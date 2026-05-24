"""End-to-end training entry point.

Run this on the training machine. It:
  1. Loads cached daily bars for the universe from analysis/_cache.
  2. Computes features + labels (panel).
  3. Runs two backtests on 2024-2025:
       (a) baseline LightGBM (equal sample weights)
       (b) LightGBM + DDG-DA-style sample reweighting
     and saves equity curves, trades, and per-retrain diagnostics.
  4. Trains a *final* model using the most-recent 3 years of data
     (label horizon trimmed) and saves it as the deployment artifact.

Outputs are written under analysis/models/:
  - final_model.txt           (LightGBM booster, plain text format)
  - feature_spec.json         (feature list + label horizon)
  - backtest_baseline.csv     (equity curve)
  - backtest_ddgda.csv        (equity curve, DDG-DA variant)
  - trades_baseline.csv
  - trades_ddgda.csv
  - model_log.csv             (per-retrain diagnostics)
  - report.json               (summary stats)

Usage:
    python train.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_io import fetch_range  # noqa: E402
from universe import get_universe  # noqa: E402
from features import FEATURE_COLS, LABEL_COL, compute_features  # noqa: E402
from backtest import BacktestConfig, run_backtest  # noqa: E402

OUT_DIR = HERE / "models"
OUT_DIR.mkdir(exist_ok=True)

BACKTEST_START = "2024-01-01"
BACKTEST_END = "2025-12-31"
HISTORY_START = "2016-01-01"
HISTORY_END = "2026-05-23"
LABEL_HORIZON = 20
TRAIN_YEARS = 3


def load_bars() -> dict[str, pd.DataFrame]:
    """Load whatever the cache has, log misses, drop empties."""
    bars: dict[str, pd.DataFrame] = {}
    for code in get_universe():
        try:
            df = fetch_range(code, HISTORY_START, HISTORY_END)
        except Exception as e:  # pragma: no cover
            print(f"[skip] {code}: {e!r}")
            continue
        if df.empty or len(df) < 200:
            print(f"[skip-thin] {code}: only {len(df)} rows")
            continue
        bars[code] = df
    print(f"loaded {len(bars)} stocks")
    return bars


def build_panel(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    for code, df in bars.items():
        feat = compute_features(df, label_h=LABEL_HORIZON)
        pieces.append(feat)
    panel = pd.concat(pieces, ignore_index=True)
    panel = panel.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"panel: {len(panel):,} rows, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    return panel


def fit_final(panel: pd.DataFrame) -> lgb.Booster:
    """Train on the most recent 3 years (labels trimmed to those that fully
    materialised). This is the booster shipped to the live trader."""
    last_label_date = panel["date"].max() - pd.Timedelta(days=LABEL_HORIZON + 5)
    train_start = last_label_date - pd.DateOffset(years=TRAIN_YEARS)
    tr = panel[(panel["date"] >= train_start) & (panel["date"] <= last_label_date)]
    tr = tr.dropna(subset=[LABEL_COL])
    X = tr[FEATURE_COLS].astype(float)
    y = tr[LABEL_COL].astype(float)
    print(f"final fit: {len(X):,} samples, "
          f"{tr['date'].min().date()} -> {tr['date'].max().date()}")

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    ds = lgb.Dataset(X, label=y)
    return lgb.train(params, ds, num_boost_round=600)


def equity_metrics(eq: pd.DataFrame, capital: float) -> dict:
    if eq.empty:
        return {}
    e = eq["equity"].values
    days = len(e)
    cum_ret = (e[-1] - capital) / capital
    daily_ret = pd.Series(e).pct_change().dropna()
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252) + 1e-9)
    peak = pd.Series(e).cummax()
    dd = (pd.Series(e) - peak) / peak
    max_dd = dd.min()
    return {
        "days": int(days),
        "final_equity": float(e[-1]),
        "cum_return": float(cum_ret),
        "ann_return": float(cum_ret / days * 252 if days else 0),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def main() -> None:
    t0 = time.monotonic()
    bars = load_bars()
    if len(bars) < 10:
        print("WARNING: fewer than 10 stocks loaded. Cache likely incomplete.")
    panel = build_panel(bars)

    cfg_base = BacktestConfig(use_ddgda=False)
    cfg_ddg = BacktestConfig(use_ddgda=True)

    print("\n--- backtest: baseline LightGBM ---")
    r_base = run_backtest(panel, bars, BACKTEST_START, BACKTEST_END, cfg_base)
    print("--- backtest: DDG-DA weighted ---")
    r_ddg = run_backtest(panel, bars, BACKTEST_START, BACKTEST_END, cfg_ddg)

    r_base.equity.to_csv(OUT_DIR / "backtest_baseline.csv", index=False)
    r_ddg.equity.to_csv(OUT_DIR / "backtest_ddgda.csv", index=False)
    r_base.trades.to_csv(OUT_DIR / "trades_baseline.csv", index=False)
    r_ddg.trades.to_csv(OUT_DIR / "trades_ddgda.csv", index=False)
    pd.concat([
        r_base.model_log.assign(variant="baseline"),
        r_ddg.model_log.assign(variant="ddgda"),
    ]).to_csv(OUT_DIR / "model_log.csv", index=False)

    print("\nbacktest summary:")
    m_base = equity_metrics(r_base.equity, cfg_base.capital)
    m_ddg = equity_metrics(r_ddg.equity, cfg_ddg.capital)
    for label, m in [("baseline", m_base), ("ddgda", m_ddg)]:
        print(f"  {label}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                         else f"{k}={v}" for k, v in m.items()))

    print("\n--- training final deployment booster ---")
    booster = fit_final(panel)
    booster.save_model(str(OUT_DIR / "final_model.txt"))

    spec = {
        "feature_cols": FEATURE_COLS,
        "label_col": LABEL_COL,
        "label_horizon_days": LABEL_HORIZON,
        "trained_at": pd.Timestamp.now().isoformat(),
        "history_start": HISTORY_START,
        "history_end": HISTORY_END,
        "universe": list(bars.keys()),
        "backtest": {
            "start": BACKTEST_START, "end": BACKTEST_END,
            "baseline": m_base, "ddgda": m_ddg,
        },
    }
    (OUT_DIR / "feature_spec.json").write_text(json.dumps(spec, indent=2))
    print(f"\nartifacts saved to {OUT_DIR}")
    print(f"total wall-time: {(time.monotonic()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
