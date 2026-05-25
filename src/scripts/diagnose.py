"""Diagnose the training result by re-running the backtest under multiple
configs against the existing predictions.csv. Also slice per-period IC to
see where the model has signal vs noise.

Run:
    .venv/Scripts/python.exe src/scripts/diagnose.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import fetch_range
from src.data.universe import get_candidate_pool
from src.evaluate.backtest import BTConfig, run_backtest_from_predictions
from src.evaluate.metrics import cross_sectional_ic

OUT = PROJECT_ROOT / "src" / "output"


def load_bars_for(codes: set[str]) -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    for c in codes:
        try:
            df = fetch_range(c, "2024-01-01", "2026-05-31")
            if not df.empty:
                bars[c] = df
        except Exception as e:
            print(f"skip {c}: {e!r}")
    return bars


def main() -> None:
    preds = pd.read_csv(OUT / "predictions.csv")
    preds["date"] = pd.to_datetime(preds["date"])
    print(f"loaded {len(preds):,} predictions across "
          f"{preds['date'].nunique()} unique dates, "
          f"{preds['stock_code_id'].nunique()} unique stocks")

    # Load bars for codes that appear in predictions.
    codes_needed = set(preds["stock_code_id"].astype(str).unique())
    print(f"loading bars for {len(codes_needed)} stocks...")
    bars = load_bars_for(codes_needed)
    print(f"  got bars for {len(bars)} of {len(codes_needed)}\n")

    # ---- 1. Signal quality across IL window ----
    print("=" * 78)
    print("1. SIGNAL QUALITY (raw IL predictions)")
    print("=" * 78)
    p_arr = preds["pred"].to_numpy()
    a_arr = preds["actual"].to_numpy()
    d_arr = preds["date"].to_numpy().astype("datetime64[ns]")
    ic_all, icir_all, ric_all, ricir_all = cross_sectional_ic(p_arr, a_arr, d_arr)
    print(f"  overall: IC={ic_all:+.4f}  ICIR={icir_all:+.3f}  "
          f"RankIC={ric_all:+.4f}  RankICIR={ricir_all:+.3f}")

    # Quarterly slice
    print("\n  per-quarter signal quality:")
    print(f"  {'quarter':<10} {'n_rows':>7} {'IC':>9} {'RankIC':>9}")
    print(f"  {'-'*10} {'-'*7} {'-'*9} {'-'*9}")
    preds["quarter"] = preds["date"].dt.to_period("Q")
    for q, grp in preds.groupby("quarter"):
        if len(grp) < 50:
            continue
        ic, _, ric, _ = cross_sectional_ic(
            grp["pred"].to_numpy(),
            grp["actual"].to_numpy(),
            grp["date"].to_numpy().astype("datetime64[ns]"),
        )
        print(f"  {str(q):<10} {len(grp):>7} {ic:+9.4f} {ric:+9.4f}")

    # ---- 2. Distribution of predictions ----
    print()
    print("=" * 78)
    print("2. PREDICTION DISTRIBUTION")
    print("=" * 78)
    p = preds["pred"]
    print(f"  mean   {p.mean():+.5f}")
    print(f"  std    {p.std():.5f}")
    print(f"  min    {p.min():+.5f}")
    print(f"  q05    {p.quantile(0.05):+.5f}")
    print(f"  q25    {p.quantile(0.25):+.5f}")
    print(f"  q50    {p.quantile(0.50):+.5f}")
    print(f"  q75    {p.quantile(0.75):+.5f}")
    print(f"  q95    {p.quantile(0.95):+.5f}")
    print(f"  max    {p.max():+.5f}")
    print(f"  fraction with pred > 0.007 (70bps threshold): "
          f"{(p > 0.007).mean():.1%}")
    print(f"  fraction with pred > 0.015 (150bps): {(p > 0.015).mean():.1%}")

    # ---- 3. Backtest A/B across configs ----
    print()
    print("=" * 78)
    print("3. BACKTEST A/B (same predictions, different BTConfig)")
    print("=" * 78)
    configs = [
        ("baseline (current)",           BTConfig(top_k=10, n_drop=3, cost_alpha_bps=70)),
        ("no-trade sanity (thr=999%)",   BTConfig(top_k=10, n_drop=3, cost_alpha_bps=99900)),
        ("qlib-canonical 10% turnover",  BTConfig(top_k=10, n_drop=1, cost_alpha_bps=70)),
        ("conservative (n_drop=1, thr=150bps)", BTConfig(top_k=10, n_drop=1, cost_alpha_bps=150)),
        ("very selective (top_k=5)",     BTConfig(top_k=5, n_drop=1, cost_alpha_bps=150)),
        ("uncapped turnover",            BTConfig(top_k=10, n_drop=None, cost_alpha_bps=70)),
    ]

    header = (f"  {'config':<40} {'trades':>7} {'cum_ret':>9} "
              f"{'sharpe':>8} {'max_dd':>8} {'win_rt':>7} {'pf':>6}")
    print(header)
    print(f"  {'-'*40} {'-'*7} {'-'*9} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")
    for name, cfg in configs:
        bt = run_backtest_from_predictions(preds, bars, cfg)
        s = bt.summary
        pf = s["profit_factor"]
        pf_str = f"{pf:6.2f}" if np.isfinite(pf) else "  inf"
        print(f"  {name:<40} {s['n_trades']:>7} {s['cum_return']:>+9.4f} "
              f"{s['sharpe']:>+8.3f} {s['max_drawdown']:>+8.4f} "
              f"{s['win_rate']:>7.3f} {pf_str}")

    # ---- 4. Signal-inversion test ----
    print()
    print("=" * 78)
    print("4. SIGNAL-INVERSION TEST (does pred = -pred help?)")
    print("=" * 78)
    inverted = preds.copy()
    inverted["pred"] = -inverted["pred"]
    ic_i, _, ric_i, _ = cross_sectional_ic(
        inverted["pred"].to_numpy(),
        inverted["actual"].to_numpy(),
        inverted["date"].to_numpy().astype("datetime64[ns]"),
    )
    print(f"  inverted IC={ic_i:+.4f}  RankIC={ric_i:+.4f}  (sign-flip of original)")
    bt_inv = run_backtest_from_predictions(inverted, bars,
                                            BTConfig(top_k=10, n_drop=1, cost_alpha_bps=70))
    print(f"  inverted backtest (n_drop=1, 70bps): "
          f"trades={bt_inv.summary['n_trades']}, "
          f"cum_ret={bt_inv.summary['cum_return']:+.4f}, "
          f"sharpe={bt_inv.summary['sharpe']:+.3f}")

    # ---- 5. Cost-drag estimate -----
    print()
    print("=" * 78)
    print("5. COST DRAG ESTIMATE (per config)")
    print("=" * 78)
    print("  TWSE round-trip cost = 0.1425% buy + (0.1425% + 0.3%) sell = 58.5 bps")
    n_days = preds["date"].nunique()
    print(f"  IL window has {n_days} trading days")
    for name, cfg in configs:
        bt = run_backtest_from_predictions(preds, bars, cfg)
        s = bt.summary
        rt = s["n_round_trips"]
        if rt == 0:
            cost_pct = 0.0
        else:
            slot = cfg.capital / cfg.top_k
            total_cost_ntd = rt * 0.00585 * slot
            cost_pct = total_cost_ntd / cfg.capital
        print(f"  {name:<40} round_trips={rt:>5}  est_total_cost={cost_pct*100:>6.1f}%")


if __name__ == "__main__":
    main()
