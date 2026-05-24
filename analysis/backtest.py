"""Sliding-window cross-sectional backtest with realistic fills + costs.

Pipeline (per rebalance date t in test window):
  1. Train window = [t - 3y, t - H], features+labels from the universe panel.
  2. (Optional) DDG-DA weights = similarity of training samples to the
     most recent 20 trading days of features (no labels needed).
  3. Fit LightGBM regressor predicting 20-day forward return.
  4. Score every stock at date t. Take top-K predictions.
  5. Submit a one-shot limit order at prev_close for entries / exits.
     Fill iff price within [low_t, high_t] per Rule 02. Costs per Rule 05.
  6. Mark portfolio to close_t. Repeat next day.

Designed to be driven from train.py — this module exposes `run_backtest`
and never reads files directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from features import FEATURE_COLS, LABEL_COL
from meta_weighting import compute_weights
from trading_rules import (
    SHARES_PER_LOT,
    buy_cost,
    round_buy,
    round_sell,
    sell_proceeds,
)


@dataclass
class BacktestConfig:
    train_years: float = 3.0
    label_horizon: int = 20
    rebalance_every: int = 20            # trading days between retrains
    top_k: int = 10                      # long basket size
    capital: float = 1e8
    cost_alpha_bps: float = 70.0         # min predicted return (bps) to act
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 400,
    })
    use_ddgda: bool = False              # ablation switch
    near_future_days: int = 20           # DDG-DA "what does soon look like"
    seed: int = 0


@dataclass
class BacktestResult:
    equity: pd.DataFrame                 # date, equity
    trades: pd.DataFrame
    daily: pd.DataFrame                  # diagnostics per day
    model_log: pd.DataFrame              # per-retrain stats


def _fit_one(
    panel: pd.DataFrame, asof: pd.Timestamp, cfg: BacktestConfig,
) -> tuple[lgb.Booster, dict]:
    train_end = asof - pd.Timedelta(days=cfg.label_horizon)
    train_start = train_end - pd.DateOffset(years=int(cfg.train_years))
    tr = panel[(panel["date"] >= train_start) & (panel["date"] <= train_end)]
    if tr.empty:
        raise RuntimeError(f"no training data at asof={asof.date()}")

    X = tr[FEATURE_COLS].astype(float)
    y = tr[LABEL_COL].astype(float)

    sample_weight = None
    if cfg.use_ddgda:
        near_start = asof - pd.Timedelta(days=int(cfg.near_future_days * 1.5))
        nf = panel[(panel["date"] >= near_start) & (panel["date"] < asof)]
        if len(nf) >= 20:
            sample_weight = compute_weights(X, nf[FEATURE_COLS].astype(float))

    ds = lgb.Dataset(X, label=y, weight=sample_weight)
    params = dict(cfg.lgbm_params)
    n_rounds = params.pop("n_estimators", 400)
    booster = lgb.train(params, ds, num_boost_round=n_rounds)

    diag = {
        "asof": asof, "n_train": len(tr),
        "ddgda": cfg.use_ddgda,
        "weight_mean": float(np.mean(sample_weight)) if sample_weight is not None else 1.0,
        "weight_std": float(np.std(sample_weight)) if sample_weight is not None else 0.0,
    }
    return booster, diag


def run_backtest(
    panel: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    start: str,
    end: str,
    cfg: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """panel: long-format features+labels; bars: per-stock OHLC for fills."""
    cfg = cfg or BacktestConfig()
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    # ordered list of unique trading dates in the test window
    test_dates = sorted(d for d in panel["date"].unique()
                        if start_dt <= d <= end_dt)
    if not test_dates:
        raise RuntimeError("no test dates in panel")

    # per-stock close lookup for marking
    closes = {c: bars[c].set_index("date")["close"] for c in bars}
    highs = {c: bars[c].set_index("date")["high"] for c in bars}
    lows = {c: bars[c].set_index("date")["low"] for c in bars}
    prev_close = {c: bars[c].set_index("date")["close"].shift(1) for c in bars}

    cash = cfg.capital
    positions: dict[str, int] = {}      # stock_code -> shares held
    trades = []
    equity_rows = []
    model_log = []
    daily_rows = []

    last_retrain = None
    booster: Optional[lgb.Booster] = None
    next_retrain_idx = 0

    for i, today in enumerate(test_dates):
        # retrain if needed
        if booster is None or i >= next_retrain_idx:
            try:
                booster, diag = _fit_one(panel, today, cfg)
                model_log.append(diag)
                last_retrain = today
                next_retrain_idx = i + cfg.rebalance_every
            except RuntimeError:
                pass

        # score today
        today_panel = panel[panel["date"] == today]
        if today_panel.empty or booster is None:
            mark = sum(positions.get(c, 0) * float(closes[c].get(today, np.nan))
                       for c in positions if c in closes)
            equity_rows.append((today, cash + (mark if not np.isnan(mark) else 0)))
            continue

        X_today = today_panel[FEATURE_COLS].astype(float)
        scores = booster.predict(X_today)
        today_panel = today_panel.assign(pred=scores)

        # top-K longs that clear the cost-aware alpha threshold
        thresh = cfg.cost_alpha_bps / 1e4
        target = (
            today_panel[today_panel["pred"] >= thresh]
            .sort_values("pred", ascending=False)
            .head(cfg.top_k)["stock_code_id"].astype(str).tolist()
        )

        # exits: holdings not in target
        for code in list(positions.keys()):
            if code in target or positions[code] == 0:
                continue
            pc = prev_close[code].get(today, np.nan)
            if np.isnan(pc):
                continue
            limit = round_sell(float(pc))
            hi = highs[code].get(today, np.nan)
            lo = lows[code].get(today, np.nan)
            if np.isnan(hi) or np.isnan(lo):
                continue
            if not (lo <= limit <= hi):
                continue
            lots = positions[code] // SHARES_PER_LOT
            if lots <= 0:
                continue
            proceeds = sell_proceeds(limit, lots)
            cash += proceeds
            trades.append((today, code, "SELL", limit, lots, proceeds))
            positions.pop(code, None)

        # entries: targets we don't already hold
        held = set(positions)
        new_targets = [c for c in target if c not in held]
        slot_capital = cfg.capital / cfg.top_k
        for code in new_targets:
            pc = prev_close[code].get(today, np.nan)
            hi = highs[code].get(today, np.nan)
            lo = lows[code].get(today, np.nan)
            if np.isnan(pc) or np.isnan(hi) or np.isnan(lo):
                continue
            limit = round_buy(float(pc))
            if not (lo <= limit <= hi):
                continue
            max_lots = int(slot_capital // (limit * SHARES_PER_LOT))
            if max_lots <= 0:
                continue
            cost = buy_cost(limit, max_lots)
            if cost > cash:
                continue
            cash -= cost
            positions[code] = positions.get(code, 0) + max_lots * SHARES_PER_LOT
            trades.append((today, code, "BUY", limit, max_lots, cost))

        # mark-to-close
        mark = 0.0
        for code, sh in positions.items():
            cl = closes[code].get(today, np.nan)
            if not np.isnan(cl):
                mark += sh * float(cl)
        equity_rows.append((today, cash + mark))
        daily_rows.append({
            "date": today, "cash": cash, "mark_value": mark,
            "n_positions": len(positions),
            "n_targets": len(target),
        })

    return BacktestResult(
        equity=pd.DataFrame(equity_rows, columns=["date", "equity"]),
        trades=pd.DataFrame(
            trades,
            columns=["date", "code", "side", "price", "lots", "cashflow"],
        ),
        daily=pd.DataFrame(daily_rows),
        model_log=pd.DataFrame(model_log),
    )
