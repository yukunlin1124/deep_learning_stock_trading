"""Portfolio backtest that consumes IL-phase predictions.

Trading rules implemented (from stock_project_for_class/rules.md):
  - Rule 01: NT$100,000,000 starting capital            (BTConfig.capital default)
  - Rule 02: limit fills iff low <= limit <= high       (entry/exit code)
  - Rule 04: TWSE banded tick + buy-down/sell-up        (round_buy / round_sell)
  - Rule 05: 0.1425% commission (min NT$20) + 0.3% tax  (trading_rules helpers)
  - Rule 06: cumulative / annualized / per-trade return,
             win rate, profit factor                    (_equity_summary)

qlib-style portfolio realism (in addition to the rules above):
  - n_drop turnover throttling: at most n_drop names swapped per day
  - hold_thresh: minimum trading days held before sell allowed

Top-K long-only over the IL predictions:
  Each day, rank the universe by pred and identify the new top-K target list.
  EXITS:  for held names not in target, oldest-eligible first, capped by n_drop
  ENTRIES: for target names not held, highest-pred first, capped by n_drop
  FILL:   one-shot limit order at round_{buy,sell}(prev_close), fills iff
          low <= limit <= high on the trade day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .trading_rules import (
    SHARES_PER_LOT, buy_cost, round_buy, round_sell, sell_proceeds,
)


@dataclass
class BTConfig:
    capital: float = 1e8                # Rule 01
    top_k: int = 10                     # portfolio width
    cost_alpha_bps: float = 70.0        # min predicted return to act (covers round-trip cost)
    n_drop: int | None = None           # qlib-style daily turnover cap; None = no throttle
    hold_thresh: int = 1                # min trading days held before sell


@dataclass
class BTResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    daily: pd.DataFrame
    summary: dict


# ----------------- summary metrics (Rule 06) -----------------

def _equity_summary(eq: pd.DataFrame, capital: float, trades_df: pd.DataFrame
                    ) -> dict:
    """Portfolio + per-trade metrics per rules.md section 06."""
    if eq.empty:
        return {"days": 0, "final_equity": capital, "cum_return": 0.0,
                "ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "n_trades": 0, "n_round_trips": 0,
                "win_rate": 0.0, "profit_factor": 0.0,
                "avg_profit_return": 0.0, "avg_loss_return": 0.0}
    e = eq["equity"].values
    days = len(e)
    cum_ret = (e[-1] - capital) / capital                       # Rule 06: cum return
    daily_ret = pd.Series(e).pct_change().dropna()
    ann_vol = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) else 0.0
    sharpe = (float(daily_ret.mean()) * 252) / (ann_vol + 1e-9) if ann_vol > 0 else 0.0
    peak = pd.Series(e).cummax()
    dd = (pd.Series(e) - peak) / peak
    max_dd = float(dd.min())

    # Rule 06: per-trade metrics (closed round-trips only)
    n_round_trips = 0
    win_rate = 0.0
    profit_factor = 0.0
    avg_profit = 0.0
    avg_loss = 0.0
    if not trades_df.empty and "trade_return" in trades_df.columns:
        closed = trades_df.loc[
            (trades_df["side"] == "SELL") & trades_df["trade_return"].notna()
        ]
        if not closed.empty:
            r = closed["trade_return"].astype(float)
            wins = r[r > 0]
            losses = r[r < 0]
            n_round_trips = int(len(r))
            win_rate = float(len(wins) / len(r))                # Rule 06: win rate
            gains = float(wins.sum())                            # Rule 06: profit factor
            losses_abs = float(-losses.sum())
            profit_factor = float(gains / losses_abs) if losses_abs > 0 else float("inf")
            avg_profit = float(wins.mean()) if len(wins) else 0.0  # Rule 06: avg gain
            avg_loss = float(losses.mean()) if len(losses) else 0.0  # Rule 06: avg loss

    return {
        "days": int(days),
        "final_equity": float(e[-1]),
        "cum_return": float(cum_ret),
        "ann_return": float(cum_ret / days * 252 if days else 0.0),    # Rule 06: annualized
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "n_trades": int(len(trades_df)) if not trades_df.empty else 0,
        "n_round_trips": n_round_trips,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_profit_return": avg_profit,
        "avg_loss_return": avg_loss,
    }


# ----------------- main backtest -----------------

def run_backtest_from_predictions(
    preds_df: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    cfg: BTConfig | None = None,
) -> BTResult:
    cfg = cfg or BTConfig()
    empty_trades = pd.DataFrame(columns=[
        "date", "code", "side", "price", "lots", "cashflow", "trade_return"
    ])
    if preds_df.empty:
        return BTResult(
            equity=pd.DataFrame(columns=["date", "equity"]),
            trades=empty_trades,
            daily=pd.DataFrame(),
            summary=_equity_summary(pd.DataFrame(columns=["equity"]), cfg.capital,
                                    empty_trades),
        )

    preds = preds_df.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    preds["stock_code_id"] = preds["stock_code_id"].astype(str)

    closes = {c: bars[c].set_index("date")["close"]
              for c in bars if "date" in bars[c].columns}
    highs = {c: bars[c].set_index("date")["high"] for c in closes}
    lows = {c: bars[c].set_index("date")["low"] for c in closes}
    prev_close = {c: closes[c].shift(1) for c in closes}

    # Lookup: today's pred score for a code (for sorting holdings/targets).
    # Built per-day inside the loop.

    test_dates = sorted(preds["date"].unique())
    cash = cfg.capital
    positions: dict[str, int] = {}                  # code -> shares
    entry: dict[str, dict] = {}                     # code -> {price, lots, cost, day_idx}

    trades: list[dict] = []
    equity_rows: list[tuple] = []
    daily_rows: list[dict] = []
    thresh = cfg.cost_alpha_bps / 1e4

    for today_idx, today in enumerate(test_dates):
        today_preds = preds[preds["date"] == today]
        pred_lookup = dict(zip(today_preds["stock_code_id"].tolist(),
                               today_preds["pred"].astype(float).tolist()))

        # Build target list (top-K above threshold).
        target = (
            today_preds[today_preds["pred"] >= thresh]
            .sort_values("pred", ascending=False)
            .head(cfg.top_k)["stock_code_id"].tolist()
        )
        target_set = set(target)
        held_set = set(positions)

        # --- candidate exits & entries ---
        # Sell candidates: held names not in target, oldest-eligible first
        # (= lowest pred, qlib's method_sell="bottom").
        sell_candidates = [c for c in held_set - target_set]
        sell_candidates.sort(key=lambda c: pred_lookup.get(c, -np.inf))
        # Buy candidates: target names not held, highest pred first.
        buy_candidates = [c for c in target if c not in held_set]

        # Apply n_drop turnover cap (qlib parity).
        if cfg.n_drop is not None:
            sell_candidates = sell_candidates[: cfg.n_drop]
            buy_candidates = buy_candidates[: cfg.n_drop]

        # --- exits ---
        for code in sell_candidates:
            if code not in prev_close:
                continue
            # hold_thresh: skip if held fewer than hold_thresh trading days
            entered_idx = entry.get(code, {}).get("day_idx", -10**9)
            if today_idx - entered_idx < cfg.hold_thresh:
                continue
            pc = prev_close[code].get(today, np.nan)
            hi = highs[code].get(today, np.nan)
            lo = lows[code].get(today, np.nan)
            if not (np.isfinite(pc) and np.isfinite(hi) and np.isfinite(lo)):
                continue
            limit = round_sell(float(pc))                      # Rule 04: sell rounds up
            if limit <= 0:
                continue
            if not (lo <= limit <= hi):                        # Rule 02: range fill
                continue
            lots = positions[code] // SHARES_PER_LOT
            if lots <= 0:
                continue
            proceeds = sell_proceeds(limit, lots)              # Rule 05: commission + tax
            cash += proceeds

            # Per-trade return: (sell_price - cost_price) / cost_price  (Rule 06)
            buy_price = entry.get(code, {}).get("price", limit)
            trade_return = ((limit - buy_price) / buy_price) if buy_price > 0 else 0.0
            trades.append({
                "date": today, "code": code, "side": "SELL",
                "price": limit, "lots": lots, "cashflow": proceeds,
                "trade_return": trade_return,
            })
            positions.pop(code, None)
            entry.pop(code, None)

        # --- entries ---
        slot_capital = cfg.capital / cfg.top_k
        for code in buy_candidates:
            if code not in prev_close:
                continue
            pc = prev_close[code].get(today, np.nan)
            hi = highs[code].get(today, np.nan)
            lo = lows[code].get(today, np.nan)
            if not (np.isfinite(pc) and np.isfinite(hi) and np.isfinite(lo)):
                continue
            limit = round_buy(float(pc))                       # Rule 04: buy rounds down
            if limit <= 0:
                continue
            if not (lo <= limit <= hi):                        # Rule 02: range fill
                continue
            max_lots = int(slot_capital // (limit * SHARES_PER_LOT))
            if max_lots <= 0:
                continue
            cost = buy_cost(limit, max_lots)                   # Rule 05: commission
            if cost > cash:
                continue
            cash -= cost
            positions[code] = positions.get(code, 0) + max_lots * SHARES_PER_LOT
            entry[code] = {"price": limit, "lots": max_lots,
                           "cost": cost, "day_idx": today_idx}
            trades.append({
                "date": today, "code": code, "side": "BUY",
                "price": limit, "lots": max_lots, "cashflow": cost,
                "trade_return": np.nan,                         # only set on SELL
            })

        # --- mark-to-close ---
        mark = 0.0
        for code, sh in positions.items():
            if code in closes:
                cl = closes[code].get(today, np.nan)
                if np.isfinite(cl):
                    mark += sh * float(cl)
        equity_rows.append((today, cash + mark))
        daily_rows.append({
            "date": today, "cash": cash, "mark_value": mark,
            "n_positions": len(positions), "n_targets": len(target),
        })

    eq = pd.DataFrame(equity_rows, columns=["date", "equity"])
    trades_df = pd.DataFrame(trades) if trades else empty_trades.copy()
    summary = _equity_summary(eq, cfg.capital, trades_df)
    return BTResult(equity=eq, trades=trades_df, daily=pd.DataFrame(daily_rows),
                    summary=summary)
