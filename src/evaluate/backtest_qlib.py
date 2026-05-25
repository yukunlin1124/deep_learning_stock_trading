"""qlib-style backtest: TopkDropoutStrategy + close-price execution.

Mirrors qlib's default benchmark setup
(qlib/examples/benchmarks/GRU/workflow_config_gru_Alpha360.yaml):

  strategy:
      class: TopkDropoutStrategy
      kwargs: { topk: 50, n_drop: 5 }
  backtest:
      exchange_kwargs:
          deal_price: close          # every order fills at today's close
          open_cost: 0.0005          # 5 bps buy commission
          close_cost: 0.0015         # 15 bps sell commission + tax (bundled)
          min_cost: 5                # 5 CNY min per trade
          limit_threshold: 0.095     # block at limit-up/down

Key differences from src/evaluate/backtest.py (our realistic backtest):
  - **Always fills**: no `low <= limit <= high` check; every order executes
    at today's close
  - **No tick rounding**: prices are continuous (no TWSE-band snapping)
  - **No lot rounding**: shares are continuous (no 1000-share lot constraint)
  - **Look-ahead bias**: uses today's close to decide today's order (you
    can't actually do this in live trading — order has to be placed before
    market close)
  - **No NT$20 min commission**: uses qlib's `min_cost` floor instead

Use the `costs` flag to switch between qlib's CN-A-share defaults and
TWSE-correct rates so we can compare apples-to-apples on the SAME pricing
model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class QlibBTConfig:
    capital: float = 1e8
    top_k: int = 50                      # qlib default; for our 50-stock universe, set to 10 for fair comparison
    n_drop: int = 5                      # qlib default
    cost_alpha_bps: float = 0.0          # qlib default: no threshold (every top-K name is targeted)
    costs: Literal["qlib", "twse"] = "qlib"
    limit_threshold: float = 0.095       # block trades at +/- 9.5% limit-up/down (TWSE limit is 10%)


_COST_TABLES = {
    "qlib": dict(open_cost=0.0005, close_cost=0.0015, min_cost=5.0),
    "twse": dict(open_cost=0.001425,                           # 0.1425% commission
                 close_cost=0.001425 + 0.003,                  # 0.1425% + 0.3% sell tax
                 min_cost=20.0),
}


@dataclass
class BTResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    daily: pd.DataFrame
    summary: dict


def _equity_summary(eq: pd.DataFrame, capital: float, trades_df: pd.DataFrame
                    ) -> dict:
    if eq.empty:
        return {"days": 0, "final_equity": capital, "cum_return": 0.0,
                "ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "n_trades": 0, "n_round_trips": 0,
                "win_rate": 0.0, "profit_factor": 0.0,
                "avg_profit_return": 0.0, "avg_loss_return": 0.0,
                "fill_rate": 1.0}
    e = eq["equity"].values
    days = len(e)
    cum_ret = (e[-1] - capital) / capital
    daily_ret = pd.Series(e).pct_change().dropna()
    ann_vol = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) else 0.0
    sharpe = (float(daily_ret.mean()) * 252) / (ann_vol + 1e-9) if ann_vol > 0 else 0.0
    peak = pd.Series(e).cummax()
    max_dd = float(((pd.Series(e) - peak) / peak).min())

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
            win_rate = float(len(wins) / len(r))
            gains, losses_abs = float(wins.sum()), float(-losses.sum())
            profit_factor = float(gains / losses_abs) if losses_abs > 0 else float("inf")
            avg_profit = float(wins.mean()) if len(wins) else 0.0
            avg_loss = float(losses.mean()) if len(losses) else 0.0

    return {
        "days": int(days),
        "final_equity": float(e[-1]),
        "cum_return": float(cum_ret),
        "ann_return": float(cum_ret / days * 252 if days else 0.0),
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "n_trades": int(len(trades_df)) if not trades_df.empty else 0,
        "n_round_trips": n_round_trips,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_profit_return": avg_profit,
        "avg_loss_return": avg_loss,
        "fill_rate": 1.0,                    # qlib model: always fills
    }


def _is_limit_locked(today_high: float, today_low: float, prev_close: float,
                     threshold: float) -> bool:
    """Approximate limit-up/down detection: if the day's range is at the
    +/-threshold edge of prev_close. We block trades on those days."""
    if not np.isfinite(prev_close) or prev_close <= 0:
        return False
    limit_up = prev_close * (1 + threshold)
    limit_down = prev_close * (1 - threshold)
    # if both high and low are at one limit, it's locked
    return (today_low >= limit_up * 0.999) or (today_high <= limit_down * 1.001)


def run_qlib_backtest_from_predictions(
    preds_df: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    cfg: QlibBTConfig | None = None,
) -> BTResult:
    """qlib-style: top-K dropout strategy, deal_price=close, always fills."""
    cfg = cfg or QlibBTConfig()
    costs = _COST_TABLES[cfg.costs]
    open_cost, close_cost, min_cost = costs["open_cost"], costs["close_cost"], costs["min_cost"]

    empty_trades = pd.DataFrame(columns=[
        "date", "code", "side", "price", "shares", "cashflow", "trade_return"
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

    test_dates = sorted(preds["date"].unique())
    cash = cfg.capital
    positions: dict[str, float] = {}                   # code -> shares (float, no lot rounding)
    entry: dict[str, dict] = {}                        # code -> {price, cost}

    trades: list[dict] = []
    equity_rows: list[tuple] = []
    daily_rows: list[dict] = []
    thresh = cfg.cost_alpha_bps / 1e4

    for today in test_dates:
        today_preds = preds[preds["date"] == today]
        pred_lookup = dict(zip(today_preds["stock_code_id"].tolist(),
                               today_preds["pred"].astype(float).tolist()))

        target = (
            today_preds[today_preds["pred"] >= thresh]
            .sort_values("pred", ascending=False)
            .head(cfg.top_k)["stock_code_id"].tolist()
        )
        target_set = set(target)
        held_set = set(positions)

        sell_candidates = sorted(held_set - target_set,
                                 key=lambda c: pred_lookup.get(c, -np.inf))
        buy_candidates = [c for c in target if c not in held_set]
        if cfg.n_drop is not None:
            sell_candidates = sell_candidates[: cfg.n_drop]
            buy_candidates = buy_candidates[: cfg.n_drop]

        # --- SELLS at today's close (no fill check, no tick/lot rounding) ---
        for code in sell_candidates:
            if code not in closes:
                continue
            cl = closes[code].get(today, np.nan)
            hi = highs[code].get(today, np.nan)
            lo = lows[code].get(today, np.nan)
            pc = prev_close[code].get(today, np.nan)
            if not np.isfinite(cl):
                continue
            if _is_limit_locked(hi, lo, pc, cfg.limit_threshold):
                continue                                            # halted
            shares = positions[code]
            if shares <= 0:
                continue
            notional = shares * float(cl)
            cost = max(notional * close_cost, min_cost)
            proceeds = notional - cost
            cash += proceeds

            buy_price = entry.get(code, {}).get("price", float(cl))
            trade_return = ((float(cl) - buy_price) / buy_price) if buy_price > 0 else 0.0
            trades.append({
                "date": today, "code": code, "side": "SELL",
                "price": float(cl), "shares": shares, "cashflow": proceeds,
                "trade_return": trade_return,
            })
            positions.pop(code, None)
            entry.pop(code, None)

        # --- BUYS at today's close ---
        slot_capital = cfg.capital / cfg.top_k
        for code in buy_candidates:
            if code not in closes:
                continue
            cl = closes[code].get(today, np.nan)
            hi = highs[code].get(today, np.nan)
            lo = lows[code].get(today, np.nan)
            pc = prev_close[code].get(today, np.nan)
            if not np.isfinite(cl) or cl <= 0:
                continue
            if _is_limit_locked(hi, lo, pc, cfg.limit_threshold):
                continue
            shares = slot_capital / float(cl)                       # fractional shares (qlib does this)
            notional = shares * float(cl)
            cost = max(notional * open_cost, min_cost)
            total = notional + cost
            if total > cash:
                continue
            cash -= total
            positions[code] = shares
            entry[code] = {"price": float(cl), "cost": total}
            trades.append({
                "date": today, "code": code, "side": "BUY",
                "price": float(cl), "shares": shares, "cashflow": total,
                "trade_return": np.nan,
            })

        # --- Mark-to-close ---
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
