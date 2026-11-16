"""qlib-style backtest: TopkDropoutStrategy + close-price execution.

Mirrors qlib's default benchmark setup
(thirdparty/qlib/examples/benchmarks/GRU/workflow_config_gru_Alpha360.yaml):

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

from src.trading.rules import COST_TABLES, is_limit_locked
from src.trading.strategy import buy_slot, topk_dropout


@dataclass
class QlibBTConfig:
    capital: float = 1e8  # starting cash: 100M NT$
    top_k: int = 50  # qlib default; for our 50-stock universe, set to 10 for fair comparison
    n_drop: int = 5  # qlib default
    cost_alpha_bps: float = 0.0  # qlib default: no threshold (every top-K name is targeted)
    costs: Literal["qlib", "twse"] = "qlib"
    limit_threshold: float = 0.095  # block trades at +/- 9.5% limit-up/down (TWSE limit is 10%)
    deal_price: Literal["close", "open"] = "close"
    risk_degree: float = 0.95  # qlib default: 95% deployed, 5% cash reserve
    # "close" = qlib default (today's close; look-ahead, unimplementable live)
    # "open"  = today's open (achievable live via pre-9:00 order targeting the opening auction)


@dataclass
class BTResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    daily: pd.DataFrame
    summary: dict


def _equity_summary(eq: pd.DataFrame, capital: float, trades_df: pd.DataFrame) -> dict:
    if eq.empty:
        return {
            "days": 0,
            "final_equity": capital,
            "cum_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
            "n_round_trips": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_profit_return": 0.0,
            "avg_loss_return": 0.0,
            "fill_rate": 1.0,
        }
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
        closed = trades_df.loc[(trades_df["side"] == "SELL") & trades_df["trade_return"].notna()]
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
        "fill_rate": 1.0,  # qlib model: always fills
    }


class PortfolioSim:
    """Day-by-day portfolio simulator: topk-dropout selection, fills at the
    day's deal price (close or open), fractional shares, per-side costs,
    no trades on limit-locked days, mark-to-close equity.

    Used by run_qlib_backtest_from_predictions (replay) and by the live-IL
    walk in trainer/incremental.py, which feeds it one day at a time as the
    model adapts.
    """

    def __init__(self, bars: dict[str, pd.DataFrame], cfg: QlibBTConfig):
        self.cfg = cfg
        costs = COST_TABLES[cfg.costs]
        self.open_cost, self.close_cost, self.min_cost = costs["open_cost"], costs["close_cost"], costs["min_cost"]
        self.closes = {c: bars[c].set_index("date")["close"] for c in bars if "date" in bars[c].columns}
        opens = {c: bars[c].set_index("date")["open"] for c in self.closes}
        self.highs = {c: bars[c].set_index("date")["high"] for c in self.closes}
        self.lows = {c: bars[c].set_index("date")["low"] for c in self.closes}
        self.prev_close = {c: self.closes[c].shift(1) for c in self.closes}
        # Pick the deal-price series based on cfg.deal_price (close or open).
        self.deal_series = self.closes if cfg.deal_price == "close" else opens

        self.cash = cfg.capital
        self.positions: dict[str, float] = {}  # code -> shares (float, no lot rounding)
        self.entry: dict[str, dict] = {}  # code -> {price, cost}
        self.trades: list[dict] = []
        self.equity_rows: list[tuple] = []
        self.daily_rows: list[dict] = []

    def trade_day(self, today: pd.Timestamp, target: list[str], scores: dict[str, float]) -> None:
        """Rebalance on `today` towards `target` (ranked, best first), then mark to close."""
        cfg = self.cfg
        closes, highs, lows, prev_close, deal_series = (
            self.closes,
            self.highs,
            self.lows,
            self.prev_close,
            self.deal_series,
        )
        positions, entry = self.positions, self.entry
        sell_candidates, buy_candidates = topk_dropout(target, positions, scores, cfg.top_k, cfg.n_drop)

        # --- SELLS at today's deal_price (no fill check, no tick/lot rounding) ---
        for code in sell_candidates:
            if code not in closes:
                continue
            px = deal_series[code].get(today, np.nan)
            hi = highs[code].get(today, np.nan)
            lo = lows[code].get(today, np.nan)
            pc = prev_close[code].get(today, np.nan)
            if not np.isfinite(px):
                continue
            if is_limit_locked(hi, lo, pc, cfg.limit_threshold):
                continue  # halted
            shares = positions[code]
            if shares <= 0:
                continue
            notional = shares * float(px)
            cost = max(notional * self.close_cost, self.min_cost)
            proceeds = notional - cost
            self.cash += proceeds

            buy_price = entry.get(code, {}).get("price", float(px))
            trade_return = ((float(px) - buy_price) / buy_price) if buy_price > 0 else 0.0
            self.trades.append(
                {
                    "date": today,
                    "code": code,
                    "side": "SELL",
                    "price": float(px),
                    "shares": shares,
                    "cashflow": proceeds,
                    "trade_return": trade_return,
                }
            )
            positions.pop(code, None)
            entry.pop(code, None)

        # --- BUYS at today's deal_price ---
        # qlib's exact formula: slot per buy = current cash * risk_degree / len(buys).
        # Sells above have already added their proceeds to cash, so this reinvests them.
        n_buys = len(buy_candidates)
        slot_capital = buy_slot(self.cash, cfg.risk_degree, n_buys)
        for code in buy_candidates:
            if code not in closes:
                continue
            px = deal_series[code].get(today, np.nan)
            hi = highs[code].get(today, np.nan)
            lo = lows[code].get(today, np.nan)
            pc = prev_close[code].get(today, np.nan)
            if not np.isfinite(px) or px <= 0:
                continue
            if is_limit_locked(hi, lo, pc, cfg.limit_threshold):
                continue
            shares = slot_capital / float(px)  # fractional shares (qlib does this)
            notional = shares * float(px)
            cost = max(notional * self.open_cost, self.min_cost)
            total = notional + cost
            if total > self.cash:
                continue
            self.cash -= total
            positions[code] = shares
            entry[code] = {"price": float(px), "cost": total}
            self.trades.append(
                {
                    "date": today,
                    "code": code,
                    "side": "BUY",
                    "price": float(px),
                    "shares": shares,
                    "cashflow": total,
                    "trade_return": np.nan,
                }
            )

        # --- Mark-to-close ---
        mark = 0.0
        for code, sh in positions.items():
            if code in closes:
                cl = closes[code].get(today, np.nan)
                if np.isfinite(cl):
                    mark += sh * float(cl)
        self.equity_rows.append((today, self.cash + mark))
        self.daily_rows.append(
            {
                "date": today,
                "cash": self.cash,
                "mark_value": mark,
                "n_positions": len(positions),
                "n_targets": len(target),
            }
        )

    @property
    def last_equity(self) -> float | None:
        return self.equity_rows[-1][1] if self.equity_rows else None

    def result(self) -> BTResult:
        eq = pd.DataFrame(self.equity_rows, columns=["date", "equity"])
        trades_df = pd.DataFrame(self.trades) if self.trades else _empty_trades()
        summary = _equity_summary(eq, self.cfg.capital, trades_df)
        return BTResult(equity=eq, trades=trades_df, daily=pd.DataFrame(self.daily_rows), summary=summary)


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "code", "side", "price", "shares", "cashflow", "trade_return"])


def run_qlib_backtest_from_predictions(
    preds_df: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    cfg: QlibBTConfig | None = None,
) -> BTResult:
    """qlib-style: top-K dropout strategy, deal_price=close, always fills.

    Replays a fixed predictions table (date, stock_code_id, pred) day by day.
    For the mode where the model keeps adapting during the backtest, see
    trainer.incremental.run_qlib_backtest_with_il.
    """
    cfg = cfg or QlibBTConfig()
    if preds_df.empty:
        empty_trades = _empty_trades()
        return BTResult(
            equity=pd.DataFrame(columns=["date", "equity"]),
            trades=empty_trades,
            daily=pd.DataFrame(),
            summary=_equity_summary(pd.DataFrame(columns=["equity"]), cfg.capital, empty_trades),
        )

    preds = preds_df.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    preds["stock_code_id"] = preds["stock_code_id"].astype(str)

    sim = PortfolioSim(bars, cfg)
    thresh = cfg.cost_alpha_bps / 1e4
    for today in sorted(preds["date"].unique()):
        today_preds = preds[preds["date"] == today]
        pred_lookup = dict(zip(today_preds["stock_code_id"].tolist(), today_preds["pred"].astype(float).tolist()))
        target = (
            today_preds[today_preds["pred"] >= thresh]
            .sort_values("pred", ascending=False)
            .head(cfg.top_k)["stock_code_id"]
            .tolist()
        )
        sim.trade_day(today, target, pred_lookup)
    return sim.result()
