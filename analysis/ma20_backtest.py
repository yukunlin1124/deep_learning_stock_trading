"""Step 4 — Honest MA20 backtest with realistic fills + costs.

Signal (computed at end of day t-1, no look-ahead):
  - prev_close > MA20  AND  not in position  -> buy 1 lot at t
  - prev_close < MA20  AND      in position  -> sell 1 lot at t

Order fills only if limit price is within [low_t, high_t].
Limit price = prev_close (rounded for TWSE ticks).
Costs from trading_rules: 0.1425% commission both sides + 0.3% sell tax.

Reports cumulative return on a notional account.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import fetch_range  # noqa: E402
from trading_rules import (  # noqa: E402
    SHARES_PER_LOT,
    buy_cost,
    round_buy,
    round_sell,
    sell_proceeds,
)


def run_ma20(df: pd.DataFrame, lots_per_trade: int = 1, capital: float = 1e8):
    df = df.copy().reset_index(drop=True)
    df["ma20"] = df["close"].rolling(20).mean()
    df["prev_close"] = df["close"].shift(1)
    df["prev_ma20"] = df["ma20"].shift(1)

    cash = capital
    shares = 0
    trades = []
    equity_curve = []

    for _, row in df.iterrows():
        date, hi, lo, close = row["date"], row["high"], row["low"], row["close"]
        pc, pma = row["prev_close"], row["prev_ma20"]

        if pd.isna(pc) or pd.isna(pma):
            equity_curve.append((date, cash + shares * close))
            continue

        # signal on info known by yesterday's close
        want_buy = pc > pma and shares == 0
        want_sell = pc < pma and shares > 0

        if want_buy:
            limit = round_buy(pc)
            if lo <= limit <= hi:
                cost = buy_cost(limit, lots_per_trade)
                if cost <= cash:
                    cash -= cost
                    shares += lots_per_trade * SHARES_PER_LOT
                    trades.append((date, "BUY", limit, lots_per_trade, cost))

        elif want_sell:
            limit = round_sell(pc)
            if lo <= limit <= hi:
                lots = shares // SHARES_PER_LOT
                proceeds = sell_proceeds(limit, lots)
                cash += proceeds
                shares = 0
                trades.append((date, "SELL", limit, lots, proceeds))

        equity_curve.append((date, cash + shares * close))

    eq = pd.DataFrame(equity_curve, columns=["date", "equity"])
    tr = pd.DataFrame(trades, columns=["date", "side", "price", "lots", "cashflow"])
    return eq, tr, cash, shares


def per_trade_returns(tr: pd.DataFrame) -> list[float]:
    """Realized-trade returns: pair each SELL with the most recent BUY."""
    rets = []
    open_buy = None
    for _, t in tr.iterrows():
        if t["side"] == "BUY":
            open_buy = t
        elif t["side"] == "SELL" and open_buy is not None:
            buy_px = open_buy["price"]
            sell_px = t["price"]
            rets.append((sell_px - buy_px) / buy_px)
            open_buy = None
    return rets


def main():
    df = fetch_range("2330", "2023-01-01", "2025-12-31")
    initial = 1e8
    eq, tr, cash, shares = run_ma20(df, lots_per_trade=1, capital=initial)

    last_equity = eq["equity"].iloc[-1]
    last_close = df["close"].iloc[-1]
    n_days = len(eq)

    cum_ret = (last_equity - initial) / initial
    buy_hold = (last_close - df["close"].iloc[0]) / df["close"].iloc[0]
    ann = cum_ret / n_days * 252 if n_days else 0

    rets = per_trade_returns(tr)
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    win_rate = len(wins) / len(rets) if rets else 0
    profit_factor = (
        sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    )

    print(f"TSMC 2330 — MA20 strategy, 1 lot per trade, NT${initial:,.0f} capital")
    print(f"period: {df['date'].min().date()} → {df['date'].max().date()} "
          f"({n_days} trading days)")
    print(f"final equity:      NT${last_equity:,.0f}")
    print(f"final cash:        NT${cash:,.0f}")
    print(f"final shares:      {shares} (mkt val NT${shares*last_close:,.0f})")
    print(f"trades executed:   {len(tr)} ({sum(tr['side']=='BUY')} buys, "
          f"{sum(tr['side']=='SELL')} sells)")
    print(f"completed rounds:  {len(rets)}")
    print()
    print(f"cumulative return: {cum_ret*100:+.2f}%")
    print(f"annualized (rules formula): {ann*100:+.2f}%")
    print(f"buy & hold same window:     {buy_hold*100:+.2f}%")
    print(f"per-trade win rate:         {win_rate*100:.1f}%")
    print(f"profit factor:              {profit_factor:.2f}")

    print("\nlast 8 trades:")
    print(tr.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
