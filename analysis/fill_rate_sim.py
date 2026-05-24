"""Step 3 — Fill-rate simulator.

Rule 02 says an order fills iff its limit price is within [day_low, day_high].
We submit on day t for fill on day t (using only info known by t-1 close),
then mark the realized fill price (the post-tick-round limit if inside,
else no fill).

Compares several naive pricing rules against TSMC history.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import fetch_range  # noqa: E402
from trading_rules import (  # noqa: E402
    round_buy,
    round_sell,
    round_trip_cost_bps,
)


def simulate(df: pd.DataFrame, side: str, offset_pct: float):
    """For each day t, submit a limit at prev_close * (1 + offset_pct).
    Buy: round down to tick; Sell: round up. Fill if within [low_t, high_t]."""
    out = df.copy().reset_index(drop=True)
    prev_close = out["close"].shift(1)
    raw_limit = prev_close * (1 + offset_pct)

    if side == "buy":
        out["limit"] = raw_limit.apply(lambda x: round_buy(x) if pd.notna(x) else x)
    else:
        out["limit"] = raw_limit.apply(lambda x: round_sell(x) if pd.notna(x) else x)

    out["filled"] = (out["limit"] >= out["low"]) & (out["limit"] <= out["high"])
    out.loc[out["limit"].isna(), "filled"] = False
    out["fill_price"] = out["limit"].where(out["filled"])
    out["slip_vs_close"] = (out["fill_price"] - out["close"]) / out["close"]
    return out


def report(df: pd.DataFrame, label: str):
    n = df["limit"].notna().sum()
    filled = df["filled"].sum()
    rate = filled / n if n else float("nan")
    avg_slip = df.loc[df["filled"], "slip_vs_close"].mean()
    p25 = df.loc[df["filled"], "slip_vs_close"].quantile(0.25)
    p75 = df.loc[df["filled"], "slip_vs_close"].quantile(0.75)
    print(
        f"  {label:<32} "
        f"fill={filled}/{n} ({rate:.1%})  "
        f"slip_mean={avg_slip*100:+.3f}%  "
        f"p25/p75={p25*100:+.3f}/{p75*100:+.3f}%"
    )


def main():
    df = fetch_range("2330", "2023-01-01", "2025-12-31")
    print(f"TSMC 2330 — {len(df)} trading days {df['date'].min().date()} → "
          f"{df['date'].max().date()}")
    avg_px = df["close"].mean()
    print(f"avg close = {avg_px:.1f}, round-trip cost at avg = "
          f"{round_trip_cost_bps(avg_px):.1f} bps "
          f"(min commission can dominate small lots)\n")

    for side in ("buy", "sell"):
        print(f"[{side.upper()}] limit = prev_close * (1 + offset), one shot")
        for offset in (-0.010, -0.005, -0.002, 0.0, +0.002, +0.005, +0.010):
            sim = simulate(df, side, offset)
            label = f"offset={offset*100:+.1f}%"
            report(sim, label)
        print()


if __name__ == "__main__":
    main()
