"""TWSE market rules: tick rounding, lot size, trading costs, limit-lock detection."""

from __future__ import annotations
import math

import numpy as np

SHARES_PER_LOT = 1000


def tick_size(price: float) -> float:
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    if price < 100:
        return 0.10
    if price < 500:
        return 0.50
    if price < 1000:
        return 1.00
    return 5.00


def round_buy(price: float) -> float:
    t = tick_size(price)
    return math.floor(price / t + 1e-9) * t


def round_sell(price: float) -> float:
    t = tick_size(price)
    return math.ceil(price / t - 1e-9) * t


# Per-side trading costs used by the backtest (QlibBTConfig.costs selects one).
COST_TABLES = {
    "qlib": dict(open_cost=0.0005, close_cost=0.0015, min_cost=5.0),
    "twse": dict(
        open_cost=0.001425,  # 0.1425% commission
        close_cost=0.001425 + 0.003,  # 0.1425% commission + 0.3% sell tax
        min_cost=20.0,
    ),
}


def is_limit_locked(today_high: float, today_low: float, prev_close: float, threshold: float) -> bool:
    """Approximate limit-up/down detection: if the day's range is at the
    +/-threshold edge of prev_close. We block trades on those days."""
    if not np.isfinite(prev_close) or prev_close <= 0:
        return False
    limit_up = prev_close * (1 + threshold)
    limit_down = prev_close * (1 - threshold)
    # if both high and low are at one limit, it's locked
    return (today_low >= limit_up * 0.999) or (today_high <= limit_down * 1.001)
