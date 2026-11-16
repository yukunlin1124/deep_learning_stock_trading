"""TWSE tick rounding + lot size. (Trading costs live in backtest_qlib._COST_TABLES.)"""
from __future__ import annotations
import math

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

