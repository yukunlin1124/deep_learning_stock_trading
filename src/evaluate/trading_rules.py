"""TWSE tick rounding + cost model. Mirrors qlib's exchange-rule helpers."""
from __future__ import annotations
import math

COMMISSION_RATE = 0.001425
COMMISSION_MIN = 20.0
TAX_RATE = 0.003
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


def commission(notional: float) -> float:
    return max(notional * COMMISSION_RATE, COMMISSION_MIN)


def buy_cost(price: float, lots: int) -> float:
    notional = price * SHARES_PER_LOT * lots
    return notional + commission(notional)


def sell_proceeds(price: float, lots: int) -> float:
    notional = price * SHARES_PER_LOT * lots
    return notional - commission(notional) - notional * TAX_RATE


def round_trip_cost_bps(price: float) -> float:
    notional = price * SHARES_PER_LOT
    b = commission(notional)
    s = commission(notional) + notional * TAX_RATE
    return (b + s) / notional * 10000
