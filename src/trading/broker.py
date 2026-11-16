"""Broker API: the course's virtual-TWSE `stock_api` (holdings, orders) plus the
yfinance intraday quote used by deploy_today.py --cheat.

This is the only module that imports `stock_api`. Scripts that talk to the
broker (deploy_today.py, holdings.py) go through here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from datetime import datetime

import numpy as np
import pandas as pd
from stock_api import Buy_Stock, Get_User_Stocks, Sell_Stock

from src.trading.rules import SHARES_PER_LOT


def get_raw_holdings(account: str, password: str) -> Any:
    """Raw Get_User_Stocks response (shape varies; see parse_holdings)."""
    return Get_User_Stocks(account, password)


def parse_holdings(raw: Any) -> dict[str, int]:
    """Convert Get_User_Stocks response into {code: shares_count}.

    Course API observed shape (one row per held stock):
      {'usid': ..., 'stock_name': ..., 'beginning_price': ...,
       'shares': 87,                  # <- LOTS, despite the name
       'stock_code_id': '2303',       # <- the canonical code key
       'createtime': ..., 'user_uid_id': ...}

    The `shares` field is in LOTS (1 lot = 1000 actual shares); the rest of
    deploy_today treats holdings[code] as a SHARE count and divides by
    SHARES_PER_LOT downstream, so we multiply on parse to keep one consistent
    internal unit.
    """
    holdings: dict[str, int] = {}
    if raw is None or raw == [] or raw == {}:
        return holdings

    def code_from(item: dict) -> str:
        return str(item.get("stock_code_id") or item.get("stock_code") or item.get("code") or "").strip()

    def lots_from(item: dict) -> int:
        v = item.get("shares") or item.get("quantity") or item.get("stock_shares") or 0
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    if isinstance(raw, dict):
        for key in ("data", "stocks", "holdings", "user_stocks"):
            if key in raw and isinstance(raw[key], (list, dict)):
                return parse_holdings(raw[key])
        for code, val in raw.items():
            code = str(code).strip()
            if not re.fullmatch(r"\d{4,5}", code):
                continue
            if isinstance(val, dict):
                lots = lots_from(val)
            else:
                try:
                    lots = int(val)
                except (TypeError, ValueError):
                    continue
            if lots > 0:
                holdings[code] = lots * SHARES_PER_LOT
        return holdings

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                code = code_from(item)
                lots = lots_from(item)
                if code and lots > 0:
                    holdings[code] = lots * SHARES_PER_LOT
        return holdings

    if isinstance(raw, str):
        for m in re.finditer(r"\b(\d{4})\b\D{1,40}?(\d+)", raw):
            holdings.setdefault(m.group(1), int(m.group(2)) * SHARES_PER_LOT)
        return holdings

    return holdings


def fetch_today_intraday(codes: list[str]) -> dict[str, tuple[float, float]]:
    """Cheat-mode helper: pull today's intraday OHLC via yfinance (~15 min
    delayed for the free tier).

    Returns {code: (today_low, today_high)} for stocks that have a valid
    today bar. Stocks without today's data (delisted, post-IPO blackout,
    market not open yet, network error, etc.) are omitted — the caller
    should skip them rather than fall back to prev_close, so that --cheat
    mode never silently degrades to the regular causal flow.
    """
    import yfinance as yf  # pylint: disable=import-outside-toplevel  # optional: only --cheat needs it

    today = datetime.now().date()
    tomorrow_str = (today.fromordinal(today.toordinal() + 1)).isoformat()
    tickers = " ".join(f"{c}.TW" for c in codes)
    df = yf.download(
        tickers,
        start=today.isoformat(),
        end=tomorrow_str,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )
    ranges: dict[str, tuple[float, float]] = {}
    if df is None or df.empty:
        return ranges
    multi = isinstance(df.columns, pd.MultiIndex)
    for code in codes:
        try:
            sub = df[f"{code}.TW"] if multi else df
            sub = sub.dropna(subset=["Low", "High"])
            if sub.empty:
                continue
            low = float(sub["Low"].iloc[-1])
            high = float(sub["High"].iloc[-1])
            if np.isfinite(low) and np.isfinite(high) and 0 < low <= high:
                ranges[code] = (low, high)
        except (KeyError, IndexError):
            continue
    return ranges


@dataclass
class IntendedOrder:
    side: str  # "BUY" or "SELL"
    code: str
    lots: int
    price: float  # int for the legacy prev-close path; tick-aligned float
    # in cheat mode so sub-NT$50 stocks land inside today's
    # [low, high] (tick=0.05 there, int rounding loses 95%
    # of valid limits and pushes the order outside the
    # fillable range).
    limit_source: str = "prev-close"  # "prev-close" | "cheat-high" | "cheat-low"


def submit_orders(orders: list[IntendedOrder], account: str, password: str, dry_run: bool) -> list[dict]:
    log_rows = []
    for o in orders:
        attempted_at = datetime.now().isoformat(timespec="seconds")
        if dry_run:
            success = None
        else:
            fn = Buy_Stock if o.side == "BUY" else Sell_Stock
            try:
                success = bool(fn(account, password, int(o.code), o.lots, o.price))
            except Exception as e:
                print(f"[error] {o.side} {o.code} x{o.lots} @ {o.price}: {e!r}")
                success = False
        status = "DRY_RUN" if dry_run else ("OK" if success else "FAIL")
        print(f"  [{status}] {o.side:4s} {o.code} lots={o.lots:3d} " f"price={o.price:>7.2f}  [{o.limit_source}]")
        log_rows.append(
            {
                "attempted_at": attempted_at,
                "side": o.side,
                "code": o.code,
                "lots": o.lots,
                "price": o.price,
                "limit_source": o.limit_source,
                "status": status,
            }
        )
    return log_rows
