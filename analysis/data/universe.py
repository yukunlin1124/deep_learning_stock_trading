"""TWSE universe: 150-code candidate pool, top-50 active per training period.

Layout:
  - CANDIDATE_CODES: ~170 well-known TWSE large/mid-caps, filtered down to
    the first 150 valid by load_symbol_map(). This is the SCRAPE target.
  - Per-rebalance ACTIVE training universe = top TOP_N (=50) of the pool by
    trailing-180-day mean(close * capacity). Rebalances twice a year.

So the **scrape pool** is fixed at 150 stocks; the **training subset** is the
top 50 of those 150 per six-month period, drifting with relative liquidity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stock_api import load_symbol_map

POOL_TARGET = 150               # candidate pool size (= scrape target)
TOP_N = 50                      # active training universe per period
REBALANCE_PER_YEAR = 2          # twice a year
LIQUIDITY_WINDOW_DAYS = 180     # trailing window for the ranking metric
MIN_TRAILING_ROWS = 10          # need at least this many rows to rank (cold-start safety)

CANDIDATE_CODES: list[str] = sorted(set(
    [str(c) for c in range(1101, 1111)]
    + [str(c) for c in range(1201, 1218)]
    + [str(c) for c in range(1301, 1322)]
    + [str(c) for c in range(1402, 1410)]
    + ["1326", "1434", "1440", "1476", "1503", "1504", "1590",
       "1605", "1717", "1722", "1789", "1802", "1907"]
    + [str(c) for c in range(2002, 2010)]
    + [str(c) for c in range(2104, 2110)]
    + [str(c) for c in range(2201, 2208)]
    + ["2227", "2231", "2603", "2609", "2615", "2618", "2633"]
    + ["2301", "2303", "2308", "2317", "2324", "2327", "2330", "2337",
       "2344", "2345", "2347", "2353", "2354", "2356", "2357", "2360",
       "2376", "2379", "2382", "2383", "2385", "2388", "2392", "2395",
       "2408", "2409", "2412", "2421", "2439", "2441", "2449", "2451",
       "2454", "2474", "2480", "2492", "2498"]
    + ["2801", "2809", "2812", "2816", "2820", "2823", "2832", "2834",
       "2845", "2849", "2867", "2880", "2881", "2882", "2883", "2884",
       "2885", "2886", "2887", "2888", "2890", "2891", "2892", "5876",
       "5880"]
    + ["2912", "3008", "3017", "3034", "3037", "3044", "3045", "3056",
       "3231", "3406", "3443", "3481", "3532", "3653", "3661", "3673",
       "3711", "4904", "4938", "4958", "5483", "5871", "6005", "6116",
       "6176", "6213", "6239", "6285", "6415", "6446", "6505", "6669",
       "8046", "8454", "9904", "9910", "9921", "9933", "9941", "9945",
       "9958"]
))


def get_candidate_pool() -> list[str]:
    """First POOL_TARGET (=150) valid TWSE codes from the candidate list."""
    m = load_symbol_map()
    out: list[str] = []
    seen: set[str] = set()
    for code in CANDIDATE_CODES:
        if code in seen:
            continue
        info = m.get(code)
        if info is None:
            continue
        if info.get("type") != "TWSE":
            continue
        out.append(code)
        seen.add(code)
        if len(out) >= POOL_TARGET:
            break
    return out


def get_universe() -> list[str]:
    """Legacy alias for the disk-scrape phase."""
    return get_candidate_pool()


def rebalance_dates(start: str, end: str,
                    rebalances_per_year: int = REBALANCE_PER_YEAR) -> list[pd.Timestamp]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    months_per_rb = max(1, 12 // rebalances_per_year)
    out: list[pd.Timestamp] = []
    cur = pd.Timestamp(year=s.year, month=1, day=1)
    while cur <= e:
        if cur >= s:
            out.append(cur)
        cur = cur + pd.DateOffset(months=months_per_rb)
    if not out or out[0] > s:
        out = [s] + out
    return out


def _trailing_liquidity(df: pd.DataFrame, asof: pd.Timestamp,
                        window_days: int) -> float:
    """Mean(close*capacity) over up to `window_days` rows ending <= asof."""
    if df.empty:
        return 0.0
    mask = df["date"] <= asof
    if not mask.any():
        return 0.0
    tail = df.loc[mask].sort_values("date").tail(window_days)
    if len(tail) < MIN_TRAILING_ROWS:
        return 0.0
    vals = (tail["close"].astype(float) * tail["capacity"].astype(float))
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else 0.0


def get_universe_at(asof: str | pd.Timestamp,
                    bars: dict[str, pd.DataFrame],
                    top_n: int = TOP_N,
                    window_days: int = LIQUIDITY_WINDOW_DAYS) -> list[str]:
    asof = pd.Timestamp(asof)
    scored = []
    for code, df in bars.items():
        liq = _trailing_liquidity(df, asof, window_days)
        if liq > 0:
            scored.append((code, liq))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:top_n]]


def build_universe_schedule(bars: dict[str, pd.DataFrame],
                            start: str, end: str,
                            top_n: int = TOP_N,
                            rebalances_per_year: int = REBALANCE_PER_YEAR,
                            window_days: int = LIQUIDITY_WINDOW_DAYS,
                            ) -> pd.DataFrame:
    rows = []
    for asof in rebalance_dates(start, end, rebalances_per_year):
        codes = get_universe_at(asof, bars, top_n=top_n, window_days=window_days)
        rows.append({"asof": asof.date(), "n_active": len(codes), "codes": codes})
    return pd.DataFrame(rows)


def active_codes_for_date(date: pd.Timestamp, schedule: pd.DataFrame) -> set[str]:
    if schedule.empty:
        return set()
    asofs = pd.to_datetime(schedule["asof"])
    mask = asofs <= date
    if not mask.any():
        return set(schedule.iloc[0]["codes"])
    idx = int(mask.values.nonzero()[0][-1])
    return set(schedule.iloc[idx]["codes"])


if __name__ == "__main__":
    u = get_candidate_pool()
    print(f"candidate pool: {len(u)} (target {POOL_TARGET})")
    print(f"training top-N: {TOP_N} per period, {REBALANCE_PER_YEAR}x/yr rebalance")
    print(u)
