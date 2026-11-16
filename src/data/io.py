"""Read-only cache wrapper for per-(stock, year) OHLC bars.

Cache is populated by `src/scripts/import_tej.py` from TEJ's half-year CSVs
under `data/raw/TEJ/`. Once imported, all downstream code (panel
build, training, backtest, deploy) reads cached CSVs from `data/interim/` and
never hits the network.

Schema of each cache file (`data/interim/{code}_{year}.csv`):
    date, capacity, turnover, high, low, close, change,
    transaction_volume, stock_code_id, open

This module previously had yfinance + TWSE-direct fallback fetchers; both
were removed once the TEJ pipeline became the sole data source.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import CACHE_DIR, HISTORY_START, HISTORY_END
from src.utils.progress import pbar, log
from src.data.handler import PRED_HORIZON, SEQ_LEN
from src.data.universe import get_index_universe_codes

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(stock_code: str, year: int) -> Path:
    return CACHE_DIR / f"{stock_code}_{year}.csv"


def fetch_year(stock_code: str, year: int) -> pd.DataFrame:
    """Read one year of bars for one stock from the local cache.

    Returns an empty DataFrame (with the legacy schema) if the cache file
    doesn't exist or has only a header. Callers that need full coverage
    should run `src/scripts/import_tej.py` first.
    """
    p = _cache_path(stock_code, year)
    if not p.exists():
        return _empty_frame()
    return pd.read_csv(p, parse_dates=["date"])


def fetch_range(stock_code: str, start: str, end: str) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    frames = [fetch_year(stock_code, y) for y in range(s.year, e.year + 1)]
    df = pd.concat(frames, ignore_index=True) if frames else _empty_frame()
    if df.empty:
        return df
    mask = (df["date"] >= s) & (df["date"] <= e)
    return df[mask].sort_values("date").reset_index(drop=True)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "capacity",
            "turnover",
            "high",
            "low",
            "close",
            "change",
            "transaction_volume",
            "stock_code_id",
            "open",
        ]
    )


def load_bars(universe: str, start: str = HISTORY_START, end: str = HISTORY_END) -> dict[str, pd.DataFrame]:
    """Bars for every code ever in `universe` (TW50 / TM100 / BOTH).

    Skips codes with too few rows to build one feature window + label.
    """
    bars: dict[str, pd.DataFrame] = {}
    codes = get_index_universe_codes(universe)
    print(f"loading bars for {universe} universe ({len(codes)} codes)...")
    for code in pbar(codes, desc="loading bars", unit="stock"):
        try:
            df = fetch_range(code, start, end)
        except Exception as e:
            log(f"[skip] {code}: {e!r}")
            continue
        if df.empty or len(df) < SEQ_LEN + PRED_HORIZON + 1:
            log(f"[skip-thin] {code}: only {len(df)} rows")
            continue
        bars[code] = df
    print(f"loaded {len(bars)} candidate stocks (pool)")
    return bars


if __name__ == "__main__":
    import time

    t0 = time.monotonic()
    df = fetch_range("2330", "2024-01-01", "2024-03-31")
    print(f"fetched {len(df)} rows in {time.monotonic() - t0:.2f}s")
    print(df.head())
    print("dtypes:\n", df.dtypes)
