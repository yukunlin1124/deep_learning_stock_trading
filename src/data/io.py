"""Read-only cache wrapper for per-(stock, year) OHLC bars.

Cache is populated by `src/scripts/import_tej.py` from TEJ's half-year CSVs
under `src/data/stock_data_all/`. Once imported, all downstream code (panel
build, training, backtest, deploy) reads cached CSVs from `src/_cache/` and
never hits the network.

Schema of each cache file (`src/_cache/{code}_{year}.csv`):
    date, capacity, turnover, high, low, close, change,
    transaction_volume, stock_code_id, open

This module previously had yfinance + TWSE-direct fallback fetchers; both
were removed once the TEJ pipeline became the sole data source.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "src" / "_cache"
CACHE_DIR.mkdir(exist_ok=True)


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
    return pd.DataFrame(columns=[
        "date", "capacity", "turnover", "high", "low", "close",
        "change", "transaction_volume", "stock_code_id", "open",
    ])


if __name__ == "__main__":
    import time
    t0 = time.monotonic()
    df = fetch_range("2330", "2024-01-01", "2024-03-31")
    print(f"fetched {len(df)} rows in {time.monotonic()-t0:.2f}s")
    print(df.head())
    print("dtypes:\n", df.dtypes)
