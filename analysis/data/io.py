"""Local caching wrapper around stock_api.get_taiwan_stock_data.

Each TWSE fetch sleeps 2s per month so we cache per (stock_code, year) CSV
on disk and only hit the network for missing years.

CACHE_DIR points at the project-level analysis/_cache regardless of where
this module is imported from.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from stock_api import get_taiwan_stock_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "analysis" / "_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(stock_code: str, year: int) -> Path:
    return CACHE_DIR / f"{stock_code}_{year}.csv"


def fetch_year(stock_code: str, year: int) -> pd.DataFrame:
    p = _cache_path(stock_code, year)
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        return df
    df = get_taiwan_stock_data(stock_code, f"{year}-01-01", f"{year}-12-31")
    df.to_csv(p, index=False)
    return df


def fetch_range(stock_code: str, start: str, end: str) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    frames = [fetch_year(stock_code, y) for y in range(s.year, e.year + 1)]
    df = pd.concat(frames, ignore_index=True)
    mask = (df["date"] >= s) & (df["date"] <= e)
    return df[mask].sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    df = fetch_range("2330", "2024-01-01", "2024-03-31")
    print(df.head())
    print("rows:", len(df), "cols:", list(df.columns))
    print("dtypes:\n", df.dtypes)
