"""Import TEJ half-year CSVs into our per-(stock, year) cache format.

Source: data/raw/TEJ/{train,val,test}/{YEAR}_H{1,2}.csv (cp950-encoded Big5).
        Split layout is described in data/raw/TEJ/data.yaml.
Output: data/interim/{code}_{year}.csv (UTF-8, our standard schema).

Filters to the TW50 historical union (~87 codes) so we only materialize
what training needs.

Schema mapping (TEJ -> our cache):
  date         <- col 2 (parsed YYYY/MM/DD -> ISO)
  open         <- col 3 (dividend-adjusted)
  high         <- col 4
  low          <- col 5
  close        <- col 6
  capacity     <- col 7 (成交量 in 千股 / thousand shares) * 1000 -> shares
  turnover     <- col 8 (成交值 in 百萬元 / million NT$)  * 1_000_000 -> NT$
  change       <- close.diff()
  transaction_volume <- 0  (not in TEJ; unused by compute_alpha360)
  stock_code_id<- col 0

TEJ values are SPLIT-ADJUSTED with stock-specific cumulative factors (e.g.
1101 listed since 1962 has ~35x cumulative adjustment vs 2330 listed 1994
at ~1x). So VWAP = turnover/capacity won't equal close, and absolute volume
levels aren't comparable to TWSE-raw. Alpha360's lag-0 normalization
(`win / win[-1]`) cancels per-stock scaling within each 60-day window, so
the model sees the same relative shape regardless of adjustment factor.

Usage:
    python src/scripts/import_tej.py
    python src/scripts/import_tej.py --since-year 2016
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.data.io import CACHE_DIR
from src.data.universe import get_index_universe_codes

from src.config import TEJ_DIR
# Half-year price files are split by period; see data/raw/TEJ/data.yaml.
TEJ_SPLITS = ("train", "val", "test")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _tej_path(year: int, half: str) -> Path | None:
    """Locate {year}_{half}.csv in whichever split folder holds it."""
    for split in TEJ_SPLITS:
        path = TEJ_DIR / split / f"{year}_{half}.csv"
        if path.exists():
            return path
    return None


def _to_float(s):
    if pd.isna(s) or s == "":
        return float("nan")
    return float(str(s).replace(",", ""))


def import_year(year: int, universe: set[str]) -> dict[str, int]:
    """Read H1 + H2 files for the year, return {code: n_rows_written}."""
    parts = []
    for half in ("H1", "H2"):
        path = _tej_path(year, half)
        if path is None:
            continue
        df = pd.read_csv(path, encoding="cp950", dtype=str)
        # Column indices are stable; alias them to readable names
        df.columns = [
            "stock_code_id", "name", "date_raw",
            "open", "high", "low", "close",
            "volume_k_shares",       # col 7: 成交量(千股) — thousand shares
            "turnover_M_ntd",        # col 8: 成交值(百萬元) — million NT$
            "mcap_tse", "mcap_tej", "pb_tse", "pb_tej", "ps_tej",
            "div_yield", "ret_pct",
            "warn_a", "disp_d", "circuit_y", "market",
        ]
        df = df[df["stock_code_id"].isin(universe)]
        if df.empty:
            continue
        parts.append(df)

    if not parts:
        return {}

    full = pd.concat(parts, ignore_index=True)
    full["date"] = pd.to_datetime(full["date_raw"], format="%Y/%m/%d")
    for c in ("open", "high", "low", "close", "volume_k_shares", "turnover_M_ntd"):
        full[c] = full[c].apply(_to_float)
    full = full.dropna(subset=["close"])

    # Convert from TEJ units to raw NT$ / raw shares (per stock; split-adjusted).
    full["capacity"] = (full["volume_k_shares"] * 1_000.0).round(0).astype("int64")
    full["turnover"] = (full["turnover_M_ntd"] * 1_000_000.0).round(0)
    full["transaction_volume"] = 0
    full["stock_code_id"] = full["stock_code_id"].astype(int)

    written = {}
    for code, sub in full.groupby("stock_code_id"):
        sub = sub.sort_values("date").copy()
        sub["change"] = sub["close"].diff().fillna(0.0)
        out = sub[["date", "capacity", "turnover", "high", "low", "close",
                   "change", "transaction_volume", "stock_code_id", "open"]]
        path = CACHE_DIR / f"{code}_{year}.csv"
        out.to_csv(path, index=False)
        written[str(code)] = len(out)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["TW50", "TM100", "BOTH"], default="TW50",
                    help="which TEJ index union to import (default TW50; "
                         "BOTH = union of TW50 ∪ TM100 ≈ 300 codes)")
    ap.add_argument("--since-year", type=int, default=2016,
                    help="oldest year to import (default 2016 to match HISTORY_START)")
    ap.add_argument("--until-year", type=int, default=2026,
                    help="newest year to import (default 2026)")
    args = ap.parse_args()

    if args.universe == "BOTH":
        universe = set(get_index_universe_codes("TW50")) | set(get_index_universe_codes("TM100"))
    else:
        universe = set(get_index_universe_codes(args.universe))
    print(f"{args.universe} universe: {len(universe)} codes")
    print(f"Reading TEJ files {args.since_year}_H1 .. {args.until_year}_H2 from {TEJ_DIR}")

    t0 = time.monotonic()
    total_files_written = 0
    coverage = {code: 0 for code in universe}
    for year in range(args.since_year, args.until_year + 1):
        if not any(_tej_path(year, h) for h in ("H1", "H2")):
            print(f"  {year}: no files, skip")
            continue
        written = import_year(year, universe)
        for code, n in written.items():
            coverage[code] += n
        total_files_written += len(written)
        print(f"  {year}: {len(written)} stocks, {sum(written.values()):>6,} rows")

    print(f"\nDONE in {time.monotonic()-t0:.1f}s — {total_files_written} CSVs in {CACHE_DIR}")

    # Per-stock coverage summary
    missing = [c for c, n in coverage.items() if n == 0]
    short = sorted([(c, n) for c, n in coverage.items() if 0 < n < 2000], key=lambda x: x[1])
    print(f"\n  Stocks with NO data: {len(missing)}  {missing}")
    print(f"  Stocks with <2000 bars (partial coverage): {len(short)}")
    for c, n in short[:10]:
        print(f"    {c}: {n} bars")


if __name__ == "__main__":
    main()
