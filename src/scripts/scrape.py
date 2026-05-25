"""Bulk scrape (stock, year) pairs into the local CSV cache.

Safe to interrupt and resume -- data.io.fetch_year only hits the network
on cache miss. Logs progress to scrape.log.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# Allow running as `python src/scripts/scrape.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.io import fetch_year, CACHE_DIR  # noqa: E402
from src.data.universe import get_candidate_pool  # noqa: E402

START_YEAR = 2016
END_YEAR = 2026

LOG = CACHE_DIR / "scrape.log"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    universe = get_candidate_pool()
    years = list(range(START_YEAR, END_YEAR + 1))
    total = len(universe) * len(years)
    log(f"start: {len(universe)} stocks x {len(years)} years = {total} fetches")
    t0 = time.monotonic()
    done = 0

    for i, code in enumerate(universe):
        bad_stock = False
        for y in years:
            done += 1
            try:
                df = fetch_year(code, y)
                if df.empty and y < END_YEAR - 1:
                    bad_stock = True
            except Exception as e:
                log(f"FAIL {code} {y}: {e!r}")
                traceback.print_exc()
                bad_stock = True
                continue

            if done % 20 == 0 or y == years[-1]:
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed else 0
                eta = (total - done) / rate if rate else float("inf")
                log(f"progress {done}/{total} ({done/total:.1%}), "
                    f"ETA {eta/60:.1f} min, last={code}/{y}")

            if bad_stock:
                log(f"skip rest of {code} (empty year {y})")
                break

        if (i + 1) % 5 == 0:
            log(f"stock {i+1}/{len(universe)} done ({code})")

    log(f"DONE in {(time.monotonic()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
