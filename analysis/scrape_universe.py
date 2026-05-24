"""Bulk scrape (stock, year) pairs into local CSV cache.

Designed to be safe to interrupt and resume — data_io.fetch_year only
hits the network on cache miss. Logs progress to scrape.log so you can
tail it during the overnight run.

Years 2016..2026 by default. Skip stocks whose 2025 fetch returns empty
(probably suspended) to avoid wasting more network on them.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_io import fetch_year, CACHE_DIR  # noqa: E402
from universe import get_universe  # noqa: E402

START_YEAR = 2016
END_YEAR = 2026

LOG = CACHE_DIR / "scrape.log"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    universe = get_universe()
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
                if y == END_YEAR and df.empty and bad_stock is False:
                    # current year empty is fine (early in year), don't flag
                    pass
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
