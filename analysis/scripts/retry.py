"""Retry (stock, year) pairs that failed or never landed during scrape.

Parses scrape.log for FAIL lines and ALSO scans for any (code, year) in the
universe whose cache CSV is missing -- this rescues stocks marked "bad"
mid-scrape that had their remaining years skipped.

Each pair is retried at most once. Lingering failures are logged and we move on.
"""
from __future__ import annotations

import re
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.data.io import fetch_year, CACHE_DIR  # noqa: E402
from analysis.data.universe import get_candidate_pool  # noqa: E402

START_YEAR = 2016
END_YEAR = 2026
LOG = CACHE_DIR / "retry.log"
SCRAPE_LOG = CACHE_DIR / "scrape.log"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_failures() -> list[tuple[str, int]]:
    if not SCRAPE_LOG.exists():
        return []
    pairs: list[tuple[str, int]] = []
    pat = re.compile(r"FAIL\s+(\d+)\s+(\d{4})")
    for line in SCRAPE_LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            pairs.append((m.group(1), int(m.group(2))))
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int]] = []
    for p in pairs:
        if p not in seen:
            out.append(p); seen.add(p)
    return out


def missing_pairs_in_universe() -> list[tuple[str, int]]:
    universe = get_candidate_pool()
    out: list[tuple[str, int]] = []
    for c in universe:
        for y in range(START_YEAR, END_YEAR + 1):
            if not (CACHE_DIR / f"{c}_{y}.csv").exists():
                out.append((c, y))
    return out


def main() -> None:
    fail_pairs = parse_failures()
    miss_pairs = missing_pairs_in_universe()
    todo = list(dict.fromkeys(fail_pairs + miss_pairs))
    log(f"retry: {len(fail_pairs)} prior-FAIL + {len(miss_pairs)} missing "
        f"-> {len(todo)} unique (stock, year) pairs")
    if not todo:
        log("nothing to retry"); return
    t0 = time.monotonic()
    ok = 0
    still_failed = 0
    for i, (code, year) in enumerate(todo):
        try:
            df = fetch_year(code, year)
            if df.empty:
                log(f"empty {code} {year}")
                still_failed += 1
            else:
                ok += 1
        except Exception as e:
            log(f"FAIL again {code} {year}: {e!r}")
            traceback.print_exc()
            still_failed += 1
        if (i + 1) % 10 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (len(todo) - i - 1) / rate if rate else float("inf")
            log(f"progress {i+1}/{len(todo)}, ok={ok} fail={still_failed}, "
                f"ETA {eta/60:.1f} min")
    log(f"RETRY DONE in {(time.monotonic()-t0)/60:.1f} min: "
        f"ok={ok}, still_failed={still_failed}")


if __name__ == "__main__":
    main()
