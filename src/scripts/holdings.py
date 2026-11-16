"""Print current holdings from the course virtual TWSE system.

Reads credentials from env vars TWSE_ACCOUNT / TWSE_PASSWORD (or --account /
--password CLI flags). Cross-references against the most recent
the current run's orders_YYYYMMDD.csv to mark which of today's orders filled.

Usage:
    $env:TWSE_ACCOUNT="..."
    $env:TWSE_PASSWORD="..."
    python src/scripts/holdings.py
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from pathlib import Path

import pandas as pd

from src.trading.broker import get_raw_holdings, parse_holdings

from src import runs
from src.trading.rules import SHARES_PER_LOT


def main() -> None:
    ap = argparse.ArgumentParser(description="Print current holdings + match against today's orders")
    ap.add_argument("--account", default=os.environ.get("TWSE_ACCOUNT"))
    ap.add_argument("--password", default=os.environ.get("TWSE_PASSWORD"))
    args = ap.parse_args()
    if not args.account or not args.password:
        raise SystemExit("missing TWSE_ACCOUNT / TWSE_PASSWORD (env vars or CLI flags).")

    raw = get_raw_holdings(args.account, args.password)
    holdings = parse_holdings(raw)

    print(f"=== Holdings for account ending in ...{args.account[-3:]} ===")
    print(f"raw response type: {type(raw).__name__}")
    if isinstance(raw, list):
        print(f"raw list length: {len(raw)}")
        if raw:
            print(f"  first item type: {type(raw[0]).__name__}")
            print(f"  first item: {raw[0]!r}")
            if len(raw) > 1:
                print(f"  ...and {len(raw) - 1} more")
    elif isinstance(raw, dict):
        print(f"raw dict keys: {list(raw.keys())[:10]}")
        print(f"raw sample: {repr(raw)[:300]}")
    elif isinstance(raw, str):
        print(f"raw string sample: {raw[:300]!r}")
    print(f"parsed {len(holdings)} positions\n")

    if holdings:
        rows = [{"code": c, "shares": s, "lots": s // SHARES_PER_LOT} for c, s in sorted(holdings.items())]
        print(pd.DataFrame(rows).to_string(index=False))
        print()

    # Cross-reference with today's order log if present (an artifact of the current run).
    today = datetime.now().strftime("%Y%m%d")
    run_name = runs.read_current()  # no current run -> just report "no order log"
    log_path = runs.resolve_run(run_name) / f"orders_{today}.csv" if run_name else Path("<no current run>")
    if log_path.exists():
        orders = pd.read_csv(log_path)
        attempted = orders[orders["status"] == "OK"][["side", "code", "lots", "price"]].copy()
        attempted["code"] = attempted["code"].astype(str)
        if not attempted.empty:
            print(f"=== Today's submitted orders ({len(attempted)}) vs holdings ===")
            attempted["in_holdings"] = attempted["code"].isin(holdings.keys())
            print(attempted.to_string(index=False))
            filled = attempted["in_holdings"].sum()
            print(
                f"\n{filled} / {len(attempted)} BUY orders appear in current holdings "
                f"(SELLs would show as REMOVED, not present)"
            )
    else:
        print(f"(no order log at {log_path})")


if __name__ == "__main__":
    main()
