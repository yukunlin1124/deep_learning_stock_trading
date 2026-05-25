"""Print current holdings from the course virtual TWSE system.

Reads credentials from env vars TWSE_ACCOUNT / TWSE_PASSWORD (or --account /
--password CLI flags). Cross-references against the most recent
src/output/orders_YYYYMMDD.csv to mark which of today's orders filled.

Usage:
    $env:TWSE_ACCOUNT="..."
    $env:TWSE_PASSWORD="..."
    .venv/Scripts/python.exe src/scripts/holdings.py
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.deploy_today import parse_holdings  # noqa: E402
from stock_api import Get_User_Stocks  # noqa: E402

OUT_BASE = PROJECT_ROOT / "src" / "output"


def main() -> None:
    ap = argparse.ArgumentParser(description="Print current holdings + match against today's orders")
    ap.add_argument("--account", default=os.environ.get("TWSE_ACCOUNT"))
    ap.add_argument("--password", default=os.environ.get("TWSE_PASSWORD"))
    args = ap.parse_args()
    if not args.account or not args.password:
        raise SystemExit("missing TWSE_ACCOUNT / TWSE_PASSWORD (env vars or CLI flags).")

    raw = Get_User_Stocks(args.account, args.password)
    holdings = parse_holdings(raw)

    print(f"=== Holdings for account ending in ...{args.account[-3:]} ===")
    print(f"raw response type: {type(raw).__name__}")
    print(f"parsed {len(holdings)} positions\n")

    if holdings:
        rows = [{"code": c, "shares": s, "lots": s // 1000}
                for c, s in sorted(holdings.items())]
        print(pd.DataFrame(rows).to_string(index=False))
        print()

    # Cross-reference with today's order log if present.
    today = datetime.now().strftime("%Y%m%d")
    log_path = OUT_BASE / f"orders_{today}.csv"
    if log_path.exists():
        orders = pd.read_csv(log_path)
        attempted = orders[orders["status"] == "OK"][["side", "code", "lots", "price"]].copy()
        attempted["code"] = attempted["code"].astype(str)
        if not attempted.empty:
            print(f"=== Today's submitted orders ({len(attempted)}) vs holdings ===")
            attempted["in_holdings"] = attempted["code"].isin(holdings.keys())
            print(attempted.to_string(index=False))
            filled = attempted["in_holdings"].sum()
            print(f"\n{filled} / {len(attempted)} BUY orders appear in current holdings "
                  f"(SELLs would show as REMOVED, not present)")
    else:
        print(f"(no order log at {log_path})")


if __name__ == "__main__":
    main()
