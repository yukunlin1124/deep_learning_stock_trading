"""Compare our realistic backtest vs qlib-style backtest on the SAME predictions.

Run after train.py has produced run{N}/predictions.csv. Loads predictions
from the CURRENT run (or --run runX), then runs both backtests on the
identical signal and shows the diff.

Usage:
    .venv/Scripts/python.exe src/scripts/compare_backtests.py
    .venv/Scripts/python.exe src/scripts/compare_backtests.py --run run5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import fetch_range
from src.evaluate.backtest import BTConfig, run_backtest_from_predictions
from src.evaluate.backtest_qlib import QlibBTConfig, run_qlib_backtest_from_predictions

OUT_BASE = PROJECT_ROOT / "src" / "output"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="which output/runN/ to use (default: output/CURRENT)")
    args = ap.parse_args()

    if args.run:
        run_dir = OUT_BASE / args.run
    else:
        cur_file = OUT_BASE / "CURRENT"
        if not cur_file.exists():
            raise SystemExit(f"missing {cur_file}; pass --run runX.")
        run_dir = OUT_BASE / cur_file.read_text().strip()
    print(f"loading predictions from {run_dir.name}/")

    preds_path = run_dir / "predictions.csv"
    if not preds_path.exists():
        raise SystemExit(f"missing {preds_path}; run train.py first.")
    preds = pd.read_csv(preds_path)
    preds["date"] = pd.to_datetime(preds["date"])
    print(f"  {len(preds):,} predictions across {preds['date'].nunique()} dates, "
          f"{preds['stock_code_id'].nunique()} stocks\n")

    # Fetch bars for all stocks that appear in predictions
    codes_needed = set(preds["stock_code_id"].astype(str).unique())
    bars = {}
    for c in codes_needed:
        try:
            df = fetch_range(c, "2024-01-01", "2026-05-31")
            if not df.empty:
                bars[c] = df
        except Exception:
            pass
    print(f"  loaded bars for {len(bars)}/{len(codes_needed)} stocks\n")

    # ---- Configurations to compare ----
    configs = [
        ("ours (run5 config: top_k=10, n_drop=1, limit at prev_close)",
         "ours", BTConfig(capital=1e8, top_k=10, n_drop=1, cost_alpha_bps=70.0)),
        ("ours @ top_k=50, n_drop=5 (apples-to-apples count)",
         "ours", BTConfig(capital=1e8, top_k=50, n_drop=5, cost_alpha_bps=70.0)),
        ("qlib literal default (top_k=50, n_drop=5, close-fill, qlib CN costs)",
         "qlib", QlibBTConfig(capital=1e8, top_k=50, n_drop=5, costs="qlib")),
        ("qlib pricing + TWSE costs (top_k=50, n_drop=5)",
         "qlib", QlibBTConfig(capital=1e8, top_k=50, n_drop=5, costs="twse")),
        ("qlib pricing matched to our portfolio shape (top_k=10, n_drop=1, TWSE costs)",
         "qlib", QlibBTConfig(capital=1e8, top_k=10, n_drop=1, costs="twse")),
    ]

    rows = []
    for name, kind, cfg in configs:
        if kind == "ours":
            bt = run_backtest_from_predictions(preds, bars, cfg)
        else:
            bt = run_qlib_backtest_from_predictions(preds, bars, cfg)
        s = bt.summary
        rows.append({
            "config": name,
            "trades": s["n_trades"],
            "round_trips": s["n_round_trips"],
            "cum_return_%": round(s["cum_return"] * 100, 2),
            "sharpe": round(s["sharpe"], 3),
            "max_dd_%": round(s["max_drawdown"] * 100, 2),
            "win_rate_%": round(s["win_rate"] * 100, 1),
            "profit_factor": round(s["profit_factor"], 2) if np.isfinite(s["profit_factor"]) else float("inf"),
            "final_eq_M": round(s["final_equity"] / 1e6, 1),
        })

    df = pd.DataFrame(rows)
    print("=" * 110)
    print("Comparison (same run5 predictions, different backtest engines)")
    print("=" * 110)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    print(df.to_string(index=False))
    print()
    print("Reading the table:")
    print("  - 'ours' = limit at prev_close (tick-rounded), fill iff low<=limit<=high, TWSE costs+lot")
    print("  - 'qlib' = fills at today's close (always), no tick/lot rounding, costs as configured")
    print("  - The big diff is FILL RATE, not signal quality")


if __name__ == "__main__":
    main()
