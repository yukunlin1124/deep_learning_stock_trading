"""qlib-style cross-run metrics vs 0050 benchmark.

Reads each run's equity.csv + report.json, aligns to the 0050 daily
series from data/interim/0050_benchmark.csv, and prints the table that
lives in docs/model_performance.md.

Formulas match qlib.contrib.evaluate.risk_analysis (additive
annualization: mean * 252; additive max drawdown: cumsum then cummax).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import runs
from src.config import CACHE_DIR

BENCHMARK_CSV = CACHE_DIR / "0050_benchmark.csv"
RUNS = [
    "run1",
    "run2",
    "run3",
    "run4",
    "run5",
    "run6",
    "run7",
    "run8",
    "run9",
    "run10",
    "run11",
    "run12",
    "run13",
    "run14",
    "run15",
    "run16",
    "run17",
    "run18",
    "run19",
    "run20",
]


def _load_benchmark() -> pd.DataFrame:
    bm = pd.read_csv(BENCHMARK_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    bm["bm_ret"] = bm["close"].pct_change()
    return bm


def _run_metrics(run: str, bm: pd.DataFrame) -> dict:
    d = runs.resolve_run(run)
    rpt = json.loads((d / "report.json").read_text())
    bt = rpt["backtest"]
    eq = pd.read_csv(d / "equity.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    eq["st_ret"] = eq["equity"].pct_change()

    m = pd.merge(eq[["date", "st_ret"]], bm[["date", "bm_ret"]], on="date", how="inner").dropna()
    m["excess"] = m["st_ret"] - m["bm_ret"]

    excess_ann = float(m["excess"].mean() * 252)
    ir = float(m["excess"].mean() / m["excess"].std() * np.sqrt(252))
    cum_excess = m["excess"].cumsum()
    excess_dd = float((cum_excess - cum_excess.cummax()).min())

    return {
        "run": run,
        "IC": rpt["il_overall_ic"],
        "ICIR": rpt["il_overall_icir"],
        "RankIC": rpt["il_overall_ric"],
        "RankICIR": rpt["il_overall_ricir"],
        "excess_ann_ret": excess_ann,
        "IR": ir,
        "max_dd": bt["max_drawdown"],
        "excess_max_dd": excess_dd,
    }


def main() -> None:
    bm = _load_benchmark()
    bm_in = bm.dropna()
    bm_24 = bm_in[(bm_in["date"] >= "2024-01-02") & (bm_in["date"] <= "2026-04-30")]

    print(f"Benchmark 0050.TW, {bm_24['date'].min().date()} -> {bm_24['date'].max().date()} (n={len(bm_24)}):")
    print(f"  qlib-ann ret (mean*252)   : {bm_24['bm_ret'].mean() * 252:+.2%}")
    print(f"  cumulative return         : {(1 + bm_24['bm_ret']).prod() - 1:+.2%}")
    print(f"  annualized vol            : {bm_24['bm_ret'].std() * np.sqrt(252):.2%}")
    print()

    rows = [_run_metrics(r, bm) for r in RUNS]
    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", "{:+.4f}".format)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
