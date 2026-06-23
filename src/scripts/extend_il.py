"""Run a single missing IL step against an existing run's final.pt.

Surgical continuation: loads run9 (or any run)'s final.pt + norm_stats +
universe, builds the panel through whatever bars are now cached, then fires
one FOMAML step that picks up exactly where il_log.csv left off (train =
prev test window, test = next r=20 trading days). Saves the updated weights
back to final.pt with a timestamped backup, appends one row to il_log.csv,
and appends per-stock preds to predictions.csv.

Usage:
    python src/scripts/extend_il.py --run run9            # auto: next step
    python src/scripts/extend_il.py --run run9 --dry-run  # show plan, no writes

Skips automatically if the next test window isn't fully past yet (i.e., not
enough trading days with labels exist to close the next r-day query window).
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.data.io import fetch_range
from src.data.universe import get_index_universe_codes, build_universe_schedule
from src.data.handler import build_panel, PanelTensors, PRED_HORIZON, SEQ_LEN, N_FIELDS
from src.data.processor import apply_robust_zscore, fillna
from src.model.double_adapt import DoubleAdapt
from src.trainer.maml import FOMAMLConfig
from src.trainer.incremental import ILConfig, run_incremental
from src.scripts.train import apply_time_varying_universe, HISTORY_START


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir name, e.g. run9")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan; do not mutate final.pt / il_log.csv / predictions.csv")
    args = ap.parse_args()

    run_dir = PROJECT_ROOT / "src" / "output" / args.run
    if not run_dir.exists():
        sys.exit(f"missing run dir: {run_dir}")

    report = json.loads((run_dir / "report.json").read_text())
    universe = report["universe"]
    fcfg_dict = report["model_config"]["fomaml"]
    fcfg = FOMAMLConfig(**{k: v for k, v in fcfg_dict.items()
                           if k in FOMAMLConfig.__dataclass_fields__})

    scheme_path = run_dir / "feature_scheme.json"
    feature_mode = (json.loads(scheme_path.read_text()).get("mode", "qlib")
                    if scheme_path.exists() else "per_field")
    print(f"run: {args.run}  universe: {universe}  feature_mode: {feature_mode}  r: {fcfg.r}")

    il_log = pd.read_csv(run_dir / "il_log.csv")
    last_step = int(il_log["step"].iloc[-1])
    test_end_col = "test_end" if "test_end" in il_log.columns else "query_end"
    last_test_end = pd.Timestamp(il_log[test_end_col].iloc[-1])
    next_test_start = last_test_end + pd.Timedelta(days=1)
    print(f"last IL step: {last_step}  test_end: {last_test_end.date()}  "
          f"-> next step will train through {last_test_end.date()}, "
          f"test starting {next_test_start.date()}+")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"loading bars ({universe})...")
    bars = {}
    today = pd.Timestamp.today().normalize()
    for code in get_index_universe_codes(universe):
        try:
            df = fetch_range(code, HISTORY_START, str(today.date()))
        except Exception as e:
            print(f"  [skip] {code}: {e!r}")
            continue
        if df.empty or len(df) < SEQ_LEN + PRED_HORIZON + 1:
            continue
        bars[code] = df
    print(f"loaded {len(bars)} stocks")

    panel = build_panel(bars, pred_h=PRED_HORIZON, mode=feature_mode)
    print(f"raw panel: {len(panel):,}  dates: "
          f"{pd.Timestamp(panel.dates.min()).date()} -> {pd.Timestamp(panel.dates.max()).date()}")

    schedule = build_universe_schedule(universe, HISTORY_START, str(today.date()))
    panel = apply_time_varying_universe(panel, schedule)

    with open(run_dir / "norm_stats.pkl", "rb") as f:
        norm_stats = pickle.load(f)
    panel = PanelTensors(
        X=fillna(apply_robust_zscore(panel.X, norm_stats)),
        y=panel.y, dates=panel.dates, codes=panel.codes,
    )
    print(f"after universe mask + norm: {len(panel):,} rows")

    udates = np.unique(panel.dates)
    next_start_np = np.datetime64(next_test_start)
    i_next = int(np.searchsorted(udates, next_start_np, side="left"))
    if i_next < fcfg.r:
        sys.exit(f"not enough lookback before {next_test_start.date()} "
                 f"(need {fcfg.r} prior trading days for train window)")
    if i_next + fcfg.r > len(udates):
        avail = len(udates) - i_next
        sys.exit(f"next test window not fully closed yet: only {avail}/{fcfg.r} "
                 f"trading days available after {last_test_end.date()}. "
                 f"Wait until panel extends another {fcfg.r - avail} trading days.")

    test_start = pd.Timestamp(udates[i_next])
    test_end_predicted = pd.Timestamp(udates[i_next + fcfg.r - 1])
    train_start = pd.Timestamp(udates[i_next - fcfg.r])
    print(f"PLAN: train {train_start.date()}..{pd.Timestamp(udates[i_next-1]).date()}  "
          f"test {test_start.date()}..{test_end_predicted.date()}")

    if args.dry_run:
        print("--dry-run -> not mutating anything")
        return

    print(f"loading final.pt -> model")
    model = DoubleAdapt(seq_len=SEQ_LEN, n_fields=N_FIELDS).to(device)
    model.load_state_dict(torch.load(run_dir / "final.pt", map_location=device))

    icfg = ILConfig(start_date=str(test_start.date()), end_date=str(test_end_predicted.date()))
    t0 = time.monotonic()
    new_log, new_preds = run_incremental(model, panel, fcfg, icfg, device)
    print(f"IL step done in {time.monotonic() - t0:.1f}s; got {len(new_log)} step(s)")

    if new_log.empty:
        print("WARNING: run_incremental produced 0 steps -- nothing to append, no save")
        return

    new_log["step"] = new_log["step"] + (last_step + 1)
    if test_end_col == "query_end":
        new_log = new_log.rename(columns={
            "train_start": "support_start", "train_end": "support_end",
            "test_start": "query_start", "test_end": "query_end",
            "n_train": "n_support", "n_test": "n_query",
        })
    print("new step(s):")
    print(new_log.to_string(index=False))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = run_dir / f"final_pre_step{last_step + 1}_{stamp}.pt"
    shutil.copy(run_dir / "final.pt", backup)
    torch.save(model.state_dict(), run_dir / "final.pt")
    print(f"backed up old weights -> {backup.name}; wrote new final.pt")

    pd.concat([il_log, new_log], ignore_index=True).to_csv(run_dir / "il_log.csv", index=False)
    print(f"appended {len(new_log)} row(s) to il_log.csv")

    preds_path = run_dir / "predictions.csv"
    if preds_path.exists() and not new_preds.empty:
        old_preds = pd.read_csv(preds_path)
        pd.concat([old_preds, new_preds], ignore_index=True).to_csv(preds_path, index=False)
        print(f"appended {len(new_preds)} row(s) to predictions.csv")


if __name__ == "__main__":
    main()
