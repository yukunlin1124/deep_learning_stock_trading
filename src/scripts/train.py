"""DoubleAdapt training pipeline orchestrator.

Pipeline:
  1. Load cached daily bars for the candidate pool.
  2. Build Alpha360 features (60-day lookback x 6 fields = 360 dims per row).
  3. Build semi-annual top-50 active-universe schedule (by trailing 180d $vol).
  4. Mask panel by active universe; fit RobustZScoreNorm on pretrain rows.
  5. Offline FOMAML pretrain on [2016..2021] with val on [2022..2023];
     early-stop on val IC.
  6. Fake-online IL across [2024..panel_max], r=20 trading days/step.
  7. Portfolio backtest over the IL window using top-K limit-at-prev-close.
  8. Live forecast on the freshest 20 label-less dates.

Artifacts -> src/models/.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow running as `python src/scripts/train.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.data.io import fetch_range
from src.data.universe import (
    get_index_universe_codes, build_universe_schedule,
)

# Which TEJ index to train on. Override per-run by setting --universe on the
# CLI (parsed in main()) — this default is used by both __main__ and the
# bookkeeping fields written to report.json.
DEFAULT_UNIVERSE = "TW50"
from src.data.handler import (
    build_panel, PanelTensors, PRED_HORIZON, SEQ_LEN, N_FIELDS, FEATURE_MODE, FIELDS,
)
from src.data.processor import (
    fit_robust_zscore, apply_robust_zscore, fillna,
)
from src.model.double_adapt import DoubleAdapt
from src.trainer.maml import (
    FOMAMLConfig, PretrainConfig, pretrain_offline,
)
from src.trainer.incremental import ILConfig, run_incremental
from src.evaluate.metrics import cross_sectional_ic
from src.evaluate.backtest_qlib import (
    QlibBTConfig as BTConfig,
    run_qlib_backtest_with_il,
)
from src.workflow.forecast import forecast_latest

OUT_BASE = PROJECT_ROOT / "src" / "output"
OUT_BASE.mkdir(exist_ok=True)
LOGS_DIR = PROJECT_ROOT / "src" / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def _next_run_dir(base: Path) -> Path:
    """Find next available output/runN/. Auto-increments across re-runs.

    Skips empty `runN/` directories so test-imports / killed runs don't
    create permanent gaps in the numbering.
    """
    nums = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("run") and p.name[3:].isdigit():
            # Treat empty dirs as never-existed so we reuse the slot
            if any(p.iterdir()):
                nums.append(int(p.name[3:]))
    next_n = (max(nums) + 1) if nums else 1
    return base / f"run{next_n}"


class _Tee:
    """Write stdout to BOTH terminal and a log file."""
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def _init_run_output(base: Path) -> Path:
    """Allocate the next run dir, point CURRENT at it, tee stdout to its log.

    Called only from main() so importing train.py doesn't allocate a run.
    """
    out_dir = _next_run_dir(base)
    out_dir.mkdir(exist_ok=True)
    (base / "CURRENT").write_text(out_dir.name)
    print(f"writing artifacts to {out_dir}")
    import time as _time
    log_path = LOGS_DIR / f"train_{out_dir.name}_{_time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"logging full training session to {log_path}")
    return out_dir


# OUT_DIR is set inside main() at runtime; default to OUT_BASE so module-level
# imports don't create directories. References later in the file pick it up
# via the global rebinding in main().
OUT_DIR = OUT_BASE

# Date windows
HISTORY_START = "2016-01-01"
HISTORY_END = "2026-05-23"

# Three operational phases, each consuming one of Algorithm 1's meta-sets:
#   - PRETRAIN phase (2016-2021)         → consumes the meta_train_set
#   - VALIDATION phase (2022-2023)       → consumes the meta_valid_set
#       Dual-purpose per the paper: (a) during pretrain, held-out for
#       early-stop / best-state restore (read-only via train=False);
#       (b) after pretrain, re-fed through DoubleAdapt for the "final
#       preparation" warmup IL with real bi-level updates.
#   - ONLINE_TRAINING phase (2024-now)   → consumes the meta_test_set
#       The qlib backtest fine-tunes the model every r=20 days while
#       simulating the portfolio (Algorithm 1 line 9 second half).
PRETRAIN_START         = "2016-01-01"
PRETRAIN_END           = "2021-12-31"   # 6 years
VALIDATION_START       = "2022-01-01"
VALIDATION_END         = "2023-12-31"   # 2 years -- held-out val + warmup IL
ONLINE_TRAINING_START  = "2024-01-01"   # live-IL backtest starts here
ONLINE_TRAINING_END    = "auto"         # extend to panel max


def load_bars(universe: str) -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    codes = get_index_universe_codes(universe)
    print(f"loading bars for {universe} universe ({len(codes)} codes)...")
    for code in codes:
        try:
            df = fetch_range(code, HISTORY_START, HISTORY_END)
        except Exception as e:
            print(f"[skip] {code}: {e!r}")
            continue
        if df.empty or len(df) < SEQ_LEN + PRED_HORIZON + 1:
            print(f"[skip-thin] {code}: only {len(df)} rows")
            continue
        bars[code] = df
    print(f"loaded {len(bars)} candidate stocks (pool)")
    return bars


def apply_time_varying_universe(panel: PanelTensors, schedule: pd.DataFrame
                                ) -> PanelTensors:
    """Keep only rows whose code is active on its date per the schedule."""
    if panel.X.shape[0] == 0 or schedule.empty:
        return panel
    keep = np.zeros(len(panel), dtype=bool)
    sched_asofs = pd.to_datetime(schedule["asof"]).values
    sched_sets = [set(codes) for codes in schedule["codes"]]
    panel_dates = pd.to_datetime(panel.dates).values
    asof_idx = np.searchsorted(sched_asofs, panel_dates, side="right") - 1
    asof_idx = np.clip(asof_idx, 0, len(sched_asofs) - 1)
    for row_i, (sidx, code) in enumerate(zip(asof_idx, panel.codes)):
        if str(code) in sched_sets[sidx]:
            keep[row_i] = True
    return PanelTensors(
        X=panel.X[keep], y=panel.y[keep],
        dates=panel.dates[keep], codes=panel.codes[keep],
    )


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["TW50", "TM100", "BOTH"], default=DEFAULT_UNIVERSE,
                    help="which TEJ index to use as the time-varying active universe "
                         "(BOTH = TW50 ∪ TM100, ~150 stocks/period)")
    args = ap.parse_args()
    universe_name = args.universe

    global OUT_DIR
    OUT_DIR = _init_run_output(OUT_BASE)
    t0 = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)} "
              f"capability={torch.cuda.get_device_capability(0)}")
    print(f"universe: {universe_name}")
    torch.manual_seed(0)
    np.random.seed(0)

    bars = load_bars(universe_name)
    if len(bars) < 5:
        print("WARNING: very few stocks loaded. Cache likely incomplete.")

    panel = build_panel(bars, pred_h=PRED_HORIZON)
    print(f"raw panel rows: {len(panel):,}; "
          f"dates: {pd.Timestamp(panel.dates.min()).date()} -> "
          f"{pd.Timestamp(panel.dates.max()).date()}")

    # Time-varying mask from the chosen TEJ index (monthly snapshots).
    schedule = build_universe_schedule(universe_name, HISTORY_START, HISTORY_END)
    schedule.to_csv(OUT_DIR / "universe_schedule.csv", index=False)
    typical_size = int(schedule["n_active"].iloc[-1]) if not schedule.empty else 0
    print(f"universe schedule: {len(schedule)} rebalances, "
          f"~{typical_size} active per period ({universe_name})")
    panel = apply_time_varying_universe(panel, schedule)
    print(f"after universe mask: {len(panel):,} rows")

    meta_train_X = panel.slice_by_date(PRETRAIN_START, PRETRAIN_END).X
    norm_stats = fit_robust_zscore(meta_train_X)
    panel = PanelTensors(
        X=fillna(apply_robust_zscore(panel.X, norm_stats)),
        y=panel.y, dates=panel.dates, codes=panel.codes,
    )
    print(f"feature norm fit on {len(meta_train_X):,} meta-train rows")
    # Persist for deploy_today.py
    with open(OUT_DIR / "norm_stats.pkl", "wb") as f:
        pickle.dump(norm_stats, f)
    # Record which Alpha360 normalization was used so deploy_today.py can
    # reproduce matching features at inference time. Runs without this file
    # are assumed legacy "per_field" (= runs 1-16).
    (OUT_DIR / "feature_scheme.json").write_text(json.dumps({
        "mode": FEATURE_MODE,
        "seq_len": SEQ_LEN,
        "n_fields": N_FIELDS,
        "fields": list(FIELDS),
    }, indent=2))

    # Algorithm-1 meta-sets:
    #   meta_train_set ← PRETRAIN window         (offline pretrain)
    #   meta_valid_set ← VALIDATION window       (held-out val + warmup IL)
    #   meta_test_set  ← ONLINE_TRAINING window  (live-IL backtest)
    meta_train_set = panel.slice_by_date(PRETRAIN_START, PRETRAIN_END).drop_nan_labels()
    meta_valid_set = panel.slice_by_date(VALIDATION_START, VALIDATION_END).drop_nan_labels()
    # The IL panel needs to extend back ~90 days before VALIDATION_START so the
    # first warmup task's train window (VALIDATION_START − r days) has features.
    il_panel = panel.slice_by_date(
        pd.Timestamp(VALIDATION_START) - pd.Timedelta(days=90), HISTORY_END
    )
    print(f"meta_train_set: {len(meta_train_set):,}  |  "
          f"meta_valid_set: {len(meta_valid_set):,}  |  IL panel: {len(il_panel):,}")

    model = DoubleAdapt(seq_len=SEQ_LEN, n_fields=N_FIELDS).to(device)
    print(f"#DA params: {sum(p.numel() for p in model.da_parameters()):,}")
    print(f"#MA params: {sum(p.numel() for p in model.ma_parameters()):,}")

    # Run-3 tuning:
    #   - lr_psi 1e-3 (carried over from run 2): slower DA drift
    #   - tasks_per_epoch 200 -> 1000 (= r * TOP_N = 20 * 50): one task's worth
    #     of cross-sectional samples per epoch -> more diverse meta-batches
    #   - epochs 5 -> 30, patience 2 -> 8: more room for the meta-learner to
    #     plateau (paper-style patience instead of run-2's aggressive cut)
    fcfg = FOMAMLConfig(lr_psi=1e-3)
    pcfg = PretrainConfig(epochs=30, tasks_per_epoch=1000, early_stop_patience=8)
    online_end_actual = (str(pd.Timestamp(panel.dates.max()).date())
                         if ONLINE_TRAINING_END == "auto" else ONLINE_TRAINING_END)
    print(f"IL plan: warmup (meta-valid set) {VALIDATION_START} → {ONLINE_TRAINING_START}, "
          f"then online training / live-IL (meta-test set) "
          f"{ONLINE_TRAINING_START} → {online_end_actual}")

    # ---- Stage 1: offline pretrain on meta_train_set, validate on meta_valid_set ----
    print("\n=== offline FOMAML pretrain (meta_train_set) ===")
    pre_res = pretrain_offline(model, meta_train_set, meta_valid_set, fcfg, pcfg, device)
    torch.save(model.state_dict(), OUT_DIR / "pretrained.pt")
    pd.DataFrame(pre_res["history"]).to_csv(OUT_DIR / "pretrain_log.csv", index=False)
    print(f"saved pretrained.pt; best meta-valid IC = {pre_res['best_val_ic']:.4f}")

    # ---- Stage 2: IL warmup ("final preparation") on meta_valid_set ----
    # Re-feeds the meta_valid_set through DoubleAdapt with REAL bi-level updates
    # (Algorithm 1 line 9 first half). Walks the pretrained checkpoint forward
    # through the validation window producing the "after final preparation" model.
    print(f"\n=== IL warmup ({VALIDATION_START} → {ONLINE_TRAINING_START}) — "
          f"final preparation on meta_valid_set ===")
    warmup_cfg = ILConfig(start_date=VALIDATION_START, end_date=ONLINE_TRAINING_START)
    warmup_log, warmup_preds = run_incremental(model, il_panel, fcfg, warmup_cfg, device)
    torch.save(model.state_dict(), OUT_DIR / "final_prep.pt")
    warmup_log.to_csv(OUT_DIR / "warmup_log.csv", index=False)
    print(f"saved final_prep.pt; warmup did {len(warmup_log)} IL steps")

    # Persist trading config + active universe before the live-IL backtest.
    bt_cfg = BTConfig(top_k=10, n_drop=1, cost_alpha_bps=70.0,
                      costs="twse", deal_price="close")
    (OUT_DIR / "bt_config.json").write_text(json.dumps(asdict(bt_cfg), indent=2))
    final_universe = [str(c) for c in schedule.iloc[-1]["codes"]] if len(schedule) else list(bars.keys())
    (OUT_DIR / "active_universe.json").write_text(
        json.dumps({"asof": str(schedule.iloc[-1]["asof"]) if len(schedule) else None,
                    "codes": final_universe}, indent=2))

    # ---- Stage 3: online training / live-IL backtest on meta_test_set ----
    # Algorithm 1 line 9 second half: starts from final_prep.pt, walks forward
    # in r=20-day windows. At each step does fomaml_step (adapts model in-place
    # AND simulates portfolio trades over the test window). Final model state
    # -> final.pt.
    print(f"\n=== online training / live-IL qlib backtest "
          f"({ONLINE_TRAINING_START} → {online_end_actual}) — meta_test_set ===")
    bt, step_log, preds = run_qlib_backtest_with_il(
        model, il_panel, bars, fcfg, bt_cfg,
        start_date=ONLINE_TRAINING_START, end_date=online_end_actual, device=device,
    )
    torch.save(model.state_dict(), OUT_DIR / "final.pt")
    step_log.to_csv(OUT_DIR / "il_log.csv", index=False)
    preds.to_csv(OUT_DIR / "predictions.csv", index=False)
    bt.equity.to_csv(OUT_DIR / "equity.csv", index=False)
    bt.trades.to_csv(OUT_DIR / "trades.csv", index=False)
    bt.daily.to_csv(OUT_DIR / "backtest_daily.csv", index=False)
    print(f"saved final.pt; live-IL backtest: "
          f"{bt.summary['days']} days, {bt.summary['n_trades']} trades, "
          f"final equity={bt.summary['final_equity']:.0f}, "
          f"cum_ret={bt.summary['cum_return']:.4f}, "
          f"sharpe={bt.summary['sharpe']:.3f}, "
          f"max_dd={bt.summary['max_drawdown']:.4f}")

    live = forecast_latest(model, il_panel, n_days=20, device=device)
    live.to_csv(OUT_DIR / "live_forecast.csv", index=False)
    print(f"live forecast: {len(live)} rows over "
          f"{live['date'].nunique() if not live.empty else 0} dates")

    if not preds.empty:
        ic_all, icir_all, ric_all, ricir_all = cross_sectional_ic(
            preds["pred"].to_numpy(), preds["actual"].to_numpy(),
            preds["date"].to_numpy().astype("datetime64[ns]"),
        )
    else:
        ic_all = icir_all = ric_all = ricir_all = float("nan")

    report = {
        "history_start": HISTORY_START,
        "history_end": HISTORY_END,
        # Operational-phase date windows (the date-range constants in train.py)
        "pretrain_window":        [PRETRAIN_START, PRETRAIN_END],
        "validation_window":      [VALIDATION_START, VALIDATION_END],
        "online_training_window": [ONLINE_TRAINING_START, online_end_actual],
        # Mapping to Algorithm 1 of the DoubleAdapt paper (T_train, T_valid, T_test).
        # Each operational phase consumes one meta-set drawn from the matching window.
        "meta_train_set_window": [PRETRAIN_START, PRETRAIN_END],
        "meta_valid_set_window": [VALIDATION_START, VALIDATION_END],
        "meta_test_set_window":  [ONLINE_TRAINING_START, online_end_actual],
        "il_window":              [VALIDATION_START, online_end_actual],  # full rolling-IL span
        "il_warmup_window":       [VALIDATION_START, ONLINE_TRAINING_START],
        "universe": universe_name,
        "n_candidate_pool": len(bars),
        "n_active_per_period": typical_size,
        "rebalances_per_year": 2,
        "n_rebalances": int(len(schedule)),
        "n_panel_rows": int(len(panel)),
        "n_meta_train_set_rows": int(len(meta_train_set)),
        "n_meta_valid_set_rows": int(len(meta_valid_set)),
        "n_il_rows": int(len(il_panel)),
        "pretrain_best_val_ic": float(pre_res["best_val_ic"]),
        "pretrain_epochs_run": len(pre_res["history"]),
        "il_warmup_n_steps": int(len(warmup_log)),
        "il_online_n_steps": int(len(step_log)),
        "il_mean_ic": float(step_log["ic"].mean()) if len(step_log) else float("nan"),
        "il_mean_ric": float(step_log["ric"].mean()) if len(step_log) else float("nan"),
        "il_overall_ic": ic_all,
        "il_overall_icir": icir_all,
        "il_overall_ric": ric_all,
        "il_overall_ricir": ricir_all,
        "backtest": bt.summary,
        "backtest_cfg": {"capital": bt_cfg.capital, "top_k": bt_cfg.top_k,
                         "cost_alpha_bps": bt_cfg.cost_alpha_bps},
        "model_config": {
            "seq_len": SEQ_LEN, "n_fields": N_FIELDS, "pred_horizon": PRED_HORIZON,
            "fomaml": asdict(fcfg),
            "pretrain": asdict(pcfg),
            "il_warmup": asdict(warmup_cfg),
            "online_training_start": ONLINE_TRAINING_START,
        },
        "trained_at": pd.Timestamp.now().isoformat(),
        "universe_codes": list(bars.keys()),
        "wall_time_min": (time.monotonic() - t0) / 60.0,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n=== summary ===")
    print(f"  pretrain best val IC : {pre_res['best_val_ic']:.4f}")
    print(f"  IL steps             : {len(step_log)}")
    print(f"  IL overall IC        : {ic_all:.4f}  (ICIR {icir_all:.3f})")
    print(f"  IL overall RankIC    : {ric_all:.4f}  (RankICIR {ricir_all:.3f})")
    print(f"  Backtest cum return  : {bt.summary['cum_return']:+.4f}")
    print(f"  Backtest Sharpe      : {bt.summary['sharpe']:+.3f}")
    print(f"  Backtest max drawdown: {bt.summary['max_drawdown']:.4f}")
    print(f"  Backtest # trades    : {bt.summary['n_trades']}")
    print(f"  total wall-time      : {(time.monotonic() - t0) / 60:.1f} min")
    print(f"  artifacts in         : {OUT_DIR}")


if __name__ == "__main__":
    main()
