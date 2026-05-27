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
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import fetch_range  # noqa: E402
from src.data.universe import (  # noqa: E402
    get_index_universe_codes, build_universe_schedule,
)

# Which TEJ index to train on. Override per-run by setting --universe on the
# CLI (parsed in main()) — this default is used by both __main__ and the
# bookkeeping fields written to report.json.
DEFAULT_UNIVERSE = "TW50"
from src.data.handler import (  # noqa: E402
    build_panel, PanelTensors, LABEL_HORIZON, SEQ_LEN, N_FIELDS,
)
from src.data.processor import (  # noqa: E402
    fit_robust_zscore, apply_robust_zscore, fillna,
)
from src.model.double_adapt import DoubleAdapt  # noqa: E402
from src.trainer.maml import (  # noqa: E402
    FOMAMLConfig, PretrainConfig, pretrain_offline,
)
from src.trainer.incremental import ILConfig, run_incremental  # noqa: E402
from src.evaluate.metrics import cross_sectional_ic  # noqa: E402
from src.evaluate.backtest_qlib import (  # noqa: E402
    QlibBTConfig as BTConfig,
    run_qlib_backtest_with_il,
)
from src.workflow.forecast import forecast_latest  # noqa: E402

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

# Pretrain on 2016-2021. The 2022-2023 window serves TWO roles in sequence:
#   1. During pretrain: per-epoch held-out validation (read-only, for early stop)
#   2. After pretrain: IL warm-up training (model receives bi-level updates)
# Backtest reporting starts in 2024 (data never touched by pretrain training,
# only by IL adaptation).
PRETRAIN_START = "2016-01-01"
PRETRAIN_END = "2021-12-31"      # 6 years
VAL_START = "2022-01-01"
VAL_END = "2023-12-31"           # 2 years -- clean held-out val DURING pretrain

IL_START = "2022-01-01"          # IL begins right after pretrain, including val period
IL_END = "auto"
BACKTEST_START = "2024-01-01"    # backtest reports only on 2024+ (post-warmup)


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
        if df.empty or len(df) < SEQ_LEN + LABEL_HORIZON + 1:
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

    panel = build_panel(bars, label_h=LABEL_HORIZON)
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

    pretrain_X = panel.slice_by_date(PRETRAIN_START, PRETRAIN_END).X
    norm_stats = fit_robust_zscore(pretrain_X)
    panel = PanelTensors(
        X=fillna(apply_robust_zscore(panel.X, norm_stats)),
        y=panel.y, dates=panel.dates, codes=panel.codes,
    )
    print(f"feature norm fit on {len(pretrain_X):,} pretrain rows")
    # Persist for deploy_today.py
    with open(OUT_DIR / "norm_stats.pkl", "wb") as f:
        pickle.dump(norm_stats, f)

    pre_train = panel.slice_by_date(PRETRAIN_START, PRETRAIN_END).drop_nan_labels()
    pre_val = panel.slice_by_date(VAL_START, VAL_END).drop_nan_labels()
    il_panel = panel.slice_by_date(
        pd.Timestamp(IL_START) - pd.Timedelta(days=90), HISTORY_END
    )
    print(f"pretrain: {len(pre_train):,}  |  val: {len(pre_val):,}  |  IL: {len(il_panel):,}")

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
    il_end_actual = (str(pd.Timestamp(panel.dates.max()).date())
                     if IL_END == "auto" else IL_END)
    print(f"IL plan: warmup {IL_START} → {BACKTEST_START}, then live-IL backtest "
          f"{BACKTEST_START} → {il_end_actual}")

    print("\n=== offline FOMAML pretrain ===")
    pre_res = pretrain_offline(model, pre_train, pre_val, fcfg, pcfg, device)
    torch.save(model.state_dict(), OUT_DIR / "pretrained.pt")
    pd.DataFrame(pre_res["history"]).to_csv(OUT_DIR / "pretrain_log.csv", index=False)
    print(f"saved pretrained.pt; best val IC = {pre_res['best_val_ic']:.4f}")

    # ---- Stage 2': IL warmup (2022-01 → BACKTEST_START) ----
    # Walks the pretrained checkpoint forward through the val window doing
    # per-20-day FOMAML adapts, producing the "after final preparation" model.
    print(f"\n=== IL warmup ({IL_START} → {BACKTEST_START}) — final preparation ===")
    warmup_cfg = ILConfig(start_date=IL_START, end_date=BACKTEST_START)
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

    # ---- Stage 3: live-IL backtest (BACKTEST_START → now) ----
    # Starts from final_prep.pt; walks forward in r=20-day windows; at each
    # step does fomaml_step (adapts model in-place) AND simulates portfolio
    # trades over the query window. Final model state -> final.pt.
    print(f"\n=== live-IL qlib backtest ({BACKTEST_START} → {il_end_actual}) ===")
    bt, step_log, preds = run_qlib_backtest_with_il(
        model, il_panel, bars, fcfg, bt_cfg,
        start_date=BACKTEST_START, end_date=il_end_actual, device=device,
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
        "pretrain_window": [PRETRAIN_START, PRETRAIN_END],
        "val_window": [VAL_START, VAL_END],
        "il_window": [IL_START, il_end_actual],
        "il_warmup": [IL_START, BACKTEST_START],
        "backtest_window": [BACKTEST_START, il_end_actual],
        "universe": universe_name,
        "n_candidate_pool": len(bars),
        "n_active_per_period": typical_size,
        "rebalances_per_year": 2,
        "n_rebalances": int(len(schedule)),
        "n_panel_rows": int(len(panel)),
        "n_pretrain_rows": int(len(pre_train)),
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
            "seq_len": SEQ_LEN, "n_fields": N_FIELDS, "label_horizon": LABEL_HORIZON,
            "fomaml": asdict(fcfg),
            "pretrain": asdict(pcfg),
            "il_warmup": asdict(warmup_cfg),
            "il_online_start": BACKTEST_START,
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
