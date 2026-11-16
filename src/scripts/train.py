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

Artifacts -> models/run{N}/ (see src/runs.py).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from src.data.io import load_bars
from src.data.universe import build_universe_schedule, apply_time_varying_universe

# Which TEJ index to train on. Override per-run by setting --universe on the
# CLI (parsed in main()) — this default is used by both __main__ and the
# bookkeeping fields written to report.json.
DEFAULT_UNIVERSE = "TW50"
from src.data.handler import (
    build_panel,
    PanelTensors,
    PRED_HORIZON,
    SEQ_LEN,
    N_FIELDS,
    FEATURE_MODE,
)
from src.data.processor import (
    fit_robust_zscore,
    apply_robust_zscore,
    fillna,
)
from src.model.double_adapt import DoubleAdapt
from src.trainer.maml import (
    FOMAMLConfig,
    PretrainConfig,
    pretrain_offline,
)
from src.trainer.incremental import ILConfig, run_incremental, run_qlib_backtest_with_il
from src.evaluate.metrics import cross_sectional_ic
from src.evaluate.backtest_qlib import QlibBTConfig as BTConfig
from src.inference import forecast_latest
from src import runs
from src.config import (
    DEFAULT_BT,
    HISTORY_START,
    HISTORY_END,
    PRETRAIN_START,
    PRETRAIN_END,
    VALIDATION_START,
    VALIDATION_END,
    ONLINE_TRAINING_START,
    ONLINE_TRAINING_END,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--universe",
        choices=["TW50", "TM100", "BOTH"],
        default=DEFAULT_UNIVERSE,
        help="which TEJ index to use as the time-varying active universe " "(BOTH = TW50 ∪ TM100, ~150 stocks/period)",
    )
    args = ap.parse_args()
    universe_name = args.universe

    OUT_DIR = runs.init_run()  # next free models/runN/, CURRENT -> it, tee to logs/
    t0 = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)} " f"capability={torch.cuda.get_device_capability(0)}")
    print(f"universe: {universe_name}")
    torch.manual_seed(0)
    np.random.seed(0)

    bars = load_bars(universe_name)
    if len(bars) < 5:
        print("WARNING: very few stocks loaded. Cache likely incomplete.")

    panel = build_panel(bars, pred_h=PRED_HORIZON)
    print(
        f"raw panel rows: {len(panel):,}; "
        f"dates: {pd.Timestamp(panel.dates.min()).date()} -> "
        f"{pd.Timestamp(panel.dates.max()).date()}"
    )

    # Time-varying mask from the chosen TEJ index (monthly snapshots).
    schedule = build_universe_schedule(universe_name, HISTORY_START, HISTORY_END)
    schedule.to_csv(OUT_DIR / "universe_schedule.csv", index=False)
    typical_size = int(schedule["n_active"].iloc[-1]) if not schedule.empty else 0
    print(f"universe schedule: {len(schedule)} rebalances, " f"~{typical_size} active per period ({universe_name})")
    panel = apply_time_varying_universe(panel, schedule)
    print(f"after universe mask: {len(panel):,} rows")

    meta_train_X = panel.slice_by_date(PRETRAIN_START, PRETRAIN_END).X
    norm_stats = fit_robust_zscore(meta_train_X)
    panel = PanelTensors(
        X=fillna(apply_robust_zscore(panel.X, norm_stats)),
        y=panel.y,
        dates=panel.dates,
        codes=panel.codes,
    )
    print(f"feature norm fit on {len(meta_train_X):,} meta-train rows")
    # Persist for deploy_today.py (feature scheme = which Alpha360 normalization).
    runs.save_norm_stats(OUT_DIR, norm_stats)
    runs.save_feature_scheme(OUT_DIR, FEATURE_MODE)

    # Algorithm-1 meta-sets:
    #   meta_train_set ← PRETRAIN window         (offline pretrain)
    #   meta_valid_set ← VALIDATION window       (held-out val + warmup IL)
    #   meta_test_set  ← ONLINE_TRAINING window  (live-IL backtest)
    meta_train_set = panel.slice_by_date(PRETRAIN_START, PRETRAIN_END).drop_nan_labels()
    meta_valid_set = panel.slice_by_date(VALIDATION_START, VALIDATION_END).drop_nan_labels()
    # The IL panel needs to extend back ~90 days before VALIDATION_START so the
    # first warmup task's train window (VALIDATION_START − r days) has features.
    il_panel = panel.slice_by_date(pd.Timestamp(VALIDATION_START) - pd.Timedelta(days=90), HISTORY_END)
    print(
        f"meta_train_set: {len(meta_train_set):,}  |  "
        f"meta_valid_set: {len(meta_valid_set):,}  |  IL panel: {len(il_panel):,}"
    )

    model = DoubleAdapt(seq_len=SEQ_LEN, n_fields=N_FIELDS).to(device)
    print(f"#DA params: {sum(p.numel() for p in model.da_parameters()):,}")
    print(f"#MA params: {sum(p.numel() for p in model.ma_parameters()):,}")

    # Run-3 tuning:
    #   - lr_psi 1e-3 (carried over from run 2): slower DA drift
    #   - tasks_per_epoch 200 -> 1000 (= r * universe size = 20 * 50): one task's worth
    #     of cross-sectional samples per epoch -> more diverse meta-batches
    #   - epochs 5 -> 30, patience 2 -> 8: more room for the meta-learner to
    #     plateau (paper-style patience instead of run-2's aggressive cut)
    fcfg = FOMAMLConfig(lr_psi=1e-3)
    pcfg = PretrainConfig(epochs=30, tasks_per_epoch=1000, early_stop_patience=8)
    online_end_actual = (
        str(pd.Timestamp(panel.dates.max()).date()) if ONLINE_TRAINING_END == "auto" else ONLINE_TRAINING_END
    )
    print(
        f"IL plan: warmup (meta-valid set) {VALIDATION_START} → {ONLINE_TRAINING_START}, "
        f"then online training / live-IL (meta-test set) "
        f"{ONLINE_TRAINING_START} → {online_end_actual}"
    )

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
    print(
        f"\n=== IL warmup ({VALIDATION_START} → {ONLINE_TRAINING_START}) — " f"final preparation on meta_valid_set ==="
    )
    warmup_cfg = ILConfig(start_date=VALIDATION_START, end_date=ONLINE_TRAINING_START)
    warmup_log, warmup_preds = run_incremental(model, il_panel, fcfg, warmup_cfg, device, desc="IL warmup")
    torch.save(model.state_dict(), OUT_DIR / "final_prep.pt")
    warmup_log.to_csv(OUT_DIR / "warmup_log.csv", index=False)
    print(f"saved final_prep.pt; warmup did {len(warmup_log)} IL steps")

    # Persist trading config + active universe before the live-IL backtest.
    bt_cfg = BTConfig(**DEFAULT_BT)
    runs.save_bt_config(OUT_DIR, bt_cfg)
    runs.save_active_universe(OUT_DIR, schedule, fallback_codes=list(bars.keys()))

    # ---- Stage 3: online training / live-IL backtest on meta_test_set ----
    # Algorithm 1 line 9 second half: starts from final_prep.pt, walks forward
    # in r=20-day windows. At each step does fomaml_step (adapts model in-place
    # AND simulates portfolio trades over the test window). Final model state
    # -> final.pt.
    print(
        f"\n=== online training / live-IL qlib backtest "
        f"({ONLINE_TRAINING_START} → {online_end_actual}) — meta_test_set ==="
    )
    bt, step_log, preds = run_qlib_backtest_with_il(
        model,
        il_panel,
        bars,
        fcfg,
        bt_cfg,
        start_date=ONLINE_TRAINING_START,
        end_date=online_end_actual,
        device=device,
    )
    torch.save(model.state_dict(), OUT_DIR / "final.pt")
    step_log.to_csv(OUT_DIR / "il_log.csv", index=False)
    preds.to_csv(OUT_DIR / "predictions.csv", index=False)
    runs.save_backtest(OUT_DIR, bt)
    print(
        f"saved final.pt; live-IL backtest: "
        f"{bt.summary['days']} days, {bt.summary['n_trades']} trades, "
        f"final equity={bt.summary['final_equity']:.0f}, "
        f"cum_ret={bt.summary['cum_return']:.4f}, "
        f"sharpe={bt.summary['sharpe']:.3f}, "
        f"max_dd={bt.summary['max_drawdown']:.4f}"
    )

    live = forecast_latest(model, il_panel, n_days=20, device=device)
    live.to_csv(OUT_DIR / "live_forecast.csv", index=False)
    print(f"live forecast: {len(live)} rows over " f"{live['date'].nunique() if not live.empty else 0} dates")

    if not preds.empty:
        ic_all, icir_all, ric_all, ricir_all = cross_sectional_ic(
            preds["pred"].to_numpy(),
            preds["actual"].to_numpy(),
            preds["date"].to_numpy().astype("datetime64[ns]"),
        )
    else:
        ic_all = icir_all = ric_all = ricir_all = float("nan")

    report = {
        "history_start": HISTORY_START,
        "history_end": HISTORY_END,
        # Operational-phase date windows (the date-range constants in train.py)
        "pretrain_window": [PRETRAIN_START, PRETRAIN_END],
        "validation_window": [VALIDATION_START, VALIDATION_END],
        "online_training_window": [ONLINE_TRAINING_START, online_end_actual],
        # Mapping to Algorithm 1 of the DoubleAdapt paper (T_train, T_valid, T_test).
        # Each operational phase consumes one meta-set drawn from the matching window.
        "meta_train_set_window": [PRETRAIN_START, PRETRAIN_END],
        "meta_valid_set_window": [VALIDATION_START, VALIDATION_END],
        "meta_test_set_window": [ONLINE_TRAINING_START, online_end_actual],
        "il_window": [VALIDATION_START, online_end_actual],  # full rolling-IL span
        "il_warmup_window": [VALIDATION_START, ONLINE_TRAINING_START],
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
        "backtest_cfg": {"capital": bt_cfg.capital, "top_k": bt_cfg.top_k, "cost_alpha_bps": bt_cfg.cost_alpha_bps},
        "model_config": {
            "seq_len": SEQ_LEN,
            "n_fields": N_FIELDS,
            "pred_horizon": PRED_HORIZON,
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
