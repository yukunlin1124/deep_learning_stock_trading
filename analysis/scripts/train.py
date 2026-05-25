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

Artifacts -> analysis/models/.
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

# Allow running as `python analysis/scripts/train.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.data.io import fetch_range  # noqa: E402
from analysis.data.universe import (  # noqa: E402
    get_candidate_pool, build_universe_schedule, TOP_N as UNIVERSE_TOP_N,
)
from analysis.data.handler import (  # noqa: E402
    build_panel, PanelTensors, LABEL_HORIZON, SEQ_LEN, N_FIELDS,
)
from analysis.data.processor import (  # noqa: E402
    fit_robust_zscore, apply_robust_zscore, fillna,
)
from analysis.model.double_adapt import DoubleAdapt  # noqa: E402
from analysis.trainer.maml import (  # noqa: E402
    FOMAMLConfig, PretrainConfig, pretrain_offline,
)
from analysis.trainer.incremental import ILConfig, run_incremental  # noqa: E402
from analysis.evaluate.metrics import cross_sectional_ic  # noqa: E402
from analysis.evaluate.backtest import BTConfig, run_backtest_from_predictions  # noqa: E402
from analysis.workflow.forecast import forecast_latest  # noqa: E402

OUT_BASE = PROJECT_ROOT / "analysis" / "output"
OUT_BASE.mkdir(exist_ok=True)


def _next_run_dir(base: Path) -> Path:
    """Find next available output/runN/. Auto-increments across re-runs."""
    nums = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("run") and p.name[3:].isdigit():
            nums.append(int(p.name[3:]))
    next_n = (max(nums) + 1) if nums else 1
    return base / f"run{next_n}"


OUT_DIR = _next_run_dir(OUT_BASE)
OUT_DIR.mkdir()
# Update the CURRENT pointer so deploy_today.py picks up this run by default.
(OUT_BASE / "CURRENT").write_text(OUT_DIR.name)
print(f"writing artifacts to {OUT_DIR}")

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


def load_bars() -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    for code in get_candidate_pool():
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
    t0 = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)} "
              f"capability={torch.cuda.get_device_capability(0)}")
    torch.manual_seed(0)
    np.random.seed(0)

    bars = load_bars()
    if len(bars) < 5:
        print("WARNING: very few stocks loaded. Cache likely incomplete.")

    panel = build_panel(bars, label_h=LABEL_HORIZON)
    print(f"raw panel rows: {len(panel):,}; "
          f"dates: {pd.Timestamp(panel.dates.min()).date()} -> "
          f"{pd.Timestamp(panel.dates.max()).date()}")

    schedule = build_universe_schedule(bars, HISTORY_START, HISTORY_END,
                                       top_n=UNIVERSE_TOP_N)
    schedule.to_csv(OUT_DIR / "universe_schedule.csv", index=False)
    print(f"universe schedule: {len(schedule)} rebalances, "
          f"top {UNIVERSE_TOP_N} per period")
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
    print(f"IL window: {IL_START} -> {il_end_actual}")
    icfg = ILConfig(start_date=IL_START, end_date=il_end_actual)

    print("\n=== offline FOMAML pretrain ===")
    pre_res = pretrain_offline(model, pre_train, pre_val, fcfg, pcfg, device)
    torch.save(model.state_dict(), OUT_DIR / "pretrained.pt")
    pd.DataFrame(pre_res["history"]).to_csv(OUT_DIR / "pretrain_log.csv", index=False)
    print(f"saved pretrained.pt; best val IC = {pre_res['best_val_ic']:.4f}")

    print("\n=== fake-online incremental learning ===")
    step_log, preds = run_incremental(model, il_panel, fcfg, icfg, device)
    torch.save(model.state_dict(), OUT_DIR / "final.pt")
    step_log.to_csv(OUT_DIR / "il_log.csv", index=False)
    preds.to_csv(OUT_DIR / "predictions.csv", index=False)
    print(f"saved final.pt and IL logs")

    # Run 2: n_drop=1 to match qlib's 10% daily turnover (run 1 used n_drop=3
    # and saw ~30% annualized cost drag).
    bt_cfg = BTConfig(n_drop=1)
    # Slice predictions to the backtest reporting window (2022-2023 is warmup).
    preds_for_backtest = preds[pd.to_datetime(preds["date"]) >= pd.Timestamp(BACKTEST_START)].copy()
    print(f"backtest window: {BACKTEST_START} -> {il_end_actual} "
          f"({len(preds_for_backtest):,} of {len(preds):,} preds; "
          f"warmup pre-{BACKTEST_START} excluded)")
    # Persist for deploy_today.py (it uses the SAME trading config as the backtest).
    (OUT_DIR / "bt_config.json").write_text(json.dumps(asdict(bt_cfg), indent=2))
    # Persist the final active universe so deploy_today.py knows what to score.
    final_universe = [str(c) for c in schedule.iloc[-1]["codes"]] if len(schedule) else list(bars.keys())
    (OUT_DIR / "active_universe.json").write_text(
        json.dumps({"asof": str(schedule.iloc[-1]["asof"]) if len(schedule) else None,
                    "codes": final_universe}, indent=2))
    bt = run_backtest_from_predictions(preds_for_backtest, bars, bt_cfg)
    bt.equity.to_csv(OUT_DIR / "equity.csv", index=False)
    bt.trades.to_csv(OUT_DIR / "trades.csv", index=False)
    bt.daily.to_csv(OUT_DIR / "backtest_daily.csv", index=False)
    print(f"backtest: {bt.summary['days']} days, {bt.summary['n_trades']} trades, "
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
        "n_candidate_pool": len(bars),
        "n_active_per_period": int(UNIVERSE_TOP_N),
        "rebalances_per_year": 2,
        "n_rebalances": int(len(schedule)),
        "n_panel_rows": int(len(panel)),
        "n_pretrain_rows": int(len(pre_train)),
        "n_il_rows": int(len(il_panel)),
        "pretrain_best_val_ic": float(pre_res["best_val_ic"]),
        "pretrain_epochs_run": len(pre_res["history"]),
        "il_n_steps": int(len(step_log)),
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
            "il": asdict(icfg),
        },
        "trained_at": pd.Timestamp.now().isoformat(),
        "universe": list(bars.keys()),
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
