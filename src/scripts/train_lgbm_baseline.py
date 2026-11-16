"""LightGBM baseline: gradient-boosted trees on the same Alpha360 panel.

Mirrors qlib's `workflow_config_lightgbm_Alpha360.yaml` benchmark, on our
data and protocol, so it lines up against run9 (DoubleAdapt) and run18
(plain-GRU IL baseline):

  - Same Alpha360 features (per_field), same universe + time-varying mask,
    same robust z-score fit on pretrain rows
  - Each (60 lag x 6 field) window is flattened to 360 tabular columns
  - Label = per-date CSRankNorm of the 1-day c2c return (qlib's learn processor)
  - Fit on PRETRAIN, early-stop on VALIDATION (l2 on CSRankNorm'd labels)
  - Hyperparameters = qlib's tuned Alpha360 LightGBM config
  - Same backtest: TopkDropoutStrategy top_k=10 / n_drop=1, TWSE costs,
    cost_alpha_bps=70, deal_price=close

Two modes:
  static  (default) one fit, frozen through the whole test window; this is
          what qlib's benchmark table reports.
  rolling refit every r=20 trading days on all labelled history up to that
          point (expanding window, fixed #trees = best_iter of the first fit).
          This is the closest tree analogue to the GRU's IL fine-tuning.

Usage:
  python -m src.scripts.train_lgbm_baseline                  # -> next free runN (MLflow, experiment lgbm_baseline)
  python -m src.scripts.train_lgbm_baseline --mode rolling --run run20
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from src.data.handler import build_panel, PanelTensors, PRED_HORIZON, SEQ_LEN, N_FIELDS, FIELDS
from src.data.processor import (
    fit_robust_zscore,
    apply_robust_zscore,
    fillna,
    csrank_normalize_per_date,
)
from src.data.universe import build_universe_schedule
from src.evaluate.backtest_qlib import QlibBTConfig, run_qlib_backtest_from_predictions
from src.evaluate.metrics import cross_sectional_ic
from src.config import (
    DEFAULT_BT,
    HISTORY_START,
    HISTORY_END,
    PRETRAIN_START,
    PRETRAIN_END,
    VALIDATION_START,
    VALIDATION_END,
    ONLINE_TRAINING_START,
)
from src.data.io import load_bars
from src.data.universe import apply_time_varying_universe
from src.utils.progress import pbar, log
from src import runs

FEATURE_MODE = "per_field"  # match run9 / run18
REFIT_EVERY = 20  # r=20, matches the IL cadence

# qlib examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha360.yaml
LGB_PARAMS = dict(
    objective="regression",  # loss: mse
    learning_rate=0.0421,
    colsample_bytree=0.8879,
    subsample=0.8789,
    subsample_freq=1,
    lambda_l1=205.6999,
    lambda_l2=580.9768,
    max_depth=8,
    num_leaves=210,
    num_threads=8,
    seed=0,
    verbosity=-1,
)
NUM_BOOST_ROUND = 1000
EARLY_STOPPING = 50


def _flat(p: PanelTensors) -> np.ndarray:
    return p.X.reshape(len(p), -1)


def _csrank(y: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Same CSRankNorm the GRU baselines train on (reuses the torch helper)."""
    t = torch.from_numpy(y.astype(np.float32))
    return csrank_normalize_per_date(t, dates).numpy()


def _dataset(p: PanelTensors, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    return lgb.Dataset(_flat(p), label=_csrank(p.y, p.dates), reference=reference, free_raw_data=False)


class _TqdmCallback:
    """LightGBM callback: advance a tqdm bar once per boosting round.

    Shows the latest eval metrics (e.g. train_l2 / valid_l2) as the bar postfix and,
    if log_every > 0, writes a log line every log_every rounds so the session log
    keeps a record (bar redraws are kept out of log files by the Tee).
    """

    order = 15  # after lgb.log_evaluation (10), before lgb.early_stopping (30)
    before_iteration = False

    def __init__(self, bar: Any, log_every: int = 0) -> None:
        self.bar = bar
        self.log_every = log_every

    def __call__(self, env: Any) -> None:
        self.bar.update(1)
        metrics = {f"{r[0]}_{r[1]}": r[2] for r in (env.evaluation_result_list or [])}
        if metrics:
            self.bar.set_postfix({k: f"{v:.5f}" for k, v in metrics.items()}, refresh=False)
            runs.log_metrics(metrics, step=env.iteration)
            if self.log_every and (env.iteration + 1) % self.log_every == 0:
                log(f"[lgbm] round {env.iteration + 1:4d}: " + "  ".join(f"{k}={v:.5f}" for k, v in metrics.items()))


def fit_static(train: PanelTensors, valid: PanelTensors) -> tuple[lgb.Booster, list[dict]]:
    dtrain = _dataset(train)
    dvalid = _dataset(valid, reference=dtrain)
    evals: dict = {}
    # Early stopping usually ends well before NUM_BOOST_ROUND, so the bar can stop short of 100%.
    with pbar(total=NUM_BOOST_ROUND, desc="lgbm fit", unit="round") as bar:
        booster = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                _TqdmCallback(bar, log_every=50),
                lgb.record_evaluation(evals),
            ],
        )
    log(f"[lgbm] stopped after {booster.current_iteration()} rounds; best iteration = {booster.best_iteration}")
    hist = [
        {"iter": i + 1, "train_l2": tr, "valid_l2": va}
        for i, (tr, va) in enumerate(zip(evals["train"]["l2"], evals["valid"]["l2"]))
    ]
    return booster, hist


def _pred_rows(p: PanelTensors, preds: np.ndarray) -> list[dict]:
    return [
        {"date": pd.Timestamp(d).date(), "stock_code_id": str(c), "pred": float(pr), "actual": float(a)}
        for d, c, pr, a in zip(p.dates, p.codes, preds, p.y)
    ]


def predict_static(
    booster: lgb.Booster, panel: PanelTensors, start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen model; logs IC per r-day window so il_log.csv matches run9/run18."""
    test = panel.slice_by_date(start, end).drop_nan_labels()
    preds = booster.predict(_flat(test), num_iteration=booster.best_iteration)
    udates = np.unique(test.dates)
    rows = []
    for step, s in enumerate(range(0, len(udates), REFIT_EVERY)):
        win = udates[s : s + REFIT_EVERY]
        m = np.isin(test.dates, win)
        ic, icir, ric, ricir = cross_sectional_ic(preds[m], test.y[m], test.dates[m])
        rows.append(
            {
                "step": step,
                "test_start": pd.Timestamp(win[0]).date(),
                "test_end": pd.Timestamp(win[-1]).date(),
                "n_test": int(m.sum()),
                "ic": ic,
                "icir": icir,
                "ric": ric,
                "ricir": ricir,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(_pred_rows(test, preds))


def predict_rolling(panel: PanelTensors, n_rounds: int, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every r days: refit on all labelled rows dated < window start, predict window.

    The newest training label (date i-1) uses close[i], which is known at the
    window's first deal (close of day i) -- same timing as the GRU IL loop.
    """
    labelled = panel.drop_nan_labels()
    udates = np.unique(labelled.dates)
    s = np.datetime64(pd.Timestamp(start))
    e = np.datetime64(pd.Timestamp(end))
    first = int(np.searchsorted(udates, s, side="left"))
    starts = list(range(first, len(udates), REFIT_EVERY))
    starts = [i for i in starts if udates[i] <= e]

    rows, pred_rows = [], []
    bar = pbar(starts, desc="rolling refit", unit="step")
    for step, i in enumerate(bar):
        win = udates[i : i + REFIT_EVERY]
        win = win[win <= e]
        tr_m = (labelled.dates >= np.datetime64(pd.Timestamp(PRETRAIN_START))) & (labelled.dates < udates[i])
        te_m = np.isin(labelled.dates, win)
        tr = PanelTensors(
            X=labelled.X[tr_m], y=labelled.y[tr_m], dates=labelled.dates[tr_m], codes=labelled.codes[tr_m]
        )
        te = PanelTensors(
            X=labelled.X[te_m], y=labelled.y[te_m], dates=labelled.dates[te_m], codes=labelled.codes[te_m]
        )
        with pbar(total=n_rounds, desc=f"refit {step} rounds", unit="round", leave=False) as round_bar:
            booster = lgb.train(
                LGB_PARAMS, _dataset(tr), num_boost_round=n_rounds, callbacks=[_TqdmCallback(round_bar)]
            )
        preds = booster.predict(_flat(te))
        ic, icir, ric, ricir = cross_sectional_ic(preds, te.y, te.dates)
        rows.append(
            {
                "step": step,
                "train_end": pd.Timestamp(udates[i - 1]).date(),
                "test_start": pd.Timestamp(win[0]).date(),
                "test_end": pd.Timestamp(win[-1]).date(),
                "n_train": len(tr),
                "n_test": len(te),
                "ic": ic,
                "icir": icir,
                "ric": ric,
                "ricir": ricir,
            }
        )
        pred_rows += _pred_rows(te, preds)
        log(
            f"[refit] step {step:3d}: t={pd.Timestamp(win[0]).date()} "
            f"n_train={len(tr):,} ic={ic:+.4f} ric={ric:+.4f}"
        )
        runs.log_metrics({"refit_ic": ic, "refit_ric": ric}, step=step)
        bar.set_postfix(ic=f"{ic:+.4f}")
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="TW50")
    ap.add_argument("--mode", choices=["static", "rolling"], default="static")
    ap.add_argument("--run", default=None, help="MLflow run name (default: next runN)")
    args = ap.parse_args()

    # deploy_today.py can't load a LightGBM run, so leave the current tag alone.
    out_dir = runs.init_run(args.run, experiment="lgbm_baseline", set_as_current=False)
    t0 = time.monotonic()
    np.random.seed(0)
    print(f"universe: {args.universe}  feature_mode: {FEATURE_MODE}  mode: {args.mode}")

    bars = load_bars(args.universe)
    panel = build_panel(bars, pred_h=PRED_HORIZON, mode=FEATURE_MODE)
    schedule = build_universe_schedule(args.universe, HISTORY_START, HISTORY_END)
    schedule.to_csv(out_dir / "universe_schedule.csv", index=False)
    panel = apply_time_varying_universe(panel, schedule)
    print(f"after universe mask: {len(panel):,} rows")

    norm_stats = fit_robust_zscore(panel.slice_by_date(PRETRAIN_START, PRETRAIN_END).X)
    panel = PanelTensors(
        X=fillna(apply_robust_zscore(panel.X, norm_stats)), y=panel.y, dates=panel.dates, codes=panel.codes
    )
    runs.save_norm_stats(out_dir, norm_stats)
    runs.save_feature_scheme(out_dir, FEATURE_MODE, flattened=True)

    train_set = panel.slice_by_date(PRETRAIN_START, PRETRAIN_END).drop_nan_labels()
    valid_set = panel.slice_by_date(VALIDATION_START, VALIDATION_END).drop_nan_labels()
    print(f"train: {len(train_set):,}  valid: {len(valid_set):,}")

    print("\n=== Stage 1: fit on pretrain, early-stop on validation ===")
    booster, hist = fit_static(train_set, valid_set)
    pd.DataFrame(hist).to_csv(out_dir / "pretrain_log.csv", index=False)
    booster.save_model(str(out_dir / "pretrained.txt"))
    vpred = booster.predict(_flat(valid_set), num_iteration=booster.best_iteration)
    val_ic, _, val_ric, _ = cross_sectional_ic(vpred, valid_set.y, valid_set.dates)
    print(f"best_iter={booster.best_iteration}  val_ic={val_ic:+.4f}  val_ric={val_ric:+.4f}")

    imp = pd.DataFrame(
        {
            "feature": [f"{FIELDS[j % N_FIELDS]}_lag{SEQ_LEN - 1 - j // N_FIELDS}" for j in range(SEQ_LEN * N_FIELDS)],
            "gain": booster.feature_importance("gain"),
        }
    ).sort_values("gain", ascending=False)
    imp.to_csv(out_dir / "feature_importance.csv", index=False)

    online_end = str(pd.Timestamp(panel.dates.max()).date())
    print(f"\n=== Stage 2: {args.mode} predictions ({ONLINE_TRAINING_START} -> {online_end}) ===")
    if args.mode == "static":
        step_log, preds = predict_static(booster, panel, ONLINE_TRAINING_START, online_end)
    else:
        step_log, preds = predict_rolling(panel, booster.best_iteration, ONLINE_TRAINING_START, online_end)
    step_log.to_csv(out_dir / "il_log.csv", index=False)
    preds.to_csv(out_dir / "predictions.csv", index=False)

    print("\n=== Stage 3: qlib backtest from predictions ===")
    bt_cfg = QlibBTConfig(**DEFAULT_BT)
    runs.save_bt_config(out_dir, bt_cfg)
    bt = run_qlib_backtest_from_predictions(preds, bars, bt_cfg)
    runs.save_backtest(out_dir, bt)

    ic_all, icir_all, ric_all, ricir_all = cross_sectional_ic(
        preds["pred"].to_numpy(),
        preds["actual"].to_numpy(),
        preds["date"].to_numpy().astype("datetime64[ns]"),
    )
    report = {
        "model_kind": f"lgbm_{args.mode}",
        "history_start": HISTORY_START,
        "history_end": HISTORY_END,
        "pretrain_window": [PRETRAIN_START, PRETRAIN_END],
        "validation_window": [VALIDATION_START, VALIDATION_END],
        "online_training_window": [ONLINE_TRAINING_START, online_end],
        "universe": args.universe,
        "feature_mode": FEATURE_MODE,
        "n_candidate_pool": len(bars),
        "n_panel_rows": int(len(panel)),
        "n_train_rows": int(len(train_set)),
        "n_valid_rows": int(len(valid_set)),
        "best_iteration": int(booster.best_iteration),
        "pretrain_best_val_ic": float(val_ic),
        "pretrain_best_val_ric": float(val_ric),
        "refit_every": REFIT_EVERY if args.mode == "rolling" else None,
        "il_overall_ic": ic_all,
        "il_overall_icir": icir_all,
        "il_overall_ric": ric_all,
        "il_overall_ricir": ricir_all,
        "backtest": bt.summary,
        "backtest_cfg": asdict(bt_cfg),
        "lgb_params": LGB_PARAMS,
        "trained_at": pd.Timestamp.now().isoformat(),
        "wall_time_min": (time.monotonic() - t0) / 60.0,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    runs.log_report(report)

    s = bt.summary
    print("\n=== summary ===")
    print(f"  val IC / RankIC      : {val_ic:.4f} / {val_ric:.4f}  (best_iter {booster.best_iteration})")
    print(f"  test IC / RankIC     : {ic_all:.4f} / {ric_all:.4f}")
    print(f"  Backtest ann return  : {s['ann_return']:+.4f}")
    print(f"  Backtest Sharpe      : {s['sharpe']:+.3f}")
    print(f"  Backtest max DD      : {s['max_drawdown']:+.4f}")
    print(f"  Total wall-time      : {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
