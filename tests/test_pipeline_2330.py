"""End-to-end pipeline integration test on stock 2330 (TSMC) only.

Mirrors src/scripts/train.py EXACTLY -- same fixed date splits:
  - Pretrain: 2016-01-01 -> 2021-12-31
  - Validate: 2022-01-01 -> 2023-12-31
  - Online IL + backtest: 2024-01-01 -> panel max

Requires all 11 yearly cache files for 2330 (2016..2026) to exist. Exits
with a clear message if any are missing.

IMPORTANT CAVEAT
----------------
Cross-sectional metrics (IC / RankIC) require multiple stocks per date.
With only 2330, each date has n=1 row, so:
  - CSRankNorm degenerates: y_norm = 0 for every sample
  - L_train collapses to MSE(pred, 0) -> model learns to predict ~0
  - per-date IC/RankIC are 0 / NaN
  - top-K backtest with K=10 holds just 2330 whenever pred > threshold

So THIS TEST VERIFIES PLUMBING, NOT MODEL QUALITY. It catches:
  * import / shape / type bugs across all modules
  * dataflow regressions in panel build, IL loop, backtest engine
  * artifact-writing bugs

For a real signal-quality test, run the full pipeline on the 50-stock
universe via `python src/scripts/train.py`.

Outputs go to a temp folder (test_2330_*), not to a tracked MLflow run.

Run (either works; pytest skips instead of passing when the cache is incomplete):
    pytest -m slow
    python tests/test_pipeline_2330.py
"""

from __future__ import annotations

import json
import time

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import pytest
except ImportError:  # plain-script run without pytest installed
    pytest = None
else:
    pytestmark = pytest.mark.slow  # whole module: trains a model end to end


from src.data.io import fetch_range, CACHE_DIR
from src.config import (
    HISTORY_START,
    HISTORY_END,
    PRETRAIN_START,
    PRETRAIN_END,
    VALIDATION_START,
    VALIDATION_END,
    ONLINE_TRAINING_START,
    ONLINE_TRAINING_END,
)
from src.data.handler import (
    build_panel,
    PanelTensors,
    SEQ_LEN,
    N_FIELDS,
    PRED_HORIZON,
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
from src.trainer.incremental import ILConfig, run_incremental
from src.evaluate.backtest_qlib import (
    QlibBTConfig as BTConfig,
    run_qlib_backtest_from_predictions as run_backtest_from_predictions,
)
from src.inference import forecast_latest

TEST_STOCK = "2330"
# Smoke-test outputs go to a temp folder; they are not a tracked MLflow run.
OUT_DIR = Path(tempfile.mkdtemp(prefix="test_2330_"))

# Date splits come from src/config.py, the same values train.py uses.
REQUIRED_YEARS = list(range(2016, 2027))  # 2016..2026 inclusive


def cached_years(code: str) -> set[int]:
    out = set()
    for p in CACHE_DIR.glob(f"{code}_*.csv"):
        try:
            out.add(int(p.stem.split("_")[1]))
        except (ValueError, IndexError):
            continue
    return out


def require_full_cache(code: str) -> None:
    """Hard-stop if any of the REQUIRED_YEARS are not yet on disk."""
    have = cached_years(code)
    missing = [y for y in REQUIRED_YEARS if y not in have]
    if missing:
        print(f"\n*** SKIP: {code} cache is incomplete ***")
        print(f"have: {sorted(have)}")
        print(f"missing: {missing}")
        print(
            "Drop fresh TEJ CSVs into data/raw/TEJ/test/ and run "
            f"src/scripts/import_tej.py to populate {code}'s cache, "
            "then rerun this test."
        )
        raise SystemExit(0)


def check(cond: bool, msg: str) -> None:
    """Lightweight assertion that prints a clean failure message."""
    if not cond:
        raise AssertionError(f"ASSERTION FAILED: {msg}")
    print(f"  OK: {msg}")


def main() -> None:
    t0 = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- 1. Data ----
    print(f"\n=== 1. Data layer (stock {TEST_STOCK}) ===")
    require_full_cache(TEST_STOCK)
    print(f"all {len(REQUIRED_YEARS)} years cached for {TEST_STOCK}")

    df = fetch_range(TEST_STOCK, HISTORY_START, HISTORY_END)
    check(not df.empty, f"fetched {len(df)} bars for {TEST_STOCK}")
    required_cols = {"date", "open", "high", "low", "close", "capacity", "turnover"}
    check(required_cols.issubset(df.columns), f"all required OHLCV columns present ({required_cols})")
    check(
        len(df) >= SEQ_LEN + PRED_HORIZON + 1, f"enough bars for Alpha360 ({len(df)} >= {SEQ_LEN + PRED_HORIZON + 1})"
    )

    bars = {TEST_STOCK: df}

    # ---- 2. Panel ----
    print("\n=== 2. Build panel ===")
    panel = build_panel(bars, pred_h=PRED_HORIZON)
    check(len(panel) > 0, f"panel built ({len(panel)} rows)")
    check(panel.X.shape[1:] == (SEQ_LEN, N_FIELDS), f"X shape per row is (60, 6)")
    check(len(panel.dates) == len(panel.X), "dates/X aligned")
    print(
        f"panel: {len(panel):,} rows, "
        f"{pd.Timestamp(panel.dates.min()).date()} -> "
        f"{pd.Timestamp(panel.dates.max()).date()}"
    )

    # ---- 3. Fixed date splits (match src/scripts/train.py) ----
    pretrain_start = pd.Timestamp(PRETRAIN_START)
    pretrain_end = pd.Timestamp(PRETRAIN_END)
    val_start = pd.Timestamp(VALIDATION_START)
    val_end = pd.Timestamp(VALIDATION_END)
    online_start = pd.Timestamp(ONLINE_TRAINING_START)
    online_end = pd.Timestamp(panel.dates.max()) if ONLINE_TRAINING_END == "auto" else pd.Timestamp(ONLINE_TRAINING_END)
    print(f"pretrain        : {pretrain_start.date()} -> {pretrain_end.date()}")
    print(f"validation      : {val_start.date()} -> {val_end.date()}")
    print(f"online training : {online_start.date()} -> {online_end.date()}")

    # ---- 4. Feature norm (RobustZScoreNorm fit on pretrain) ----
    print("\n=== 3. Feature norm ===")
    pretrain_X = panel.slice_by_date(pretrain_start, pretrain_end).X
    stats = fit_robust_zscore(pretrain_X)
    panel = PanelTensors(
        X=fillna(apply_robust_zscore(panel.X, stats)),
        y=panel.y,
        dates=panel.dates,
        codes=panel.codes,
    )
    check(np.isfinite(panel.X).all(), "no NaNs/inf after normalize+fillna")
    check(np.abs(panel.X).max() <= 3.0 + 1e-5, f"clipped to +-3 ({np.abs(panel.X).max():.3f})")

    # Algorithm-1 meta-sets carved out of each operational date window.
    meta_train_set = panel.slice_by_date(pretrain_start, pretrain_end).drop_nan_labels()
    meta_valid_set = panel.slice_by_date(val_start, val_end).drop_nan_labels()
    il_panel = panel.slice_by_date(val_start - pd.Timedelta(days=90), online_end)
    check(len(meta_train_set) > 0, f"meta_train_set non-empty ({len(meta_train_set)})")
    check(len(meta_valid_set) > 0, f"meta_valid_set non-empty ({len(meta_valid_set)})")
    check(len(il_panel) > 0, f"IL panel non-empty ({len(il_panel)})")

    # ---- 5. Model construction ----
    print("\n=== 4. Model ===")
    model = DoubleAdapt(seq_len=SEQ_LEN, n_fields=N_FIELDS).to(device)
    n_da = sum(p.numel() for p in model.da_parameters())
    n_ma = sum(p.numel() for p in model.ma_parameters())
    check(n_da > 0 and n_ma > 0, f"params: DA={n_da:,}, MA={n_ma:,}")

    # ---- 6. Pretrain (small budget for test speed; r=20 matches main pipeline) ----
    print("\n=== 5. Pretrain (3 epochs, 20 tasks each; degenerate CSRankNorm) ===")
    fcfg = FOMAMLConfig()  # r=20, paper defaults
    pcfg = PretrainConfig(
        epochs=3,
        tasks_per_epoch=20,
        val_n_tasks=20,
        early_stop_patience=99,
        seed=0,
    )
    pre_res = pretrain_offline(model, meta_train_set, meta_valid_set, fcfg, pcfg, device)
    check(
        "history" in pre_res and len(pre_res["history"]) >= 1,
        f"pretrain produced history ({len(pre_res['history'])} epochs)",
    )

    # ---- 7. IL ----
    print("\n=== 6. Incremental learning ===")
    icfg = ILConfig(
        start_date=str(online_start.date()),
        end_date=str(online_end.date()),
    )
    step_log, preds = run_incremental(model, il_panel, fcfg, icfg, device)
    # Note: may produce 0 steps if the IL window is too short for r=10
    print(f"IL: {len(step_log)} steps, {len(preds)} predictions")
    check(isinstance(step_log, pd.DataFrame), "step_log is a DataFrame")
    check(isinstance(preds, pd.DataFrame), "predictions is a DataFrame")
    if not preds.empty:
        for col in ["date", "stock_code_id", "pred", "actual"]:
            check(col in preds.columns, f"predictions has '{col}' column")

    # ---- 8. Backtest ----
    print("\n=== 7. Backtest (top-K=10 will hold just 2330 when pred passes threshold) ===")
    bt = run_backtest_from_predictions(preds, bars, BTConfig())
    check(bt.summary["days"] >= 0, f"backtest produced summary")
    print(
        f"backtest: days={bt.summary['days']}, trades={bt.summary['n_trades']}, "
        f"cum_ret={bt.summary['cum_return']:.4f}, "
        f"sharpe={bt.summary['sharpe']:.3f}"
    )

    # ---- 9. Live forecast ----
    print("\n=== 8. Live forecast ===")
    live = forecast_latest(model, il_panel, n_days=5, device=device)
    check(isinstance(live, pd.DataFrame), "live forecast is a DataFrame")
    print(f"live forecast: {len(live)} rows")

    # ---- 10. Write outputs ----
    print(f"\n=== 9. Write outputs to {OUT_DIR} ===")
    pd.DataFrame(pre_res["history"]).to_csv(OUT_DIR / "pretrain_log.csv", index=False)
    step_log.to_csv(OUT_DIR / "il_log.csv", index=False)
    preds.to_csv(OUT_DIR / "predictions.csv", index=False)
    bt.equity.to_csv(OUT_DIR / "equity.csv", index=False)
    bt.trades.to_csv(OUT_DIR / "trades.csv", index=False)
    live.to_csv(OUT_DIR / "live_forecast.csv", index=False)
    torch.save(model.state_dict(), OUT_DIR / "model.pt")

    report = {
        "test": "pipeline_2330",
        "stock": TEST_STOCK,
        "pretrain_window": [PRETRAIN_START, PRETRAIN_END],
        "validation_window": [VALIDATION_START, VALIDATION_END],
        "online_training_window": [ONLINE_TRAINING_START, str(online_end.date())],
        "n_panel_rows": int(len(panel)),
        "n_meta_train_set_rows": int(len(meta_train_set)),
        "n_meta_valid_set_rows": int(len(meta_valid_set)),
        "n_il_rows": int(len(il_panel)),
        "pretrain_epochs": int(len(pre_res["history"])),
        "il_steps": int(len(step_log)),
        "n_predictions": int(len(preds)),
        "backtest_summary": bt.summary,
        "wall_time_sec": round(time.monotonic() - t0, 1),
        "caveat": (
            "Single-stock test: CSRankNorm degenerates, so model quality "
            "is meaningless. Test verifies pipeline plumbing only."
        ),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n=== TEST PIPELINE 2330 PASSED ===")
    print(f"wall time: {time.monotonic() - t0:.1f}s")
    print(f"artifacts: {OUT_DIR}")


def test_pipeline_2330() -> None:
    """pytest entry point: the full main() run. A missing cache is a skip, not a pass."""
    try:
        main()
    except SystemExit as e:
        if e.code == 0:
            pytest.skip("2330 cache incomplete; run src/scripts/import_tej.py first")
        raise


if __name__ == "__main__":
    main()
