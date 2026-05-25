"""End-to-end pipeline integration test on stock 2330 (TSMC) only.

Mirrors analysis/scripts/train.py EXACTLY -- same fixed date splits:
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
universe via `python analysis/scripts/train.py`.

Outputs are written to analysis/test/output/ so they don't clobber analysis/output/.

Run:
    .venv/Scripts/python.exe analysis/test/test_pipeline_2330.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.data.io import fetch_range, CACHE_DIR
from analysis.data.handler import (
    build_panel, PanelTensors, SEQ_LEN, N_FIELDS, LABEL_HORIZON,
)
from analysis.data.processor import (
    fit_robust_zscore, apply_robust_zscore, fillna,
)
from analysis.model.double_adapt import DoubleAdapt
from analysis.trainer.maml import (
    FOMAMLConfig, PretrainConfig, pretrain_offline,
)
from analysis.trainer.incremental import ILConfig, run_incremental
from analysis.evaluate.metrics import cross_sectional_ic
from analysis.evaluate.backtest import BTConfig, run_backtest_from_predictions
from analysis.workflow.forecast import forecast_latest

TEST_STOCK = "2330"
OUT_DIR = PROJECT_ROOT / "analysis" / "test" / "output"
OUT_DIR.mkdir(exist_ok=True)

# Fixed date splits — match analysis/scripts/train.py exactly.
HISTORY_START = "2016-01-01"
HISTORY_END = "2026-05-23"
PRETRAIN_START = "2016-01-01"
PRETRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2023-12-31"
IL_START = "2024-01-01"
IL_END = "auto"                      # extend to panel max
REQUIRED_YEARS = list(range(2016, 2027))   # 2016..2026 inclusive


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
        print("Wait for the scrape (analysis/scripts/scrape.py) to finish "
              f"all 11 years for {code}, then rerun this test.")
        raise SystemExit(0)


def assert_(cond: bool, msg: str) -> None:
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
    assert_(not df.empty, f"fetched {len(df)} bars for {TEST_STOCK}")
    required_cols = {"date", "open", "high", "low", "close", "capacity", "turnover"}
    assert_(required_cols.issubset(df.columns),
            f"all required OHLCV columns present ({required_cols})")
    assert_(len(df) >= SEQ_LEN + LABEL_HORIZON + 1,
            f"enough bars for Alpha360 ({len(df)} >= {SEQ_LEN + LABEL_HORIZON + 1})")

    bars = {TEST_STOCK: df}

    # ---- 2. Panel ----
    print("\n=== 2. Build panel ===")
    panel = build_panel(bars, label_h=LABEL_HORIZON)
    assert_(len(panel) > 0, f"panel built ({len(panel)} rows)")
    assert_(panel.X.shape[1:] == (SEQ_LEN, N_FIELDS),
            f"X shape per row is (60, 6)")
    assert_(len(panel.dates) == len(panel.X),
            "dates/X aligned")
    print(f"panel: {len(panel):,} rows, "
          f"{pd.Timestamp(panel.dates.min()).date()} -> "
          f"{pd.Timestamp(panel.dates.max()).date()}")

    # ---- 3. Fixed date splits (match analysis/scripts/train.py) ----
    pretrain_start = pd.Timestamp(PRETRAIN_START)
    pretrain_end = pd.Timestamp(PRETRAIN_END)
    val_start = pd.Timestamp(VAL_START)
    val_end = pd.Timestamp(VAL_END)
    il_start = pd.Timestamp(IL_START)
    il_end = (pd.Timestamp(panel.dates.max())
              if IL_END == "auto" else pd.Timestamp(IL_END))
    print(f"pretrain: {pretrain_start.date()} -> {pretrain_end.date()}")
    print(f"val     : {val_start.date()} -> {val_end.date()}")
    print(f"IL      : {il_start.date()} -> {il_end.date()}")

    # ---- 4. Feature norm (RobustZScoreNorm fit on pretrain) ----
    print("\n=== 3. Feature norm ===")
    pretrain_X = panel.slice_by_date(pretrain_start, pretrain_end).X
    stats = fit_robust_zscore(pretrain_X)
    panel = PanelTensors(
        X=fillna(apply_robust_zscore(panel.X, stats)),
        y=panel.y, dates=panel.dates, codes=panel.codes,
    )
    assert_(np.isfinite(panel.X).all(), "no NaNs/inf after normalize+fillna")
    assert_(np.abs(panel.X).max() <= 3.0 + 1e-5,
            f"clipped to +-3 ({np.abs(panel.X).max():.3f})")

    pre_train = panel.slice_by_date(pretrain_start, pretrain_end).drop_nan_labels()
    pre_val = panel.slice_by_date(val_start, val_end).drop_nan_labels()
    il_panel = panel.slice_by_date(
        val_start - pd.Timedelta(days=90), il_end
    )
    assert_(len(pre_train) > 0, f"pretrain panel non-empty ({len(pre_train)})")
    assert_(len(pre_val) > 0, f"val panel non-empty ({len(pre_val)})")
    assert_(len(il_panel) > 0, f"IL panel non-empty ({len(il_panel)})")

    # ---- 5. Model construction ----
    print("\n=== 4. Model ===")
    model = DoubleAdapt(seq_len=SEQ_LEN, n_fields=N_FIELDS).to(device)
    n_da = sum(p.numel() for p in model.da_parameters())
    n_ma = sum(p.numel() for p in model.ma_parameters())
    assert_(n_da > 0 and n_ma > 0, f"params: DA={n_da:,}, MA={n_ma:,}")

    # ---- 6. Pretrain (small budget for test speed; r=20 matches main pipeline) ----
    print("\n=== 5. Pretrain (3 epochs, 20 tasks each; degenerate CSRankNorm) ===")
    fcfg = FOMAMLConfig()                            # r=20, paper defaults
    pcfg = PretrainConfig(
        epochs=3, tasks_per_epoch=20, val_n_tasks=20,
        early_stop_patience=99, seed=0,
    )
    pre_res = pretrain_offline(model, pre_train, pre_val, fcfg, pcfg, device)
    assert_("history" in pre_res and len(pre_res["history"]) >= 1,
            f"pretrain produced history ({len(pre_res['history'])} epochs)")

    # ---- 7. IL ----
    print("\n=== 6. Incremental learning ===")
    icfg = ILConfig(
        start_date=str(il_start.date()),
        end_date=str(il_end.date()),
    )
    step_log, preds = run_incremental(model, il_panel, fcfg, icfg, device)
    # Note: may produce 0 steps if the IL window is too short for r=10
    print(f"IL: {len(step_log)} steps, {len(preds)} predictions")
    assert_(isinstance(step_log, pd.DataFrame), "step_log is a DataFrame")
    assert_(isinstance(preds, pd.DataFrame), "predictions is a DataFrame")
    if not preds.empty:
        for col in ["date", "stock_code_id", "pred", "actual"]:
            assert_(col in preds.columns, f"predictions has '{col}' column")

    # ---- 8. Backtest ----
    print("\n=== 7. Backtest (top-K=10 will hold just 2330 when pred passes threshold) ===")
    bt = run_backtest_from_predictions(preds, bars, BTConfig())
    assert_(bt.summary["days"] >= 0, f"backtest produced summary")
    print(f"backtest: days={bt.summary['days']}, trades={bt.summary['n_trades']}, "
          f"cum_ret={bt.summary['cum_return']:.4f}, "
          f"sharpe={bt.summary['sharpe']:.3f}")

    # ---- 9. Live forecast ----
    print("\n=== 8. Live forecast ===")
    live = forecast_latest(model, il_panel, n_days=5, device=device)
    assert_(isinstance(live, pd.DataFrame), "live forecast is a DataFrame")
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
        "val_window": [VAL_START, VAL_END],
        "il_window": [IL_START, str(il_end.date())],
        "n_panel_rows": int(len(panel)),
        "n_pretrain_rows": int(len(pre_train)),
        "n_val_rows": int(len(pre_val)),
        "n_il_rows": int(len(il_panel)),
        "pretrain_epochs": int(len(pre_res["history"])),
        "il_steps": int(len(step_log)),
        "n_predictions": int(len(preds)),
        "backtest_summary": bt.summary,
        "wall_time_sec": round(time.monotonic() - t0, 1),
        "caveat": ("Single-stock test: CSRankNorm degenerates, so model quality "
                   "is meaningless. Test verifies pipeline plumbing only."),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n=== TEST PIPELINE 2330 PASSED ===")
    print(f"wall time: {time.monotonic() - t0:.1f}s")
    print(f"artifacts: {OUT_DIR}")


if __name__ == "__main__":
    main()
