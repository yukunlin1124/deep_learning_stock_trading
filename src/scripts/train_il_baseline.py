"""IL-baseline training: plain GRU + vanilla pretrain + per-step fine-tune.

Strips DoubleAdapt's distinguishing machinery to isolate its contribution:
  - No FeatureAdapter (G) — features pass through untouched
  - No LabelAdapter (H) — predict CSRankNorm'd y directly, no inverse transform
  - No FOMAML bi-level — pretrain = vanilla MSE; IL = N=10 vanilla SGD steps

Everything else matches run9 exactly: Alpha360 features (per_field), TW50
universe, robust z-score on pretrain rows, CSRankNorm wrap on labels for
stability, r=20 IL cadence, TopkDropoutStrategy with top_k=10/n_drop=1,
TWSE costs.

Output: models/run18/ with same artifact shape as run9.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.data.io import fetch_range
from src.data.universe import get_index_universe_codes, build_universe_schedule
from src.data.handler import build_panel, PanelTensors, PRED_HORIZON, SEQ_LEN, N_FIELDS, FIELDS
from src.data.processor import (
    fit_robust_zscore, apply_robust_zscore, fillna, csrank_normalize_per_date,
)
from src.model.gru import GRUBase
from src.evaluate.metrics import cross_sectional_ic
from src.evaluate.backtest_qlib import QlibBTConfig, run_qlib_backtest_from_predictions
from src.scripts.train import (
    apply_time_varying_universe, HISTORY_START, HISTORY_END,
    PRETRAIN_START, PRETRAIN_END, VALIDATION_START, VALIDATION_END,
    ONLINE_TRAINING_START,
)
from src.utils.progress import Tee, pbar, log, n_walk_steps

from src.config import LOGS_DIR, MODELS_DIR

OUT_BASE = MODELS_DIR

# Match run9's feature scheme for fair comparison.
FEATURE_MODE = "per_field"

# Baseline-specific hyperparameters.
PRETRAIN_EPOCHS = 30
PRETRAIN_BATCH_SIZE = 1024
PRETRAIN_LR = 1e-3
PRETRAIN_PATIENCE = 8
IL_STEPS_PER_WINDOW = 10            # the "10 SGD steps" called out in the report
IL_LR = 1e-3
IL_WINDOW = 20                      # r=20, matches run9


def _init_run_output(base: Path, run_name: str = "run18") -> Path:
    out_dir = base / run_name
    out_dir.mkdir(exist_ok=True)
    (base / "CURRENT").write_text(out_dir.name)
    log_path = LOGS_DIR / f"train_{out_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print(f"writing artifacts to {out_dir}")
    print(f"logging full session to {log_path}")
    return out_dir


def load_bars(universe: str) -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    codes = get_index_universe_codes(universe)
    print(f"loading bars for {universe} universe ({len(codes)} codes)...")
    for code in pbar(codes, desc="loading bars", unit="stock"):
        try:
            df = fetch_range(code, HISTORY_START, HISTORY_END)
        except Exception as e:
            log(f"[skip] {code}: {e!r}")
            continue
        if df.empty or len(df) < SEQ_LEN + PRED_HORIZON + 1:
            continue
        bars[code] = df
    print(f"loaded {len(bars)} candidate stocks")
    return bars


def _predict_batched(model: GRUBase, X: np.ndarray, device: torch.device,
                     batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,), dtype=np.float32)


def vanilla_pretrain(model: GRUBase, train_panel: PanelTensors,
                     val_panel: PanelTensors, device: torch.device) -> dict:
    """MSE pretrain on CSRankNorm'd labels with early-stop on val IC (vs raw y)."""
    optim = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    y_train_raw_t = torch.from_numpy(train_panel.y.astype(np.float32)).to(device)
    y_train_norm_t = csrank_normalize_per_date(y_train_raw_t, train_panel.dates)
    X_train_t = torch.from_numpy(train_panel.X).to(device)

    n = len(train_panel)
    steps_per_epoch = max(1, n // PRETRAIN_BATCH_SIZE)
    rng = np.random.default_rng(0)
    best_ic = -np.inf
    bad = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = []

    epoch_bar = pbar(range(PRETRAIN_EPOCHS), desc="pretrain", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        idx = rng.permutation(n)
        loss_sum = 0.0
        batch_bar = pbar(range(steps_per_epoch), desc=f"epoch {epoch} train",
                         unit="batch", leave=False)
        for s in batch_bar:
            b = idx[s * PRETRAIN_BATCH_SIZE:(s + 1) * PRETRAIN_BATCH_SIZE]
            if len(b) == 0:
                continue
            optim.zero_grad(set_to_none=True)
            pred = model(X_train_t[b])
            loss = F.mse_loss(pred, y_train_norm_t[b])
            loss.backward()
            optim.step()
            loss_sum += float(loss.detach().item())
            if (s + 1) % 50 == 0:
                batch_bar.set_postfix(loss=f"{loss_sum / (s + 1):.4f}")
        train_loss = loss_sum / max(1, steps_per_epoch)

        preds = _predict_batched(model, val_panel.X, device)
        ic, icir, ric, ricir = cross_sectional_ic(preds, val_panel.y, val_panel.dates)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_ic": ic, "val_icir": icir,
                        "val_ric": ric, "val_ricir": ricir})
        log(f"[pretrain] ep {epoch:3d}: train_loss={train_loss:.4f}  "
            f"val_ic={ic:+.4f}  val_ric={ric:+.4f}")

        if ic > best_ic + 1e-6:
            best_ic = ic
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        epoch_bar.set_postfix(loss=f"{train_loss:.4f}", val_ic=f"{ic:+.4f}",
                              best=f"{best_ic:.4f}", bad=bad)
        if bad and bad >= PRETRAIN_PATIENCE:
            log(f"[pretrain] early stop @ epoch {epoch} (best val_ic={best_ic:.4f})")
            break
    epoch_bar.close()

    model.load_state_dict(best_state)
    return {"best_val_ic": best_ic, "history": history}


def il_step(model: GRUBase, X_train: torch.Tensor, y_train_norm: torch.Tensor,
            optim: torch.optim.Optimizer, n_steps: int) -> float:
    """N vanilla SGD steps on the past r days. Returns final train loss."""
    model.train()
    last_loss = 0.0
    for _ in range(n_steps):
        optim.zero_grad(set_to_none=True)
        pred = model(X_train)
        loss = F.mse_loss(pred, y_train_norm)
        loss.backward()
        optim.step()
        last_loss = float(loss.detach().item())
    return last_loss


def run_il_baseline(model: GRUBase, panel: PanelTensors, bars: dict,
                    start_date: str, end_date: str,
                    device: torch.device, desc: str = "IL") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Walk the panel in r=20-day windows; per window: fine-tune N steps, then
    predict the next r days. Returns (step_log, predictions, backtest_summary)."""
    udates = np.unique(panel.dates)
    start = np.datetime64(pd.Timestamp(start_date))
    end = np.datetime64(pd.Timestamp(end_date))
    test_first_idx = int(np.searchsorted(udates, start, side="left"))
    test_first_idx = max(test_first_idx, IL_WINDOW)

    optim = torch.optim.Adam(model.parameters(), lr=IL_LR)

    rows = []
    preds_rows = []
    step = 0
    i = test_first_idx
    bar = pbar(total=n_walk_steps(udates, test_first_idx, IL_WINDOW, end),
               desc=desc, unit="step")
    while i + IL_WINDOW <= len(udates):
        train_dates = udates[i - IL_WINDOW: i]
        test_dates = udates[i: i + IL_WINDOW]
        if udates[i] > end:
            break
        bar.update(1)

        train_mask = np.isin(panel.dates, train_dates) & np.isfinite(panel.y)
        test_mask = np.isin(panel.dates, test_dates) & np.isfinite(panel.y)
        if train_mask.sum() < 8 or test_mask.sum() < 8:
            i += IL_WINDOW; step += 1
            continue

        X_train = torch.from_numpy(panel.X[train_mask]).to(device)
        d_train = panel.dates[train_mask]
        y_train_raw_t = torch.from_numpy(panel.y[train_mask].astype(np.float32)).to(device)
        y_train_norm_t = csrank_normalize_per_date(y_train_raw_t, d_train)

        train_loss = il_step(model, X_train, y_train_norm_t, optim, IL_STEPS_PER_WINDOW)

        # Frozen prediction over the next 20 days.
        X_test = panel.X[test_mask]
        y_test_raw = panel.y[test_mask]
        d_test = panel.dates[test_mask]
        c_test = panel.codes[test_mask]
        preds = _predict_batched(model, X_test, device)
        ic, icir, ric, ricir = cross_sectional_ic(preds, y_test_raw, d_test)

        rows.append({
            "step": step,
            "train_start": pd.Timestamp(train_dates[0]).date(),
            "train_end": pd.Timestamp(train_dates[-1]).date(),
            "test_start": pd.Timestamp(test_dates[0]).date(),
            "test_end": pd.Timestamp(test_dates[-1]).date(),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "train_loss": train_loss,
            "ic": float(ic), "icir": float(icir),
            "ric": float(ric), "ricir": float(ricir),
        })
        for d, c, p, a in zip(d_test, c_test, preds, y_test_raw):
            preds_rows.append({
                "date": pd.Timestamp(d).date(),
                "stock_code_id": str(c),
                "pred": float(p), "actual": float(a),
            })

        log(f"[IL] step {step:3d}: t={pd.Timestamp(test_dates[0]).date()}.."
            f"{pd.Timestamp(test_dates[-1]).date()} "
            f"n_train={int(train_mask.sum()):5d} n_test={int(test_mask.sum()):5d} "
            f"loss={train_loss:.5f} ic={ic:+.4f} ric={ric:+.4f}")
        bar.set_postfix(t=str(pd.Timestamp(test_dates[0]).date()), ic=f"{ic:+.4f}")

        i += IL_WINDOW
        step += 1
    bar.close()

    step_log = pd.DataFrame(rows)
    preds_df = pd.DataFrame(preds_rows)
    return step_log, preds_df, {}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="TW50")
    ap.add_argument("--run", default="run18", help="output dir name")
    args = ap.parse_args()

    out_dir = _init_run_output(OUT_BASE, args.run)
    t0 = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(0)
    np.random.seed(0)
    print(f"universe: {args.universe}  feature_mode: {FEATURE_MODE}  "
          f"IL_steps_per_window: {IL_STEPS_PER_WINDOW}  IL_window(r): {IL_WINDOW}")

    bars = load_bars(args.universe)
    panel = build_panel(bars, pred_h=PRED_HORIZON, mode=FEATURE_MODE)
    print(f"raw panel: {len(panel):,}  dates: "
          f"{pd.Timestamp(panel.dates.min()).date()} -> {pd.Timestamp(panel.dates.max()).date()}")

    schedule = build_universe_schedule(args.universe, HISTORY_START, HISTORY_END)
    schedule.to_csv(out_dir / "universe_schedule.csv", index=False)
    panel = apply_time_varying_universe(panel, schedule)
    print(f"after universe mask: {len(panel):,} rows")

    meta_train_X = panel.slice_by_date(PRETRAIN_START, PRETRAIN_END).X
    norm_stats = fit_robust_zscore(meta_train_X)
    panel = PanelTensors(
        X=fillna(apply_robust_zscore(panel.X, norm_stats)),
        y=panel.y, dates=panel.dates, codes=panel.codes,
    )
    with open(out_dir / "norm_stats.pkl", "wb") as f:
        pickle.dump(norm_stats, f)
    (out_dir / "feature_scheme.json").write_text(json.dumps({
        "mode": FEATURE_MODE, "seq_len": SEQ_LEN, "n_fields": N_FIELDS,
        "fields": list(FIELDS),
    }, indent=2))

    meta_train_set = panel.slice_by_date(PRETRAIN_START, PRETRAIN_END).drop_nan_labels()
    meta_valid_set = panel.slice_by_date(VALIDATION_START, VALIDATION_END).drop_nan_labels()
    il_panel = panel.slice_by_date(
        pd.Timestamp(VALIDATION_START) - pd.Timedelta(days=90), HISTORY_END
    )
    print(f"meta_train_set: {len(meta_train_set):,}  meta_valid_set: {len(meta_valid_set):,}  "
          f"il_panel: {len(il_panel):,}")

    model = GRUBase(input_dim=N_FIELDS, hidden_dim=64, num_layers=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"#GRU params: {n_params:,} (vs run9's #MA=38,849 + #DA=12,176)")

    print("\n=== Stage 1: vanilla pretrain (no FOMAML, no DA) ===")
    pre_res = vanilla_pretrain(model, meta_train_set, meta_valid_set, device)
    torch.save(model.state_dict(), out_dir / "pretrained.pt")
    pd.DataFrame(pre_res["history"]).to_csv(out_dir / "pretrain_log.csv", index=False)
    print(f"saved pretrained.pt; best val_ic={pre_res['best_val_ic']:.4f}")

    bt_cfg = QlibBTConfig(top_k=10, n_drop=1, cost_alpha_bps=70.0,
                          costs="twse", deal_price="close")
    (out_dir / "bt_config.json").write_text(json.dumps(asdict(bt_cfg), indent=2))
    final_universe = [str(c) for c in schedule.iloc[-1]["codes"]] if len(schedule) else []
    (out_dir / "active_universe.json").write_text(
        json.dumps({"asof": str(schedule.iloc[-1]["asof"]) if len(schedule) else None,
                    "codes": final_universe}, indent=2))

    print(f"\n=== Stage 2: warmup IL ({VALIDATION_START} -> {ONLINE_TRAINING_START}) ===")
    warmup_log, _, _ = run_il_baseline(model, il_panel, bars,
                                       start_date=VALIDATION_START,
                                       end_date=ONLINE_TRAINING_START, device=device)
    torch.save(model.state_dict(), out_dir / "final_prep.pt")
    warmup_log.to_csv(out_dir / "warmup_log.csv", index=False)
    print(f"saved final_prep.pt; warmup did {len(warmup_log)} IL steps")

    online_end = str(pd.Timestamp(panel.dates.max()).date())
    print(f"\n=== Stage 3: live-IL backtest ({ONLINE_TRAINING_START} -> {online_end}) ===")
    step_log, preds, _ = run_il_baseline(model, il_panel, bars,
                                         start_date=ONLINE_TRAINING_START,
                                         end_date=online_end, device=device)
    torch.save(model.state_dict(), out_dir / "final.pt")
    step_log.to_csv(out_dir / "il_log.csv", index=False)
    preds.to_csv(out_dir / "predictions.csv", index=False)

    print(f"\n=== running qlib backtest from predictions ===")
    bt = run_qlib_backtest_from_predictions(preds, bars, bt_cfg)
    bt.equity.to_csv(out_dir / "equity.csv", index=False)
    bt.trades.to_csv(out_dir / "trades.csv", index=False)
    bt.daily.to_csv(out_dir / "backtest_daily.csv", index=False)
    print(f"backtest summary: cum_return={bt.summary['cum_return']:+.4f}  "
          f"sharpe={bt.summary['sharpe']:+.3f}  max_dd={bt.summary['max_drawdown']:+.4f}  "
          f"n_trades={bt.summary['n_trades']}")

    if not preds.empty:
        ic_all, icir_all, ric_all, ricir_all = cross_sectional_ic(
            preds["pred"].to_numpy(), preds["actual"].to_numpy(),
            preds["date"].to_numpy().astype("datetime64[ns]"),
        )
    else:
        ic_all = icir_all = ric_all = ricir_all = float("nan")

    report = {
        "model_kind": "il_baseline_no_DA",
        "history_start": HISTORY_START, "history_end": HISTORY_END,
        "pretrain_window": [PRETRAIN_START, PRETRAIN_END],
        "validation_window": [VALIDATION_START, VALIDATION_END],
        "online_training_window": [ONLINE_TRAINING_START, online_end],
        "universe": args.universe,
        "feature_mode": FEATURE_MODE,
        "n_candidate_pool": len(bars),
        "n_panel_rows": int(len(panel)),
        "n_meta_train_set_rows": int(len(meta_train_set)),
        "n_meta_valid_set_rows": int(len(meta_valid_set)),
        "pretrain_best_val_ic": float(pre_res["best_val_ic"]),
        "pretrain_epochs_run": len(pre_res["history"]),
        "il_warmup_n_steps": int(len(warmup_log)),
        "il_online_n_steps": int(len(step_log)),
        "il_steps_per_window": IL_STEPS_PER_WINDOW,
        "il_lr": IL_LR,
        "il_window": IL_WINDOW,
        "il_overall_ic": ic_all, "il_overall_icir": icir_all,
        "il_overall_ric": ric_all, "il_overall_ricir": ricir_all,
        "backtest": bt.summary,
        "backtest_cfg": {"capital": bt_cfg.capital, "top_k": bt_cfg.top_k,
                         "cost_alpha_bps": bt_cfg.cost_alpha_bps},
        "model_config": {"seq_len": SEQ_LEN, "n_fields": N_FIELDS,
                         "pred_horizon": PRED_HORIZON,
                         "hidden_dim": 64, "num_layers": 2,
                         "n_params": n_params},
        "trained_at": pd.Timestamp.now().isoformat(),
        "wall_time_min": (time.monotonic() - t0) / 60.0,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\n=== summary ===")
    print(f"  pretrain best val IC : {pre_res['best_val_ic']:.4f}")
    print(f"  IL steps (warmup)    : {len(warmup_log)}")
    print(f"  IL steps (live)      : {len(step_log)}")
    print(f"  IL overall IC        : {ic_all:.4f}  (RankIC {ric_all:.4f})")
    print(f"  Backtest cum return  : {bt.summary['cum_return']:+.4f}")
    print(f"  Backtest Sharpe      : {bt.summary['sharpe']:+.3f}")
    print(f"  Backtest max DD      : {bt.summary['max_drawdown']:+.4f}")
    print(f"  Total wall-time      : {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
