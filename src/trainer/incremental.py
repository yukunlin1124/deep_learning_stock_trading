"""Fake-online incremental learning phase.

Walks forward through the IL window in steps of r=20 trading days. At each
step:
  - train (= meta-train / support set) = past r trading days
  - test  (= meta-test  / query   set) = next r trading days
  - Run one FOMAML bi-level step:
       inner: theta = phi - eta * grad_phi L_train  (on train set)
       outer: L_test backprop -> update DA (psi) and MA (phi)
  - Record predictions + IC/RankIC over the test window.

Per the paper: keep meta-learner LRs ON during deployment so adapters drift
with the market.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.data.handler import PanelTensors
from src.model.double_adapt import DoubleAdapt
from src.evaluate.metrics import cross_sectional_ic
from src.trainer.maml import FOMAMLConfig, panel_to_task, fomaml_step
from src.evaluate.backtest_qlib import BTResult, PortfolioSim, QlibBTConfig
from src.utils.progress import pbar, log, n_walk_steps
from src.config import ONLINE_TRAINING_START


@dataclass
class ILConfig:
    start_date: str = ONLINE_TRAINING_START
    end_date: str = "2025-12-31"


def run_incremental(
    model: DoubleAdapt,
    panel: PanelTensors,
    fcfg: FOMAMLConfig,
    icfg: ILConfig,
    device: torch.device,
    log_fn: Callable[[str], None] = log,
    desc: str = "IL",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    optim_phi = torch.optim.Adam(model.ma_parameters(), lr=fcfg.lr_phi, weight_decay=fcfg.weight_decay)
    optim_psi = torch.optim.Adam(model.da_parameters(), lr=fcfg.lr_psi, weight_decay=fcfg.weight_decay)

    udates = np.unique(panel.dates)
    start = np.datetime64(pd.Timestamp(icfg.start_date))
    end = np.datetime64(pd.Timestamp(icfg.end_date))
    test_first_idx = int(np.searchsorted(udates, start, side="left"))
    test_first_idx = max(test_first_idx, fcfg.r)

    rows = []
    preds_rows = []
    step = 0
    i = test_first_idx
    bar = pbar(total=n_walk_steps(udates, test_first_idx, fcfg.r, end), desc=desc, unit="step")
    while i + fcfg.r <= len(udates):
        train_start = udates[i - fcfg.r]
        if udates[i] > end:
            break
        bar.update(1)
        task = panel_to_task(panel, i - fcfg.r, fcfg.r, device)
        if task is None:
            i += fcfg.r
            step += 1
            continue

        optim_phi.zero_grad(set_to_none=True)
        optim_psi.zero_grad(set_to_none=True)
        test_loss, pred_test, y_test = fomaml_step(model, task, fcfg, train=True)
        optim_phi.step()
        optim_psi.step()

        preds_np = pred_test.cpu().numpy()
        ys_np = y_test.cpu().numpy()
        d_np = task.test_dates
        c_np = task.test_codes
        ic, icir, ric, ricir = cross_sectional_ic(preds_np, ys_np, d_np)

        rows.append(
            {
                "step": step,
                "train_start": pd.Timestamp(train_start).date(),
                "train_end": pd.Timestamp(udates[i - 1]).date(),
                "test_start": pd.Timestamp(udates[i]).date(),
                "test_end": pd.Timestamp(udates[min(i + fcfg.r - 1, len(udates) - 1)]).date(),
                "n_train": int(task.x_train.shape[0]),
                "n_test": int(task.x_test.shape[0]),
                "test_loss": float(test_loss),
                "ic": float(ic),
                "icir": float(icir),
                "ric": float(ric),
                "ricir": float(ricir),
            }
        )
        for d, c, p, a in zip(d_np, c_np, preds_np, ys_np):
            preds_rows.append(
                {
                    "date": pd.Timestamp(d).date(),
                    "stock_code_id": str(c),
                    "pred": float(p),
                    "actual": float(a),
                }
            )

        log_fn(
            f"[IL] step {step:3d}: t={pd.Timestamp(udates[i]).date()}.."
            f"{pd.Timestamp(udates[min(i + fcfg.r - 1, len(udates) - 1)]).date()} "
            f"n_train={task.x_train.shape[0]:5d} n_test={task.x_test.shape[0]:5d} "
            f"loss={test_loss:.5f} ic={ic:+.4f} ric={ric:+.4f}"
        )
        bar.set_postfix(t=str(pd.Timestamp(udates[i]).date()), ic=f"{ic:+.4f}")

        i += fcfg.r
        step += 1

    bar.close()
    return pd.DataFrame(rows), pd.DataFrame(preds_rows)


# ---------- Live-IL backtest: model fine-tunes every r days during the run ----------


def run_qlib_backtest_with_il(
    model: DoubleAdapt,  # mutated in-place by FOMAML steps
    il_panel: PanelTensors,  # covers [start_date - 90d, end_date]
    bars: dict[str, pd.DataFrame],
    fcfg: FOMAMLConfig,  # inner_lr, lr_phi, lr_psi, r, etc.
    cfg: QlibBTConfig | None = None,
    start_date: str = ONLINE_TRAINING_START,
    end_date: str | None = None,
    device: torch.device | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[BTResult, pd.DataFrame, pd.DataFrame]:
    """Walk il_panel forward in r-day windows. Each window:
      1. Build (train, test) task; do one FOMAML bi-level update on the train set;
         get predictions for the test days.
      2. For each day in the test window: take top-K from those predictions and
         trade them through evaluate.backtest_qlib.PortfolioSim (same rules as
         `run_qlib_backtest_from_predictions`).

    This is the "live IL during backtest" mode — the model state at trade time
    on day D reflects everything it has adapted through D - 20. Reproduces
    deploy-style behavior more faithfully than the replay-from-predictions path.

    Returns:
      (BTResult, step_log_df, predictions_df)
        step_log_df: per-IL-step IC/RankIC/loss (same shape as run_incremental)
        predictions_df: per-(date, stock) pred/actual (same shape as predictions.csv)
    """
    log_fn = log_fn or log
    cfg = cfg or QlibBTConfig()
    thresh = cfg.cost_alpha_bps / 1e4

    # Optimizers — adapt model state in-place during the backtest
    optim_phi = torch.optim.Adam(model.ma_parameters(), lr=fcfg.lr_phi, weight_decay=fcfg.weight_decay)
    optim_psi = torch.optim.Adam(model.da_parameters(), lr=fcfg.lr_psi, weight_decay=fcfg.weight_decay)

    sim = PortfolioSim(bars, cfg)

    # IL walk
    udates = np.unique(il_panel.dates)
    start = np.datetime64(pd.Timestamp(start_date))
    end = np.datetime64(pd.Timestamp(end_date)) if end_date else udates[-1]
    test_first_idx = max(int(np.searchsorted(udates, start, side="left")), fcfg.r)

    step_log_rows = []
    pred_rows = []
    i = test_first_idx
    step = 0
    bar = pbar(total=n_walk_steps(udates, test_first_idx, fcfg.r, end), desc="live-IL backtest", unit="step")
    while i + fcfg.r <= len(udates):
        if udates[i] > end:
            break
        bar.update(1)
        task = panel_to_task(il_panel, i - fcfg.r, fcfg.r, device)
        if task is None:
            i += fcfg.r
            step += 1
            continue

        # ---- (1) IL step: adapt + predict ----
        optim_phi.zero_grad(set_to_none=True)
        optim_psi.zero_grad(set_to_none=True)
        test_loss, pred_test, y_test = fomaml_step(model, task, fcfg, train=True)
        optim_phi.step()
        optim_psi.step()

        preds_np = pred_test.cpu().detach().numpy()
        ys_np = y_test.cpu().detach().numpy()
        d_np = task.test_dates
        c_np = task.test_codes
        ic, icir, ric, ricir = cross_sectional_ic(preds_np, ys_np, d_np)

        step_log_rows.append(
            {
                "step": step,
                "train_start": pd.Timestamp(udates[i - fcfg.r]).date(),
                "train_end": pd.Timestamp(udates[i - 1]).date(),
                "test_start": pd.Timestamp(udates[i]).date(),
                "test_end": pd.Timestamp(udates[min(i + fcfg.r - 1, len(udates) - 1)]).date(),
                "n_train": int(task.x_train.shape[0]),
                "n_test": int(task.x_test.shape[0]),
                "test_loss": float(test_loss),
                "ic": float(ic),
                "icir": float(icir),
                "ric": float(ric),
                "ricir": float(ricir),
            }
        )

        # ---- (2) Build per-(date, code) pred lookup for the test window ----
        # task.test_dates is parallel with preds_np
        day_preds: dict[pd.Timestamp, dict[str, float]] = {}
        for d, c, p, a in zip(d_np, c_np, preds_np, ys_np):
            d_ts = pd.Timestamp(d)
            day_preds.setdefault(d_ts, {})[str(c)] = float(p)
            pred_rows.append(
                {
                    "date": d_ts.date(),
                    "stock_code_id": str(c),
                    "pred": float(p),
                    "actual": float(a),
                }
            )

        # ---- (3) Simulate portfolio for each day in the test window ----
        for day_idx in range(fcfg.r):
            di = i + day_idx
            if di >= len(udates):
                break
            today = pd.Timestamp(udates[di])
            if today > pd.Timestamp(end):
                break

            today_preds = day_preds.get(today, {})
            target = sorted(
                ((c, p) for c, p in today_preds.items() if p >= thresh),
                key=lambda x: -x[1],
            )[: cfg.top_k]
            sim.trade_day(today, [c for c, _ in target], today_preds)

        log_fn(
            f"[live-IL] step {step:3d}: q={pd.Timestamp(udates[i]).date()}.."
            f"{pd.Timestamp(udates[min(i + fcfg.r - 1, len(udates) - 1)]).date()} "
            f"loss={test_loss:.5f} ic={ic:+.4f} ric={ric:+.4f} "
            f"equity={sim.last_equity:,.0f}"
            if sim.equity_rows
            else ""
        )
        if sim.equity_rows:
            bar.set_postfix(ic=f"{ic:+.4f}", equity=f"{sim.last_equity:,.0f}")
        i += fcfg.r
        step += 1
    bar.close()

    return sim.result(), pd.DataFrame(step_log_rows), pd.DataFrame(pred_rows)
