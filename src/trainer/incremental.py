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

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.data.handler import PanelTensors
from src.model.double_adapt import DoubleAdapt
from src.evaluate.metrics import cross_sectional_ic
from src.trainer.maml import FOMAMLConfig, panel_to_task, fomaml_step
from src.utils.progress import pbar, log, n_walk_steps


@dataclass
class ILConfig:
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"


def run_incremental(
    model: DoubleAdapt,
    panel: PanelTensors,
    fcfg: FOMAMLConfig,
    icfg: ILConfig,
    device: torch.device,
    log_fn=log,
    desc: str = "IL",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    optim_phi = torch.optim.Adam(model.ma_parameters(), lr=fcfg.lr_phi,
                                 weight_decay=fcfg.weight_decay)
    optim_psi = torch.optim.Adam(model.da_parameters(), lr=fcfg.lr_psi,
                                 weight_decay=fcfg.weight_decay)

    udates = np.unique(panel.dates)
    start = np.datetime64(pd.Timestamp(icfg.start_date))
    end = np.datetime64(pd.Timestamp(icfg.end_date))
    test_first_idx = int(np.searchsorted(udates, start, side="left"))
    test_first_idx = max(test_first_idx, fcfg.r)

    rows = []
    preds_rows = []
    step = 0
    i = test_first_idx
    bar = pbar(total=n_walk_steps(udates, test_first_idx, fcfg.r, end),
               desc=desc, unit="step")
    while i + fcfg.r <= len(udates):
        train_start = udates[i - fcfg.r]
        if udates[i] > end:
            break
        bar.update(1)
        task = panel_to_task(panel, i - fcfg.r, fcfg.r, device)
        if task is None:
            i += fcfg.r; step += 1
            continue

        optim_phi.zero_grad(set_to_none=True)
        optim_psi.zero_grad(set_to_none=True)
        test_loss, pred_test, y_test = fomaml_step(model, task, fcfg, train=True)
        optim_phi.step(); optim_psi.step()

        preds_np = pred_test.cpu().numpy()
        ys_np = y_test.cpu().numpy()
        d_np = task.test_dates
        c_np = task.test_codes
        ic, icir, ric, ricir = cross_sectional_ic(preds_np, ys_np, d_np)

        rows.append({
            "step": step,
            "train_start": pd.Timestamp(train_start).date(),
            "train_end": pd.Timestamp(udates[i - 1]).date(),
            "test_start": pd.Timestamp(udates[i]).date(),
            "test_end": pd.Timestamp(udates[min(i + fcfg.r - 1, len(udates) - 1)]).date(),
            "n_train": int(task.x_train.shape[0]),
            "n_test": int(task.x_test.shape[0]),
            "test_loss": float(test_loss),
            "ic": float(ic), "icir": float(icir),
            "ric": float(ric), "ricir": float(ricir),
        })
        for d, c, p, a in zip(d_np, c_np, preds_np, ys_np):
            preds_rows.append({
                "date": pd.Timestamp(d).date(),
                "stock_code_id": str(c),
                "pred": float(p), "actual": float(a),
            })

        log_fn(f"[IL] step {step:3d}: t={pd.Timestamp(udates[i]).date()}.."
               f"{pd.Timestamp(udates[min(i+fcfg.r-1, len(udates)-1)]).date()} "
               f"n_train={task.x_train.shape[0]:5d} n_test={task.x_test.shape[0]:5d} "
               f"loss={test_loss:.5f} ic={ic:+.4f} ric={ric:+.4f}")
        bar.set_postfix(t=str(pd.Timestamp(udates[i]).date()), ic=f"{ic:+.4f}")

        i += fcfg.r
        step += 1

    bar.close()
    return pd.DataFrame(rows), pd.DataFrame(preds_rows)
