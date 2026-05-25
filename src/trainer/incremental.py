"""Fake-online incremental learning phase.

Walks forward through the IL window in steps of r=20 trading days. At each
step:
  - support = past r trading days (with labels)
  - query   = next r trading days
  - Run one FOMAML bi-level step:
       inner: theta = phi - eta * grad_phi L_train  (support)
       outer: L_test backprop -> update DA (psi) and MA (phi)
  - Record predictions + IC/RankIC over the query window.

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
    log_fn=print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    optim_phi = torch.optim.Adam(model.ma_parameters(), lr=fcfg.lr_phi,
                                 weight_decay=fcfg.weight_decay)
    optim_psi = torch.optim.Adam(model.da_parameters(), lr=fcfg.lr_psi,
                                 weight_decay=fcfg.weight_decay)

    udates = np.unique(panel.dates)
    start = np.datetime64(pd.Timestamp(icfg.start_date))
    end = np.datetime64(pd.Timestamp(icfg.end_date))
    qry_first_idx = int(np.searchsorted(udates, start, side="left"))
    qry_first_idx = max(qry_first_idx, fcfg.r)

    rows = []
    preds_rows = []
    step = 0
    i = qry_first_idx
    while i + fcfg.r <= len(udates):
        sup_start = udates[i - fcfg.r]
        task = panel_to_task(panel, i - fcfg.r, fcfg.r, device)
        if task is None:
            i += fcfg.r; step += 1
            continue
        if udates[i] > end:
            break

        optim_phi.zero_grad(set_to_none=True)
        optim_psi.zero_grad(set_to_none=True)
        test_loss, pred_qry, y_qry = fomaml_step(model, task, fcfg, train=True)
        optim_phi.step(); optim_psi.step()

        preds_np = pred_qry.cpu().numpy()
        ys_np = y_qry.cpu().numpy()
        d_np = task.qry_dates
        c_np = task.qry_codes
        ic, icir, ric, ricir = cross_sectional_ic(preds_np, ys_np, d_np)

        rows.append({
            "step": step,
            "support_start": pd.Timestamp(sup_start).date(),
            "support_end": pd.Timestamp(udates[i - 1]).date(),
            "query_start": pd.Timestamp(udates[i]).date(),
            "query_end": pd.Timestamp(udates[min(i + fcfg.r - 1, len(udates) - 1)]).date(),
            "n_support": int(task.x_sup.shape[0]),
            "n_query": int(task.x_qry.shape[0]),
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

        log_fn(f"[IL] step {step:3d}: q={pd.Timestamp(udates[i]).date()}.."
               f"{pd.Timestamp(udates[min(i+fcfg.r-1, len(udates)-1)]).date()} "
               f"n_sup={task.x_sup.shape[0]:5d} n_qry={task.x_qry.shape[0]:5d} "
               f"loss={test_loss:.5f} ic={ic:+.4f} ric={ric:+.4f}")

        i += fcfg.r
        step += 1

    return pd.DataFrame(rows), pd.DataFrame(preds_rows)
