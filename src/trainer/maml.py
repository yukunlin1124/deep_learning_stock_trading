"""MAML (FOMAML) training core for DoubleAdapt.

Per the paper:
  - Inner: theta_k = phi_{k-1} - eta_theta * grad_phi L_train   (single step)
  - Outer: L_test = MSE(H^-1(F_theta(G(x_query))), y_query)
                  + alpha * MSE(H(y_query), y_query)            (regularization)
  - First-order MAML: treat theta_k as phi - constant when backproping L_test.
  - Optimizers: Adam, lr_inner=1e-3, lr_phi (MA)=1e-3, lr_psi (DA)=1e-2, alpha=0.5.

Tasks are sliced by trading-date index from a flat PanelTensors blob:
  support = rows whose date is in dates[i:i+r],
  query   = rows whose date is in dates[i+r:i+2r],
where r = 20 (task interval). Pretrain samples i uniformly (or sweeps all
when n_tasks is None).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from src.data.handler import PanelTensors
from src.data.processor import csrank_normalize_per_date
from src.model.double_adapt import DoubleAdapt
from src.evaluate.metrics import cross_sectional_ic


# ----------------- task sampling -----------------

@dataclass
class Task:
    x_sup: torch.Tensor
    y_sup: torch.Tensor
    x_qry: torch.Tensor
    y_qry: torch.Tensor
    sup_dates: np.ndarray
    qry_dates: np.ndarray
    qry_codes: np.ndarray


def _unique_sorted_dates(panel: PanelTensors) -> np.ndarray:
    return np.unique(panel.dates)


def panel_to_task(panel: PanelTensors, sup_start_idx: int, r: int,
                  device: torch.device) -> Task | None:
    udates = _unique_sorted_dates(panel)
    if sup_start_idx + 2 * r > len(udates):
        return None
    sup_dates = udates[sup_start_idx : sup_start_idx + r]
    qry_dates = udates[sup_start_idx + r : sup_start_idx + 2 * r]
    sup_mask = np.isin(panel.dates, sup_dates) & np.isfinite(panel.y)
    qry_mask = np.isin(panel.dates, qry_dates) & np.isfinite(panel.y)
    if sup_mask.sum() < 8 or qry_mask.sum() < 8:
        return None
    return Task(
        x_sup=torch.from_numpy(panel.X[sup_mask]).to(device),
        y_sup=torch.from_numpy(panel.y[sup_mask]).to(device),
        x_qry=torch.from_numpy(panel.X[qry_mask]).to(device),
        y_qry=torch.from_numpy(panel.y[qry_mask]).to(device),
        sup_dates=panel.dates[sup_mask],
        qry_dates=panel.dates[qry_mask],
        qry_codes=panel.codes[qry_mask],
    )


def sample_pretrain_tasks(
    panel: PanelTensors, r: int, n_tasks: int | None,
    rng: np.random.Generator, device: torch.device,
) -> list[Task]:
    """If n_tasks is None, sweep ALL valid task starts deterministically."""
    udates = _unique_sorted_dates(panel)
    max_start = len(udates) - 2 * r
    if max_start <= 0:
        return []
    if n_tasks is None:
        starts = np.arange(max_start)
    else:
        starts = rng.integers(0, max_start, size=n_tasks)
    tasks: list[Task] = []
    for s in starts:
        t = panel_to_task(panel, int(s), r, device)
        if t is not None:
            tasks.append(t)
    return tasks


# ----------------- core FOMAML step -----------------

@dataclass
class FOMAMLConfig:
    r: int = 20            # task interval (support len = query len = r)
    inner_lr: float = 1e-3
    alpha_reg: float = 0.5
    lr_phi: float = 1e-3   # MA (base init)
    lr_psi: float = 1e-2   # DA (feature + label adapters)
    weight_decay: float = 0.0


def fomaml_step(
    model: DoubleAdapt,
    task: Task,
    cfg: FOMAMLConfig,
    train: bool = True,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """One FOMAML step on a single task.
    Returns (test_loss_scalar, pred_query (Nq,), y_query_raw (Nq,)).
    cudnn RNN backward requires train() mode; the `train` flag only controls
    whether the OUTER loss is backproped.
    """
    model.train()
    y_sup_norm = csrank_normalize_per_date(task.y_sup, task.sup_dates)
    y_qry_norm = csrank_normalize_per_date(task.y_qry, task.qry_dates)

    # Inner forward on support. H sees the ADAPTED x_tilde (qlib semantics).
    x_sup_tilde = model.feature_adapter(task.x_sup)
    y_sup_tilde = model.label_adapter(x_sup_tilde, y_sup_norm, inverse=False)
    pred_sup_raw = model.base(x_sup_tilde)
    L_train = F.mse_loss(pred_sup_raw, y_sup_tilde)

    # Inner step on base params (FOMAML: detach grads).
    phi_named = dict(model.base.named_parameters())
    phi_names = list(phi_named.keys())
    phi_params = list(phi_named.values())
    grads = torch.autograd.grad(L_train, phi_params, create_graph=False)
    theta_named = {
        k: p - cfg.inner_lr * g.detach()
        for k, p, g in zip(phi_names, phi_params, grads)
    }

    # Outer forward on query using theta.
    x_qry_tilde = model.feature_adapter(task.x_qry)
    pred_qry_raw = functional_call(model.base, theta_named, (x_qry_tilde,))
    pred_qry = model.label_adapter(x_qry_tilde, pred_qry_raw, inverse=True)
    y_qry_tilde = model.label_adapter(x_qry_tilde, y_qry_norm, inverse=False)

    L_test = (F.mse_loss(pred_qry, y_qry_norm)
              + cfg.alpha_reg * F.mse_loss(y_qry_tilde, y_qry_norm))

    if train:
        L_test.backward()

    return float(L_test.detach().item()), pred_qry.detach(), task.y_qry.detach()


# ----------------- pretrain loop -----------------

@dataclass
class PretrainConfig:
    epochs: int = 200                # qlib GRU n_epochs
    tasks_per_epoch: int = 200
    batch_size: int = 1
    early_stop_patience: int = 20    # qlib GRU early_stop
    seed: int = 0
    val_n_tasks: int | None = None   # None -> sweep all val starts


def pretrain_offline(
    model: DoubleAdapt,
    train_panel: PanelTensors,
    val_panel: PanelTensors,
    fcfg: FOMAMLConfig,
    pcfg: PretrainConfig,
    device: torch.device,
    log_fn=print,
) -> dict:
    """Offline meta-train on train_panel; early-stop on val IC."""
    rng = np.random.default_rng(pcfg.seed)
    optim_phi = torch.optim.Adam(model.ma_parameters(), lr=fcfg.lr_phi,
                                 weight_decay=fcfg.weight_decay)
    optim_psi = torch.optim.Adam(model.da_parameters(), lr=fcfg.lr_psi,
                                 weight_decay=fcfg.weight_decay)

    best_ic = -np.inf
    bad_epochs = 0
    history = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(pcfg.epochs):
        model.train()
        train_tasks = sample_pretrain_tasks(train_panel, fcfg.r,
                                            pcfg.tasks_per_epoch, rng, device)
        train_losses = []
        optim_phi.zero_grad(); optim_psi.zero_grad()
        ti = -1
        for ti, task in enumerate(train_tasks):
            loss, _, _ = fomaml_step(model, task, fcfg, train=True)
            train_losses.append(loss)
            if (ti + 1) % pcfg.batch_size == 0:
                optim_phi.step(); optim_psi.step()
                optim_phi.zero_grad(); optim_psi.zero_grad()
        if ti >= 0 and (ti + 1) % pcfg.batch_size != 0:
            optim_phi.step(); optim_psi.step()
            optim_phi.zero_grad(); optim_psi.zero_grad()

        # validation: still do inner step, just don't backprop outer.
        val_tasks = sample_pretrain_tasks(val_panel, fcfg.r,
                                          pcfg.val_n_tasks, rng, device)
        all_p, all_y, all_d = [], [], []
        for task in val_tasks:
            model.zero_grad(set_to_none=True)
            _, pred_qry, y_qry = fomaml_step(model, task, fcfg, train=False)
            model.zero_grad(set_to_none=True)
            all_p.append(pred_qry.cpu().numpy())
            all_y.append(y_qry.cpu().numpy())
            all_d.append(task.qry_dates)
        if all_p:
            preds = np.concatenate(all_p)
            ys = np.concatenate(all_y)
            dates = np.concatenate(all_d)
            ic, icir, ric, ricir = cross_sectional_ic(preds, ys, dates)
        else:
            ic = icir = ric = ricir = 0.0
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_ic": ic, "val_icir": icir,
            "val_ric": ric, "val_ricir": ricir,
        })
        log_fn(f"[pretrain] epoch {epoch:3d}: train_loss={train_loss:.5f} "
               f"val_ic={ic:.4f} icir={icir:.3f} ric={ric:.4f}")

        if ic > best_ic + 1e-6:
            best_ic = ic
            bad_epochs = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= pcfg.early_stop_patience:
                log_fn(f"[pretrain] early stop at epoch {epoch} "
                       f"(no IC improvement for {pcfg.early_stop_patience} epochs)")
                break

    model.load_state_dict(best_state)
    return {"history": history, "best_val_ic": best_ic}
