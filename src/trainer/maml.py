"""MAML (FOMAML) training core for DoubleAdapt.

Per the paper:
  - Inner: theta_k = phi_{k-1} - eta_theta * grad_phi L_train   (single step)
  - Outer: L_test = MSE(H^-1(F_theta(G(x_test))), y_test)
                  + alpha * MSE(H(y_test), y_test)              (regularization)
  - First-order MAML: treat theta_k as phi - constant when backproping L_test.
  - Optimizers: Adam, lr_inner=1e-3, lr_phi (MA)=1e-3, lr_psi (DA)=1e-2, alpha=0.5.

Tasks are sliced by trading-date index from a flat PanelTensors blob:
  train (= meta-train / support set) = rows whose date is in dates[i:i+r],
  test  (= meta-test  / query   set) = rows whose date is in dates[i+r:i+2r],
where r = 20 (task interval). Pretrain samples i uniformly (or sweeps all
when n_tasks is None).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from src.data.handler import PanelTensors
from src.data.processor import csrank_normalize_per_date
from src.model.double_adapt import DoubleAdapt
from src.evaluate.metrics import cross_sectional_ic
from src import runs
from src.utils.progress import pbar, log

# ----------------- task sampling -----------------


@dataclass
class Task:
    """A meta-learning task = (train, test) split adjacent in time.
    `train` is the inner-step's "support" set (past r days).
    `test`  is the outer-step's "query"  set (next r days).
    """

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    train_dates: np.ndarray
    test_dates: np.ndarray
    test_codes: np.ndarray


def _unique_sorted_dates(panel: PanelTensors) -> np.ndarray:
    return np.unique(panel.dates)


def panel_to_task(panel: PanelTensors, train_start_idx: int, r: int, device: torch.device) -> Task | None:
    udates = _unique_sorted_dates(panel)
    if train_start_idx + 2 * r > len(udates):
        return None
    train_dates = udates[train_start_idx : train_start_idx + r]
    test_dates = udates[train_start_idx + r : train_start_idx + 2 * r]
    train_mask = np.isin(panel.dates, train_dates) & np.isfinite(panel.y)
    test_mask = np.isin(panel.dates, test_dates) & np.isfinite(panel.y)
    if train_mask.sum() < 8 or test_mask.sum() < 8:
        return None
    return Task(
        x_train=torch.from_numpy(panel.X[train_mask]).to(device),
        y_train=torch.from_numpy(panel.y[train_mask]).to(device),
        x_test=torch.from_numpy(panel.X[test_mask]).to(device),
        y_test=torch.from_numpy(panel.y[test_mask]).to(device),
        train_dates=panel.dates[train_mask],
        test_dates=panel.dates[test_mask],
        test_codes=panel.codes[test_mask],
    )


def sample_pretrain_tasks(
    panel: PanelTensors,
    r: int,
    n_tasks: int | None,
    rng: np.random.Generator,
    device: torch.device,
    desc: str = "sampling tasks",
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
    for s in pbar(starts, desc=desc, unit="task", leave=False):
        t = panel_to_task(panel, int(s), r, device)
        if t is not None:
            tasks.append(t)
    return tasks


# ----------------- core FOMAML step -----------------


@dataclass
class FOMAMLConfig:
    r: int = 20  # task interval (support len = query len = r)
    inner_lr: float = 1e-3
    alpha_reg: float = 0.5
    lr_phi: float = 1e-3  # MA (base init)
    lr_psi: float = 1e-2  # DA (feature + label adapters)
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
    y_train_norm = csrank_normalize_per_date(task.y_train, task.train_dates)
    y_test_norm = csrank_normalize_per_date(task.y_test, task.test_dates)

    # Inner forward on the train set. H sees the ADAPTED x_tilde (qlib semantics).
    x_train_tilde = model.feature_adapter(task.x_train)
    y_train_tilde = model.label_adapter(x_train_tilde, y_train_norm, inverse=False)
    pred_train_raw = model.base(x_train_tilde)
    L_train = F.mse_loss(pred_train_raw, y_train_tilde)

    # Inner step on base params (FOMAML: detach grads).
    phi_named = dict(model.base.named_parameters())
    phi_names = list(phi_named.keys())
    phi_params = list(phi_named.values())
    grads = torch.autograd.grad(L_train, phi_params, create_graph=False)
    theta_named = {k: p - cfg.inner_lr * g.detach() for k, p, g in zip(phi_names, phi_params, grads)}

    # Outer forward on the test set using theta.
    x_test_tilde = model.feature_adapter(task.x_test)
    pred_test_raw = functional_call(model.base, theta_named, (x_test_tilde,))
    pred_test = model.label_adapter(x_test_tilde, pred_test_raw, inverse=True)
    y_test_tilde = model.label_adapter(x_test_tilde, y_test_norm, inverse=False)

    L_test = F.mse_loss(pred_test, y_test_norm) + cfg.alpha_reg * F.mse_loss(y_test_tilde, y_test_norm)

    if train:
        L_test.backward()

    return float(L_test.detach().item()), pred_test.detach(), task.y_test.detach()


# ----------------- pretrain loop -----------------


@dataclass
class PretrainConfig:
    epochs: int = 200  # qlib GRU n_epochs
    tasks_per_epoch: int = 200
    batch_size: int = 1
    early_stop_patience: int = 20  # qlib GRU early_stop
    seed: int = 0
    val_n_tasks: int | None = None  # None -> sweep all val starts


def pretrain_offline(
    model: DoubleAdapt,
    train_panel: PanelTensors,
    val_panel: PanelTensors,
    fcfg: FOMAMLConfig,
    pcfg: PretrainConfig,
    device: torch.device,
    log_fn: Callable[[str], None] = log,
) -> dict:
    """Offline meta-train on train_panel; early-stop on val IC."""
    rng = np.random.default_rng(pcfg.seed)
    optim_phi = torch.optim.Adam(model.ma_parameters(), lr=fcfg.lr_phi, weight_decay=fcfg.weight_decay)
    optim_psi = torch.optim.Adam(model.da_parameters(), lr=fcfg.lr_psi, weight_decay=fcfg.weight_decay)

    best_ic = -np.inf
    bad_epochs = 0
    history = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    epoch_bar = pbar(range(pcfg.epochs), desc="pretrain", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_tasks = sample_pretrain_tasks(
            train_panel, fcfg.r, pcfg.tasks_per_epoch, rng, device, desc="sampling train tasks"
        )
        train_losses = []
        optim_phi.zero_grad()
        optim_psi.zero_grad()
        ti = -1
        task_bar = pbar(train_tasks, desc=f"epoch {epoch} train", unit="task", leave=False)
        for ti, task in enumerate(task_bar):
            loss, _, _ = fomaml_step(model, task, fcfg, train=True)
            train_losses.append(loss)
            if (ti + 1) % 50 == 0:
                task_bar.set_postfix(loss=f"{np.mean(train_losses[-50:]):.5f}")
            if (ti + 1) % pcfg.batch_size == 0:
                optim_phi.step()
                optim_psi.step()
                optim_phi.zero_grad()
                optim_psi.zero_grad()
        if ti >= 0 and (ti + 1) % pcfg.batch_size != 0:
            optim_phi.step()
            optim_psi.step()
            optim_phi.zero_grad()
            optim_psi.zero_grad()

        # validation: still do inner step, just don't backprop outer.
        val_tasks = sample_pretrain_tasks(val_panel, fcfg.r, pcfg.val_n_tasks, rng, device, desc="sampling val tasks")
        all_p, all_y, all_d = [], [], []
        for task in pbar(val_tasks, desc=f"epoch {epoch} val", unit="task", leave=False):
            model.zero_grad(set_to_none=True)
            _, pred_test, y_test = fomaml_step(model, task, fcfg, train=False)
            model.zero_grad(set_to_none=True)
            all_p.append(pred_test.cpu().numpy())
            all_y.append(y_test.cpu().numpy())
            all_d.append(task.test_dates)
        if all_p:
            preds = np.concatenate(all_p)
            ys = np.concatenate(all_y)
            dates = np.concatenate(all_d)
            ic, icir, ric, ricir = cross_sectional_ic(preds, ys, dates)
        else:
            ic = icir = ric = ricir = 0.0
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_ic": ic,
                "val_icir": icir,
                "val_ric": ric,
                "val_ricir": ricir,
            }
        )
        log_fn(
            f"[pretrain] epoch {epoch:3d}: train_loss={train_loss:.5f} "
            f"val_ic={ic:.4f} icir={icir:.3f} ric={ric:.4f}"
        )

        if ic > best_ic + 1e-6:
            best_ic = ic
            bad_epochs = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        runs.log_metrics(
            {"train_loss": train_loss, "val_ic": ic, "val_icir": icir, "val_ric": ric, "val_ricir": ricir},
            step=epoch,
        )
        epoch_bar.set_postfix(loss=f"{train_loss:.5f}", val_ic=f"{ic:.4f}", best=f"{best_ic:.4f}", bad=bad_epochs)
        if bad_epochs and bad_epochs >= pcfg.early_stop_patience:
            log_fn(
                f"[pretrain] early stop at epoch {epoch} " f"(no IC improvement for {pcfg.early_stop_patience} epochs)"
            )
            break
    epoch_bar.close()

    model.load_state_dict(best_state)
    return {"history": history, "best_val_ic": best_ic}
