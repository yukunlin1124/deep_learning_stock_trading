"""DoubleAdapt: FeatureAdapter (G) + LabelAdapter (H) + base GRU.

Architecture mirrors qlib reference at
double_adapter/qlib/qlib/contrib/meta/incremental/net.py exactly:

  * FeatureAdapter: per-time-step cosine-similarity gating against learned
    prototypes; multi-head residual transform with shared (across time) head
    weights. Output s has shape (B, T, N).
  * LabelAdapter: INDEPENDENT gating (Linear(x_dim, hid_dim) -> cosine vs
    own prototypes); per-head affine h_i(y) = gamma_i * y + beta_i with
    beta initialized to 1/8 and gamma ~ U(0.75, 1.25).
  * DoubleAdapt.forward(X) computes x_tilde = G(X) -> base(x_tilde) ->
    H_inverse(x_tilde, pred). Same path as qlib's DoubleAdapt.forward
    when transform=True.

Hyperparameters (N=8, tau=10) follow the DoubleAdapt paper; algorithmic
structure follows qlib.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gru import GRUBase

DEFAULT_NUM_HEAD = 8
DEFAULT_TEMPERATURE = 10.0
DEFAULT_HID_DIM = 32


class FeatureAdapter(nn.Module):
    """Per-time-step cosine-gated multi-head residual transform.
    Same logic as qlib FeatureAdapter (net.py:76-91)."""

    def __init__(self, in_dim: int, num_head: int = DEFAULT_NUM_HEAD,
                 temperature: float = DEFAULT_TEMPERATURE):
        super().__init__()
        self.in_dim = in_dim
        self.num_head = num_head
        self.temperature = temperature
        self.P = nn.Parameter(torch.empty(num_head, in_dim))
        nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.heads = nn.ModuleList([
            nn.Linear(in_dim, in_dim, bias=True) for _ in range(num_head)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sims = torch.cat([
            F.cosine_similarity(x, self.P[i], dim=-1).unsqueeze(-1)
            for i in range(self.num_head)
        ], dim=-1)
        s = F.softmax(sims / self.temperature, dim=-1).unsqueeze(-1)
        head_outs = torch.stack([h(x) for h in self.heads], dim=-2)
        return x + (s * head_outs).sum(dim=-2)


class LabelAdaptHeads(nn.Module):
    """h_i(y) = gamma_i * y + beta_i. Mirrors qlib LabelAdaptHeads."""

    def __init__(self, num_head: int = DEFAULT_NUM_HEAD):
        super().__init__()
        self.num_head = num_head
        self.weight = nn.Parameter(torch.empty(1, num_head))
        self.bias = nn.Parameter(torch.ones(1, num_head) / 8.0)
        nn.init.uniform_(self.weight, 0.75, 1.25)

    def forward(self, y: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if inverse:
            return (y.view(-1, 1) - self.bias) / (self.weight + 1e-9)
        return (self.weight + 1e-9) * y.view(-1, 1) + self.bias


class LabelAdapter(nn.Module):
    """Multi-head label transform with independent cosine gating.
    Mirrors qlib LabelAdapter (net.py:47-63)."""

    def __init__(self, x_dim: int, num_head: int = DEFAULT_NUM_HEAD,
                 temperature: float = DEFAULT_TEMPERATURE,
                 hid_dim: int = DEFAULT_HID_DIM):
        super().__init__()
        self.x_dim = x_dim
        self.num_head = num_head
        self.temperature = temperature
        self.hid_dim = hid_dim
        self.linear = nn.Linear(x_dim, hid_dim, bias=False)
        self.P = nn.Parameter(torch.empty(num_head, hid_dim))
        nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.heads = LabelAdaptHeads(num_head)

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        v = self.linear(flat)
        sims = F.cosine_similarity(v.unsqueeze(1), self.P.unsqueeze(0), dim=-1)
        return F.softmax(sims / self.temperature, dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                inverse: bool = False) -> torch.Tensor:
        gate = self._gate(x)
        heads_out = self.heads(y, inverse=inverse)
        return (gate * heads_out).sum(dim=-1)


class DoubleAdapt(nn.Module):
    """G -> base -> H wrapper. forward(x) returns inverse-transformed prediction."""

    def __init__(self, seq_len: int = 60, n_fields: int = 6,
                 hidden_dim: int = 64, num_layers: int = 2,
                 num_head: int = DEFAULT_NUM_HEAD,
                 temperature: float = DEFAULT_TEMPERATURE,
                 hid_dim: int = DEFAULT_HID_DIM):
        super().__init__()
        self.feature_adapter = FeatureAdapter(in_dim=n_fields, num_head=num_head,
                                              temperature=temperature)
        self.base = GRUBase(input_dim=n_fields, hidden_dim=hidden_dim,
                            num_layers=num_layers)
        self.label_adapter = LabelAdapter(
            x_dim=seq_len * n_fields, num_head=num_head,
            temperature=temperature, hid_dim=hid_dim,
        )
        self.seq_len = seq_len
        self.n_fields = n_fields

    def da_parameters(self):
        return list(self.feature_adapter.parameters()) + list(self.label_adapter.parameters())

    def ma_parameters(self):
        return list(self.base.parameters())

    def base_named_parameters(self):
        return dict(self.base.named_parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tilde = self.feature_adapter(x)
        pred_raw = self.base(x_tilde)
        return self.label_adapter(x_tilde, pred_raw, inverse=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoubleAdapt().to(device)
    print(f"device: {device}")
    print(f"#DA params: {sum(p.numel() for p in model.da_parameters()):,}")
    print(f"#MA params: {sum(p.numel() for p in model.ma_parameters()):,}")
    x = torch.randn(8, 60, 6, device=device)
    y = torch.randn(8, device=device)
    x_t = model.feature_adapter(x)
    y_t = model.label_adapter(x_t, y, inverse=False)
    y_b = model.label_adapter(x_t, y_t, inverse=True)
    print(f"round-trip max err: {(y_b - y).abs().max().item():.5f}")
    print(f"pred shape: {tuple(model(x).shape)}")
