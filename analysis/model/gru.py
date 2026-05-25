"""Base GRU predictor F. Defaults mirror qlib's pytorch_gru."""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUBase(nn.Module):
    """GRU(input=6, hidden=64, num_layers=2, dropout=0) -> Linear -> scalar.
    Defaults follow qlib qlib/contrib/model/pytorch_gru.py:47-55.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)
