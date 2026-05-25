"""qlib-style preprocessors for the Alpha360 panel.

Mirrors qlib's standard GRU+Alpha360 workflow processors:
  - RobustZScoreNorm (features): fit on pretrain rows only;
        x_norm = clip((x - median) / (1.4826 * MAD), -3, +3)
  - Fillna (features): replace remaining NaN with 0.0
  - CSRankNorm (labels, per cross-section): for each date, rank labels and
        normalize to mean 0, scaled by 1/sqrt(12).

CSRankNorm is applied at TRAINING time inside fomaml_step (not baked into the
panel) because the per-date grouping depends on the task slice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # noqa: F401  (kept for callers re-importing through this module)
import torch


@dataclass
class RobustZScoreStats:
    median: np.ndarray   # (60, 6)
    mad: np.ndarray      # (60, 6)


def fit_robust_zscore(X_pretrain: np.ndarray) -> RobustZScoreStats:
    """X_pretrain: (N, 60, 6). Computes per-(lag, field) median + MAD."""
    median = np.nanmedian(X_pretrain, axis=0).astype(np.float32)
    mad = np.nanmedian(np.abs(X_pretrain - median), axis=0).astype(np.float32)
    mad = np.where(mad < 1e-6, 1.0, mad)
    return RobustZScoreStats(median=median, mad=mad)


def apply_robust_zscore(X: np.ndarray, stats: RobustZScoreStats,
                        clip: float = 3.0) -> np.ndarray:
    out = (X - stats.median[None]) / (1.4826 * stats.mad[None])
    if clip is not None:
        np.clip(out, -clip, clip, out=out)
    return out.astype(np.float32)


def fillna(X: np.ndarray, value: float = 0.0) -> np.ndarray:
    return np.where(np.isnan(X), value, X).astype(np.float32)


def csrank_normalize_per_date(y: torch.Tensor, dates: np.ndarray
                              ) -> torch.Tensor:
    """Per-date CSRankNorm. Within each unique date, rank values 1..N, then
    normalize so the result has unit variance (under uniform-rank assumption).
    """
    out = torch.empty_like(y)
    udates = np.unique(dates)
    for d in udates:
        m = (dates == d)
        idx = torch.from_numpy(m).to(y.device)
        vals = y[idx]
        n = int(vals.numel())
        if n < 2:
            out[idx] = 0.0
            continue
        ranks = vals.argsort().argsort().float() + 1.0
        normed = (ranks / float(n)) - 0.5
        out[idx] = normed / (1.0 / np.sqrt(12.0))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 60, 6)).astype(np.float32)
    X[0, 0, 0] = np.nan
    stats = fit_robust_zscore(X)
    Xn = apply_robust_zscore(X, stats)
    Xn = fillna(Xn)
    print("post-norm mean/std:", Xn.mean(), Xn.std(), "nan count:", np.isnan(Xn).sum())
