"""DDG-DA-style sample re-weighting layer (simplified reimplementation).

Idea (from arXiv:2201.04038 in plain English):
  When training to predict period [t, t+H], not all historical samples are
  equally informative. Some past periods resemble what is about to happen
  next; others don't. DDG-DA learns a model that predicts the *near-future
  data distribution* and uses that prediction to up-weight similar history.

What this module does:
  Given a training panel (long-format features + label) and a `near_future`
  feature panel (the most-recent K trading days, no label yet), return a
  vector of sample weights for the training panel based on similarity to
  the near-future distribution.

Similarity uses a low-rank Gaussian-RBF kernel on the standardised features:
  w_i = mean_j exp(-||z_i - z_j||^2 / (2*bandwidth^2))
where z's are PCA-projected (default 8 components) feature vectors.

This is a faithful but compact reimplementation that drops qlib's heavier
bi-level optimisation in exchange for being usable in a 23-day project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_weights(
    train_X: pd.DataFrame,
    near_future_X: pd.DataFrame,
    n_components: int = 8,
    bandwidth: float | None = None,
    floor: float = 0.1,
) -> np.ndarray:
    """Return weights of length len(train_X), normalised to mean 1.

    `floor` clips per-sample weight from below so no sample is fully ignored
    (avoids the bi-level optimisation collapse the paper warns about).
    """
    if len(near_future_X) == 0:
        return np.ones(len(train_X))

    scaler = StandardScaler().fit(train_X)
    Z_train = scaler.transform(train_X)
    Z_near = scaler.transform(near_future_X)

    n_comp = min(n_components, Z_train.shape[1], Z_train.shape[0])
    pca = PCA(n_components=n_comp).fit(Z_train)
    P_train = pca.transform(Z_train)
    P_near = pca.transform(Z_near)

    if bandwidth is None:
        # Silverman-ish rule on the projected near-future cloud
        bandwidth = max(np.std(P_near, axis=0).mean(), 1e-3)

    # Pairwise squared distances chunked to bound memory
    n_train = P_train.shape[0]
    n_near = P_near.shape[0]
    weights = np.zeros(n_train)
    chunk = 4096
    for s in range(0, n_train, chunk):
        e = min(s + chunk, n_train)
        diff = P_train[s:e, None, :] - P_near[None, :, :]
        d2 = (diff ** 2).sum(axis=2)
        weights[s:e] = np.exp(-d2 / (2 * bandwidth ** 2)).mean(axis=1)

    # normalise to mean 1 and floor
    weights = weights / weights.mean()
    weights = np.maximum(weights, floor)
    weights = weights / weights.mean()
    return weights


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    feat_cols = [f"f{i}" for i in range(5)]
    tr = pd.DataFrame(rng.normal(size=(200, 5)), columns=feat_cols)
    nf = pd.DataFrame(rng.normal(loc=1.0, size=(20, 5)), columns=feat_cols)
    w = compute_weights(tr, nf, n_components=4)
    print("weights mean/std/min/max:",
          w.mean(), w.std(), w.min(), w.max())
