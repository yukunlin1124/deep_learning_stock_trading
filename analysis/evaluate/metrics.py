"""Cross-sectional IC / RankIC / ICIR metrics."""
from __future__ import annotations

import numpy as np


def cross_sectional_ic(preds: np.ndarray, ys: np.ndarray, dates: np.ndarray
                       ) -> tuple[float, float, float, float]:
    """Per-date Pearson and Spearman correlation, aggregated.
    Returns (mean IC, ICIR, mean RankIC, RankICIR).
    """
    udates = np.unique(dates)
    ics: list[float] = []
    rics: list[float] = []
    for d in udates:
        m = dates == d
        if m.sum() < 5:
            continue
        p = preds[m]
        a = ys[m]
        if np.std(p) < 1e-9 or np.std(a) < 1e-9:
            continue
        ic = float(np.corrcoef(p, a)[0, 1])
        rp = np.argsort(np.argsort(p))
        ra = np.argsort(np.argsort(a))
        ric = float(np.corrcoef(rp, ra)[0, 1])
        if np.isfinite(ic):
            ics.append(ic)
        if np.isfinite(ric):
            rics.append(ric)
    if not ics:
        return 0.0, 0.0, 0.0, 0.0
    ics_a = np.array(ics)
    rics_a = np.array(rics)
    icir = float(ics_a.mean() / (ics_a.std() + 1e-9))
    ricir = float(rics_a.mean() / (rics_a.std() + 1e-9))
    return float(ics_a.mean()), icir, float(rics_a.mean()), ricir
