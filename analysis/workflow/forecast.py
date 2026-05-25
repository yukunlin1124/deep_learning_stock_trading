"""Live forecast: G -> base -> H^-1 on the freshest (label-less) panel rows.

After the IL phase, the model state is the most up-to-date. We can score the
most recent dates whose features exist but whose labels do not yet exist
(because the next 20 trading days haven't elapsed). Produces a "what does
the model think today's top picks are" CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from analysis.data.handler import PanelTensors
from analysis.model.double_adapt import DoubleAdapt


def forecast_latest(
    model: DoubleAdapt,
    panel: PanelTensors,
    n_days: int = 20,
    device: torch.device | None = None,
) -> pd.DataFrame:
    if device is None:
        device = next(model.parameters()).device
    if len(panel) == 0:
        return pd.DataFrame(columns=["date", "stock_code_id", "pred"])

    udates = np.unique(panel.dates)
    take_dates = udates[-n_days:]
    mask = np.isin(panel.dates, take_dates)
    if not mask.any():
        return pd.DataFrame(columns=["date", "stock_code_id", "pred"])

    X = torch.from_numpy(panel.X[mask]).to(device)
    dates = panel.dates[mask]
    codes = panel.codes[mask]

    # Pure inference — no backward needed, so eval() (not train()) is correct.
    model.eval()
    with torch.no_grad():
        preds = model(X).cpu().numpy()

    df = pd.DataFrame({
        "date": [pd.Timestamp(d).date() for d in dates],
        "stock_code_id": [str(c) for c in codes],
        "pred": preds.astype(float),
    })
    return df.sort_values(["date", "pred"], ascending=[True, False]).reset_index(drop=True)
