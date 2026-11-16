"""Live forecast: G -> base -> H^-1 on the freshest (label-less) panel rows.

After the IL phase, the model state is the most up-to-date. We can score the
most recent dates whose features exist but whose labels do not yet exist
(because the next 20 trading days haven't elapsed). Produces a "what does
the model think today's top picks are" CSV.

predict_for_yesterday() is the live-trading path: it builds features from the
cached bars up to the latest close and scores the run's active universe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from datetime import datetime

from src.config import LIVE_HISTORY_START
from src.data.handler import PanelTensors, SEQ_LEN, compute_alpha360
from src.data.io import fetch_range
from src.data.processor import apply_robust_zscore, fillna
from src.model.double_adapt import DoubleAdapt
from src.utils.progress import log, pbar


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

    df = pd.DataFrame(
        {
            "date": [pd.Timestamp(d).date() for d in dates],
            "stock_code_id": [str(c) for c in codes],
            "pred": preds.astype(float),
        }
    )
    return df.sort_values(["date", "pred"], ascending=[True, False]).reset_index(drop=True)


def predict_for_yesterday(
    model: DoubleAdapt,
    norm_stats: Any,
    universe: list[str],
    device: torch.device,
    feature_mode: str = "qlib",
) -> pd.DataFrame:
    """Build features at lag-0 = the most recent cached bar (= yesterday's close)
    and return predictions per stock.

    `feature_mode` must match the scheme the run was trained with
    ("qlib" for run17+, "per_field" for runs 1-16). Wrong mode = silent
    feature distortion → garbage predictions.

    Returns: DataFrame(code, asof, pred, prev_close)
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for code in pbar(universe, desc="building features", unit="stock"):
        try:
            df = fetch_range(code, LIVE_HISTORY_START, today_str)
        except Exception as e:
            log(f"[skip] {code}: fetch failed: {e!r}")
            continue
        if len(df) < SEQ_LEN + 1:
            log(f"[skip] {code}: only {len(df)} bars (need >= {SEQ_LEN + 1})")
            continue
        feat = compute_alpha360(df, pred_h=1, mode=feature_mode).tail(1)  # pred horizon irrelevant here
        if feat.empty:
            continue
        rows.append(
            {
                "code": code,
                "asof": pd.Timestamp(feat["date"].iloc[0]),
                "X": feat["X"].iloc[0],
                "prev_close": float(df["close"].iloc[-1]),  # yesterday's close
            }
        )
    if not rows:
        raise RuntimeError("no stocks produced features; check the cache.")
    X = np.stack([r["X"] for r in rows]).astype(np.float32)
    X = fillna(apply_robust_zscore(X, norm_stats))
    with torch.no_grad():
        preds = model(torch.from_numpy(X).to(device)).cpu().numpy()
    out = pd.DataFrame(
        {
            "code": [r["code"] for r in rows],
            "asof": [r["asof"] for r in rows],
            "pred": preds.astype(float),
            "prev_close": [r["prev_close"] for r in rows],
        }
    )
    return out.sort_values("pred", ascending=False).reset_index(drop=True)
