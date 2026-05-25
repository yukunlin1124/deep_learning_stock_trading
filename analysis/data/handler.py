"""Alpha360 feature builder + sequence panel.

For each (stock, date) where >=60 prior bars exist, build a 360-dim vector:
    6 fields x 60 lags
Fields: CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME.
Each field at lag d in [59, 58, ..., 1, 0] is normalized by the lag-0 value:
    feature = field[t-d] / field[t]
The result reshapes naturally to (60, 6) for a sequence model.

Label: 20-day forward return on close.

build_panel(bars) returns a PanelTensors with:
    X     : (N, 60, 6) float32
    y     : (N,)       float32
    dates : (N,)       datetime64
    codes : (N,)       object/str
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEQ_LEN = 60
N_FIELDS = 6
FEATURE_DIM = SEQ_LEN * N_FIELDS  # 360
LABEL_HORIZON = 20

FIELDS = ["close", "open", "high", "low", "vwap", "volume"]


def compute_alpha360(df: pd.DataFrame, label_h: int = LABEL_HORIZON) -> pd.DataFrame:
    """Build per-stock Alpha360 frame.
    df columns required: date, open, high, low, close, capacity, turnover.
    Returns DataFrame with columns: date, stock_code_id, X, y_fwd20.
    Rows with insufficient history are dropped.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    c = df["close"].astype(float).values
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    lo = df["low"].astype(float).values
    cap = df["capacity"].astype(float).values
    tov = df["turnover"].astype(float).values

    vwap = np.where(cap > 0, tov / np.maximum(cap, 1e-9), c)

    fields_arr = np.stack([c, o, h, lo, vwap, cap], axis=1)  # (T, 6)
    T = fields_arr.shape[0]
    if T < SEQ_LEN + label_h:
        return pd.DataFrame(columns=["date", "stock_code_id", "X", "y_fwd20"])

    label = np.full(T, np.nan, dtype=float)
    label[: T - label_h] = c[label_h:] / c[: T - label_h] - 1.0

    out_rows = []
    code = str(df["stock_code_id"].iloc[0])
    dates = pd.to_datetime(df["date"]).values

    eps = 1e-9
    for t in range(SEQ_LEN - 1, T):
        win = fields_arr[t - SEQ_LEN + 1 : t + 1]
        denom = win[-1]
        denom = np.where(np.abs(denom) < eps, eps, denom)
        X = win / denom
        if not np.isfinite(X).all():
            continue
        y = label[t]
        out_rows.append({
            "date": dates[t],
            "stock_code_id": code,
            "X": X.astype(np.float32),
            "y_fwd20": float(y) if np.isfinite(y) else np.nan,
        })
    return pd.DataFrame(out_rows)


@dataclass
class PanelTensors:
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    codes: np.ndarray
    feature_dim: int = FEATURE_DIM
    seq_len: int = SEQ_LEN
    n_fields: int = N_FIELDS

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def slice_by_date(self, start: pd.Timestamp, end: pd.Timestamp) -> "PanelTensors":
        s = np.datetime64(pd.Timestamp(start))
        e = np.datetime64(pd.Timestamp(end))
        mask = (self.dates >= s) & (self.dates <= e)
        return PanelTensors(X=self.X[mask], y=self.y[mask],
                            dates=self.dates[mask], codes=self.codes[mask])

    def drop_nan_labels(self) -> "PanelTensors":
        m = np.isfinite(self.y)
        return PanelTensors(X=self.X[m], y=self.y[m],
                            dates=self.dates[m], codes=self.codes[m])


def build_panel(bars: dict[str, pd.DataFrame],
                label_h: int = LABEL_HORIZON) -> PanelTensors:
    pieces = []
    for code, df in bars.items():
        if df.empty or len(df) < SEQ_LEN + label_h:
            continue
        feat = compute_alpha360(df, label_h=label_h)
        if not feat.empty:
            pieces.append(feat)
    if not pieces:
        return PanelTensors(
            X=np.zeros((0, SEQ_LEN, N_FIELDS), dtype=np.float32),
            y=np.zeros((0,), dtype=np.float32),
            dates=np.zeros((0,), dtype="datetime64[ns]"),
            codes=np.zeros((0,), dtype=object),
        )
    panel = pd.concat(pieces, ignore_index=True)
    panel = panel.sort_values(["date", "stock_code_id"]).reset_index(drop=True)
    X = np.stack(panel["X"].values, axis=0).astype(np.float32)
    y = panel["y_fwd20"].astype(np.float32).values
    dates = pd.to_datetime(panel["date"]).values
    codes = panel["stock_code_id"].astype(str).values
    return PanelTensors(X=X, y=y, dates=dates, codes=codes)


if __name__ == "__main__":
    from analysis.data.io import fetch_range
    df = fetch_range("1101", "2021-01-01", "2023-12-31")
    feat = compute_alpha360(df)
    print(f"feature rows: {len(feat)}")
    if not feat.empty:
        x0 = feat["X"].iloc[0]
        print(f"X shape per row: {x0.shape}, dtype: {x0.dtype}")
        print(f"first date with feature: {feat['date'].iloc[0]}")
