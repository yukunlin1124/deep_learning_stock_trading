"""Alpha360 feature builder + sequence panel — qlib-exact normalization.

For each (stock, date) where >=60 prior bars exist, build a 360-dim vector:
    6 fields x 60 lags
Fields: CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME.

Normalization matches qlib's Alpha360 (qlib/contrib/data/handler.py:103-132):
  - All 5 price-like fields (close, open, high, low, vwap) at every lag d
    are divided by TODAY'S close (`$close`, lag-0 close), NOT by their own
    lag-0 value. This makes the 5 fields commensurate on a single
    close-anchor scale and preserves today's intraday bar shape at lag-0
    (OPEN0 = $open/$close, HIGH0 = $high/$close, etc. — meaningful values,
    not all 1.0).
  - VOLUME is divided by TODAY'S volume (`$volume`, lag-0 volume).
  - VOLUME0 = 1.0; CLOSE0 = 1.0; the other 4 lag-0 fields encode today's
    intraday structure.

(Prior to 2026-06-04 this module used per-field normalization where every
lag-0 was forced to 1.0 — runs 1-16 were trained with that scheme.)

Label: next-day close-to-close return (prediction horizon = 1 day).
    label[t] = close[t + PRED_HORIZON] / close[t] - 1.0
This matches the DoubleAdapt paper's spec — the model predicts a single
1-day forward return; the 20-day cadence elsewhere in the pipeline
(FOMAMLConfig.r) is the *task interval* (= deployment window between
meta-updates), NOT the prediction horizon.

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
PRED_HORIZON = 1

FIELDS = ["close", "open", "high", "low", "vwap", "volume"]

# Default feature normalization scheme used by NEW runs:
#   "qlib"      — 5 price fields / close[t], volume / volume[t]
#                 (paper-faithful; preserves intraday bar shape at lag-0)
#   "per_field" — each field / its own lag-0 value
#                 (legacy: every lag-0 forced to 1.0; what runs 1-16 used)
#
# train.py records the active mode to each run's `feature_scheme.json`
# so deploy_today.py can reproduce the matching features at inference time.
# Older runs without that file default to "per_field" for backward compat.
FEATURE_MODE = "qlib"


def compute_alpha360(df: pd.DataFrame, pred_h: int = PRED_HORIZON,
                     mode: str = FEATURE_MODE) -> pd.DataFrame:
    """Build per-stock Alpha360 frame.
    df columns required: date, open, high, low, close, capacity, turnover.
    Returns DataFrame with columns: date, stock_code_id, X, y_fwd.
    Rows with insufficient history are dropped.

    `mode` controls the per-(stock, date) lag-0 normalization:
      "qlib"      — qlib-exact (5 price fields / close[t], vol / vol[t])
      "per_field" — legacy (each field / its own lag-0 value)
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
    if T < SEQ_LEN + pred_h:
        return pd.DataFrame(columns=["date", "stock_code_id", "X", "y_fwd"])

    label = np.full(T, np.nan, dtype=float)
    label[: T - pred_h] = c[pred_h:] / c[: T - pred_h] - 1.0

    out_rows = []
    code = str(df["stock_code_id"].iloc[0])
    dates = pd.to_datetime(df["date"]).values

    eps = 1e-9
    # Field order is [close, open, high, low, vwap, volume] — first 5 are price.
    for t in range(SEQ_LEN - 1, T):
        win = fields_arr[t - SEQ_LEN + 1 : t + 1]
        if mode == "qlib":
            # qlib-exact: 5 price fields / close[t], volume / volume[t]
            close_t = win[-1, 0]
            vol_t = win[-1, 5]
            if abs(close_t) < eps:
                close_t = eps
            if abs(vol_t) < eps:
                vol_t = eps
            denom = np.array([close_t, close_t, close_t, close_t, close_t, vol_t],
                             dtype=fields_arr.dtype)
        elif mode == "per_field":
            # legacy: each field / its own lag-0 (lag-0 row becomes [1,1,1,1,1,1])
            denom = win[-1].copy()
            denom = np.where(np.abs(denom) < eps, eps, denom)
        else:
            raise ValueError(f"unknown feature mode {mode!r}; expected 'qlib' or 'per_field'")
        X = win / denom
        if not np.isfinite(X).all():
            continue
        y = label[t]
        out_rows.append({
            "date": dates[t],
            "stock_code_id": code,
            "X": X.astype(np.float32),
            "y_fwd": float(y) if np.isfinite(y) else np.nan,
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
                pred_h: int = PRED_HORIZON,
                mode: str = FEATURE_MODE) -> PanelTensors:
    pieces = []
    for code, df in bars.items():
        if df.empty or len(df) < SEQ_LEN + pred_h:
            continue
        feat = compute_alpha360(df, pred_h=pred_h, mode=mode)
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
    y = panel["y_fwd"].astype(np.float32).values
    dates = pd.to_datetime(panel["date"]).values
    codes = panel["stock_code_id"].astype(str).values
    return PanelTensors(X=X, y=y, dates=dates, codes=codes)


if __name__ == "__main__":
    from src.data.io import fetch_range
    df = fetch_range("1101", "2021-01-01", "2023-12-31")
    feat = compute_alpha360(df)
    print(f"feature rows: {len(feat)}")
    if not feat.empty:
        x0 = feat["X"].iloc[0]
        print(f"X shape per row: {x0.shape}, dtype: {x0.dtype}")
        print(f"first date with feature: {feat['date'].iloc[0]}")
