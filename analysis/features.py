"""30 hand-picked features for TWSE daily ranking.

All features are computed *causal* (use values at and before the current bar
so concatenation with a shifted label gives no look-ahead).

Inputs: a DataFrame with columns
    date, open, high, low, close, capacity, turnover, transaction_volume

Returns a new DataFrame with the same `date` index plus 30 feature columns
and a target column `y_fwd20` (20-day forward return based on close).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS: list[str] = [
    # returns / momentum (8)
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "ma5_over_ma20", "ma20_over_ma60", "close_over_ma20",
    # mean reversion / position (4)
    "zscore_20d", "zscore_60d", "range_pos_20d", "range_pos_60d",
    # volatility (4)
    "vol_5d", "vol_20d", "vol_60d", "vol_ratio_5_20",
    # volume / liquidity (5)
    "log_capacity", "volume_z20", "volume_z60",
    "turnover_z20", "volume_ratio_5_20",
    # range / bar shape (3)
    "hl_range", "atr20", "body_ratio",
    # vwap-based (2)
    "vwap_dev", "vwap_ma20_dev",
    # trend persistence + gap (4)
    "up_days_5", "drawdown_20", "drawup_20", "gap_open",
]
assert len(FEATURE_COLS) == 30, len(FEATURE_COLS)

LABEL_COL = "y_fwd20"


def _zscore(s: pd.Series, w: int) -> pd.Series:
    m = s.rolling(w).mean()
    sd = s.rolling(w).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)


def _true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs(),
    ], axis=1).max(axis=1)


def compute_features(df: pd.DataFrame, label_h: int = 20) -> pd.DataFrame:
    """Compute features + forward-return label for a single stock."""
    df = df.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": df["date"]})

    c = df["close"]
    o = df["open"]
    h = df["high"]
    lo = df["low"]
    cap = df["capacity"].astype(float)
    tov = df["turnover"].astype(float)

    # returns
    out["ret_1d"] = c.pct_change(1)
    out["ret_5d"] = c.pct_change(5)
    out["ret_10d"] = c.pct_change(10)
    out["ret_20d"] = c.pct_change(20)
    out["ret_60d"] = c.pct_change(60)

    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    out["ma5_over_ma20"] = ma5 / ma20 - 1
    out["ma20_over_ma60"] = ma20 / ma60 - 1
    out["close_over_ma20"] = c / ma20 - 1

    # mean reversion
    std20 = c.rolling(20).std(ddof=0)
    std60 = c.rolling(60).std(ddof=0)
    out["zscore_20d"] = (c - ma20) / std20.replace(0, np.nan)
    out["zscore_60d"] = (c - ma60) / std60.replace(0, np.nan)
    rng20 = c.rolling(20).max() - c.rolling(20).min()
    rng60 = c.rolling(60).max() - c.rolling(60).min()
    out["range_pos_20d"] = (c - c.rolling(20).min()) / rng20.replace(0, np.nan)
    out["range_pos_60d"] = (c - c.rolling(60).min()) / rng60.replace(0, np.nan)

    # volatility (of returns)
    r1 = out["ret_1d"]
    out["vol_5d"] = r1.rolling(5).std(ddof=0)
    out["vol_20d"] = r1.rolling(20).std(ddof=0)
    out["vol_60d"] = r1.rolling(60).std(ddof=0)
    out["vol_ratio_5_20"] = out["vol_5d"] / out["vol_20d"].replace(0, np.nan)

    # volume / liquidity
    out["log_capacity"] = np.log1p(cap)
    out["volume_z20"] = _zscore(cap, 20)
    out["volume_z60"] = _zscore(cap, 60)
    out["turnover_z20"] = _zscore(tov, 20)
    out["volume_ratio_5_20"] = (
        cap.rolling(5).mean() / cap.rolling(20).mean().replace(0, np.nan)
    )

    # range / bar shape
    out["hl_range"] = (h - lo) / c.replace(0, np.nan)
    tr = _true_range(df)
    out["atr20"] = tr.rolling(20).mean() / c.replace(0, np.nan)
    body = (c - o)
    span = (h - lo).replace(0, np.nan)
    out["body_ratio"] = body / span

    # vwap-based
    vwap = (tov / cap.replace(0, np.nan))
    out["vwap_dev"] = c / vwap - 1
    out["vwap_ma20_dev"] = vwap.rolling(20).mean() / vwap - 1

    # trend persistence
    up = (out["ret_1d"] > 0).astype(int)
    out["up_days_5"] = up.rolling(5).sum()
    out["drawdown_20"] = 1 - c / c.rolling(20).max()
    out["drawup_20"] = c / c.rolling(20).min() - 1
    out["gap_open"] = o / c.shift(1) - 1

    # label: 20d forward return based on close
    out[LABEL_COL] = c.shift(-label_h) / c - 1

    out["stock_code_id"] = df["stock_code_id"].astype(str)
    return out


def compute_for_universe(
    frames: dict[str, pd.DataFrame], label_h: int = 20
) -> pd.DataFrame:
    """Concatenate per-stock feature frames into a panel (long-format).

    Drops rows missing label or any feature."""
    pieces = []
    for code, df in frames.items():
        if df.empty:
            continue
        feat = compute_features(df, label_h=label_h)
        pieces.append(feat)
    panel = pd.concat(pieces, ignore_index=True)
    needed = FEATURE_COLS + [LABEL_COL]
    panel = panel.dropna(subset=needed).reset_index(drop=True)
    return panel


if __name__ == "__main__":
    # quick sanity check on the TSMC cache
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_io import fetch_range
    df = fetch_range("2330", "2023-01-01", "2025-12-31")
    feat = compute_features(df)
    print("feature rows:", len(feat))
    print("non-null counts:")
    print(feat[FEATURE_COLS + [LABEL_COL]].notna().sum())
    print("\nlast 3 rows of selected features:")
    cols = ["date"] + FEATURE_COLS[:5] + [LABEL_COL]
    print(feat.tail(3)[cols].to_string(index=False))
