"""TWSE training universe — official TEJ index masks (TW50 + TM100).

Two indices are supported, both loaded from TEJ's monthly snapshot CSVs
under src/data/stock_data_all/:

  TW50  — FTSE TWSE Taiwan 50 (the top-50 large-caps; ~50 names per month,
          ~99 unique codes across 2011-2026)
  TM100 — FTSE TWSE Taiwan Mid-Cap 100 (the next 100 below TW50; ~100 names
          per month, ~266 unique codes across 2011-2026)

Both files have the same TEJ format:
  Row 1: title (comma-separated rebalance dates)
  Row 2: column headers (年月日, 證券, 權數因子, 流通係數, ..., 股數, 除數,
                         前日調整收盤, 前日市值比)
  Data rows: one row per (snapshot_date, stock) listing the index's
             constituents on that date with their free-float weight factor.

The training pipeline (`src/scripts/train.py`) consumes one of these indices
as its time-varying active universe; the per-day predictions then run
through the qlib or course backtest engine.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Active training universe size per period (informational; the actual size
# comes from the index file — TW50 = 50, TM100 = 100).
TOP_N = 50

REBALANCE_PER_YEAR = 2          # semi-annual rebalance for build_universe_schedule_*
MIN_TRAILING_ROWS = 10          # cold-start safety for any liquidity helper


# ---------- TEJ index CSV loader (TW50 / TM100) ----------

_STOCK_DATA_DIR = Path(__file__).resolve().parent / "stock_data_all"
_INDEX_PATHS = {
    "TW50":  _STOCK_DATA_DIR / "TW50.csv",
    "TM100": _STOCK_DATA_DIR / "TM100.csv",
}

# Per-index caches keyed by index name
_index_state_caches: dict[str, dict[pd.Timestamp, list[str]]] = {}
_index_weight_caches: dict[str, dict[pd.Timestamp, dict[str, float]]] = {}


def _parse_tej_index_csv(path: Path) -> tuple[dict[pd.Timestamp, list[str]],
                                              dict[pd.Timestamp, dict[str, float]]]:
    """Parse a TEJ index-constituent CSV into (state_dict, weights_dict).

    state[asof]   = sorted list of constituent codes (4-digit strings)
    weights[asof] = {code: free-float-normalized weight summing to 1.0}
    """
    df = pd.read_csv(path, encoding="cp950", skiprows=1, dtype=str)
    df.columns = [
        "date_raw", "code_name", "weight_factor", "float_factor", "limit_factor",
        "shares_out", "divisor", "prev_close", "prev_mcap_ratio",
    ]
    df["date"] = pd.to_datetime(df["date_raw"], format="%Y/%m/%d")
    # "1101 台泥" -> "1101"
    df["code"] = df["code_name"].str.split().str[0]
    df = df[df["code"].str.match(r"^\d{4}$")]

    state: dict[pd.Timestamp, list[str]] = {}
    weights: dict[pd.Timestamp, dict[str, float]] = {}
    for asof_raw, grp in df.groupby("date"):
        asof = pd.Timestamp(asof_raw)  # type: ignore[arg-type]
        codes = sorted(grp["code"].unique())
        state[asof] = codes
        w = pd.to_numeric(grp["weight_factor"], errors="coerce").fillna(0.0)
        total_w = float(w.sum())
        if total_w > 0:
            weights[asof] = dict(zip(grp["code"], (w / total_w).astype(float)))
        else:
            weights[asof] = {c: 1.0 / len(codes) for c in codes}
    return state, weights


def _get_index_history(index_name: str) -> dict[pd.Timestamp, list[str]]:
    """Return cached {snapshot_date -> [codes]} for one of the known indices."""
    if index_name in _index_state_caches:
        return _index_state_caches[index_name]
    path = _INDEX_PATHS.get(index_name)
    if path is None:
        raise ValueError(f"unknown index {index_name!r}; expected one of {list(_INDEX_PATHS)}")
    if not path.exists():
        raise FileNotFoundError(f"missing TEJ index CSV: {path}")
    state, weights = _parse_tej_index_csv(path)
    _index_state_caches[index_name] = state
    _index_weight_caches[index_name] = weights
    return state


def _index_lookup(index_name: str, asof: str | pd.Timestamp) -> pd.Timestamp:
    """Find the latest snapshot_date in `index_name` that is <= asof."""
    history = _get_index_history(index_name)
    asof = pd.Timestamp(asof)
    keys = sorted(history.keys())
    chosen = None
    for k in keys:
        if k <= asof:
            chosen = k
        else:
            break
    return chosen if chosen is not None else keys[0]


def get_index_codes_at(index_name: str, asof: str | pd.Timestamp) -> list[str]:
    """Constituent codes for the given index on the given date."""
    return list(_get_index_history(index_name)[_index_lookup(index_name, asof)])


def get_index_weights_at(index_name: str, asof: str | pd.Timestamp) -> dict[str, float]:
    """Free-float weights (sum to 1.0) for the given index on the given date."""
    _get_index_history(index_name)  # populate caches
    weights_cache = _index_weight_caches.get(index_name, {})
    if not weights_cache:
        return {}
    return dict(weights_cache[_index_lookup(index_name, asof)])


def get_index_universe_codes(index_name: str) -> list[str]:
    """Union of all historical constituents of the given index.

    Use as a scrape target (we only need bars for stocks that have ever been
    in the index). TW50: ~99 codes, TM100: ~266 codes.
    """
    history = _get_index_history(index_name)
    union: set[str] = set()
    for codes in history.values():
        union.update(codes)
    return sorted(union)


# ---------- TW50 wrappers (back-compat with existing imports) ----------

def get_tw50_at(asof: str | pd.Timestamp) -> list[str]:
    return get_index_codes_at("TW50", asof)


def get_tw50_weights_at(asof: str | pd.Timestamp) -> dict[str, float]:
    return get_index_weights_at("TW50", asof)


def get_tw50_universe_codes() -> list[str]:
    return get_index_universe_codes("TW50")


# ---------- TM100 wrappers ----------

def get_tm100_at(asof: str | pd.Timestamp) -> list[str]:
    return get_index_codes_at("TM100", asof)


def get_tm100_weights_at(asof: str | pd.Timestamp) -> dict[str, float]:
    return get_index_weights_at("TM100", asof)


def get_tm100_universe_codes() -> list[str]:
    return get_index_universe_codes("TM100")


# ---------- Rebalance calendar + schedule builder ----------

def rebalance_dates(start: str, end: str,
                    rebalances_per_year: int = REBALANCE_PER_YEAR) -> list[pd.Timestamp]:
    """Semi-annual (Jan 1 + Jul 1 by default) rebalance dates in [start, end]."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    months_per_rb = max(1, 12 // rebalances_per_year)
    out: list[pd.Timestamp] = []
    cur = pd.Timestamp(year=s.year, month=1, day=1)
    while cur <= e:
        if cur >= s:
            out.append(cur)
        cur = cur + pd.DateOffset(months=months_per_rb)
    if not out or out[0] > s:
        out = [s] + out
    return out


def build_universe_schedule(index_name: str, start: str, end: str,
                            rebalances_per_year: int = REBALANCE_PER_YEAR,
                            ) -> pd.DataFrame:
    """Schedule of (asof, codes, weights) per rebalance for the chosen index.

    Looks up the official TEJ history; no liquidity ranking needed.
    """
    rows = []
    for asof in rebalance_dates(start, end, rebalances_per_year):
        codes = get_index_codes_at(index_name, asof)
        weights = get_index_weights_at(index_name, asof)
        rows.append({
            "asof": asof.date(),
            "n_active": len(codes),
            "codes": codes,
            "weights": weights,
        })
    return pd.DataFrame(rows)


def build_universe_schedule_tw50(start: str, end: str,
                                  rebalances_per_year: int = REBALANCE_PER_YEAR,
                                  ) -> pd.DataFrame:
    return build_universe_schedule("TW50", start, end, rebalances_per_year)


def build_universe_schedule_tm100(start: str, end: str,
                                   rebalances_per_year: int = REBALANCE_PER_YEAR,
                                   ) -> pd.DataFrame:
    return build_universe_schedule("TM100", start, end, rebalances_per_year)


def active_codes_for_date(date: pd.Timestamp, schedule: pd.DataFrame) -> set[str]:
    """Which codes were active on the given date, per the schedule."""
    if schedule.empty:
        return set()
    asofs = pd.to_datetime(schedule["asof"])
    mask = asofs <= date
    if not mask.any():
        return set(schedule.iloc[0]["codes"])
    idx = int(mask.values.nonzero()[0][-1])
    return set(schedule.iloc[idx]["codes"])


if __name__ == "__main__":
    for idx in ("TW50", "TM100"):
        union = get_index_universe_codes(idx)
        latest = _index_lookup(idx, pd.Timestamp.now())
        snap = get_index_codes_at(idx, latest)
        print(f"{idx}: {len(union)} unique codes over history, "
              f"latest snapshot {latest.date()} has {len(snap)} codes")
