"""Daily auto-trader for the TWSE virtual trading system.

Schedule (typical):
   T-1 15:30-16:00  TWSE settlement window (course system updates bars)
   T-1 16:00 onward (or T 11:00)  scrape today's bar:
                                      python analysis/scripts/scrape.py
   T   12:00        run THIS script -- 1 hour before order window closes:
                                      python analysis/scripts/deploy_today.py

Causal contract:
   - Features for prediction use bars through (T - 1)  (yesterday's close).
   - The model outputs a prediction PER stock (rank-norm 20d forward return).
   - We pick top-K above the cost-alpha threshold, throttle by n_drop, then
     submit Buy_Stock / Sell_Stock at limit prices rounded from (T - 1) close.
   - TWSE will fill iff today's [low, high] covers the limit (project Rule 02).

Trading config is loaded from analysis/output/bt_config.json so it stays
identical to whatever the backtest used.

Credentials:
   Read from env vars TWSE_ACCOUNT and TWSE_PASSWORD. CLI args override.

Run:
   # Dry run (no orders submitted, just prints intended actions):
   python analysis/scripts/deploy_today.py --dry-run

   # Live trading (will hit Buy_Stock / Sell_Stock):
   $env:TWSE_ACCOUNT="..."
   $env:TWSE_PASSWORD="..."
   python analysis/scripts/deploy_today.py

Logs every intended/submitted order to analysis/output/orders_YYYYMMDD.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.data.io import fetch_range
from analysis.data.handler import (
    compute_alpha360, SEQ_LEN, N_FIELDS, LABEL_HORIZON,
)
from analysis.data.processor import apply_robust_zscore, fillna
from analysis.model.double_adapt import DoubleAdapt
from analysis.evaluate.trading_rules import (
    SHARES_PER_LOT, round_buy, round_sell,
)
from stock_api import Buy_Stock, Sell_Stock, Get_User_Stocks

OUT_BASE = PROJECT_ROOT / "analysis" / "output"
HISTORY_START = "2024-01-01"   # ~2 years of bars is plenty for a 60-day window

# Daily order log lives at output/ root (spans runs, not per-run).
LOG_PATH = OUT_BASE / f"orders_{datetime.now().strftime('%Y%m%d')}.csv"


def resolve_run_dir(run_override: str | None) -> Path:
    """Pick which output/runN/ to load. CLI flag overrides CURRENT pointer."""
    if run_override:
        d = OUT_BASE / run_override
    else:
        cur_file = OUT_BASE / "CURRENT"
        if not cur_file.exists():
            raise SystemExit(f"missing {cur_file}; pass --run runN or run train.py first.")
        d = OUT_BASE / cur_file.read_text().strip()
    if not d.exists():
        raise SystemExit(f"run dir {d} does not exist.")
    return d


# ----------------- loaders -----------------

def load_artifacts(device: torch.device, run_dir: Path) -> tuple:
    model_path = run_dir / "final.pt"
    if not model_path.exists():
        raise SystemExit(f"missing {model_path}. Run analysis/scripts/train.py first.")
    model = DoubleAdapt(seq_len=SEQ_LEN, n_fields=N_FIELDS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(run_dir / "norm_stats.pkl", "rb") as f:
        norm_stats = pickle.load(f)

    universe_blob = json.loads((run_dir / "active_universe.json").read_text())
    universe = universe_blob["codes"]

    bt_cfg = json.loads((run_dir / "bt_config.json").read_text())
    return model, norm_stats, universe, bt_cfg


# ----------------- inference -----------------

def predict_for_yesterday(
    model: DoubleAdapt,
    norm_stats,
    universe: list[str],
    device: torch.device,
) -> pd.DataFrame:
    """Build features at lag-0 = the most recent cached bar (= yesterday's close)
    and return predictions per stock.

    Returns: DataFrame(code, asof, pred, prev_close)
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for code in universe:
        try:
            df = fetch_range(code, HISTORY_START, today_str)
        except Exception as e:
            print(f"[skip] {code}: fetch failed: {e!r}")
            continue
        if len(df) < SEQ_LEN + 1:
            print(f"[skip] {code}: only {len(df)} bars (need >= {SEQ_LEN + 1})")
            continue
        feat = compute_alpha360(df, label_h=1).tail(1)  # label horizon irrelevant here
        if feat.empty:
            continue
        rows.append({
            "code": code,
            "asof": pd.Timestamp(feat["date"].iloc[0]),
            "X": feat["X"].iloc[0],
            "prev_close": float(df["close"].iloc[-1]),   # yesterday's close
        })
    if not rows:
        raise RuntimeError("no stocks produced features; check the cache.")
    X = np.stack([r["X"] for r in rows]).astype(np.float32)
    X = fillna(apply_robust_zscore(X, norm_stats))
    with torch.no_grad():
        preds = model(torch.from_numpy(X).to(device)).cpu().numpy()
    out = pd.DataFrame({
        "code": [r["code"] for r in rows],
        "asof": [r["asof"] for r in rows],
        "pred": preds.astype(float),
        "prev_close": [r["prev_close"] for r in rows],
    })
    return out.sort_values("pred", ascending=False).reset_index(drop=True)


# ----------------- holdings + orders -----------------

def parse_holdings(raw) -> dict[str, int]:
    """Convert Get_User_Stocks response into {code: shares}. The course API's
    return shape isn't strictly specified; this handles list-of-dicts and
    string fallback (treat unknown as 'held' so we don't double-buy).
    """
    holdings: dict[str, int] = {}
    if raw is None:
        return holdings
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                code = str(item.get("stock_code") or item.get("code") or "").strip()
                shares = int(item.get("shares") or item.get("quantity") or 0)
                if code:
                    holdings[code] = shares
    elif isinstance(raw, str):
        # Best-effort: regex pull 4-digit codes from a stringified blob.
        for m in re.finditer(r"\b(\d{4})\b\D{1,40}?(\d+)", raw):
            holdings.setdefault(m.group(1), int(m.group(2)))
    return holdings


@dataclass
class IntendedOrder:
    side: str          # "BUY" or "SELL"
    code: str
    lots: int
    price: int         # int per stock_api requirements


def plan_orders(
    preds_df: pd.DataFrame,
    holdings: dict[str, int],
    bt_cfg: dict,
) -> list[IntendedOrder]:
    capital = float(bt_cfg.get("capital", 1e8))
    top_k = int(bt_cfg.get("top_k", 10))
    cost_alpha = float(bt_cfg.get("cost_alpha_bps", 70.0)) / 1e4
    n_drop = bt_cfg.get("n_drop", None)
    n_drop = int(n_drop) if n_drop is not None else None

    target = (
        preds_df[preds_df["pred"] >= cost_alpha]
        .sort_values("pred", ascending=False)
        .head(top_k)["code"].tolist()
    )
    target_set = set(target)
    held_set = set(holdings.keys())

    # Sell candidates: held names not in target, lowest pred first.
    pred_lookup = dict(zip(preds_df["code"], preds_df["pred"]))
    sell_candidates = sorted(held_set - target_set,
                             key=lambda c: pred_lookup.get(c, -np.inf))
    buy_candidates = [c for c in target if c not in held_set]

    if n_drop is not None:
        sell_candidates = sell_candidates[: n_drop]
        buy_candidates = buy_candidates[: n_drop]

    prev_close = dict(zip(preds_df["code"], preds_df["prev_close"]))
    orders: list[IntendedOrder] = []
    slot_capital = capital / top_k

    # SELL first to free cash
    for code in sell_candidates:
        pc = prev_close.get(code)
        if pc is None or not np.isfinite(pc):
            continue
        sell_price = int(round(round_sell(pc)))
        lots = int(holdings[code] // SHARES_PER_LOT)
        if lots <= 0:
            continue
        orders.append(IntendedOrder("SELL", code, lots, sell_price))

    # BUY
    for code in buy_candidates:
        pc = prev_close.get(code)
        if pc is None or not np.isfinite(pc):
            continue
        buy_price = int(round(round_buy(pc)))
        if buy_price <= 0:
            continue
        lots = int(slot_capital // (buy_price * SHARES_PER_LOT))
        if lots <= 0:
            continue
        orders.append(IntendedOrder("BUY", code, lots, buy_price))
    return orders


# ----------------- submission + logging -----------------

def submit_orders(orders: list[IntendedOrder], account: str, password: str,
                  dry_run: bool) -> list[dict]:
    log_rows = []
    for o in orders:
        attempted_at = datetime.now().isoformat(timespec="seconds")
        if dry_run:
            success = None
        else:
            fn = Buy_Stock if o.side == "BUY" else Sell_Stock
            try:
                success = bool(fn(account, password, int(o.code), o.lots, o.price))
            except Exception as e:
                print(f"[error] {o.side} {o.code} x{o.lots} @ {o.price}: {e!r}")
                success = False
        status = "DRY_RUN" if dry_run else ("OK" if success else "FAIL")
        print(f"  [{status}] {o.side:4s} {o.code} lots={o.lots:3d} price={o.price}")
        log_rows.append({
            "attempted_at": attempted_at,
            "side": o.side, "code": o.code,
            "lots": o.lots, "price": o.price,
            "status": status,
        })
    return log_rows


# ----------------- main -----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Daily auto-trader for the course virtual TWSE")
    ap.add_argument("--dry-run", action="store_true",
                    help="print intended orders without calling Buy_Stock/Sell_Stock")
    ap.add_argument("--account", default=os.environ.get("TWSE_ACCOUNT"),
                    help="course account (default: $TWSE_ACCOUNT)")
    ap.add_argument("--password", default=os.environ.get("TWSE_PASSWORD"),
                    help="course password (default: $TWSE_PASSWORD)")
    ap.add_argument("--n-drop", type=int, default=None,
                    help="override BTConfig.n_drop for this run only "
                         "(use n_drop=top_k on day 1 to fill the book)")
    ap.add_argument("--run", default=None,
                    help="which output/runN/ to load (default: read output/CURRENT)")
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.run)

    if not args.dry_run and (not args.account or not args.password):
        raise SystemExit("missing credentials. Set TWSE_ACCOUNT + TWSE_PASSWORD "
                         "env vars, or pass --account/--password, or use --dry-run.")

    t0 = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, dry_run={args.dry_run}")

    print(f"\n[1/4] loading artifacts from {run_dir.name}/ (model, norm stats, universe, bt_config)...")
    model, norm_stats, universe, bt_cfg = load_artifacts(device, run_dir)
    if args.n_drop is not None:
        print(f"      OVERRIDE: n_drop {bt_cfg.get('n_drop')} -> {args.n_drop} (CLI flag)")
        bt_cfg["n_drop"] = args.n_drop
    print(f"      universe size: {len(universe)}, bt_cfg: {bt_cfg}")

    print("\n[2/4] running inference on yesterday's features...")
    preds = predict_for_yesterday(model, norm_stats, universe, device)
    asof = preds["asof"].iloc[0]
    print(f"      asof (= 'yesterday' = lag-0): {pd.Timestamp(asof).date()}, "
          f"got predictions for {len(preds)} stocks")
    print(f"      top 5: {preds.head(5)[['code','pred','prev_close']].to_string(index=False)}")

    print("\n[3/4] reading current holdings + planning orders...")
    if args.dry_run and not args.account:
        print("      (dry-run + no credentials -> assuming empty book)")
        holdings = {}
    else:
        try:
            raw = Get_User_Stocks(args.account, args.password)
        except Exception as e:
            print(f"      Get_User_Stocks failed: {e!r}  (assuming empty book)")
            raw = None
        holdings = parse_holdings(raw)
    print(f"      current holdings: {len(holdings)} positions")
    orders = plan_orders(preds, holdings, bt_cfg)
    print(f"      planned: {sum(1 for o in orders if o.side=='SELL')} sells, "
          f"{sum(1 for o in orders if o.side=='BUY')} buys")

    print("\n[4/4] submitting orders...")
    log_rows = submit_orders(orders, args.account, args.password, args.dry_run)

    if log_rows:
        log_df = pd.DataFrame(log_rows)
        if LOG_PATH.exists():
            log_df = pd.concat([pd.read_csv(LOG_PATH), log_df], ignore_index=True)
        log_df.to_csv(LOG_PATH, index=False)
        print(f"\nlogged {len(log_rows)} order(s) to {LOG_PATH}")
    else:
        print("\nno orders to log.")

    print(f"\ntotal wall-time: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
