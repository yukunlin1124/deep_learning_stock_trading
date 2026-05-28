"""Daily auto-trader for the TWSE virtual trading system.

Schedule (typical):
   T-1 15:30-16:00  TWSE settlement window (course system updates bars)
   weekly/monthly   drop fresh TEJ half-year CSV into src/data/stock_data_all/
                    and re-run: python src/scripts/import_tej.py
   T   12:00        run THIS script -- 1 hour before order window closes:
                                      python src/scripts/deploy_today.py --cash NT$X

Causal contract:
   - Features for prediction use bars through (T - 1)  (yesterday's close).
   - The model outputs a prediction PER stock (next-day close-to-close return).
   - We pick top-K above the cost-alpha threshold, throttle by n_drop, then
     submit Buy_Stock / Sell_Stock at limit prices rounded from (T - 1) close.
   - TWSE will fill iff today's [low, high] covers the limit (project Rule 02).

Cheat mode (--cheat):
   - Breaks the causal contract by fetching today's intraday OHLC via
     yfinance (~15 min delayed for free tier) and pricing orders against
     today's high (for sells) and today's low (for buys). Use for backtest
     sanity-checking ("what's the best execution we could get today?") —
     NOT for live trading. Stocks without today's yfinance bar are skipped.

Trading config is loaded from src/output/bt_config.json so it stays
identical to whatever the backtest used.

Credentials:
   Read from env vars TWSE_ACCOUNT and TWSE_PASSWORD. CLI args override.

Run (--cash is REQUIRED; copy the liquid cash figure from the course website):
   # Dry run (no orders submitted, just prints intended actions):
   python src/scripts/deploy_today.py --dry-run --cash 9500000

   # Live trading (will hit Buy_Stock / Sell_Stock):
   $env:TWSE_ACCOUNT="..."
   $env:TWSE_PASSWORD="..."
   python src/scripts/deploy_today.py --cash 9500000

Logs every intended/submitted order to src/output/run{N}/orders_YYYYMMDD.csv.
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

from src.data.io import fetch_range
from src.data.handler import (
    compute_alpha360, SEQ_LEN, N_FIELDS, PRED_HORIZON,
)
from src.data.processor import apply_robust_zscore, fillna
from src.model.double_adapt import DoubleAdapt
from src.evaluate.trading_rules import (
    SHARES_PER_LOT, round_buy, round_sell,
)
from stock_api import Buy_Stock, Sell_Stock, Get_User_Stocks

OUT_BASE = PROJECT_ROOT / "src" / "output"
LOGS_DIR = PROJECT_ROOT / "src" / "logs"
LOGS_DIR.mkdir(exist_ok=True)
HISTORY_START = "2024-01-01"   # ~2 years of bars is plenty for a 60-day window
TOP_N_LOG = 10                 # how many top predictions to log per inference


class _Tee:
    """Write stdout to BOTH terminal and a log file."""
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
    def flush(self):
        self.stream.flush()
        self.log_file.flush()


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
        raise SystemExit(f"missing {model_path}. Run src/scripts/train.py first.")
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
        feat = compute_alpha360(df, pred_h=1).tail(1)  # pred horizon irrelevant here
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
    """Convert Get_User_Stocks response into {code: shares_count}.

    Course API observed shape (one row per held stock):
      {'usid': ..., 'stock_name': ..., 'beginning_price': ...,
       'shares': 87,                  # <- LOTS, despite the name
       'stock_code_id': '2303',       # <- the canonical code key
       'createtime': ..., 'user_uid_id': ...}

    The `shares` field is in LOTS (1 lot = 1000 actual shares); the rest of
    deploy_today treats holdings[code] as a SHARE count and divides by
    SHARES_PER_LOT downstream, so we multiply on parse to keep one consistent
    internal unit.
    """
    holdings: dict[str, int] = {}
    if raw is None or raw == [] or raw == {}:
        return holdings

    def code_from(item: dict) -> str:
        return str(
            item.get("stock_code_id") or item.get("stock_code")
            or item.get("code") or ""
        ).strip()

    def lots_from(item: dict) -> int:
        v = (item.get("shares") or item.get("quantity")
             or item.get("stock_shares") or 0)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    if isinstance(raw, dict):
        for key in ("data", "stocks", "holdings", "user_stocks"):
            if key in raw and isinstance(raw[key], (list, dict)):
                return parse_holdings(raw[key])
        for code, val in raw.items():
            code = str(code).strip()
            if not re.fullmatch(r"\d{4,5}", code):
                continue
            if isinstance(val, dict):
                lots = lots_from(val)
            else:
                try:
                    lots = int(val)
                except (TypeError, ValueError):
                    continue
            if lots > 0:
                holdings[code] = lots * SHARES_PER_LOT
        return holdings

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                code = code_from(item)
                lots = lots_from(item)
                if code and lots > 0:
                    holdings[code] = lots * SHARES_PER_LOT
        return holdings

    if isinstance(raw, str):
        for m in re.finditer(r"\b(\d{4})\b\D{1,40}?(\d+)", raw):
            holdings.setdefault(m.group(1), int(m.group(2)) * SHARES_PER_LOT)
        return holdings

    return holdings


def parse_holdings_detail(raw) -> dict[str, dict]:
    """Like parse_holdings but keeps per-position cost basis info.

    Returns {code: {shares, lots, beginning_price, stock_name}} where shares
    is in actual share count (lots * SHARES_PER_LOT), beginning_price is the
    per-share entry price the broker reports.
    """
    detail: dict[str, dict] = {}
    if not isinstance(raw, list):
        return detail
    for item in raw:
        if not isinstance(item, dict):
            continue
        code = str(item.get("stock_code_id") or item.get("stock_code")
                   or item.get("code") or "").strip()
        try:
            lots = int(item.get("shares") or item.get("quantity")
                       or item.get("stock_shares") or 0)
        except (TypeError, ValueError):
            continue
        if not code or lots <= 0:
            continue
        try:
            beg_px = float(item.get("beginning_price") or item.get("avg_price") or 0)
        except (TypeError, ValueError):
            beg_px = 0.0
        detail[code] = {
            "shares": lots * SHARES_PER_LOT,
            "lots": lots,
            "beginning_price": beg_px,
            "stock_name": str(item.get("stock_name") or ""),
        }
    return detail


def fetch_today_intraday(codes: list[str]) -> dict[str, tuple[float, float]]:
    """Cheat-mode helper: pull today's intraday OHLC via yfinance (~15 min
    delayed for the free tier).

    Returns {code: (today_low, today_high)} for stocks that have a valid
    today bar. Stocks without today's data (delisted, post-IPO blackout,
    market not open yet, network error, etc.) are omitted — the caller
    should skip them rather than fall back to prev_close, so that --cheat
    mode never silently degrades to the regular causal flow.
    """
    import yfinance as yf
    today = datetime.now().date()
    tomorrow_str = (today.fromordinal(today.toordinal() + 1)).isoformat()
    tickers = " ".join(f"{c}.TW" for c in codes)
    df = yf.download(
        tickers, start=today.isoformat(), end=tomorrow_str,
        progress=False, auto_adjust=False, group_by="ticker", threads=True,
    )
    ranges: dict[str, tuple[float, float]] = {}
    if df is None or df.empty:
        return ranges
    multi = isinstance(df.columns, pd.MultiIndex)
    for code in codes:
        try:
            sub = df[f"{code}.TW"] if multi else df
            sub = sub.dropna(subset=["Low", "High"])
            if sub.empty:
                continue
            low = float(sub["Low"].iloc[-1])
            high = float(sub["High"].iloc[-1])
            if np.isfinite(low) and np.isfinite(high) and 0 < low <= high:
                ranges[code] = (low, high)
        except (KeyError, IndexError):
            continue
    return ranges


@dataclass
class IntendedOrder:
    side: str          # "BUY" or "SELL"
    code: str
    lots: int
    price: float       # int for the legacy prev-close path; tick-aligned float
                       # in cheat mode so sub-NT$50 stocks land inside today's
                       # [low, high] (tick=0.05 there, int rounding loses 95%
                       # of valid limits and pushes the order outside the
                       # fillable range).
    limit_source: str = "prev-close"   # "prev-close" | "cheat-high" | "cheat-low"


def plan_orders(
    preds_df: pd.DataFrame,
    holdings: dict[str, int],
    bt_cfg: dict,
    cash: float,
    today_ranges: dict[str, tuple[float, float]] | None = None,
) -> list[IntendedOrder]:
    """Plan orders for today.

    cash: liquid cash available for new buys (NT$). Pass the value shown on
    the course website. Slot per buy is cash * risk_degree / len(buys)
    (conservative: does NOT pre-add expected sell proceeds, unlike the
    qlib backtest which assumes today's sells fill first).

    today_ranges: optional `--cheat` payload from fetch_today_intraday().
    When provided:
      - SELL limit = round_buy(today_high)  (rounds DOWN to a valid tick,
        lands just below today's high, inside [low, high] so it fills).
      - BUY limit  = round_sell(today_low)  (rounds UP to a valid tick,
        lands just above today's low, inside [low, high] so it fills).
    Stocks with no today_ranges entry are skipped — cheat mode is
    all-or-nothing per stock; we never fall back to prev_close silently.
    """
    top_k = int(bt_cfg.get("top_k", 10))
    cost_alpha = float(bt_cfg.get("cost_alpha_bps", 70.0)) / 1e4
    n_drop = bt_cfg.get("n_drop", None)
    n_drop = int(n_drop) if n_drop is not None else None
    risk_degree = float(bt_cfg.get("risk_degree", 0.95))     # qlib default

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
    # qlib TopkDropoutStrategy buy formula: sells + (topk - holdings).
    # Empty/partial book -> auto-fill open slots; full book -> n_drop replacement.
    buy_budget = len(sell_candidates) + (top_k - len(held_set))
    buy_candidates = buy_candidates[: max(buy_budget, 0)]

    prev_close = dict(zip(preds_df["code"], preds_df["prev_close"]))
    orders: list[IntendedOrder] = []

    # qlib's slot formula: cash * risk_degree / len(buys).
    # Conservative: does NOT pre-add expected sell proceeds.
    n_buys = len(buy_candidates)
    slot_capital = (cash * risk_degree / n_buys) if n_buys > 0 else 0.0
    if n_buys > 0:
        print(f"      sizing: cash={cash:,.0f}  risk={risk_degree}  "
              f"len(buys)={n_buys}  -> slot={slot_capital:,.0f} NT$/name "
              f"(conservative: not assuming today's sells fill)")

    cheat = today_ranges is not None

    # SELL first
    for code in sell_candidates:
        lots = int(holdings[code] // SHARES_PER_LOT)
        if lots <= 0:
            continue
        if cheat:
            rng = today_ranges.get(code)
            if rng is None:
                print(f"      [cheat-skip SELL] {code}: no yfinance bar today")
                continue
            _, today_high = rng
            sell_price = round(float(round_buy(today_high)), 2)
            limit_source = "cheat-high"
        else:
            pc = prev_close.get(code)
            if pc is None or not np.isfinite(pc):
                continue
            sell_price = int(round(round_sell(pc)))
            limit_source = "prev-close"
        if sell_price <= 0:
            continue
        orders.append(IntendedOrder("SELL", code, lots, sell_price, limit_source))

    # BUY
    for code in buy_candidates:
        if cheat:
            rng = today_ranges.get(code)
            if rng is None:
                print(f"      [cheat-skip BUY] {code}: no yfinance bar today")
                continue
            today_low, _ = rng
            buy_price = round(float(round_sell(today_low)), 2)
            limit_source = "cheat-low"
        else:
            pc = prev_close.get(code)
            if pc is None or not np.isfinite(pc):
                continue
            buy_price = int(round(round_buy(pc)))
            limit_source = "prev-close"
        if buy_price <= 0:
            continue
        lots = int(slot_capital // (buy_price * SHARES_PER_LOT))
        if lots <= 0:
            continue
        orders.append(IntendedOrder("BUY", code, lots, buy_price, limit_source))
    return orders


# ----------------- inference logging -----------------

def log_top_inference(preds_df: pd.DataFrame, run_dir: Path,
                      cost_alpha_bps: float, top_n: int = TOP_N_LOG) -> Path:
    """Append a snapshot of the top-N predictions (rank ordered) to
    run_dir/inferences_YYYYMMDD.csv. One block per inference call so we
    can see how predictions evolve day-to-day for a given model.

    Columns: attempted_at, rank, code, pred, prev_close, above_threshold
    """
    log_path = run_dir / f"inferences_{datetime.now().strftime('%Y%m%d')}.csv"
    ranked = preds_df.sort_values("pred", ascending=False).head(top_n).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked.insert(0, "attempted_at", datetime.now().isoformat(timespec="seconds"))
    ranked["above_threshold"] = ranked["pred"] >= (cost_alpha_bps / 1e4)
    ranked["pred"] = ranked["pred"].round(5)
    ranked["prev_close"] = ranked["prev_close"].round(2)
    ranked["asof"] = ranked["asof"].apply(lambda d: pd.Timestamp(d).date())
    ranked = ranked[["attempted_at", "asof", "rank", "code", "pred",
                     "prev_close", "above_threshold"]]
    if log_path.exists():
        existing = pd.read_csv(log_path)
        ranked = pd.concat([existing, ranked], ignore_index=True)
    ranked.to_csv(log_path, index=False)
    return log_path


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
        print(f"  [{status}] {o.side:4s} {o.code} lots={o.lots:3d} "
              f"price={o.price:>7.2f}  [{o.limit_source}]")
        log_rows.append({
            "attempted_at": attempted_at,
            "side": o.side, "code": o.code,
            "lots": o.lots, "price": o.price,
            "limit_source": o.limit_source,
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
                    help="override n_drop for this run only "
                         "(use n_drop=top_k on day 1 to fill the book)")
    ap.add_argument("--cash", type=float, required=True,
                    help="REQUIRED. Liquid cash for new buys (NT$). The course "
                         "website shows this; pass it here. Slot per buy = "
                         "cash * risk_degree / len(buys).")
    ap.add_argument("--run", default=None,
                    help="which output/runN/ to load (default: read output/CURRENT)")
    ap.add_argument("--cheat", action="store_true",
                    help="BREAKS THE CAUSAL CONTRACT. Pull today's intraday "
                         "OHLC via yfinance (~15 min delayed) and place orders "
                         "at today's high (sells) / today's low (buys) for "
                         "best-execution sanity-checking. Stocks without "
                         "today's bar are skipped (no silent fallback).")
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.run)

    if not args.dry_run and (not args.account or not args.password):
        raise SystemExit("missing credentials. Set TWSE_ACCOUNT + TWSE_PASSWORD "
                         "env vars, or pass --account/--password, or use --dry-run.")

    # Tee stdout/stderr to a daily deploy log (one file per calendar day, appended).
    deploy_log_path = LOGS_DIR / f"deploy_{datetime.now().strftime('%Y%m%d')}.log"
    deploy_log_file = open(deploy_log_path, "a", encoding="utf-8", buffering=1)
    deploy_log_file.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} "
                          f"dry_run={args.dry_run} =====\n")
    sys.stdout = _Tee(sys.__stdout__, deploy_log_file)
    sys.stderr = _Tee(sys.__stderr__, deploy_log_file)

    t0 = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, dry_run={args.dry_run}")
    print(f"logging session to {deploy_log_path}")

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

    # Log top-N predictions for every inference (regardless of dry-run / live).
    inference_path = log_top_inference(
        preds, run_dir, float(bt_cfg.get("cost_alpha_bps", 70.0)), top_n=TOP_N_LOG
    )
    print(f"      logged top {TOP_N_LOG} predictions to {inference_path.name}")

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
        # Diagnostic: if API returned data but we parsed 0 positions, the response
        # shape may not match what parse_holdings expects. Show a sample (trimmed).
        if not holdings and raw not in (None, {}, []):
            sample = repr(raw)[:400]
            print(f"      WARNING: Get_User_Stocks returned non-empty {type(raw).__name__} "
                  f"but parsed 0 positions. Sample: {sample}")
    print(f"      current holdings: {len(holdings)} positions")

    today_ranges: dict[str, tuple[float, float]] | None = None
    if args.cheat:
        cheat_codes = sorted(set(holdings.keys()) | set(preds["code"].tolist()))
        print(f"\n      [cheat] yfinance batched download for {len(cheat_codes)} "
              f"tickers (~15 min delayed)...")
        today_ranges = fetch_today_intraday(cheat_codes)
        print(f"      [cheat] got today's [low, high] for "
              f"{len(today_ranges)}/{len(cheat_codes)} candidates")

    orders = plan_orders(preds, holdings, bt_cfg, cash=args.cash,
                         today_ranges=today_ranges)
    print(f"      planned: {sum(1 for o in orders if o.side=='SELL')} sells, "
          f"{sum(1 for o in orders if o.side=='BUY')} buys")

    print("\n[4/4] submitting orders...")
    log_rows = submit_orders(orders, args.account, args.password, args.dry_run)

    # Order log goes into run_dir/ so each model's full activity is co-located.
    orders_log_path = run_dir / f"orders_{datetime.now().strftime('%Y%m%d')}.csv"
    if log_rows:
        log_df = pd.DataFrame(log_rows)
        if orders_log_path.exists():
            log_df = pd.concat([pd.read_csv(orders_log_path), log_df], ignore_index=True)
        log_df.to_csv(orders_log_path, index=False)
        print(f"\nlogged {len(log_rows)} order(s) to {orders_log_path}")
    else:
        print("\nno orders to log.")

    print(f"\ntotal wall-time: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
