"""Daily auto-trader for the TWSE virtual trading system.

Schedule (typical):
   T-1 15:30-16:00  TWSE settlement window (course system updates bars)
   weekly/monthly   drop fresh TEJ half-year CSV into data/raw/TEJ/test/
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

Trading config is loaded from the run's bt_config.json so it stays
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

Logs every intended/submitted order to orders_YYYYMMDD.csv in the run's MLflow artifacts.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.inference import predict_for_yesterday
from src.data.handler import SEQ_LEN, N_FIELDS
from src.model.double_adapt import DoubleAdapt
from src.trading.strategy import buy_slot, topk_dropout
from src.trading.rules import (
    SHARES_PER_LOT,
    round_buy,
    round_sell,
)
from src.trading.broker import (
    IntendedOrder,
    fetch_today_intraday,
    get_raw_holdings,
    parse_holdings,
    submit_orders,
)
from src.config import LOGS_DIR
from src import runs

TOP_N_LOG = 10  # how many top predictions to log per inference


# ----------------- loaders -----------------


def load_artifacts(device: torch.device, run_dir: Path) -> tuple:
    model_path = run_dir / "final.pt"
    if not model_path.exists():
        raise SystemExit(f"missing {model_path}. Run src/scripts/train.py first.")
    model = DoubleAdapt(seq_len=SEQ_LEN, n_fields=N_FIELDS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    norm_stats = runs.load_norm_stats(run_dir)
    universe = runs.load_active_universe(run_dir)
    bt_cfg = runs.load_bt_config(run_dir)
    feature_mode = runs.load_feature_mode(run_dir)  # legacy runs 1-16 -> per_field
    print(f"      feature scheme: {feature_mode}")

    return model, norm_stats, universe, bt_cfg, feature_mode


# Inference on the newest bars (predict_for_yesterday): src/inference.py


# Broker I/O (holdings, intraday quote, order submission): src/trading/broker.py


def plan_orders(
    preds_df: pd.DataFrame,
    holdings: dict[str, int],
    bt_cfg: dict,
    cash: float,
    today_ranges: dict[str, tuple[float, float]] | None = None,
) -> list[IntendedOrder]:
    """Plan orders for today.

    cash: liquid cash currently available (NT$). Pass the value shown on
    the course website. Today's planned SELLs add their expected gross
    proceeds to this number BEFORE sizing the buy slot — same convention
    as qlib's TopkDropoutStrategy (assumes today's sells fill first, so
    that cash is recyclable into the same day's buys).

    Effective slot = (cash + Σ sell_proceeds) * risk_degree / len(buys).

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
    risk_degree = float(bt_cfg.get("risk_degree", 0.95))  # qlib default

    target = preds_df[preds_df["pred"] >= cost_alpha].sort_values("pred", ascending=False).head(top_k)["code"].tolist()
    # Same topk-dropout selection as the backtest (src/trading/strategy.py).
    pred_lookup = dict(zip(preds_df["code"], preds_df["pred"]))
    sell_candidates, buy_candidates = topk_dropout(target, holdings.keys(), pred_lookup, top_k, n_drop)

    prev_close = dict(zip(preds_df["code"], preds_df["prev_close"]))
    orders: list[IntendedOrder] = []

    # ---- SELL first; accumulate expected proceeds so they're recyclable
    # into today's buy slots (mirrors qlib's TopkDropoutStrategy assumption
    # that sells fill before buys settle).
    sell_proceeds = 0.0
    for code in sell_candidates:
        lots = int(holdings[code] // SHARES_PER_LOT)
        if lots <= 0:
            continue
        if today_ranges is not None:
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
        sell_proceeds += sell_price * lots * SHARES_PER_LOT  # GROSS, no cost subtracted

    # ---- Compute buy slot AFTER sells are known
    n_buys = len(buy_candidates)
    effective_cash = cash + sell_proceeds
    slot_capital = buy_slot(effective_cash, risk_degree, n_buys)
    if n_buys > 0:
        print(
            f"      sizing: cash={cash:,.0f}  + sell_proceeds={sell_proceeds:,.0f}"
            f"  = effective={effective_cash:,.0f}  risk={risk_degree}  "
            f"len(buys)={n_buys}  -> slot={slot_capital:,.0f} NT$/name"
        )

    # ---- BUY using slot informed by post-sell cash
    for code in buy_candidates:
        if today_ranges is not None:
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


def log_top_inference(preds_df: pd.DataFrame, run_dir: Path, cost_alpha_bps: float, top_n: int = TOP_N_LOG) -> Path:
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
    ranked = ranked[["attempted_at", "asof", "rank", "code", "pred", "prev_close", "above_threshold"]]
    if log_path.exists():
        existing = pd.read_csv(log_path)
        ranked = pd.concat([existing, ranked], ignore_index=True)
    ranked.to_csv(log_path, index=False)
    return log_path


# ----------------- main -----------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily auto-trader for the course virtual TWSE")
    ap.add_argument("--dry-run", action="store_true", help="print intended orders without calling Buy_Stock/Sell_Stock")
    ap.add_argument("--account", default=os.environ.get("TWSE_ACCOUNT"), help="course account (default: $TWSE_ACCOUNT)")
    ap.add_argument(
        "--password", default=os.environ.get("TWSE_PASSWORD"), help="course password (default: $TWSE_PASSWORD)"
    )
    ap.add_argument(
        "--n-drop",
        type=int,
        default=None,
        help="override n_drop for this run only " "(use n_drop=top_k on day 1 to fill the book)",
    )
    ap.add_argument(
        "--cash",
        type=float,
        required=True,
        help="REQUIRED. Liquid cash for new buys (NT$). The course "
        "website shows this; pass it here. Slot per buy = "
        "cash * risk_degree / len(buys).",
    )
    ap.add_argument("--run", default=None, help="MLflow run to load, e.g. run9 (default: the run tagged current)")
    ap.add_argument(
        "--cheat",
        action="store_true",
        help="BREAKS THE CAUSAL CONTRACT. Pull today's intraday "
        "OHLC via yfinance (~15 min delayed) and place orders "
        "at today's high (sells) / today's low (buys) for "
        "best-execution sanity-checking. Stocks without "
        "today's bar are skipped (no silent fallback).",
    )
    args = ap.parse_args()
    run_dir = runs.resolve_run(args.run)

    if not args.dry_run and (not args.account or not args.password):
        raise SystemExit(
            "missing credentials. Set TWSE_ACCOUNT + TWSE_PASSWORD "
            "env vars, or pass --account/--password, or use --dry-run."
        )

    # Tee stdout/stderr to a daily deploy log (one file per calendar day, appended).
    deploy_log_path = LOGS_DIR / f"deploy_{datetime.now().strftime('%Y%m%d')}.log"
    runs.tee_session(
        deploy_log_path,
        mode="a",
        header=f"\n===== {datetime.now().isoformat(timespec='seconds')} dry_run={args.dry_run} =====\n",
    )

    t0 = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, dry_run={args.dry_run}")
    print(f"logging session to {deploy_log_path}")

    print(f"\n[1/4] loading artifacts from {run_dir.name}/ (model, norm stats, universe, bt_config)...")
    model, norm_stats, universe, bt_cfg, feature_mode = load_artifacts(device, run_dir)
    if args.n_drop is not None:
        print(f"      OVERRIDE: n_drop {bt_cfg.get('n_drop')} -> {args.n_drop} (CLI flag)")
        bt_cfg["n_drop"] = args.n_drop
    print(f"      universe size: {len(universe)}, bt_cfg: {bt_cfg}")

    print("\n[2/4] running inference on yesterday's features...")
    preds = predict_for_yesterday(model, norm_stats, universe, device, feature_mode=feature_mode)
    asof = preds["asof"].iloc[0]
    print(
        f"      asof (= 'yesterday' = lag-0): {pd.Timestamp(asof).date()}, " f"got predictions for {len(preds)} stocks"
    )
    print(f"      top 5: {preds.head(5)[['code', 'pred', 'prev_close']].to_string(index=False)}")

    # Log top-N predictions for every inference (regardless of dry-run / live).
    inference_path = log_top_inference(preds, run_dir, float(bt_cfg.get("cost_alpha_bps", 70.0)), top_n=TOP_N_LOG)
    print(f"      logged top {TOP_N_LOG} predictions to {inference_path.name}")

    print("\n[3/4] reading current holdings + planning orders...")
    if args.dry_run and not args.account:
        print("      (dry-run + no credentials -> assuming empty book)")
        holdings = {}
    else:
        try:
            raw = get_raw_holdings(args.account, args.password)
        except Exception as e:
            print(f"      Get_User_Stocks failed: {e!r}  (assuming empty book)")
            raw = None
        holdings = parse_holdings(raw)
        # Diagnostic: if API returned data but we parsed 0 positions, the response
        # shape may not match what parse_holdings expects. Show a sample (trimmed).
        if not holdings and raw not in (None, {}, []):
            sample = repr(raw)[:400]
            print(
                f"      WARNING: Get_User_Stocks returned non-empty {type(raw).__name__} "
                f"but parsed 0 positions. Sample: {sample}"
            )
    print(f"      current holdings: {len(holdings)} positions")

    today_ranges: dict[str, tuple[float, float]] | None = None
    if args.cheat:
        cheat_codes = sorted(set(holdings.keys()) | set(preds["code"].tolist()))
        print(f"\n      [cheat] yfinance batched download for {len(cheat_codes)} " f"tickers (~15 min delayed)...")
        today_ranges = fetch_today_intraday(cheat_codes)
        print(f"      [cheat] got today's [low, high] for " f"{len(today_ranges)}/{len(cheat_codes)} candidates")

    orders = plan_orders(preds, holdings, bt_cfg, cash=args.cash, today_ranges=today_ranges)
    print(
        f"      planned: {sum(1 for o in orders if o.side == 'SELL')} sells, "
        f"{sum(1 for o in orders if o.side == 'BUY')} buys"
    )

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
