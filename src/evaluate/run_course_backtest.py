"""Run DoubleAdapt predictions through the course's official backtest framework.

Mirrors the structure of
[thirdparty/stock_project_for_class/backtest_package/stock_backtest_v1/examplebacktest.py]
but plugs in the DoubleAdapt model's per-day predictions for stock selection.

Pattern (matches the example):
  1. Define `select_stock(all_stock_info, prev_pool, cash) -> List[str]`
     Returns today's top-K codes by predicted alpha.
  2. Define `trade_strategy(stock_info_list, user_inventory, cash, tool)`
     Sell positions that dropped out of target; buy new entries.
     Trade price = today's `price_close` (the example's choice).
  3. Driver: `BacktestSystem(account, password)` -> set period + cash ->
     `execute_strategy(name, select, trade)` -> `calculate_performance` ->
     `save_performance_to_xls` -> optional `upload_*_to_web`.

Differences from examplebacktest.py:
  - Pre-builds the framework's stock cache from our data/interim/ files so it
    doesn't spend an hour downloading every TWSE code from the course API.
  - Monkey-patches Stock_API.get_all_stock_information to return only the
    codes that appear in our model's predictions.csv (~80 codes).
  - select_stock uses model preds; trade_strategy uses the size-by-cash slot
    formula (qlib's `value = cash * risk_degree / len(buys)`).

Usage:
    python src/evaluate/run_course_backtest.py
    python src/evaluate/run_course_backtest.py --upload
    python src/evaluate/run_course_backtest.py --run run5 --upload
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRAMEWORK_ROOT = PROJECT_ROOT / "thirdparty" / "stock_project_for_class" / "backtest_package" / "stock_backtest_v1"
sys.path.insert(0, str(FRAMEWORK_ROOT))

from src.config import CACHE_DIR, MODELS_DIR

OUT_BASE = MODELS_DIR
RISK_DEGREE = 0.95                              # qlib default: 95% deployed, 5% cash reserve


@contextmanager
def chdir(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ if not already set."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def build_framework_cache(codes: List[str], start_date: str, end_date: str) -> None:
    """Pre-populate the framework's ./tmp/stock_data.csv from our local cache
    so it skips the (very slow) per-stock API downloads.

    The framework's `Stock_Information.get_price()` only works when
    `df.loc[(code, date)]` returns a DataFrame (>=2 rows). Duplicate every
    row to satisfy this quirk.
    """
    parts = []
    years = list(range(int(start_date[:4]) - 1, int(end_date[:4]) + 1))
    for code in codes:
        for year in years:
            csv_path = CACHE_DIR / f"{code}_{year}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, usecols=["date", "open", "high", "low", "close"])
            df = df.dropna(subset=["close"])
            df["stock_code"] = str(code)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
            parts.append(df[["stock_code", "date", "open", "high", "low", "close"]])
    if not parts:
        raise SystemExit("no cache files found under data/interim/")
    full = pd.concat(parts, ignore_index=True)
    full = pd.concat([full, full], ignore_index=True)         # framework quirk: needs duplicates
    full = full.sort_values(["stock_code", "date"]).reset_index(drop=True)
    full.set_index(["stock_code", "date"], inplace=True)

    tmp_dir = FRAMEWORK_ROOT / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    full.to_csv(tmp_dir / "stock_data.csv")

    buffered_start = (pd.to_datetime(start_date) - pd.Timedelta(days=90)).strftime("%Y%m%d")
    with open(tmp_dir / "save_data_info.yaml", "w") as f:
        yaml.dump({"start_date": buffered_start, "end_date": end_date}, f)

    n_dates = full.index.get_level_values("date").nunique()
    print(f"  framework cache: {len(full)//2:,} unique rows x 2 (duplicated), "
          f"{len(codes)} codes x {n_dates} dates")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="which models/runN/ to use (default: models/CURRENT)")
    ap.add_argument("--start-date", default=None,
                    help="YYYYMMDD; default = earliest date in predictions.csv")
    ap.add_argument("--end-date", default=None,
                    help="YYYYMMDD; default = latest date in predictions.csv")
    ap.add_argument("--cash", type=float, default=100_000.0,
                    help="framework cash units (100k ~= 100M NTD at the framework's "
                         "implicit 1000x scaling). Default = 100M starting cash.")
    ap.add_argument("--strategy-name", default=None,
                    help="leaderboard label; default 'DoubleAdapt {run} (full IL)'")
    ap.add_argument("--upload", action="store_true",
                    help="upload results to course leaderboard after backtest")
    ap.add_argument("--account", default=os.environ.get("TWSE_ACCOUNT"))
    ap.add_argument("--password", default=os.environ.get("TWSE_PASSWORD"))
    args = ap.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    if args.account is None:
        args.account = os.environ.get("TWSE_ACCOUNT")
    if args.password is None:
        args.password = os.environ.get("TWSE_PASSWORD")

    if args.run:
        run_dir = OUT_BASE / args.run
    else:
        cur_file = OUT_BASE / "CURRENT"
        if not cur_file.exists():
            raise SystemExit(f"missing {cur_file}; pass --run runX")
        run_dir = OUT_BASE / cur_file.read_text().strip()
    print(f"using {run_dir.name}/")

    if args.upload and (not args.account or not args.password):
        raise SystemExit("--upload requires TWSE_ACCOUNT + TWSE_PASSWORD "
                         "(env vars, --account/--password, or .env file)")

    preds = pd.read_csv(run_dir / "predictions.csv")
    preds["date"] = pd.to_datetime(preds["date"])
    preds["stock_code_id"] = preds["stock_code_id"].astype(str)
    bt_cfg = json.loads((run_dir / "bt_config.json").read_text())
    top_k = int(bt_cfg["top_k"])
    threshold = float(bt_cfg.get("cost_alpha_bps", 0.0)) / 1e4
    print(f"  bt_config: top_k={top_k}, cost_alpha={threshold:.4f}")

    start_date = args.start_date or preds["date"].min().strftime("%Y%m%d")
    end_date = args.end_date or preds["date"].max().strftime("%Y%m%d")
    print(f"  IL window: {start_date} -> {end_date} ({preds['date'].nunique()} dates)")

    codes = sorted(preds["stock_code_id"].unique().tolist())
    # The course framework uses 2330 (TSMC) as its trading-day reference,
    # so it must be in the framework's bar cache + universe even if the model
    # never predicts on it (e.g., TM100 backtests where 2330 is in TW50, not TM100).
    # Predictions themselves still come from pred_lookup, so adding 2330 here
    # only affects the framework's calendar lookup — not stock selection.
    if "2330" not in codes:
        codes = sorted(set(codes) | {"2330"})
        print(f"  + injected 2330 for framework day-reference")
    print(f"  universe (from predictions + framework refs): {len(codes)} codes")

    pred_lookup: dict[str, dict[str, float]] = {}
    for date, grp in preds.groupby("date"):
        pred_lookup[date.strftime("%Y%m%d")] = dict(
            zip(grp["stock_code_id"].tolist(), grp["pred"].astype(float).tolist())
        )

    print("building framework cache from data/interim/ ...")
    build_framework_cache(codes, start_date, end_date)

    strategy_name = args.strategy_name or f"DoubleAdapt {run_dir.name} (full IL)"
    print(f"  strategy name: {strategy_name!r}")

    # ---------- strategy callbacks (mirror examplebacktest.py shape) ----------

    def select_stock(all_stock_info, previous_stock_pool, user_cash):
        """Today's top-K codes by DoubleAdapt pred. Returns List[str] of codes."""
        if not all_stock_info:
            return []
        today_str = all_stock_info[0].init_time.strftime("%Y%m%d")
        today_preds = pred_lookup.get(today_str, {})
        if not today_preds:
            return []
        ranked = sorted(
            ((c, p) for c, p in today_preds.items() if p >= threshold),
            key=lambda x: -x[1],
        )[:top_k]
        return [c for c, _ in ranked]

    def trade_strategy(stock_code_list, user_inventory, user_cash, transaction_tool):
        """Sell positions no longer in target; buy new entries. Trade at today's
        close (matches examplebacktest.py's use of `stock.price_close`)."""
        held_codes = {inv.stock_info.stock_code for inv in user_inventory}
        target_set = {s.stock_code for s in stock_code_list}

        # SELL: held but not in today's target
        for inv in user_inventory:
            code = inv.stock_info.stock_code
            if code in target_set:
                continue
            px = inv.stock_info.price_close
            if px is None:
                continue
            transaction_tool.sell_stock(code, int(round(px)), int(inv.shares))

        # BUY: new entries not yet held. Slot = cash * 0.95 / num_new_buys
        # (qlib's value-per-buy formula).
        new_buys = [s for s in stock_code_list if s.stock_code not in held_codes]
        n_buys = len(new_buys)
        if n_buys == 0:
            return transaction_tool.transaction_record
        slot = user_cash * RISK_DEGREE / n_buys
        for stock in new_buys:
            px = stock.price_close
            if px is None or px <= 0:
                continue
            lots = int(slot // px)
            if lots > 0:
                transaction_tool.buy_stock(stock.stock_code, int(round(px)), lots)

        return transaction_tool.transaction_record

    # ---------- driver (mirrors examplebacktest.py's __main__) ----------

    # Skip the framework's all-TWSE download; only fetch codes in our universe
    from backtest.Stock_API import Stock_API as _SA  # noqa: E402
    _SA.get_all_stock_information = staticmethod(lambda: codes)

    from backtest.backtest import BacktestSystem  # noqa: E402

    print(f"\nrunning backtest...")
    with chdir(FRAMEWORK_ROOT):
        backtest = BacktestSystem(args.account or "dummy", args.password or "dummy")
        backtest.set_backtest_period(start_date, end_date)
        backtest.set_cash_balance(args.cash)
        backtest.execute_strategy(strategy_name, select_stock, trade_strategy)

        performance, performance_detail = backtest.calculate_performance()
        if performance is None:
            raise SystemExit("calculate_performance returned None (no trades?)")

        # Framework's upload reads "backtest_performance.xlsx" from CWD (hardcoded).
        backtest.save_performance_to_xls(
            strategy_name, performance, performance_detail, "backtest_performance.xlsx"
        )
        run_xls = run_dir / "course_backtest.xlsx"
        shutil.copy("backtest_performance.xlsx", run_xls)

    total_return_pct = (performance.total_assets - args.cash) / args.cash * 100
    print(f"\n  saved {run_xls}")
    print(f"  realized_profit:   {performance.realized_profit:>15,.2f}  (framework units; *1000 ~= NTD)")
    print(f"  unrealized_profit: {performance.unrealized_profit:>15,.2f}")
    print(f"  final cash:        {performance.cash_assets:>15,.2f}")
    print(f"  total_assets:      {performance.total_assets:>15,.2f}   (initial {args.cash:,.0f})")
    print(f"  total return:      {total_return_pct:>14.2f}%")

    if args.upload:
        print(f"\nuploading to leaderboard at 140.116.86.242:8081 ...")
        with chdir(FRAMEWORK_ROOT):
            backtest.upload_performance_to_web()
            backtest.upload_performance_detail_to_web()
            backtest.upload_performance_record_to_web()
        print("  upload done")


if __name__ == "__main__":
    main()
