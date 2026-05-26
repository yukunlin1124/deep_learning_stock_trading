"""qlib-style backtest: TopkDropoutStrategy + close-price execution.

Mirrors qlib's default benchmark setup
(qlib/examples/benchmarks/GRU/workflow_config_gru_Alpha360.yaml):

  strategy:
      class: TopkDropoutStrategy
      kwargs: { topk: 50, n_drop: 5 }
  backtest:
      exchange_kwargs:
          deal_price: close          # every order fills at today's close
          open_cost: 0.0005          # 5 bps buy commission
          close_cost: 0.0015         # 15 bps sell commission + tax (bundled)
          min_cost: 5                # 5 CNY min per trade
          limit_threshold: 0.095     # block at limit-up/down

Key differences from src/evaluate/backtest.py (our realistic backtest):
  - **Always fills**: no `low <= limit <= high` check; every order executes
    at today's close
  - **No tick rounding**: prices are continuous (no TWSE-band snapping)
  - **No lot rounding**: shares are continuous (no 1000-share lot constraint)
  - **Look-ahead bias**: uses today's close to decide today's order (you
    can't actually do this in live trading — order has to be placed before
    market close)
  - **No NT$20 min commission**: uses qlib's `min_cost` floor instead

Use the `costs` flag to switch between qlib's CN-A-share defaults and
TWSE-correct rates so we can compare apples-to-apples on the SAME pricing
model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class QlibBTConfig:
    capital: float = 1e8                 # starting cash: 100M NT$
    top_k: int = 50                      # qlib default; for our 50-stock universe, set to 10 for fair comparison
    n_drop: int = 5                      # qlib default
    cost_alpha_bps: float = 0.0          # qlib default: no threshold (every top-K name is targeted)
    costs: Literal["qlib", "twse"] = "qlib"
    limit_threshold: float = 0.095       # block trades at +/- 9.5% limit-up/down (TWSE limit is 10%)
    deal_price: Literal["close", "open"] = "close"
    risk_degree: float = 0.95            # qlib default: 95% deployed, 5% cash reserve
    # "close" = qlib default (today's close; look-ahead, unimplementable live)
    # "open"  = today's open (achievable live via pre-9:00 order targeting the opening auction)


_COST_TABLES = {
    "qlib": dict(open_cost=0.0005, close_cost=0.0015, min_cost=5.0),
    "twse": dict(open_cost=0.001425,                           # 0.1425% commission
                 close_cost=0.001425 + 0.003,                  # 0.1425% + 0.3% sell tax
                 min_cost=20.0),
}


@dataclass
class BTResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    daily: pd.DataFrame
    summary: dict


def _equity_summary(eq: pd.DataFrame, capital: float, trades_df: pd.DataFrame
                    ) -> dict:
    if eq.empty:
        return {"days": 0, "final_equity": capital, "cum_return": 0.0,
                "ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "n_trades": 0, "n_round_trips": 0,
                "win_rate": 0.0, "profit_factor": 0.0,
                "avg_profit_return": 0.0, "avg_loss_return": 0.0,
                "fill_rate": 1.0}
    e = eq["equity"].values
    days = len(e)
    cum_ret = (e[-1] - capital) / capital
    daily_ret = pd.Series(e).pct_change().dropna()
    ann_vol = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) else 0.0
    sharpe = (float(daily_ret.mean()) * 252) / (ann_vol + 1e-9) if ann_vol > 0 else 0.0
    peak = pd.Series(e).cummax()
    max_dd = float(((pd.Series(e) - peak) / peak).min())

    n_round_trips = 0
    win_rate = 0.0
    profit_factor = 0.0
    avg_profit = 0.0
    avg_loss = 0.0
    if not trades_df.empty and "trade_return" in trades_df.columns:
        closed = trades_df.loc[
            (trades_df["side"] == "SELL") & trades_df["trade_return"].notna()
        ]
        if not closed.empty:
            r = closed["trade_return"].astype(float)
            wins = r[r > 0]
            losses = r[r < 0]
            n_round_trips = int(len(r))
            win_rate = float(len(wins) / len(r))
            gains, losses_abs = float(wins.sum()), float(-losses.sum())
            profit_factor = float(gains / losses_abs) if losses_abs > 0 else float("inf")
            avg_profit = float(wins.mean()) if len(wins) else 0.0
            avg_loss = float(losses.mean()) if len(losses) else 0.0

    return {
        "days": int(days),
        "final_equity": float(e[-1]),
        "cum_return": float(cum_ret),
        "ann_return": float(cum_ret / days * 252 if days else 0.0),
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "n_trades": int(len(trades_df)) if not trades_df.empty else 0,
        "n_round_trips": n_round_trips,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_profit_return": avg_profit,
        "avg_loss_return": avg_loss,
        "fill_rate": 1.0,                    # qlib model: always fills
    }


def _is_limit_locked(today_high: float, today_low: float, prev_close: float,
                     threshold: float) -> bool:
    """Approximate limit-up/down detection: if the day's range is at the
    +/-threshold edge of prev_close. We block trades on those days."""
    if not np.isfinite(prev_close) or prev_close <= 0:
        return False
    limit_up = prev_close * (1 + threshold)
    limit_down = prev_close * (1 - threshold)
    # if both high and low are at one limit, it's locked
    return (today_low >= limit_up * 0.999) or (today_high <= limit_down * 1.001)


def run_qlib_backtest_from_predictions(
    preds_df: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    cfg: QlibBTConfig | None = None,
) -> BTResult:
    """qlib-style: top-K dropout strategy, deal_price=close, always fills."""
    cfg = cfg or QlibBTConfig()
    costs = _COST_TABLES[cfg.costs]
    open_cost, close_cost, min_cost = costs["open_cost"], costs["close_cost"], costs["min_cost"]

    empty_trades = pd.DataFrame(columns=[
        "date", "code", "side", "price", "shares", "cashflow", "trade_return"
    ])
    if preds_df.empty:
        return BTResult(
            equity=pd.DataFrame(columns=["date", "equity"]),
            trades=empty_trades,
            daily=pd.DataFrame(),
            summary=_equity_summary(pd.DataFrame(columns=["equity"]), cfg.capital,
                                    empty_trades),
        )

    preds = preds_df.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    preds["stock_code_id"] = preds["stock_code_id"].astype(str)

    closes = {c: bars[c].set_index("date")["close"]
              for c in bars if "date" in bars[c].columns}
    opens = {c: bars[c].set_index("date")["open"] for c in closes}
    highs = {c: bars[c].set_index("date")["high"] for c in closes}
    lows = {c: bars[c].set_index("date")["low"] for c in closes}
    prev_close = {c: closes[c].shift(1) for c in closes}
    # Pick the deal-price series based on cfg.deal_price (close or open).
    deal_series = closes if cfg.deal_price == "close" else opens

    test_dates = sorted(preds["date"].unique())
    cash = cfg.capital
    positions: dict[str, float] = {}                   # code -> shares (float, no lot rounding)
    entry: dict[str, dict] = {}                        # code -> {price, cost}

    trades: list[dict] = []
    equity_rows: list[tuple] = []
    daily_rows: list[dict] = []
    thresh = cfg.cost_alpha_bps / 1e4

    for today in test_dates:
        today_preds = preds[preds["date"] == today]
        pred_lookup = dict(zip(today_preds["stock_code_id"].tolist(),
                               today_preds["pred"].astype(float).tolist()))

        target = (
            today_preds[today_preds["pred"] >= thresh]
            .sort_values("pred", ascending=False)
            .head(cfg.top_k)["stock_code_id"].tolist()
        )
        target_set = set(target)
        held_set = set(positions)

        sell_candidates = sorted(held_set - target_set,
                                 key=lambda c: pred_lookup.get(c, -np.inf))
        buy_candidates = [c for c in target if c not in held_set]
        if cfg.n_drop is not None:
            sell_candidates = sell_candidates[: cfg.n_drop]
        # qlib TopkDropoutStrategy: buy = sells + (topk - holdings).
        # Auto-fills partial book; falls back to n_drop replacement when full.
        buy_budget = len(sell_candidates) + (cfg.top_k - len(held_set))
        buy_candidates = buy_candidates[: max(buy_budget, 0)]

        # --- SELLS at today's deal_price (no fill check, no tick/lot rounding) ---
        for code in sell_candidates:
            if code not in closes:
                continue
            px = deal_series[code].get(today, np.nan)
            hi = highs[code].get(today, np.nan)
            lo = lows[code].get(today, np.nan)
            pc = prev_close[code].get(today, np.nan)
            if not np.isfinite(px):
                continue
            if _is_limit_locked(hi, lo, pc, cfg.limit_threshold):
                continue                                            # halted
            shares = positions[code]
            if shares <= 0:
                continue
            notional = shares * float(px)
            cost = max(notional * close_cost, min_cost)
            proceeds = notional - cost
            cash += proceeds

            buy_price = entry.get(code, {}).get("price", float(px))
            trade_return = ((float(px) - buy_price) / buy_price) if buy_price > 0 else 0.0
            trades.append({
                "date": today, "code": code, "side": "SELL",
                "price": float(px), "shares": shares, "cashflow": proceeds,
                "trade_return": trade_return,
            })
            positions.pop(code, None)
            entry.pop(code, None)

        # --- BUYS at today's deal_price ---
        # qlib's exact formula: slot per buy = current cash * risk_degree / len(buys).
        # Sells above have already added their proceeds to cash, so this reinvests them.
        n_buys = len(buy_candidates)
        slot_capital = (cash * cfg.risk_degree / n_buys) if n_buys > 0 else 0.0
        for code in buy_candidates:
            if code not in closes:
                continue
            px = deal_series[code].get(today, np.nan)
            hi = highs[code].get(today, np.nan)
            lo = lows[code].get(today, np.nan)
            pc = prev_close[code].get(today, np.nan)
            if not np.isfinite(px) or px <= 0:
                continue
            if _is_limit_locked(hi, lo, pc, cfg.limit_threshold):
                continue
            shares = slot_capital / float(px)                       # fractional shares (qlib does this)
            notional = shares * float(px)
            cost = max(notional * open_cost, min_cost)
            total = notional + cost
            if total > cash:
                continue
            cash -= total
            positions[code] = shares
            entry[code] = {"price": float(px), "cost": total}
            trades.append({
                "date": today, "code": code, "side": "BUY",
                "price": float(px), "shares": shares, "cashflow": total,
                "trade_return": np.nan,
            })

        # --- Mark-to-close ---
        mark = 0.0
        for code, sh in positions.items():
            if code in closes:
                cl = closes[code].get(today, np.nan)
                if np.isfinite(cl):
                    mark += sh * float(cl)
        equity_rows.append((today, cash + mark))
        daily_rows.append({
            "date": today, "cash": cash, "mark_value": mark,
            "n_positions": len(positions), "n_targets": len(target),
        })

    eq = pd.DataFrame(equity_rows, columns=["date", "equity"])
    trades_df = pd.DataFrame(trades) if trades else empty_trades.copy()
    summary = _equity_summary(eq, cfg.capital, trades_df)
    return BTResult(equity=eq, trades=trades_df, daily=pd.DataFrame(daily_rows),
                    summary=summary)


# ---------- Live-IL backtest: model fine-tunes every r days during the run ----------

def run_qlib_backtest_with_il(
    model,                              # DoubleAdapt; mutated in-place by FOMAML steps
    il_panel,                           # PanelTensors covering [start_date - 90d, end_date]
    bars: dict[str, pd.DataFrame],
    fcfg,                               # FOMAMLConfig (inner_lr, lr_phi, lr_psi, r, etc.)
    cfg: QlibBTConfig | None = None,
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    device=None,
    log_fn=print,
) -> tuple[BTResult, pd.DataFrame, pd.DataFrame]:
    """Walk il_panel forward in r-day windows. Each window:
      1. Build (support, query) task; do one FOMAML bi-level update on support;
         get predictions for the query days.
      2. For each day in the query window: take top-K from those predictions,
         simulate sells/buys/mark-to-close (same rules as
         `run_qlib_backtest_from_predictions`).

    This is the "live IL during backtest" mode — the model state at trade time
    on day D reflects everything it has adapted through D - 20. Reproduces
    deploy-style behavior more faithfully than the replay-from-predictions path.

    Returns:
      (BTResult, step_log_df, predictions_df)
        step_log_df: per-IL-step IC/RankIC/loss (same shape as run_incremental)
        predictions_df: per-(date, stock) pred/actual (same shape as predictions.csv)
    """
    import torch
    from src.trainer.maml import panel_to_task, fomaml_step
    from src.evaluate.metrics import cross_sectional_ic

    cfg = cfg or QlibBTConfig()
    costs = _COST_TABLES[cfg.costs]
    open_cost, close_cost, min_cost = costs["open_cost"], costs["close_cost"], costs["min_cost"]
    thresh = cfg.cost_alpha_bps / 1e4

    # Optimizers — adapt model state in-place during the backtest
    optim_phi = torch.optim.Adam(model.ma_parameters(), lr=fcfg.lr_phi,
                                 weight_decay=fcfg.weight_decay)
    optim_psi = torch.optim.Adam(model.da_parameters(), lr=fcfg.lr_psi,
                                 weight_decay=fcfg.weight_decay)

    # Price series for portfolio sim
    closes = {c: bars[c].set_index("date")["close"]
              for c in bars if "date" in bars[c].columns}
    opens = {c: bars[c].set_index("date")["open"] for c in closes}
    highs = {c: bars[c].set_index("date")["high"] for c in closes}
    lows = {c: bars[c].set_index("date")["low"] for c in closes}
    prev_close = {c: closes[c].shift(1) for c in closes}
    deal_series = closes if cfg.deal_price == "close" else opens

    # Portfolio state
    cash = cfg.capital
    positions: dict[str, float] = {}
    entry: dict[str, dict] = {}
    trades: list[dict] = []
    equity_rows: list[tuple] = []
    daily_rows: list[dict] = []

    # IL walk
    udates = np.unique(il_panel.dates)
    start = np.datetime64(pd.Timestamp(start_date))
    end = (np.datetime64(pd.Timestamp(end_date))
           if end_date else udates[-1])
    qry_first_idx = max(int(np.searchsorted(udates, start, side="left")), fcfg.r)

    step_log_rows = []
    pred_rows = []
    i = qry_first_idx
    step = 0
    while i + fcfg.r <= len(udates):
        if udates[i] > end:
            break
        task = panel_to_task(il_panel, i - fcfg.r, fcfg.r, device)
        if task is None:
            i += fcfg.r; step += 1; continue

        # ---- (1) IL step: adapt + predict ----
        optim_phi.zero_grad(set_to_none=True)
        optim_psi.zero_grad(set_to_none=True)
        test_loss, pred_qry, y_qry = fomaml_step(model, task, fcfg, train=True)
        optim_phi.step(); optim_psi.step()

        preds_np = pred_qry.cpu().detach().numpy()
        ys_np = y_qry.cpu().detach().numpy()
        d_np = task.qry_dates
        c_np = task.qry_codes
        ic, icir, ric, ricir = cross_sectional_ic(preds_np, ys_np, d_np)

        step_log_rows.append({
            "step": step,
            "support_start": pd.Timestamp(udates[i - fcfg.r]).date(),
            "support_end": pd.Timestamp(udates[i - 1]).date(),
            "query_start": pd.Timestamp(udates[i]).date(),
            "query_end": pd.Timestamp(udates[min(i + fcfg.r - 1, len(udates) - 1)]).date(),
            "n_support": int(task.x_sup.shape[0]),
            "n_query": int(task.x_qry.shape[0]),
            "test_loss": float(test_loss),
            "ic": float(ic), "icir": float(icir),
            "ric": float(ric), "ricir": float(ricir),
        })

        # ---- (2) Build per-(date, code) pred lookup for the query window ----
        # task.qry_dates is parallel with preds_np
        day_preds: dict[pd.Timestamp, dict[str, float]] = {}
        for d, c, p, a in zip(d_np, c_np, preds_np, ys_np):
            d_ts = pd.Timestamp(d)
            day_preds.setdefault(d_ts, {})[str(c)] = float(p)
            pred_rows.append({
                "date": d_ts.date(),
                "stock_code_id": str(c),
                "pred": float(p), "actual": float(a),
            })

        # ---- (3) Simulate portfolio for each day in the query window ----
        for day_idx in range(fcfg.r):
            di = i + day_idx
            if di >= len(udates):
                break
            today = pd.Timestamp(udates[di])
            if today > pd.Timestamp(end):
                break

            today_preds = day_preds.get(today, {})
            target = sorted(
                ((c, p) for c, p in today_preds.items() if p >= thresh),
                key=lambda x: -x[1],
            )[:cfg.top_k]
            target_codes = [c for c, _ in target]
            target_set = set(target_codes)
            held_set = set(positions)

            sell_candidates = sorted(
                held_set - target_set,
                key=lambda c: today_preds.get(c, -np.inf),
            )
            buy_candidates = [c for c in target_codes if c not in held_set]
            if cfg.n_drop is not None:
                sell_candidates = sell_candidates[: cfg.n_drop]
            buy_budget = len(sell_candidates) + (cfg.top_k - len(held_set))
            buy_candidates = buy_candidates[: max(buy_budget, 0)]

            # SELLS
            for code in sell_candidates:
                if code not in closes:
                    continue
                px = deal_series[code].get(today, np.nan)
                hi = highs[code].get(today, np.nan)
                lo = lows[code].get(today, np.nan)
                pc = prev_close[code].get(today, np.nan)
                if not np.isfinite(px):
                    continue
                if _is_limit_locked(hi, lo, pc, cfg.limit_threshold):
                    continue
                shares = positions[code]
                if shares <= 0:
                    continue
                notional = shares * float(px)
                cost = max(notional * close_cost, min_cost)
                proceeds = notional - cost
                cash += proceeds
                buy_price = entry.get(code, {}).get("price", float(px))
                trade_return = ((float(px) - buy_price) / buy_price) if buy_price > 0 else 0.0
                trades.append({
                    "date": today, "code": code, "side": "SELL",
                    "price": float(px), "shares": shares, "cashflow": proceeds,
                    "trade_return": trade_return,
                })
                positions.pop(code, None)
                entry.pop(code, None)

            # BUYS — qlib slot formula
            n_buys = len(buy_candidates)
            slot_capital = (cash * cfg.risk_degree / n_buys) if n_buys > 0 else 0.0
            for code in buy_candidates:
                if code not in closes:
                    continue
                px = deal_series[code].get(today, np.nan)
                hi = highs[code].get(today, np.nan)
                lo = lows[code].get(today, np.nan)
                pc = prev_close[code].get(today, np.nan)
                if not np.isfinite(px) or px <= 0:
                    continue
                if _is_limit_locked(hi, lo, pc, cfg.limit_threshold):
                    continue
                shares = slot_capital / float(px)
                notional = shares * float(px)
                cost = max(notional * open_cost, min_cost)
                total = notional + cost
                if total > cash:
                    continue
                cash -= total
                positions[code] = shares
                entry[code] = {"price": float(px), "cost": total}
                trades.append({
                    "date": today, "code": code, "side": "BUY",
                    "price": float(px), "shares": shares, "cashflow": total,
                    "trade_return": np.nan,
                })

            # Mark-to-close
            mark = 0.0
            for code, sh in positions.items():
                if code in closes:
                    cl = closes[code].get(today, np.nan)
                    if np.isfinite(cl):
                        mark += sh * float(cl)
            equity_rows.append((today, cash + mark))
            daily_rows.append({
                "date": today, "cash": cash, "mark_value": mark,
                "n_positions": len(positions), "n_targets": len(target_codes),
            })

        log_fn(f"[live-IL] step {step:3d}: q={pd.Timestamp(udates[i]).date()}.."
               f"{pd.Timestamp(udates[min(i+fcfg.r-1, len(udates)-1)]).date()} "
               f"loss={test_loss:.5f} ic={ic:+.4f} ric={ric:+.4f} "
               f"equity={equity_rows[-1][1]:,.0f}" if equity_rows else "")
        i += fcfg.r
        step += 1

    eq = pd.DataFrame(equity_rows, columns=["date", "equity"])
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=[
        "date", "code", "side", "price", "shares", "cashflow", "trade_return"])
    summary = _equity_summary(eq, cfg.capital, trades_df)
    bt_result = BTResult(equity=eq, trades=trades_df,
                         daily=pd.DataFrame(daily_rows), summary=summary)
    step_log_df = pd.DataFrame(step_log_rows)
    predictions_df = pd.DataFrame(pred_rows)
    return bt_result, step_log_df, predictions_df
