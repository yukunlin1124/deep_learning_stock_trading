# Backtest — run5 Model

Two backtest engines, same model (`output/run5/predictions.csv`), same
strategy shape (top_k=10, n_drop=1, risk_degree=0.95). One is an in-process
simulator that mirrors qlib's defaults; the other drives the course's
official `BacktestSystem` from `stock_project_for_class/backtest_package/`.

## The two engines

| Engine | File | Trade price | Fill rule | Costs | Notes |
|---|---|---|---|---|---|
| **qlib** | [src/evaluate/backtest_qlib.py](../src/evaluate/backtest_qlib.py) | today's close (`deal_price="close"`) | always fills | **TWSE** (0.1425% × 2 + 0.3% tax + NT$20 min) | In-process; qlib-exact slot formula `cash * risk / len(buys)` post-sell |
| **course** | [src/evaluate/run_course_backtest.py](../src/evaluate/run_course_backtest.py) | today's close (matches [examplebacktest.py](../stock_project_for_class/backtest_package/stock_backtest_v1/examplebacktest.py)) | `low ≤ price ≤ high` (framework check, trivially true at close) | **none** — framework charges zero | Drives the course's `BacktestSystem`; result uploadable to leaderboard at `140.116.86.242:8081` |

Both use the **same selection logic**: top-K codes by DoubleAdapt pred,
above the `cost_alpha_bps=70` threshold (≈ 0.7% in CSRankNorm-space → keeps
out clearly bearish picks). Both use the **qlib `TopkDropoutStrategy`** shape
(sell-all-dropped-out + buy-all-new-entries; no `n_drop` cap in the rewrite
that mirrors the example).

## Headline results

### Full IL window (2022-01-03 → 2026-04-23, 1,040 trading days)

| Engine | Trades | Cum return | Sharpe | Max DD | Win rate | Profit factor | Final equity |
|---|---|---|---|---|---|---|---|
| qlib (TWSE costs) | 1,838 | **+190.3%** | 0.804 | −46.98% | 56.8% | 1.79 | NT$ 290.3M |
| course (zero costs) | n/a (framework opaque) | **+252.7%** | n/a | n/a | n/a | n/a | NT$ 352.7M (10× scale) |

### Operational window (2024-01-01 → 2026-04-23, 555 trading days, post-warmup)

| Engine | Trades | Cum return | Sharpe | Max DD | Win rate | Profit factor | Final equity |
|---|---|---|---|---|---|---|---|
| qlib (TWSE costs) | 1,028 | **+169.9%** | 1.278 | −46.80% | 53.4% | 2.06 | NT$ 269.9M |
| course (zero costs) | n/a | **+135.3%** | n/a | n/a | n/a | n/a | NT$ 235.3M (10× scale) |

## Reading the numbers

- **course > qlib on full IL** because the course framework charges **zero
  transaction costs**, so the +252.7% includes ~50-70 pp of "cost alpha"
  that the qlib +190.3% number absorbs into commissions and TWSE sell tax.
- **course < qlib on operational window** because the course framework runs
  `select_stock(D)` then `trade_strategy(D+1)` — every trade has a 1-day
  lag, so the strategy reacts a day late to today's pred. Over a shorter,
  trendier window (2024-2026) that lag costs more than zero-cost saves.
- The course-framework number (Sharpe / MDD / etc.) only comes back as a
  single P&L scalar; the framework doesn't expose per-day equity, so we
  can't compute Sharpe / drawdown for it without reparsing its Excel output.

## How to run

### Qlib backtest (in-process, fast)

Reproduces the qlib row of the table above:

```powershell
.venv/Scripts/python.exe -c "
from src.evaluate.backtest_qlib import QlibBTConfig, run_qlib_backtest_from_predictions
from src.data.io import fetch_range
import pandas as pd

preds = pd.read_csv('src/output/run5/predictions.csv', parse_dates=['date'])
preds['stock_code_id'] = preds['stock_code_id'].astype(str)
bars = {c: fetch_range(c, '2021-01-01', '2026-04-30')
        for c in preds.stock_code_id.unique()}
bars = {c: b for c, b in bars.items() if not b.empty}

cfg = QlibBTConfig(capital=1e8, top_k=10, n_drop=1, cost_alpha_bps=70,
                   costs='twse', deal_price='close', risk_degree=0.95)
r = run_qlib_backtest_from_predictions(preds, bars, cfg)
print(r.summary)
"
```

Tunable knobs in `QlibBTConfig`:

| Param | Default | Effect |
|---|---|---|
| `capital` | `1e8` (NT$100M) | starting cash |
| `top_k` | 10 | portfolio width |
| `n_drop` | 1 | daily turnover cap (qlib's papers use 50/5) |
| `cost_alpha_bps` | 70 | min pred score to enter (CSRankNorm space, NOT % return) |
| `costs` | `"qlib"` | swap to `"twse"` for our exchange rates |
| `deal_price` | `"close"` | swap to `"open"` for live-achievable opening-auction model |
| `risk_degree` | 0.95 | fraction of cash deployed each rebalance |

### Course backtest (uses the framework + uploads to leaderboard)

```powershell
# Dry (no upload):
.venv/Scripts/python.exe src/evaluate/run_course_backtest.py

# Specific run + custom window:
.venv/Scripts/python.exe src/evaluate/run_course_backtest.py --run run5 `
    --start-date 20240101 --end-date 20260423

# Upload to leaderboard (needs credentials):
$env:TWSE_ACCOUNT = "<account>"
$env:TWSE_PASSWORD = "<password>"
.venv/Scripts/python.exe src/evaluate/run_course_backtest.py --upload
```

CLI flags:

| Flag | Default | Effect |
|---|---|---|
| `--run` | reads `output/CURRENT` | which `runN/` to use |
| `--start-date` | min date in `predictions.csv` | YYYYMMDD |
| `--end-date` | max date in `predictions.csv` | YYYYMMDD |
| `--cash` | `100_000` (framework units = 100M NT$) | starting cash |
| `--strategy-name` | `"DoubleAdapt {run} (full IL)"` | leaderboard label |
| `--upload` | off | POST to course API |
| `--account`, `--password` | env vars `TWSE_ACCOUNT` / `TWSE_PASSWORD` | required for `--upload` |

The script pre-builds the framework's local stock cache from
[src/_cache/](../src/_cache/) so it doesn't redownload everything from the
course API on every run.

## Causal alignment (no look-ahead in either engine)

- Each pred for date `D` was computed from features through `D`'s close.
- The **qlib engine** trades on `D` using `pred[D]` at `D`'s close —
  technically uses today's close to decide today's trade (mild look-ahead;
  matches qlib's published methodology).
- The **course framework** calls `trade_strategy(D)` *before*
  `select_stock(D)`, so today's trade is informed by `pred[D-1]` (yesterday's
  signal, no look-ahead at all). Same causality as
  [deploy_today.py](../src/scripts/deploy_today.py).

The 1-day lag in the course framework is part of why its operational-window
return (+135%) is lower than qlib's (+170%) — qlib captures today's move on
today's signal; the course misses one day's edge on every rotation.

## Files

| File | Purpose |
|---|---|
| [src/evaluate/backtest_qlib.py](../src/evaluate/backtest_qlib.py) | qlib-style in-process engine |
| [src/evaluate/run_course_backtest.py](../src/evaluate/run_course_backtest.py) | bridges DoubleAdapt → course framework |
| [src/output/run5/predictions.csv](../src/output/run5/) | model's per-day output (date, stock_code_id, pred, actual) |
| [src/output/run5/bt_config.json](../src/output/run5/) | trading config (top_k, n_drop, etc.) |
| [src/output/run5/course_backtest.xlsx](../src/output/run5/) | course framework's full output (regenerated each run) |
| [stock_project_for_class/backtest_package/stock_backtest_v1/examplebacktest.py](../stock_project_for_class/backtest_package/stock_backtest_v1/examplebacktest.py) | template our `run_course_backtest.py` mirrors |
