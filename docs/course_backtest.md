# Course Backtest — via the official BacktestSystem

Bridge that runs DoubleAdapt predictions through the course's official
`BacktestSystem` (from
[stock_project_for_class/backtest_package/](../stock_project_for_class/backtest_package/)).
Results are uploadable to the leaderboard at `140.116.86.242:8081`.

**Source of predictions:** this script reads `predictions.csv` from a run
dir. In the current pipeline that file is produced by `train.py`'s
**live-IL qlib backtest** (Stage 3 — see
[qlib_backtest.md](qlib_backtest.md) and the README's training-schedule
section). So `predictions.csv` reflects a model that was being fine-tuned
every 20 days during the 2024+ window. The course backtest then replays
those predictions through the zero-cost framework; **it does not itself
re-run inference or fine-tune** — the model isn't even loaded.

## Engine

| Property | Value |
|---|---|
| File | [src/evaluate/run_course_backtest.py](../src/evaluate/run_course_backtest.py) |
| Framework | `BacktestSystem` from `stock_project_for_class/backtest_package/` |
| Trade price | today's close (matches [examplebacktest.py](../stock_project_for_class/backtest_package/stock_backtest_v1/examplebacktest.py)) |
| Fill rule | `low ≤ price ≤ high` (framework check; trivially true at close) |
| **Costs** | **zero** — the course framework charges no commission, no tax, no min |
| Slot sizing | qlib exact: `cash * 0.95 / len(buys)` |
| Causal lag | **1 day** — framework calls `trade_strategy(D)` *before* `select_stock(D)`, so trades on `D` use `pred[D-1]` |
| Output | per-day asset value (in framework Excel), no per-day equity exposed |

The 1-day causal lag means the course backtest is **strictly more honest**
than the qlib engine (which uses today's pred on today's trade). Same
causality as [deploy_today.py](../src/scripts/deploy_today.py).

## Cross-run comparison

All 8 runs evaluated on the **operational window 2024-01-02 → 2026-04-23**
(555 trading days), initial cash NT$100M (framework units = 100k):

| Run | Universe | Cum return | Final equity | Notes |
|---|---|---:|---:|---|
| run1 | TW50-era | +59.3% | NT$ 159.3M | qlib defaults baseline |
| run2 | TW50-era | +100.7% | NT$ 200.7M | + tuned pretrain |
| run3 | TW50-era | +112.2% | NT$ 212.2M | + longer pretrain |
| run4 | TW50-era | +124.4% | NT$ 224.4M | + 2022-2023 IL warmup |
| **run5** | TW50-era | **+135.3%** | **NT$ 235.3M** | + clean held-out val |
| run6 | TW50 | +76.0% | NT$ 176.0M | TEJ cache + monthly TW50 mask |
| **run7** | TW50 | **+124.0%** | **NT$ 224.0M** | + live-IL backtest flow (predictions generated during the qlib backtest, not as a separate pre-pass) |
| run8 | **TM100** | +55.4% | NT$ 155.4M | --universe TM100 (100-stock mid-cap mask, 24 codes auto-injected for the framework's 2330 day-reference) |

### Full-IL view (2022-01 → 2026-04, 1,040 days)

Runs 4-6 have predictions for 2022-2023 (the IL warmup window); runs 1-3
and run7 don't (run7's predictions.csv is scoped to 2024+ only by design).

| Run | Cum return | Final equity |
|---|---:|---:|
| run4 | +138.7% | NT$ 238.7M |
| **run5** | **+252.7%** | **NT$ 352.7M** |
| run6 | (not yet measured on full window) | |
| run7 | N/A — predictions.csv starts at 2024-01-01 | — |

### Why run7's course-backtest number (+124%) is higher than its qlib-backtest number (+81%)

Same predictions, different engines. Run7's qlib backtest charges TWSE
costs (~58 bps round-trip) and uses today's close as the deal price; the
course backtest charges nothing and runs with the 1-day causal lag. Net:
zero-cost gives a tailwind that more than compensates the lag-cost over
the 2024-2026 trending market.

For run7, both engines agree on the **strategy character**: profitable
with moderate trade frequency (~1100 trades / 555 days = ~2/day).

### Why run6 looks lower here than in [qlib_backtest.md](qlib_backtest.md)

Same data-source story: TEJ-adjusted prices compress run6's prediction
magnitudes, leading to smaller positions and smaller P&L swings in either
direction. The Sharpe / drawdown advantage that the qlib engine reveals
(Sharpe 1.52, MDD −19%) isn't visible in the course framework because the
course doesn't expose per-day equity — only the final P&L scalar.

## How to run

### Dry (no leaderboard upload)

```powershell
# Default: read output/CURRENT, full IL window in predictions.csv
.venv/Scripts/python.exe src/evaluate/run_course_backtest.py

# Specific run + operational window
.venv/Scripts/python.exe src/evaluate/run_course_backtest.py --run run6 `
    --start-date 20240101 --end-date 20260522
```

### Live upload to leaderboard

```powershell
$env:TWSE_ACCOUNT = "<your_account>"
$env:TWSE_PASSWORD = "<your_password>"
.venv/Scripts/python.exe src/evaluate/run_course_backtest.py --upload `
    --strategy-name "DoubleAdapt run6 (TEJ)"
```

The script:
1. Pre-builds the framework's local stock cache from [src/_cache/](../src/_cache/) so it doesn't redownload from the course API
2. Writes `course_backtest.xlsx` to the run dir
3. Optionally POSTs summary + per-trade detail to the leaderboard API

### CLI flags

| Flag | Default | Effect |
|---|---|---|
| `--run` | reads `output/CURRENT` | which `runN/` to use |
| `--start-date` | min date in `predictions.csv` | YYYYMMDD |
| `--end-date` | max date in `predictions.csv` | YYYYMMDD |
| `--cash` | `100_000` (framework units = NT$100M at framework's implicit 1000× scaling) | starting cash |
| `--strategy-name` | `"DoubleAdapt {run} (full IL)"` | leaderboard label |
| `--upload` | off | POST to course API |
| `--account`, `--password` | env vars `TWSE_ACCOUNT` / `TWSE_PASSWORD` | required for `--upload` |

### Reproduce the cross-run table

```powershell
foreach ($run in @("run1","run2","run3","run4","run5","run6")) {
    .venv/Scripts/python.exe src/evaluate/run_course_backtest.py `
        --run $run --start-date 20240101 `
        --strategy-name "DoubleAdapt $run"
}
```

Each call rebuilds the framework's local stock cache (~5s) and writes the
per-run Excel output to `src/output/{run}/course_backtest.xlsx`.

## Caveats

- **No transaction costs.** The course framework charges nothing. Live
  trading still pays the 14.25 bps + 30 bps tax + NT$20 min — see
  [qlib_backtest.md](qlib_backtest.md) for the TWSE-cost view.
- **No per-day equity exposed.** Only the final P&L scalar comes back from
  `calculate_performance()`. So Sharpe, max drawdown, etc. aren't directly
  computable from the framework's output (would require parsing its Excel).
- **1-day causal lag** is stricter than qlib's same-day fill. This matches
  the deploy reality but reduces operational-window absolute returns.
- **Framework reads bars from its own `./tmp/stock_data.csv` cache** — we
  pre-populate it from `src/_cache/` to avoid the slow per-stock course-API
  download.

## Related

- [model_env.md](model_env.md) — what changed in each run (the inputs to this table)
- [model_performance.md](model_performance.md) — qlib-style metrics (IC, ICIR, IR, excess return) vs 0050 benchmark
- [qlib_backtest.md](qlib_backtest.md) — the other backtest engine (TWSE costs + per-day equity)
- [auto_trader.md](auto_trader.md) — daily deploy pipeline (same causal pattern as this engine)
