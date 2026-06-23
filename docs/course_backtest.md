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

All 15 runs evaluated on the **operational window 2024-01-02 → 2026-04-30**
(560 trading days), initial cash NT$100M (framework units = 100k):

| Run | Label h | Universe | Mode | Cum return | Final equity | Notes |
|---|---|---|---|---:|---:|---|
| run1 | 20d | TW50-era | replay | +58.7% | NT$ 158.7M | qlib defaults baseline |
| run2 | 20d | TW50-era | replay | +95.1% | NT$ 195.1M | + tuned pretrain |
| run3 | 20d | TW50-era | replay | +114.1% | NT$ 214.1M | + longer pretrain |
| **run4** | 20d | TW50-era | replay | **+147.8%** | **NT$ 247.8M** | + 2022-2023 IL warmup (new course-backtest leader) |
| run5 | 20d | TW50-era | replay | +99.8% | NT$ 199.8M | + clean held-out val |
| run6 | 20d | TW50 | replay | +76.0% | NT$ 176.0M | TEJ cache + monthly TW50 mask |
| **run7** | 20d | TW50 | **live-IL** | **+124.0%** | **NT$ 224.0M** | + live-IL backtest flow (predictions generated during the qlib backtest, not as a separate pre-pass) |
| run8 | 20d | **TM100** | **live-IL** | +55.4% | NT$ 155.4M | --universe TM100 (100-stock mid-cap mask, 24 codes auto-injected for the framework's 2330 day-reference) |
| run9 | **1d** | TW50 | **live-IL** | +49.5% | NT$ 149.5M | + paper-faithful 1-day c2c label (production recommendation) |
| run10 | 1d | **BOTH** | live-IL | +37.1% | NT$ 137.1M | + universe TW50 ∪ TM100 (~150/period) |
| run11 | 1d | TW50 | live-IL **no-CSRankNorm** | +11.4% | NT$ 111.4M | ablation: strip CSRankNorm wrap (inner FOMAML collapses) |
| run12 | 1d | TW50 | live-IL **no best-state restore** | +35.7% | NT$ 135.7M | ablation: skip restore-to-best-IC after early stop |
| run13 | 1d | TW50 | live-IL **patience=16** | +49.5% | NT$ 149.5M | ablation: bit-identical to run9 (best-state never beaten) |
| run14 | 1d | TW50 | live-IL **r=5** | +40.1% | NT$ 140.1M | ablation: 5-day task span — more steps but noisier per-step preds |
| run17 | 1d | TW50 | live-IL **qlib-exact Alpha360** | +31.5% | NT$ 131.5M | paper-faithful Alpha360 — wins +20% on pretrain val IC but the course 1-day lag erodes the intraday-shape signal (qlib gap was only −3 pp; course gap is −18 pp) |

> **Table refreshed 2026-05-28**. Re-ran the course backtest for every
> run with `--start-date 20240101` (operational window) against the
> framework's bar cache pre-populated from `src/_cache/`. Numbers above
> are from the framework's `total return` field; final equity uses the
> framework's implicit 1000× scaling (100k framework units = NT$100M
> starting cash).

### Why run9 (live-IL, 1-day label) lags the 20-day-label runs here

The course framework has **no transaction costs**, so the optimal play
is to make bigger bets — exactly what the 20-day-label runs do. The
1-day-label runs (run9-run14) take more conservative position sizes,
which is the right trade in real markets (TWSE pays ~58 bps round-trip)
but leaves return on the table in this zero-cost simulator.

For the **paper-faithful & deploy-honest** ranking, run9 still wins
across both engines on risk-adjusted metrics (Sharpe + max DD — see
[qlib_backtest.md](qlib_backtest.md) and [model_performance.md](model_performance.md)).

### Why run7's course-backtest number (+124%) is higher than its qlib-backtest number (+79%)

Same predictions, different engines. Run7's qlib backtest charges TWSE
costs (~58 bps round-trip) and uses today's close as the deal price; the
course backtest charges nothing and runs with the 1-day causal lag. Net:
zero-cost gives a tailwind that more than compensates the lag-cost over
the 2024-2026 trending market.

For run7, both engines agree on the **strategy character**: profitable
with moderate trade frequency (~1100 trades / 560 days = ~2/day).

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
