# Daily Auto-Trader (`deploy_today.py`)

Bridge from the trained DoubleAdapt model to the course's virtual TWSE
trading system. Runs **once per trading day** and submits orders that mirror
the same `BTConfig` the backtest uses.

## Daily schedule

The auto-trader fits cleanly into the TWSE intraday session:

```
TIME         WHAT                                                                 BY
-----------  -------------------------------------------------------------------  ----------
09:00        TWSE market opens                                                    -
13:00        Order window closes (project rules.md Rule 02)                       -
13:30        TWSE market closes                                                   -
15:30-16:00  Course system settlement + bar update window                         -
            -- end of day T-1 --
~16:00       Scrape yesterday's bar into cache                                    scheduled
            -- start of day T --
~11:30       (optional) recheck cache; backfill any missing bar                   scheduled
12:00        deploy_today.py runs -> predict using T-1 features, submit orders    scheduled  <==
13:00        Order window closes; placed orders carry to intraday matching        -
13:30        Day T's bars finalize                                                -
14:30        Day T's settled positions visible via Get_User_Stocks                -
```

The 12:00 trigger is **1 hour before the order window closes**, giving margin
for the API call + retries.

## Causal contract (no look-ahead)

| Date | Feature data used | Prediction is "for" | Orders placed on | Fills happen on |
|---|---|---|---|---|
| T-1's close | bars through T-1 (lag-0 = T-1) | 20d forward from T-1 | T morning | T intraday |

Predictions are computed using **only data available before today's open**,
so there's no use of T's prices to decide T's orders. The
[backtest](../analysis/evaluate/backtest.py) has a mild look-ahead (uses
T's preds to trade on T); the deployment loop fixes it by using T-1's preds
for T's orders.

## Two cadences

| Cadence | What runs | Cost |
|---|---|---|
| **Daily** (12:00) | `deploy_today.py` — inference + order submission | ~30 sec |
| **Every ~20 trading days** | `train.py` — re-run full pipeline to roll the IL bi-level update one step forward (or write a slim `online_update.py` that loads `final.pt`, does one more FOMAML step, saves) | ~30-60 min |

`deploy_today.py` does **not** retrain — it loads `final.pt` and just runs
the forward pass. The model is frozen between FOMAML updates.

## Setup

### 1. One-time: train and persist artifacts

After the scrape finishes:

```powershell
.venv/Scripts/python.exe analysis/scripts/train.py
```

This now writes (in addition to the IL/backtest artifacts):

| File | Used by |
|---|---|
| `analysis/output/run{N}/final.pt` | model weights |
| `analysis/output/run{N}/norm_stats.pkl` | RobustZScoreNorm `(median, MAD)` from pretrain |
| `analysis/output/run{N}/active_universe.json` | the 50 stocks the model is scoring |
| `analysis/output/run{N}/bt_config.json` | trading config (`top_k=10, n_drop=3, ...`) |
| `analysis/output/CURRENT` | text file naming the active run (e.g. `run5`) |

`train.py` auto-increments to the next available `run{N}/` and updates `CURRENT`
to point at the new run. `deploy_today.py` reads `CURRENT` (or `--run runX` to override).

### 2. Credentials

Set environment variables before running deploy:

```powershell
$env:TWSE_ACCOUNT = "your_course_account"
$env:TWSE_PASSWORD = "your_course_password"
```

Or pass on the CLI: `--account ... --password ...`. The `.env` file is
gitignored.

### 3. Run (manual test)

```powershell
# Preview without submitting orders:
.venv/Scripts/python.exe analysis/scripts/deploy_today.py --dry-run

# Live:
.venv/Scripts/python.exe analysis/scripts/deploy_today.py
```

### 4. Schedule on Windows (Task Scheduler)

Action → Start a program:

| Field | Value |
|---|---|
| Program | `C:\workspace\deep_learning_stock_trading\.venv\Scripts\python.exe` |
| Arguments | `analysis/scripts/deploy_today.py` |
| Start in | `C:\workspace\deep_learning_stock_trading` |

Trigger: daily at **12:00**. Add a separate trigger at **16:00** running
`analysis/scripts/scrape.py` to keep the cache current.

Skip weekends with the "Filter by days" condition (Monday-Friday only).

### 5. Schedule on Linux/macOS (cron)

```cron
# Scrape yesterday's bar at 16:00 weekdays
0 16 * * 1-5  cd /path/to/deep_learning_stock_trading && .venv/bin/python analysis/scripts/scrape.py

# Trade at 12:00 weekdays
0 12 * * 1-5  cd /path/to/deep_learning_stock_trading && TWSE_ACCOUNT=... TWSE_PASSWORD=... .venv/bin/python analysis/scripts/deploy_today.py
```

## What the script does (step by step)

[analysis/scripts/deploy_today.py](../analysis/scripts/deploy_today.py)
in 4 phases:

```
[1/4] load artifacts
       final.pt           -> model weights
       norm_stats.pkl     -> feature normalization
       active_universe    -> 50 codes the model knows
       bt_config.json     -> trading hyperparameters
                              (top_k=10, n_drop=3, cost_alpha_bps=70)

[2/4] inference for yesterday's features
       for each code in universe:
           fetch_range(code, history_start, today)
           compute_alpha360(df).tail(1)   # lag-0 = yesterday's close
       stack to (50, 60, 6)
       apply_robust_zscore(X, norm_stats) + fillna(0)
       preds = model.forward(X)            # rank-norm 20d return per stock
       sort desc by pred

[3/4] read holdings + plan orders
       holdings = parse(Get_User_Stocks(account, password))
       target   = top_k(preds where pred >= 0.007)
       sells    = (held - target), lowest pred first, capped by n_drop
       buys     = (target - held), highest pred first, capped by n_drop
       limit prices: round_sell/round_buy(yesterday's close), int-cast

[4/4] submit orders
       for SELL: Sell_Stock(account, password, code, lots, price)
       for BUY:  Buy_Stock(account, password, code, lots, price)
       log every attempt to analysis/output/orders_YYYYMMDD.csv
```

## Output

`analysis/output/orders_YYYYMMDD.csv` accumulates one row per attempted
order each run:

| col | example |
|---|---|
| `attempted_at` | `2026-05-25T12:00:01` |
| `side` | `BUY` or `SELL` |
| `code` | `2330` |
| `lots` | `15` |
| `price` | `608` (int per `stock_api`) |
| `status` | `OK` / `FAIL` / `DRY_RUN` |

## Re-running between FOMAML updates

The model is frozen for ~20 trading days at a time. To keep predictions
fresh **without** retraining, the daily script:

- Always loads the **latest** `final.pt` from disk
- Fetches the **latest** bars (so today's input is genuinely yesterday's
  close, not stale data)
- Reuses `norm_stats.pkl` (computed once, on pretrain)

This means:

- If you re-run `train.py` weekly → daily inference picks up the newer model the next morning
- If you don't re-run `train.py` → daily inference still works, the model just doesn't absorb new data; only its inputs do

## Edge cases handled

| Case | Behavior |
|---|---|
| Missing `final.pt` | Hard exit with a clear message ("run train.py first") |
| Some stock's cache too thin | Logged + skipped (< 61 bars needed for Alpha360) |
| `Get_User_Stocks` returns format we don't recognize | Best-effort regex parse; falls back to "no holdings" |
| Sub-NT$1 tick price for a low-priced stock | `int(round(...))` may lose some precision (course API takes `int` for price) |
| API call fails | Per-order try/except; logged as FAIL; continues with rest |
| Dry-run + no credentials | Assumes empty book, no API calls |

## Things explicitly NOT done (deliberately)

- **No retrying of FAIL orders.** If the limit isn't fillable today, that's a
  signal to try again tomorrow — not to chase the price.
- **No fractional shares.** All `lots` are integer; sub-lot fractional
  capital just sits as cash.
- **No leverage.** The script never goes short. Buy is skipped if cash is
  insufficient; the trading_rules in the backtest enforce the same.
- **No same-day flip.** `hold_thresh=1` from `bt_config.json` is enforced
  implicitly via "if held, don't buy again" (we already own it).
- **No automatic FOMAML update.** Deployment is **inference only**;
  retraining is a separate (slower) job.

## Recommended ops rhythm

| Frequency | What | Why |
|---|---|---|
| Each trading day | `deploy_today.py` at 12:00 | Submit fresh orders for the day |
| Each trading day | `scrape.py` at 16:00 | Keep cache current with yesterday's bar |
| Every ~20 trading days | Re-run `train.py` | Roll one more FOMAML bi-level update into the model |
| Every 6 months | Universe rebalance happens automatically inside `train.py`; manually exit positions in stocks that drop out of the new active 50 | Match the schedule in `data/universe.py` |
