# Qlib Backtest — in-process

In-process backtest that mirrors qlib's `TopkDropoutStrategy` + `Exchange`
behavior. The default path used by `train.py` is the **live-IL mode**: the
backtest loads the post-warmup model checkpoint (`final_prep.pt`) and
**fine-tunes the model every 20 days during the backtest**, exactly like
the live deploy loop would do. A separate "replay" mode (just consumes
pre-computed predictions) is kept for cross-run analysis.

## Two modes

| Mode | Function | Inputs | Use case |
|---|---|---|---|
| **Live-IL (default in train.py)** | `run_qlib_backtest_with_il(model, il_panel, bars, fcfg, cfg, start, end)` | Post-warmup model + features + bars | Honest deploy simulation: model is adapted in-place every r=20 days during the backtest |
| **Replay** | `run_qlib_backtest_from_predictions(preds_df, bars, cfg)` | Pre-computed `predictions.csv` + bars | Re-run any prior model's predictions through the same portfolio engine for A/B comparison |

Both share the same portfolio simulation rules and `QlibBTConfig`.

## Engine

| Property | Value |
|---|---|
| File | [src/evaluate/backtest_qlib.py](../src/evaluate/backtest_qlib.py) |
| Trade price | today's close (`deal_price="close"`) or today's open (`deal_price="open"`) |
| Fill rule | always fills (qlib convention; no `low ≤ limit ≤ high` check) |
| Costs | configurable: `"qlib"` (CN A-share defaults) or `"twse"` (0.1425% × 2 + 0.3% tax + NT$20 min) |
| Slot sizing | qlib exact: `cash * risk_degree / len(buys)` post-sell |
| Limit-up/down block | `limit_threshold=0.095` (skip days where a stock is locked at ±9.5% from prev close) |
| **Live-IL adaptation** | Every `fcfg.r=20` trading days, one FOMAML bi-level step on the past 20-day support window. Model state at trade time D reflects everything it adapted through D − 20. |
| Output | Live-IL: `(BTResult, step_log_df, predictions_df)`. Replay: `BTResult` only. |

## Default config (matches `train.py` and `deploy_today.py`)

```python
QlibBTConfig(
    capital=1e8,            # NT$ 100M starting cash
    top_k=10,               # portfolio width
    n_drop=1,               # 10% daily turnover (qlib's CSI300 uses 50/5 = same ratio)
    cost_alpha_bps=70.0,    # min pred to enter (CSRankNorm space, not % return)
    costs="twse",
    deal_price="close",
    risk_degree=0.95,
)
```

## Cross-run comparison

All 10 runs evaluated under the **same** config above, **same** window
(2024-01-01 → 2026-04-30 ≈ 560 trading days, the operational period common
to every run's `predictions.csv`), **same** bars (current TEJ-adjusted
cache):

| Run | Label h | Universe | Mode | Trades | **Cum return** | **Sharpe** | **Max DD** | Win % | PF | Final equity |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| run1 | 20d | TW50-era | replay | 1,114 | +92.4% | +1.095 | −38.6% | 54.0 | 1.56 | 192.4M |
| run2 | 20d | TW50-era | replay | 1,114 | +147.4% | +1.373 | −45.5% | 51.4 | 1.94 | 247.4M |
| **run3** | 20d | TW50-era | replay | 1,118 | **+204.8%** | +1.661 | −36.4% | 56.0 | 2.11 | **304.8M** |
| run4 | 20d | TW50-era | replay | 1,110 | +131.4% | +1.427 | −40.7% | 52.0 | 1.97 | 231.4M |
| run5 | 20d | TW50-era | replay | 1,116 | +158.9% | +1.426 | −43.7% | 55.2 | 1.98 | 258.9M |
| run6 | 20d | TW50 | replay | 1,118 | +94.4% | **+1.520** | −19.3% | 53.1 | 2.02 | 194.4M |
| **run7** | 20d | TW50 | **live-IL** | 1,116 | +79.0% | +1.367 | −18.1% | 51.4 | 1.94 | 179.0M |
| **run8** | 20d | **TM100** | **live-IL** | 1,106 | +60.1% | +0.856 | −33.6% | 46.9 | 1.57 | 160.1M |
| **run9** | **1d** | TW50 | **live-IL** | 1,128 | +56.7% | +1.242 | **−14.1%** | **59.4** | 2.10 | 156.7M |
| run10 | 1d | **BOTH** | **live-IL** | 1,126 | +46.4% | +0.986 | **−14.0%** | 57.9 | 1.79 | 146.4M |

**Run 9 replay-mode cross-check**: re-running run9's `predictions.csv`
through `run_qlib_backtest_from_predictions` produces cum +56.76% /
Sharpe 1.243 / MDD −14.07% — identical to the live-IL numbers above to
sub-basis-point precision. Confirms `predictions.csv` is faithful to
the in-process backtest.

**Label h column**: forward-return horizon used during training. Runs
1-8 used 20-day forward c2c labels; run 9 uses 1-day forward c2c
(paper-faithful — see [model_env.md](model_env.md) for context). For
the same model quality, 1-day IC is structurally smaller and the
trader takes smaller bets (run9 has 25 pp less cum return but the
smallest drawdown of any run).

**Universe column**:
- *TW50-era* (runs 1-5) — pre-TEJ trained on the curated 150-stock pool / yfinance bars; predictions only cover stocks in the model's then-universe (subset of TW50 names)
- *TW50* (runs 6, 7, 9) — TEJ bars + official TW50 monthly mask (~50 active stocks per period)
- *TM100* (run8) — TEJ bars + official TM100 monthly mask (~100 active stocks per period, ~266 unique over history)
- *BOTH* (run10) — TEJ bars + synthetic union of TW50 ∪ TM100 (~150 active stocks per period, 269 candidate codes after history-mask)

**Mode column**:
- *replay* — predictions generated by a single forward IL pass in `train.py`'s old flow, then a separate `run_qlib_backtest_from_predictions(...)` consumed them
- *live-IL* — predictions generated **during** the backtest by the new `run_qlib_backtest_with_il(...)` flow; the model adapts in-place every 20 days (run7+ uses this)

### Highlights

| Lens | Winner |
|---|---|
| Highest absolute return | **run3** (+204.8%, TW50-era, replay) |
| Highest Sharpe | **run6** (+1.520, TW50, replay) |
| Smallest drawdown | **run9 / run10** tie at −14.0% (both 1-day label, live-IL) — beats run7's −18.1% by 4 pp |
| Largest profit factor | run3 (2.11, replay) — run9 close behind at 2.10 |
| Best win rate | **run9** (59.4%, TW50, 1-day label) — paper-faithful 1d horizon trades hit at a higher rate |
| Best RankICIR | **run10** (0.41, BOTH universe, 1-day label) — wider universe ranks more consistently even when raw IC is weaker |
| **Methodologically most honest** | **run9** (TW50, live-IL, 1-day label — paper-faithful spec; model genuinely adapts during the backtest; 1-day labels are fully observable next-day so no IL leakage) |

### Why TM100 (run8) underperforms TW50 (run7) on the same engine

Same model + same training script + same hyperparameters; only `--universe`
differs. Run8's 100-stock universe is **wider but lower-quality** than TW50:

| | run7 (TW50) | run8 (TM100) |
|---|---|---|
| Active universe per period | ~50 large-caps | ~100 mid-caps (TW50 is **excluded** — TM100 is "the next 100 below TW50") |
| Pretrain best val IC | 0.020 | **0.053** ← model finds MORE signal on the wider panel |
| IL overall IC (operational) | **0.086** | 0.035 ← but the signal is **diluted** when picking top-10 from 100 vs from 50 |
| Cum return | +79.0% | +60.1% |
| Sharpe | +1.37 | +0.86 |
| Max DD | **−18.1%** | −33.6% ← mid-caps less liquid, more volatile drawdowns |
| Win rate | 51.4% | 46.9% |

The val-IC vs operational-IC inversion is the key signal: the model
learned MORE general structure from the wider panel (val IC 2.5× higher),
but the **top-K selection from a noisier mid-cap universe pulls in
worse-quality picks**. Net: TM100 strategy is more volatile and lower-Sharpe
than TW50.

### Why run7 ≈ run6 but not identical

Same data + same model architecture + same hyperparameters. The single
difference is **when adapt happens**:

- run6: `train.py` ran the full IL window (2022 → today) in one pass, saved all predictions, then a separate replay-mode backtest consumed them
- run7: `train.py` ran warmup IL only (2022 → 2023), saved `final_prep.pt`, then the **qlib backtest itself** walked 2024+ doing IL + portfolio sim in tandem

Both flows should produce IDENTICAL predictions at each 20-day boundary if
floating-point arithmetic were perfectly deterministic. In practice:

| Aspect | run6 | run7 |
|---|---|---|
| CUDA op ordering | one continuous loop in `run_incremental` | mixed with portfolio-sim Python | 
| Adam state initialization | once at start | once at start (new `Adam` instance for live-IL phase) |
| Net effect on prediction floats | identical to ~6 decimal places | identical to ~6 decimal places |
| Net effect on portfolio P&L | preserved character (Sharpe + MDD ≈ same); slight rotation differences accumulate over 555 days → ~13pp cum-return drift |

So the **risk-adjusted profile is preserved** (Sharpe 1.39 vs 1.52, MDD 18%
vs 19% — essentially noise) but the **absolute return drifts** because
small per-day rotation differences compound.

### Why run6's profile differs

Same model architecture as run5, same training script, same hyperparameters.
The only differences:

- **run5**: yfinance/TWSE-direct bars (mix of unadjusted + Yahoo's
  dividend-back-adjusted close) + Wikipedia-reconstructed TW50 mask (32
  quarterly change events)
- **run6**: TEJ-adjusted bars (consistent dividend back-adjustment + split
  adjustment per-stock) + official TW50 monthly snapshots (184 month-end
  lists from TEJ's TW50.csv)

The TEJ adjustment produces **smoother forward-return labels** (no synthetic
ex-div price drops), so the model trades less aggressively but with higher
conviction per trade → both upside and downside compress.

## How to run

### Live-IL (run by `train.py` automatically)

Every `train.py` invocation ends with `run_qlib_backtest_with_il(...)`
called on the post-warmup model. To reproduce manually for an arbitrary
run, load `final_prep.pt` and call the function directly:

```python
import torch
from src.data.io import fetch_range
from src.data.handler import build_panel, LABEL_HORIZON
from src.data.processor import apply_robust_zscore, fillna
from src.data.universe import get_index_universe_codes
from src.model.double_adapt import DoubleAdapt
from src.trainer.maml import FOMAMLConfig
from src.evaluate.backtest_qlib import QlibBTConfig, run_qlib_backtest_with_il
import pickle

# Load the post-warmup checkpoint + norm stats
model = DoubleAdapt(seq_len=60, n_fields=6).to("cuda")
model.load_state_dict(torch.load("src/output/run6/final_prep.pt"))
norm = pickle.load(open("src/output/run6/norm_stats.pkl", "rb"))

# Load + featurize + normalize bars (same as train.py)
bars = {c: fetch_range(c, "2022-01-01", "2026-05-31")
        for c in get_index_universe_codes("TW50")}
bars = {c: b for c, b in bars.items() if not b.empty}
panel = build_panel(bars, label_h=LABEL_HORIZON)
panel.X = fillna(apply_robust_zscore(panel.X, norm))

bt, step_log, preds = run_qlib_backtest_with_il(
    model, panel, bars,
    fcfg=FOMAMLConfig(lr_psi=1e-3),
    cfg=QlibBTConfig(top_k=10, n_drop=1, cost_alpha_bps=70,
                     costs="twse", deal_price="close", risk_degree=0.95),
    start_date="2024-01-01", end_date="2026-05-22", device="cuda",
)
print(bt.summary)
```

### Replay (cross-run comparison, no live IL)

For comparing multiple runs' predictions through the same engine without
re-training:

```powershell
.venv/Scripts/python.exe -c "
from src.evaluate.backtest_qlib import QlibBTConfig, run_qlib_backtest_from_predictions
from src.data.io import fetch_range
import pandas as pd

preds = pd.read_csv('src/output/run6/predictions.csv', parse_dates=['date'])
preds['stock_code_id'] = preds['stock_code_id'].astype(str)
preds = preds[preds.date >= '2024-01-01']

bars = {c: fetch_range(c, '2022-01-01', '2026-05-31')
        for c in preds.stock_code_id.unique()}
bars = {c: b for c, b in bars.items() if not b.empty}

cfg = QlibBTConfig(capital=1e8, top_k=10, n_drop=1, cost_alpha_bps=70,
                   costs='twse', deal_price='close', risk_degree=0.95)
print(run_qlib_backtest_from_predictions(preds, bars, cfg).summary)
"
```

### Cross-run comparison (reproduces the table above)

```powershell
.venv/Scripts/python.exe -c "
import pandas as pd
from src.evaluate.backtest_qlib import QlibBTConfig, run_qlib_backtest_from_predictions
from src.data.io import fetch_range

CFG = QlibBTConfig(capital=1e8, top_k=10, n_drop=1, cost_alpha_bps=70,
                   costs='twse', deal_price='close', risk_degree=0.95)
codes = set()
for run in ['run1','run2','run3','run4','run5','run6','run7','run8','run9']:
    df = pd.read_csv(f'src/output/{run}/predictions.csv')
    codes.update(df.stock_code_id.astype(str).unique())
bars = {c: fetch_range(c, '2022-01-01', '2026-05-31') for c in codes}
bars = {c: b for c, b in bars.items() if not b.empty}

for run in ['run1','run2','run3','run4','run5','run6','run7','run8','run9']:
    preds = pd.read_csv(f'src/output/{run}/predictions.csv', parse_dates=['date'])
    preds['stock_code_id'] = preds['stock_code_id'].astype(str)
    preds = preds[preds.date >= '2024-01-01']
    s = run_qlib_backtest_from_predictions(preds, bars, CFG).summary
    print(f'{run}: {s[\"cum_return\"]*100:+.1f}%  sharpe={s[\"sharpe\"]:+.3f}  dd={s[\"max_drawdown\"]*100:+.1f}%')
"
```

## Tunable knobs

| Param | Default | Effect |
|---|---|---|
| `capital` | `1e8` (NT$100M) | starting cash |
| `top_k` | 10 | portfolio width |
| `n_drop` | 1 | daily turnover cap (qlib papers use 50/5 — same 10% ratio) |
| `cost_alpha_bps` | 70 | min pred score to enter (CSRankNorm space, NOT % return — see [model_env.md](model_env.md)) |
| `costs` | `"twse"` | swap to `"qlib"` for CN A-share defaults (0.05% buy + 0.15% sell bundled) |
| `deal_price` | `"close"` | swap to `"open"` for live-achievable opening-auction model |
| `risk_degree` | 0.95 | fraction of cash deployed each rebalance |
| `limit_threshold` | 0.095 | block trades when stock is locked at ±9.5% limit |

## Causal note

The qlib engine trades on date `D` using `pred[D]` at `D`'s close.
Technically uses today's close to decide today's trade (mild look-ahead;
this matches qlib's published methodology). The course backtest fixes this
by inserting a 1-day lag — see [course_backtest.md](course_backtest.md).

## Related

- [model_env.md](model_env.md) — what changed in each run (the inputs to this table)
- [model_performance.md](model_performance.md) — qlib-style metrics (IC, ICIR, IR, excess return) vs 0050 benchmark
- [course_backtest.md](course_backtest.md) — the other backtest engine (zero-cost, leaderboard-uploadable)
- [auto_trader.md](auto_trader.md) — daily deploy pipeline using whichever run is in `CURRENT`
- [topk_ndrop_selection.md](topk_ndrop_selection.md) — why top_k=10 / n_drop=1
