# Selecting `top_k` and `n_drop`

These two `BTConfig` fields ([analysis/evaluate/backtest.py](../analysis/evaluate/backtest.py))
drive the entire trading policy. Picking them is more important than any
model hyperparameter — get them wrong and the model's alpha is either
diluted away (too high `top_k`) or eaten by costs (too high `n_drop`).

## What they do

| Field | Meaning |
|---|---|
| `top_k` | Number of long positions held at the same time (basket size) |
| `n_drop` | Daily turnover cap: at most `n_drop` names swapped per day (set to `None` for no cap) |

The pair defines **selectivity** and **turnover**:
- `top_k / universe_size` = **selectivity ratio** — what fraction of the universe is in the book at any time
- `n_drop / top_k` = **daily turnover** — fraction of the book that rotates each day

## Why both matter

```
Universe (N stocks)
    │
    ├── model scores each stock daily
    ▼
Top-K candidates by predicted return
    │
    ├── n_drop cap: only swap up to n_drop names vs. yesterday's book
    ▼
Today's actual trades  ────►  cost drag ≈ 58.5 bps per round-trip on TWSE
```

Bigger `top_k` → more names held → less concentration (more diversified, less alpha
expression). Bigger `n_drop` → faster reaction to new signals but more cost drag.

## qlib's convention

Across **every** benchmark workflow in `qlib/examples/benchmarks/` — covering
GRU, LSTM, ALSTM, GATs, LightGBM, XGBoost, CatBoost, MLP, Linear, ADD, ADARNN,
DoubleEnsemble, HIST, IGMTF, KRNN, KEnhance, KEMLP, Sandwich — the strategy
config is:

```yaml
strategy:
    class: TopkDropoutStrategy
    kwargs:
        topk: 50
        n_drop: 5
```

And it stays the same regardless of market:

| Market | Universe | `top_k` | `n_drop` | Selectivity | Daily turnover |
|---|---|---|---|---|---|
| CSI300 | ~300 | 50 | 5 | 16.7% | 10% |
| CSI500 | ~500 | 50 | 5 | 10.0% | 10% |

So qlib fixes the **absolute** numbers, and the **selectivity ratio**
varies with universe size. The **daily-turnover ratio (`n_drop/top_k =
10%`) is the design invariant** — basket fully recycles every ~10 trading
days.

## Two-stage filtering in this project

Unlike qlib (universe = index constituents), ours has a **liquidity
pre-filter**:

```
220 TWSE candidate codes (data/universe.py CANDIDATE_CODES)
        │
        ├── Stage 1: trailing 180d $vol ranking, semi-annual rebalance
        ▼
   Active 50-stock universe (TOP_N = 50)
        │
        ├── Stage 2: model predicts 20-day forward return per stock
        │             top_k by pred above 70 bps threshold
        ▼
   Actual long basket (top_k positions held)
```

So the **universe is already pre-selected to 50 names by liquidity**.
`top_k` and `n_drop` then control how aggressively the model picks within
those 50.

## Decision framework

### Step 1 — pick `top_k` from selectivity ratio

Match a target ratio of held / universe. Three reference regimes:

| Goal | Ratio | `top_k` (universe = 50) | Behavior |
|---|---|---|---|
| **Concentrated alpha** | ~10% | **5** | Match CSI500's qlib selectivity — best 10% only |
| **Balanced selection** (current default) | ~20% | **10** | Slightly less selective than CSI300 |
| **Diversified, market-tracking** | ~50% | 25 | Half the universe — alpha diluted |
| **Index-like** | ~100% | 50 | Buy the universe; model only allocates cash |

**Anti-pattern**: setting `top_k = universe_size`. The model loses its
"selection" job — it only re-allocates marginal cash among names you'd own
anyway. This is what `top_k=50` would degenerate into for our 50-stock
universe.

### Step 2 — pick `n_drop` from daily turnover ratio

Once `top_k` is set, pick `n_drop` as a fraction of `top_k`:

| Daily turnover | `n_drop` (top_k = 10) | Avg holding period | Annual cost drag (TWSE) |
|---|---|---|---|
| Conservative (~10%) | **1** | ~20 days | ~15% |
| qlib-canonical (10%) | 1 | ~20 days | ~15% |
| Moderate (~30%) | 3 | ~7 days | ~37% |
| Aggressive (~50%) | 5 | ~4 days | ~60% |
| Unconstrained | `None` | varies | up to ~100% |

Cost-drag rule of thumb on TWSE:
```
annual cost drag ≈ trades_per_day × 58.5 bps × 252 days / capital
                 ≈ (2 × n_drop) × 58.5 bps × 252
```

`58.5 bps` is one TWSE round-trip (14.25 bps commission × 2 sides + 30 bps
sell tax). At `n_drop = 5` you trade ~10 contracts/day, eating ~60%/yr —
the model must beat that just to break even.

### Step 3 — sanity-check slot capital

```
slot_capital = capital / top_k
```

For `capital = NT$100M`:

| `top_k` | slot_capital | Notes |
|---|---|---|
| 5 | NT$20M | Comfortable — even high-priced names (TSMC ~600) fit ~30 lots |
| 10 | NT$10M | Comfortable — most names fit ~5-15 lots |
| 25 | NT$4M | Tight on TSMC (~6 lots); fine on mid-caps |
| 50 | NT$2M | Pinched on high-priced names (TSMC = 3 lots = NT$1.8M) |

Slot size below ~NT$5M starts to round-down significantly because lot
granularity (1000 shares) becomes coarse relative to slot size.

## Concrete recommendations for this project

Active universe = **50 stocks** ([data/universe.py:24](../analysis/data/universe.py#L24)).

| Use case | `top_k` | `n_drop` | Rationale |
|---|---|---|---|
| **Default** (current) | 10 | `None` | Slightly under CSI300 selectivity; no daily turnover cap |
| **Mirror CSI300 selectivity** | 8 | 1 | 16% selectivity, 12.5% daily turnover (close to qlib's 10%) |
| **Mirror CSI500 selectivity** | 5 | 1 | 10% selectivity, 20% daily turnover (more reactive) |
| **Target ~5 trades/day** | 10 | 3 | 30% daily turnover, ~37% annual cost drag |
| **Aggressive rotation** | 10 | 5 | 50% daily turnover, ~60% annual cost drag — model must be very strong |
| **Diversified, low alpha** | 25 | 3 | 12% daily turnover but only 50% selectivity |
| **Anti-pattern: don't do this** | 50 | 5 | top_k = universe → degenerate "long the index" |

The **default `BTConfig(top_k=10, n_drop=None)`** in
[analysis/evaluate/backtest.py:41-47](../analysis/evaluate/backtest.py#L41-L47)
is a reasonable starting point: 20% selectivity (between CSI300 and
CSI500), no turnover cap so the model fully expresses its rebalance
intent. Switch to `n_drop=1` if you want to control daily costs.

## How to change

Edit the `bt_cfg = BTConfig()` line in
[analysis/scripts/train.py:170](../analysis/scripts/train.py#L170):

```python
# Default
bt_cfg = BTConfig()

# Conservative, qlib-CSI300-like selectivity, capped turnover
bt_cfg = BTConfig(top_k=8, n_drop=1)

# Target ~5 trades/day
bt_cfg = BTConfig(top_k=10, n_drop=3)
```

## When to revisit

Re-tune `top_k` / `n_drop` when:

1. **Universe size changes** — if `TOP_N` in `data/universe.py` changes from
   50, recompute the selectivity ratio
2. **Model quality improves** — stronger alpha justifies tighter `top_k`
   (more selective) and higher `n_drop` (faster rotation)
3. **Cost regime changes** — if you start paying lower commissions or
   trade a non-TWSE market, the cost-drag math shifts
4. **Slot capital becomes binding** — if max_lots is rounding down to 0
   for several names, reduce `top_k`

## References

- [analysis/evaluate/backtest.py](../analysis/evaluate/backtest.py) — `BTConfig`, `run_backtest_from_predictions`
- [analysis/evaluate/trading_rules.py](../analysis/evaluate/trading_rules.py) — commission, tax, lot, tick rules
- [stock_project_for_class/rules.md](../stock_project_for_class/rules.md) — TWSE trading-rules source of truth
- qlib reference: [qlib/contrib/strategy/signal_strategy.py:75-128](../qlib/qlib/contrib/strategy/signal_strategy.py#L75-L128) — `TopkDropoutStrategy` class
- qlib YAML conventions: any file under [qlib/examples/benchmarks/*/workflow_config_*.yaml](../qlib/examples/benchmarks/)
