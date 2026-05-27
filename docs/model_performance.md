# Model Performance — qlib-style metrics vs 0050 benchmark

Apples-to-apples comparison of all 8 runs using qlib's standard prediction
+ portfolio metrics. Benchmark is **0050.TW** (Yuanta Taiwan 50 ETF) over
the same calendar window as each run's backtest.

## Why 0050?

It's the natural benchmark for any TWSE long-equity strategy:
- Tracks the FTSE TWSE 50 index, the same large-cap universe runs 1-7
  draw from (run8 uses TM100, but 0050 remains the most-liquid passive
  alternative).
- Investable: tight spreads, low TER, no model risk.
- Listed since 2003; full 2024-2026 history is on yfinance (TEJ doesn't
  cover ETFs).

Cache file: [src/_cache/0050_benchmark.csv](../src/_cache/0050_benchmark.csv)
(auto-adjusted close from yfinance, 597 rows 2023-12 → 2026-05).

## Metrics computed (qlib convention)

All formulas match `qlib.contrib.evaluate.risk_analysis` so numbers are
directly comparable to any qlib report:

| Metric | Formula | Source |
|---|---|---|
| **IC** | mean per-day Pearson(pred, label) | `il_overall_ic` in `report.json` (computed during live-IL backtest on the 2024+ window) |
| **ICIR** | mean(IC) / std(IC) | `il_overall_icir` |
| **RankIC** | mean per-day Spearman(pred, label) | `il_overall_ric` |
| **RankICIR** | mean(RankIC) / std(RankIC) | `il_overall_ricir` |
| **Excess Ann Return (w/ cost)** | `mean(strat_ret − bm_ret) × 252` | computed here; strat returns already net of TWSE costs |
| **Information Ratio (w/ cost)** | `mean(excess) / std(excess) × sqrt(252)` | computed here |
| **Max Drawdown (w/ cost)** | min of strategy equity drawdown | `backtest.max_drawdown` in `report.json` |
| **Excess Max DD (w/ cost)** | min of cumulative excess (`cumsum(excess) − cummax`) | computed here |

> **qlib convention note.** qlib uses *additive* annualization
> (`mean × 252`) and *additive* drawdown (`cumsum`, not `cumprod`), so its
> "annualized return" differs from CAGR. Excess and IR follow the same
> additive convention. This is what qlib's standard tables report —
> mixing CAGR here would not be qlib-comparable.

## Benchmark — 0050 over 2024-01-02 → 2026-04-30 (560 trading days)

| Stat | Value |
|---|---:|
| Cumulative return | +184.6% |
| qlib-ann return (mean × 252) | +50.4% |
| CAGR | +60.1% |
| Annualized volatility | 25.7% |

0050 was an exceptional benchmark in this window — TWSE rode the AI/TSMC
rally for ~2.5× cumulative return. Beating it with a long-only TW50/TM100
model that pays TWSE round-trip costs is a *very* high bar; **all 8 runs
underperform 0050 on excess-return basis**. The model story is in IC /
RankIC / Sharpe (raw P&L is still attractive in absolute terms — runs
4-8 cleared +30%/yr after costs — just not vs a passively-held 50-stock
mega-cap basket in a bull market).

## Cross-run table (vs 0050)

| Run | Label horizon | Universe | Mode | IC | ICIR | RankIC | RankICIR | Excess Ann Ret (w/ cost) | IR (w/ cost) | Max DD (w/ cost) | Excess Max DD (w/ cost) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| run1 | 20d | TW50-era | replay | −0.0105 | −0.0521 | −0.0249 | −0.1235 | −101.6% | −3.78 | −83.0% | −2.40 |
| run2 | 20d | TW50-era | replay | +0.0398 | +0.1882 | +0.0158 | +0.0835 | −39.0% | −0.93 | −58.9% | −1.36 |
| run3 | 20d | TW50-era | replay | +0.0695 | +0.3655 | +0.0551 | +0.3083 | −25.3% | −1.00 | −33.4% | −0.74 |
| run4 | 20d | TW50-era | replay | +0.0134 | +0.0828 | +0.0150 | +0.0944 | −3.5% | −0.11 | −37.9% | −0.61 |
| run5 | 20d | TW50-era | replay | +0.0490 | +0.2346 | +0.0322 | +0.1650 | −17.8% | −0.52 | −27.1% | −0.72 |
| **run6** | 20d | TW50 | replay | +0.0499 | +0.2608 | +0.0605 | **+0.3363** | −16.3% | −1.01 | −19.3% | **−0.48** |
| **run7** | 20d | TW50 | **live-IL** | **+0.0856** | **+0.4748** | **+0.0937** | **+0.5131** | −21.6% | −1.32 | −18.1% | −0.60 |
| run8 | 20d | **TM100** | **live-IL** | +0.0348 | +0.2763 | +0.0318 | +0.2553 | −21.1% | −0.94 | −31.4% | −0.71 |
| **run9** | **1d** | TW50 | **live-IL** | +0.0459 | +0.2058 | +0.0672 | +0.3260 | −28.8% | −1.69 | **−14.07%** | −0.69 |
| run10 | 1d | **BOTH** | live-IL | +0.0263 | +0.1767 | +0.0588 | +0.4141 | −31.5% | −1.70 | **−14.01%** | −0.73 |

**Bold** = best-in-column (Excess Ann Ret and IR are negative for every
run, so no winners are bolded; "least negative" is run4 for Excess Ann
Ret and IR).

> **Comparing IC across label horizons is unfair.** Run9 is the first
> 1-day-label run; runs 1-8 used 20-day forward labels. For the same
> underlying model quality, 20-day forward IC is structurally **3-5×
> larger** than 1-day forward IC (because aggregating returns over more
> days raises the signal-to-noise ratio). The fair lens for run9 is
> *drawdown* and *risk-adjusted return* — see the run7 vs run9 callout
> below.

## What this table actually says

### Signal quality (IC, ICIR, RankIC, RankICIR)
- **Among 20-day-label runs, run7 wins on every signal-quality metric.**
  IC = 0.086, RankIC = 0.094, with the highest ICIR (0.47) and RankICIR
  (0.51). This is consistent with the live-IL flow giving the model a
  continuously-warmer adapter state throughout 2024-2026.
- **run1 is the only run with negative IC** — TW50-era's pretrained-only
  baseline (no IL tuning) had a sign-flipped signal on the operational
  window. Everything after run2 (pretrain + IL) is +IC.
- **run3 was already near-best on signal quality** (IC 0.070, RankIC 0.055)
  despite being TW50-era / replay — the "+ longer pretrain" tweak in run3
  hit a high-water mark for the old pipeline.
- **TM100 (run8) is mid-pack** on signal (IC 0.035, RankIC 0.032) — the
  wider universe finds *more* signal in aggregate but with noisier
  per-stock estimates than TW50.
- **run9 (1-day label) lands at IC = 0.046 / RankIC = 0.067** — would
  rank mid-pack against the 20-day-label runs, but the comparison is
  not fair (see the horizon warning above). Per-step IC during live-IL
  is also more volatile — 6 of run9's 28 IL steps had negative IC,
  vs 2 of run7's 28 steps.
- **run10 (BOTH universe, 1-day label) has the weakest IC of the
  paper-faithful pair** (0.026 vs run9's 0.046) but the highest
  RankICIR (0.41 — beats run9's 0.33). The wider 150-stock universe
  finds *more rank-consistent* signal but smaller absolute correlations
  — same pattern as run8 vs run7 (TM100 vs TW50) under the 20-day
  label.

### Excess return vs 0050 (all negative — context matters)
- Everyone loses to 0050 on raw excess return, but the *gap* is the
  story:
  - **run4** (TW50-era, IL warmup) is closest to 0050 (only −3.5% ann)
  — it caught more of the 2024 TSMC rally because the curated 150-stock
  pool was TSMC-heavy.
  - **run1** is a catastrophe (−101.6% ann excess) — its signal was
  inverted (negative IC) so the strategy was systematically short the
  market direction.
- IR ranks runs by *consistency* of underperformance — closest to 0 is
  best:
  - **run4** has IR = −0.11 (essentially noise-level underperformance)
  - **run7** has IR = −1.32 (consistent underperformance — the model is
  good at picking stocks but the picks systematically miss 0050's
  TSMC-driven beta)

### Drawdown (where the new pipeline shines)
- **run9 and run10 tie at the new best max-DD ≈ −14%** (run9 −14.07%,
  run10 −14.01%) — 4 pp tighter than run7 (−18.1%). Both are 1-day-label
  live-IL runs; the universe (TW50 vs BOTH) barely moves the drawdown.
  Tighter labels → the adapter's per-task `θ` doesn't have to commit
  to 20-day directional views, so position sizing is more reactive.
- **run7 / run6 still lead in absolute return** at −18% / −19% MDD with
  cum +81% / +94%, vs run9's +57%. The 20-day-label runs commit to
  larger bets that pay off in a trending market.
- Excess max DD is best at run6 (−0.48) — i.e., the *cumulative-excess*
  curve never falls more than 48 percentage points below its peak
  relative to 0050. Run9's excess MDD (−0.69) is mid-pack despite the
  best raw MDD, because 0050's own drawdowns are smaller than run9's
  raw position drawdowns at the days that matter most.

### run7 vs run9 — the only apples-to-apples (controlled-experiment) comparison

| | run7 | run9 | Δ |
|---|---:|---:|---:|
| Universe | TW50 | TW50 | — |
| Mode | live-IL | live-IL | — |
| **Label horizon** | **20d** | **1d** | the change |
| Pretrain best val IC | 0.019 | **0.051** | +2.7× (1d val IC is paradoxically higher) |
| IL overall IC | 0.086 | 0.046 | −47% (expected — 1d IC is structurally smaller) |
| IL overall RankIC | 0.094 | 0.067 | −29% |
| Cum return | **+81.3%** | +56.7% | −24.6 pp |
| qlib Sharpe | **+1.38** | +1.24 | −0.14 |
| **Max DD** | −18.1% | **−14.1%** | **+4.0 pp (smaller drawdown)** |
| qlib excess ann ret | −21.6% | −28.8% | −7.2 pp (worse vs 0050) |
| qlib IR | −1.32 | −1.69 | −0.37 |
| Trades | 1116 | 1128 | +12 (~same) |

Reading: switching to the paper-faithful 1-day label gives a **tighter
drawdown** and **higher pretrain val IC**, but loses absolute return
and IR. The cost is real — 24.6 pp of cum return — and trades off
against deploy-honesty. **For live deployment, run9 is the more
defensible model** (matches paper spec, smaller drawdown, no IL leakage
since 1-day labels are fully observable the next day).

### run9 vs run10 — universe ablation under the 1-day-label regime

| | run9 | run10 | Δ |
|---|---:|---:|---:|
| Label horizon | 1d | 1d | — |
| **Universe** | **TW50 (~50/period)** | **BOTH (~150/period)** | the change |
| Mode | live-IL | live-IL | — |
| Pretrain best val IC | 0.051 | 0.041 | −0.010 (TW50 actually finds *more* pretrain signal here) |
| IL overall IC | **0.046** | 0.026 | −0.020 (wider universe halves IC, as in run7 vs run8) |
| IL overall RankIC | 0.067 | 0.059 | −0.008 |
| IL overall RankICIR | 0.326 | **0.414** | +0.088 (BOTH ranks more consistently — biggest RankICIR of any run) |
| Cum return | **+56.7%** | +46.4% | −10.3 pp |
| qlib Sharpe | **+1.24** | +0.99 | −0.26 |
| Max DD | −14.07% | **−14.01%** | +0.06 pp (statistically identical) |
| Win rate | **59.4%** | 57.9% | −1.5 pp |
| Profit factor | **2.10** | 1.79 | −0.31 |
| Trades | 1128 | 1126 | −2 (~same) |
| Wall-time | 9 min | **85 min** | +76 min (BOTH is ~9× slower to train) |

Reading: the BOTH-universe ablation is **strictly worse than TW50
on every operational metric** except RankICIR (which improves) and
max-DD (which is a tie). Same finding as run7 vs run8 (TW50 vs TM100)
under the 20-day label, now confirmed under the 1-day label too:
**wider universes don't help when the strategy picks top-K**.

The mechanism is consistent: a noisier per-stock IC dilutes the
top-K selection. The wider universe gives more candidates to *exclude*,
not more confident picks to *include*.

## How to regenerate this table

```powershell
# 1) Refresh 0050 benchmark (yfinance, auto-adjusted)
.venv/Scripts/python.exe -c @"
import yfinance as yf, pandas as pd
df = yf.download('0050.TW', start='2023-12-01', end='2026-05-31',
                 auto_adjust=True, progress=False).dropna()
df.columns = ['close','high','low','open','volume']
df.reset_index().rename(columns={'Date':'date'}).to_csv(
    'src/_cache/0050_benchmark.csv', index=False)
"@

# 2) Recompute metrics (reads each run's equity.csv + report.json)
.venv/Scripts/python.exe src/evaluate/compute_qlib_metrics.py
```

(The compute script is small — it reads `equity.csv` per run, aligns to
the benchmark by trade date, and applies the qlib formulas above.)

## Caveats

1. **Run1-5 equity.csv was rebuilt by the cross-run replay**, not their
   original (deleted) backtest engine. Their predictions.csv is original
   but the equity curve reflects the *current* TEJ-adjusted bar cache
   replayed through the new qlib engine. This is fine for fair
   cross-run comparison but it's **not** what the runs originally
   computed during training.
2. **All 10 runs are evaluated on the same 2024-01 → 2026-04 operational
   window** even though runs 4-6 also trained the IL on 2022-2023. The
   2022-2023 inner-loop window is not reflected in this table — see
   [model_env.md](model_env.md) for the full IL view.
3. **The benchmark is dividend-adjusted close, not total-return-index.**
   Yfinance's `auto_adjust=True` adjusts for splits and dividends, so
   `close[t] / close[0] − 1` already includes reinvested distributions.
   This matches how the strategy equity is measured (no separate
   dividend handling).
4. **No risk-free rate in the IR.** qlib's `risk_analysis` doesn't
   subtract a risk-free rate from excess; it's pure long-short of
   strategy vs benchmark. If you wanted a Sharpe-style metric you'd
   subtract TWSE 1Y bill ~1.5%/yr, which would shift IR magnitudes by
   `1.5% / 25% ≈ 0.06` per year — too small to change rank order.

## Related

- [model_env.md](model_env.md) — what changed in each run (run1 → run8 timeline)
- [qlib_backtest.md](qlib_backtest.md) — the backtest engine used to
  generate every `equity.csv` in this table
- [course_backtest.md](course_backtest.md) — the zero-cost course
  framework (different convention, not qlib-comparable)
