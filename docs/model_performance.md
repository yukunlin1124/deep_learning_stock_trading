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
model that pays TWSE round-trip costs is a *very* high bar; only **run3
clears it convincingly** (+7.7% excess ann ret, IR +0.34) and **run5 is
statistically break-even** (+0.01%, IR +0.001). Every other run loses
to the passive ETF on excess basis — most by 15-40 pp annualized. The
1-day-label runs (run9 onwards) consistently underperform on excess
return because they trade more conservatively; their value shows up in
drawdown (−14% vs run3's −33%) and risk-adjusted metrics (IC, RankIC).

## Cross-run table (vs 0050)

| Run | Label horizon | Universe | Mode | IC | ICIR | RankIC | RankICIR | Excess Ann Ret (w/ cost) | IR (w/ cost) | Max DD (w/ cost) | Excess Max DD (w/ cost) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| run1 | 20d | TW50-era | replay | −0.0105 | −0.0521 | −0.0249 | −0.1235 | −15.0% | −0.70 | −83.0% | −0.60 |
| run2 | 20d | TW50-era | replay | +0.0398 | +0.1882 | +0.0158 | +0.0835 | −1.7% | −0.08 | −58.9% | −0.60 |
| **run3** | 20d | TW50-era | replay | +0.0695 | +0.3655 | +0.0551 | +0.3083 | **+7.7%** | **+0.34** | −33.4% | **−0.44** |
| run4 | 20d | TW50-era | replay | +0.0134 | +0.0828 | +0.0150 | +0.0944 | −6.2% | −0.33 | −37.9% | −0.47 |
| run5 | 20d | TW50-era | replay | +0.0490 | +0.2346 | +0.0322 | +0.1650 | **+0.01%** | **+0.001** | −27.1% | −0.49 |
| **run6** | 20d | TW50 | replay | +0.0499 | +0.2608 | +0.0605 | +0.3363 | −16.3% | −1.01 | −19.3% | −0.48 |
| **run7** | 20d | TW50 | **live-IL** | **+0.0856** | **+0.4748** | **+0.0937** | **+0.5131** | −20.2% | −1.25 | −18.1% | −0.60 |
| run8 | 20d | **TM100** | **live-IL** | +0.0348 | +0.2763 | +0.0318 | +0.2553 | −23.2% | −1.04 | −31.4% | −0.76 |
| **run9** | **1d** | TW50 | **live-IL** | +0.0459 | +0.2058 | +0.0672 | +0.3260 | −28.8% | −1.69 | **−14.07%** | −0.69 |
| run10 | 1d | **BOTH** | live-IL | +0.0263 | +0.1767 | +0.0588 | **+0.4141** | −31.5% | −1.70 | **−14.01%** | −0.73 |
| run11 | 1d | TW50 | live-IL **no-CSRankNorm** | +0.0222 | +0.1043 | +0.0230 | +0.1113 | −40.3% | −1.53 | −39.9% | −1.01 |
| run12 | 1d | TW50 | live-IL **no best-state restore** | +0.0098 | +0.0517 | +0.0208 | +0.1207 | −32.3% | −1.95 | −31.7% | −0.87 |
| run13 | 1d | TW50 | live-IL **patience=16** (vs default 8) | +0.0459 | +0.2058 | +0.0672 | +0.3260 | −28.8% | −1.69 | −14.07% | −0.69 |
| run14 | 1d | TW50 | live-IL **r=5** (5-day task span) | +0.0450 | +0.1989 | +0.0652 | +0.3115 | −40.0% | −2.47 | −27.7% | −0.94 |

> **Table refreshed 2026-05-28**. Re-derived from the freshly-replayed
> qlib equity.csv for every run (181-code unified bars covering TW50 +
> TM100 + the TW50-era curated pool). Previous version had stale
> numbers for runs 1-5 because the earlier cross-run replay used a
> smaller bars set that skipped some TW50-era prediction codes — fixed.

**Bold** = best-in-column. Notably, **run3** is the only run with
positive excess return AND positive IR vs 0050; run5 is statistically
break-even (excess ann ret ≈ 0). Every other run loses to the
passively-held 0050 ETF in this window.

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

### Excess return vs 0050
- **run3** is the only run that *beats* 0050 on excess return (+7.7%
  ann ret, IR +0.34) — the TW50-era curated 150-stock pool with longer
  pretrain (run3's diff vs run2) found a signal that exceeds the
  passive benchmark even after TWSE costs.
- **run5** is statistically tied with 0050 (+0.01% ann, IR ≈ 0) — the
  clean held-out val variant of run4.
- **run1** is the worst (−15.0% ann excess) because its overall IC was
  *negative* (−0.0105) — the signal was inverted, so the strategy
  systematically picked the wrong side of every TSMC-led rally.
- Runs 6+ have negative excess (−16% to −40%) because TEJ-adjusted
  prices compress prediction magnitudes → smaller bets → less ride on
  the TSMC mega-rally that drove 0050. Trade-off documented in the
  drawdown column.
- The 1-day-label runs (run9, run10, run13) cluster at −29 to −32%
  excess — paper-faithful regime, but every gain in drawdown
  protection costs ~7-10 pp of annualized return vs 0050.

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

### run9 vs run11 — CSRankNorm ablation (the qlib stability fork)

Only difference: run11 strips `csrank_normalize_per_date(y)` before H, so
L_train, L_test, and L_reg all compare against raw next-day c2c returns
(paper-faithful, see [overview.md](overview.md) deviation table). LRs
unchanged (`lr_phi=lr_psi=1e-3`, `inner_lr=1e-3`).

| Metric | run9 (CSRankNorm) | run11 (raw y) | Δ |
|---|---:|---:|---:|
| Pretrain best val IC | **0.051** | 0.050 | −0.001 (Adam's per-param variance norm absorbs the 360× loss-scale shrink) |
| IL overall IC | **0.046** | 0.022 | **−52%** |
| IL overall RankIC | **0.067** | 0.023 | **−66%** |
| IL overall RankICIR | **0.326** | 0.111 | **−66%** |
| Cum return | **+56.7%** | +17.8% | −38.9 pp |
| qlib Sharpe | **+1.24** | +0.43 | −0.81 |
| **Max DD** | **−14.07%** | **−39.91%** | **−25.8 pp (much worse)** |
| Win rate | **59.4%** | 51.6% | −7.8 pp |
| Trades | 1128 | **566** | **half** — raw-y predictions in the ±0.01-0.05 range mostly fail to clear `cost_alpha_bps=70` (= 0.007 in pred space) |
| Excess Ann Ret vs 0050 | −28.8% | **−40.3%** | −11.5 pp |

**Why the regression is so large despite identical pretrain quality:**

1. **Pretrain still works.** Adam's per-parameter gradient-variance
   normalization absorbs the 360× loss-scale shrink, so val IC peaks
   nearly identically to run9 (0.050 vs 0.051).
2. **The FOMAML inner step collapses to a no-op.** The inner step is
   *plain SGD* — `θ = φ − inner_lr · ∇_φ L_train` with `inner_lr=1e-3`.
   With L_train ≈ 2.5e-4 (vs ≈ 1.0 in CSRankNorm mode), the gradient is
   ~360× smaller, so `θ ≈ φ` for every task. The bi-level meta-learning
   structure that DoubleAdapt depends on collapses to "pretrained model
   + small Adam fine-tuning on G + H". The model loses its key
   advantage of per-task adaptation.
3. **The threshold mismatch halves the trade count.** With raw-y
   predictions clustered in ±0.01-0.05, the `cost_alpha_bps=70`
   (= 0.007) threshold is now near the *median* of predictions instead
   of the ~60th percentile in CSRankNorm space. Half the days have no
   target above threshold, so positions persist longer and the
   portfolio doesn't react to changing market state — explains the
   wider drawdowns.

**Lesson confirmed**: the CSRankNorm deviation flagged in
[overview.md](overview.md) is the largest deviation from the paper in
this codebase, but removing it without rescaling LRs is **strictly
worse** than keeping it. A paper-faithful run requires (a) stripping
CSRankNorm AND (b) rescaling `inner_lr` upward by ~100-360× to keep
the inner SGD step meaningful AND (c) recalibrating `cost_alpha_bps`
for the raw-return prediction scale. Run11 is the "do (a) only" data
point — useful evidence, not a deploy candidate.

### run9 vs run12 — best-state restore ablation (does early-stop's "rollback to best epoch" matter?)

Only difference: run12 disables `model.load_state_dict(best_state)`
in [src/trainer/maml.py:237](../src/trainer/maml.py#L237), so warmup IL
starts from the **post-early-stop weights (epoch 9, val IC ≈ 0.012)**
instead of the **best-IC weights (epoch 1, val IC = 0.051)**. Same
seed, same data, same LRs — the pretrain trajectory is bit-identical
through epoch 9, only the handoff to Stage 2 changes.

| Metric | run9 (restore to best) | run12 (keep epoch-9 weights) | Δ |
|---|---:|---:|---:|
| Pretrain trajectory epochs 0-9 | identical | identical | seed = 0 |
| Pretrain best val IC | 0.0507 (used as Stage-2 init) | 0.0507 (logged but NOT used) | same |
| Pretrain epoch-9 val IC | 0.0117 (discarded) | 0.0117 (carried into Stage 2) | same |
| IL overall IC | **0.046** | 0.010 | **−78%** |
| IL overall ICIR | **0.206** | 0.052 | −75% |
| IL overall RankIC | **0.067** | 0.021 | −69% |
| IL overall RankICIR | **0.326** | 0.121 | −63% |
| Cum return | **+56.7%** | +42.0% | −14.7 pp |
| qlib Sharpe | **+1.24** | +0.82 | −0.42 |
| Max DD | **−14.07%** | **−31.66%** | **−17.6 pp (more than 2×)** |
| Win rate | **59.4%** | 54.9% | −4.5 pp |
| Profit factor | **2.10** | 1.70 | −0.40 |
| Trades | 1128 | 1124 | ~same |

**The early-stop + best-state-restore mechanism is worth ~15 pp of cum
return and halves max drawdown.** Stage 2 + 3 cannot fully recover
from a degraded pretrain init even with 53 rolling FOMAML updates
(25 warmup + 28 live-IL). The warmup loss curve makes this visible:

| Step | run9 warmup loss | run12 warmup loss |
|---|---:|---:|
| 0 | 1.001 | **1.426** |
| 5 | 0.981 | 1.474 |
| 10 | 1.012 | 1.318 |
| 15 | 0.992 | 1.181 |
| 20 | 0.993 | 1.128 |
| 24 (last) | 1.000 | ~1.115 |

Run12's warmup loss decays from 1.43 toward 1.12 over 25 steps — it's
*still* recovering from the degraded init when Stage 2 ends. Run9
starts already at the plateau (~1.00) and uses its 25 steps for actual
2022-2023 regime adaptation. By the time live-IL begins (Stage 3),
run9 has 25 steps of productive adapt; run12 has burned its budget on
recovery.

**Lesson confirmed**: the `load_state_dict(best_state)` line at
[src/trainer/maml.py:237](../src/trainer/maml.py#L237) is load-bearing
for downstream performance. The early-stop counter only matters
*because* of the restore — without it, early stopping degrades to
"stop training early at a worse checkpoint", which is strictly worse
than just letting it train. Skipping restore inflates max DD from
14% to 32% and halves the Sharpe-via-IC contribution.

### run9 vs run13 — patience ablation (8 → 16)

Only difference: `PretrainConfig.early_stop_patience` raised from 8
to 16, letting pretrain run 8 more non-improving epochs before
breaking. Same seed, same data, everything else identical.

| Metric | run9 (patience=8) | run13 (patience=16) | Δ |
|---|---:|---:|---:|
| Pretrain epochs run | 10 | **18** | +8 wasted epochs |
| Pretrain best val IC | 0.0507 @ ep1 | **0.0507 @ ep1** | same — peak never beaten |
| Wall-clock | 8.67 min | **14.62 min** | **+5.95 min wasted GPU time** |
| Best-state restored from | epoch 1 | epoch 1 | same checkpoint |
| IL overall IC | 0.0459 | **0.0459** | bit-identical |
| IL overall RankIC | 0.0672 | **0.0672** | bit-identical |
| Cum return | +56.73% | **+56.73%** | bit-identical |
| Sharpe | 1.242 | **1.242** | bit-identical |
| Max DD | −14.07% | **−14.07%** | bit-identical |
| Trades | 1128 | **1128** | bit-identical |
| Final equity | 156,725,275.95 | **156,725,275.95** | identical to the cent |

**Lesson**: `early_stop_patience` only matters if pretrain finds a NEW
val-IC peak during the extra epochs. For this seed/data combination,
val IC peaks at epoch 1 and never returns — confirmed across run9,
run10, and now run13. Raising patience from 8 to 16 burns 6 minutes
of GPU for **zero** downstream change. The current default of 8 is
already loose enough (no run since run6 has triggered early-stop with
a "still-improving-late" trajectory). A tighter `patience=4` would
save more time without any operational risk.

### run9 vs run14 — task-span ablation (r=20 → r=5)

Only difference: `FOMAMLConfig.r` set to 5 (4× shorter task span and
4× more rolling steps). Support window = 5 trading days × ~50 stocks
= ~250 samples per task (vs ~1000 with r=20). Warmup IL now runs 97
steps and live-IL 115 (vs 25 and 28).

| Metric | run9 (r=20) | run14 (r=5) | Δ |
|---|---:|---:|---:|
| **Wall-clock** | 8.67 min | **4.14 min** | **−52% (FASTER)** |
| Warmup IL steps | 25 | **97** | +4× |
| Live-IL steps | 28 | **115** | +4× |
| **Pretrain best val IC** | 0.0507 | **0.0561** | **+11%** (more diverse meta-batches/epoch) |
| IL overall IC | 0.046 | 0.045 | ~same (averaged over 4× more, individually noisier windows) |
| IL overall RankIC | 0.067 | 0.065 | ~same |
| Per-step IC variance | ±0.13 typical | **±0.28 typical** | 2× wider — 250-row IC estimates are inherently noisier than 1000-row |
| **Cum return** | **+56.7%** | +27.4% | **−29.3 pp** |
| **Sharpe** | **+1.24** | +0.65 | **−0.59 (halves)** |
| **Max DD** | **−14.07%** | **−27.71%** | **doubles** |
| Excess Ann Ret vs 0050 | −28.8% | **−40.1%** | −11.3 pp |
| IR vs 0050 | −1.69 | **−2.48** | further from 0 |
| n_trades | 1128 | 1157 | +29 (~same — n_drop=1 still caps daily turnover) |
| Win rate | 59.4% | 59.1% | ~same |
| Profit factor | 2.10 | 1.69 | −0.40 |

**Three counter-intuitive findings:**

1. **Smaller tasks train FASTER, not slower.** Despite 4× more IL
   steps, run14 finished in half the time of run9. Smaller per-task
   tensors (250 vs 1000 rows) make each FOMAML inner+outer pass
   cheaper enough to more than offset the 4× step count. Pretrain also
   benefits — same tasks_per_epoch=1000 but each task processes 4× less
   data per step.
2. **Smaller tasks find BETTER pretrain val IC.** 0.056 vs 0.051 — the
   meta-learner sees more diverse start dates per epoch (1000 random
   samples drawn from a much larger pool of valid r=5 windows than r=20
   windows), which improves the regression head's generalization.
3. **But the trading strategy is substantially worse.** Sharpe halves,
   MDD doubles, cum return drops 29 pp. The per-step IC is similar in
   *aggregate*, but the per-window IC variance doubles (±0.28 vs
   ±0.13) — predictions are 4× more reactive to short-term noise.
   Compounding the issue: the portfolio's frozen-prediction deployment
   window is now 5 days instead of 20, so predictions go stale 4× faster
   during the period the model is *not* adapting. Result: more
   aggressive position shifts on noisier signals → wider drawdowns,
   especially in choppy markets (April 2025 drawdown to NT$86M
   visible in the equity curve).

**Lesson**: task span trades off **meta-learner adaptation frequency**
against **per-task signal quality**. r=20 sits closer to the qlib
canonical CSI300 setup and produces the best risk-adjusted trading
even though pretrain val IC is slightly worse. Shortening to r=5
is tempting (faster training, higher pretrain IC) but the noisier
per-task predictions translate directly into worse Sharpe and MDD.
This matches the DoubleAdapt paper's choice of 20-day deployment
windows — the value comes from the *match* between deployment cadence
and prediction confidence horizon, not from adapting as frequently
as possible.

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
2. **All 14 runs are evaluated on the same 2024-01 → 2026-04 operational
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
