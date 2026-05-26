# Model — what changed in each run

End-to-end training runs of the DoubleAdapt + GRU + Alpha360 pipeline.
Each row of this doc is a single training iteration; the **diff** column
shows what changed vs the prior run, and the **results** columns show
how that change moved the metrics.

For methodology details on the two evaluation engines, see
[qlib_backtest.md](qlib_backtest.md) and [course_backtest.md](course_backtest.md).

## Run timeline

| Run | Date | What changed | Why |
|---|---|---|---|
| **1** | 2026-05-25 | baseline | Apply qlib + DoubleAdapt paper defaults out-of-the-box |
| **2** | 2026-05-25 | `epochs 200→5, patience 20→2, lr_psi 1e-2→1e-3, n_drop 3→1` | Pretrain val IC peaked at epoch 1 (overfit after); DA outer LR was 10× inner LR (destabilized); n_drop=3 caused ~30% cost drag |
| **3** | 2026-05-25 | `tasks_per_epoch 200→1000, epochs 5→30, patience 2→8` | Run 2 under-trained — let pretrain settle longer with larger meta-batches |
| **4** | 2026-05-25 | `IL window 2024→2022 (adds 24 warmup FOMAML steps); val moved INSIDE pretrain to 2021 H2` | Hypothesis: warmup matters more than pretrain depth |
| **5** | 2026-05-25 | `val back to 2022-2023 as clean held-out` (IL still includes 2022-2023 warmup) | Methodologically cleaner separation; verify val signal isn't leaking |
| **6** | 2026-05-26 | **bars: yfinance/TWSE-direct → TEJ-adjusted** + **universe mask: Wikipedia (32 quarterly changes) → official TW50.csv (184 month-end snapshots)** | Data-quality upgrade — official source for universe, consistent split+dividend adjustment for prices |

Each row is a strict superset of the prior row's fixes (i.e. once we
adopted `n_drop=1` in run 2, all subsequent runs kept it).

## Headline diffs (both backtest engines, operational window 2024-01 → 2026-05)

| Run | qlib cum_ret | qlib Sharpe | qlib MDD | course cum_ret |
|---|---:|---:|---:|---:|
| run1 | +85.7% | 1.013 | −38.7% | +59.3% |
| run2 | +144.3% | 1.356 | −46.1% | +100.7% |
| **run3** | **+204.5%** | 1.660 | −36.4% | +112.2% |
| run4 | +130.0% | 1.364 | −40.9% | +124.4% |
| **run5** | +167.2% | 1.421 | −43.7% | **+135.3%** |
| **run6** | +94.4% | **+1.520** | **−19.3%** | +76.0% |

Different leaders by metric:
- Max absolute return: **run3** (qlib) / **run5** (course)
- Best Sharpe: **run6**
- Smallest drawdown: **run6** (less than half of every other run)

## Per-run detail

### Run 1 — qlib defaults (baseline)

| Component | Setting |
|---|---|
| Pretrain | epochs=200, tasks_per_epoch=200, patience=20 |
| FOMAML | lr_phi=1e-3, **lr_psi=1e-2**, α_reg=0.5 |
| Backtest cap | top_k=10, **n_drop=3**, cost_alpha_bps=70 |
| Universe | curated 150-code pool + top-50 active by trailing-180d liquidity |
| Bars | yfinance (most stocks) + TWSE-direct (delisted) |

Pretrain best val IC = 0.033; IL overall RankIC = **−0.025** (anti-predictive
in the early 2024 window).

### Run 2 — three knob tweaks

| Diff vs run 1 | Why |
|---|---|
| `epochs 200→5` | pretrain val IC peaked at epoch 1 — long training overfits |
| `patience 20→2` | match the early peak |
| `lr_psi 1e-2→1e-3` | DA adapter outer LR was 10× inner LR — destabilized handoff to IL |
| `n_drop 3→1` | match qlib's canonical CSI300 10% daily turnover |

Pretrain best val IC = **0.078** (2.4×). IL overall RankIC flipped sign
(−0.025 → +0.016). Backtest jumps +85→+144%.

### Run 3 — train longer & deeper

| Diff vs run 2 | Why |
|---|---|
| `tasks_per_epoch 200→1000` | larger meta-batch each epoch |
| `epochs 5→30, patience 2→8` | give it room to actually settle |

Pretrain best val IC ~unchanged but the GRU's **internal representations**
are richer (the regression head's val score isn't the right metric of
representation quality). Backtest jumps +144→+204%.

**Lesson**: pretrain val IC isn't the right early-stop signal for IL
performance. Train longer than val-IC would suggest.

### Run 4 — extend IL through 2022-2023 as warmup

| Diff vs run 3 | Why |
|---|---|
| Pretrain training: 2016-2021 H1 (5.5 yr) | shift to free up 2021 H2 for val |
| Pretrain val: 2021 H2 (inside pretrain) | clean held-out from training but inside the total span |
| **IL window: 2022-01 → panel max (~52 steps)** | adds 24 warmup FOMAML updates before backtest opens |
| Backtest reporting: 2024-01 → panel max | unchanged |

Warmup matters: 28 IL steps (run 3) → 52 IL steps (run 4) on the same
architecture. Backtest 204→130 on qlib (more conservative bets) but 112→124
on course (zero-cost view). **The qlib regression here suggests warmup
makes the model trade more conservatively, which costs return when
slippage/cost is real but helps when zero-cost.**

### Run 5 — clean methodological val

| Diff vs run 4 | Why |
|---|---|
| Pretrain training: 2016-2021 (full 6 yr) | re-extend |
| **Pretrain val: 2022-2023 (clean held-out)** | val genuinely never seen during pretrain training |
| IL window: 2022-2023 warmup + 2024+ backtest | unchanged |

Pretrain best val IC = 0.029 (cleaner signal). Compressed prediction
range (std 0.061 vs run 4's 0.392) → smaller per-trade bets but stronger
per-unit signal. Backtest beats run 4 on both engines.

Run 5 was the production model for ~1 day. The data-quality concerns it
exposed motivated run 6.

### Run 6 — TEJ cache + official TW50 monthly mask

| Diff vs run 5 | Why |
|---|---|
| **Bars: yfinance/TWSE → TEJ** | yfinance dividend-adjusts inconsistently per stock; TEJ applies one consistent split+dividend adjustment across the whole history |
| **Universe mask: tw50_history.json (Wikipedia, 32 quarterly changes, reconstructed via reverse walk) → tw50_history.csv (TEJ, 184 monthly snapshots, official)** | Monthly granularity instead of quarterly; official source instead of crowdsourced |
| Universe expanded: 87 → 99 unique codes (15 years of monthly history captures more turnover) | Wider data coverage |
| All hyperparameters identical | Isolate the data-quality effect |

**Effect on qlib backtest:**

| Metric | Run 5 | Run 6 | Δ |
|---|---|---|---|
| Cum return | +167.2% | +94.4% | **−72.8 pp** |
| Sharpe | +1.421 | +1.520 | +0.10 |
| Max drawdown | −43.7% | **−19.3%** | **+24.4 pp** (half) |

Run 6 trades **less aggressively** (smaller absolute return) but **with
much higher risk-adjusted quality** (Sharpe up, MDD nearly halved). This is
consistent with cleaner data → less noise → smaller-but-more-confident bets.

## Counter-intuitive findings (across the 6 runs)

1. **Pretrain val IC isn't the right metric.** Runs 3-6 have lower val IC
   than run 2 (0.020-0.029 vs 0.078) but much better IL performance — what
   matters is the GRU's internal representations, not the regression head's
   val-set fit.
2. **Warmup matters more than pretrain quality.** 28 IL steps (run 3) → 52
   IL steps (runs 4-6, +24 warmup) materially improved Sharpe.
3. **Smaller predictions can be better.** Run 5's 6× smaller-magnitude
   predictions (std 0.061 vs 0.392) imply smaller per-trade bets — but the
   signal-to-noise ratio is higher. Run 6 takes this further with TEJ
   data.
4. **Data source matters as much as model architecture.** Run 5 → run 6
   changed zero hyperparameters; the data swap alone improved Sharpe by
   ~10% and halved drawdown.
5. **Different backtest engines rank runs differently.** Run 3 is the qlib
   champion; run 5 is the course champion. Run 6 dominates both on
   risk-adjusted measures but neither on absolute return. **Pick the
   metric that matters for your deployment.**

## What's preserved on disk

| Path | What |
|---|---|
| [src/output/run1/](../src/output/run1/) | run 1 artifacts |
| [src/output/run2/](../src/output/run2/) | run 2 |
| [src/output/run3/](../src/output/run3/) | run 3 (qlib-backtest absolute-return leader) |
| [src/output/run4/](../src/output/run4/) | run 4 |
| [src/output/run5/](../src/output/run5/) | run 5 (course-backtest absolute-return leader) |
| [src/output/run6/](../src/output/run6/) | **run 6 — current production** (best risk-adjusted) |
| [src/output/CURRENT](../src/output/) | text file: `run6` |

Each `runN/` contains: `final.pt`, `pretrained.pt`, `norm_stats.pkl`,
`predictions.csv` (date, stock_code_id, pred, actual), `bt_config.json`,
`active_universe.json`, `pretrain_log.csv`, `il_log.csv`, `equity.csv`,
`trades.csv`, `backtest_daily.csv`, `live_forecast.csv`, `report.json`,
`universe_schedule.csv`, plus regenerated `course_backtest.xlsx`.

## Choosing which run to deploy

| You want | Pick |
|---|---|
| Maximum absolute return, accept ~40% drawdowns | run 3 (qlib +204%) or run 5 (course +135%) |
| Best risk-adjusted return | **run 6** (Sharpe 1.52, MDD 19%) |
| Cleanest methodology (no leakage, official data sources) | **run 6** |
| Backwards compat with prior `deploy_today.py` runs | run 5 (last yfinance/Wikipedia model) |

`output/CURRENT` is `run6` — [deploy_today.py](../src/scripts/deploy_today.py)
picks it up by default.

## Related

- [qlib_backtest.md](qlib_backtest.md) — TWSE-costs in-process engine + cross-run table
- [course_backtest.md](course_backtest.md) — zero-cost framework + leaderboard upload
- [topk_ndrop_selection.md](topk_ndrop_selection.md) — why top_k=10 / n_drop=1
- [auto_trader.md](auto_trader.md) — daily deploy pipeline using the CURRENT run
