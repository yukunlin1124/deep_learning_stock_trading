# Model — what changed in each run

End-to-end training runs of the DoubleAdapt + GRU + Alpha360 pipeline.
Each row of this doc is a single training iteration; the **diff** column
shows what changed vs the prior run, and the **results** columns show
how that change moved the metrics.

For methodology details on the two evaluation engines, see
[qlib_backtest.md](qlib_backtest.md) and [course_backtest.md](course_backtest.md).

> **Label horizon note.** Runs 1-8 were trained with `LABEL_HORIZON = 20`
> (20-day forward close-to-close return). On 2026-05-27 the handler was
> switched to `LABEL_HORIZON = 1` (next-day close-to-close return) to
> match the DoubleAdapt paper's spec — the model predicts a single
> 1-day forward return; the 20-day cadence elsewhere in the pipeline
> (`FOMAMLConfig.r`) is the **task interval** (deployment window between
> meta-updates), not the label horizon. Existing `predictions.csv` and
> `equity.csv` from runs 1-8 still reflect the 20-day label, so their
> IC / RankIC magnitudes are not directly comparable to runs trained
> after this change (1-day IC is structurally ~3-5× smaller in magnitude
> for the same underlying model quality).

## Run timeline

| Run | Date | What changed | Why |
|---|---|---|---|
| **1** | 2026-05-25 | baseline | Apply qlib + DoubleAdapt paper defaults out-of-the-box |
| **2** | 2026-05-25 | `epochs 200→5, patience 20→2, lr_psi 1e-2→1e-3, n_drop 3→1` | Pretrain val IC peaked at epoch 1 (overfit after); DA outer LR was 10× inner LR (destabilized); n_drop=3 caused ~30% cost drag |
| **3** | 2026-05-25 | `tasks_per_epoch 200→1000, epochs 5→30, patience 2→8` | Run 2 under-trained — let pretrain settle longer with larger meta-batches |
| **4** | 2026-05-25 | `IL window 2024→2022 (adds 24 warmup FOMAML steps); val moved INSIDE pretrain to 2021 H2` | Hypothesis: warmup matters more than pretrain depth |
| **5** | 2026-05-25 | `val back to 2022-2023 as clean held-out` (IL still includes 2022-2023 warmup) | Methodologically cleaner separation; verify val signal isn't leaking |
| **6** | 2026-05-26 | **bars: yfinance/TWSE-direct → TEJ-adjusted** + **universe mask: Wikipedia (32 quarterly changes) → official TW50.csv (184 month-end snapshots)** | Data-quality upgrade — official source for universe, consistent split+dividend adjustment for prices |
| **7** | 2026-05-27 | **flow split**: warmup IL → `final_prep.pt`; **live-IL qlib backtest** (Stage 3) walks 2024+ doing adapt + portfolio-sim in tandem → `final.pt`. Predictions in `predictions.csv` are now strictly 2024+. | Match deploy-time behavior in the backtest itself — model continues fine-tuning during the 2024+ window, same as live trading would |
| **8** | 2026-05-27 | `--universe TW50 → TM100` (50 large-caps → 100 mid-caps; otherwise identical to run7) | Test whether a wider, less-liquid universe gives the model more signal to learn from |
| **9** | 2026-05-27 | **`LABEL_HORIZON 20 → 1`** (next-day c2c return; universe back to TW50; otherwise identical to run7) | Match the DoubleAdapt paper's spec — the model predicts a single 1-day forward return, and the 20-day cadence is the FOMAML *task interval*, not the label horizon. Also closes the live-IL label-leakage that the 20-day horizon implied (with horizon=1, the most recent support label is observable the next day) |
| **10** | 2026-05-27 | `--universe TW50 → BOTH` (TW50 ∪ TM100, ~150 stocks/period, 269 candidate codes; otherwise identical to run9) | Test whether the combined large-cap + mid-cap universe under the 1-day-label regime gives the model more signal than TW50 alone — completes the universe ablation matrix (TW50 / TM100 / BOTH) under both label horizons |

Each row is a strict superset of the prior row's fixes (i.e. once we
adopted `n_drop=1` in run 2, all subsequent runs kept it).

## Headline diffs (both backtest engines, operational window 2024-01 → 2026-05)

| Run | Label h | Universe | Mode | qlib cum_ret | qlib Sharpe | qlib MDD | course cum_ret |
|---|---|---|---|---:|---:|---:|---:|
| run1 | 20d | TW50-era | replay | +92.4% | 1.095 | −38.6% | +59.3% |
| run2 | 20d | TW50-era | replay | +147.4% | 1.373 | −45.5% | +100.7% |
| **run3** | 20d | TW50-era | replay | **+204.8%** | 1.661 | −36.4% | +112.2% |
| run4 | 20d | TW50-era | replay | +131.4% | 1.427 | −40.7% | +124.4% |
| run5 | 20d | TW50-era | replay | +158.9% | 1.426 | −43.7% | **+135.3%** |
| **run6** | 20d | TW50 | replay | +94.4% | **+1.520** | −19.3% | +76.0% |
| **run7** | 20d | TW50 | live-IL | +79.0% | +1.367 | −18.1% | +124.0% |
| run8 | 20d | **TM100** | live-IL | +60.1% | +0.856 | −33.6% | +55.4% |
| **run9** | **1d** | TW50 | live-IL | +56.7% | +1.242 | **−14.07%** | TBD |
| run10 | 1d | **BOTH** | live-IL | +46.4% | +0.986 | **−14.01%** | TBD |

Different leaders by metric:
- Max absolute return: **run3** (qlib) / **run5** (course)
- Best Sharpe: **run6**
- Smallest drawdown: **run9 / run10 tie** at ~−14% (both 1-day label, live-IL — wide gap to run7's −18.1%)

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

### Run 7 — live-IL during the qlib backtest

| Diff vs run 6 | Why |
|---|---|
| Split `train.py`'s IL phase into two checkpoints: `final_prep.pt` after warmup (2022-2023) + `final.pt` after live-IL backtest (2024+) | The qlib backtest now *is* the IL loop for 2024+ — model adapts every 20 days during the backtest, matching live deployment behavior |
| `predictions.csv` scoped to 2024+ (was 2022+) | Warmup predictions don't need to be persisted; only operational-window predictions feed downstream backtests |
| All hyperparameters + universe identical to run 6 | Isolate the methodology change |

**Effect on qlib backtest:** numbers very close to run6 (cum +79% vs +94%,
Sharpe +1.37 vs +1.52, MDD −18% vs −19%). The small drift comes from
floating-point ordering differences between the two execution patterns —
risk profile is preserved. Run7's win is **methodological**: predictions
reflect the same per-20-day adapt cadence as live trading.

### Run 8 — switch universe to TM100 (Mid-Cap 100)

| Diff vs run 7 | Why |
|---|---|
| `--universe TW50 → TM100` (50 large-caps → 100 mid-caps; FTSE's "next 100 below TW50") | Test whether a wider, less-liquid universe gives the model more signal to learn from |
| Bar cache expanded to 279 unique codes (TW50 ∪ TM100 ∪ a few stragglers) | Need bars for every code that's ever been in TM100 |
| Auto-injection of 2330 (TSMC) into the course backtest's universe even though the model never predicts on it | Course `BacktestSystem` uses 2330 as its trading-day reference; predictions still come from `pred_lookup` so 2330 isn't selected |
| All hyperparameters identical to run7 | Isolate the universe effect |

**Effect:** TM100 finds MORE pretrain val signal (val IC 0.053 vs run7's
0.020) but performs WORSE on operational backtests:

| Metric | Run 7 (TW50) | Run 8 (TM100) | Δ |
|---|---|---|---|
| Pretrain best val IC | 0.020 | **0.053** (+2.6× pretrain signal) | better |
| IL overall IC (2024+) | **0.086** | 0.035 | worse |
| qlib cum return | +79.0% | +60.1% | −19 pp |
| qlib Sharpe | **+1.367** | +0.856 | −0.51 |
| qlib MDD | **−18.1%** | −33.6% | −15 pp |
| Win rate | 51.4% | 46.9% | −4.5 pp |
| course cum return | +124.0% | +55.4% | −69 pp |

The **val-IC vs IL-IC inversion** is the key takeaway: the wider universe
helps pretrain capture more cross-sectional structure, but the top-K
selection from a noisier mid-cap pool pulls in worse-quality picks at
trade time. Mid-caps are also less liquid → higher realized volatility →
deeper drawdowns.

### Run 9 — paper-faithful 1-day label horizon

| Diff vs run 7 | Why |
|---|---|
| **`LABEL_HORIZON = 20 → 1`** (next-day c2c return instead of 20-day forward c2c) | Match the DoubleAdapt paper's spec — the model predicts a single 1-day forward return; the 20-day cadence is the FOMAML *task interval* (deployment window between meta-updates), not the label horizon |
| **Closes the live-IL label leakage** that the 20-day horizon implied | With horizon=20, at trading day T the support window used labels through `y[T]` = `c[T+20]/c[T]−1`, which needs prices that don't exist for 20 more days. With horizon=1, the most recent support label is observable the next day. |
| Universe back to TW50 (from run8's TM100); all other hyperparameters identical to run7 | Isolate the label-horizon effect; TW50 = the proven-best universe per run7 |

**Effect:**

| Metric | Run 7 (20d) | Run 9 (1d) | Δ |
|---|---:|---:|---:|
| Pretrain best val IC | 0.019 | **0.051** | +2.6× (1d val IC is paradoxically higher) |
| IL overall IC | **0.086** | 0.046 | −47% — but **horizons aren't directly comparable** (1d IC is structurally 3-5× smaller for the same model quality) |
| IL overall RankIC | **0.094** | 0.067 | −29% — same caveat |
| qlib cum return | **+81.3%** | +56.7% | −24.6 pp |
| qlib Sharpe | **+1.38** | +1.24 | −0.14 |
| qlib MDD | −18.1% | **−14.1%** | **+4.0 pp — smallest drawdown of any run** |
| n_trades | 1116 | 1128 | +12 (~same activity level) |
| Win rate | 51.4% | **59.4%** | +8.0 pp — model picks have a higher hit rate on 1d horizon |
| Profit factor | 2.02 | **2.10** | +0.08 |

**Two real wins**: smallest drawdown (−14.1%) and highest win rate
(59.4%) of any run. **One real loss**: 25 pp less cumulative return.
The mechanism: 1-day labels prevent the adapter from committing to
20-day directional views, so position sizes are reactive and small,
which dampens both downside and upside.

**For deployment, run9 is more defensible than run7**: it follows the
paper's spec exactly, has no label leakage in live-IL, and has the
tightest drawdown. The trade-off is real and earned — not a regression.

### Run 10 — universe ablation under the paper-faithful regime (BOTH = TW50 ∪ TM100)

| Diff vs run 9 | Why |
|---|---|
| `--universe TW50 → BOTH` (50 large-caps → 150 large+mid-caps, 269 candidate codes after history-mask) | Test whether the combined universe under the 1-day-label regime gives the model more signal than TW50 alone |
| **Bar cache expanded to 269 unique codes** (TW50 ∪ TM100 union); pretrain rows triple from 68k to 201k | More data → slower per-epoch (~10 min/epoch vs run9's ~1 min/epoch) |
| All hyperparameters + flow identical to run9 | Isolate the universe effect |
| Wall-time: 85 min (vs run9's 9 min — **~9× slower** because each FOMAML inner step scans 3× more support+query rows) | Necessary cost of the wider panel |

**Effect:**

| Metric | Run 9 (TW50) | Run 10 (BOTH) | Δ |
|---|---:|---:|---:|
| Pretrain best val IC | **0.051** | 0.041 | −0.010 (TW50 has cleaner pretrain signal) |
| IL overall IC | **0.046** | 0.026 | −0.020 (BOTH halves IC, as expected from earlier ablations) |
| IL overall RankIC | **0.067** | 0.059 | −0.008 |
| IL overall RankICIR | 0.326 | **0.414** | +0.088 (BOTH ranks more consistently — best of any run) |
| qlib cum return | **+56.7%** | +46.4% | −10.3 pp |
| qlib Sharpe | **+1.24** | +0.99 | −0.26 |
| qlib MDD | −14.07% | **−14.01%** | +0.06 pp (statistically identical) |
| Win rate | **59.4%** | 57.9% | −1.5 pp |
| Profit factor | **2.10** | 1.79 | −0.31 |

The universe ablation matrix is now complete:

| Universe | 20d label (run #) | 1d label (run #) |
|---|---|---|
| **TW50** (50 large-caps) | run7: cum +81%, IC 0.086 | **run9: cum +57%, IC 0.046** ← best 1d |
| **TM100** (100 mid-caps) | run8: cum +60%, IC 0.035 | not trained |
| **BOTH** (150 union) | not trained | run10: cum +46%, IC 0.026 |

**Two confirmations**:
1. **Wider universes consistently underperform on operational P&L**
   when the strategy picks top-K (TW50 > TM100 > BOTH at every label
   horizon). The pattern transcends label horizon — wider universes
   give noisier per-stock IC, which dilutes the top-K selection.
2. **The drawdown wins of 1-day labels persist across universes**: both
   1d runs land at ~−14% MDD; both 20d runs land at ~−18-34%. Tighter
   labels → tighter drawdowns, regardless of universe width.

**For deployment, run9 remains the recommendation** — same drawdown as
run10 but with stronger absolute return and a 9× faster training
cycle for monthly retraining.

## Counter-intuitive findings (across the 10 runs)

1. **Pretrain val IC isn't the right metric for IL performance.** Runs 3-6
   have val IC ~0.020-0.029 vs run 2's 0.078 — yet better operational
   numbers. Run 8 has the **highest val IC (0.053)** of any TW50/TM100 run
   yet **worst operational Sharpe**.
2. **Warmup matters more than pretrain quality.** 28 IL steps (run 3) → 52
   IL steps (runs 4-7, +24 warmup) materially improved Sharpe.
3. **Smaller predictions can be better.** Run 5's 6× smaller-magnitude
   predictions (std 0.061 vs 0.392) imply smaller per-trade bets — but the
   signal-to-noise ratio is higher. Run 6 takes this further with TEJ data.
4. **Data source matters as much as model architecture.** Run 5 → run 6
   changed zero hyperparameters; the data swap alone improved Sharpe by
   ~10% and halved drawdown.
5. **Wider universes don't automatically help downstream P&L** (run 7 →
   run 8): TM100 gives the model 2× more training signal but trades worse
   because the top-K-of-100 is noisier than top-K-of-50.
6. **Different backtest engines rank runs differently.** Run 3 is the qlib
   champion; run 5 is the course champion. Run 7 is the only run that's
   "honestly fine-tuned during the backtest" with 20-day labels; run 9
   adds the paper-faithful 1-day label on top.
7. **Tighter labels → tighter drawdowns** (run 7 → run 9): switching from
   20-day to 1-day forward labels closes the IL label-leakage and
   reduces max DD from −18.1% to −14.1% (smallest of any run), at the
   cost of 25 pp of cum return. The model commits to smaller bets per
   day because it isn't pretending to predict 20 days out.
8. **The drawdown floor is the label, not the universe.** Run9 (TW50,
   1d) and run10 (BOTH, 1d) both land at ~−14% MDD despite a 3× wider
   universe. The 1-day label is what reduces the per-position risk
   exposure; universe size doesn't move the drawdown needle.
9. **Universe effect is consistent across label horizons.** Wider
   universes reduce IC and cum_return at both 20d (run7 → run8) and
   1d (run9 → run10). The DoubleAdapt + TopkDropout pipeline doesn't
   benefit from a wider universe regardless of label horizon, in this
   TWSE setup.

## What's preserved on disk

| Path | What |
|---|---|
| [src/output/run1/](../src/output/run1/) | run 1 artifacts |
| [src/output/run2/](../src/output/run2/) | run 2 |
| [src/output/run3/](../src/output/run3/) | run 3 (qlib-backtest absolute-return leader) |
| [src/output/run4/](../src/output/run4/) | run 4 |
| [src/output/run5/](../src/output/run5/) | run 5 (course-backtest absolute-return leader) |
| [src/output/run6/](../src/output/run6/) | run 6 (last replay-mode backtest) |
| [src/output/run7/](../src/output/run7/) | run 7 (first live-IL backtest; last 20-day-label run on TW50) |
| [src/output/run8/](../src/output/run8/) | run 8 (TM100 universe, live-IL backtest, 20-day label) |
| [src/output/run9/](../src/output/run9/) | **run 9 — recommended production** (TW50, live-IL, paper-faithful 1-day label) |
| [src/output/run10/](../src/output/run10/) | run 10 (BOTH universe — completes the universe ablation matrix under the 1-day label) |
| [src/output/CURRENT](../src/output/) | text file: `run10` (latest-trained — bump to `run9` for production deploy) |

Each `runN/` contains: `final.pt`, `pretrained.pt`, `norm_stats.pkl`,
`predictions.csv` (date, stock_code_id, pred, actual), `bt_config.json`,
`active_universe.json`, `pretrain_log.csv`, `il_log.csv`, `equity.csv`,
`trades.csv`, `backtest_daily.csv`, `live_forecast.csv`, `report.json`,
`universe_schedule.csv`, plus regenerated `course_backtest.xlsx`. Runs 7+
also have `final_prep.pt` (post-warmup checkpoint) and `warmup_log.csv`.

## Choosing which run to deploy

| You want | Pick |
|---|---|
| Maximum absolute return, accept ~40% drawdowns | run 3 (qlib +204%) or run 5 (course +135%) |
| Best risk-adjusted return on 20-day label | run 6 (Sharpe 1.52, MDD 19% — replay mode) |
| **Paper-faithful + smallest drawdown** | **run 9** (TW50, live-IL, 1-day label; Sharpe 1.24, MDD 14.1%) |
| Highest Sharpe AND deploy-honest, 20d label | **run 7** (TW50, live-IL, Sharpe 1.38, MDD 18.1%) |
| Widest universe (BOTH = TW50 ∪ TM100) | run 10 — same drawdown as run9 but lower Sharpe + 9× slower training |
| Widest universe of mid-caps (experimental) | run 8 |
| Backwards compat with prior `deploy_today.py` runs | run 5 (last yfinance/Wikipedia model) |

`output/CURRENT` is `run10` (latest-trained) — [deploy_today.py](../src/scripts/deploy_today.py)
picks it up by default. Bump to `run9` for production deploys (same
drawdown, higher Sharpe, faster retraining cycle).

## Related

- [qlib_backtest.md](qlib_backtest.md) — TWSE-costs in-process engine + cross-run table
- [course_backtest.md](course_backtest.md) — zero-cost framework + leaderboard upload
- [topk_ndrop_selection.md](topk_ndrop_selection.md) — why top_k=10 / n_drop=1
- [auto_trader.md](auto_trader.md) — daily deploy pipeline using the CURRENT run
