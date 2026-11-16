# Training Procedure Overview

End-to-end visual reference for how `train.py` produces a deployable
model. Combines the three-stage schedule, the DoubleAdapt architecture
(G / H / F / MA / DA), the rolling FOMAML task window, and the per-task
bi-level update into a single map. For per-run config diffs see
[model_env.md](model_env.md); for backtest engines see
[qlib_backtest.md](qlib_backtest.md) / [course_backtest.md](course_backtest.md);
for cross-run metrics see [model_performance.md](model_performance.md).

## TL;DR

One `python src/scripts/train.py --universe TW50` invocation executes:

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: OFFLINE PRETRAIN  (2016-01 → 2021-12,  val 2022-2023)     │
│    random task sampling, FOMAML bi-level, early-stop on val IC      │
│    ───────────────────────────────────────────────────────────────  │
│    saves pretrained.pt    (~10 min wall-clock for TW50 on RTX 5060) │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ φ, ψ
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: WARMUP IL  (2022-01 → 2024-01)                            │
│    25 rolling FOMAML steps (r=20-day cadence), real optim.step()    │
│    ───────────────────────────────────────────────────────────────  │
│    saves final_prep.pt  +  warmup_log.csv                           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ φ, ψ (warmer adapters)
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: LIVE-IL QLIB BACKTEST  (2024-01 → today)                  │
│    28 rolling FOMAML steps  +  portfolio simulation in tandem       │
│    ───────────────────────────────────────────────────────────────  │
│    saves final.pt + predictions.csv + equity.csv + report.json      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                  MLflow run runN in mlruns/    (deployable to deploy_today.py)
```

The split into three stages is operational, not architectural — the
*same* DoubleAdapt model carries its `(φ, ψ)` state straight through
all three. Stage 1 randomly samples from a fixed offline panel; Stages
2 + 3 walk a continuous rolling timeline (Algorithm 1's online phase),
with Stage 3 also running portfolio simulation.

## Data flow (cache → features → labels → tasks)

```
                TEJ half-year CSVs (cp950, data/raw/TEJ/)
                              │
                              ▼  src/scripts/import_tej.py
                  per-(stock, year) cache (data/interim/{code}_{year}.csv)
                              │
                              ▼  src/data/io.py: fetch_range
                  bars dict: {code: DataFrame[date, open, high, low,
                                              close, capacity, turnover]}
                              │
                              ▼  src/data/handler.py: compute_alpha360
                  per-(stock, date):
                    X = (60, 6) — 60-day lookback × 6 fields, each
                          normalized by its lag-0 (X[-1, :] = [1,1,1,1,1,1])
                    y = next-day c2c return (close[t+1] / close[t] − 1)
                              │
                              ▼  build_panel + apply_time_varying_universe
                  PanelTensors{X: (N, 60, 6), y: (N,), dates, codes}
                  restricted to TW50/TM100/BOTH membership-as-of(date)
                              │
                              ▼  fit_robust_zscore on pretrain rows only
                              ▼  fillna(apply_robust_zscore(X, stats))
                  Normalized panel (same shape, lag-0 → 0, clipped to ±3)
                              │
                              ▼  sample / walk tasks of length r=20
                              ▼  per task: (x_train, y_train, x_test, y_test)
                          → FOMAML step (see "Per-task" below)
```

`norm_stats` is persisted to `runN/norm_stats.pkl` so `deploy_today.py`
applies the exact same per-(lag, field) normalization that the model
trained on. The 6 fields are `[close, open, high, low, vwap, volume]`
in that order — vwap is computed as `turnover / capacity` (or `close`
if volume is zero).

## DoubleAdapt architecture

```
                              x ∈ R^(B, 60, 6)
                                       │
                                       ▼
         ┌─────────────────────────────────────────────────────────┐
         │   DA(ψ) =  G  ⊕  H                                       │
         │   ───────────────────────────────────────────────────    │
         │                                                          │
         │   G (FeatureAdapter):                                    │
         │     per-time-step cosine gate vs N=8 learned prototypes  │
         │     P ∈ R^(8, 6) → softmax(sims / τ) → multi-head        │
         │     residual: x_tilde = x + Σ_i s_i · head_i(x)          │
         │                                                          │
         │   H (LabelAdapter):                                      │
         │     gate(x_tilde) (cosine vs hidden prototypes) →        │
         │     per-head affine on y:  h_i(y) = γ_i · y + β_i        │
         │     (γ ~ U(0.75, 1.25), β = 1/8)                         │
         │     forward: ỹ = H(x_tilde, y); inverse: y = H⁻¹(x_tilde,│
         │     ỹ) — used at inference to map prediction back to     │
         │     real-world distribution                              │
         │                                                          │
         └─────────────────────────────────────────────────────────┘
                                       │
                                       ▼  x_tilde
         ┌─────────────────────────────────────────────────────────┐
         │   F(φ) = GRUBase                                         │
         │   ───────────────────────────────────────────────────    │
         │   GRU(input=6, hidden=64, num_layers=2)                  │
         │     → last hidden state                                  │
         │     → Linear(64, 1) → squeeze                            │
         │     → pred_raw ∈ R^B  (in CSRankNorm-scaled label space) │
         └─────────────────────────────────────────────────────────┘
                                       │
                                       ▼  pred_raw
                              H⁻¹(x_tilde, pred_raw)
                                       │
                                       ▼
                              ŷ ∈ R^B  (real-world prediction)
```

Param counts (default config): **DA (ψ) = 12,176** params (G's 6×6
head matrices + N=8 prototypes; H's hidden gate + per-head γ, β),
**MA / F base (φ) = 38,849** params (GRU weights). The base is the
thing FOMAML's inner step fine-tunes per task; DA is updated only by
the outer step.

> "MA" in the paper is a *concept* (a meta-learned initializer for F's
> params). In code there's no separate MA module — the meta-update IS
> the Adam step on `model.base` parameters in the outer loop. So when
> the diagram shows `MA(φ^(k−1)) → θ^k`, the implementation is the
> FOMAML inner step: `θ^k = φ^(k−1) − η_θ ∇_φ L_train`.

## Per-task: one FOMAML bi-level step

This is what runs *once per task* during Stage 1 (offline pretrain) and
*once per r=20-day window* during Stages 2 + 3 (warmup + live-IL). See
[src/trainer/maml.py:109-152](../src/trainer/maml.py#L109-L152) for the code.

```
                                        Inner / lower-level
                                      ──────────────────────
   x_train, y_train ──┐                       y_train
   (past r days)      │                          │
                      ▼                          ▼ CSRankNorm
                G(x_train) → x_train_tilde     y_train_norm
                      │                          │
                      ├──────────────────────────┤
                      ▼                          ▼
              F(φ)(x_train_tilde)          H(x_train_tilde, y_train_norm)
              = pred_train_raw             = y_train_tilde
                      │                          │
                      └──────► MSE(·, ·) ◄───────┘
                                  │
                                  ▼
                              L_train
                                  │
                                  ▼  ∇_φ L_train  (autograd.grad, detached)
                  θ^k = φ − η_θ · ∇_φ L_train          ← FOMAML inner step
                                  │
                                  │  (θ used functionally — model.base.φ untouched)
                                  │
                                       Outer / upper-level
                                      ──────────────────────
   x_test, y_test ──┐                         y_test
   (next r days)    │                           │
                    ▼                           ▼ CSRankNorm
              G(x_test) → x_test_tilde     y_test_norm
                    │                           │
                    ▼                           │
            functional_call(F, θ, x_test_tilde) │
              = pred_test_raw                   │
                    │                           │
                    ▼ H⁻¹                       │
              ŷ = pred_test                     │
                    │                           │
                    └──────► MSE(ŷ, y_test_norm) ─┴── + α · MSE(H(y), y_test_norm)
                                │                                                  │
                                ▼                                                  │
                            L_test  =  L_MSE  +  α · L_reg ◄────────────────────┘
                                │
                                ▼  L_test.backward()
                  optim_phi.step()  →  φ ← Adam(lr_phi=1e-3)   (updates F's GRU)
                  optim_psi.step()  →  ψ ← Adam(lr_psi=1e-3)   (updates G + H)
```

> **Terminology note.** Throughout this codebase, "train" / "test" refer
> to the per-task meta-train / meta-test split (the support / query sets
> in standard MAML literature) — NOT the project-level pretrain vs
> backtest windows. They share names because they play the same logical
> role inside each task: train is what the inner step adapts on, test
> is what the outer loss is evaluated against. The `train: bool` flag on
> `fomaml_step` is a separate concept — it controls whether the outer
> step actually backpropagates (`True` for training/IL passes, `False`
> for read-only val passes).

Three separate learning rates, all configurable via `FOMAMLConfig`.
Values shown are what `train.py` *actually passes* (which overrides the
dataclass defaults in maml.py — train.py line 242 sets `lr_psi=1e-3`):

| Rate | Symbol | train.py value | Updates |
|---|---|---:|---|
| `inner_lr` | η_θ | 1e-3 | The inner step θ = φ − η_θ ∇L_train (SGD, plain) |
| `lr_phi` | η_φ | 1e-3 | F (GRU base) outer step via Adam |
| `lr_psi` | η_ψ | **1e-3** (overridden from maml.py default 1e-2) | DA (G + H) outer step via Adam |
| `alpha_reg` | α | 0.5 | Weight of `L_reg = MSE(H(y), y)` in L_test |

The outer uses **Adam**, not plain SGD — substitution from the paper's
pseudocode that every published DoubleAdapt-style implementation makes.
The `lr_psi` override traces back to run2's finding that the
paper-default 10× ratio (lr_psi / lr_phi = 10) destabilized the
handoff from pretrain to IL (see [model_env.md](model_env.md)).

## Rolling task window (Stages 2 + 3)

```
   t=T+1      T+r      T+r+1     T+2r       T+2r+1   ...
   ├──────────┤├─────────────────┤├─────────────────┤
   │ D_train¹ ││     D_test¹     ││     D_test²     │
   │          ││     D_train²    ││     D_train³    │
   │          ││                 ││                 │
   └──────────┘└─────────────────┘└─────────────────┘

   train window of task k+1   ≡   test window of task k
   ─────────────────────────────────────────────────────────

   At trade-day D in Stage 3:
     - model state reflects everything adapted through D − r
     - next FOMAML step on train=[D−r, D−1] only fires once
       those labels are observable (1-day publish lag, since
       PRED_HORIZON=1)
     - within the r-day test window, the model is FROZEN and
       emits 20 sequential 1-day predictions
```

Verified against run9's `il_log.csv` (an MLflow artifact): for
all 28 consecutive live-IL tasks, `train_start[k+1] == test_start[k]`
holds — chained rolling, no overlap, no gap. Same for the 25 warmup
tasks in run9's `warmup_log.csv`.
(Earlier runs use the legacy `support_*` / `query_*` column names —
re-running training under the renamed code emits the new column names.)

Stage 1 (pretrain) is the exception: tasks are sampled *randomly* from
the 2016-2021 panel with replacement (`tasks_per_epoch=1000` per
epoch, ~1460 valid start dates). No chaining — pretrain teaches the
model the task structure across many independent starts.

## Stage 1 outer loop (pretrain with early stop)

```
   for epoch in range(epochs=30):
       │
       ├─ sample 1000 random tasks from [2016-01, 2021-12]
       │
       ├─ for each task:
       │     FOMAML step (train=True)  →  L_test.backward()
       │     optim_phi.step();  optim_psi.step()
       │
       ├─ build val task set from [2022-01, 2023-12]
       │
       ├─ for each val task:
       │     FOMAML step (train=False)  ← NO L_test.backward()
       │                                   so meta-learners untouched
       │     collect pred_test for IC computation
       │
       ├─ val_ic = cross_sectional_ic(preds, ys, dates)
       │
       ├─ if val_ic > best_val_ic + 1e-6:
       │       best_state = clone(model.state_dict())
       │   else:
       │       bad_epochs += 1
       │       if bad_epochs >= 8:
       │           break                ← EARLY STOP
       │
   model.load_state_dict(best_state)    ← roll back to best val IC
   save pretrained.pt
```

The val pass implements Algorithm 1 line 5 ("make a temporary copy of
φ and ψ") via the `train=False` flag instead of an explicit deepcopy.
Functionally equivalent — the actual model.base + DA params aren't
mutated by val tasks.

## Stage 2 + 3 outer loop (online — Algorithm 1 line 9)

```
   i = first_idx_in_window
   while i + r <= len(udates):
       task = panel_to_task(panel, i − r, r)
       │
       ├─ FOMAML step (train=True)  →  L_test.backward()
       │  optim_phi.step();  optim_psi.step()    ← real updates this time
       │
       ├─ collect (date, code, pred, actual) for predictions.csv
       │
       ├─ STAGE 3 ONLY: for each day in test window:
       │     pick top-K above threshold,
       │     simulate sells / buys (TopkDropout + n_drop=1),
       │     apply TWSE costs (0.1425% × 2 + 0.3% tax + NT$20 min),
       │     update positions, cash, equity
       │
       i += r                          ← advance rolling window by r
   end

   STAGE 2 saves final_prep.pt + warmup_log.csv
   STAGE 3 saves final.pt + predictions.csv + equity.csv + trades.csv
                    + backtest_daily.csv + report.json
```

## Key date windows (defaults from train.py)

Each operational phase consumes one of Algorithm 1's meta-sets:

| Date constant | Value | Operational phase | Meta-set (Algorithm 1) |
|---|---|---|---|
| `HISTORY_START` | 2016-01-01 | (universal cache floor) | — |
| `PRETRAIN_START` / `PRETRAIN_END` | 2016-01-01 / 2021-12-31 | Stage 1 — pretrain (offline) | `meta_train_set` (T_train) |
| `VALIDATION_START` / `VALIDATION_END` | 2022-01-01 / 2023-12-31 | (a) read-only val during Stage 1; (b) warmup IL "final preparation" in Stage 2 | `meta_valid_set` (T_valid) — dual-purpose per the paper |
| `ONLINE_TRAINING_START` / `ONLINE_TRAINING_END` | 2024-01-01 / `auto` | Stage 3 — online training / live-IL backtest | `meta_test_set` (T_test) |
| `HISTORY_END` | 2026-05-23 (rolling) | (clamps `ONLINE_TRAINING_END=auto` to the panel max) | — |

The 2022-2023 block serves dual duty: held-out val during Stage 1
(no updates) then warmup IL during Stage 2 (real updates). This is
exactly what Algorithm 1 prescribes for T_valid.

In `train.py` the panel slices are named to mirror the paper's meta-set
naming:
```python
meta_train_set = panel.slice_by_date(PRETRAIN_START, PRETRAIN_END)
meta_valid_set = panel.slice_by_date(VALIDATION_START, VALIDATION_END)
# meta_test_set is implicit — consumed by the live-IL backtest via
# start_date=ONLINE_TRAINING_START, end_date=ONLINE_TRAINING_END.
```

## Key model + training hyperparameters

| Param | Source | Default | Role |
|---|---|---:|---|
| `SEQ_LEN` | handler.py | 60 | Days of lookback per row |
| `N_FIELDS` | handler.py | 6 | Fields per timestep (close, open, high, low, vwap, volume) |
| `PRED_HORIZON` | handler.py | **1** | Forward c2c return horizon (paper-faithful) |
| `FEATURE_MODE` | handler.py | **"qlib"** | Alpha360 normalization scheme — `"qlib"` (5 price fields / close[t], vol / vol[t]) for run17+; `"per_field"` (each field / its own lag-0) for runs 1-16. `train.py` writes the active mode to `runN/feature_scheme.json` so `deploy_today.py` can match it. |
| `FOMAMLConfig.r` | maml.py default | 20 | Task interval (support len = query len = deployment window) |
| `FOMAMLConfig.inner_lr` | maml.py default | 1e-3 | η_θ — inner step |
| `FOMAMLConfig.lr_phi` | maml.py default | 1e-3 | η_φ — outer step on F (GRU base) |
| `FOMAMLConfig.lr_psi` | **train.py override** | **1e-3** (default 1e-2) | η_ψ — outer step on DA (G + H) |
| `FOMAMLConfig.alpha_reg` | maml.py default | 0.5 | α — weight of L_reg in L_test |
| `PretrainConfig.epochs` | **train.py override** | 30 (default 200) | Max Stage 1 epochs |
| `PretrainConfig.tasks_per_epoch` | **train.py override** | 1000 (default 200) | Random samples per epoch |
| `PretrainConfig.early_stop_patience` | **train.py override** | 8 (default 20) | ζ in Algorithm 1 |
| `DoubleAdapt.num_head` | double_adapt.py | 8 | N in paper |
| `DoubleAdapt.temperature` | double_adapt.py | 10 | τ in paper |
| `DoubleAdapt.hidden_dim` | double_adapt.py | 64 | GRU hidden width |
| `DoubleAdapt.num_layers` | double_adapt.py | 2 | GRU depth |
| `QlibBTConfig.top_k` | backtest_qlib.py | 10 | Stage 3 portfolio width |
| `QlibBTConfig.n_drop` | backtest_qlib.py | 1 | Stage 3 daily turnover cap |
| `QlibBTConfig.cost_alpha_bps` | backtest_qlib.py | 70 | Min pred score to enter |
| `QlibBTConfig.risk_degree` | backtest_qlib.py | 0.95 | Fraction of cash deployed |

## What lands on disk per run

```
mlruns/artifacts/doubleadapt/<run_id>/artifacts/     (MLflow run run{N})
├── pretrained.pt          ← Stage 1 best-val-IC checkpoint
├── final_prep.pt          ← Stage 2 post-warmup checkpoint
├── final.pt               ← Stage 3 final checkpoint (used by deploy_today)
├── norm_stats.pkl         ← Per-(lag, field) median + MAD from pretrain rows
├── universe_schedule.csv  ← Semi-annual TEJ index membership snapshots
├── active_universe.json   ← Latest snapshot's stock list (deploy uses this)
├── bt_config.json         ← QlibBTConfig used for the backtest
├── pretrain_log.csv       ← Per-epoch train_loss / val_ic / val_icir
├── warmup_log.csv         ← Per-step IL log for Stage 2
├── il_log.csv             ← Per-step IL log for Stage 3
├── predictions.csv        ← Per-(date, code) pred / actual for Stage 3
├── equity.csv             ← Per-day strategy equity from Stage 3 portfolio sim
├── trades.csv             ← Per-trade fills (Stage 3)
├── backtest_daily.csv     ← Per-day positions, cash, deal events (Stage 3)
├── live_forecast.csv      ← Predictions for the most recent r days (no labels yet)
├── report.json            ← Roll-up: IC/ICIR/RankIC/RankICIR + backtest summary
├── train.log              ← Full console log of the session
└── code_diff.txt          ← Uncommitted code at training time (qlib-style)
```

The same numbers are also MLflow params/metrics, so runs can be compared in
`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`. The run tagged `current=true` is what
`deploy_today.py` reads by default; override with `--run runX`.

## Known deviations from the DoubleAdapt paper

All previously flagged in earlier audits; consolidated here for one
reference table:

| # | Paper says | Code does | Effect |
|---|---|---|---|
| 1 | `T'_train ← shuffle(T_train)` (one sweep per epoch) | Random sampling with replacement, `tasks_per_epoch=1000` | Functionally near-identical; ~1000 unique tasks/epoch on average |
| 2 | `φ', ψ' ← copy(φ, ψ)` then val on copies | `train=False` skips outer backward (no copy) | Zero — both ensure val doesn't update meta-learners |
| 3 | Online predictions span T_valid ∪ T_test | `predictions.csv` only covers T_test (2024+) | Low — warmup-window predictions aren't persisted |
| 4 | Paper outer step is plain SGD | Adam (`optim_phi` / `optim_psi`) | Standard substitution; same η_φ / η_ψ as labels |
| 5 | H operates on raw y | CSRankNorm wraps y before H (qlib stability trick); reported IC still computed against raw y | Re-tuning all LRs would be needed to remove the wrap. **Tested in run11**: stripping CSRankNorm without LR rescaling halves Sharpe (1.24 → 0.43) and nearly triples MDD (−14% → −40%) vs run9 — the inner FOMAML step collapses because plain SGD on a 360× smaller loss is a no-op. See [model_env.md Run 11](model_env.md) and [model_performance.md run9 vs run11](model_performance.md) for the controlled ablation. |

## Related docs

- [model_env.md](model_env.md) — per-run config diffs (run1 → run10)
- [model_performance.md](model_performance.md) — qlib-style metrics vs 0050
- [qlib_backtest.md](qlib_backtest.md) — Stage 3 portfolio engine details
- [course_backtest.md](course_backtest.md) — zero-cost framework bridge
- [auto_trader.md](auto_trader.md) — daily `deploy_today.py` pipeline
- [topk_ndrop_selection.md](topk_ndrop_selection.md) — why top_k=10, n_drop=1
- [project_structure.md](project_structure.md) — file-by-file walkthrough
