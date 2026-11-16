# Deep-Learning Stock Trading — DoubleAdapt on TWSE

A meta-learning forecasting pipeline for Taiwan Stock Exchange (TWSE) equities,
built around the **DoubleAdapt** architecture (KDD 2023) with **Alpha360**
features. Targets the **TW50** (50-stock blue-chip) or **TM100** (100-stock
mid-cap) index as a time-varying active universe, masked from official TEJ
monthly snapshots.

## Training schedule — the three-stage pattern

The 2022-2023 window plays **two roles in sequence**, by design:

| Stage | Window | What happens to the model | Role of the 2022-2023 window |
|---|---|---|---|
| **1. Pretrain** | 2016-01 → 2021-12 | Offline FOMAML meta-train on the past. Adapters + base GRU update via bi-level optimization. | *(future)* |
| **2. Validate** | 2022-01 → 2023-12 | Model is **read-only** per pretrain epoch — val IC drives early-stop (`patience=8` epochs of no improvement). Output: `pretrained.pt`. | **Role A: held-out validation signal** during pretrain. |
| **2'. Final preparation (IL warmup)** | 2022-01 → 2023-12 | After pretrain checkpoint is frozen, IL begins at the same date. ~24 bi-level updates (`r=20` trading days) walk through the val window, fine-tuning the model on recent post-pretrain regime **before going online**. Output: `final_prep.pt`. | **Role B: warmup IL** — the "final preparation before live deployment." |
| **3. Live-IL backtest** | 2024-01 → now | The qlib backtest **loads `final_prep.pt`** and walks forward in 20-day windows. Each window: one FOMAML adapt + portfolio sim on the next 20 days. So **the model continues fine-tuning during the backtest itself** — the trade at date D uses the model state at D, which reflects everything it has adapted through D − 20. Output: `final.pt` + `predictions.csv` + `equity.csv`. | *(this is the deployed model)* |

So the val window is **both** the early-stop signal AND the post-pretrain
warmup — the pipeline cleanly separates these two roles in time (read-only
during pretrain, then write during IL) without leaking val labels into
pretrain training.

Three checkpoints get written per `train.py` run:

| Checkpoint | When saved | Role |
|---|---|---|
| `pretrained.pt` | end of Stage 1 (best val IC) | Pre-IL baseline; useful for warmup-vs-no-warmup A/B |
| `final_prep.pt` | end of Stage 2' (warmup complete) | "After final preparation" — the starting point for the live-IL backtest in Stage 3 |
| `final.pt` | end of Stage 3 (live-IL backtest done) | What `deploy_today.py` loads tomorrow; reflects all online IL adaptation |

Edit the five constants at [src/scripts/train.py:119-124](src/scripts/train.py#L119-L124)
to shift any window — nothing downstream is hardcoded to those dates.

## What it does

1. **Pretrain (offline, 2016 → 2021)** — meta-train a GRU base predictor and
   two adapters (FeatureAdapter `G`, LabelAdapter `H`) on rolling
   (support, query) tasks via first-order MAML (FOMAML).
2. **Validate (2022 → 2023)** — early-stop pretraining on validation IC
   *(Role A above)*.
3. **IL warmup (2022 → 2023)** — walk forward in 20-trading-day
   steps; at each step do one bi-level update on the past 20 days (support).
   This fine-tunes the pretrained checkpoint into `final_prep.pt` *(Role B)*.
4. **Live-IL qlib backtest (2024 → now)** — load `final_prep.pt`, then walk
   forward in 20-day windows doing adapt + portfolio simulation in tandem.
   Model continues fine-tuning during the backtest; final state → `final.pt`.
   See [docs/qlib_backtest.md](docs/qlib_backtest.md). The course-framework
   backtest ([docs/course_backtest.md](docs/course_backtest.md)) replays
   `predictions.csv` produced by this step through the course's zero-cost
   `BacktestSystem`.
5. **Live forecast + daily deploy** — `deploy_today.py` loads `final.pt`,
   scores the freshest universe, and submits limit orders to the course's
   virtual TWSE system.

Architecture and processor stack mirror the qlib reference implementation
(`thirdparty/qlib/qlib/contrib/meta/incremental/net.py` and
`thirdparty/qlib/examples/benchmarks/GRU/workflow_config_gru_Alpha360.yaml`).
Hyperparameters follow the DoubleAdapt paper.

## Project layout

```
deep_learning_stock_trading/
├── src/                       # the pipeline (data, model, trainer, evaluate, scripts)
├── thirdparty/                # external code (gitignored)
│   ├── qlib/                  # qlib clone — reference only, never imported
│   └── stock_project_for_class/  # TWSE trading API + course's BacktestSystem
├── docs/                      # architecture + result + structure docs
└── environment.yml            # conda env spec (`conda activate dlst`)
```

Docs:
- [docs/overview.md](docs/overview.md) — training-procedure diagram (three-stage schedule + DoubleAdapt architecture + rolling task window + per-task FOMAML)
- [docs/project_structure.md](docs/project_structure.md) — file-by-file walkthrough
- [docs/model_env.md](docs/model_env.md) — per-run training log (runs 1-17: what changed each iteration and why)
- [docs/model_performance.md](docs/model_performance.md) — qlib-style metrics (IC, ICIR, IR, excess return) vs 0050 benchmark
- [docs/qlib_backtest.md](docs/qlib_backtest.md) — in-process qlib backtest engine
- [docs/course_backtest.md](docs/course_backtest.md) — course-framework backtest + leaderboard upload
- [docs/auto_trader.md](docs/auto_trader.md) — daily deploy schedule
- [docs/topk_ndrop_selection.md](docs/topk_ndrop_selection.md) — picking `top_k` and `n_drop`

## Tests

```powershell
# End-to-end pipeline test on stock 2330 (TSMC). Requires all 11 yearly
# cache files (2016-2026) for 2330. Verifies the full workflow (data ->
# panel -> pretrain -> IL -> backtest -> live forecast) wires together
# correctly using the same fixed splits as the main pipeline.
python tests/test_pipeline_2330.py
```

Single-stock test is **plumbing-only** — it cannot validate model quality
because CSRankNorm degenerates with n=1 per date (see the script docstring
for details).

## Prerequisites — external clones

Two repos must live under **`thirdparty/`** in the project root (gitignored here):

| Path | Source | Purpose |
|---|---|---|
| `thirdparty/stock_project_for_class/` | `git clone https://ciot.imis.ncku.edu.tw:25388/Amy/stock_project_for_class.git thirdparty/stock_project_for_class` | Course's TWSE trading API (`Buy_Stock`, `Sell_Stock`, `Get_User_Stocks`) + the `BacktestSystem` framework |
| `thirdparty/qlib/` | `git clone https://github.com/SJTU-DMTai/DoubleAdapt.git thirdparty/qlib` (or any qlib fork) | qlib reference clone — used by the docs for architecture cross-checks; **never imported at runtime** |

Without `thirdparty/stock_project_for_class/` the deploy script can't submit orders or
upload to the leaderboard. `thirdparty/qlib/` is optional but referenced in `docs/`.

## Data source — TEJ

Bars and TW50/TM100 constituent history are sourced from **TEJ** (Taiwan
Economic Journal) exports placed under `data/raw/TEJ/`:

| File | What |
|---|---|
| `data.yaml` | Describes the split layout, file schema and row counts |
| `{train,val,test}/{YEAR}_H{1,2}.csv` | Half-year price+volume CSVs (cp950-encoded; 31 files, 2011-2026 → split-adjusted OHLC, capacity, turnover). `train/` = 2011–2021 (pretrain uses 2016–2021), `val/` = 2022–2023, `test/` = 2024 onward |
| `TW50.csv` | 184 monthly snapshots of FTSE TWSE Taiwan 50 constituents (2011-2026, 99 unique codes) |
| `TM100.csv` | 184 monthly snapshots of FTSE TWSE Mid-Cap 100 constituents (2011-2026, 266 unique codes) |

The importer ([src/scripts/import_tej.py](src/scripts/import_tej.py))
reads these and writes per-(stock, year) cache files to `data/interim/`.
No network access during training — everything reads from local disk.

The repo follows the [cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/)
layout: `data/raw/` (immutable inputs), `data/interim/` (rebuildable cache),
`data/processed/` and `data/external/` (unused for now), `models/` (training
runs) and `logs/`. All paths are defined in [src/config.py](src/config.py).

## Quickstart

```powershell
# 1. Create the conda env (env name: dlst). environment.yml pulls CUDA-12.8
#    PyTorch from the PyTorch wheel index and installs this repo + the course
#    package (thirdparty/stock_project_for_class) as editable. Edit the torch
#    line in environment.yml for CPU-only machines.
conda env create -f environment.yml
conda activate dlst

# 2. (After editing environment.yml later) sync the env:
#    conda env update -f environment.yml --prune

# 3. Populate data/interim/ from the TEJ CSVs (~30 seconds, no network).
#    --universe TW50 (default) ≈ 99 codes; TM100 ≈ 266 codes; BOTH ≈ 300.
python src/scripts/import_tej.py

# 4. Pretrain + warmup IL + online IL + backtest + live forecast.
#    Artifacts -> models/run{N}/, CURRENT pointer updates automatically.
python src/scripts/train.py                  # uses TW50
python src/scripts/train.py --universe TM100 # broader universe

# 5. (Daily after train.py) auto-trade on the course virtual TWSE.
#    Schedule at 12:00 (1hr before the 13:00 order window closes).
#    --cash is REQUIRED (read it from the course website).
$env:TWSE_ACCOUNT="..."; $env:TWSE_PASSWORD="..."
python src/scripts/deploy_today.py --cash 100000000
# Add --dry-run to preview orders without submitting.

# 6. Backtest cross-comparison (qlib engine):
#    See docs/qlib_backtest.md for the reproducible Python one-liner that
#    runs every run's predictions.csv through the same engine.
```

## Outputs (`models/run{N}/`)

| File | What it is |
|---|---|
| `pretrained.pt` | Model state at best validation IC during pretrain (Stage 1 end) |
| `final_prep.pt` | Model state after IL warmup (Stage 2' end) — "ready to deploy" starting point for the live-IL backtest |
| `final.pt` | Model state after Stage 3's live-IL backtest — what `deploy_today.py` loads |
| `norm_stats.pkl` | `(median, MAD)` for RobustZScoreNorm, fit on pretrain rows only |
| `pretrain_log.csv` | Per-epoch train loss + val IC/RankIC |
| `warmup_log.csv` | Per-step IL stats for the 2022-2023 warmup phase |
| `il_log.csv` | Per-step IL stats for the 2024+ live-IL backtest phase |
| `predictions.csv` | Per-(date, stock) predicted vs actual next-day close-to-close return |
| `equity.csv` | Daily portfolio equity over the IL window (qlib backtest) |
| `trades.csv` | Every fill (date, code, side, price, lots, cashflow) |
| `backtest_daily.csv` | Per-day cash / mark-value / position count |
| `live_forecast.csv` | Predictions for the freshest label-less dates |
| `universe_schedule.csv` | Per-rebalance active universe (semi-annual snapshots from TW50/TM100) |
| `active_universe.json` | Current period's constituent list (deploy_today reads this) |
| `bt_config.json` | Trading config used by backtest + deploy_today |
| `course_backtest.xlsx` | Course-framework backtest output (regenerated on each `run_course_backtest.py` invocation) |
| `report.json` | Summary: best IC, IL stats, backtest metrics, full config |

## Key design choices

| Component | Setting | Source |
|---|---|---|
| Features | Alpha360 (60 lags × 6 fields = 360 dims) | qlib |
| Base model | GRU `input=6, hidden=64, num_layers=2, dropout=0.0` | qlib defaults |
| FeatureAdapter | 8-head residual, per-time-step cosine gating, τ=10 | paper + qlib structure |
| LabelAdapter | 8-head affine `γy+β`, independent gating, hid_dim=32 | paper + qlib structure |
| Feature norm | RobustZScoreNorm (fit on pretrain) + Fillna(0) | qlib |
| Label norm | CSRankNorm per cross-section | qlib |
| Training | FOMAML — 1 inner SGD step on base (lr=1e-3), outer Adam on `(φ, ψ)` | paper |
| Outer LRs | MA (`φ`) 1e-3, DA (`ψ`) 1e-3 | runs 2+ tuning (paper's 1e-2 was unstable) |
| Loss | `L_test = MSE(pred, y_norm) + 0.5 · MSE(H(y_norm), y_norm)` | paper |
| Pretrain budget | 30 epochs, early-stop patience 8 on val IC | runs 3+ tuning |
| Universe | TW50 (default, 50 codes/period) or TM100 (100 codes/period), monthly from TEJ | this project |
| Backtest | Top-K long-only, qlib `TopkDropoutStrategy` shape | this project |

## Caveats

- **Survivorship cleaner than yfinance**: TEJ ships data for delisted stocks
  via half-year exports, so 2823 (China Life, delisted 2021), 2888 (Shin
  Kong Financial, delisted 2025), 3474 (Inotera, delisted 2016) all have
  historical bars in the cache. yfinance would have returned empty for
  those.
- **Dividend back-adjustment**: TEJ applies one consistent split + dividend
  adjustment per stock across its full history. Compared to TWSE-direct
  unadjusted prices, this smooths forward-return labels (no synthetic
  ex-div drops) but inflates older volumes by stock-specific factors.
- **Determinism**: `torch.manual_seed(0)` is set but `cudnn.deterministic`
  is not. Two runs of `train.py` differ slightly (~5% in IC).
- **No portfolio backtest during pretrain** — IC/RankIC are signal-quality
  metrics, not P&L. The portfolio backtest only runs over the IL window
  (2022-onwards) and reports on the 2024+ subset.

## References

- Du et al., *DoubleAdapt: A Meta-learning Approach to Incremental Learning for
  Stock Trend Forecasting*, KDD 2023
- Yang et al., *Qlib: An AI-oriented Quantitative Investment Platform*, 2020
  ([thirdparty/qlib/](thirdparty/qlib/) is the reference fork)
- FTSE Russell, *FTSE TWSE Taiwan 50 Index Ground Rules* (the index spec
  TEJ's `TW50.csv` follows)
- FTSE Russell, *FTSE TWSE Taiwan Mid-Cap 100 Index Ground Rules*
