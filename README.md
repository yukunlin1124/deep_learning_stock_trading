# Deep-Learning Stock Trading — DoubleAdapt on TWSE

A meta-learning forecasting pipeline for Taiwan Stock Exchange (TWSE) equities,
built around the **DoubleAdapt** architecture (KDD 2023) with **Alpha360**
features. Targets a top-50 active universe drawn from a 150-stock pool that
rebalances twice a year by trailing dollar volume.

## What it does

1. **Pretrain (offline, 2016 → 2021)** — meta-train a GRU base predictor and
   two adapters (FeatureAdapter `G`, LabelAdapter `H`) on rolling
   (support, query) tasks via first-order MAML (FOMAML).
2. **Validate (2022 → 2023)** — early-stop pretraining on validation IC.
3. **Online IL (2024 → present)** — walk forward in 20-trading-day steps; at
   each step do one bi-level update on the past 20 days (support) and score
   the next 20 days (query). Keeps the adapters drifting with the market.
4. **Portfolio backtest** over the same IL window — top-K long-only with
   one-shot limit-at-prev-close orders + TWSE-realistic costs (commission,
   sell tax, lot size, tick rounding).
5. **Live forecast** — score the freshest label-less rows for "what does the
   model think tomorrow's winners are."

Architecture and processor stack mirror the qlib reference implementation
(`qlib/qlib/contrib/meta/incremental/net.py` and
`qlib/examples/benchmarks/GRU/workflow_config_gru_Alpha360.yaml`).
Hyperparameters follow the DoubleAdapt paper.

## Project layout

```
deep_learning_stock_trading/
├── analysis/                  # the pipeline (this project, incl. test/)
├── qlib/                      # qlib clone — reference only, never imported
├── stock_project_for_class/   # TWSE data API + coursework materials
├── docs/                      # architecture + structure docs
└── .venv/                     # Python venv (gitignored)
```

See:
- [docs/project_structure.md](docs/project_structure.md) — file-by-file walkthrough
- [docs/topk_ndrop_selection.md](docs/topk_ndrop_selection.md) — picking `top_k` and `n_drop`
- [docs/auto_trader.md](docs/auto_trader.md) — daily deploy schedule
- [docs/training_experiments.md](docs/training_experiments.md) — what each training run showed (5 runs to date; production candidates: run 4 **+136.78% / Sharpe 1.27** [aggressive] or run 5 **+76.16% / Sharpe 0.95 / max DD −27%** [conservative, methodologically cleanest])

## Tests

```powershell
# End-to-end pipeline test on stock 2330 (TSMC). Requires all 11 yearly
# cache files (2016-2026) for 2330. Verifies the full workflow (data ->
# panel -> pretrain -> IL -> backtest -> live forecast) wires together
# correctly using the same fixed splits as the main pipeline.
.venv/Scripts/python.exe src/test/test_pipeline_2330.py
```

Single-stock test is **plumbing-only** — it cannot validate model quality
because CSRankNorm degenerates with n=1 per date (see the script docstring
for details).

## Prerequisites — external clones

Two repos must live as **siblings under the project root** (gitignored here):

| Path | Source | Purpose |
|---|---|---|
| `stock_project_for_class/` | `git clone https://ciot.imis.ncku.edu.tw:25388/Amy/stock_project_for_class.git` | Course's TWSE API (`Buy_Stock`, `Sell_Stock`, `get_taiwan_stock_data`, etc.) |
| `qlib/` | `git clone https://github.com/SJTU-DMTai/DoubleAdapt.git qlib` (or any qlib fork) | qlib reference clone — used by the docs for architecture cross-checks; **never imported at runtime** |

Without `stock_project_for_class/` the pipeline can't fetch TWSE bars or
submit orders. `qlib/` is optional but referenced in `docs/`.

## Quickstart

```powershell
# 1. Create venv and install all Python deps in one shot. requirements.txt
#    pulls CUDA-12.8 PyTorch from NVIDIA's wheel index by default; edit the
#    --extra-index-url line for CPU-only or older CUDA.
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. Install the course package as editable (after cloning per Prerequisites):
.venv/Scripts/python.exe -m pip install -e ./stock_project_for_class

# 2. Scrape TWSE bars into src/_cache (~5 hr fresh, resumable).
.venv/Scripts/python.exe src/scripts/scrape.py

# 3. Retry any (stock, year) pairs the main scrape failed or skipped.
.venv/Scripts/python.exe src/scripts/retry.py

# 4. Pretrain + IL + backtest + live forecast. Artifacts -> src/output/.
.venv/Scripts/python.exe src/scripts/train.py

# 5. (Daily after train.py) auto-trade on the course virtual TWSE.
#    Schedule this at 12:00 (1hr before the 13:00 order window closes):
$env:TWSE_ACCOUNT="..."; $env:TWSE_PASSWORD="..."
.venv/Scripts/python.exe src/scripts/deploy_today.py
# Add --dry-run to preview orders without submitting.
```

## Outputs (`src/output/`)

| File | What it is |
|---|---|
| `pretrained.pt` | Model state dict at best validation IC during pretrain |
| `final.pt` | Model state dict after the IL phase |
| `pretrain_log.csv` | Per-epoch train loss + val IC/RankIC |
| `il_log.csv` | Per-step IL stats (loss, IC, RankIC) over each 20-day window |
| `predictions.csv` | Per-(date, stock) predicted vs actual 20-day forward return |
| `equity.csv` | Daily portfolio equity over the IL window |
| `trades.csv` | Every fill (date, code, side, price, lots, cashflow) |
| `backtest_daily.csv` | Per-day cash / mark-value / position count |
| `live_forecast.csv` | Predictions for the freshest label-less dates |
| `universe_schedule.csv` | Per-rebalance active universe (semi-annual snapshots) |
| `report.json` | Summary: best IC, IL stats, backtest metrics, config |

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
| Outer LRs | MA (`φ`) 1e-3, DA (`ψ`) 1e-2 | paper |
| Loss | `L_test = MSE(pred, y_norm) + 0.5 · MSE(H(y_norm), y_norm)` | paper |
| Pretrain budget | 200 epochs, early-stop patience 20 on val IC | qlib |
| Universe | 150 candidates → top-50 active by trailing 180d $vol, semi-annual | this project |
| Backtest | Top-K long-only, limit-at-prev-close, fill iff `low ≤ limit ≤ high` | this project |

## Caveats

- **Survivorship + selection bias** in the universe: candidate pool comes from
  today's `load_symbol_map()`, so truly delisted/dead names are absent.
  Semi-annual rerank changes the *active* set within the pool, but the pool
  itself reflects 2025 knowledge.
- **TWSE rate limit**: scrape sleeps 2s/month/stock. Full 150-stock × 11-year
  scrape takes ~5 hours fresh.
- **Determinism**: `torch.manual_seed(0)` is set but `cudnn.deterministic` is
  not. Two runs of `train.py` differ slightly (~5% in IC).
- **No portfolio backtest during pretrain** — IC/RankIC are signal-quality
  metrics, not P&L. The portfolio backtest only runs over the IL window.

## References

- Du et al., *DoubleAdapt: A Meta-learning Approach to Incremental Learning for
  Stock Trend Forecasting*, KDD 2023
- Yang et al., *Qlib: An AI-oriented Quantitative Investment Platform*, 2020
  ([qlib/](qlib/) is the reference fork)
