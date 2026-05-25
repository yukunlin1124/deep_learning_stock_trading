# Project Structure

File-by-file walkthrough. Mirrors qlib's `contrib/` split
(`data / model / trainer / evaluate / workflow / scripts`).

## Top-level

```
deep_learning_stock_trading/
├── analysis/                  # the pipeline (this project, incl. test/)
├── qlib/                      # qlib clone — reference only (never imported)
├── stock_project_for_class/   # TWSE data API + coursework materials
├── docs/                      # architecture + structure docs
├── .venv/                     # Python venv (gitignored)
├── README.md
└── .gitignore
```

| Path | Purpose |
|---|---|
| `analysis/` | All code that runs (data, model, trainer, evaluation, workflow, scripts) |
| `qlib/` | Microsoft's qlib fork — referenced for architecture parity, *not* imported at runtime |
| `stock_project_for_class/` | The TWSE fetch API + course materials (`course_example.py`, `rules.md`, etc.) |
| `docs/` | Project docs |

## `analysis/` — the pipeline

```
analysis/
├── __init__.py
├── data/
│   ├── io.py            # TWSE CSV cache wrapper
│   ├── handler.py       # Alpha360 feature builder + PanelTensors
│   ├── processor.py     # RobustZScoreNorm + Fillna + CSRankNorm
│   └── universe.py      # 150-code pool + semi-annual top-50 schedule
├── model/
│   ├── gru.py           # GRUBase (qlib pytorch_gru defaults)
│   └── double_adapt.py  # FeatureAdapter + LabelAdapter + DoubleAdapt wrapper
├── trainer/
│   ├── maml.py          # FOMAML pretrain loop, Task, sample/eval helpers
│   └── incremental.py   # fake-online IL loop (walks 2024 → panel max)
├── evaluate/
│   ├── metrics.py       # cross_sectional_ic (Pearson IC + Spearman RankIC)
│   ├── trading_rules.py # TWSE tick rounding, commission, sell tax, lot size
│   └── backtest.py      # top-K equity simulator over IL predictions
├── workflow/
│   └── forecast.py      # live forecast on label-less newest dates
├── scripts/
│   ├── scrape.py        # cache populator (run once)
│   ├── retry.py         # post-scrape rescue (failed/missing pairs)
│   └── train.py         # main entry point (pretrain → IL → backtest → forecast)
├── test/
│   ├── __init__.py
│   ├── test_pipeline_2330.py   # end-to-end smoke test on stock 2330 only
│   └── output/                 # test artifacts (gitignored)
├── output/              # training artifacts (gitignored)
└── _cache/              # per-(stock, year) CSV cache (gitignored)
```

### `analysis/data/` — data layer

| File | Key items |
|---|---|
| `io.py` | `fetch_year(code, year)`, `fetch_range(code, start, end)` — TWSE cache wrapper. `CACHE_DIR = analysis/_cache/`. Network calls only on cache miss. |
| `handler.py` | `compute_alpha360(df)` builds (60, 6) tensor per (stock, date). `build_panel(bars)` stacks all stocks into a long-format `PanelTensors(X, y, dates, codes)`. |
| `processor.py` | `fit_robust_zscore(X_pretrain)` returns `(median, MAD)`; `apply_robust_zscore` + `fillna(value=0)` for features. `csrank_normalize_per_date(y, dates)` for labels. |
| `universe.py` | `CANDIDATE_CODES` list, `get_candidate_pool() → 150 codes`, `build_universe_schedule(bars, ...)` returns one row per semi-annual rebalance with the active top-50. |

### `analysis/model/` — neural modules

| File | Key items |
|---|---|
| `gru.py` | `GRUBase(input_dim=6, hidden_dim=64, num_layers=2, dropout=0.0)` → `Linear(64, 1)` → scalar. Matches qlib `pytorch_gru.py` defaults. |
| `double_adapt.py` | `FeatureAdapter`, `LabelAdaptHeads`, `LabelAdapter`, and the `DoubleAdapt` wrapper that wires `G → base → H_inverse`. Mirrors qlib `meta/incremental/net.py` exactly. |

### `analysis/trainer/` — training loops

| File | Key items |
|---|---|
| `maml.py` | `Task` dataclass; `panel_to_task`, `sample_pretrain_tasks` for slicing tasks by trading-date index; `fomaml_step` for one FOMAML bi-level update; `pretrain_offline` for the meta-train loop with early-stop on val IC. |
| `incremental.py` | `run_incremental(model, panel, ...)` walks forward by `r=20` trading days through the IL window, applies one `fomaml_step` per step, accumulates predictions + per-step IC/RankIC. |

### `analysis/evaluate/` — metrics + backtest

| File | Key items |
|---|---|
| `metrics.py` | `cross_sectional_ic(preds, ys, dates)` → `(mean_IC, ICIR, mean_RankIC, RankICIR)`. Per-date correlation aggregated. |
| `trading_rules.py` | TWSE-specific cost model: `tick_size`, `round_buy/sell`, `commission`, `buy_cost`, `sell_proceeds`. `SHARES_PER_LOT=1000`, `TAX_RATE=0.003`, `COMMISSION_RATE=0.001425`. |
| `backtest.py` | `run_backtest_from_predictions(preds_df, bars, BTConfig)` — top-K long-only over IL predictions; one-shot limit-at-prev-close orders; fill iff `low ≤ limit ≤ high`. Returns equity, trades, daily logs, summary. |

### `analysis/workflow/` — production helpers

| File | Key items |
|---|---|
| `forecast.py` | `forecast_latest(model, panel, n_days)` — scores the freshest `n_days` unique dates whose labels don't exist yet. Returns DataFrame sorted by `(date, pred desc)`. |

### `analysis/scripts/` — entry points

| File | What it does |
|---|---|
| `scrape.py` | Iterates `get_candidate_pool() × range(2016, 2027)` and calls `fetch_year` for each. Logs progress to `_cache/scrape.log`. Resumable. |
| `retry.py` | Parses `scrape.log` for `FAIL` lines + scans for missing cache files; retries each pair once. Logs to `_cache/retry.log`. |
| `train.py` | The orchestrator. Loads bars → builds panel → applies universe schedule → fits feature norm → pretrain → IL → backtest → live forecast. Writes all artifacts to `analysis/models/`. |

### `analysis/output/` — training artifacts

| File | What it is |
|---|---|
| `pretrained.pt` | Model state dict at best val IC during pretrain |
| `final.pt` | Model state dict after IL |
| `pretrain_log.csv` | Per-epoch train loss + val IC/RankIC |
| `il_log.csv` | Per-step IL stats |
| `predictions.csv` | Per-(date, stock) `pred` vs `actual` |
| `equity.csv` | Daily portfolio equity |
| `trades.csv` | Every fill |
| `backtest_daily.csv` | Per-day cash / mark / position count |
| `live_forecast.csv` | Predictions for label-less newest dates |
| `universe_schedule.csv` | Semi-annual active-universe snapshots |
| `report.json` | Summary metadata |

## `qlib/` — reference clone

Microsoft's qlib repository (the fork from
`https://github.com/SJTU-DMTai/DoubleAdapt`). **Never imported at runtime** —
kept for cross-referencing architectural and hyperparameter choices in our
own implementation. Notable files:

| Path | What we reference |
|---|---|
| `qlib/contrib/meta/incremental/net.py` | `FeatureAdapter`, `LabelAdapter`, `DoubleAdapt` — algorithmic template for `analysis/model/double_adapt.py` |
| `qlib/contrib/meta/incremental/model.py` | Bi-level training pattern |
| `qlib/contrib/model/pytorch_gru.py` | GRU default hyperparameters |
| `qlib/contrib/data/handler.py` | `Alpha158`, `Alpha360` feature formulas |
| `examples/benchmarks/GRU/workflow_config_gru_Alpha360.yaml` | Standard processor stack (RobustZScoreNorm + Fillna + CSRankNorm) |

## `stock_project_for_class/` — external dependency + coursework

```
stock_project_for_class/
├── stock_api/                 # the actual API package
│   ├── __init__.py
│   ├── core.py                # get_taiwan_stock_data dispatcher
│   ├── fetchers.py            # TWSE / TPEX / ESB HTML+JSON scrapers (2s sleep per call)
│   ├── symbols.py             # symbol lookups
│   ├── utils.py               # date conversion helpers
│   └── stock_symbol_map.json  # ~2565 listed codes
├── course_example.py          # course's reference example
├── project_objective.md       # coursework spec
├── rules.md                   # coursework rules (trading constraints)
└── requirements.txt           # pandas, requests
```

Imported as `from stock_api import get_taiwan_stock_data` after
`sys.path.insert(0, "stock_project_for_class")` (done in `analysis/data/io.py`
and `analysis/data/universe.py`).

## Data flow

```
TWSE web    ── fetch ──>  _cache/{code}_{year}.csv      (analysis/scripts/scrape.py)
                              │
                              ├── fetch_range  ──>  raw bars dict
                              │                         │
                              │                         ├── compute_alpha360 per stock ──>  PanelTensors
                              │                         │                                       │
                              │                         │                                       ├── universe schedule mask
                              │                         │                                       ├── RobustZScoreNorm + Fillna
                              │                         │                                       │
                              │                         │                                       ├── pretrain (FOMAML, 2016-2021)  ──>  pretrained.pt
                              │                         │                                       │       └── early-stop on 2022-2023 val IC
                              │                         │                                       │
                              │                         │                                       ├── IL (FOMAML, 2024-now)         ──>  final.pt
                              │                         │                                       │       └── per-step IC/RankIC          predictions.csv
                              │                         │                                       │                                       il_log.csv
                              │                         │                                       │
                              │                         │                                       ├── backtest (top-K longs)        ──>  equity.csv
                              │                         │                                       │                                       trades.csv
                              │                         │                                       │
                              │                         │                                       └── live forecast (latest 20d)    ──>  live_forecast.csv
                              │                         │
                              │                         └── (bars also passed directly to backtest for OHLC fills)
```

## Conventions

- All `analysis/*` modules use **absolute imports** through the `analysis`
  package (e.g. `from analysis.data.handler import build_panel`).
- Scripts in `analysis/scripts/` add the project root to `sys.path` so
  `python analysis/scripts/train.py` works from the repo root.
- The qlib clone is **never** imported — it's reference material only.
- `stock_api` lives outside `analysis/` to keep the third-party package
  surface obvious.
- Data and training artifacts (`_cache/`, `output/`) are gitignored — they're
  reproducible from `scripts/train.py`.
