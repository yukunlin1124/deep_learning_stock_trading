# Project Structure

File-by-file walkthrough. Mirrors qlib's `contrib/` split
(`data / model / trainer / evaluate / workflow / scripts`).

## Top-level

```
deep_learning_stock_trading/
├── src/                       # the pipeline (this project, incl. test/); paths in src/config.py
├── data/                      # cookiecutter-data-science layout
│   ├── raw/                   # immutable inputs, never edited by code
│   │   └── TEJ/               # raw TEJ dataset, split by period (CSVs gitignored)
│   │       ├── data.yaml      # split boundaries, file schema, row counts
│   │       ├── train/         # 2011_H1 … 2021_H2 (pretrain uses 2016-2021)
│   │       ├── val/           # 2022_H1 … 2023_H2
│   │       ├── test/          # 2024_H1 … 2026_H1 (online / live-IL backtest)
│   │       ├── TW50.csv       # monthly index snapshots (all periods)
│   │       └── TM100.csv
│   ├── interim/               # per-(stock, year) bar cache + 0050_benchmark.csv (gitignored; from import_tej.py)
│   ├── processed/             # final model-ready datasets (unused for now)
│   └── external/              # non-TEJ third-party data (unused for now)
├── models/                    # trained runs: run{N}/, CURRENT pointer, test_2330/ (gitignored)
├── logs/                      # train / deploy session logs (gitignored)
├── thirdparty/                # external code (gitignored)
│   ├── qlib/                  # qlib clone — reference only (never imported)
│   └── stock_project_for_class/  # TWSE data API + course backtest framework
├── docs/                      # architecture + result + structure docs
├── environment.yml            # conda env spec (env name: dlst)
├── README.md
└── .gitignore
```

| Path | Purpose |
|---|---|
| `src/` | All code that runs (data, model, trainer, evaluation, workflow, scripts) |
| `data/interim/` | Rebuildable bar cache written by `import_tej.py`, read through `src/data/io.py`. Safe to delete. |
| `models/` | One `run{N}/` folder per training run (see [`models/`](#models--training-artifacts) below), the `CURRENT` pointer, and `test_2330/` from the smoke test. |
| `logs/` | Session logs (see [`logs/`](#logs--session-logs-gitignored) below). |
| `data/raw/TEJ/` | TEJ raw inputs, described in `data/raw/TEJ/data.yaml`: 31 half-year price CSVs split into `train/` (`2011_H1` → `2021_H2`), `val/` (`2022_H1` → `2023_H2`) and `test/` (`2024_H1` → `2026_H1`), plus `TW50.csv` (184 monthly TW50 snapshots, 2011-2026) + `TM100.csv` (184 monthly TM100 snapshots). cp950-encoded. Read by `src/scripts/import_tej.py` and `src/data/universe.py`. |
| `thirdparty/qlib/` | Microsoft's qlib fork — referenced for architecture parity, *not* imported at runtime |
| `thirdparty/stock_project_for_class/` | TWSE fetch API (`stock_api/`) + course `backtest_package/` (`BacktestSystem` + `examplebacktest.py` template) + course materials (`rules.md`, etc.) |
| `docs/` | Project docs — see [Docs](#docs) section below |

## `src/` — the pipeline

```
src/
├── __init__.py
├── config.py                  # all project paths (data/, models/, logs/)
├── data/
│   ├── io.py                  # read-only OHLC cache wrapper (cache populated by import_tej.py)
│   ├── handler.py             # Alpha360 feature builder + PanelTensors
│   ├── processor.py           # RobustZScoreNorm + Fillna + CSRankNorm
│   └── universe.py            # TEJ index loader: TW50 + TM100 monthly snapshots
├── model/
│   ├── gru.py                 # GRUBase (qlib pytorch_gru defaults)
│   └── double_adapt.py        # FeatureAdapter + LabelAdapter + DoubleAdapt wrapper
├── trainer/
│   ├── maml.py                # FOMAML pretrain loop, Task, sample/eval helpers
│   └── incremental.py         # fake-online IL loop (walks 2022 → panel max)
├── evaluate/
│   ├── metrics.py             # cross_sectional_ic (Pearson IC + Spearman RankIC)
│   ├── trading_rules.py       # TWSE tick rounding + lot size
│   ├── backtest_qlib.py       # qlib-style in-process backtest (TopkDropoutStrategy)
│   ├── run_course_backtest.py # bridge model preds → course's BacktestSystem
│   └── compute_qlib_metrics.py # IC / IR / excess-return table vs 0050 for every run
├── workflow/
│   └── forecast.py            # live forecast on label-less newest dates
├── scripts/
│   ├── import_tej.py          # populate data/interim/ from TEJ half-year CSVs
│   ├── train.py               # main entry point (pretrain → IL → backtest → forecast)
│   ├── train_il_baseline.py   # ablation: plain GRU + vanilla fine-tune IL (no DoubleAdapt), run18
│   ├── extend_il.py           # append the next IL step(s) to an existing run as new data arrives
│   ├── deploy_today.py        # daily inference + order submission
│   └── holdings.py            # query current holdings via Get_User_Stocks
├── test/
│   ├── __init__.py
│   └── test_pipeline_2330.py  # end-to-end smoke test on stock 2330 only (writes models/test_2330/)
└── utils/
    └── progress.py            # Tee, progress bars, logging helpers
```

### `src/data/` — data layer

| File | Key items |
|---|---|
| `io.py` | `fetch_year(code, year)` / `fetch_range(code, start, end)` — **read-only** wrapper around the local CSV cache at `data/interim/`. Returns empty DataFrame for missing files; the cache is populated by `import_tej.py`. No network access. |
| `handler.py` | `compute_alpha360(df)` builds (60, 6) tensor per (stock, date). `build_panel(bars)` stacks all stocks into a long-format `PanelTensors(X, y, dates, codes)`. |
| `processor.py` | `fit_robust_zscore(X_pretrain)` returns `(median, MAD)`; `apply_robust_zscore` + `fillna(value=0)` for features. `csrank_normalize_per_date(y, dates)` for labels. |
| `universe.py` | **TEJ index mask loader.** `get_index_universe_codes(name)` returns the historical union for `name ∈ {TW50, TM100}` (~99 / ~266 codes). `build_universe_schedule(name, start, end)` returns one row per semi-annual rebalance with `{asof, codes, weights}` looked up from TEJ's monthly snapshots in `data/raw/TEJ/TW50.csv` and `data/raw/TEJ/TM100.csv`. |

### `src/model/` — neural modules

| File | Key items |
|---|---|
| `gru.py` | `GRUBase(input_dim=6, hidden_dim=64, num_layers=2, dropout=0.0)` → `Linear(64, 1)` → scalar. Matches qlib `pytorch_gru.py` defaults. |
| `double_adapt.py` | `FeatureAdapter`, `LabelAdaptHeads`, `LabelAdapter`, and the `DoubleAdapt` wrapper that wires `G → base → H_inverse`. Mirrors qlib `meta/incremental/net.py` exactly. |

### `src/trainer/` — training loops

| File | Key items |
|---|---|
| `maml.py` | `Task` dataclass; `panel_to_task`, `sample_pretrain_tasks` for slicing tasks by trading-date index; `fomaml_step` for one FOMAML bi-level update; `pretrain_offline` for the meta-train loop with early-stop on val IC. |
| `incremental.py` | `run_incremental(model, panel, ...)` walks forward by `r=20` trading days through the IL window, applies one `fomaml_step` per step, accumulates predictions + per-step IC/RankIC. |

### `src/evaluate/` — metrics + backtest

| File | Key items |
|---|---|
| `metrics.py` | `cross_sectional_ic(preds, ys, dates)` → `(mean_IC, ICIR, mean_RankIC, RankICIR)`. Per-date correlation aggregated. |
| `trading_rules.py` | TWSE tick rounding for limit orders: `tick_size`, `round_buy/sell`, plus `SHARES_PER_LOT=1000`. Trading costs live in `backtest_qlib._COST_TABLES`. |
| `backtest_qlib.py` | `run_qlib_backtest_with_il(model, il_panel, bars, ...)` — the live-IL backtest `train.py` runs: one FOMAML adapt per 20-day window, then trades those 20 days. `run_qlib_backtest_from_predictions(preds, bars, QlibBTConfig)` replays a saved `predictions.csv`. Both mirror qlib's `TopkDropoutStrategy` + `deal_price="close"` (or `"open"`) execution, with configurable `costs="twse"` or `"qlib"` and `risk_degree=0.95`. Slot per buy = `cash * risk_degree / len(buys)` (qlib exact). |
| `compute_qlib_metrics.py` | Prints the qlib-style metrics table used in [model_performance.md](model_performance.md) (IC, ICIR, IR, excess return vs the 0050 benchmark) from each run's `equity.csv` + `report.json`. |
| `run_course_backtest.py` | Bridges DoubleAdapt predictions into the course's `BacktestSystem` (from `thirdparty/stock_project_for_class/backtest_package/`). Pre-builds the framework's stock cache from `data/interim/`, defines `select_stock` / `trade_strategy` mirroring [examplebacktest.py](../thirdparty/stock_project_for_class/backtest_package/stock_backtest_v1/examplebacktest.py), saves `course_backtest.xlsx` per run, optionally uploads to leaderboard via `--upload`. |

### `src/workflow/` — production helpers

| File | Key items |
|---|---|
| `forecast.py` | `forecast_latest(model, panel, n_days)` — scores the freshest `n_days` unique dates whose labels don't exist yet. Returns DataFrame sorted by `(date, pred desc)`. |

### `src/scripts/` — entry points

| File | What it does |
|---|---|
| `import_tej.py` | Reads TEJ half-year CSVs from `data/raw/TEJ/` and writes per-(stock, year) cache files to `data/interim/`. `--universe TW50` (default) / `TM100` / `BOTH` filters which codes to keep. Runs in <30s, no network. |
| `extend_il.py` | Runs the next r=20-day IL step on an existing run (`--run runN`, `--dry-run` to preview) and appends to its `final.pt`, `il_log.csv` and `predictions.csv`, backing up the old weights first. |
| `train_il_baseline.py` | Ablation baseline: plain GRU with vanilla MSE pretrain and N=10 SGD fine-tune steps per IL window (no adapters, no FOMAML). Same splits, universe and backtest as `train.py`; writes `models/run18/`. |
| `train.py` | The orchestrator. Loads bars → builds panel → applies universe schedule → fits feature norm → pretrain → IL → backtest → live forecast. `--universe TW50` (default) / `TM100` picks which TEJ index to train against. Writes all artifacts to `models/run{N}/`. Tees full stdout/stderr to `logs/train_run{N}_YYYYMMDD_HHMMSS.log`. |
| `deploy_today.py` | Daily inference + order submission. **Requires `--cash NT$`** (read from the course website). Logs orders to `models/run{N}/orders_YYYYMMDD.csv`, inferences to `inferences_YYYYMMDD.csv`, and tees full stdout to `logs/deploy_YYYYMMDD.log` (appended across multiple calls per day). |
| `holdings.py` | Query current settled holdings via `Get_User_Stocks`, cross-reference against today's order log to show which submitted orders filled. |

### `logs/` — session logs (gitignored)

| File | Contents |
|---|---|
| `train_run{N}_YYYYMMDD_HHMMSS.log` | Per-run full training session capture (every print from train.py) |
| `deploy_YYYYMMDD.log` | Append-mode daily deploy session log (every invocation of `deploy_today.py`) |

### `models/` — training artifacts

One `run{N}/` per `train.py` run; the full file list is in the
[README's Outputs table](../README.md#outputs-modelsrunn).

## `thirdparty/qlib/` — reference clone

Microsoft's qlib repository (the fork from
`https://github.com/SJTU-DMTai/DoubleAdapt`). **Never imported at runtime** —
kept for cross-referencing architectural and hyperparameter choices in our
own implementation. Notable files:

| Path | What we reference |
|---|---|
| `qlib/contrib/meta/incremental/net.py` | `FeatureAdapter`, `LabelAdapter`, `DoubleAdapt` — algorithmic template for `src/model/double_adapt.py` |
| `qlib/contrib/meta/incremental/model.py` | Bi-level training pattern |
| `qlib/contrib/model/pytorch_gru.py` | GRU default hyperparameters |
| `qlib/contrib/data/handler.py` | `Alpha158`, `Alpha360` feature formulas |
| `examples/benchmarks/GRU/workflow_config_gru_Alpha360.yaml` | Standard processor stack (RobustZScoreNorm + Fillna + CSRankNorm) |

## `thirdparty/stock_project_for_class/` — external dependency + coursework

```
thirdparty/stock_project_for_class/
├── stock_api/                 # the actual API package
│   ├── __init__.py
│   ├── core.py                # get_taiwan_stock_data + Get_User_Stocks + Buy_Stock + Sell_Stock
│   ├── fetchers.py            # TWSE / TPEX / ESB HTML+JSON scrapers (2s sleep per call)
│   ├── symbols.py             # symbol lookups
│   ├── utils.py               # date conversion helpers + safe_get_json
│   └── stock_symbol_map.json  # ~2565 listed codes (TWSE/OTC/ETF/ESB)
├── backtest_package/          # course's BacktestSystem framework
│   └── stock_backtest_v1/
│       ├── backtest/
│       │   ├── backtest.py    # BacktestSystem class (set_period → execute_strategy → calc_perf)
│       │   ├── Stock_API.py   # local copy bundled with the backtest framework
│       │   └── type.py        # TransactionRecord / User_Inventory / Daily_Asset dataclasses
│       ├── examplebacktest.py # template our src/evaluate/run_course_backtest.py mirrors
│       └── example.py         # live API usage example
├── course_example.py          # course's reference example
├── project_objective.md       # coursework spec
├── rules.md                   # coursework rules (trading constraints)
└── requirements.txt           # pandas, requests
```

`stock_api` is pip-installed in editable mode by `environment.yml`, so
`from stock_api import Buy_Stock` works anywhere. `run_course_backtest.py` additionally adds
`backtest_package/stock_backtest_v1/` to `sys.path` and imports
`BacktestSystem` from there.

## Data flow

```
TEJ half-year CSVs  ── import_tej.py ──>  data/interim/{code}_{year}.csv
(data/raw/TEJ/{train,val,test}/)   (TW50 or TM100 union filter)
                                              │
                                              ├── fetch_range ──> raw bars dict
                                              │                         │
                                              │                         ├── compute_alpha360 per stock ──> PanelTensors
                                              │                         │                                       │
TEJ TW50.csv / TM100.csv  ─ universe.py ──────┼──> universe schedule mask
                                              │                         │
                                              │                         ├── RobustZScoreNorm + Fillna
                                              │                         │
                                              │                         ├── pretrain (FOMAML, 2016-2021)  ──> pretrained.pt
                                              │                         │       └── early-stop on 2022-2023 val IC
                                              │                         │
                                              │                         ├── IL (FOMAML, 2022-now, r=20)   ──> final.pt
                                              │                         │       └── per-step IC/RankIC         predictions.csv
                                              │                         │                                      il_log.csv
                                              │                         │
                                              │                         ├── qlib backtest (top-K longs)   ──> equity.csv
                                              │                         │                                      trades.csv
                                              │                         │
                                              │                         └── live forecast (latest 20d)    ──> live_forecast.csv
                                              │
                                              └── (bars also passed to qlib backtest engine for fills)
```

## Docs

| File | Contents |
|---|---|
| [overview.md](overview.md) | End-to-end training-procedure diagram: three-stage schedule, DoubleAdapt architecture, rolling task window, per-task FOMAML detail, key hyperparameters |
| [project_structure.md](project_structure.md) | This file — code layout and module map |
| [model_env.md](model_env.md) | Per-run training log: what changed for each of runs 1-17 and why, plus headline diffs in both backtest engines |
| [model_performance.md](model_performance.md) | qlib-style metrics (IC, ICIR, RankIC, RankICIR, IR, excess return, max DD) for runs 1-17 vs 0050 benchmark |
| [qlib_backtest.md](qlib_backtest.md) | In-process qlib-style backtest engine — TWSE costs, `deal_price=close`, cross-run comparison table, tunable knobs |
| [course_backtest.md](course_backtest.md) | Course `BacktestSystem` bridge — zero costs, 1-day causal lag, leaderboard upload |
| [auto_trader.md](auto_trader.md) | Daily `deploy_today.py` pipeline + scheduling guide |
| [topk_ndrop_selection.md](topk_ndrop_selection.md) | Why `top_k=10, n_drop=1` and how to pick for other universe sizes |

## Conventions

- All `src/*` modules use **absolute imports** through the `src` package
  (e.g. `from src.data.handler import build_panel`).
- `src` is pip-installed in editable mode (`pip install -e .`, done by
  `environment.yml`), so `python src/scripts/train.py` works from anywhere.
- The qlib clone is **never** imported — it's reference material only.
- `stock_api` and `backtest_package` live outside `src/` to keep the
  third-party package surface obvious.
- Data and training artifacts (`data/interim/`, `models/`, `logs/`, and the
  raw CSVs under `data/raw/`) are gitignored. Everything except `data/raw/`
  is reproducible from `scripts/import_tej.py` + `scripts/train.py`.
- `src/` holds code only. Every path comes from `src/config.py`, so moving a
  folder means changing one constant.
