# Project Structure

File-by-file walkthrough. Mirrors qlib's `contrib/` split
(`data / model / trainer / evaluate / workflow / scripts`).

## Top-level

```
deep_learning_stock_trading/
├── src/                       # the pipeline (this project, incl. test/)
├── qlib/                      # qlib clone — reference only (never imported)
├── stock_project_for_class/   # TWSE data API + course backtest framework
├── docs/                      # architecture + result + structure docs
├── .venv/                     # Python venv (gitignored)
├── README.md
└── .gitignore
```

| Path | Purpose |
|---|---|
| `src/` | All code that runs (data, model, trainer, evaluation, workflow, scripts) |
| `qlib/` | Microsoft's qlib fork — referenced for architecture parity, *not* imported at runtime |
| `stock_project_for_class/` | TWSE fetch API (`stock_api/`) + course `backtest_package/` (`BacktestSystem` + `examplebacktest.py` template) + course materials (`rules.md`, etc.) |
| `docs/` | Project docs — see [Docs](#docs) section below |

## `src/` — the pipeline

```
src/
├── __init__.py
├── data/
│   ├── io.py                  # read-only OHLC cache wrapper (cache populated by import_tej.py)
│   ├── handler.py             # Alpha360 feature builder + PanelTensors
│   ├── processor.py           # RobustZScoreNorm + Fillna + CSRankNorm
│   ├── universe.py            # TEJ index loader: TW50 + TM100 monthly snapshots
│   └── stock_data_all/        # raw TEJ files (half-year CSVs + TW50.csv + TM100.csv)
├── model/
│   ├── gru.py                 # GRUBase (qlib pytorch_gru defaults)
│   └── double_adapt.py        # FeatureAdapter + LabelAdapter + DoubleAdapt wrapper
├── trainer/
│   ├── maml.py                # FOMAML pretrain loop, Task, sample/eval helpers
│   └── incremental.py         # fake-online IL loop (walks 2022 → panel max)
├── evaluate/
│   ├── metrics.py             # cross_sectional_ic (Pearson IC + Spearman RankIC)
│   ├── trading_rules.py       # TWSE tick rounding, commission, sell tax, lot size
│   ├── backtest_qlib.py       # qlib-style in-process backtest (TopkDropoutStrategy)
│   └── run_course_backtest.py # bridge model preds → course's BacktestSystem
├── workflow/
│   └── forecast.py            # live forecast on label-less newest dates
├── scripts/
│   ├── import_tej.py          # populate src/_cache/ from TEJ half-year CSVs
│   ├── train.py               # main entry point (pretrain → IL → backtest → forecast)
│   ├── deploy_today.py        # daily inference + order submission
│   └── holdings.py            # query current holdings via Get_User_Stocks
├── test/
│   ├── __init__.py
│   ├── test_pipeline_2330.py  # end-to-end smoke test on stock 2330 only
│   └── output/                # test artifacts (gitignored)
├── output/                    # training artifacts (gitignored)
└── _cache/                    # per-(stock, year) CSV cache (gitignored; populated by import_tej.py)
```

### `src/data/` — data layer

| File | Key items |
|---|---|
| `io.py` | `fetch_year(code, year)` / `fetch_range(code, start, end)` — **read-only** wrapper around the local CSV cache at `src/_cache/`. Returns empty DataFrame for missing files; the cache is populated by `import_tej.py`. No network access. |
| `handler.py` | `compute_alpha360(df)` builds (60, 6) tensor per (stock, date). `build_panel(bars)` stacks all stocks into a long-format `PanelTensors(X, y, dates, codes)`. |
| `processor.py` | `fit_robust_zscore(X_pretrain)` returns `(median, MAD)`; `apply_robust_zscore` + `fillna(value=0)` for features. `csrank_normalize_per_date(y, dates)` for labels. |
| `universe.py` | **TEJ index mask loader.** `get_index_universe_codes(name)` returns the historical union for `name ∈ {TW50, TM100}` (~99 / ~266 codes). `build_universe_schedule(name, start, end)` returns one row per semi-annual rebalance with `{asof, codes, weights}` looked up from TEJ's monthly snapshots in `stock_data_all/TW50.csv` and `stock_data_all/TM100.csv`. Per-index thin wrappers also exposed: `get_tw50_*` / `get_tm100_*`. |
| `stock_data_all/` | TEJ raw inputs: 22 half-year price CSVs (`2016_H1.csv` → `2026_H1.csv`) + `TW50.csv` (184 monthly TW50 snapshots, 2011-2026) + `TM100.csv` (184 monthly TM100 snapshots). cp950-encoded. |

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
| `trading_rules.py` | TWSE-specific cost model: `tick_size`, `round_buy/sell`, `commission`, `buy_cost`, `sell_proceeds`. `SHARES_PER_LOT=1000`, `TAX_RATE=0.003`, `COMMISSION_RATE=0.001425`. |
| `backtest_qlib.py` | `run_qlib_backtest_from_predictions(preds, bars, QlibBTConfig)` — mirrors qlib's `TopkDropoutStrategy` + `deal_price="close"` (or `"open"`) execution, with configurable `costs="twse"` or `"qlib"` and `risk_degree=0.95`. Slot per buy = `cash * risk_degree / len(buys)` (qlib exact). |
| `run_course_backtest.py` | Bridges DoubleAdapt predictions into the course's `BacktestSystem` (from `stock_project_for_class/backtest_package/`). Pre-builds the framework's stock cache from `src/_cache/`, defines `select_stock` / `trade_strategy` mirroring [examplebacktest.py](../stock_project_for_class/backtest_package/stock_backtest_v1/examplebacktest.py), saves `course_backtest.xlsx` per run, optionally uploads to leaderboard via `--upload`. |

### `src/workflow/` — production helpers

| File | Key items |
|---|---|
| `forecast.py` | `forecast_latest(model, panel, n_days)` — scores the freshest `n_days` unique dates whose labels don't exist yet. Returns DataFrame sorted by `(date, pred desc)`. |

### `src/scripts/` — entry points

| File | What it does |
|---|---|
| `import_tej.py` | Reads TEJ half-year CSVs from `src/data/stock_data_all/` and writes per-(stock, year) cache files to `src/_cache/`. `--universe TW50` (default) / `TM100` / `BOTH` filters which codes to keep. Runs in <30s, no network. |
| `train.py` | The orchestrator. Loads bars → builds panel → applies universe schedule → fits feature norm → pretrain → IL → backtest → live forecast. `--universe TW50` (default) / `TM100` picks which TEJ index to train against. Writes all artifacts to `src/output/run{N}/`. Tees full stdout/stderr to `src/logs/train_run{N}_YYYYMMDD_HHMMSS.log`. |
| `deploy_today.py` | Daily inference + order submission. **Requires `--cash NT$`** (read from the course website). Logs orders to `src/output/run{N}/orders_YYYYMMDD.csv`, inferences to `inferences_YYYYMMDD.csv`, and tees full stdout to `src/logs/deploy_YYYYMMDD.log` (appended across multiple calls per day). |
| `holdings.py` | Query current settled holdings via `Get_User_Stocks`, cross-reference against today's order log to show which submitted orders filled. |

### `src/logs/` — session logs (gitignored)

| File | Contents |
|---|---|
| `train_run{N}_YYYYMMDD_HHMMSS.log` | Per-run full training session capture (every print from train.py) |
| `deploy_YYYYMMDD.log` | Append-mode daily deploy session log (every invocation of `deploy_today.py`) |

### `src/output/` — training artifacts

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
| `qlib/contrib/meta/incremental/net.py` | `FeatureAdapter`, `LabelAdapter`, `DoubleAdapt` — algorithmic template for `src/model/double_adapt.py` |
| `qlib/contrib/meta/incremental/model.py` | Bi-level training pattern |
| `qlib/contrib/model/pytorch_gru.py` | GRU default hyperparameters |
| `qlib/contrib/data/handler.py` | `Alpha158`, `Alpha360` feature formulas |
| `examples/benchmarks/GRU/workflow_config_gru_Alpha360.yaml` | Standard processor stack (RobustZScoreNorm + Fillna + CSRankNorm) |

## `stock_project_for_class/` — external dependency + coursework

```
stock_project_for_class/
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

Imported as `from stock_api import get_taiwan_stock_data` after
`sys.path.insert(0, "stock_project_for_class")` (done in `src/data/io.py`
and `src/data/universe.py`). `run_course_backtest.py` additionally adds
`backtest_package/stock_backtest_v1/` to `sys.path` and imports
`BacktestSystem` from there.

## Data flow

```
TEJ half-year CSVs  ── import_tej.py ──>  src/_cache/{code}_{year}.csv
(stock_data_all/)        (TW50 or TM100 union filter)
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
| [project_structure.md](project_structure.md) | This file — code layout and module map |
| [model_env.md](model_env.md) | Per-run training log: what changed in the env for each of runs 1-8 and why, plus headline diffs in both backtest engines |
| [model_performance.md](model_performance.md) | qlib-style metrics (IC, ICIR, RankIC, RankICIR, IR, excess return, max DD) for all 8 runs vs 0050 benchmark |
| [qlib_backtest.md](qlib_backtest.md) | In-process qlib-style backtest engine — TWSE costs, `deal_price=close`, cross-run comparison table, tunable knobs |
| [course_backtest.md](course_backtest.md) | Course `BacktestSystem` bridge — zero costs, 1-day causal lag, leaderboard upload |
| [auto_trader.md](auto_trader.md) | Daily `deploy_today.py` pipeline + scheduling guide |
| [topk_ndrop_selection.md](topk_ndrop_selection.md) | Why `top_k=10, n_drop=1` and how to pick for other universe sizes |

## Conventions

- All `src/*` modules use **absolute imports** through the `src` package
  (e.g. `from src.data.handler import build_panel`).
- Scripts in `src/scripts/` add the project root to `sys.path` so
  `python src/scripts/train.py` works from the repo root.
- The qlib clone is **never** imported — it's reference material only.
- `stock_api` and `backtest_package` live outside `src/` to keep the
  third-party package surface obvious.
- Data and training artifacts (`_cache/`, `output/`, `logs/`) are gitignored
  — they're reproducible from `scripts/import_tej.py` + `scripts/train.py`.
