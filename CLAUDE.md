# deep_learning_stock_trading

DoubleAdapt + GRU + Alpha360 on TWSE data (TEJ exports), trained with
incremental learning and traded on the course's virtual exchange.
Environment: conda env `dlst` (`environment.yml`). Full file walkthrough:
[docs/project_structure.md](docs/project_structure.md).

## Where things live

- Paths, date windows, default backtest config: `src/config.py`. Never hard-code these.
- Data loading, features, universe masks: `src/data/`. Raw TEJ: `data/raw/TEJ/` (see `data.yaml`).
- Networks: `src/model/`. Training loops (FOMAML, incremental learning, live-IL backtest): `src/trainer/`.
- Inference on the newest dates: `src/inference.py`.
- TWSE market rules, portfolio strategy, broker API (`stock_api`): `src/trading/`.
- Metrics and the backtest simulator: `src/evaluate/`.
- Run folders (`models/run{N}/`, `CURRENT`, artifacts, session logs): only through `src/runs.py`.
- Everything you run: `src/scripts/` (flat). Scripts import from `src/`, never from each other.
- Tests: `tests/` (`pytest -m slow` runs the end-to-end 2330 test).

Skip when searching: `data/` (CSV data), `models/` (run outputs), `logs/`,
`thirdparty/` (vendored qlib + course package), `docs/*.pdf|docx|png`,
`__pycache__/`, `.venv/`.

## Rules

- Import directions: `scripts → everything`; `trainer → evaluate, data, model`;
  `evaluate → trading`; `trading → config`; `data → config`. Nothing imports `scripts/`.
- Never move or rename `src/data/processor.py` or `RobustZScoreStats`: every
  `models/*/norm_stats.pkl` pickles it by module path.
- Keep `models/run{N}/` file names and JSON keys stable; run1–run18 must stay loadable.
- Backtest outputs depend on `PYTHONHASHSEED` (tie order in
  `trading/strategy.topk_dropout`); pin it when comparing runs byte for byte.
