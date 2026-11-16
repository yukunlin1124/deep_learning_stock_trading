"""Shared constants: project paths, experiment date windows, default backtest config.

This module depends on nothing else in `src/`, so any module may import it.

Paths follow the cookiecutter-data-science layout:

    data/raw/        immutable inputs (TEJ exports). Never edited by code.
    data/interim/    derived, rebuildable data (per-stock bar cache from import_tej.py)
    data/processed/  final model-ready datasets (unused for now)
    data/external/   third-party data that isn't TEJ (unused for now)
    mlruns/          MLflow tracking store (qlib's recorder flow): one MLflow run per
                     training run, holding its model, predictions, metrics, train.log,
                     command line and git diff. Browse with `mlflow ui` (see src/runs.py).
    logs/            deploy session logs (train logs live inside each MLflow run)

Everything except data/raw/ is safe to delete and regenerate. Import paths
from here instead of rebuilding them from __file__ in each script.
"""

from __future__ import annotations

from typing import Any

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

TEJ_DIR = RAW_DATA_DIR / "TEJ"  # see data/raw/TEJ/data.yaml
CACHE_DIR = INTERIM_DATA_DIR  # {code}_{year}.csv bars + 0050_benchmark.csv

# MLflow store. SQLite backend because MLflow 3 no longer accepts qlib's plain
# file store (file:./mlruns); artifacts stay ordinary folders under ARTIFACT_ROOT.
MLRUNS_DIR = PROJ_ROOT / "mlruns"
TRACKING_URI = "sqlite:///" + (MLRUNS_DIR / "mlflow.db").as_posix()
ARTIFACT_ROOT = MLRUNS_DIR / "artifacts"  # <experiment>/<run_id>/artifacts/
LOGS_DIR = PROJ_ROOT / "logs"

# ---------- Experiment date windows ----------
# Bars loaded for training. Mirrors the split in data/raw/TEJ/data.yaml.
HISTORY_START = "2016-01-01"
HISTORY_END = "2026-05-23"

# Three operational phases, each consuming one of Algorithm 1's meta-sets:
#   - PRETRAIN phase (2016-2021)         → consumes the meta_train_set
#   - VALIDATION phase (2022-2023)       → consumes the meta_valid_set
#       Dual-purpose per the paper: (a) during pretrain, held-out for
#       early-stop / best-state restore (read-only via train=False);
#       (b) after pretrain, re-fed through DoubleAdapt for the "final
#       preparation" warmup IL with real bi-level updates.
#   - ONLINE_TRAINING phase (2024-now)   → consumes the meta_test_set
#       The qlib backtest fine-tunes the model every r=20 days while
#       simulating the portfolio (Algorithm 1 line 9 second half).
PRETRAIN_START = "2016-01-01"
PRETRAIN_END = "2021-12-31"  # 6 years
VALIDATION_START = "2022-01-01"
VALIDATION_END = "2023-12-31"  # 2 years -- held-out val + warmup IL
ONLINE_TRAINING_START = "2024-01-01"  # live-IL backtest starts here
ONLINE_TRAINING_END = "auto"  # extend to panel max

# Bars loaded for live inference (deploy_today.py). Only needs enough history
# for one 60-day feature window, so it is not tied to the training split.
LIVE_HISTORY_START = "2024-01-01"

# ---------- Default backtest config ----------
# Keyword arguments for evaluate.backtest_qlib.QlibBTConfig used by every
# training run (written to each run's bt_config.json). A plain dict so this
# module stays free of imports from the rest of src/.
DEFAULT_BT: dict[str, Any] = dict(top_k=10, n_drop=1, cost_alpha_bps=70.0, costs="twse", deal_price="close")
