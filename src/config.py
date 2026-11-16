"""Project paths, following the cookiecutter-data-science layout.

    data/raw/        immutable inputs (TEJ exports). Never edited by code.
    data/interim/    derived, rebuildable data (per-stock bar cache from import_tej.py)
    data/processed/  final model-ready datasets (unused for now)
    data/external/   third-party data that isn't TEJ (unused for now)
    models/          trained runs: weights, predictions, backtest reports
    logs/            session logs (train / deploy)

Everything except data/raw/ is safe to delete and regenerate. Import paths
from here instead of rebuilding them from __file__ in each script.
"""
from __future__ import annotations

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

TEJ_DIR = RAW_DATA_DIR / "TEJ"       # see data/raw/TEJ/data.yaml
CACHE_DIR = INTERIM_DATA_DIR         # {code}_{year}.csv bars + 0050_benchmark.csv

MODELS_DIR = PROJ_ROOT / "models"    # run{N}/ + CURRENT pointer
LOGS_DIR = PROJ_ROOT / "logs"
