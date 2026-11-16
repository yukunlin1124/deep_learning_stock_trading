"""The run-folder contract: `models/run{N}/`, the `CURRENT` pointer, and the
artifact files that training writes and deploy / extend_il / the course
bridge read back.

Every script that creates, finds, or reads a run goes through here, so the
file names and formats live in one place. Formats must stay readable by
run1..run18, which were written before this module existed: do not rename
files or JSON keys.

Artifacts handled here (others, like report.json and the per-stage logs,
differ per script and are written inline):
    norm_stats.pkl        RobustZScoreStats (pickled by module path: src.data.processor)
    feature_scheme.json   {"mode", "seq_len", "n_fields", "fields", ...}
    bt_config.json        asdict(QlibBTConfig)
    active_universe.json  {"asof", "codes"} from the last universe rebalance
    equity.csv / trades.csv / backtest_daily.csv   BTResult frames
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import asdict
from typing import Any
from pathlib import Path

import pandas as pd

from src.config import LOGS_DIR, MODELS_DIR
from src.data.handler import FIELDS, N_FIELDS, SEQ_LEN
from src.utils.progress import Tee

CURRENT_FILE = MODELS_DIR / "CURRENT"


# ---------- locating runs ----------


def next_run_dir(base: Path = MODELS_DIR) -> Path:
    """Next free models/runN/. Empty runN/ dirs count as free, so killed runs
    don't leave permanent gaps in the numbering."""
    nums = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("run") and p.name[3:].isdigit():
            if any(p.iterdir()):
                nums.append(int(p.name[3:]))
    next_n = (max(nums) + 1) if nums else 1
    return base / f"run{next_n}"


def read_current() -> str | None:
    """Name of the run CURRENT points at, or None if there is no pointer."""
    return CURRENT_FILE.read_text().strip() if CURRENT_FILE.exists() else None


def set_current(run_dir: Path) -> None:
    CURRENT_FILE.write_text(run_dir.name)


def resolve_run(run: str | None = None) -> Path:
    """Run dir to read: `run` if given (CLI override), else the CURRENT pointer."""
    if run:
        d = MODELS_DIR / run
    else:
        name = read_current()
        if name is None:
            raise SystemExit(f"missing {CURRENT_FILE}; pass --run runN or run train.py first.")
        d = MODELS_DIR / name
    if not d.exists():
        raise SystemExit(f"run dir {d} does not exist.")
    return d


# ---------- starting a run ----------


def tee_session(log_path: Path, mode: str = "w", header: str | None = None) -> None:
    """Copy all further stdout/stderr into log_path (line-buffered).

    mode="a" appends (deploy_today.py keeps one log per calendar day);
    `header` is written to the file only, before teeing starts.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Kept open for the life of the process: the Tee writes to it until exit.
    log_file = open(log_path, mode, encoding="utf-8", buffering=1)  # pylint: disable=consider-using-with
    if header:
        log_file.write(header)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)


def init_run(run_name: str | None = None, *, set_as_current: bool = True) -> Path:
    """Create the run dir (named, or the next free runN), optionally point
    CURRENT at it, and tee the session to logs/train_<run>_<timestamp>.log.

    Pass set_as_current=False for runs deploy_today.py can't load (e.g. the
    LightGBM baseline), so CURRENT keeps pointing at a deployable model.
    """
    out_dir = MODELS_DIR / run_name if run_name else next_run_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    if set_as_current:
        set_current(out_dir)
    log_path = LOGS_DIR / f"train_{out_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    tee_session(log_path)
    print(f"writing artifacts to {out_dir}")
    print(f"logging full session to {log_path}")
    return out_dir


# ---------- writing artifacts ----------


def save_norm_stats(run_dir: Path, norm_stats: Any) -> None:
    with open(run_dir / "norm_stats.pkl", "wb") as f:
        pickle.dump(norm_stats, f)


def save_feature_scheme(run_dir: Path, mode: str, **extra: Any) -> None:
    """Record the Alpha360 normalization so deploy_today.py rebuilds matching
    features. Runs without this file are legacy "per_field" (runs 1-16)."""
    (run_dir / "feature_scheme.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "seq_len": SEQ_LEN,
                "n_fields": N_FIELDS,
                "fields": list(FIELDS),
                **extra,
            },
            indent=2,
        )
    )


def save_bt_config(run_dir: Path, bt_cfg: Any) -> None:
    (run_dir / "bt_config.json").write_text(json.dumps(asdict(bt_cfg), indent=2))


def save_active_universe(run_dir: Path, schedule: pd.DataFrame, fallback_codes: list[str] | None = None) -> None:
    """Codes active at the last rebalance (what deploy_today.py will score)."""
    if len(schedule):
        codes = [str(c) for c in schedule.iloc[-1]["codes"]]
        asof = str(schedule.iloc[-1]["asof"])
    else:
        codes, asof = list(fallback_codes or []), None
    (run_dir / "active_universe.json").write_text(json.dumps({"asof": asof, "codes": codes}, indent=2))


def save_backtest(run_dir: Path, bt: Any) -> None:
    bt.equity.to_csv(run_dir / "equity.csv", index=False)
    bt.trades.to_csv(run_dir / "trades.csv", index=False)
    bt.daily.to_csv(run_dir / "backtest_daily.csv", index=False)


# ---------- reading artifacts ----------


def load_norm_stats(run_dir: Path) -> Any:
    with open(run_dir / "norm_stats.pkl", "rb") as f:
        return pickle.load(f)


def load_bt_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "bt_config.json").read_text())


def load_active_universe(run_dir: Path) -> list[str]:
    return json.loads((run_dir / "active_universe.json").read_text())["codes"]


def load_feature_mode(run_dir: Path) -> str:
    """Alpha360 scheme of the run; legacy runs without the file are per_field."""
    path = run_dir / "feature_scheme.json"
    if path.exists():
        return json.loads(path.read_text()).get("mode", "qlib")
    return "per_field"
