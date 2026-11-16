"""Runs, tracked with MLflow the way qlib's recorder does it.

Each training run is one MLflow run (named run1, run2, ...) in an experiment
per workflow: "doubleadapt" (train.py), "il_baseline", "lgbm_baseline". Like a
qlib recorder, a run records the command line (param `cmd-sys.argv`), the
uncommitted code (code_diff.txt / code_status.txt), per-epoch and per-step
metrics, and every artifact. Unlike qlib, the console log (train.log) is an
artifact too, so model and log live together.

The run's artifact folder is a plain local directory, so the rest of the code
reads and writes files there by path: `init_run` and `resolve_run` return it.
Browse runs with:  mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

The run deploy_today.py trades with carries the tag `current=true`.

Artifacts handled here (others, like report.json and the per-stage logs,
differ per script and are written inline). Formats must stay readable by
run1..run20, written before MLflow was adopted: do not rename files or keys.
    norm_stats.pkl        RobustZScoreStats (pickled by module path: src.data.processor)
    feature_scheme.json   {"mode", "seq_len", "n_fields", "fields", ...}
    bt_config.json        asdict(QlibBTConfig)
    active_universe.json  {"asof", "codes"} from the last universe rebalance
    equity.csv / trades.csv / backtest_daily.csv   BTResult frames
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any
from pathlib import Path

import pandas as pd

from src.config import ARTIFACT_ROOT, MLRUNS_DIR, PROJ_ROOT, TRACKING_URI
from src.data.handler import FIELDS, N_FIELDS, SEQ_LEN
from src.utils.progress import Tee

CURRENT_TAG = "current"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # mlflow 3 prints an assistant hint on import


def _mlflow() -> Any:
    """Import mlflow lazily (it is slow to import) and point it at our store."""
    import mlflow  # pylint: disable=import-outside-toplevel

    if mlflow.get_tracking_uri() != TRACKING_URI:
        MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow


def get_experiment_id(name: str) -> str:
    """Experiment id for `name`, creating it with a local artifact folder if needed."""
    mlflow = _mlflow()
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return mlflow.create_experiment(name, artifact_location=(ARTIFACT_ROOT / name).as_uri())


def artifact_dir(run: Any) -> Path:
    """Local folder holding a run's artifacts (created if missing)."""
    from mlflow.utils.file_utils import local_file_uri_to_path  # pylint: disable=import-outside-toplevel

    d = Path(local_file_uri_to_path(run.info.artifact_uri))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _all_runs(filter_string: str = "") -> list:
    mlflow = _mlflow()
    exp_ids = [e.experiment_id for e in mlflow.search_experiments()]
    if not exp_ids:
        return []
    return mlflow.search_runs(
        experiment_ids=exp_ids, filter_string=filter_string, output_format="list", order_by=["start_time DESC"]
    )


# ---------- locating runs ----------


def next_run_name() -> str:
    """Next free runN across all experiments, so run names stay globally unique."""
    nums = [int(r.info.run_name[3:]) for r in _all_runs() if r.info.run_name and r.info.run_name[3:].isdigit()]
    return f"run{(max(nums) + 1) if nums else 1}"


def find_run(name: str) -> Any | None:
    """Newest MLflow run named `name`, or None."""
    found = _all_runs(f"tags.mlflow.runName = '{name}'")
    return found[0] if found else None


def read_current() -> str | None:
    """Name of the run tagged current=true, or None."""
    found = _all_runs(f"tags.{CURRENT_TAG} = 'true'")
    return found[0].info.run_name if found else None


def set_current(run_id: str) -> None:
    """Make `run_id` the run deploy_today.py trades with (clears the old tag)."""
    client = _mlflow().MlflowClient()
    for r in _all_runs(f"tags.{CURRENT_TAG} = 'true'"):
        if r.info.run_id != run_id:
            client.set_tag(r.info.run_id, CURRENT_TAG, "false")
    client.set_tag(run_id, CURRENT_TAG, "true")


def resolve_run(run: str | None = None) -> Path:
    """Artifact folder to read: run `run` if given (CLI override), else the current run."""
    name = run or read_current()
    if name is None:
        raise SystemExit("no current run; pass --run runN or run train.py first.")
    found = find_run(name)
    if found is None:
        raise SystemExit(f"no MLflow run named {name!r} in {MLRUNS_DIR}.")
    return artifact_dir(found)


def list_runs() -> dict[str, Path]:
    """{run name: artifact folder} for every run, oldest first."""
    return {r.info.run_name: artifact_dir(r) for r in reversed(_all_runs())}


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


def _log_uncommitted_code(out_dir: Path) -> None:
    """qlib's MLflowRecorder._log_uncommitted_code: save what git can't recover later."""
    for cmd, fname in [
        (["git", "diff"], "code_diff.txt"),
        (["git", "status"], "code_status.txt"),
        (["git", "diff", "--cached"], "code_cached.txt"),
    ]:
        try:
            # Bytes + explicit UTF-8: the diff has non-ASCII text, which the Windows locale codec can't decode.
            raw = subprocess.run(cmd, cwd=PROJ_ROOT, capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            continue  # no git / not a repo: record nothing rather than fail the run
        out = raw.decode("utf-8", errors="replace")
        if out.strip():
            (out_dir / fname).write_text(out, encoding="utf-8")


def init_run(run_name: str | None = None, *, experiment: str, set_as_current: bool = True) -> Path:
    """Start an MLflow run (named, or the next free runN) and return its artifact folder.

    Records the command and uncommitted code like a qlib recorder, tees the
    session to <artifacts>/train.log, and (unless set_as_current=False, for
    runs deploy_today.py can't load, e.g. the LightGBM baseline) tags it as
    the current run. The run ends when the process exits; an uncaught
    exception marks it FAILED.
    """
    mlflow = _mlflow()
    name = run_name or next_run_name()
    if find_run(name) is not None:
        raise SystemExit(f"a run named {name!r} already exists; omit --run to use the next free name.")
    mlflow.set_experiment(experiment_id=get_experiment_id(experiment))
    run = mlflow.start_run(run_name=name)

    # Installed first, so a crash anywhere after start_run marks the run FAILED
    # (otherwise mlflow's exit hook would record it as FINISHED).
    prev_hook = sys.excepthook

    def _fail_run(*exc_info: Any) -> None:
        mlflow.end_run(status="FAILED")
        prev_hook(*exc_info)

    sys.excepthook = _fail_run

    mlflow.log_param("cmd-sys.argv", " ".join(sys.argv))
    out_dir = artifact_dir(run)
    _log_uncommitted_code(out_dir)
    if set_as_current:
        set_current(run.info.run_id)
    tee_session(out_dir / "train.log")
    print(f"MLflow run {name} (experiment {experiment!r}, id {run.info.run_id})")
    print(f"writing artifacts + train.log to {out_dir}")
    return out_dir


# ---------- logging params / metrics (no-ops outside an MLflow run) ----------


def _active() -> Any | None:
    if "mlflow" not in sys.modules:  # nothing started a run in this process
        return None
    return sys.modules["mlflow"].active_run()


def _flatten(d: Mapping, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def log_metrics(metrics: Mapping[str, Any], step: int | None = None) -> None:
    """Log finite numeric values as MLflow metrics; silently skipped outside a run."""
    if _active() is None:
        return
    clean = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and pd.notna(v)
    }
    if clean:
        sys.modules["mlflow"].log_metrics(clean, step=step)


def log_params(params: Mapping[str, Any]) -> None:
    """Log (nested) params, flattened with dotted keys; skipped outside a run."""
    if _active() is None:
        return
    flat = {k: str(v)[:6000] for k, v in _flatten(params).items()}
    if flat:
        sys.modules["mlflow"].log_params(flat)


# report.json sections that hold configuration: logged as params even when numeric.
CONFIG_SECTIONS = ("model_config.", "backtest_cfg.", "lgb_params.")


def split_report(report: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    """Split a report.json-style dict into (metrics, params).

    Numeric results become metrics (e.g. backtest.sharpe, il_overall_ic); config
    sections and non-numeric values become params, as qlib logs config. Long
    lists (universe_codes) stay in report.json only.
    """
    flat = _flatten(report)
    metrics = {
        k: v
        for k, v in flat.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not k.startswith(CONFIG_SECTIONS)
    }
    params = {k: v for k, v in flat.items() if k not in metrics and not (isinstance(v, list) and len(v) > 20)}
    return metrics, params


def log_report(report: Mapping[str, Any]) -> None:
    """Log a run's report.json as MLflow metrics + params; skipped outside a run."""
    if _active() is None:
        return
    metrics, params = split_report(report)
    log_metrics(metrics)
    log_params(params)


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
