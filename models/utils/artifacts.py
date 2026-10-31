"""Artifact path helpers shared across model pipelines."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_artifacts_base() -> Path:
    """Absolute artifacts/ directory (honours NCAR_ARTIFACTS_ROOT)."""
    override = os.environ.get("NCAR_ARTIFACTS_ROOT")
    if override:
        return Path(override)
    return _REPO_ROOT / "artifacts"


def default_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run_dir(artifacts_root: Path, run_id: str) -> Path:
    path = artifacts_root / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def plots_dir(artifacts_root: Path, run_id: str) -> Path:
    path = run_dir(artifacts_root, run_id) / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def anomaly_plots_dir(artifacts_root: Path, run_id: str) -> Path:
    path = plots_dir(artifacts_root, run_id) / "anomalies"
    path.mkdir(parents=True, exist_ok=True)
    return path


def class_summary_plots_dir(artifacts_root: Path, run_id: str) -> Path:
    path = plots_dir(artifacts_root, run_id) / "class_summary"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reconstruction_plots_dir(artifacts_root: Path, run_id: str) -> Path:
    path = plots_dir(artifacts_root, run_id) / "reconstruction"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_path(artifacts_root: Path, run_id: str, *, ext: str = "parquet") -> Path:
    return run_dir(artifacts_root, run_id) / f"dataset.{ext}"


def dataset_meta_path(artifacts_root: Path, run_id: str) -> Path:
    return run_dir(artifacts_root, run_id) / "dataset_meta.json"


def scaler_path(artifacts_root: Path, run_id: str) -> Path:
    return run_dir(artifacts_root, run_id) / "scaler.json"


def model_path(artifacts_root: Path, run_id: str) -> Path:
    return run_dir(artifacts_root, run_id) / "model.pt"


def train_log_path(artifacts_root: Path, run_id: str) -> Path:
    return run_dir(artifacts_root, run_id) / "train_log.json"


def scores_path(artifacts_root: Path, run_id: str) -> Path:
    return run_dir(artifacts_root, run_id) / "scores.parquet"


def thresholds_path(artifacts_root: Path, run_id: str) -> Path:
    return run_dir(artifacts_root, run_id) / "thresholds.json"


def anomalies_path(artifacts_root: Path, run_id: str) -> Path:
    return run_dir(artifacts_root, run_id) / "anomalies.json"


def metrics_path(artifacts_root: Path, run_id: str) -> Path:
    return run_dir(artifacts_root, run_id) / "metrics.json"


def comparison_path(artifacts_root: Path, run_id: str) -> Path:
    return run_dir(artifacts_root, run_id) / "comparison.json"
