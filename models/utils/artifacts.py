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


def dataset_path(artifacts_root: Path, run_id: str, *, ext: str = "parquet") -> Path:
    return run_dir(artifacts_root, run_id) / f"dataset.{ext}"


def dataset_meta_path(artifacts_root: Path, run_id: str) -> Path:
    return run_dir(artifacts_root, run_id) / "dataset_meta.json"
