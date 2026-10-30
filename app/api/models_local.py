"""Read-only API for local model run artifacts (multi-model registry)."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.api.regions import MAX_POINTS_HINT

logger = logging.getLogger("traffic_monitor.models_local")

from models.TimesNet import config as _timesnet_cfg

MODEL_TIMESNET = "timesnet"

FAMILY_ANOMALY = "anomaly"
FAMILY_FORECAST = "forecasting"


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata for a model whose artifacts we can browse read-only."""

    id: str
    label: str
    family: str
    subdir: str  # relative to the artifacts base (NCAR_ARTIFACTS_ROOT or repo/artifacts)
    tagline: str  # one-line summary for the model-selection tile
    focus: str  # what the run is good at / what the viewer emphasises


MODEL_REGISTRY: dict[str, ModelSpec] = {
    MODEL_TIMESNET: ModelSpec(
        MODEL_TIMESNET,
        "TimesNet",
        FAMILY_ANOMALY,
        "anomaly/timesnet",
        "Periodicity-aware anomaly detection",
        "2D temporal blocks capture multi-period structure; a traffic-domain "
        "filter refines the statistical anomaly gates.",
    ),
}


def _spec(model_id: str) -> ModelSpec:
    spec = MODEL_REGISTRY.get((model_id or "").strip().lower())
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id!r}")
    return spec


def _require_family(model_id: str, family: str) -> ModelSpec:
    spec = _spec(model_id)
    if spec.family != family:
        raise HTTPException(
            status_code=400,
            detail=f"Model {spec.id!r} is a {spec.family} model, not {family}",
        )
    return spec


_REPO_ROOT = Path(__file__).resolve().parents[2]

_GT_EVENTS_CSV = _REPO_ROOT / "models" / "data" / "gt_anomalies" / "gt_events.csv"
_ANOMALY_MODES = frozenset({"percentile", "combined"})
_DEFAULT_ANOMALY_MODE = "combined"
_COMBINED_ALERT_Z = _timesnet_cfg.EVAL_COMBINED_ALERT_Z
_PERCENTILE_QUANTILE = 0.999


def _default_combined_robust_z() -> float:
    return float(_timesnet_cfg.EVAL_COMBINED_ROBUST_Z)


def _combined_robust_z_for_run(run_path: Path) -> float:
    """Read robust_z gate used at eval time (falls back to config default)."""
    for name in ("anomalies_combined.json", "metrics.json"):
        data = _read_json(run_path / name)
        if not data:
            continue
        z = data.get("params", {}).get("robust_z") if name.startswith("anomalies") else None
        if z is None and name == "metrics.json":
            z = (data.get("meta") or {}).get("combined_robust_z")
        if z is not None:
            return float(z)
    return _default_combined_robust_z()


_SEVERITY_Z_ALERT = 3.5 + 1.5


def _artifacts_base() -> Path:
    """Root that holds the anomaly/ and forecasting/ family folders."""
    override = os.environ.get("NCAR_ARTIFACTS_ROOT")
    if override:
        return Path(override)
    return _REPO_ROOT / "artifacts"


def _artifacts_root(model_id: str = MODEL_TIMESNET) -> Path:
    return _artifacts_base() / _spec(model_id).subdir


def _normalize_run_id(run_id: str) -> str:
    """Strip accidental API suffixes if a client appended /series or /scores to run_id."""
    normalized = run_id.strip().replace("\\", "/")
    for suffix in ("/series", "/scores", "/anomalies", "/run"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized.strip("/")


def _resolve_run_dir(run_id: str, model_id: str = MODEL_TIMESNET) -> Path:
    """Resolve run_id as a path relative to artifacts root (supports v5/weather_calendar)."""
    normalized = _normalize_run_id(run_id)
    if not normalized or ".." in normalized.split("/"):
        raise HTTPException(status_code=400, detail="Invalid run_id")

    root = _artifacts_root(model_id).resolve()
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run_id") from exc

    return candidate


def _normalize_anomaly_mode(mode: str | None) -> str:
    normalized = (mode or _DEFAULT_ANOMALY_MODE).strip().lower()
    if normalized not in _ANOMALY_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode={mode!r}; expected percentile or combined",
        )
    return normalized


def _anomalies_json_path(run_path: Path, mode: str) -> Path:
    if mode == "combined":
        candidate = run_path / "anomalies_combined.json"
    else:
        candidate = run_path / "anomalies_percentile.json"
    if candidate.is_file():
        return candidate
    legacy = run_path / "anomalies.json"
    if legacy.is_file():
        return legacy
    raise HTTPException(
        status_code=404,
        detail=f"No anomalies file for mode={mode!r} in {run_path.name}",
    )


def _thresholds_json_path(run_path: Path, mode: str) -> Path | None:
    if mode == "combined":
        path = run_path / "thresholds_robust.json"
    else:
        path = run_path / "thresholds_percentile.json"
    if path.is_file():
        return path
    legacy = run_path / "thresholds.json"
    return legacy if legacy.is_file() else None


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_has_eval(run_dir: Path) -> bool:
    return (run_dir / "comparison.json").is_file() or (run_dir / "metrics.json").is_file()


def _iso_mtime(path: Path) -> str | None:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except OSError:
        return None


def _summary_from_run_dir(run_dir: Path) -> dict:
    comp = _read_json(run_dir / "comparison.json")
    metrics = _read_json(run_dir / "metrics.json")
    summary = (comp or {}).get("summary") or {}
    if summary.get("anomaly_rate") is None and metrics:
        for row in metrics.get("splits") or []:
            if row.get("split") == "test":
                return {
                    "anomaly_rate": row.get("anomaly_rate"),
                    "series_count": row.get("series"),
                    "point_count": row.get("points"),
                }
    return summary


def _discover_run_dirs(root: Path, markers: tuple[str, ...]) -> list[tuple[str, Path]]:
    """Return (run_id, path) for every run under root that has any marker file."""
    if not root.is_dir():
        return []

    seen: set[str] = set()
    found: list[tuple[str, Path]] = []

    for marker in markers:
        for path in root.rglob(marker):
            if not path.is_file():
                continue
            run_dir = path.parent
            rel = run_dir.relative_to(root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            found.append((rel, run_dir))

    found.sort(key=lambda item: item[1].stat().st_mtime, reverse=True)
    return found


_ANOMALY_RUN_MARKERS = ("comparison.json", "metrics.json")


def _forecast_run_markers(model_id: str) -> tuple[str, ...]:
    return (
        "comparison.json",
        f"{model_id}_metrics.json",
        "baseline_metrics.json",
        f"{model_id}_predictions.csv",
    )


def _anomaly_run_summary(run_id: str, run_dir: Path) -> dict:
    summary = _summary_from_run_dir(run_dir)
    comp = _read_json(run_dir / "comparison.json")
    return {
        "run_id": run_id,
        "modified_at": _iso_mtime(run_dir / "comparison.json")
        or _iso_mtime(run_dir / "metrics.json")
        or _iso_mtime(run_dir),
        "anomaly_rate": summary.get("anomaly_rate"),
        "series_count": summary.get("series_count"),
        "eval_split": (comp or {}).get("eval_split"),
    }


def _forecast_run_summary(run_id: str, run_dir: Path, model_id: str) -> dict:
    comp = _read_json(run_dir / "comparison.json")
    metrics = _read_json(run_dir / f"{model_id}_metrics.json")
    baseline = _read_json(run_dir / "baseline_metrics.json")
    macro = (
        (comp or {}).get(f"{model_id}_macro")
        or (metrics or {}).get("macro")
        or (comp or {}).get("baseline_macro")
        or (baseline or {}).get("macro")
        or {}
    )
    return {
        "run_id": run_id,
        "modified_at": _iso_mtime(run_dir / "comparison.json")
        or _iso_mtime(run_dir / f"{model_id}_metrics.json")
        or _iso_mtime(run_dir / "baseline_metrics.json")
        or _iso_mtime(run_dir),
        "mae": macro.get("mae"),
        "rmse": macro.get("rmse"),
        "smape": macro.get("smape"),
        "series_count": macro.get("series_count"),
        "has_eval": (run_dir / "comparison.json").is_file(),
    }


def list_local_models() -> dict:
    """List every registered model with its discovered runs (read-only)."""
    models_out: list[dict] = []
    for spec in MODEL_REGISTRY.values():
        root = _artifacts_root(spec.id)
        if spec.family == FAMILY_ANOMALY:
            runs = [
                _anomaly_run_summary(rid, rd)
                for rid, rd in _discover_run_dirs(root, _ANOMALY_RUN_MARKERS)
            ]
        else:
            runs = [
                _forecast_run_summary(rid, rd, spec.id)
                for rid, rd in _discover_run_dirs(root, _forecast_run_markers(spec.id))
            ]
        models_out.append(
            {
                "id": spec.id,
                "label": spec.label,
                "family": spec.family,
                "tagline": spec.tagline,
                "focus": spec.focus,
                "artifacts_root": str(root),
                "run_count": len(runs),
                "runs": runs,
            }
        )
    return {"models": models_out}


def _filter_series_with_anomalies(rows: list[dict]) -> list[dict]:
    return [r for r in rows if int(r.get("n_anomalies") or 0) > 0]


_SCORE_COLUMNS = (
    "ds",
    "unique_id",
    "score",
    "score_macro",
    "score_micro",
    "split",
    "pred_val",
    "recon_val",
    "actual_val",
)
_LOCATION_WIDE_V1 = "location_wide_v1"
_MACRO_MARKER_CLASS = -1

_VEHICLE_CLASS_LABELS: dict[int, str] = {
    1: "motorcycle",
    2: "car+trailer",
    3: "truck",
    5: "bus",
    7: "car",
    8: "truck+trailer",
    9: "tractor+trailer",
    10: "unknown-10",
    11: "LCV",
}

_CLASS_COLORS: dict[int, str] = {
    1: "#999999",
    2: "#56B4E9",
    3: "#D55E00",
    5: "#CC79A7",
    7: "#009E73",
    8: "#F0E442",
    9: "#0072B2",
    10: "#E69F00",
    11: "#882255",
}
_FALLBACK_CLASS_COLOR = "#AAAAAA"


def _ds_lookup_key(ds: Any) -> str:
    ts = pd.Timestamp(ds)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _build_anomaly_lookup(anomalies_doc: dict) -> dict[tuple[str, str], dict]:
    """Map (unique_id, ds_iso) -> anomaly record from eval JSON (source of truth)."""
    lookup: dict[tuple[str, str], dict] = {}
    for rec in anomalies_doc.get("anomalies") or []:
        uid = str(rec.get("unique_id", ""))
        ds_raw = rec.get("ds")
        if not uid or ds_raw is None:
            continue
        lookup[(uid, _ds_lookup_key(ds_raw))] = rec
    return lookup


def _chart_anomaly_lookup(run_path: Path, mode: str) -> dict[tuple[str, str], dict] | None:
    doc = _read_json(_anomalies_json_path(run_path, mode))
    if not doc:
        return None
    return _build_anomaly_lookup(doc)


def _thresholds_run_path(run_path: Path) -> Path:
    """Resolve the directory containing threshold JSON."""
    return run_path


def _cutoff_by_uid(run_path: Path) -> dict[str, float]:
    th_root = _thresholds_run_path(run_path)
    if _location_wide_meta(run_path) is not None:
        out: dict[str, float] = {}
        for name in ("thresholds_macro.json", "thresholds_micro.json"):
            th = _read_json(th_root / name) or _read_json(run_path / name) or {}
            for uid, info in (th.get("per_target") or {}).items():
                out[str(uid)] = float(info.get("cutoff", 0.0))
        return out
    th = (
        _read_json(th_root / "thresholds_percentile.json")
        or _read_json(th_root / "thresholds.json")
        or {}
    )
    return {
        str(uid): float(info.get("cutoff", 0.0))
        for uid, info in (th.get("per_target") or {}).items()
    }


def _score_column_for_uid(uid: str, columns: pd.Index, location_wide: bool) -> str:
    if not location_wide:
        return "score"
    if str(uid).endswith("__total") and "score_macro" in columns:
        return "score_macro"
    if "__cls" in str(uid) and "score_micro" in columns:
        return "score_micro"
    return "score"


def _merge_anomaly_flags(
    df: pd.DataFrame,
    lookup: dict[tuple[str, str], dict],
    cutoffs: dict[str, float],
    *,
    location_wide: bool,
) -> pd.DataFrame:
    out = df.copy()
    uids = out["unique_id"].astype(str)
    ds_keys = out["ds"].map(_ds_lookup_key)
    keys = list(zip(uids, ds_keys, strict=False))

    out["is_anomaly"] = [key in lookup for key in keys]
    out["severity"] = [str((lookup.get(key) or {}).get("severity") or "") for key in keys]
    out["z"] = np.array(
        [float((lookup.get(key) or {}).get("z", 0.0)) for key in keys],
        dtype=np.float32,
    )
    out["cutoff"] = np.array(
        [float(cutoffs.get(uid, 0.0)) for uid in uids],
        dtype=np.float32,
    )

    score_col = _score_column_for_uid(str(uids.iloc[0]), out.columns, location_wide)
    display_scores = out[score_col].to_numpy(dtype=np.float64, copy=True)
    for i, key in enumerate(keys):
        rec = lookup.get(key)
        if rec is not None and rec.get("score") is not None:
            display_scores[i] = float(rec["score"])
    out["score"] = display_scores.astype(np.float32)
    return out


CHART_WINDOWS = frozenset({"full", "test", "tail7d", "tail30d"})


def _normalize_chart_window(window: str | None) -> str:
    w = (window or "full").strip().lower()
    if w not in CHART_WINDOWS:
        raise HTTPException(
            status_code=400,
            detail=f"window must be one of: {', '.join(sorted(CHART_WINDOWS))}",
        )
    return w


def _filter_df_chart_window(
    df: pd.DataFrame,
    window: str,
    *,
    time_col: str = "ds",
) -> pd.DataFrame:
    """Restrict chart rows to the UI window before downsampling."""
    if window == "full" or df.empty:
        return df
    out = df.sort_values(time_col)
    if window == "test":
        if "split" not in out.columns:
            return out
        test = out[out["split"].astype(str) == "test"]
        return test if not test.empty else out
    days = 7 if window == "tail7d" else 30
    ts = pd.to_datetime(out[time_col], utc=True)
    end = ts.max()
    cutoff = end - pd.Timedelta(days=days)
    filtered = out.loc[ts >= cutoff]
    return filtered if not filtered.empty else out


def _split_ranges_from_labels(labels: Sequence[str]) -> list[dict[str, Any]]:
    """Contiguous index ranges for train/val/test labels on a chart axis."""
    if not labels:
        return []
    ranges: list[dict[str, Any]] = []
    current = str(labels[0])
    start = 0
    for i, raw in enumerate(labels[1:], start=1):
        split_name = str(raw)
        if split_name == current:
            continue
        ranges.append({"split": current, "start": start, "end": i - 1})
        current = split_name
        start = i
    ranges.append({"split": current, "start": start, "end": len(labels) - 1})
    return ranges


def _downsample_preserving(
    df: pd.DataFrame,
    max_points: int,
    *,
    time_col: str = "ds",
) -> pd.DataFrame:
    """Reduce point count but keep every flagged timestamp."""
    if len(df) <= max_points:
        return df.sort_values(time_col).reset_index(drop=True)
    df = df.sort_values(time_col)
    anom = df[df["is_anomaly"]]
    normal = df[~df["is_anomaly"]]
    budget = max_points - len(anom)
    if budget <= 0:
        return anom.head(max_points).reset_index(drop=True)
    if len(normal) <= budget:
        return pd.concat([anom, normal]).sort_values(time_col).reset_index(drop=True)
    # Keep first/last timesteps so the chart still reaches the dataset boundary.
    if budget >= 2 and len(normal) > 2:
        ends = normal.iloc[[0, len(normal) - 1]]
        middle = normal.iloc[1:-1]
        mid_budget = max(0, budget - 2)
        if len(middle) <= mid_budget:
            normal_sampled = pd.concat([ends, middle])
        else:
            step = max(1, len(middle) // mid_budget)
            normal_sampled = pd.concat([ends, middle.iloc[::step].head(mid_budget)])
        normal_sampled = normal_sampled.drop_duplicates(subset=[time_col], keep="first")
    else:
        step = max(1, len(normal) // budget)
        normal_sampled = normal.iloc[::step].head(budget)
    return pd.concat([anom, normal_sampled]).sort_values(time_col).reset_index(drop=True)


def _downsample_index_preserving(
    index: pd.DatetimeIndex,
    must_keep: pd.DatetimeIndex,
    max_points: int,
) -> pd.DatetimeIndex:
    if len(index) <= max_points:
        return index
    boundaries = pd.DatetimeIndex([index[0], index[-1]])
    must_keep = must_keep.union(boundaries).intersection(index).sort_values()
    if len(must_keep) >= max_points:
        return must_keep[:max_points]
    budget = max_points - len(must_keep)
    others = index.difference(must_keep).sort_values()
    if len(others) <= budget:
        return must_keep.union(others).sort_values()
    step = max(1, len(others) // budget)
    sampled = others[::step][:budget]
    return must_keep.union(sampled).sort_values()


def _present_score_columns(scores_path: Path) -> list[str]:
    """Intersect the canonical score columns with the parquet schema.

    Anomaly models differ: TimesNet writes pred_val/score_macro/score_micro,
    while TimesNet may omit them. Requesting a missing column makes
    pandas raise, so we only ask for columns that exist.
    """
    names: set[str] | None = None
    try:
        import pyarrow.parquet as pq

        names = set(pq.ParquetFile(scores_path).schema.names)
    except Exception:
        try:
            names = set(pd.read_parquet(scores_path).columns)
        except Exception:
            names = None
    if not names:
        return list(_SCORE_COLUMNS)
    cols = [c for c in _SCORE_COLUMNS if c in names]
    # ds/unique_id/score/split are required downstream; keep canonical list if any is missing.
    if not {"ds", "unique_id", "score"}.issubset(set(cols)):
        return list(_SCORE_COLUMNS)
    return cols


def _read_scores_for_series(
    scores_path: Path,
    unique_id: str,
    *,
    split: str | None = None,
) -> pd.DataFrame:
    """Read one series from scores.parquet without loading the full file."""
    uid = unique_id.strip()
    columns = _present_score_columns(scores_path)
    filters: list[tuple[str, str, Any]] = [("unique_id", "==", uid)]
    if split and "split" in columns:
        filters.append(("split", "==", split))
    try:
        df = pd.read_parquet(
            scores_path,
            columns=columns,
            filters=filters,
        )
    except Exception as exc:
        logger.warning("Parquet filter read failed for %s: %s; falling back to full scan", uid, exc)
        df = pd.read_parquet(scores_path, columns=columns)
        df = df[df["unique_id"].astype(str) == uid]
        if split and "split" in df.columns:
            df = df[df["split"].astype(str) == split]
    return df


def _read_scores_for_uids(
    scores_path: Path,
    unique_ids: list[str],
    *,
    split: str | None = None,
) -> pd.DataFrame:
    uid_set = {str(u) for u in unique_ids}
    columns = _present_score_columns(scores_path)
    try:
        df = pd.read_parquet(scores_path, columns=columns)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read scores: {exc}") from exc
    df = df[df["unique_id"].astype(str).isin(uid_set)].copy()
    if split and "split" in df.columns:
        df = df[df["split"].astype(str) == split]
    return df


def _json_float(value: float) -> float:
    if isinstance(value, (np.floating, float)) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return float(value)


def _parse_cls(uid: str) -> int | None:
    if "__cls" not in uid:
        return None
    try:
        return int(uid.rsplit("__cls", 1)[1])
    except (ValueError, IndexError):
        return None


def _location_from_uid(uid: str) -> str:
    if "__cls" in uid:
        return uid.rsplit("__cls", 1)[0]
    return uid


def _downsample_index(index: pd.DatetimeIndex, max_points: int) -> pd.DatetimeIndex:
    if len(index) <= max_points:
        return index
    step = max(1, len(index) // max_points)
    return index[::step]


def _load_actual_series(
    dataset_path: Path,
    unique_id: str,
    ds_index: pd.DatetimeIndex,
) -> list[float]:
    col = f"y__{unique_id}"
    try:
        frame = pd.read_parquet(dataset_path, columns=["ds", col])
    except Exception:
        return [0.0] * len(ds_index)
    if col not in frame.columns:
        return [0.0] * len(ds_index)
    frame["ds"] = pd.to_datetime(frame["ds"], utc=True)
    frame = frame.set_index("ds").sort_index()
    aligned = frame[col].reindex(ds_index).fillna(0.0)
    return [_json_float(x) for x in aligned.to_numpy(dtype=np.float64)]


def get_timesnet_run_manifest(run_id: str, model_id: str = MODEL_TIMESNET) -> dict:
    spec = _require_family(model_id, FAMILY_ANOMALY)
    run_path = _resolve_run_dir(run_id, model_id)
    if not run_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id!r}")

    comparison = _read_json(run_path / "comparison.json")
    metrics = _read_json(run_path / "metrics.json")
    dataset_meta = _read_json(run_path / "dataset_meta.json")
    thresholds_pct = _read_json(run_path / "thresholds_percentile.json") or _read_json(
        run_path / "thresholds.json"
    )
    thresholds_rob = _read_json(run_path / "thresholds_robust.json")
    anomalies_combined = _read_json(run_path / "anomalies_combined.json")
    anomalies_pct = _read_json(run_path / "anomalies_percentile.json")
    anomalies_legacy = _read_json(run_path / "anomalies.json")
    gt_eval = _read_json(run_path / "gt_eval.json")

    if not comparison and not metrics:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id!r} has no eval artifacts (comparison.json / metrics.json)",
        )

    artifacts = sorted(
        p.name for p in run_path.iterdir() if p.is_file() and not p.name.startswith(".")
    )

    anomaly_modes: list[str] = []
    if (run_path / "anomalies_combined.json").is_file():
        anomaly_modes.append("combined")
    if (run_path / "anomalies_percentile.json").is_file():
        anomaly_modes.append("percentile")
    if not anomaly_modes and anomalies_legacy:
        anomaly_modes.append("legacy")

    top_series = (comparison or {}).get("top_series_by_rate") or []
    if top_series:
        top_series = _filter_series_with_anomalies(top_series)

    summary_combined = (anomalies_combined or anomalies_legacy or {}).get("summary")
    summary_pct = (anomalies_pct or {}).get("summary")

    schema_version = (dataset_meta or {}).get("schema_version") or ""
    summary = (comparison or {}).get("summary") or summary_combined

    return {
        "model": spec.id,
        "label": spec.label,
        "family": FAMILY_ANOMALY,
        "run_id": run_id,
        "artifacts": artifacts,
        "schema_version": schema_version,
        "has_scores": (run_path / "scores.parquet").is_file(),
        "anomaly_modes": anomaly_modes,
        "default_mode": _DEFAULT_ANOMALY_MODE,
        "has_gt_eval": gt_eval is not None,
        "comparison": comparison,
        "metrics": metrics,
        "dataset_meta": dataset_meta,
        "thresholds_percentile": thresholds_pct,
        "thresholds_robust": thresholds_rob,
        "thresholds": thresholds_pct,
        "summary": summary,
        "summary_percentile": (comparison or {}).get("summary_percentile") or summary_pct,
        "summary_combined": summary_combined,
        "macro_count": (summary or {}).get("macro_count"),
        "micro_count": (summary or {}).get("micro_count"),
        "combined_robust_z": _combined_robust_z_for_run(run_path),
        "threshold": (comparison or {}).get("threshold"),
        "gt_eval": gt_eval,
        "top_series_by_rate": top_series,
        "score_quantiles_by_split": (comparison or {}).get("score_quantiles_by_split"),
    }


def get_timesnet_run_series(
    run_id: str,
    mode: str | None = None,
    model_id: str = MODEL_TIMESNET,
    *,
    run_path: Path | None = None,
) -> dict:
    if run_path is None:
        _require_family(model_id, FAMILY_ANOMALY)
        run_path = _resolve_run_dir(run_id, model_id)
    else:
        run_path = Path(run_path)
    mode = _normalize_anomaly_mode(mode)
    anomalies = _read_json(_anomalies_json_path(run_path, mode))
    if anomalies and anomalies.get("per_series"):
        all_rows = anomalies["per_series"]
        rows = _filter_series_with_anomalies(all_rows)
        # Reconstruction models such as TimesNet may flag nothing; keep the
        # series browsable so its reconstruction chart is still reachable.
        if not rows:
            rows = all_rows
        return {
            "run_id": run_id,
            "mode": mode,
            "per_series": rows,
            "summary": anomalies.get("summary"),
        }

    scores_path = run_path / "scores.parquet"
    if not scores_path.is_file():
        raise HTTPException(status_code=404, detail="No per_series data for this run")

    df = pd.read_parquet(scores_path)
    rows: list[dict] = []
    for uid, group in df.groupby("unique_id", sort=True):
        n_total = int(len(group))
        n_anom = 0
        try:
            ann = _annotate_for_mode(group.copy(), run_path, mode)
            n_anom = int(ann["is_anomaly"].sum())
        except HTTPException:
            raise
        except Exception:
            n_anom = 0
        rows.append(
            {
                "unique_id": str(uid),
                "n": n_total,
                "n_anomalies": n_anom,
                "rate": (n_anom / n_total) if n_total > 0 else 0.0,
                "score_max": float(group["score"].max()) if n_total else 0.0,
                "score_median": float(group["score"].median()) if n_total else 0.0,
                "cutoff": 0.0,
            }
        )
    filtered = _filter_series_with_anomalies(rows)
    return {
        "run_id": run_id,
        "mode": mode,
        "per_series": filtered if filtered else rows,
        "summary": None,
    }


def _threshold_arrays(
    out: pd.DataFrame,
    thresholds: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_target = thresholds.get("per_target") or {}
    method = thresholds.get("method", "robust_z")
    cutoff_col = np.zeros(len(out), dtype=np.float64)
    z_col = np.zeros(len(out), dtype=np.float64)
    for uid, info in per_target.items():
        mask = (out["unique_id"].astype(str) == str(uid)).to_numpy()
        if not mask.any():
            continue
        s = out.loc[mask, "score"].to_numpy(dtype=np.float64)
        cutoff_col[mask] = info["cutoff"]
        if method == "robust_z":
            z_col[mask] = (s - info["median"]) / info["scale"]
        else:
            denom = max(info["cutoff"], 1e-6)
            z_col[mask] = s / denom
    is_anom = out["score"].to_numpy(dtype=np.float64) > cutoff_col
    return cutoff_col, z_col, is_anom


def _annotate_scores(scores: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Lightweight annotate for a single threshold dict (legacy / percentile)."""
    out = scores.copy()
    cutoff_col, z_col, is_anom = _threshold_arrays(out, thresholds)
    out["cutoff"] = cutoff_col.astype(np.float32)
    out["z"] = z_col.astype(np.float32)
    out["is_anomaly"] = is_anom
    method = thresholds.get("method", "robust_z")
    if method == "percentile":
        severity = np.where(~is_anom, "", "warning")
    else:
        abs_z = np.abs(z_col)
        severity = np.where(
            ~is_anom,
            "",
            np.where(abs_z >= _SEVERITY_Z_ALERT, "alert", "warning"),
        )
    out["severity"] = severity
    return out


def _location_wide_meta(run_path: Path) -> dict | None:
    meta = _read_json(run_path / "dataset_meta.json")
    if not meta or meta.get("schema_version") != _LOCATION_WIDE_V1:
        return None
    column_roles = meta.get("column_roles") or {}
    target_columns = meta.get("target_columns") or []
    if not column_roles or not target_columns:
        return None
    return {
        "column_roles": column_roles,
        "target_columns": target_columns,
    }




def _annotate_combined_scores(
    scores: pd.DataFrame,
    thresholds_percentile: dict,
    thresholds_robust: dict,
    run_path: Path,
) -> pd.DataFrame:
    out = scores.copy()
    pct_cutoff, _, is_pct = _threshold_arrays(out, thresholds_percentile)
    _, robust_z, _ = _threshold_arrays(out, thresholds_robust)
    z_confirm = _combined_robust_z_for_run(run_path)
    is_combined = is_pct & (robust_z >= z_confirm)
    out["cutoff"] = pct_cutoff.astype(np.float32)
    out["z"] = robust_z.astype(np.float32)
    out["is_anomaly"] = is_combined
    out["severity"] = np.where(
        ~is_combined,
        "",
        np.where(np.abs(robust_z) >= _COMBINED_ALERT_Z, "alert", "warning"),
    )
    return out


def _annotate_for_mode(
    scores: pd.DataFrame,
    run_path: Path,
    mode: str,
) -> pd.DataFrame:
    mode = _normalize_anomaly_mode(mode)
    th_root = _thresholds_run_path(run_path)
    if mode == "combined":
        th_pct = _read_json(th_root / "thresholds_percentile.json") or _read_json(
            th_root / "thresholds.json"
        )
        th_rob = _read_json(th_root / "thresholds_robust.json")
        if th_pct and th_rob:
            return _annotate_combined_scores(scores, th_pct, th_rob, run_path)
        th = th_rob or th_pct
        if not th:
            raise HTTPException(status_code=404, detail="thresholds not found for combined mode")
        return _annotate_scores(scores, th)
    th_path = _thresholds_json_path(th_root, "percentile")
    th = _read_json(th_path) if th_path else None
    if not th:
        raise HTTPException(status_code=404, detail="thresholds_percentile.json not found")
    return _annotate_scores(scores, th)


def get_timesnet_series_scores(
    run_id: str,
    unique_id: str,
    max_points: int = MAX_POINTS_HINT,
    split: str | None = None,
    window: str | None = None,
    mode: str | None = None,
    model_id: str = MODEL_TIMESNET,
    *,
    run_path: Path | None = None,
) -> dict:
    t0 = time.perf_counter()
    if run_path is None:
        _require_family(model_id, FAMILY_ANOMALY)
        run_path = _resolve_run_dir(run_id, model_id)
    else:
        run_path = Path(run_path)
    scores_path = run_path / "scores.parquet"
    if not scores_path.is_file():
        raise HTTPException(status_code=404, detail="scores.parquet not found")

    mode = _normalize_anomaly_mode(mode)
    chart_window = _normalize_chart_window(window)

    uid = unique_id.strip()
    if not uid:
        raise HTTPException(status_code=400, detail="unique_id is required")

    sub = _read_scores_for_series(scores_path, uid, split=split)
    if sub.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No score rows for split={split!r} on series {uid!r}"
            if split
            else f"Series not found: {uid!r}",
        )

    sub = sub.sort_values("ds")
    if "split" not in sub.columns:
        sub = sub.assign(split=split or "test")
    location_wide = _location_wide_meta(run_path) is not None
    lookup = _chart_anomaly_lookup(run_path, mode)
    if lookup is not None:
        annotated = _merge_anomaly_flags(
            sub,
            lookup,
            _cutoff_by_uid(run_path),
            location_wide=location_wide,
        )
    else:
        annotated = _annotate_for_mode(sub, run_path, mode)

    flags_in_split = int(annotated["is_anomaly"].sum())
    points_total = int(len(sub))

    windowed = _filter_df_chart_window(annotated, chart_window)
    points_in_window = int(len(windowed))

    logger.info(
        "scores run_id=%s unique_id=%s rows=%s window=%s in_window=%s flags=%s read=%.2fs",
        run_id,
        uid,
        points_total,
        chart_window,
        points_in_window,
        flags_in_split,
        time.perf_counter() - t0,
    )

    annotated = _downsample_preserving(windowed, max(50, min(max_points, 8000)))

    flagged = annotated[annotated["is_anomaly"]]
    anomaly_records = [
        {
            "ds": row.ds.isoformat() if hasattr(row.ds, "isoformat") else str(row.ds),
            "score": _json_float(row.score),
            "z": _json_float(row.z),
            "severity": str(row.severity) if row.severity else None,
        }
        for row in flagged.itertuples(index=False)
    ]

    if len(annotated):
        cutoff_arr = annotated["cutoff"].to_numpy(dtype=np.float64)
        positive = cutoff_arr[cutoff_arr > 0]
        cutoff_val = float(positive.max()) if len(positive) else float(cutoff_arr.max())
    else:
        cutoff_val = 0.0
    timestamps = [
        t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "strftime") else str(t)[:16]
        for t in annotated["ds"]
    ]

    annotated = annotated.reset_index(drop=True)
    splits_present = sorted(annotated["split"].astype(str).unique().tolist())
    split_ranges = _split_ranges_from_labels(
        [str(x) for x in annotated["split"].astype(str).tolist()]
    )

    actual_vals: list[float] = []
    predicted_vals: list[float] = []
    # Prefer forecast values when available; current TimesNet outputs reconstruction.
    if "pred_val" in annotated.columns:
        predicted_vals = [_json_float(x) for x in annotated["pred_val"]]
        value_kind = "forecast"
    elif "recon_val" in annotated.columns:
        predicted_vals = [_json_float(x) for x in annotated["recon_val"]]
        value_kind = "reconstruction"
    else:
        value_kind = "forecast"
    if "actual_val" in annotated.columns:
        actual_vals = [_json_float(x) for x in annotated["actual_val"]]
    else:
        dataset_path = run_path / "dataset.parquet"
        if dataset_path.is_file():
            ds_index = pd.DatetimeIndex(pd.to_datetime(annotated["ds"], utc=True))
            actual_vals = _load_actual_series(dataset_path, uid, ds_index)
    severity_vals = [
        str(x) if x else None for x in annotated.get("severity", pd.Series([""] * len(annotated)))
    ]

    return {
        "run_id": run_id,
        "mode": mode,
        "unique_id": uid,
        "cutoff": cutoff_val,
        "timestamps": timestamps,
        "score": [_json_float(x) for x in annotated["score"]],
        "actual": actual_vals,
        "predicted": predicted_vals,
        "value_kind": value_kind,
        "severity": severity_vals,
        "split": [str(x) for x in annotated["split"]],
        "is_anomaly": [bool(x) for x in annotated["is_anomaly"]],
        "anomalies": anomaly_records,
        "points": len(timestamps),
        "points_in_window": points_in_window,
        "points_total": points_total,
        "window": chart_window,
        "flags_in_split": flags_in_split,
        "flags_visible": int(annotated["is_anomaly"].sum()),
        "splits_present": splits_present,
        "split_ranges": split_ranges,
    }


def get_timesnet_series_anomalies(
    run_id: str,
    unique_id: str | None = None,
    limit: int = 200,
    mode: str | None = None,
    kind: str | None = None,
    model_id: str = MODEL_TIMESNET,
    *,
    run_path: Path | None = None,
) -> dict:
    if run_path is None:
        _require_family(model_id, FAMILY_ANOMALY)
        run_path = _resolve_run_dir(run_id, model_id)
    else:
        run_path = Path(run_path)
    mode = _normalize_anomaly_mode(mode)
    anomalies_doc = _read_json(_anomalies_json_path(run_path, mode))
    if not anomalies_doc:
        raise HTTPException(status_code=404, detail="anomalies file not found")

    records = anomalies_doc.get("anomalies") or []
    if unique_id:
        records = [r for r in records if str(r.get("unique_id")) == unique_id.strip()]
    if kind:
        kind_norm = kind.strip().lower()
        if kind_norm in ("macro", "micro"):
            records = [r for r in records if str(r.get("anomaly_kind", "")).lower() == kind_norm]

    records = records[: max(1, min(limit, 5000))]
    return {
        "run_id": run_id,
        "mode": mode,
        "unique_id": unique_id,
        "records": records,
        "total": len(anomalies_doc.get("anomalies") or []),
    }


def get_timesnet_location_chart(
    run_id: str,
    location_id: str,
    max_points: int = MAX_POINTS_HINT,
    split: str | None = None,
    window: str | None = None,
    mode: str | None = None,
    model_id: str = MODEL_TIMESNET,
) -> dict:
    """Stacked class breakdown + predictions for one location (plots/class_summary style)."""
    t0 = time.perf_counter()
    _require_family(model_id, FAMILY_ANOMALY)
    run_path = _resolve_run_dir(run_id, model_id)
    scores_path = run_path / "scores.parquet"
    dataset_path = run_path / "dataset.parquet"
    if not scores_path.is_file():
        raise HTTPException(status_code=404, detail="scores.parquet not found")
    if not dataset_path.is_file():
        raise HTTPException(status_code=404, detail="dataset.parquet not found")

    mode = _normalize_anomaly_mode(mode)
    chart_window = _normalize_chart_window(window)

    loc = location_id.strip()
    if not loc:
        raise HTTPException(status_code=400, detail="location_id is required")

    try:
        uid_df = pd.read_parquet(scores_path, columns=["unique_id"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read scores: {exc}") from exc

    class_uids = sorted(
        {
            str(u)
            for u in uid_df["unique_id"].astype(str).unique()
            if _location_from_uid(str(u)) == loc and _parse_cls(str(u)) is not None
        }
    )
    if not class_uids:
        raise HTTPException(
            status_code=404,
            detail=f"No class-level series for location {loc!r}",
        )

    total_uid = f"{loc}__total"
    uids_for_chart = list(class_uids)
    all_uids = {str(u) for u in uid_df["unique_id"].astype(str).unique()}
    if total_uid in all_uids:
        uids_for_chart.append(total_uid)

    location_wide = _location_wide_meta(run_path) is not None
    df_scores = _read_scores_for_uids(scores_path, uids_for_chart, split=split)
    df_scores["ds"] = pd.to_datetime(df_scores["ds"], utc=True)
    if df_scores.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No score rows for split={split!r} at location {loc!r}",
        )

    lookup = _chart_anomaly_lookup(run_path, mode)
    if lookup is not None:
        annotated = _merge_anomaly_flags(
            df_scores,
            lookup,
            _cutoff_by_uid(run_path),
            location_wide=location_wide,
        )
    else:
        annotated = _annotate_for_mode(df_scores, run_path, mode)

    points_total = int(annotated["ds"].nunique())
    annotated = _filter_df_chart_window(annotated, chart_window)
    points_in_window = int(annotated["ds"].nunique())

    ts_index = pd.DatetimeIndex(sorted(annotated["ds"].unique()))
    anom_ds = pd.DatetimeIndex(
        annotated.loc[annotated["is_anomaly"], "ds"].drop_duplicates().sort_values()
    )
    ts_index = _downsample_index_preserving(
        ts_index,
        anom_ds,
        max(50, min(max_points, 8000)),
    )

    y_cols = [f"y__{uid}" for uid in class_uids]
    try:
        ds_frame = pd.read_parquet(dataset_path, columns=["ds"] + y_cols)
    except Exception:
        ds_frame = pd.read_parquet(dataset_path)
    ds_frame["ds"] = pd.to_datetime(ds_frame["ds"], utc=True)
    ds_frame = ds_frame.set_index("ds").sort_index()
    ds_sub = ds_frame.reindex(ts_index).fillna(0.0)

    classes_out: list[dict[str, Any]] = []
    anom_per_index: dict[int, list[int]] = {}

    if total_uid in uids_for_chart:
        total_chunk = (
            annotated[annotated["unique_id"].astype(str) == total_uid]
            .set_index("ds")
            .reindex(ts_index)
        )
        total_is_anom = total_chunk["is_anomaly"].fillna(False).to_numpy(dtype=bool)
        for i, flag in enumerate(total_is_anom):
            if flag:
                anom_per_index.setdefault(i, []).append(_MACRO_MARKER_CLASS)

    for uid in class_uids:
        cls_id = _parse_cls(uid)
        if cls_id is None:
            continue
        col = f"y__{uid}"
        chunk = (
            annotated[annotated["unique_id"].astype(str) == uid].set_index("ds").reindex(ts_index)
        )
        actual_col = ds_sub[col] if col in ds_sub.columns else pd.Series(0.0, index=ts_index)
        pred_col = (
            chunk["pred_val"].fillna(0.0)
            if "pred_val" in chunk.columns
            else pd.Series(0.0, index=ts_index)
        )
        is_anom = chunk["is_anomaly"].fillna(False).to_numpy(dtype=bool)

        for i, flag in enumerate(is_anom):
            if flag:
                anom_per_index.setdefault(i, []).append(cls_id)

        classes_out.append(
            {
                "class_id": cls_id,
                "unique_id": uid,
                "label": _VEHICLE_CLASS_LABELS.get(cls_id, f"cls{cls_id}"),
                "color": _CLASS_COLORS.get(cls_id, _FALLBACK_CLASS_COLOR),
                "actual": [_json_float(x) for x in actual_col.to_numpy(dtype=np.float64)],
                "predicted": [_json_float(x) for x in pred_col.to_numpy(dtype=np.float64)],
            }
        )

    classes_out.sort(key=lambda row: sum(row["actual"]), reverse=True)

    total_actual = [
        _json_float(sum(classes_out[j]["actual"][i] for j in range(len(classes_out))))
        for i in range(len(ts_index))
    ]
    total_predicted = [
        _json_float(sum(classes_out[j]["predicted"][i] for j in range(len(classes_out))))
        for i in range(len(ts_index))
    ]

    anomaly_markers: list[dict[str, Any]] = []
    for i, cls_list in sorted(anom_per_index.items()):
        labels = [
            "macro" if c == _MACRO_MARKER_CLASS else _VEHICLE_CLASS_LABELS.get(c, f"cls{c}")
            for c in sorted(set(cls_list))
        ]
        anomaly_markers.append(
            {
                "index": i,
                "classes": sorted(set(cls_list)),
                "label": ", ".join(labels),
                "mode": "zone" if len(cls_list) >= 1 else "point",
            }
        )

    timestamps = [
        t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "strftime") else str(t)[:16] for t in ts_index
    ]

    split_labels: list[str] = []
    if class_uids:
        ref_uid = class_uids[0]
        ref_chunk = (
            annotated[annotated["unique_id"].astype(str) == ref_uid]
            .set_index("ds")
            .reindex(ts_index)
        )
        if "split" in ref_chunk.columns:
            split_labels = [str(x) for x in ref_chunk["split"].fillna("unknown").tolist()]
    split_ranges = _split_ranges_from_labels(split_labels)

    logger.info(
        "location-chart run_id=%s location=%s window=%s classes=%s points=%s in_window=%s %.2fs",
        run_id,
        loc,
        chart_window,
        len(classes_out),
        len(timestamps),
        points_in_window,
        time.perf_counter() - t0,
    )

    return {
        "run_id": run_id,
        "mode": mode,
        "location_id": loc,
        "split": split,
        "window": chart_window,
        "timestamps": timestamps,
        "classes": classes_out,
        "total_actual": total_actual,
        "total_predicted": total_predicted,
        "anomaly_markers": anomaly_markers,
        "points": len(timestamps),
        "points_in_window": points_in_window,
        "points_total": points_total,
        "split_ranges": split_ranges,
        "flags_in_split": int(annotated.loc[annotated["is_anomaly"], "ds"].nunique()),
        "flags_visible": len(anomaly_markers),
    }


_MAP_DATA_DIR = _REPO_ROOT / "app" / "core" / "data"
_DATASET_A_MAP_ANOMALIES_PATH = _MAP_DATA_DIR / "dataset_a_map_anomalies.json"
_MAX_MAP_ANOMALIES_PER_STATION = 40
_SUPPORTED_MAP_DATASETS = ("dataset-a",)


def station_id_from_unique_id(unique_id: str) -> str | None:
    """Leading digits in the original unique_id match station number (e.g. 10001-seg-l1__cls7)."""
    uid = unique_id.strip()
    i = 0
    while i < len(uid) and uid[i].isdigit():
        i += 1
    return uid[:i] if i else None


def _run_compare_row(run_id: str, model_id: str = MODEL_TIMESNET) -> dict[str, Any]:
    run_path = _resolve_run_dir(run_id, model_id)
    if not run_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id!r}")

    comparison = _read_json(run_path / "comparison.json")
    metrics = _read_json(run_path / "metrics.json")
    if not comparison and not metrics:
        raise HTTPException(status_code=404, detail=f"No eval artifacts for run {run_id!r}")

    summary = (comparison or {}).get("summary") or _summary_from_run_dir(run_path)
    test_q = (comparison or {}).get("score_quantiles_by_split") or {}
    test = test_q.get("test") if isinstance(test_q, dict) else {}
    threshold = (comparison or {}).get("threshold") or {}

    return {
        "run_id": run_id,
        "anomaly_rate": summary.get("anomaly_rate"),
        "anomaly_count": summary.get("anomaly_count"),
        "point_count": summary.get("point_count"),
        "series_count": summary.get("series_count"),
        "score_p95": test.get("p95") if test else None,
        "score_p99": test.get("p99") if test else None,
        "threshold_method": threshold.get("method"),
        "threshold_profile": threshold.get("profile"),
        "eval_split": (comparison or {}).get("eval_split"),
    }


def compare_timesnet_runs(run_ids: list[str], model_id: str = MODEL_TIMESNET) -> dict:
    _require_family(model_id, FAMILY_ANOMALY)
    if len(run_ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 run_id values")
    if len(run_ids) > 4:
        raise HTTPException(status_code=400, detail="At most 4 runs per comparison")

    rows = [_run_compare_row(rid, model_id) for rid in run_ids]
    baseline_rate = rows[0].get("anomaly_rate")
    if baseline_rate is None:
        for row in rows[1:]:
            row["delta_pp"] = None
    else:
        for row in rows:
            rate = row.get("anomaly_rate")
            if rate is None:
                row["delta_pp"] = None
            else:
                row["delta_pp"] = float((rate - baseline_rate) * 100.0)

    return {
        "model": model_id,
        "baseline_run_id": run_ids[0],
        "runs": rows,
    }


def build_map_by_station_from_run(
    run_id: str,
    *,
    max_per_station: int = _MAX_MAP_ANOMALIES_PER_STATION,
    mode: str | None = _DEFAULT_ANOMALY_MODE,
    model_id: str = MODEL_TIMESNET,
) -> dict[str, list[dict]]:
    _require_family(model_id, FAMILY_ANOMALY)
    run_path = _resolve_run_dir(run_id, model_id)
    mode = _normalize_anomaly_mode(mode)
    anomalies_doc = _read_json(_anomalies_json_path(run_path, mode))
    if not anomalies_doc:
        raise HTTPException(status_code=404, detail="anomalies file not found")

    records = sorted(
        [r for r in anomalies_doc.get("anomalies") or [] if isinstance(r, dict)],
        key=lambda r: abs(float(r.get("z") or 0.0)),
        reverse=True,
    )

    buckets: dict[str, list[dict]] = {}
    for rec in records:
        uid = str(rec.get("unique_id", ""))
        sid = station_id_from_unique_id(uid)
        if not sid:
            continue
        if len(buckets.get(sid, [])) >= max_per_station:
            continue
        ds = rec.get("ds", "")
        z = rec.get("z")
        score = rec.get("score")
        summary = f"score={score:.4f}" if score is not None else "anomaly"
        if z is not None:
            summary += f" · z={float(z):.2f}"
        item = {
            "id": f"{run_id}|{uid}|{ds}",
            "ds_utc": str(ds),
            "summary": summary,
            "severity": rec.get("severity") or "warning",
            "location_ids": [uid],
        }
        buckets.setdefault(sid, []).append(item)

    return buckets


def get_gt_events(
    *,
    location_prefix: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    csv_path = _GT_EVENTS_CSV
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail=f"GT CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    if location_prefix:
        prefix = location_prefix.strip()

        def _matches(row: pd.Series) -> bool:
            mp = str(row.get("match_prefixes") or "")
            key = str(row.get("gt_location_key") or "")
            return (
                any(
                    p.strip().startswith(prefix) or prefix.startswith(p.strip())
                    for p in mp.split(",")
                    if p.strip()
                )
                or key.startswith(prefix)
                or prefix.startswith(key)
            )

        df = df[df.apply(_matches, axis=1)]

    if date_from:
        df = df[df["date_local"].astype(str) >= date_from]
    if date_to:
        df = df[df["date_local"].astype(str) <= date_to]

    events = df.replace({np.nan: None}).to_dict(orient="records")
    return {
        "csv_path": str(csv_path),
        "count": len(events),
        "events": events,
    }


_STATIONS_GEO_PATH = _MAP_DATA_DIR / "dataset_a_stations_geo.json"












def get_gt_eval(run_id: str, model_id: str = MODEL_TIMESNET) -> dict:
    run_path = _resolve_run_dir(run_id, model_id)
    gt_eval = _read_json(run_path / "gt_eval.json")
    if not gt_eval:
        raise HTTPException(
            status_code=404,
            detail=f"gt_eval.json not found for run {run_id!r} (run eval with --gt-eval)",
        )
    return {"run_id": run_id, "gt_eval": gt_eval}


def push_run_anomalies_to_map(
    run_id: str,
    *,
    dataset: str = "dataset-a",
    write: bool = True,
    max_per_station: int = _MAX_MAP_ANOMALIES_PER_STATION,
    mode: str | None = _DEFAULT_ANOMALY_MODE,
    model_id: str = MODEL_TIMESNET,
) -> dict:
    if dataset not in _SUPPORTED_MAP_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Map export supports datasets {list(_SUPPORTED_MAP_DATASETS)!r}, got {dataset!r}",
        )

    by_station = build_map_by_station_from_run(
        run_id, max_per_station=max_per_station, mode=mode, model_id=model_id
    )
    station_count = len(by_station)
    anomaly_count = sum(len(v) for v in by_station.values())
    now = datetime.now(UTC).isoformat()

    payload = {
        "dataset": dataset,
        "updated_at": now,
        "source_run_id": run_id,
        "note": f"Exported from model run {run_id!r} (mode={mode}).",
        "by_station": by_station,
    }

    written_path: str | None = None
    if write:
        _MAP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(_DATASET_A_MAP_ANOMALIES_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        written_path = str(_DATASET_A_MAP_ANOMALIES_PATH)
        logger.info(
            "map push run_id=%s stations=%s anomalies=%s -> %s",
            run_id,
            station_count,
            anomaly_count,
            written_path,
        )

    return {
        "dataset": dataset,
        "run_id": run_id,
        "written": write,
        "path": written_path,
        "station_count": station_count,
        "anomaly_count": anomaly_count,
        "updated_at": now,
    }


# ======================================================================
# Generic forecasting viewer reserved for future TimesNet forecasting
# ----------------------------------------------------------------------
# These runs store long-format CSV predictions (unique_id, ds, y, y_hat,
# cutoff) per scope plus {scope}_metrics.json and a residual anomalies.json
# with per-scope blocks. The viewer shows actual vs forecast, MAE/RMSE/sMAPE
# and residual-based anomalies, mirroring the anomaly workspace layout.
# ======================================================================


def _forecast_pred_path(run_path: Path, scope: str) -> Path:
    return run_path / f"{scope}_predictions.csv"


def _forecast_metrics_path(run_path: Path, scope: str) -> Path:
    return run_path / f"{scope}_metrics.json"


def _forecast_available_scopes(run_path: Path, model_id: str) -> list[str]:
    """Prediction scopes present on disk; model first, baseline second."""
    scopes: list[str] = []
    if _forecast_pred_path(run_path, model_id).is_file():
        scopes.append(model_id)
    if _forecast_pred_path(run_path, "baseline").is_file():
        scopes.append("baseline")
    return scopes


def _forecast_scope_block(run_path: Path, scope: str) -> dict:
    doc = _read_json(run_path / "anomalies.json")
    if not doc:
        return {}
    return (doc.get("scopes") or {}).get(scope) or {}


def get_forecast_run_manifest(run_id: str, model_id: str) -> dict:
    spec = _require_family(model_id, FAMILY_FORECAST)
    run_path = _resolve_run_dir(run_id, model_id)
    if not run_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id!r}")

    comparison = _read_json(run_path / "comparison.json")
    metrics_model = _read_json(_forecast_metrics_path(run_path, model_id))
    metrics_baseline = _read_json(_forecast_metrics_path(run_path, "baseline"))
    dataset_meta = _read_json(run_path / "dataset_meta.json")
    anomalies = _read_json(run_path / "anomalies.json")
    domain = _read_json(run_path / "domain_anomalies.json")

    if not (metrics_model or metrics_baseline or comparison):
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id!r} has no forecasting artifacts (train/eval first)",
        )

    artifacts = sorted(
        p.name for p in run_path.iterdir() if p.is_file() and not p.name.startswith(".")
    )

    scopes = _forecast_available_scopes(run_path, model_id)
    default_scope = scopes[0] if scopes else model_id

    macro_model = (comparison or {}).get(f"{model_id}_macro") or (metrics_model or {}).get("macro")
    macro_baseline = (comparison or {}).get("baseline_macro") or (metrics_baseline or {}).get(
        "macro"
    )

    anomaly_scopes = sorted((anomalies or {}).get("scopes", {}).keys()) if anomalies else []
    domain_summary = (domain or {}).get("summary") if domain else None

    return {
        "model": spec.id,
        "label": spec.label,
        "family": FAMILY_FORECAST,
        "run_id": run_id,
        "artifacts": artifacts,
        "dataset_meta": dataset_meta,
        "comparison": comparison,
        "macro": macro_model or macro_baseline,
        "macro_model": macro_model,
        "macro_baseline": macro_baseline,
        "delta": (comparison or {}).get("delta"),
        "scopes": scopes,
        "default_scope": default_scope,
        "has_anomalies": anomalies is not None,
        "anomaly_params": (anomalies or {}).get("params"),
        "anomaly_scopes": anomaly_scopes,
        "has_domain_anomalies": domain is not None,
        "domain_summary": domain_summary,
        "has_predictions": bool(scopes),
    }


def get_forecast_series(
    run_id: str,
    model_id: str,
    scope: str | None = None,
) -> dict:
    spec = _require_family(model_id, FAMILY_FORECAST)
    run_path = _resolve_run_dir(run_id, model_id)
    scopes = _forecast_available_scopes(run_path, model_id)
    scope = (scope or (scopes[0] if scopes else model_id)).strip()

    metrics = _read_json(_forecast_metrics_path(run_path, scope))
    rows_in = (metrics or {}).get("per_series") or []

    block = _forecast_scope_block(run_path, scope)
    anom_by_uid: dict[str, dict] = {
        str(r.get("unique_id")): r for r in (block.get("per_series") or [])
    }

    per_series: list[dict] = []
    for r in rows_in:
        uid = str(r.get("unique_id"))
        a = anom_by_uid.get(uid, {})
        per_series.append(
            {
                "unique_id": uid,
                "n": int(r.get("n") or 0),
                "mae": _json_float(r.get("mae") or 0.0),
                "rmse": _json_float(r.get("rmse") or 0.0),
                "smape": _json_float(r.get("smape") or 0.0),
                "n_anomalies": int(a.get("n_anomalies") or 0),
                "rate": _json_float(a.get("rate") or 0.0),
            }
        )

    # If metrics are missing but anomalies exist, surface those series anyway.
    if not per_series and anom_by_uid:
        for uid, a in anom_by_uid.items():
            per_series.append(
                {
                    "unique_id": uid,
                    "n": int(a.get("n") or 0),
                    "mae": 0.0,
                    "rmse": 0.0,
                    "smape": 0.0,
                    "n_anomalies": int(a.get("n_anomalies") or 0),
                    "rate": _json_float(a.get("rate") or 0.0),
                }
            )

    return {
        "run_id": run_id,
        "model": spec.id,
        "scope": scope,
        "scopes": scopes,
        "per_series": per_series,
        "macro": (metrics or {}).get("macro"),
    }


def get_forecast_chart(
    run_id: str,
    model_id: str,
    unique_id: str,
    scope: str | None = None,
    max_points: int = MAX_POINTS_HINT,
    window: str | None = None,
    include_baseline: bool = True,
) -> dict:
    spec = _require_family(model_id, FAMILY_FORECAST)
    run_path = _resolve_run_dir(run_id, model_id)
    scopes = _forecast_available_scopes(run_path, model_id)
    scope = (scope or (scopes[0] if scopes else model_id)).strip()

    uid = unique_id.strip()
    if not uid:
        raise HTTPException(status_code=400, detail="unique_id is required")
    chart_window = _normalize_chart_window(window)

    preds_path = _forecast_pred_path(run_path, scope)
    if not preds_path.is_file():
        raise HTTPException(status_code=404, detail=f"{scope}_predictions.csv not found")

    df = pd.read_csv(preds_path)
    df = df[df["unique_id"].astype(str) == uid].copy()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Series not found: {uid!r}")
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    df = df.sort_values("ds").reset_index(drop=True)
    n_total = len(df)

    records = _forecast_scope_block(run_path, scope).get("anomalies") or []
    anom_lookup = {
        _ds_lookup_key(r.get("ds")): r
        for r in records
        if str(r.get("unique_id")) == uid and r.get("ds") is not None
    }
    ds_keys = [_ds_lookup_key(d) for d in df["ds"]]
    df["is_anomaly"] = [k in anom_lookup for k in ds_keys]
    df["severity"] = [str((anom_lookup.get(k) or {}).get("severity") or "") for k in ds_keys]
    df["residual"] = (df["y"].astype(float) - df["y_hat"].astype(float)).astype(float)
    df["score"] = [float((anom_lookup.get(k) or {}).get("score") or 0.0) for k in ds_keys]

    has_baseline = (
        include_baseline
        and scope != "baseline"
        and _forecast_pred_path(run_path, "baseline").is_file()
    )
    if has_baseline:
        try:
            bdf = pd.read_csv(_forecast_pred_path(run_path, "baseline"))
            bdf = bdf[bdf["unique_id"].astype(str) == uid].copy()
            bdf["ds"] = pd.to_datetime(bdf["ds"], utc=True)
            bmap = {
                _ds_lookup_key(d): float(v)
                for d, v in zip(bdf["ds"], bdf["y_hat"].astype(float), strict=False)
            }
            df["baseline"] = [bmap.get(k, float("nan")) for k in ds_keys]
        except Exception:
            has_baseline = False

    points_total = int(n_total)
    windowed = _filter_df_chart_window(df, chart_window)
    points_in_window = int(len(windowed))
    df = _downsample_preserving(windowed, max(50, min(max_points, 8000)))

    timestamps = [
        t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "strftime") else str(t)[:16] for t in df["ds"]
    ]
    flagged = df[df["is_anomaly"]]
    anomaly_records = [
        {
            "ds": row.ds.isoformat() if hasattr(row.ds, "isoformat") else str(row.ds),
            "score": _json_float(row.score),
            "residual": _json_float(row.residual),
            "severity": str(row.severity) if row.severity else None,
        }
        for row in flagged.itertuples(index=False)
    ]

    return {
        "run_id": run_id,
        "model": spec.id,
        "scope": scope,
        "scopes": scopes,
        "unique_id": uid,
        "timestamps": timestamps,
        "actual": [_json_float(x) for x in df["y"]],
        "predicted": [_json_float(x) for x in df["y_hat"]],
        "baseline": [_json_float(x) for x in df["baseline"]] if has_baseline else [],
        "residual": [_json_float(x) for x in df["residual"]],
        "score": [_json_float(x) for x in df["score"]],
        "severity": [str(x) if x else None for x in df["severity"]],
        "is_anomaly": [bool(x) for x in df["is_anomaly"]],
        "anomalies": anomaly_records,
        "points": len(timestamps),
        "points_in_window": points_in_window,
        "points_total": points_total,
        "window": chart_window,
        "split_ranges": [],
    }
