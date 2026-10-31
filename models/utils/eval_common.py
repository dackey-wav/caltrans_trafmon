"""Shared evaluation report builders for anomaly-detection models."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd


def save_json(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def build_metrics_report(
    annotated_scores: pd.DataFrame,
    extra: dict | None = None,
) -> dict:
    rows: list[dict] = []
    for split, group in annotated_scores.groupby("split", sort=True):
        rows.append(
            {
                "split": str(split),
                "points": int(len(group)),
                "series": int(group["unique_id"].nunique()),
                "anomalies": int(group["is_anomaly"].sum()),
                "anomaly_rate": float(group["is_anomaly"].mean()) if len(group) else 0.0,
                "score_mean": float(group["score"].mean()),
                "score_p95": float(group["score"].quantile(0.95)),
                "score_p99": float(group["score"].quantile(0.99)),
            }
        )
    report: dict = {"splits": rows}
    if extra:
        report["meta"] = dict(extra)
    return report


def _per_series_rows(eval_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for uid, group in eval_df.groupby("unique_id", sort=True):
        n_total = int(len(group))
        n_anom = int(group["is_anomaly"].sum())
        rows.append(
            {
                "unique_id": str(uid),
                "n": n_total,
                "n_anomalies": n_anom,
                "rate": (n_anom / n_total) if n_total > 0 else 0.0,
                "score_max": float(group["score"].max()) if n_total else 0.0,
                "score_median": float(group["score"].median()) if n_total else 0.0,
                "cutoff": float(group["cutoff"].iloc[0]) if n_total else 0.0,
            }
        )
    return rows


def _flagged_records(
    eval_df: pd.DataFrame,
    *,
    max_records: int | None,
    record_builder: Callable | None = None,
) -> list[dict]:
    flagged = eval_df[eval_df["is_anomaly"]].copy()
    if max_records is not None and len(flagged) > max_records:
        flagged = (
            flagged.assign(_abs=flagged["z"].abs())
            .sort_values("_abs", ascending=False)
            .head(int(max_records))
            .drop(columns=["_abs"])
        )
    flagged = flagged.sort_values(["unique_id", "ds"])
    if record_builder is None:
        return [
            {
                "unique_id": str(row.unique_id),
                "ds": row.ds.isoformat(),
                "score": float(row.score),
                "z": float(row.z),
                "cutoff": float(row.cutoff),
                "severity": str(row.severity),
            }
            for row in flagged.itertuples(index=False)
        ]
    return [record_builder(row) for row in flagged.itertuples(index=False)]


def build_anomaly_report_core(
    annotated_scores: pd.DataFrame,
    *,
    run_id: str | None,
    scope: str,
    eval_split: str,
    max_records: int | None,
    params: dict,
    record_builder: Callable | None = None,
    extra_top_level: dict | None = None,
) -> dict:
    eval_df = annotated_scores[annotated_scores["split"] == eval_split].copy()
    if eval_df.empty:
        raise ValueError(f"No rows in eval split={eval_split!r}")

    report = {
        "run_id": run_id,
        "params": params,
        "summary": {
            "series_count": int(eval_df["unique_id"].nunique()),
            "point_count": int(len(eval_df)),
            "anomaly_count": int(eval_df["is_anomaly"].sum()),
            "anomaly_rate": float(eval_df["is_anomaly"].mean()) if len(eval_df) else 0.0,
        },
        "per_series": _per_series_rows(eval_df),
        "anomalies": _flagged_records(
            eval_df,
            max_records=max_records,
            record_builder=record_builder,
        ),
    }
    if extra_top_level:
        report.update(extra_top_level)
    return report
