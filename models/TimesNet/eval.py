"""Build anomaly reports and summary metrics from TimesNet scores."""

from __future__ import annotations

import pandas as pd

from models.TimesNet.config import (
    DEFAULT_THRESHOLD_METHOD,
    EVAL_COMBINED_ROBUST_Z,
    EVAL_PERCENTILE_QUANTILE,
    MIN_TRAFFIC_INTENSITY_PCT,
)
from models.TimesNet.domain_filter import domain_filter_summary
from models.utils.eval_common import (
    build_anomaly_report_core,
    build_metrics_report,
    save_json,
)

__all__ = ["build_anomaly_report", "build_metrics_report", "save_json"]


def _timesnet_record(row) -> dict:
    rec = {
        "unique_id": str(row.unique_id),
        "ds": row.ds.isoformat(),
        "score": float(row.score),
        "z": float(row.z),
        "cutoff": float(row.cutoff),
        "severity": str(row.severity),
    }
    if hasattr(row, "anomaly_kind") and row.anomaly_kind:
        rec["anomaly_kind"] = str(row.anomaly_kind)
    if hasattr(row, "hourly_pct_change"):
        rec["hourly_pct_change"] = float(row.hourly_pct_change)
    return rec


def build_anomaly_report(
    annotated_scores: pd.DataFrame,
    thresholds: dict,
    run_id: str | None = None,
    scope: str = "timesnet",
    eval_split: str = "test",
    max_records: int | None = 5000,
    date_from: str | None = None,
    date_to: str | None = None,
    *,
    mode: str = "domain",
    domain_filter_enabled: bool = True,
) -> dict:
    eval_df = annotated_scores[annotated_scores["split"] == eval_split].copy()
    if eval_df.empty:
        raise ValueError(f"No rows in eval split={eval_split!r}")

    method = thresholds.get("method", DEFAULT_THRESHOLD_METHOD)
    params: dict = {
        "scope": scope,
        "mode": mode,
        "method": method,
        "eval_split": eval_split,
        "calibration_split": thresholds.get("calibration_split", "train"),
        "max_records": max_records,
        "date_from": date_from,
        "date_to": date_to,
        "domain_filter": domain_filter_enabled,
    }
    if method == "robust_z":
        first = next(iter(thresholds.get("per_target", {}).values()), {})
        params["robust_z"] = first.get("threshold", EVAL_COMBINED_ROBUST_Z)
    else:
        first = next(iter(thresholds.get("per_target", {}).values()), {})
        params["quantile"] = first.get("quantile", EVAL_PERCENTILE_QUANTILE)
    if mode == "combined":
        params["combined_rule"] = (
            f"percentile_q={EVAL_PERCENTILE_QUANTILE} AND robust_z>={EVAL_COMBINED_ROBUST_Z}"
        )
    if domain_filter_enabled:
        params["domain_gates"] = {
            "traffic_intensity_pct": MIN_TRAFFIC_INTENSITY_PCT,
            "aggregate_freq": "1h",
        }

    report = build_anomaly_report_core(
        annotated_scores,
        run_id=run_id,
        scope=scope,
        eval_split=eval_split,
        max_records=max_records,
        params=params,
        record_builder=_timesnet_record,
        extra_top_level={"thresholds": thresholds},
    )
    if domain_filter_enabled and "domain_relevant" in eval_df.columns:
        report["summary"]["domain_filter"] = domain_filter_summary(eval_df)
    return report
