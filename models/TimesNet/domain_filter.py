"""Hourly gate for TimesNet anomaly scores.

A statistical flag stays only when the hour also moves by at least
MIN_TRAFFIC_INTENSITY_PCT. Flow is summed inside the hour. Speed and
occupancy are averaged. Series ids are the channel names flow, speed,
and occupancy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.TimesNet.config import (
    DOMAIN_AGG_FREQ,
    MIN_HOURLY_VOLUME,
    MIN_TRAFFIC_INTENSITY_PCT,
)

_RATE_CHANNELS = frozenset({"speed", "occupancy"})


def _hour_bucket(ds: pd.Series) -> pd.Series:
    return pd.to_datetime(ds, utc=True).dt.floor(DOMAIN_AGG_FREQ)


def _pct_change(actual: float, expected: float, *, unique_id: str) -> float:
    if unique_id in _RATE_CHANNELS:
        denom = max(abs(float(expected)), 1e-3)
    else:
        denom = max(float(expected), MIN_HOURLY_VOLUME)
    return abs(float(actual) - float(expected)) / denom


def _hourly_totals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = _hour_bucket(out["ds"])
    rows: list[dict] = []
    for (uid, hour), group in out.groupby(["unique_id", "hour"], sort=False):
        how = "mean" if str(uid) in _RATE_CHANNELS else "sum"
        rows.append(
            {
                "unique_id": uid,
                "hour": hour,
                "actual_val": float(getattr(group["actual_val"], how)()),
                "recon_val": float(getattr(group["recon_val"], how)()),
            }
        )
    return pd.DataFrame(rows)


def _traffic_intensity_keys(hourly: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    keys: set[tuple[str, pd.Timestamp]] = set()
    for row in hourly.itertuples(index=False):
        pct = _pct_change(row.actual_val, row.recon_val, unique_id=str(row.unique_id))
        if pct >= MIN_TRAFFIC_INTENSITY_PCT:
            keys.add((str(row.unique_id), pd.Timestamp(row.hour)))
    return keys


def enrich_domain_columns(scores: pd.DataFrame) -> pd.DataFrame:
    """Add hourly change and a relevance flag to score rows."""
    required = {"ds", "unique_id", "actual_val", "recon_val"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores missing columns for domain filter: {sorted(missing)}")

    out = scores.copy()
    out["hour"] = _hour_bucket(out["ds"])
    hourly = _hourly_totals(out)
    traffic_keys = _traffic_intensity_keys(hourly)
    hourly_pct = {
        (str(r.unique_id), pd.Timestamp(r.hour)): _pct_change(
            r.actual_val, r.recon_val, unique_id=str(r.unique_id)
        )
        for r in hourly.itertuples(index=False)
    }

    domain_relevant: list[bool] = []
    anomaly_kind: list[str] = []
    hourly_pct_change: list[float] = []
    for row in out.itertuples(index=False):
        key = (str(row.unique_id), pd.Timestamp(row.hour))
        pct = float(hourly_pct.get(key, 0.0))
        relevant = key in traffic_keys
        domain_relevant.append(relevant)
        anomaly_kind.append("traffic_intensity" if relevant else "")
        hourly_pct_change.append(pct)

    out["hourly_pct_change"] = np.asarray(hourly_pct_change, dtype=np.float32)
    out["domain_relevant"] = np.asarray(domain_relevant, dtype=bool)
    out["anomaly_kind"] = anomaly_kind
    return out


def apply_domain_filter(
    annotated_scores: pd.DataFrame,
    *,
    enabled: bool = True,
) -> pd.DataFrame:
    """Keep statistically flagged points whose hour also moved enough."""
    if not enabled:
        out = annotated_scores.copy()
        out["domain_relevant"] = True
        out["anomaly_kind"] = ""
        if "hourly_pct_change" not in out.columns:
            out["hourly_pct_change"] = np.float32(0.0)
        return out

    enriched = enrich_domain_columns(annotated_scores)
    out = annotated_scores.copy()
    out["hourly_pct_change"] = enriched["hourly_pct_change"]
    out["domain_relevant"] = enriched["domain_relevant"]
    out["anomaly_kind"] = enriched["anomaly_kind"]
    is_anom = out["is_anomaly"].to_numpy() & out["domain_relevant"].to_numpy()
    out["is_anomaly"] = is_anom

    severity = out["severity"].to_numpy(copy=True)
    severity[~is_anom] = ""
    out["severity"] = severity
    return out


def domain_filter_summary(scores: pd.DataFrame) -> dict:
    """Compact counts for logging."""
    if "domain_relevant" not in scores.columns:
        scores = enrich_domain_columns(scores)
    kinds: dict[str, int] = {}
    for kind_str in scores.loc[scores["domain_relevant"], "anomaly_kind"]:
        for k in str(kind_str).split("|"):
            if k:
                kinds[k] = kinds.get(k, 0) + 1
    return {
        "domain_relevant_points": int(scores["domain_relevant"].sum()),
        "domain_relevant_rate": float(scores["domain_relevant"].mean()) if len(scores) else 0.0,
        "by_kind": kinds,
    }
