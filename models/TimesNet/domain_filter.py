"""Domain-aware anomaly gates aligned with operational definitions.

Basic anomaly types (60-minute aggregates):
- traffic_intensity: |actual - expected| / expected >= 20%
- class_share: relative change in vehicle-class share >= 30%
- lane_shift: lane-ratio shift on dual-carriageway SCPRs (S3a/A4)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from models.TimesNet.config import (
    DOMAIN_AGG_FREQ,
    LANE_SHIFT_SCPRS,
    MIN_CLASS_SHARE_PCT,
    MIN_HOURLY_VOLUME,
    MIN_LANE_RATIO_SHIFT,
    MIN_TRAFFIC_INTENSITY_PCT,
)

_LOCATION_RE = re.compile(r"^(?P<scpr>\d+)-(?P<slug>[^-]+)-(?P<lane>l\d+)(?:__cls(?P<cls>\d+))?$")


def parse_unique_id(uid: str) -> dict | None:
    m = _LOCATION_RE.match(str(uid))
    if not m:
        return None
    d = m.groupdict()
    d["location_base"] = f"{d['scpr']}-{d['slug']}"
    d["is_class"] = d["cls"] is not None
    return d


def _hour_bucket(ds: pd.Series) -> pd.Series:
    return pd.to_datetime(ds, utc=True).dt.floor(DOMAIN_AGG_FREQ)


def _pct_change(actual: float, expected: float, *, min_denom: float) -> float:
    denom = max(float(expected), min_denom)
    return abs(float(actual) - float(expected)) / denom


def _hourly_totals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = _hour_bucket(out["ds"])
    return (
        out.groupby(["unique_id", "hour"], sort=False)
        .agg(
            actual_val=("actual_val", "sum"),
            recon_val=("recon_val", "sum"),
        )
        .reset_index()
    )


def _traffic_intensity_keys(hourly: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    keys: set[tuple[str, pd.Timestamp]] = set()
    for row in hourly.itertuples(index=False):
        parsed = parse_unique_id(row.unique_id)
        if parsed is None or parsed["is_class"]:
            continue
        pct = _pct_change(row.actual_val, row.recon_val, min_denom=MIN_HOURLY_VOLUME)
        if pct >= MIN_TRAFFIC_INTENSITY_PCT:
            keys.add((str(row.unique_id), pd.Timestamp(row.hour)))
    return keys


def _class_share_keys(hourly: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    rows: list[dict] = []
    for row in hourly.itertuples(index=False):
        parsed = parse_unique_id(row.unique_id)
        if parsed is None or not parsed["is_class"]:
            continue
        rows.append(
            {
                "unique_id": row.unique_id,
                "hour": row.hour,
                "location_base": parsed["location_base"],
                "actual_val": row.actual_val,
                "recon_val": row.recon_val,
            }
        )
    if not rows:
        return set()

    cls_df = pd.DataFrame(rows)
    totals = (
        cls_df.groupby(["location_base", "hour"], sort=False)[["actual_val", "recon_val"]]
        .sum()
        .rename(columns={"actual_val": "total_actual", "recon_val": "total_recon"})
        .reset_index()
    )
    merged = cls_df.merge(totals, on=["location_base", "hour"], how="left")
    share_actual = merged["actual_val"] / merged["total_actual"].clip(lower=MIN_HOURLY_VOLUME)
    share_recon = merged["recon_val"] / merged["total_recon"].clip(lower=MIN_HOURLY_VOLUME)
    share_pct = (share_actual - share_recon).abs() / share_recon.clip(lower=0.01)

    keys: set[tuple[str, pd.Timestamp]] = set()
    for i, ok in enumerate(share_pct >= MIN_CLASS_SHARE_PCT):
        if ok:
            keys.add((str(merged.iloc[i]["unique_id"]), pd.Timestamp(merged.iloc[i]["hour"])))
    return keys


def _lane_shift_keys(hourly: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    """Return (unique_id, hour) keys where a lane-ratio shift is detected."""
    lane_rows: list[dict] = []
    for row in hourly.itertuples(index=False):
        parsed = parse_unique_id(row.unique_id)
        if parsed is None or parsed["is_class"]:
            continue
        if parsed["scpr"] not in LANE_SHIFT_SCPRS:
            continue
        if parsed["lane"] not in ("l1", "l2"):
            continue
        lane_rows.append(
            {
                "unique_id": row.unique_id,
                "hour": row.hour,
                "scpr": parsed["scpr"],
                "slug": parsed["slug"],
                "lane": parsed["lane"],
                "actual_val": row.actual_val,
                "recon_val": row.recon_val,
            }
        )

    if not lane_rows:
        return set()

    lane_df = pd.DataFrame(lane_rows)
    keys: set[tuple[str, pd.Timestamp]] = set()
    for (_, _), group in lane_df.groupby(["scpr", "slug"], sort=False):
        for hour, hour_group in group.groupby("hour", sort=False):
            l1 = hour_group[hour_group["lane"] == "l1"]
            l2 = hour_group[hour_group["lane"] == "l2"]
            if l1.empty or l2.empty:
                continue
            act_total = float(l1["actual_val"].iloc[0] + l2["actual_val"].iloc[0])
            rec_total = float(l1["recon_val"].iloc[0] + l2["recon_val"].iloc[0])
            if act_total < MIN_HOURLY_VOLUME or rec_total < MIN_HOURLY_VOLUME:
                continue
            ratio_act = float(l1["actual_val"].iloc[0]) / act_total
            ratio_rec = float(l1["recon_val"].iloc[0]) / rec_total
            if abs(ratio_act - ratio_rec) >= MIN_LANE_RATIO_SHIFT:
                for uid in hour_group["unique_id"]:
                    keys.add((str(uid), pd.Timestamp(hour)))
    return keys


def enrich_domain_columns(scores: pd.DataFrame) -> pd.DataFrame:
    """Add hourly domain metrics and relevance flags to score rows."""
    required = {"ds", "unique_id", "actual_val", "recon_val"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores missing columns for domain filter: {sorted(missing)}")

    out = scores.copy()
    out["hour"] = _hour_bucket(out["ds"])
    hourly = _hourly_totals(out)

    traffic_keys = _traffic_intensity_keys(hourly)
    class_keys = _class_share_keys(hourly)
    lane_keys = _lane_shift_keys(hourly)

    hourly_pct = {
        (str(r.unique_id), pd.Timestamp(r.hour)): _pct_change(
            r.actual_val, r.recon_val, min_denom=MIN_HOURLY_VOLUME
        )
        for r in hourly.itertuples(index=False)
    }

    domain_relevant: list[bool] = []
    anomaly_kind: list[str] = []
    hourly_pct_change: list[float] = []

    for row in out.itertuples(index=False):
        key = (str(row.unique_id), pd.Timestamp(row.hour))
        pct = float(hourly_pct.get(key, 0.0))
        kinds: list[str] = []
        if key in traffic_keys:
            kinds.append("traffic_intensity")
        if key in class_keys:
            kinds.append("class_share")
        if key in lane_keys:
            kinds.append("lane_shift")
        domain_relevant.append(bool(kinds))
        anomaly_kind.append("|".join(kinds))
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
    """Keep only statistically flagged points that pass domain gates."""
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
