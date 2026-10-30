"""Tests for TimesNet domain anomaly gates."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from models.TimesNet.domain_filter import (
    apply_domain_filter,
    enrich_domain_columns,
    parse_unique_id,
)


def test_parse_unique_id_location():
    p = parse_unique_id("24008-wis-l1")
    assert p is not None
    assert p["scpr"] == "24008"
    assert not p["is_class"]


def test_parse_unique_id_class():
    p = parse_unique_id("24008-wis-l1__cls7")
    assert p is not None
    assert p["is_class"]
    assert p["cls"] == "7"


def test_traffic_intensity_gate():
    ts = datetime(2024, 12, 2, 10, 0, tzinfo=UTC)
    scores = pd.DataFrame(
        [
            {
                "ds": ts,
                "unique_id": "24005-kat-l1",
                "actual_val": 120.0,
                "recon_val": 80.0,
                "score": 1.0,
                "cutoff": 0.5,
                "z": 7.0,
                "is_anomaly": True,
                "severity": "warning",
                "split": "test",
            }
        ]
    )
    enriched = enrich_domain_columns(scores)
    assert bool(enriched["domain_relevant"].iloc[0])
    assert "traffic_intensity" in enriched["anomaly_kind"].iloc[0]


def test_low_change_filtered_out():
    ts = datetime(2024, 12, 2, 10, 0, tzinfo=UTC)
    scores = pd.DataFrame(
        [
            {
                "ds": ts,
                "unique_id": "24005-kat-l1",
                "actual_val": 82.0,
                "recon_val": 80.0,
                "score": 1.0,
                "cutoff": 0.5,
                "z": 7.0,
                "is_anomaly": True,
                "severity": "warning",
                "split": "test",
            }
        ]
    )
    out = apply_domain_filter(scores)
    assert not bool(out["is_anomaly"].iloc[0])


def test_lane_shift_detected():
    ts = datetime(2024, 12, 2, 11, 0, tzinfo=UTC)
    scores = pd.DataFrame(
        [
            {
                "ds": ts,
                "unique_id": "24008-wis-l1",
                "actual_val": 90.0,
                "recon_val": 50.0,
                "score": 1.0,
                "cutoff": 0.5,
                "z": 7.0,
                "is_anomaly": True,
                "severity": "warning",
                "split": "test",
            },
            {
                "ds": ts,
                "unique_id": "24008-wis-l2",
                "actual_val": 10.0,
                "recon_val": 50.0,
                "score": 1.0,
                "cutoff": 0.5,
                "z": 7.0,
                "is_anomaly": True,
                "severity": "warning",
                "split": "test",
            },
        ]
    )
    enriched = enrich_domain_columns(scores)
    assert all("lane_shift" in k for k in enriched["anomaly_kind"])
