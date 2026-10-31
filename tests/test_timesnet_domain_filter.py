"""Tests for the TimesNet hourly intensity gate."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from models.TimesNet.domain_filter import apply_domain_filter, enrich_domain_columns


def test_traffic_intensity_gate():
    ts = datetime(2024, 12, 2, 10, 0, tzinfo=UTC)
    scores = pd.DataFrame(
        [
            {
                "ds": ts,
                "unique_id": "flow",
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
    assert enriched["anomaly_kind"].iloc[0] == "traffic_intensity"


def test_low_change_filtered_out():
    ts = datetime(2024, 12, 2, 10, 0, tzinfo=UTC)
    scores = pd.DataFrame(
        [
            {
                "ds": ts,
                "unique_id": "speed",
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
