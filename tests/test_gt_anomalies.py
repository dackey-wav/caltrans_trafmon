"""Tests for GT source parsing and hit/miss evaluation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from models.data.gt_anomalies import gt_eval, prepare_gt_events


def test_direction_slug_segment():
    assert prepare_gt_events._direction_slug("segment-a") == "seg"
    assert prepare_gt_events._direction_slug("") == ""


def test_parse_time_range():
    assert prepare_gt_events._parse_time_range("07:30 - 16:30") == ("07:30", "16:30")


def test_parse_date_cell_december_sheet():
    assert prepare_gt_events._parse_date_cell("02.12.", 2024, 12) == "2024-12-02"


def test_match_prefixes_filters_by_direction():
    prefixes = {
        "10001": [
            "10001-seg-l1",
            "10001-seg-l2",
            "10001-nor-l1",
        ],
    }
    seg = prepare_gt_events._match_prefixes("10001", "segment-a", prefixes)
    assert seg == ["10001-seg-l1", "10001-seg-l2"]
    all_p = prepare_gt_events._match_prefixes("10001", "", prefixes)
    assert len(all_p) == 3


def test_gt_filtered_to_test_window(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scores = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2024-11-01T00:00:00Z",
                    "2024-12-02T07:00:00Z",
                    "2024-12-02T08:00:00Z",
                ],
                utc=True,
            ),
            "split": ["train", "test", "test"],
            "unique_id": ["a", "a", "a"],
            "score": [1.0, 5.0, 5.0],
        }
    )
    scores.to_parquet(run_dir / "scores.parquet", index=False)
    (run_dir / "dataset_meta.json").write_text(
        json.dumps({"locations": ["10001-seg-l1"]}),
        encoding="utf-8",
    )

    anomalies = {
        "params": {"eval_split": "test"},
        "anomalies": [
            {
                "unique_id": "10001-seg-l1",
                "ds": "2024-12-02T07:30:00+00:00",
                "score": 5.0,
                "z": 2.0,
                "severity": "warning",
            }
        ],
    }
    (run_dir / "anomalies_percentile.json").write_text(
        json.dumps(anomalies),
        encoding="utf-8",
    )

    gt_csv = tmp_path / "gt.csv"
    pd.DataFrame(
        [
            {
                "event_id": 1,
                "date_local": "2024-12-02",
                "site_id": "10001",
                "gt_location_key": "10001-seg",
                "event_type": "roadwork",
                "hourly_visible": "yes",
                "match_prefixes": "10001-seg-l1",
                "time_start_utc": "2024-12-02T06:00:00+00:00",
                "time_end_utc": "2024-12-02T15:30:00+00:00",
            },
            {
                "event_id": 2,
                "date_local": "2024-11-15",
                "site_id": "10001",
                "gt_location_key": "10001-seg",
                "event_type": "roadwork",
                "hourly_visible": "yes",
                "match_prefixes": "10001-seg-l1",
                "time_start_utc": "2024-11-15T06:00:00+00:00",
                "time_end_utc": "2024-11-15T15:30:00+00:00",
            },
        ]
    ).to_csv(gt_csv, index=False)

    report = gt_eval.evaluate_run(run_dir, modes=("percentile",), gt_csv=gt_csv)
    assert report["modes"]["percentile"]["summary"]["total"] == 1
    assert report["modes"]["percentile"]["summary"]["hits"] == 1
    assert report["scope"]["gt_events_total_csv"] == 2
    assert report["scope"]["gt_events_in_scope"] == 1


def test_gt_hit_miss_synthetic(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    start = datetime(2024, 12, 2, 6, 30, tzinfo=UTC)
    anomalies = {
        "params": {"eval_split": "test"},
        "anomalies": [
            {
                "unique_id": "10001-seg-l1__cls7",
                "ds": start.isoformat(),
                "score": 5.0,
                "z": 2.0,
                "severity": "warning",
            }
        ],
    }
    (run_dir / "anomalies_percentile.json").write_text(
        json.dumps(anomalies),
        encoding="utf-8",
    )

    gt_csv = tmp_path / "gt.csv"
    pd.DataFrame(
        [
            {
                "event_id": 1,
                "date_local": "2024-12-02",
                "site_id": "10001",
                "gt_location_key": "10001-seg",
                "event_type": "roadwork",
                "hourly_visible": "yes",
                "match_prefixes": "10001-seg-l1,10001-seg-l2",
                "time_start_utc": "2024-12-02T06:00:00+00:00",
                "time_end_utc": "2024-12-02T15:30:00+00:00",
            }
        ]
    ).to_csv(gt_csv, index=False)

    report = gt_eval.evaluate_run(run_dir, modes=("percentile",), gt_csv=gt_csv)
    assert report["modes"]["percentile"]["summary"]["hits"] == 1
    assert report["modes"]["percentile"]["summary"]["total"] == 1
