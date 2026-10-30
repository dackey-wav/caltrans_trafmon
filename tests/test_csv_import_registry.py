"""Tests for the 5-minute station CSV import."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import app.core.config as cfg
import pytest
from app.importer.importer_csv.formats.registry import (
    FORMAT,
    PreflightError,
    get_format,
    preflight,
)
from app.importer.importer_csv.formats.station_5min import (
    parse_measure,
    parse_timestamp,
)
from app.importer.importer_csv.pipeline import (
    _file_fingerprint,
    _state_file_key,
    discover_csv_files,
    scan_files,
)

SAMPLE_HEADER = (
    "timestamp,station_id,speed_mph,flow_veh_5min,occupancy_fraction,"
    "observed_percent,latitude,longitude,freeway,direction,lane_type"
)
SAMPLE_ROW = "01.01.2022 0:00,400237,69.3,72,.0128,0,37.210372,-121.723483,101,N,ML"


def test_registry_points_at_the_station_series(monkeypatch):
    monkeypatch.setattr(cfg, "DB_NAME", "traffic")

    spec = get_format()
    assert spec is FORMAT
    assert spec.format_id == "station-5min"
    assert spec.destination_resolver() == "traffic.traffic_5min"

    with pytest.raises(FrozenInstanceError):
        spec.format_id = "changed"  # type: ignore[misc]


def test_parse_timestamp_reads_day_first_california_clock_time():
    assert parse_timestamp("01.01.2022 0:00") == "2022-01-01 00:00:00"
    assert parse_timestamp("01.01.2022 13:05:09") == "2022-01-01 13:05:09"
    assert parse_timestamp("2022-01-01 00:00") == "2022-01-01 00:00:00"
    assert parse_timestamp("not-a-time") is None


def test_parse_measure_keeps_blank_and_rejects_text():
    assert parse_measure(".0128") == "0.0128"
    assert parse_measure("69,3") == "69.3"
    assert parse_measure("  ") == ""
    assert parse_measure("n/a") is None


def test_discover_csv_files_supports_file_and_recursive_directory(tmp_path):
    root_csv = tmp_path / "root.csv"
    nested_csv = tmp_path / "nested" / "sample.CSV"
    ignored = tmp_path / "nested" / "notes.txt"
    nested_csv.parent.mkdir()
    root_csv.write_text("header\n", encoding="utf-8")
    nested_csv.write_text("header\n", encoding="utf-8")
    ignored.write_text("ignore\n", encoding="utf-8")

    assert discover_csv_files(root_csv) == [str(root_csv)]
    assert discover_csv_files(tmp_path) == [str(nested_csv), str(root_csv)]
    assert discover_csv_files(ignored) == []


def test_scan_files_honors_resume_state_and_no_resume(tmp_path):
    source = tmp_path / "source.csv"
    state_path = tmp_path / "state.json"
    source.write_text("header\n", encoding="utf-8")
    file_key = _state_file_key(str(source))
    state = {
        "files": {file_key: _file_fingerprint(str(source))},
        "last_run_at": None,
        "total_inserted": 0,
    }

    queued, total, skipped = scan_files(
        str(source), state, str(state_path), resume=True, progress_every=0
    )
    assert queued == []
    assert (total, skipped) == (1, 1)

    queued, total, skipped = scan_files(
        str(source), state, str(state_path), resume=False, progress_every=0
    )
    assert len(queued) == 1
    assert queued[0]["file_key"] == file_key
    assert (total, skipped) == (1, 0)


def test_scan_files_migrates_matching_legacy_state_key(tmp_path):
    source = tmp_path / "source.csv"
    state_path = tmp_path / "state.json"
    source.write_text("header\n", encoding="utf-8")
    fingerprint = _file_fingerprint(str(source))
    state = {
        "files": {source.name: fingerprint},
        "last_run_at": None,
        "total_inserted": 0,
    }

    queued, total, skipped = scan_files(
        str(source),
        state,
        str(state_path),
        resume=True,
        state_flush_every=1,
        progress_every=0,
    )

    assert queued == []
    assert (total, skipped) == (1, 1)
    assert state["files"][_state_file_key(str(source))] == fingerprint
    assert state_path.is_file()


def test_preflight_accepts_one_station_csv(tmp_path):
    source = tmp_path / "station.csv"
    source.write_text(f"{SAMPLE_HEADER}\n{SAMPLE_ROW}\n", encoding="utf-8")

    assert preflight(source) == [str(source)]


def test_preflight_rejects_missing_empty_and_wrong_shape(tmp_path):
    with pytest.raises(PreflightError, match="does not exist"):
        preflight(tmp_path / "missing")
    with pytest.raises(PreflightError, match="contains no CSV"):
        preflight(tmp_path)

    wrong = tmp_path / "wrong.csv"
    wrong.write_text("unknown,columns\n1,2\n", encoding="utf-8")
    with pytest.raises(PreflightError, match="does not match format station-5min"):
        preflight(wrong)


def test_build_tsv_drops_constant_columns_and_bad_timestamps(tmp_path):
    source = tmp_path / "station.csv"
    source.write_text(
        f"{SAMPLE_HEADER}\n{SAMPLE_ROW}\nnot-a-time,400237,1,2,3,4,0,0,101,N,ML\n",
        encoding="utf-8",
    )
    output = tmp_path / "stage.tsv"
    rows, bad, per_file = FORMAT.adapter.build_tsv(
        [{"path": str(source), "file_key": "k"}],
        str(output),
        progress_every=0,
    )

    assert (rows, bad) == (1, 1)
    assert per_file == {"k": 1}
    header, row = output.read_text(encoding="utf-8").splitlines()
    assert header == "ts\tspeed_mph\tflow_veh_5min\toccupancy_fraction\tobserved_percent"
    assert row == "2022-01-01 00:00:00\t69.3\t72\t0.0128\t0"
