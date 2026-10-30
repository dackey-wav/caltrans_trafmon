"""Offline contracts for CLI exposure and static map payload construction."""

from __future__ import annotations

import argparse
import json

from app.api import map_payload
from app.cli.parsers import build_parser


def _top_level_commands(parser: argparse.ArgumentParser) -> set[str]:
    action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return set(action.choices)


def test_cli_exposes_only_current_import_and_model_commands():
    commands = _top_level_commands(build_parser())

    assert commands == {"import-csv", "timesnet"}
    assert {
        "import-api",
        "features-rebuild",
        "import-csv-legacy",
        "import-csv-kat",
        "import-csv-wro",
        "nhits",
        "lstm_ae",
    }.isdisjoint(commands)


def test_import_csv_parser_takes_a_file_path():
    parser = build_parser()

    args = parser.parse_args(["import-csv", "input.csv"])
    assert args.command == "import-csv"
    assert args.input_path == "input.csv"
    assert args.no_resume is False


def test_map_payload_returns_empty_shape_for_unsupported_dataset():
    assert map_payload.build_map_payload("dataset-b") == {
        "dataset": "dataset-b",
        "source_note": None,
        "stations": [],
        "message": "No static map metadata for this dataset yet.",
    }


def test_map_payload_merges_geometry_and_valid_anomalies(tmp_path, monkeypatch):
    geo = {
        "source_note": "fixture",
        "stations": [
            {
                "station_id": "001",
                "road": "A1",
                "place": "Test",
                "segment": "north",
                "right": {"lat": 50.0, "lon": 19.0},
                "left": {"lat": 52.0, "lon": 21.0},
            }
        ],
    }
    anomalies = {
        "source_run_id": "run-1",
        "updated_at": "2026-07-22T12:00:00Z",
        "by_station": {"1": [{"id": "a"}, "invalid"]},
    }
    (tmp_path / "dataset_a_stations_geo.json").write_text(json.dumps(geo), encoding="utf-8")
    (tmp_path / "dataset_a_map_anomalies.json").write_text(json.dumps(anomalies), encoding="utf-8")
    monkeypatch.setattr(map_payload, "_DATA_DIR", tmp_path)

    payload = map_payload.build_map_payload("dataset-a")

    assert payload["source_note"] == "fixture"
    assert payload["updated_at"] == "2026-07-22T12:00:00Z"
    assert "run-1" in payload["message"]
    assert payload["stations"] == [
        {
            "station_id": "001",
            "road": "A1",
            "place": "Test",
            "segment": "north",
            "right": {"lat": 50.0, "lon": 19.0},
            "left": {"lat": 52.0, "lon": 21.0},
            "lat": 51.0,
            "lon": 20.0,
            "anomaly_count": 1,
            "anomalies": [{"id": "a"}],
        }
    ]
