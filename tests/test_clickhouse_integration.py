"""ClickHouse-backed import smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.importer.importer_csv.formats.registry import get_format
from app.importer.importer_csv.pipeline import run_staged_import

pytestmark = pytest.mark.integration_ch

SAMPLE_HEADER = (
    "timestamp,station_id,speed_mph,flow_veh_5min,occupancy_fraction,"
    "observed_percent,latitude,longitude,freeway,direction,lane_type"
)
SAMPLE_ROW = "01.01.2022 0:00,400237,69.3,72,.0128,50,37.210372,-121.723483,101,N,ML"


def test_station_5min_import_smoke(clickhouse_db, tmp_path):
    csv_path = Path(tmp_path) / "station.csv"
    csv_path.write_text(f"{SAMPLE_HEADER}\n{SAMPLE_ROW}\n", encoding="utf-8")

    spec = get_format()
    run_staged_import(
        spec.adapter,
        str(csv_path),
        str(tmp_path / "state.json"),
        resume=False,
    )

    _, rows = clickhouse_db.query(
        """
        SELECT
            toString(ts),
            speed_mph,
            flow_veh_5min,
            occupancy_fraction,
            observed_percent
        FROM traffic_5min
        ORDER BY ts
        """
    )
    assert len(rows) == 1
    ts, speed, flow, occupancy, observed = rows[0]
    assert ts == "2022-01-01 00:00:00"
    assert speed == pytest.approx(69.3)
    assert flow == pytest.approx(72)
    assert occupancy == pytest.approx(0.0128)
    assert observed == pytest.approx(50)
