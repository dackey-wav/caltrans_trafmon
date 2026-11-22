"""Dashboard REST API: route inventory and the station/model payloads."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from app.api.main import app
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit
client = TestClient(app)
EXP = Path("artifacts/forecast/timesnet/sd15/experiments/w192_compact_lr1e-4")


def test_route_inventory():
    routes = {(m, r.path) for r in app.routes if r.path.startswith("/api/") for m in r.methods}
    assert routes == {
        ("GET", "/api/health"),
        ("GET", "/api/config"),
        ("GET", "/api/station"),
        ("GET", "/api/model"),
    }
    assert client.get("/api/traffic").status_code == 404


def test_station_summary_from_replay_csv(tmp_path, monkeypatch):
    csv = (
        "timestamp,station_id,speed_mph,flow_veh_5min,occupancy_fraction,observed_percent,"
        "latitude,longitude,freeway,direction,lane_type\n"
        "2025-01-31T23:55:00-08:00,7,60,100,.05,100,32.7,-117.2,5,N,ML\n"
        "2025-02-01T00:00:00-08:00,7,60,100,.05,0,32.7,-117.2,5,N,ML\n"
        "2025-02-01T00:05:00-08:00,7,60,100,.05,100,32.7,-117.2,5,N,ML\n"
        "2025-02-01T00:00:00-08:00,8,60,100,.05,100,1,1,5,S,ML\n"
    )
    (tmp_path / "7_5min.csv.gz").write_bytes(gzip.compress(csv.encode()))
    monkeypatch.setenv("IOT_STATION_ID", "7")
    monkeypatch.setenv("IOT_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("IOT_REPLAY_FILE", raising=False)

    body = client.get("/api/station").json()
    assert body["rows"] == 3 and body["freeway"] == "5" and body["direction"] == "N"
    assert [m["month"] for m in body["monthly"]] == ["2025-01", "2025-02"]
    assert body["monthly"][1]["observed_share"] == 0.5
    assert client.get("/api/config").json()["station_id"] == "7"


@pytest.mark.skipif(not (EXP / "forecast_metrics_test.json").exists(), reason="model artifacts absent")
def test_model_payload_has_offline_baselines():
    body = client.get("/api/model").json()
    assert body["hparams"]["window"] == 192
    assert {"model", "weekly"} <= set(body["test"]["methods"])
    assert len(body["test"]["lead_minutes"]) == len(body["test"]["by_lead"]["model"]["flow"])
