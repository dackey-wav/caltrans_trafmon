"""Merge static station geometry with optional per-station anomaly lists for /api/map."""

from __future__ import annotations

import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "core" / "data"


def _load_json(name: str) -> dict | None:
    path = _DATA_DIR / name
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_map_payload(dataset: str) -> dict:
    if dataset != "dataset-a":
        return {
            "dataset": dataset,
            "source_note": None,
            "stations": [],
            "message": "No static map metadata for this dataset yet.",
        }

    geo = _load_json("dataset_a_stations_geo.json") or {}
    anom = _load_json("dataset_a_map_anomalies.json") or {}
    by_station: dict = anom.get("by_station") or {}

    stations_out: list[dict] = []
    for row in geo.get("stations") or []:
        sid = str(row.get("station_id", ""))
        raw_list = by_station.get(sid) or by_station.get(sid.lstrip("0")) or []
        items = [x for x in raw_list if isinstance(x, dict)]
        stations_out.append(
            {
                "station_id": sid,
                "road": row.get("road"),
                "place": row.get("place"),
                "segment": row.get("segment"),
                "right": row.get("right"),
                "left": row.get("left"),
                "lat": _center_lat(row),
                "lon": _center_lon(row),
                "anomaly_count": len(items),
                "anomalies": items,
            }
        )

    message: str | None = None
    source_run = anom.get("source_run_id")
    if source_run:
        message = f"Anomalies from TimesNet run {source_run!r} (see dataset_a_map_anomalies.json)."

    return {
        "dataset": dataset,
        "source_note": geo.get("source_note"),
        "updated_at": anom.get("updated_at"),
        "message": message,
        "stations": stations_out,
    }


def _center_lat(row: dict) -> float | None:
    right, left = row.get("right"), row.get("left")
    if right and left and isinstance(right, dict) and isinstance(left, dict):
        if right.get("lat") is not None and left.get("lat") is not None:
            return (float(right["lat"]) + float(left["lat"])) / 2.0
    for key in ("right", "left"):
        p = row.get(key)
        if isinstance(p, dict) and p.get("lat") is not None:
            return float(p["lat"])
    return None


def _center_lon(row: dict) -> float | None:
    right, left = row.get("right"), row.get("left")
    if right and left and isinstance(right, dict) and isinstance(left, dict):
        if right.get("lon") is not None and left.get("lon") is not None:
            return (float(right["lon"]) + float(left["lon"])) / 2.0
    for key in ("right", "left"):
        p = row.get(key)
        if isinstance(p, dict) and p.get("lon") is not None:
            return float(p["lon"])
    return None
