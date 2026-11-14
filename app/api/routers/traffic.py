from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

import app.core.config as cfg
from app.api.regions import (
    MAX_POINTS_HINT,
    RANGE_PRESETS,
    bucket_expr,
    group_locations,
)
from app.api.traffic_range import resolve_traffic_window
from app.core.datasets import FEATURE_TABLES_BY_DATASET, get_dataset, list_datasets
from app.core.db import ClickHouseDB

router = APIRouter()


@dataclass(frozen=True)
class TrafficQuery:
    dataset: str
    table: str
    locations: list[str]
    locations_sql: str
    date_from: str
    date_to: str
    resolution: str
    bucket: str


def _fetch_location_ids(table: str) -> list[str]:
    query = f"""
        SELECT DISTINCT location_id
        FROM {cfg.DB_NAME}.{table}
        ORDER BY location_id
    """
    with ClickHouseDB.from_config() as db:
        result = db.query_raw(query)
    return [row[0] for row in result.result_rows]


def _table_for_dataset(dataset: str) -> str:
    if dataset not in FEATURE_TABLES_BY_DATASET:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset!r}")
    return FEATURE_TABLES_BY_DATASET[dataset]


def _parse_locations(stations: str) -> list[str]:
    return [station.strip() for station in stations.split(",") if station.strip()]


def _resolve_query(
    *,
    dataset: str,
    stations: str,
    date_from: str | None,
    date_to: str | None,
    range_preset: str | None,
    resolution: str | None,
) -> TrafficQuery | None:
    table = _table_for_dataset(dataset)
    locations = _parse_locations(stations)
    if not locations:
        return None

    window = resolve_traffic_window(
        range_preset=range_preset,
        resolution=resolution,
        date_from=date_from,
        date_to=date_to,
    )
    return TrafficQuery(
        dataset=dataset,
        table=table,
        locations=locations,
        locations_sql=", ".join(f"'{location}'" for location in locations),
        date_from=window.date_from,
        date_to=window.date_to,
        resolution=window.resolution,
        bucket=bucket_expr(window.resolution),
    )


def _query_rows(query: str):
    with ClickHouseDB.from_config() as db:
        return db.query_raw(query).result_rows


def _downsample_wide(wide: pd.DataFrame) -> pd.DataFrame:
    wide = wide.sort_index()
    if len(wide) > MAX_POINTS_HINT:
        step = max(1, len(wide) // MAX_POINTS_HINT)
        wide = wide.iloc[::step]
    return wide


@router.get("/api/datasets")
def get_datasets():
    """Datasets with stations grouped by base id and lanes."""
    datasets_out = []
    for dataset in list_datasets():
        table = dataset.feature_table
        try:
            locs = _fetch_location_ids(table)
        except Exception:
            locs = []
        if not locs:
            continue
        stations = group_locations(locs)
        datasets_out.append(
            {
                "id": dataset.id,
                "label": dataset.label,
                "table": table,
                "station_count": len(stations),
                "location_count": len(locs),
                "stations": [
                    {
                        "base_id": station.base_id,
                        "label": station.label,
                        "lanes": [
                            {"location_id": lane.location_id, "lane": lane.lane}
                            for lane in station.lanes
                        ],
                    }
                    for station in stations
                ],
            }
        )
    return {
        "datasets": datasets_out,
        "range_presets": RANGE_PRESETS,
        "max_points_hint": MAX_POINTS_HINT,
    }


@router.get("/api/traffic")
def get_traffic(
    dataset: str = Query(..., description="Dataset id: dataset-a | dataset-b"),
    stations: str = Query(..., description="Comma-separated location_id values"),
    date_from: str | None = Query(None, description="UTC start YYYY-MM-DD HH:MM:SS"),
    date_to: str | None = Query(None, description="UTC end YYYY-MM-DD HH:MM:SS"),
    range_preset: str | None = Query(
        None,
        description="Window preset: 15m | 1h | 1d | 7d | 30d (overrides span when dates omitted)",
    ),
    resolution: str | None = Query(
        None,
        description="Bucket: 15min | 1h | 6h | 1d (auto from preset if omitted)",
    ),
):
    get_dataset(dataset)
    params = _resolve_query(
        dataset=dataset,
        stations=stations,
        date_from=date_from,
        date_to=date_to,
        range_preset=range_preset,
        resolution=resolution,
    )
    if params is None:
        return {"timestamps": [], "series": {}, "meta": {}}

    query = f"""
        SELECT
            {params.bucket} AS ds,
            location_id,
            count() AS volume
        FROM {cfg.DB_NAME}.{params.table}
        WHERE location_id IN ({params.locations_sql})
          AND time_utc >= toDateTime('{params.date_from}', 'UTC')
          AND time_utc <= toDateTime('{params.date_to}', 'UTC')
        GROUP BY ds, location_id
        ORDER BY ds, location_id
    """
    df = pd.DataFrame(
        _query_rows(query),
        columns=["ds", "location_id", "volume"],
    )
    if df.empty:
        return {
            "timestamps": [],
            "series": {},
            "meta": {
                "dataset": dataset,
                "resolution": params.resolution,
                "date_from": params.date_from,
                "date_to": params.date_to,
                "points": 0,
            },
        }

    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    wide = _downsample_wide(df.pivot(index="ds", columns="location_id", values="volume").fillna(0))
    timestamps = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in wide.index]
    series = {str(column): wide[column].astype(int).tolist() for column in wide.columns}

    return {
        "timestamps": timestamps,
        "series": series,
        "meta": {
            "dataset": dataset,
            "resolution": params.resolution,
            "range_preset": range_preset,
            "date_from": params.date_from,
            "date_to": params.date_to,
            "points": len(timestamps),
            "series_count": len(series),
        },
    }
