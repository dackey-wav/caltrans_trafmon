"""Shared date window + resolution resolution for traffic endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException

from app.api.regions import RANGE_PRESETS, resolve_resolution


@dataclass(frozen=True)
class TrafficWindow:
    date_from: str
    date_to: str
    resolution: str
    range_preset: str | None


def resolve_traffic_window(
    *,
    range_preset: str | None,
    resolution: str | None,
    date_from: str | None,
    date_to: str | None,
) -> TrafficWindow:
    res = resolve_resolution(range_preset, resolution)

    if range_preset and range_preset in RANGE_PRESETS:
        if date_to:
            end = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            if end.tzinfo is None:
                end = end.replace(tzinfo=UTC)
        else:
            end = datetime.now(UTC)

        if range_preset == "1d":
            # One calendar day (UTC): 24 hourly buckets, not a rolling 24h window.
            start = end.replace(hour=0, minute=0, second=0, microsecond=0)
            end = end.replace(hour=23, minute=59, second=59, microsecond=0)
        else:
            hours = int(RANGE_PRESETS[range_preset]["hours"])
            start = end - timedelta(hours=hours)

        df = start.strftime("%Y-%m-%d %H:%M:%S")
        dt = end.strftime("%Y-%m-%d %H:%M:%S")
    else:
        if not date_from or not date_to:
            raise HTTPException(
                status_code=400,
                detail="Provide date_from and date_to, or use range_preset",
            )
        start = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
        end = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        df = start.strftime("%Y-%m-%d %H:%M:%S")
        dt = end.strftime("%Y-%m-%d %H:%M:%S")

    span_hours = (end - start).total_seconds() / 3600.0
    max_hours_by_res = {"15min": 24 * 7, "1h": 24 * 45, "6h": 24 * 120, "1d": 24 * 400}
    if span_hours > max_hours_by_res.get(res, 24 * 30):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Range too long for resolution={res} ({span_hours:.0f}h). "
                "Use a shorter preset or coarser resolution."
            ),
        )

    return TrafficWindow(
        date_from=df,
        date_to=dt,
        resolution=res,
        range_preset=range_preset,
    )
