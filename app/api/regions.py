"""Dashboard region metadata and location_id grouping helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

LANE_SUFFIX_RE = re.compile(r"^l\d+$", re.IGNORECASE)

# Preset span limits (hours) and default ClickHouse bucket for /api/traffic.
RANGE_PRESETS: dict[str, dict[str, str | int]] = {
    "15m": {"hours": 6, "resolution": "15min", "label": "6h · 15min"},
    "1h": {"hours": 12, "resolution": "1h", "label": "12h · hourly"},
    "1d": {"hours": 24, "resolution": "1h", "label": "1 day · 24×1h"},
    "7d": {"hours": 24 * 7, "resolution": "1h", "label": "7d · hourly"},
    "30d": {"hours": 24 * 30, "resolution": "6h", "label": "30d · 6h"},
}

MAX_POINTS_HINT = 2500


@dataclass
class LaneInfo:
    location_id: str
    lane: str


@dataclass
class StationGroup:
    base_id: str
    label: str
    lanes: list[LaneInfo]


def split_location(location_id: str) -> tuple[str, str | None]:
    parts = location_id.split("-")
    if len(parts) >= 2 and LANE_SUFFIX_RE.match(parts[-1]):
        return "-".join(parts[:-1]), parts[-1].lower()
    return location_id, None


def station_label(base_id: str) -> str:
    parts = base_id.split("-")
    if len(parts) >= 2 and parts[1].upper() in ("P", "L") and len(parts) >= 3:
        return f"{parts[0]} · {parts[1]}/{parts[2]}"
    if len(parts) >= 2:
        return f"{parts[0]} · {parts[1]}"
    return base_id


def group_locations(location_ids: list[str]) -> list[StationGroup]:
    buckets: dict[str, list[LaneInfo]] = {}
    for loc in sorted(location_ids):
        base, lane = split_location(loc)
        if lane:
            buckets.setdefault(base, []).append(LaneInfo(location_id=loc, lane=lane))
        else:
            buckets.setdefault(base, []).append(LaneInfo(location_id=loc, lane=""))

    groups: list[StationGroup] = []
    for base_id, lanes in sorted(buckets.items()):
        groups.append(
            StationGroup(
                base_id=base_id,
                label=station_label(base_id),
                lanes=sorted(lanes, key=lambda x: x.lane),
            )
        )
    return groups


def bucket_expr(resolution: str) -> str:
    if resolution == "15min":
        return "toStartOfFifteenMinutes(time_utc)"
    if resolution == "1h":
        return "toStartOfHour(time_utc)"
    if resolution == "6h":
        return "toStartOfInterval(time_utc, INTERVAL 6 HOUR)"
    if resolution == "1d":
        return "toStartOfDay(time_utc)"
    raise ValueError(f"Unsupported resolution: {resolution!r}")


def resolve_resolution(range_preset: str | None, resolution: str | None) -> str:
    if resolution:
        return resolution
    if range_preset and range_preset in RANGE_PRESETS:
        return str(RANGE_PRESETS[range_preset]["resolution"])
    return "1h"
