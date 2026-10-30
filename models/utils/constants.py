"""Shared constants for multivariate anomaly and forecasting pipelines."""

from __future__ import annotations

SUPPORTED_FREQS = ("1h", "15min")

DEFAULT_CALENDAR_COLUMNS: list[str] = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "month_sin",
    "month_cos",
    "weekofyear_sin",
    "weekofyear_cos",
    "dom_sin",
    "dom_cos",
    "is_month_start",
    "is_month_end",
    "days_to_nearest_holiday",
    "is_holiday_pl",
    "is_pre_holiday",
    "is_post_holiday",
    "is_bridge_day",
    "is_long_weekend_period",
    "is_pre_long_weekend",
    "is_shopping_sunday",
    "is_school_holiday",
    "is_ferie_zimowe",
    "is_wakacje_letnie",
]

EXTENDED_CALENDAR_COLUMNS = DEFAULT_CALENDAR_COLUMNS.copy()

VEHICLE_CLASS_LABELS: dict[int, str] = {
    1: "motorcycle",
    2: "car+trailer",
    3: "truck",
    5: "bus",
    7: "car",
    8: "truck+trailer",
    9: "tractor+trailer",
    10: "unknown-10",
    11: "LCV",
}

SUPPORTED_THRESHOLD_METHODS = ("robust_z", "percentile")

DEFAULT_WINDOW = 48
DEFAULT_WINDOW_15MIN = 192


def default_window_for(freq: str) -> int:
    if freq == "1h":
        return DEFAULT_WINDOW
    if freq == "15min":
        return DEFAULT_WINDOW_15MIN
    raise ValueError(f"Unsupported freq: {freq!r}")
