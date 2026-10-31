"""Shared constants for multivariate anomaly and forecasting pipelines."""

from __future__ import annotations

SUPPORTED_FREQS = ("15min", "5min")

# A 5-minute row is a real observation only at 100. Lower values were imputed
# upstream and stay out of the training mask.
OBSERVED_ACTIVE_MIN_PERCENT = 100.0

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
    "is_holiday_ca",
    "is_pre_holiday",
    "is_post_holiday",
    "is_bridge_day",
    "is_long_weekend",
    "is_pre_long_weekend",
    "is_shopping_sunday",
    "is_school_holiday",
    "is_school_break_morgan_hill",
    "is_school_break_san_jose",
    "is_college_break_gavilan",
    "is_school_year_start",
    "is_summer_vacation",
]

EXTENDED_CALENDAR_COLUMNS = DEFAULT_CALENDAR_COLUMNS.copy()

SUPPORTED_THRESHOLD_METHODS = ("robust_z", "percentile")

DEFAULT_WINDOW_15MIN = 192
DEFAULT_WINDOW_5MIN = 576


def default_window_for(freq: str) -> int:
    if freq == "15min":
        return DEFAULT_WINDOW_15MIN
    if freq == "5min":
        return DEFAULT_WINDOW_5MIN
    raise ValueError(f"Unsupported freq: {freq!r}")
