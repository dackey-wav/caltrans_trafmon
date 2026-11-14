"""Shared constants for multivariate TimesNet pipelines."""

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
    # Polish leftover. Always 0; kept so existing datasets keep their column set.
    "is_shopping_sunday",
    "is_school_break_morgan_hill",
    "is_school_break_oak_grove",
    "is_school_break_east_side",
    "is_gavilan_spring_break",
    "is_gavilan_regular_semester",
    "is_gavilan_summer_session",
]

EXTENDED_CALENDAR_COLUMNS = DEFAULT_CALENDAR_COLUMNS.copy()

LOCAL_TZ = "America/Los_Angeles"

STEP_MINUTES = {"15min": 15, "5min": 5}

# Input window and forecast horizon default to 2 days and 1 day.
DEFAULT_WINDOW_15MIN = 192
DEFAULT_WINDOW_5MIN = 576
DEFAULT_PRED_LEN_15MIN = 96
DEFAULT_PRED_LEN_5MIN = 288


def step_minutes(freq: str) -> int:
    if freq not in STEP_MINUTES:
        raise ValueError(f"Unsupported freq: {freq!r}")
    return STEP_MINUTES[freq]


def steps_per_hour(freq: str) -> int:
    return 60 // step_minutes(freq)


def steps_per_day(freq: str) -> int:
    return 24 * steps_per_hour(freq)


def default_window_for(freq: str) -> int:
    if freq == "15min":
        return DEFAULT_WINDOW_15MIN
    if freq == "5min":
        return DEFAULT_WINDOW_5MIN
    raise ValueError(f"Unsupported freq: {freq!r}")


def default_pred_len_for(freq: str) -> int:
    if freq == "15min":
        return DEFAULT_PRED_LEN_15MIN
    if freq == "5min":
        return DEFAULT_PRED_LEN_5MIN
    raise ValueError(f"Unsupported freq: {freq!r}")
