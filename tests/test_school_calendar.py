"""School and college calendar flags."""

from __future__ import annotations

import pandas as pd
import pytest
from models.utils.constants import DEFAULT_CALENDAR_COLUMNS
from models.utils.multivariate_data_prep import _calendar_columns
from models.utils.school_calendar import SCHOOL_CALENDAR_COLUMNS, build_school_calendar

pytestmark = pytest.mark.unit


def _flags(*days: str) -> pd.DataFrame:
    return build_school_calendar(pd.DatetimeIndex(pd.to_datetime(list(days))))


def test_east_side_fall_break_monday_is_break():
    assert _flags("2022-09-26").loc["2022-09-26", "is_school_break_east_side"]


def test_east_side_normal_october_school_day_is_not_break():
    assert not _flags("2022-10-12").loc["2022-10-12", "is_school_break_east_side"]


def test_east_side_first_and_last_school_day_are_not_breaks():
    flags = _flags("2022-08-09", "2023-06-01", "2023-06-02")["is_school_break_east_side"]
    assert not flags["2022-08-09"]
    assert not flags["2023-06-01"]
    assert flags["2023-06-02"]


def test_morgan_hill_extra_day_is_break():
    assert _flags("2022-10-07").loc["2022-10-07", "is_school_break_morgan_hill"]


def test_morgan_hill_has_no_fall_break_week():
    flags = _flags("2022-09-26")
    assert not flags.loc["2022-09-26", "is_school_break_morgan_hill"]


def test_oak_grove_summer_day_is_break():
    assert _flags("2023-07-10").loc["2023-07-10", "is_school_break_oak_grove"]


def test_ordinary_weekend_is_not_a_break():
    flags = _flags("2022-10-15", "2022-10-16")  # Saturday, Sunday
    assert not flags[
        [
            "is_school_break_morgan_hill",
            "is_school_break_oak_grove",
            "is_school_break_east_side",
        ]
    ].any(axis=None)


def test_gavilan_spring_break_day_is_not_regular_semester():
    row = _flags("2023-04-05").loc["2023-04-05"]
    assert row["is_gavilan_spring_break"]
    assert not row["is_gavilan_regular_semester"]
    assert not row["is_gavilan_summer_session"]


def test_gavilan_regular_semester_day():
    row = _flags("2023-03-01").loc["2023-03-01"]
    assert row["is_gavilan_regular_semester"]
    assert not row["is_gavilan_spring_break"]


def test_gavilan_summer_session_day():
    row = _flags("2023-06-20").loc["2023-06-20"]
    assert row["is_gavilan_summer_session"]
    assert not row["is_gavilan_regular_semester"]


def test_gavilan_winter_intersession_has_no_flag():
    row = _flags("2022-12-28").loc["2022-12-28"]
    assert not row[
        ["is_gavilan_spring_break", "is_gavilan_regular_semester", "is_gavilan_summer_session"]
    ].any()


def test_calendar_columns_feed_the_row_frame():
    for column in SCHOOL_CALENDAR_COLUMNS:
        assert column in DEFAULT_CALENDAR_COLUMNS
    index = pd.date_range("2022-09-26 00:00", periods=3, freq="5min", tz="America/Los_Angeles")
    out = _calendar_columns(index.tz_convert("UTC"), DEFAULT_CALENDAR_COLUMNS)
    assert list(out.columns) == DEFAULT_CALENDAR_COLUMNS
    assert (out["is_school_break_east_side"] == 1.0).all()
    assert (out["is_school_break_morgan_hill"] == 0.0).all()
