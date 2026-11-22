"""Local calendar flags for the San Diego station."""

from __future__ import annotations

import pandas as pd
import pytest
from models.utils.constants import DEFAULT_CALENDAR_COLUMNS
from models.utils.local_calendar import LOCAL_CALENDAR_COLUMNS, build_local_calendar
from models.utils.multivariate_data_prep import _calendar_columns

pytestmark = pytest.mark.unit


def _flags(*days: str) -> pd.DataFrame:
    return build_local_calendar(pd.DatetimeIndex(pd.to_datetime(list(days))))


def test_sdusd_breaks_and_school_days():
    f = _flags("2025-08-08", "2025-08-11", "2025-10-13", "2025-12-22", "2026-01-05",
               "2026-03-30", "2026-05-28", "2026-05-29")["is_school_break_sdusd"]
    # summer, first day, non-instructional day, winter, back to school,
    # spring break, last day, summer
    assert list(f) == [True, False, True, True, False, True, False, True]


def test_universities_in_session_and_breaks():
    f = _flags("2025-09-24", "2025-09-25", "2025-12-13", "2025-12-15", "2025-11-27", "2026-03-31")
    assert list(f["is_ucsd_in_session"]) == [False, True, True, False, True, True]
    # USD fall runs Sep 3 - Dec 19 minus Thanksgiving; spring break Mar 30 - Apr 6
    assert list(f["is_usd_in_session"]) == [True, True, True, True, False, False]


def test_events():
    f = _flags("2025-07-04", "2025-07-14", "2025-06-16", "2025-06-18", "2025-07-25", "2025-09-04",
               "2025-09-05", "2025-09-01")
    assert list(f["is_padres_home_game"]) == [True, False, False, False, False, False, False, True]
    # fair closed Mon/Tue, open Wed; no racing Thursday of closing week; Labor Day card
    assert list(f["is_del_mar_event"]) == [True, False, False, True, True, False, True, True]
    assert list(f["is_comic_con"]) == [False, False, False, False, True, False, False, False]


def test_flags_reach_the_model_columns():
    assert set(LOCAL_CALENDAR_COLUMNS) <= set(DEFAULT_CALENDAR_COLUMNS)
    index = pd.date_range("2025-07-24 07:00", periods=4, freq="15min", tz="UTC")
    out = _calendar_columns(index, LOCAL_CALENDAR_COLUMNS)
    assert list(out.columns) == list(LOCAL_CALENDAR_COLUMNS)
    assert (out["is_comic_con"] == 1.0).all()
    assert (out["is_school_break_sdusd"] == 1.0).all()
