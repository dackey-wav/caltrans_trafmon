"""School and college calendar flags for the Morgan Hill / Gilroy area.

Every table lists inclusive (start, end) date ranges in California local time.
Districts stay separate. Ordinary weekends are not breaks.

K-12 break flag = listed breaks + summer vacation + extra no-school days.
First and last days of school are not breaks.

Gavilan College has three disjoint flags: regular semester (fall and spring
main terms without spring break), spring break, and summer session. Thanksgiving
closures and winter intersession belong to none of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Range = tuple[str, str]

SCHOOL_CALENDAR_COLUMNS: tuple[str, ...] = (
    "is_school_break_morgan_hill",
    "is_school_break_oak_grove",
    "is_school_break_east_side",
    "is_gavilan_spring_break",
    "is_gavilan_regular_semester",
    "is_gavilan_summer_session",
)

# East Side Union HSD. First day 2022-08-09 and last day 2023-06-01 are not breaks.
EAST_SIDE_BREAKS: tuple[Range, ...] = (
    # 2022/23
    ("2022-09-26", "2022-09-30"),  # fall
    ("2022-11-23", "2022-11-25"),  # Thanksgiving
    ("2022-12-26", "2023-01-06"),  # winter
    ("2023-02-20", "2023-02-24"),  # February
    ("2023-04-10", "2023-04-14"),  # spring
    ("2023-06-02", "2023-08-02"),  # summer
    # 2023/24
    ("2023-10-02", "2023-10-06"),
    ("2023-11-20", "2023-11-24"),
    ("2023-12-22", "2024-01-05"),
    ("2024-02-19", "2024-02-23"),
    ("2024-04-01", "2024-04-05"),
    ("2024-05-31", "2024-08-06"),
    # 2024/25
    ("2024-10-07", "2024-10-11"),
    ("2024-11-25", "2024-11-29"),
    ("2024-12-23", "2025-01-03"),
    ("2025-02-17", "2025-02-21"),
    ("2025-04-21", "2025-04-25"),
    ("2025-06-04", "2025-08-06"),
    # 2025/26
    ("2025-10-06", "2025-10-10"),
    ("2025-11-24", "2025-11-28"),
    ("2025-12-22", "2026-01-02"),
    ("2026-02-16", "2026-02-20"),
    ("2026-04-06", "2026-04-10"),
    # Summer starts 2026-06-05. No end date is published. The end below is
    # inferred from the pattern of earlier years, not taken from a PDF.
    ("2026-06-05", "2026-08-06"),
)
EAST_SIDE_EXTRA_DAYS: tuple[str, ...] = ()

# Morgan Hill USD. No fall-break week.
MORGAN_HILL_BREAKS: tuple[Range, ...] = (
    # 2022/23
    ("2022-11-21", "2022-11-25"),  # Thanksgiving
    ("2022-12-26", "2023-01-06"),  # winter
    ("2023-02-20", "2023-02-24"),  # Presidents week
    ("2023-04-10", "2023-04-14"),  # spring
    ("2023-06-10", "2023-08-14"),  # summer
    # 2023/24
    ("2023-11-20", "2023-11-24"),
    ("2023-12-22", "2024-01-05"),
    ("2024-02-19", "2024-02-23"),
    ("2024-04-01", "2024-04-05"),
    ("2024-06-07", "2024-08-14"),
    # 2024/25
    ("2024-11-25", "2024-11-29"),
    ("2024-12-23", "2025-01-03"),
    ("2025-02-17", "2025-02-21"),
    ("2025-04-14", "2025-04-18"),
    ("2025-06-07", "2025-08-13"),
    # 2025/26
    ("2025-11-24", "2025-11-28"),
    ("2025-12-22", "2026-01-02"),
    ("2026-02-16", "2026-02-20"),
    ("2026-03-30", "2026-04-06"),
    # Summer starts 2026-06-06. No end date is published. The end below is
    # inferred from the pattern of earlier years, not taken from a PDF.
    ("2026-06-06", "2026-08-13"),
)
MORGAN_HILL_EXTRA_DAYS: tuple[str, ...] = (
    "2022-10-07",
    "2022-11-14",
    "2023-10-06",
    "2024-10-04",
    "2025-10-03",
)

# Oak Grove SD.
OAK_GROVE_BREAKS: tuple[Range, ...] = (
    # 2022/23
    ("2022-09-26", "2022-09-30"),  # fall
    ("2022-11-21", "2022-11-25"),  # Thanksgiving
    ("2022-12-26", "2023-01-06"),  # winter
    ("2023-02-20", "2023-02-24"),  # February
    ("2023-04-10", "2023-04-14"),  # spring
    ("2023-06-14", "2023-08-08"),  # summer
    # 2023/24
    ("2023-10-02", "2023-10-06"),
    ("2023-11-20", "2023-11-24"),
    ("2023-12-25", "2024-01-05"),
    ("2024-02-19", "2024-02-23"),
    ("2024-04-01", "2024-04-05"),
    ("2024-06-08", "2024-08-07"),
    # 2024/25
    ("2024-10-07", "2024-10-11"),
    ("2024-11-25", "2024-11-29"),
    ("2024-12-23", "2025-01-03"),
    ("2025-02-17", "2025-02-21"),
    ("2025-04-21", "2025-04-25"),
    ("2025-06-11", "2025-08-10"),
    # 2025/26
    ("2025-10-06", "2025-10-10"),
    ("2025-11-24", "2025-11-28"),
    ("2025-12-22", "2026-01-02"),
    ("2026-02-16", "2026-02-20"),
    ("2026-04-06", "2026-04-10"),
    ("2026-06-10", "2026-08-12"),
)
OAK_GROVE_EXTRA_DAYS: tuple[str, ...] = (
    "2022-11-01",
    "2022-11-08",
    "2023-01-17",
    "2023-02-17",
    "2023-04-07",
    "2023-11-01",
    "2024-01-16",
    "2024-02-16",
    "2024-03-29",
    "2024-11-01",
    "2024-11-05",
    "2025-01-21",
    "2025-02-14",
    "2025-04-18",
    "2025-11-10",
    "2026-01-20",
    "2026-03-30",
)

# Gavilan College. Fall and spring main terms (spring break is carved out below).
GAVILAN_TERMS: tuple[Range, ...] = (
    ("2022-08-22", "2022-12-10"),  # fall 2022
    ("2023-01-30", "2023-05-26"),  # spring 2023
    ("2023-08-28", "2023-12-16"),  # fall 2023
    ("2024-01-29", "2024-05-24"),  # spring 2024
    ("2024-08-26", "2024-12-14"),  # fall 2024
    ("2025-01-27", "2025-05-23"),  # spring 2025
    ("2025-08-25", "2025-12-13"),  # fall 2025
    ("2026-01-26", "2026-05-22"),  # spring 2026
)
GAVILAN_SPRING_BREAKS: tuple[Range, ...] = (
    ("2023-04-03", "2023-04-09"),
    ("2024-04-01", "2024-04-07"),
    ("2025-04-07", "2025-04-13"),
    ("2026-04-06", "2026-04-12"),
)
GAVILAN_SUMMER_SESSIONS: tuple[Range, ...] = (
    ("2023-06-12", "2023-07-21"),
    ("2024-05-28", "2024-08-24"),
    ("2025-05-27", "2025-08-16"),
    ("2026-05-26", "2026-08-22"),
)


def _in_ranges(index: pd.DatetimeIndex, ranges: tuple[Range, ...]) -> pd.Series:
    """True where a day falls inside any inclusive range."""
    days = index.to_numpy(dtype="datetime64[D]")
    flags = np.zeros(len(days), dtype=bool)
    for start, end in ranges:
        flags |= (days >= np.datetime64(start, "D")) & (days <= np.datetime64(end, "D"))
    return pd.Series(flags, index=index)


def _in_days(index: pd.DatetimeIndex, days: tuple[str, ...]) -> pd.Series:
    return pd.Series(index.isin(pd.DatetimeIndex(list(days))), index=index)


def build_school_calendar(day_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return the six boolean school flags for a tz-naive local day index."""
    idx = pd.DatetimeIndex(day_index)
    if idx.tz is not None:
        raise ValueError("day_index must be tz-naive local dates")
    idx = idx.normalize()

    out = pd.DataFrame(index=day_index)
    out["is_school_break_morgan_hill"] = (
        _in_ranges(idx, MORGAN_HILL_BREAKS) | _in_days(idx, MORGAN_HILL_EXTRA_DAYS)
    ).to_numpy()
    out["is_school_break_oak_grove"] = (
        _in_ranges(idx, OAK_GROVE_BREAKS) | _in_days(idx, OAK_GROVE_EXTRA_DAYS)
    ).to_numpy()
    out["is_school_break_east_side"] = (
        _in_ranges(idx, EAST_SIDE_BREAKS) | _in_days(idx, EAST_SIDE_EXTRA_DAYS)
    ).to_numpy()

    spring_break = _in_ranges(idx, GAVILAN_SPRING_BREAKS)
    out["is_gavilan_spring_break"] = spring_break.to_numpy()
    out["is_gavilan_regular_semester"] = (
        _in_ranges(idx, GAVILAN_TERMS) & ~spring_break
    ).to_numpy()
    out["is_gavilan_summer_session"] = _in_ranges(idx, GAVILAN_SUMMER_SESSIONS).to_numpy()
    return out
