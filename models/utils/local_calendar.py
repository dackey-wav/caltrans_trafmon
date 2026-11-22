"""Local calendar flags for PeMS VDS 1108617 (I-5 northbound at Old Town, San Diego).

The detector sits on I-5 NB just north of downtown, at the I-8 interchange.
Traffic there carries commuters from downtown and the airport to the
University of San Diego (Linda Vista, ~1.5 km north), UC San Diego and the
North County coast (Del Mar), and crowds leaving downtown events.

Flags (one value per local day, inclusive ranges in California local time):

    is_school_break_sdusd   San Diego Unified (K-12) has no school: breaks,
                            summer, non-instructional days. Weekends are not
                            breaks. Statutory holidays are left to the holiday
                            flags, except Lincoln Day, which `holidays` lacks.
    is_ucsd_in_session      UC San Diego regular quarter, first day of
                            instruction through the last final exam.
    is_usd_in_session       University of San Diego semester, first day of
                            classes through finals, minus Thanksgiving, spring
                            and Easter breaks.
    is_padres_home_game     San Diego Padres home game at Petco Park (downtown).
    is_del_mar_event        San Diego County Fair or Del Mar summer racing day
                            at the Del Mar Fairgrounds (I-5, ~25 km north).
    is_comic_con            Comic-Con International at the downtown convention
                            center, preview night included.

Sources (checked 2026-11-17):
    SDUSD 2025-26: board-adopted calendar PDF (sandiegounified.org CDN).
    SDUSD 2024-25: district news release (first day 2024-08-12) and Patch
        (Thanksgiving, winter, spring break, last day 2025-05-29).
    SDUSD 2026-27: californiaschoolcalendar.com, compiled from the district PDF;
        only dates up to the end of the downloaded data matter here.
    UCSD: blink.ucsd.edu enrollment calendars 2024, 2025, 2026.
    USD: sandiego.edu academic calendars 2024-25 and 2025-26 (PDF), 2026-27
        undergraduate catalog.
    Padres: MLB Stats API schedule, home games at Petco Park, 2025 and 2026
        seasons including the 2026 postseason. No games Nov 2024 - Mar 2025.
    County Fair: NBC San Diego / Coast News (2025), Del Mar Fairgrounds (2026);
        closed Mondays and Tuesdays. Racing: Del Mar Thoroughbred Club via
        Solana Beach and DRF; Thursday-Sunday plus Labor Day.
    Comic-Con: Comic-Con International via SD Tourism listings.

San Diego Community College District (Mesa, City) was left out: its calendar
PDFs were unreachable and Mesa College feeds SR-163 rather than this segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Range = tuple[str, str]

LOCAL_CALENDAR_COLUMNS: tuple[str, ...] = (
    "is_school_break_sdusd",
    "is_ucsd_in_session",
    "is_usd_in_session",
    "is_padres_home_game",
    "is_del_mar_event",
    "is_comic_con",
)

SDUSD_BREAKS: tuple[Range, ...] = (
    # 2024/25: first day 2024-08-12, last day 2025-05-29
    ("2024-11-25", "2024-11-29"),  # Thanksgiving
    ("2024-12-23", "2025-01-06"),  # winter
    ("2025-03-31", "2025-04-04"),  # spring
    ("2025-05-30", "2025-08-10"),  # summer
    # 2025/26: first day 2025-08-11, last day 2026-05-28
    ("2025-11-24", "2025-11-28"),
    ("2025-12-22", "2026-01-02"),
    ("2026-03-30", "2026-04-03"),
    ("2026-05-29", "2026-08-09"),
    # 2026/27: first day 2026-08-10
    ("2026-11-23", "2026-11-27"),
    ("2026-12-21", "2027-01-01"),
    ("2027-03-29", "2027-04-02"),
)
SDUSD_EXTRA_DAYS: tuple[str, ...] = (
    "2025-02-14",  # Lincoln Day (2024/25, by the Feb pattern of other years)
    "2025-10-13",  # non-instructional
    "2025-11-10",  # non-instructional
    "2026-02-13",  # Lincoln Day
    "2026-03-09",  # non-instructional
    "2027-02-12",  # Lincoln Day
)

UCSD_QUARTERS: tuple[Range, ...] = (
    ("2024-09-26", "2024-12-14"),  # fall 2024
    ("2025-01-06", "2025-03-22"),  # winter 2025
    ("2025-03-31", "2025-06-13"),  # spring 2025
    ("2025-09-25", "2025-12-13"),  # fall 2025
    ("2026-01-05", "2026-03-21"),  # winter 2026
    ("2026-03-30", "2026-06-12"),  # spring 2026
    ("2026-09-24", "2026-12-12"),  # fall 2026
)

USD_SEMESTERS: tuple[Range, ...] = (
    ("2024-09-04", "2024-12-20"),  # fall 2024
    ("2025-01-30", "2025-05-23"),  # spring 2025
    ("2025-09-03", "2025-12-19"),  # fall 2025
    ("2026-01-26", "2026-05-20"),  # spring 2026
    ("2026-09-01", "2026-12-18"),  # fall 2026
)
USD_BREAKS: tuple[Range, ...] = (
    ("2024-11-27", "2024-11-29"),  # Thanksgiving
    ("2025-03-10", "2025-03-14"),  # spring break
    ("2025-04-17", "2025-04-21"),  # Easter
    ("2025-11-26", "2025-11-28"),
    ("2026-03-30", "2026-04-06"),  # Easter/spring break
    ("2026-11-25", "2026-11-27"),
)

PADRES_HOME_GAMES: tuple[str, ...] = (
    '2025-03-27', '2025-03-28', '2025-03-29', '2025-03-30', '2025-03-31',
    '2025-04-01', '2025-04-02', '2025-04-11', '2025-04-12', '2025-04-13', '2025-04-14', '2025-04-15', '2025-04-16', '2025-04-25', '2025-04-26', '2025-04-27', '2025-04-29', '2025-04-30',
    '2025-05-12', '2025-05-13', '2025-05-14', '2025-05-16', '2025-05-17', '2025-05-18', '2025-05-26', '2025-05-27', '2025-05-28', '2025-05-30', '2025-05-31',
    '2025-06-01', '2025-06-09', '2025-06-10', '2025-06-11', '2025-06-20', '2025-06-21', '2025-06-22', '2025-06-23', '2025-06-24', '2025-06-25',
    '2025-07-04', '2025-07-05', '2025-07-06', '2025-07-07', '2025-07-08', '2025-07-09', '2025-07-10', '2025-07-11', '2025-07-12', '2025-07-13', '2025-07-28', '2025-07-29', '2025-07-30',
    '2025-08-01', '2025-08-02', '2025-08-03', '2025-08-08', '2025-08-09', '2025-08-10', '2025-08-18', '2025-08-19', '2025-08-20', '2025-08-21', '2025-08-22', '2025-08-23', '2025-08-24',
    '2025-09-01', '2025-09-02', '2025-09-03', '2025-09-08', '2025-09-09', '2025-09-10', '2025-09-11', '2025-09-12', '2025-09-13', '2025-09-14', '2025-09-22', '2025-09-23', '2025-09-24', '2025-09-26', '2025-09-27', '2025-09-28',
    '2026-03-26', '2026-03-27', '2026-03-28', '2026-03-30', '2026-03-31',
    '2026-04-01', '2026-04-09', '2026-04-10', '2026-04-11', '2026-04-12', '2026-04-14', '2026-04-15', '2026-04-16', '2026-04-27', '2026-04-28', '2026-04-29',
    '2026-05-01', '2026-05-02', '2026-05-03', '2026-05-07', '2026-05-08', '2026-05-09', '2026-05-10', '2026-05-18', '2026-05-19', '2026-05-20', '2026-05-22', '2026-05-23', '2026-05-24', '2026-05-25', '2026-05-26', '2026-05-27',
    '2026-06-05', '2026-06-06', '2026-06-07', '2026-06-08', '2026-06-09', '2026-06-10', '2026-06-22', '2026-06-23', '2026-06-24', '2026-06-26', '2026-06-27', '2026-06-28',
    '2026-07-06', '2026-07-07', '2026-07-08', '2026-07-09', '2026-07-10', '2026-07-11', '2026-07-12', '2026-07-28', '2026-07-29', '2026-07-30', '2026-07-31',
    '2026-08-01', '2026-08-02', '2026-08-07', '2026-08-08', '2026-08-09', '2026-08-10', '2026-08-11', '2026-08-12', '2026-08-21', '2026-08-22', '2026-08-23', '2026-08-24', '2026-08-25', '2026-08-26',
    '2026-09-04', '2026-09-05', '2026-09-06', '2026-09-07', '2026-09-08', '2026-09-09', '2026-09-18', '2026-09-19', '2026-09-20', '2026-09-25', '2026-09-26', '2026-09-27', '2026-09-29', '2026-09-30',
    '2026-10-06', '2026-10-07',
)  # fmt: skip

# Fair: open Wednesday-Sunday. Racing: Thursday-Sunday plus the listed extra days.
COUNTY_FAIR: tuple[Range, ...] = (("2025-06-11", "2025-07-06"), ("2026-06-10", "2026-07-05"))
FAIR_WEEKDAYS = (2, 3, 4, 5, 6)
RACING_MEETS: tuple[Range, ...] = (("2025-07-18", "2025-09-07"), ("2026-07-17", "2026-09-07"))
RACING_WEEKDAYS = (3, 4, 5, 6)
RACING_EXTRA_DAYS: tuple[str, ...] = ("2025-09-01", "2026-09-07")  # Labor Day cards
RACING_DARK_DAYS: tuple[str, ...] = ("2025-09-04",)  # no Thursday racing on closing week

COMIC_CON: tuple[Range, ...] = (("2025-07-23", "2025-07-27"), ("2026-07-22", "2026-07-26"))


def _in_ranges(days: np.ndarray, ranges: tuple[Range, ...]) -> np.ndarray:
    flags = np.zeros(len(days), dtype=bool)
    for start, end in ranges:
        flags |= (days >= np.datetime64(start, "D")) & (days <= np.datetime64(end, "D"))
    return flags


def _in_days(days: np.ndarray, listed: tuple[str, ...]) -> np.ndarray:
    return np.isin(days, np.array(listed, dtype="datetime64[D]"))


def build_local_calendar(day_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return the boolean local flags for a tz-naive local day index."""
    idx = pd.DatetimeIndex(day_index)
    if idx.tz is not None:
        raise ValueError("day_index must be tz-naive local dates")
    days = idx.normalize().to_numpy(dtype="datetime64[D]")
    weekday = idx.weekday.to_numpy()

    fair = _in_ranges(days, COUNTY_FAIR) & np.isin(weekday, FAIR_WEEKDAYS)
    racing = (
        (_in_ranges(days, RACING_MEETS) & np.isin(weekday, RACING_WEEKDAYS))
        | _in_days(days, RACING_EXTRA_DAYS)
    ) & ~_in_days(days, RACING_DARK_DAYS)

    out = pd.DataFrame(index=day_index)
    out["is_school_break_sdusd"] = _in_ranges(days, SDUSD_BREAKS) | _in_days(days, SDUSD_EXTRA_DAYS)
    out["is_ucsd_in_session"] = _in_ranges(days, UCSD_QUARTERS)
    out["is_usd_in_session"] = _in_ranges(days, USD_SEMESTERS) & ~_in_ranges(days, USD_BREAKS)
    out["is_padres_home_game"] = _in_days(days, PADRES_HOME_GAMES)
    out["is_del_mar_event"] = fair | racing
    out["is_comic_con"] = _in_ranges(days, COMIC_CON)
    return out
