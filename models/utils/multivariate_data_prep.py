"""Build a multivariate wide-format frame from a ClickHouse feature table.

Joint anomaly models expect a wide matrix where each column is one channel
observed at the same timestamps (targets + calendar context).

Schema of the produced frame (returned as a `pandas.DataFrame`):

    index : pd.DatetimeIndex (UTC, freq=1h | 15min)
    columns:
        y__<location_id>      ->  one column per traffic series (= target)
        hour_sin, hour_cos    ->  calendar context shared across targets
        dow_sin,  dow_cos
        is_weekend

Why context columns are shared (one copy, not per-target):
    Calendar features are deterministic functions of `ds`, identical for
    every series. Duplicating them per-target would just inflate the input
    width without giving the model anything new.

Why network context is NOT injected here:
    The joint input already includes all target columns at once, so the
    model has direct access to other stations' history.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import app.core.config as cfg
from app.core.db import ClickHouseDB
from models.utils.constants import (
    DEFAULT_CALENDAR_COLUMNS,
    EXTENDED_CALENDAR_COLUMNS,
    SUPPORTED_FREQS,
)
from models.utils.multivariate_grid import (
    attach_active_columns,
    fill_gap_values_in_span,
    list_active_columns,
)


@dataclass
class MultivariateDatasetMeta:
    locations: list[str]
    target_columns: list[str]
    context_columns: list[str]
    freq: str
    date_from: str | None
    date_to: str | None
    rows: int
    n_targets: int
    n_context: int
    ds_min: str
    ds_max: str
    imputed_rows: int
    active_columns: list[str] = field(default_factory=list)
    inactive_rows: int = 0
    gap_imputed_rows: int = 0
    schema_version: str = ""
    column_roles: dict[str, str] = field(default_factory=dict)


SCHEMA_LOCATION_WIDE_V1 = "location_wide_v1"


def _bucket_expr(freq: str) -> str:
    """ClickHouse expression to truncate `time_utc` to the requested bucket."""
    if freq == "1h":
        return "toStartOfHour(time_utc)"
    if freq == "15min":
        return "toStartOfFifteenMinutes(time_utc)"
    raise ValueError(f"Unsupported freq: {freq!r}; expected one of {SUPPORTED_FREQS}")


def _build_query(
    locations: Sequence[str],
    date_from: str | None,
    date_to: str | None,
    freq: str,
    feature_table: str,
) -> tuple[str, dict]:
    conditions = ["location_id IN {locations:Array(String)}"]
    parameters: dict = {"locations": list(locations)}
    if date_from:
        conditions.append("time_utc >= {date_from:DateTime('UTC')}")
        parameters["date_from"] = date_from
    if date_to:
        conditions.append("time_utc < {date_to:DateTime('UTC')}")
        parameters["date_to"] = date_to

    where = " AND ".join(conditions)
    bucket = _bucket_expr(freq)
    query = f"""
        SELECT
            location_id,
            {bucket} AS ds,
            count() AS y
        FROM {cfg.DB_NAME}.{feature_table}
        WHERE {where}
        GROUP BY location_id, ds
        ORDER BY location_id, ds
    """
    return query, parameters


def _easter_sunday(year: int) -> date:
    """Return Gregorian Easter Sunday date for the given year."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day_val = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day_val)


def _poland_holidays(year: int) -> set[date]:
    """Public holidays in Poland for a given year."""
    easter = _easter_sunday(year)
    return {
        date(year, 1, 1),    # New Year's Day
        date(year, 1, 6),    # Epiphany
        date(year, 5, 1),    # Labour Day
        date(year, 5, 3),    # Constitution Day
        date(year, 8, 15),   # Assumption Day
        date(year, 11, 1),   # All Saints' Day
        date(year, 11, 11),  # Independence Day
        date(year, 12, 25),  # Christmas Day
        date(year, 12, 26),  # Second Day of Christmas
        easter,                          # Easter Sunday
        easter + timedelta(days=1),      # Easter Monday
        easter + timedelta(days=49),     # Pentecost
        easter + timedelta(days=60),     # Corpus Christi
    }


def _build_holiday_set(years: Sequence[int]) -> set[date]:
    holidays: set[date] = set()
    for year in years:
        holidays.update(_poland_holidays(int(year)))
    return holidays


def _days_to_nearest_holiday(index: pd.DatetimeIndex, holiday_set: set[date]) -> np.ndarray:
    if not holiday_set:
        return np.zeros(len(index), dtype=np.float32)
    holiday_arr = np.array(sorted(holiday_set), dtype="datetime64[D]")
    days = index.tz_localize(None).normalize().to_numpy(dtype="datetime64[D]")
    dist = np.array([
        int(np.min(np.abs((holiday_arr - d).astype(int))))
        for d in days
    ], dtype=np.float32)
    return np.clip(dist, 0, 7)


def _build_day_calendar(day_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build day-level calendar table with holiday and bridge-day flags."""
    years = sorted(set(day_index.year.tolist()))
    holiday_set = _build_holiday_set(years)
    holiday_idx = pd.DatetimeIndex(sorted(pd.to_datetime(list(holiday_set))))

    cal = pd.DataFrame(index=day_index)
    cal["weekday"] = cal.index.weekday
    cal["is_weekend"] = cal["weekday"] >= 5
    cal["is_holiday_pl"] = cal.index.isin(holiday_idx)
    cal["is_non_working"] = cal["is_weekend"] | cal["is_holiday_pl"]

    cal["is_pre_holiday"] = cal["is_holiday_pl"].shift(-1, fill_value=False)
    cal["is_post_holiday"] = cal["is_holiday_pl"].shift(1, fill_value=False)

    cal["is_bridge_day"] = (
        (~cal["is_non_working"])
        & cal["is_non_working"].shift(1, fill_value=False)
        & cal["is_non_working"].shift(-1, fill_value=False)
    )

    extended_non_working = cal["is_non_working"] | cal["is_bridge_day"]
    change_points = (~extended_non_working).cumsum()
    run_len = extended_non_working.groupby(change_points).transform("sum")
    cal["is_long_weekend_period"] = extended_non_working & (run_len >= 3)
    
    cal["is_pre_long_weekend"] = (~extended_non_working) & cal["is_long_weekend_period"].shift(-1, fill_value=False)

    # Polish Shopping Sundays 2024
    SHOPPING_SUNDAYS_2024 = {
        date(2024, 1, 28), date(2024, 3, 24), date(2024, 4, 28),
        date(2024, 6, 30), date(2024, 8, 25), date(2024, 12, 15),
        date(2024, 12, 22)
    }
    cal["is_shopping_sunday"] = cal.index.isin(pd.DatetimeIndex(list(SHOPPING_SUNDAYS_2024)))
    
    # School Holidays (Silesia 2024)
    cal["is_ferie_zimowe"] = (cal.index >= "2024-01-29") & (cal.index <= "2024-02-11")
    cal["is_wakacje_letnie"] = (cal.index >= "2024-06-22") & (cal.index <= "2024-08-31")
    cal["is_school_holiday"] = cal["is_ferie_zimowe"] | cal["is_wakacje_letnie"]

    return cal


def _calendar_columns(index: pd.DatetimeIndex, columns: Sequence[str]) -> pd.DataFrame:
    """Vectorised calendar features. Indexed by the supplied DatetimeIndex.

    Computed in Europe/Warsaw local time to match human traffic behaviour
    (which is what the model needs to learn around). `is_weekend` uses
    Mon=0..Sun=6 → weekend = Sat|Sun.
    """
    local = index.tz_convert("Europe/Warsaw")
    out = pd.DataFrame(index=index)
    requested = set(columns)

    if {"hour_sin", "hour_cos"} & requested:
        angle = 2.0 * math.pi * (local.hour + local.minute / 60.0) / 24.0
        if "hour_sin" in requested:
            out["hour_sin"] = np.sin(angle).astype(np.float32)
        if "hour_cos" in requested:
            out["hour_cos"] = np.cos(angle).astype(np.float32)

    if {"dow_sin", "dow_cos", "is_weekend"} & requested:
        dow = local.dayofweek.to_numpy()
        if {"dow_sin", "dow_cos"} & requested:
            angle = 2.0 * math.pi * dow / 7.0
            if "dow_sin" in requested:
                out["dow_sin"] = np.sin(angle).astype(np.float32)
            if "dow_cos" in requested:
                out["dow_cos"] = np.cos(angle).astype(np.float32)
        if "is_weekend" in requested:
            out["is_weekend"] = (dow >= 5).astype(np.float32)

    # Month cyclical encoding
    if {"month_sin", "month_cos"} & requested:
        month = local.month.to_numpy()
        angle = 2.0 * math.pi * (month - 1) / 12.0
        if "month_sin" in requested:
            out["month_sin"] = np.sin(angle).astype(np.float32)
        if "month_cos" in requested:
            out["month_cos"] = np.cos(angle).astype(np.float32)

    # Week-of-year cyclical encoding
    if {"weekofyear_sin", "weekofyear_cos"} & requested:
        week = local.isocalendar().week.to_numpy(dtype=np.int16)
        angle = 2.0 * math.pi * (week - 1) / 53.0
        if "weekofyear_sin" in requested:
            out["weekofyear_sin"] = np.sin(angle).astype(np.float32)
        if "weekofyear_cos" in requested:
            out["weekofyear_cos"] = np.cos(angle).astype(np.float32)
            
    # Day-of-month encoding
    needs_dom = {"dom_sin", "dom_cos", "is_month_start", "is_month_end"} & requested
    if needs_dom:
        if "dom_sin" in requested:
            out["dom_sin"] = np.sin(2.0 * math.pi * local.day / 31.0).astype(np.float32)
        if "dom_cos" in requested:
            out["dom_cos"] = np.cos(2.0 * math.pi * local.day / 31.0).astype(np.float32)
        if "is_month_start" in requested:
            out["is_month_start"] = (local.day <= 3).astype(np.float32)
        if "is_month_end" in requested:
            out["is_month_end"] = (local.day >= 28).astype(np.float32)

    # Distance to nearest holiday
    if "days_to_nearest_holiday" in requested:
        years = sorted(set(index.year.tolist()))
        holiday_set = _build_holiday_set(years)
        out["days_to_nearest_holiday"] = _days_to_nearest_holiday(index, holiday_set)

    # Holiday-related features (computed from day-level calendar)
    holiday_feats = {
        "is_holiday_pl", "is_pre_holiday", "is_post_holiday",
        "is_bridge_day", "is_long_weekend_period", "is_pre_long_weekend",
        "is_shopping_sunday", "is_school_holiday", "is_ferie_zimowe", "is_wakacje_letnie"
    }
    if holiday_feats & requested:
        local_days = local.tz_localize(None).normalize()
        day_start = pd.to_datetime(local_days.min())
        day_end = pd.to_datetime(local_days.max())
        day_index = pd.date_range(day_start, day_end, freq="D")
        cal = _build_day_calendar(day_index)
        cal_for_rows = cal.reindex(local_days)

        for feat in holiday_feats & requested:
            out[feat] = cal_for_rows[feat].to_numpy(dtype=np.float32)

    return out[[c for c in columns if c in out.columns]]


def _attach_weather(context: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
    from pathlib import Path
    weather_path = Path("models/data/weather/weather_2024.parquet")
    if weather_path.exists():
        weather_df = pd.read_parquet(weather_path)
        if "ds" in weather_df.columns:
            weather_df["ds"] = pd.to_datetime(weather_df["ds"], utc=True)
            weather_df = weather_df.set_index("ds")
        else:
            weather_df.index = pd.to_datetime(weather_df.index, utc=True)
        # Ensure we only pick numeric columns
        weather_df = weather_df.reindex(full_index)
        weather_df = weather_df.interpolate(method="time").bfill().ffill()
        all_nan_cols = weather_df.columns[weather_df.isna().all()].tolist()
        if all_nan_cols:
            weather_df = weather_df.drop(columns=all_nan_cols)
            print(
                "[data_prep] Dropped weather columns with no overlap for this date range: "
                f"{all_nan_cols}"
            )
        if not weather_df.empty:
            weather_df = weather_df.fillna(0.0).astype(np.float32)
            context = pd.concat([context, weather_df], axis=1)
            print(f"[data_prep] Attached weather features: {list(weather_df.columns)}")
    return context


def build_multivariate_frame_from_long(
    long_df: pd.DataFrame,
    locations: Sequence[str],
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    freq: str = "1h",
    calendar_columns: Sequence[str] = DEFAULT_CALENDAR_COLUMNS,
) -> tuple[pd.DataFrame, MultivariateDatasetMeta]:
    """Pivot a long counts frame (location_id, ds, y) into the wide traffic matrix."""
    if not locations:
        raise ValueError("locations list must not be empty")
    if freq not in SUPPORTED_FREQS:
        raise ValueError(f"Unsupported freq: {freq!r}; expected one of {SUPPORTED_FREQS}")

    work = long_df.copy()
    if work.empty:
        raise RuntimeError(
            f"No rows in long counts frame for locations={list(locations)} "
            f"(range={date_from}..{date_to}, freq={freq})."
        )
    work["ds"] = pd.to_datetime(work["ds"], utc=True)
    work = work[work["location_id"].isin(locations)]
    if work.empty:
        raise RuntimeError(
            f"No rows for requested locations={list(locations)} after filtering."
        )

    pandas_freq = "h" if freq == "1h" else "15min"
    wide_raw = (
        work.pivot_table(index="ds", columns="location_id", values="y", aggfunc="sum")
        .sort_index()
    )
    for loc in locations:
        if loc not in wide_raw.columns:
            wide_raw[loc] = 0.0
    wide_raw = wide_raw[list(locations)]

    full_index = pd.date_range(
        start=wide_raw.index.min(),
        end=wide_raw.index.max(),
        freq=pandas_freq,
        tz="UTC",
    )
    target_columns = [f"y__{loc}" for loc in locations]
    wide_y = wide_raw.reindex(full_index)
    wide_y_prefixed = wide_y.copy()
    wide_y_prefixed.columns = target_columns
    active_df, inactive_rows, gap_imputed_rows = attach_active_columns(
        wide_y_prefixed, work, full_index
    )
    wide_filled = fill_gap_values_in_span(wide_y_prefixed, work, full_index)
    wide = wide_filled.fillna(0.0).astype(np.float32)
    wide.columns = target_columns

    context = _calendar_columns(full_index, calendar_columns)
    context = _attach_weather(context, full_index)
    frame = pd.concat([wide, active_df, context], axis=1)
    frame.index.name = "ds"
    active_columns = list_active_columns(target_columns)
    imputed_rows = inactive_rows + gap_imputed_rows

    meta = MultivariateDatasetMeta(
        locations=list(locations),
        target_columns=target_columns,
        context_columns=list(context.columns),
        freq=freq,
        date_from=date_from,
        date_to=date_to,
        rows=int(len(frame)),
        n_targets=len(target_columns),
        n_context=int(context.shape[1]),
        ds_min=str(frame.index.min()),
        ds_max=str(frame.index.max()),
        imputed_rows=imputed_rows,
        active_columns=active_columns,
        inactive_rows=inactive_rows,
        gap_imputed_rows=gap_imputed_rows,
    )
    return frame, meta


def build_multivariate_frame(
    locations: Sequence[str],
    date_from: str | None = None,
    date_to: str | None = None,
    freq: str = "1h",
    calendar_columns: Sequence[str] = DEFAULT_CALENDAR_COLUMNS,
    *,
    feature_table: str,
) -> tuple[pd.DataFrame, MultivariateDatasetMeta]:
    """Pull the raw long frame, pivot to wide, attach calendar columns.

    The resulting wide frame is then ready for windowed training. Internal
    gaps are masked inactive (not trained/scored); gap buckets may be
    forward-filled for the forward pass. Calendar cells outside a location's
    span are zero-filled. Counts are in ``meta.inactive_rows`` /
    ``meta.gap_imputed_rows``.
    """
    if not locations:
        raise ValueError("locations list must not be empty")
    if freq not in SUPPORTED_FREQS:
        raise ValueError(f"Unsupported freq: {freq!r}; expected one of {SUPPORTED_FREQS}")

    query, parameters = _build_query(locations, date_from, date_to, freq, feature_table)
    with ClickHouseDB.from_config() as db:
        result = db.query_raw(query, parameters=parameters)
    long_df = pd.DataFrame(result.result_rows, columns=result.column_names)
    if long_df.empty:
        raise RuntimeError(
            f"No rows returned for locations={list(locations)} "
            f"(range={date_from}..{date_to}, freq={freq}). "
            "Check the feature table and location ids."
        )

    long_df["ds"] = pd.to_datetime(long_df["ds"], utc=True)
    return build_multivariate_frame_from_long(
        long_df,
        locations,
        date_from=date_from,
        date_to=date_to,
        freq=freq,
        calendar_columns=calendar_columns,
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_dataset(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reset index so the saved parquet retains `ds` as a regular column
    # (parquet does not preserve a tz-aware index across all reader versions).
    out = frame.reset_index().rename(columns={"index": "ds"})
    out.to_parquet(path, index=False)
    return path


def save_meta(meta: MultivariateDatasetMeta, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)
    return path


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(Path(path))
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    df = df.set_index("ds").sort_index()
    return df


def load_meta(path: Path) -> MultivariateDatasetMeta:
    with open(Path(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    for key, default in (
        ("active_columns", []),
        ("inactive_rows", 0),
        ("gap_imputed_rows", 0),
        ("schema_version", ""),
        ("column_roles", {}),
    ):
        data.setdefault(key, default)
    known = {f.name for f in fields(MultivariateDatasetMeta)}
    return MultivariateDatasetMeta(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_robust_scaler(
    train_values: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Per-column robust scaler: (x - median) / (1.4826 * MAD).

    Robust scaling shrugs off the same heavy tails (incidents, holidays)
    that we want the anomaly score to flag later, so it's a better fit
    than mean/std normalisation here.

    Args:
        train_values: 2-D array (T x F). Fit only on train slice to avoid
            test-set leakage.
    """
    if train_values.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape={train_values.shape}")
    median = np.median(train_values, axis=0)
    mad = np.median(np.abs(train_values - median), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale <= eps, np.std(train_values, axis=0), scale)
    scale = np.where(scale <= eps, 1.0, scale)
    return {"median": median.astype(np.float32), "scale": scale.astype(np.float32)}


def apply_scaler(values: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    return ((values - scaler["median"]) / scaler["scale"]).astype(np.float32)


def save_scaler(
    scaler: dict[str, np.ndarray],
    columns: Sequence[str],
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "columns": list(columns),
        "median": [float(x) for x in scaler["median"]],
        "scale": [float(x) for x in scaler["scale"]],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_scaler(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    with open(Path(path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    return (
        {
            "median": np.asarray(payload["median"], dtype=np.float32),
            "scale": np.asarray(payload["scale"], dtype=np.float32),
        },
        list(payload["columns"]),
    )


def _build_class_query(
    locations: Sequence[str],
    date_from: str | None,
    date_to: str | None,
    freq: str,
    feature_table: str,
) -> tuple[str, dict]:
    conditions = ["location_id IN {locations:Array(String)}"]
    parameters: dict = {"locations": list(locations)}

    if date_from:
        conditions.append("time_utc >= {date_from:DateTime('UTC')}")
        parameters["date_from"] = date_from
    if date_to:
        conditions.append("time_utc < {date_to:DateTime('UTC')}")
        parameters["date_to"] = date_to

    where = " AND ".join(conditions)
    bucket = _bucket_expr(freq)
    query = f"""
        SELECT
            location_id,
            location_lane,
            vehicle_class,
            {bucket} AS ds,
            count() AS y
        FROM {cfg.DB_NAME}.{feature_table}
        WHERE {where}
        GROUP BY location_id, location_lane, vehicle_class, ds
        ORDER BY location_id, location_lane, vehicle_class, ds
    """
    return query, parameters


def build_multivariate_class_frame(
    locations: Sequence[str],
    date_from: str | None = None,
    date_to: str | None = None,
    freq: str = "1h",
    calendar_columns: Sequence[str] = DEFAULT_CALENDAR_COLUMNS,
    min_nonzero_hours_per_series: int = 0,
    *,
    feature_table: str,
) -> tuple[pd.DataFrame, MultivariateDatasetMeta]:
    """Phase-2: hourly counts per (location, vehicle_class).

    Pivots the long format into a wide matrix (one column per unique series).
    Applies a sparsity gate: any series with fewer than
    `min_nonzero_hours_per_series` non-zero periods is dropped.
    """
    if not locations:
        raise ValueError("locations list must not be empty")
    if freq not in SUPPORTED_FREQS:
        raise ValueError(f"Unsupported freq: {freq!r}; expected one of {SUPPORTED_FREQS}")

    query, parameters = _build_class_query(locations, date_from, date_to, freq, feature_table)

    with ClickHouseDB.from_config() as db:
        result = db.query_raw(query, parameters=parameters)

    long_df = pd.DataFrame(result.result_rows, columns=result.column_names)
    if long_df.empty:
        raise RuntimeError(
            f"No rows returned for locations={list(locations)} "
            f"(range={date_from}..{date_to}, freq={freq})."
        )

    long_df["ds"] = pd.to_datetime(long_df["ds"], utc=True)
    long_df["unique_id"] = (
        long_df["location_id"].astype(str)
        + "__cls"
        + long_df["vehicle_class"].astype(str)
    )

    if min_nonzero_hours_per_series > 0:
        keep_ids = (
            long_df[long_df["y"] > 0]
            .groupby("unique_id")["y"]
            .count()
            .loc[lambda s: s >= min_nonzero_hours_per_series]
            .index
        )
        long_df = long_df[long_df["unique_id"].isin(keep_ids)].reset_index(drop=True)
        if long_df.empty:
            raise RuntimeError(
                "All class-level series filtered out by "
                f"min_nonzero_hours_per_series={min_nonzero_hours_per_series}."
            )

    pandas_freq = "h" if freq == "1h" else "15min"

    wide_raw = (
        long_df.pivot(index="ds", columns="unique_id", values="y")
        .sort_index()
    )
    full_index = pd.date_range(
        start=wide_raw.index.min(),
        end=wide_raw.index.max(),
        freq=pandas_freq,
        tz="UTC",
    )
    target_columns = [f"y__{uid}" for uid in wide_raw.columns]
    wide_y = wide_raw.reindex(full_index)
    long_for_active = long_df[["unique_id", "ds"]].rename(
        columns={"unique_id": "location_id"}
    )
    wide_y_prefixed = wide_y.copy()
    wide_y_prefixed.columns = target_columns
    active_df, inactive_rows, gap_imputed_rows = attach_active_columns(
        wide_y_prefixed, long_for_active, full_index
    )
    wide_filled = fill_gap_values_in_span(wide_y_prefixed, long_for_active, full_index)
    wide = wide_filled.fillna(0.0).astype(np.float32)
    wide.columns = target_columns

    context = _calendar_columns(full_index, calendar_columns)
    context = _attach_weather(context, full_index)
    frame = pd.concat([wide, active_df, context], axis=1)
    frame.index.name = "ds"
    active_columns = list_active_columns(target_columns)
    imputed_rows = inactive_rows + gap_imputed_rows

    meta = MultivariateDatasetMeta(
        locations=list(locations),
        target_columns=target_columns,
        context_columns=list(context.columns),
        freq=freq,
        date_from=date_from,
        date_to=date_to,
        rows=int(len(frame)),
        n_targets=len(target_columns),
        n_context=int(context.shape[1]),
        ds_min=str(frame.index.min()),
        ds_max=str(frame.index.max()),
        imputed_rows=imputed_rows,
        active_columns=active_columns,
        inactive_rows=inactive_rows,
        gap_imputed_rows=gap_imputed_rows,
    )

    return frame, meta


def parse_target_column(col: str) -> tuple[str, str | None, int | None]:
    """Parse y__<location>__total or y__<location>__cls<N> into parts."""
    if not col.startswith("y__"):
        return col, None, None
    body = col[3:]
    if body.endswith("__total"):
        return body[: -len("__total")], "total", None
    marker = "__cls"
    if marker in body:
        loc, _, cls_s = body.partition(marker)
        try:
            return loc, "class", int(cls_s)
        except ValueError:
            return loc, "class", None
    return body, None, None


def target_column_role(col: str, column_roles: dict[str, str]) -> str | None:
    if col in column_roles:
        return column_roles[col]
    _, role, _ = parse_target_column(col)
    return role


def unique_id_from_target_column(col: str) -> str:
    """Score row id: location__total or location__cls7 (without y__ prefix)."""
    if col.startswith("y__"):
        return col[3:]
    return col


def _build_location_wide_query(
    locations: Sequence[str],
    date_from: str | None,
    date_to: str | None,
    freq: str,
    feature_table: str,
) -> tuple[str, dict]:
    conditions = ["location_id IN {locations:Array(String)}"]
    parameters: dict = {"locations": list(locations)}

    if date_from:
        conditions.append("time_utc >= {date_from:DateTime('UTC')}")
        parameters["date_from"] = date_from
    if date_to:
        conditions.append("time_utc < {date_to:DateTime('UTC')}")
        parameters["date_to"] = date_to

    where = " AND ".join(conditions)
    bucket = _bucket_expr(freq)
    query = f"""
        SELECT
            location_id,
            vehicle_class,
            {bucket} AS ds,
            count() AS y
        FROM {cfg.DB_NAME}.{feature_table}
        WHERE {where}
        GROUP BY location_id, vehicle_class, ds
        ORDER BY location_id, vehicle_class, ds
    """
    return query, parameters


def build_multivariate_location_wide_frame(
    locations: Sequence[str],
    date_from: str | None = None,
    date_to: str | None = None,
    freq: str = "15min",
    calendar_columns: Sequence[str] = DEFAULT_CALENDAR_COLUMNS,
    min_nonzero_periods: int = 0,
    *,
    feature_table: str,
) -> tuple[pd.DataFrame, MultivariateDatasetMeta]:
    """Wide matrix per location: total + per-class counts at each timestamp."""
    if not locations:
        raise ValueError("locations list must not be empty")
    if freq not in SUPPORTED_FREQS:
        raise ValueError(f"Unsupported freq: {freq!r}; expected one of {SUPPORTED_FREQS}")

    query, parameters = _build_location_wide_query(
        locations, date_from, date_to, freq, feature_table
    )
    with ClickHouseDB.from_config() as db:
        result = db.query_raw(query, parameters=parameters)

    long_df = pd.DataFrame(result.result_rows, columns=result.column_names)
    if long_df.empty:
        raise RuntimeError(
            f"No rows returned for locations={list(locations)} "
            f"(range={date_from}..{date_to}, freq={freq})."
        )

    long_df["ds"] = pd.to_datetime(long_df["ds"], utc=True)
    long_df["vehicle_class"] = long_df["vehicle_class"].astype(int)
    long_df["col_key"] = (
        long_df["location_id"].astype(str)
        + "__cls"
        + long_df["vehicle_class"].astype(str)
    )

    if min_nonzero_periods > 0:
        keep_cls = (
            long_df[long_df["y"] > 0]
            .groupby("col_key")["y"]
            .count()
            .loc[lambda s: s >= min_nonzero_periods]
            .index
        )
        long_df = long_df[long_df["col_key"].isin(keep_cls)].reset_index(drop=True)
        if long_df.empty:
            raise RuntimeError(
                f"All class columns filtered out by min_nonzero_periods={min_nonzero_periods}."
            )

    pandas_freq = "h" if freq == "1h" else "15min"
    wide_cls = long_df.pivot(index="ds", columns="col_key", values="y").sort_index()
    full_index = pd.date_range(
        start=wide_cls.index.min(),
        end=wide_cls.index.max(),
        freq=pandas_freq,
        tz="UTC",
    )
    imputed_mask = wide_cls.reindex(full_index).isna()
    imputed_rows = int(imputed_mask.values.sum())
    wide_cls = wide_cls.reindex(full_index).fillna(0.0).astype(np.float32)

    locs_in_data = sorted({k.rsplit("__cls", 1)[0] for k in wide_cls.columns})
    target_frames: list[pd.DataFrame] = []
    column_roles: dict[str, str] = {}
    imputed_context: dict[str, pd.Series] = {}

    for loc in locs_in_data:
        prefix = f"{loc}__cls"
        cls_cols = [c for c in wide_cls.columns if c.startswith(prefix)]
        if not cls_cols:
            continue
        cls_block = wide_cls[cls_cols]
        total = cls_block.sum(axis=1).astype(np.float32)
        total_col = f"y__{loc}__total"
        column_roles[total_col] = "total"

        renamed = {c: f"y__{loc}__cls{c.split('__cls', 1)[1]}" for c in cls_cols}
        loc_frame = cls_block.rename(columns=renamed)
        for c in renamed.values():
            column_roles[c] = "class"
        loc_frame[total_col] = total
        cols_order = [total_col] + [renamed[c] for c in cls_cols]
        loc_frame = loc_frame[cols_order]

        n_cls = len(cls_cols) + 1
        imputed_frac = (
            imputed_mask[cls_cols].sum(axis=1).astype(np.float32) / float(n_cls)
        ).clip(0.0, 1.0)
        imputed_context[f"is_imputed__{loc}"] = imputed_frac.astype(np.float32)

        target_frames.append(loc_frame)

    if not target_frames:
        raise RuntimeError("No location-wide target columns produced after pivot.")

    wide = pd.concat(target_frames, axis=1)
    cal = _calendar_columns(full_index, calendar_columns)
    imputed_df = pd.DataFrame(imputed_context, index=full_index)
    context = pd.concat([cal, imputed_df], axis=1)
    context = _attach_weather(context, full_index)
    frame = pd.concat([wide, context], axis=1)
    frame.index.name = "ds"

    context_columns = [c for c in frame.columns if c not in wide.columns]
    target_columns = list(wide.columns)

    meta = MultivariateDatasetMeta(
        locations=list(locations),
        target_columns=target_columns,
        context_columns=context_columns,
        freq=freq,
        date_from=date_from,
        date_to=date_to,
        rows=int(len(frame)),
        n_targets=len(target_columns),
        n_context=len(context_columns),
        ds_min=str(frame.index.min()),
        ds_max=str(frame.index.max()),
        imputed_rows=imputed_rows,
        schema_version=SCHEMA_LOCATION_WIDE_V1,
        column_roles=column_roles,
    )
    return frame, meta