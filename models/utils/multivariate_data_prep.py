"""Build one wide frame for the single station in traffic_5min.

It is multivariate: several series share one timestamp index.
The series are measurements, not stations and not vehicle classes:

    y__flow        vehicles per bucket
    y__speed       average speed, mph
    y__occupancy   fraction of the bucket when the detector was occupied
    active__*      1 only when observed_percent is 100

Occupancy is not a count of vehicles. 0.0128 means the loop was covered
for 1.28% of that 5-minute step.

15-minute buckets sum flow and average speed and occupancy.
The ClickHouse table name is FEATURE_TABLE in models/TimesNet/config.py.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path

import app.core.config as cfg
import holidays
import numpy as np
import pandas as pd
from app.core.db import ClickHouseDB

from models.utils.constants import (
    DEFAULT_CALENDAR_COLUMNS,
    OBSERVED_ACTIVE_MIN_PERCENT,
    SUPPORTED_FREQS,
)
from models.utils.local_calendar import LOCAL_CALENDAR_COLUMNS, build_local_calendar
from models.utils.multivariate_grid import (
    attach_active_columns,
    fill_gap_values_in_span,
    list_active_columns,
)


@dataclass
class MultivariateDatasetMeta:
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


_CHANNEL_SOURCE: tuple[tuple[str, str], ...] = (
    ("flow", "flow_veh_5min"),
    ("speed", "speed_mph"),
    ("occupancy", "occupancy_fraction"),
)
_LOCAL_TZ = "America/Los_Angeles"


def _bucket_expr(freq: str) -> str:
    """ClickHouse expression that buckets `ts` (California local time)."""
    if freq == "15min":
        return "toStartOfFifteenMinutes(ts)"
    if freq == "5min":
        return "ts"
    raise ValueError(f"Unsupported freq: {freq!r}; expected one of {SUPPORTED_FREQS}")


def _build_query(
    date_from: str | None,
    date_to: str | None,
    freq: str,
    table: str,
) -> tuple[str, dict]:
    """Return SQL and the bind values for `{name:Type}` placeholders.

    `conditions` are the optional WHERE fragments. `parameters` is what
    clickhouse_connect substitutes into those placeholders. An empty date
    range omits WHERE entirely.
    """
    conditions: list[str] = []
    parameters: dict = {}
    if date_from:
        conditions.append("ts >= {date_from:DateTime64(0, 'America/Los_Angeles')}")
        parameters["date_from"] = date_from
    if date_to:
        conditions.append("ts < {date_to:DateTime64(0, 'America/Los_Angeles')}")
        parameters["date_to"] = date_to

    where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    bucket = _bucket_expr(freq)
    query = f"""
        SELECT
            {bucket} AS ds,
            sum(flow_veh_5min) AS flow_veh_5min,
            avg(speed_mph) AS speed_mph,
            avg(occupancy_fraction) AS occupancy_fraction,
            min(observed_percent) AS observed_percent
        FROM {cfg.DB_NAME}.{table} FINAL
        {where_sql}
        GROUP BY ds
        ORDER BY ds
    """
    return query, parameters


def _california_holidays(year: int) -> set[date]:
    """Holidays in California for a given year."""
    ca = holidays.country_holidays(
        country="US",
        subdiv="CA",
        years=year,
        observed=True,
        language="en_US",
    )

    return {day for day in ca if day.year == year}


def _build_holiday_set(years: Sequence[int]) -> set[date]:
    found: set[date] = set()
    for year in years:
        found.update(_california_holidays(int(year)))
    return found


def _days_to_nearest_holiday(index: pd.DatetimeIndex, holiday_set: set[date]) -> np.ndarray:
    if not holiday_set:
        return np.zeros(len(index), dtype=np.float32)
    holiday_arr = np.array(sorted(holiday_set), dtype="datetime64[D]")
    local = index.tz_convert(_LOCAL_TZ).tz_localize(None) if index.tz is not None else index
    days = local.normalize().to_numpy(dtype="datetime64[D]")
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
    cal["is_holiday_ca"] = cal.index.isin(holiday_idx)
    cal["is_non_working"] = cal["is_weekend"] | cal["is_holiday_ca"]

    cal["is_pre_holiday"] = cal["is_holiday_ca"].shift(-1, fill_value=False)
    cal["is_post_holiday"] = cal["is_holiday_ca"].shift(1, fill_value=False)

    cal["is_bridge_day"] = (
        (~cal["is_non_working"])
        & cal["is_non_working"].shift(1, fill_value=False)
        & cal["is_non_working"].shift(-1, fill_value=False)
    )

    extended_non_working = cal["is_non_working"] | cal["is_bridge_day"]
    change_points = (~extended_non_working).cumsum()
    run_len = extended_non_working.groupby(change_points).transform("sum")
    cal["is_long_weekend"] = extended_non_working & (run_len >= 3)
    cal["is_pre_long_weekend"] = (~extended_non_working) & cal["is_long_weekend"].shift(
        -1, fill_value=False
    )

    local = build_local_calendar(day_index)
    for column in LOCAL_CALENDAR_COLUMNS:
        cal[column] = local[column]

    return cal


def _calendar_columns(index: pd.DatetimeIndex, columns: Sequence[str]) -> pd.DataFrame:
    """Vectorised calendar features. Indexed by the supplied DatetimeIndex.

    Clock features use America/Los_Angeles. `is_weekend` is Sat|Sun.
    """
    local = index.tz_convert(_LOCAL_TZ) if index.tz is not None else index
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
        "is_holiday_ca",
        "is_pre_holiday",
        "is_post_holiday",
        "is_bridge_day",
        "is_long_weekend",
        "is_pre_long_weekend",
        *LOCAL_CALENDAR_COLUMNS,
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


def _as_utc(values: pd.Series) -> pd.Series:
    """Interpret naive timestamps as California local time, then store UTC."""
    parsed = pd.to_datetime(values)
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(_LOCAL_TZ)
    return parsed.dt.tz_convert("UTC")


def build_multivariate_frame_from_long(
    station_df: pd.DataFrame,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    freq: str = "15min",
    calendar_columns: Sequence[str] = DEFAULT_CALENDAR_COLUMNS,
) -> tuple[pd.DataFrame, MultivariateDatasetMeta]:
    """Turn station buckets into y__flow, y__speed, y__occupancy plus a mask."""
    if freq not in SUPPORTED_FREQS:
        raise ValueError(f"Unsupported freq: {freq!r}; expected one of {SUPPORTED_FREQS}")
    if station_df.empty:
        raise RuntimeError(f"No station rows for range={date_from}..{date_to}, freq={freq}.")

    work = station_df.copy()
    work["ds"] = _as_utc(work["ds"])
    work = work.set_index("ds").sort_index()
    active_step = work["observed_percent"].fillna(0) >= OBSERVED_ACTIVE_MIN_PERCENT
    for _, source in _CHANNEL_SOURCE:
        work.loc[~active_step, source] = np.nan

    full_index = pd.date_range(
        start=work.index.min(),
        end=work.index.max(),
        freq=freq,
        tz="UTC",
    )
    channels = [name for name, _ in _CHANNEL_SOURCE]
    target_columns = [f"y__{name}" for name in channels]
    wide_y = pd.DataFrame(
        {f"y__{name}": work[source] for name, source in _CHANNEL_SOURCE},
        index=work.index,
    ).reindex(full_index)
    long_df = pd.concat(
        [pd.DataFrame({"location_id": name, "ds": work.index}) for name in channels],
        ignore_index=True,
    )
    active_df, inactive_rows, gap_imputed_rows = attach_active_columns(
        wide_y, long_df, full_index
    )
    wide = fill_gap_values_in_span(wide_y, long_df, full_index).fillna(0.0).astype(np.float32)

    context = _calendar_columns(full_index, calendar_columns)
    context = _attach_weather(context, full_index)
    frame = pd.concat([wide, active_df, context], axis=1)
    frame.index.name = "ds"
    active_columns = list_active_columns(target_columns)

    meta = MultivariateDatasetMeta(
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
        imputed_rows=inactive_rows + gap_imputed_rows,
        active_columns=active_columns,
        inactive_rows=inactive_rows,
        gap_imputed_rows=gap_imputed_rows,
    )
    return frame, meta


def _source_table() -> str:
    from models.TimesNet.config import FEATURE_TABLE

    return FEATURE_TABLE


def build_multivariate_frame(
    date_from: str | None = None,
    date_to: str | None = None,
    freq: str = "15min",
    calendar_columns: Sequence[str] = DEFAULT_CALENDAR_COLUMNS,
) -> tuple[pd.DataFrame, MultivariateDatasetMeta]:
    """Read the configured station table and build the wide training frame."""
    if freq not in SUPPORTED_FREQS:
        raise ValueError(f"Unsupported freq: {freq!r}; expected one of {SUPPORTED_FREQS}")

    table = _source_table()
    query, parameters = _build_query(date_from, date_to, freq, table)
    with ClickHouseDB.from_config() as db:
        result = db.query_raw(query, parameters=parameters)
    station_df = pd.DataFrame(result.result_rows, columns=result.column_names)
    if station_df.empty:
        raise RuntimeError(f"No rows in {table} for range={date_from}..{date_to}, freq={freq}.")
    return build_multivariate_frame_from_long(
        station_df,
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
    with open(Path(path), encoding="utf-8") as f:
        data = json.load(f)
    for key, default in (
        ("active_columns", []),
        ("inactive_rows", 0),
        ("gap_imputed_rows", 0),
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
    with open(Path(path), encoding="utf-8") as f:
        payload = json.load(f)
    return (
        {
            "median": np.asarray(payload["median"], dtype=np.float32),
            "scale": np.asarray(payload["scale"], dtype=np.float32),
        },
        list(payload["columns"]),
    )
