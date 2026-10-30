"""Shared runner helpers for anomaly-detection CLI orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd


def resolve_threshold_profile(
    threshold_profile: str | None,
    threshold_method: str,
    threshold: float,
    quantile: float,
    *,
    preset_resolver: Callable[[str], dict[str, float | str]],
) -> tuple[str, float, float]:
    if not threshold_profile:
        return threshold_method, threshold, quantile
    preset = preset_resolver(threshold_profile)
    method = str(preset.get("method", threshold_method))
    th = float(preset.get("threshold", threshold))
    q = float(preset.get("quantile", quantile))
    return method, th, q


def parse_eval_date(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True)


def freq_timedelta(freq: str) -> pd.Timedelta:
    return pd.Timedelta("15min") if freq == "15min" else pd.Timedelta("1h")


def calendar_columns_from_meta(meta) -> list[str]:
    return [c for c in meta.context_columns if not c.startswith("weather__")]


def filter_by_date_range(
    frame: pd.DataFrame,
    annotated_scores: pd.DataFrame,
    *,
    date_from: str | None,
    date_to: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not date_from and not date_to:
        return frame, annotated_scores

    lo = parse_eval_date(date_from) if date_from else None
    hi = parse_eval_date(date_to) if date_to else None

    idx = pd.to_datetime(frame.index, utc=True)
    frame_mask = pd.Series(True, index=frame.index)
    if lo is not None:
        frame_mask &= idx >= lo
    if hi is not None:
        frame_mask &= idx < hi
    frame_out = frame.loc[frame_mask]

    ds = pd.to_datetime(annotated_scores["ds"], utc=True)
    score_mask = pd.Series(True, index=annotated_scores.index)
    if lo is not None:
        score_mask &= ds >= lo
    if hi is not None:
        score_mask &= ds < hi
    scores_out = annotated_scores.loc[score_mask]
    return frame_out, scores_out


def extend_frame_for_eval_date_range(
    frame: pd.DataFrame,
    meta,
    *,
    date_from: str | None,
    date_to: str | None,
    build_multivariate_frame: Callable[..., tuple[pd.DataFrame, object]],
    log_prefix: str,
) -> tuple[pd.DataFrame, int]:
    original_len = len(frame)
    if not date_from and not date_to:
        return frame, original_len

    idx = pd.to_datetime(frame.index, utc=True)
    ds_min, ds_max = idx.min(), idx.max()
    lo = parse_eval_date(date_from) if date_from else ds_min
    hi = parse_eval_date(date_to) if date_to else ds_max + freq_timedelta(meta.freq)
    step = freq_timedelta(meta.freq)

    needs_before = lo < ds_min
    needs_after = hi > ds_max + step
    if not needs_before and not needs_after:
        return frame, original_len

    if needs_before and needs_after:
        fetch_from = lo.strftime("%Y-%m-%d %H:%M:%S")
        fetch_to = hi.strftime("%Y-%m-%d %H:%M:%S")
    elif needs_after:
        fetch_from = (ds_max + step).strftime("%Y-%m-%d %H:%M:%S")
        fetch_to = hi.strftime("%Y-%m-%d %H:%M:%S")
    else:
        fetch_from = lo.strftime("%Y-%m-%d %H:%M:%S")
        fetch_to = ds_min.strftime("%Y-%m-%d %H:%M:%S")

    print(
        f"[{log_prefix}:eval] supplemental DB fetch {fetch_from} .. {fetch_to} "
        f"(prepared span {ds_min} .. {ds_max})"
    )
    ext_frame, _ = build_multivariate_frame(
        locations=meta.locations,
        date_from=fetch_from,
        date_to=fetch_to,
        freq=meta.freq,
        calendar_columns=calendar_columns_from_meta(meta),
    )
    ext_frame = ext_frame.reindex(columns=frame.columns, fill_value=0.0)
    merged = pd.concat([frame, ext_frame])
    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    if len(merged) == original_len:
        raise ValueError(
            f"No supplemental rows returned for eval date range {date_from!r}..{date_to!r}. "
            f"Prepared dataset span is {ds_min} .. {ds_max}. "
            "Check ClickHouse data and location ids."
        )
    print(f"[{log_prefix}:eval] extended dataset rows {original_len} -> {len(merged)}")
    return merged, original_len


def eval_date_range_error(
    *,
    date_from: str | None,
    date_to: str | None,
    meta,
    annotated: pd.DataFrame,
) -> str:
    ds = pd.to_datetime(annotated["ds"], utc=True)
    scores_min = ds.min() if not annotated.empty else None
    scores_max = ds.max() if not annotated.empty else None
    return (
        f"No score rows in eval date range {date_from!r}..{date_to!r} "
        f"(--date-to is exclusive). "
        f"Prepared dataset span: {meta.ds_min} .. {meta.ds_max}. "
        f"Scored span: {scores_min} .. {scores_max}. "
        "Re-run prepare with a wider --date-to, or use "
        "--date-to '2025-01-01 00:00:00' to include all of December."
    )


def threshold_preset(name: str, presets: dict[str, dict[str, float | str]]) -> dict[str, float | str]:
    if name not in presets:
        raise ValueError(f"Unknown threshold preset: {name!r}")
    return dict(presets[name])
