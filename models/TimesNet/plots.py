"""Plots for TimesNet: score timeline + anomaly overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.utils.multivariate_grid import list_active_columns




def _trim_to_active_span(series: pd.Series, frame: pd.DataFrame, col: str) -> pd.Series:
    """Drop leading/trailing timeline where the location is inactive."""
    active_cols = list_active_columns([col])
    if active_cols and active_cols[0] in frame.columns:
        mask = frame[active_cols[0]].astype(bool)
        if mask.any():
            idx = mask[mask].index
            return series.loc[idx[0] : idx[-1]]
    nz = series[series > 0]
    if not nz.empty:
        return series.loc[nz.index[0] : nz.index[-1]]
    return series


def _mask_inactive_buckets(series: pd.Series, frame: pd.DataFrame, col: str) -> pd.Series:
    """Set inactive/gap timesteps to NaN so plots break lines instead of flat zeros."""
    out = series.astype(float).copy()
    active_cols = list_active_columns([col])
    if active_cols and active_cols[0] in frame.columns:
        active = frame[active_cols[0]].reindex(series.index).fillna(False).astype(bool)
        out.loc[~active] = np.nan
    return out


def _require_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc
    matplotlib.use("Agg")
    return plt


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def _align_eval_bounds_to_index(
    index: pd.DatetimeIndex,
    eval_min: pd.Timestamp,
    eval_max: pd.Timestamp,
    *,
    pad_frac: float = 0.02,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return [lo, hi] on the same timezone / naive convention as `index`."""
    if eval_min is None or eval_max is None or pd.isna(eval_min) or pd.isna(eval_max):
        return None
    emin = pd.Timestamp(eval_min)
    emax = pd.Timestamp(eval_max)
    tz = getattr(index, "tz", None)
    if tz is not None:
        if emin.tzinfo is None:
            emin = emin.tz_localize("UTC").tz_convert(tz)
            emax = emax.tz_localize("UTC").tz_convert(tz)
        else:
            emin = emin.tz_convert(tz)
            emax = emax.tz_convert(tz)
    else:
        if emin.tzinfo is not None:
            emin = emin.tz_convert(None)
        if emax.tzinfo is not None:
            emax = emax.tz_convert(None)
    span = emax - emin
    try:
        pad = span * pad_frac if pd.notna(span) and span > pd.Timedelta(0) else pd.Timedelta(0)
    except TypeError:
        pad = pd.Timedelta(0)
    return emin - pad, emax + pad


# When eval spans more than this many days, zoom plots to the trailing window
# (same policy as TimesNet class-summary plots).
_PLOT_FOCUS_DAYS = 7
_ANOMALY_FIGSIZE = (12, 4)
_RECON_FIGSIZE = (12, 4)


def _focus_plot_range(
    eval_min: pd.Timestamp | None,
    eval_max: pd.Timestamp | None,
    *,
    zoom_to_eval: bool,
    focus_days: int = _PLOT_FOCUS_DAYS,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return [plot_min, plot_max] for readable zoomed anomaly/recon plots."""
    if not zoom_to_eval or eval_min is None or eval_max is None:
        return None, None
    if pd.isna(eval_min) or pd.isna(eval_max):
        return None, None
    plot_min = pd.Timestamp(eval_min)
    plot_max = pd.Timestamp(eval_max)
    if (plot_max - plot_min) > pd.Timedelta(days=focus_days):
        plot_min = plot_max - pd.Timedelta(days=focus_days)
    return plot_min, plot_max


def _select_top_uids(annotated: pd.DataFrame, max_series: int | None) -> set[str]:
    all_uids = annotated["unique_id"].astype(str).unique().tolist()
    if max_series is None or max_series <= 0 or len(all_uids) <= max_series:
        return set(all_uids)
    counts = (
        annotated[annotated["is_anomaly"]]
        .groupby("unique_id", sort=False)["is_anomaly"]
        .sum()
        .sort_values(ascending=False)
    )
    selected: list[str] = list(counts.head(max_series).index.astype(str))
    if len(selected) < max_series:
        for uid in all_uids:
            if uid not in selected:
                selected.append(uid)
                if len(selected) >= max_series:
                    break
    return set(selected)


def plot_score_timeline(
    annotated_scores: pd.DataFrame,
    output_dir: Path,
    max_series: int | None = 12,
    splits_to_show: tuple[str, ...] = ("val", "test"),
    include_train: bool = False,
) -> list[Path]:
    """Per-series score(t) with cutoff and anomaly markers.

    By default hides train for readability on year-long runs.
    """
    plt = _require_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_top_uids(annotated_scores, max_series=max_series)
    paths: list[Path] = []
    for uid, group in annotated_scores.groupby("unique_id", sort=True):
        if uid not in selected:
            continue
        group = group.sort_values("ds")
        fig, ax = plt.subplots(figsize=(12, 3.8))
        split_order = ["train", "val", "test"] if include_train else ["val", "test"]
        split_color = {"train": "tab:gray", "val": "tab:blue", "test": "tab:green"}

        for split in split_order:
            chunk = group[group["split"] == split]
            if chunk.empty:
                continue
            if split not in splits_to_show and not (split == "train" and include_train):
                continue
            ax.plot(
                chunk["ds"],
                chunk["score"],
                color=split_color[split],
                linewidth=0.9,
                alpha=0.75,
                label=split,
            )

        cutoff = float(group["cutoff"].iloc[0])
        ax.axhline(cutoff, color="black", linestyle="--", linewidth=0.8, label="cutoff")

        anomalies = group[group["is_anomaly"]]
        if not anomalies.empty:
            warns = anomalies[anomalies["severity"] != "alert"]
            alerts = anomalies[anomalies["severity"] == "alert"]
            if not warns.empty:
                ax.scatter(
                    warns["ds"], warns["score"],
                    s=40, facecolors="none", edgecolors="tab:orange",
                    linewidths=1.2, label="anomaly (warn)", zorder=5,
                )
            if not alerts.empty:
                ax.scatter(
                    alerts["ds"], alerts["score"],
                    s=60, facecolors="none", edgecolors="tab:red",
                    linewidths=1.7, label="anomaly (alert)", zorder=6,
                )

        ax.set_title(f"score timeline {uid} (TimesNet)")
        ax.set_xlabel("time (UTC)")
        ax.set_ylabel("anomaly score")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        fig.autofmt_xdate()
        out = output_dir / f"score_{_safe(uid)}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        paths.append(out)
    return paths


def plot_actual_with_anomalies(
    frame: pd.DataFrame,
    target_columns: list[str],
    annotated_scores: pd.DataFrame,
    output_dir: Path,
    max_series: int | None = 12,
    eval_split: str = "test",
    zoom_to_eval: bool = True,
    include_full_span_copy: bool = True,
) -> list[Path]:
    """Plot actual y with anomaly markers.

    Default view zooms to the eval split (last 7 days when eval is longer).
    Optionally saves a full-span companion figure (``*_full.png``).
    """
    plt = _require_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_top_uids(annotated_scores, max_series=max_series)
    paths: list[Path] = []

    eval_ds = annotated_scores.loc[
        annotated_scores["split"] == eval_split,
        "ds",
    ]
    eval_min = pd.to_datetime(eval_ds.min(), utc=True) if not eval_ds.empty else None
    eval_max = pd.to_datetime(eval_ds.max(), utc=True) if not eval_ds.empty else None
    plot_min, plot_max = _focus_plot_range(
        eval_min, eval_max, zoom_to_eval=zoom_to_eval
    )

    for col in target_columns:
        uid = col[len("y__"):] if col.startswith("y__") else col
        if uid not in selected:
            continue
        series = frame[col].sort_index()

        anom = annotated_scores[
            (annotated_scores["unique_id"] == uid)
            & (annotated_scores["is_anomaly"])
            & (annotated_scores["split"] == eval_split)
        ].sort_values("ds")

        uid_eval = annotated_scores[
            (annotated_scores["unique_id"] == uid)
            & (annotated_scores["split"] == eval_split)
        ]

        def _draw(fig_name_suffix: str, xlim_zoom: bool) -> Path:
            fig, ax = plt.subplots(figsize=_ANOMALY_FIGSIZE)
            ser_plot = _mask_inactive_buckets(
                _trim_to_active_span(series.sort_index(), frame, col),
                frame,
                col,
            )
            bounds = None
            if xlim_zoom and plot_min is not None and plot_max is not None:
                bounds = _align_eval_bounds_to_index(
                    ser_plot.index, plot_min, plot_max, pad_frac=0.02
                )
                if bounds is not None:
                    lo, hi = bounds
                    sliced = ser_plot.loc[lo:hi]
                    if len(sliced) > 0:
                        ser_plot = sliced

            ax.plot(ser_plot.index, ser_plot.values, color="tab:blue", linewidth=1.2, label="actual")

            if "recon_val" in annotated_scores.columns:
                recon_series = uid_eval.set_index("ds")["recon_val"].sort_index()
                if bounds is not None:
                    lo, hi = bounds
                    recon_plot = recon_series.loc[lo:hi]
                else:
                    recon_plot = recon_series
                if not recon_plot.empty:
                    ax.plot(
                        recon_plot.index,
                        recon_plot.values,
                        color="tab:orange",
                        linewidth=1.2,
                        alpha=0.8,
                        label="reconstructed (TimesNet)",
                    )

            if not anom.empty:
                vals = series.reindex(anom["ds"]).to_numpy()
                warns_mask = anom["severity"].to_numpy() != "alert"
                alerts_mask = ~warns_mask
                if warns_mask.any():
                    ax.scatter(
                        anom.loc[warns_mask, "ds"],
                        vals[warns_mask],
                        s=55,
                        facecolors="none",
                        edgecolors="tab:orange",
                        linewidths=1.4,
                        label="anomaly (warn)",
                        zorder=5,
                    )
                if alerts_mask.any():
                    ax.scatter(
                        anom.loc[alerts_mask, "ds"],
                        vals[alerts_mask],
                        s=75,
                        facecolors="none",
                        edgecolors="tab:red",
                        linewidths=1.8,
                        label="anomaly (alert)",
                        zorder=6,
                    )

            if xlim_zoom and bounds is not None:
                lo, hi = bounds
                ax.set_xlim(lo, hi)

            ax.set_title(
                f"{uid} (TimesNet anomalies, split={eval_split}{fig_name_suffix})"
            )
            ax.set_xlabel("time (UTC)")
            ax.set_ylabel("vehicle count")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()

            suffix = "" if not fig_name_suffix else fig_name_suffix
            out = output_dir / f"actual_{_safe(uid)}{suffix}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=120)
            plt.close(fig)
            return out

        paths.append(_draw(fig_name_suffix="", xlim_zoom=zoom_to_eval))
        if include_full_span_copy:
            paths.append(_draw(fig_name_suffix="_full", xlim_zoom=False))

    return paths


def plot_reconstruction_vs_actual(
    frame: pd.DataFrame,
    target_columns: list[str],
    annotated_scores: pd.DataFrame,
    output_dir: Path,
    max_series: int | None = 12,
    eval_split: str = "test",
    zoom_to_eval: bool = True,
    include_full_span_copy: bool = True,
) -> list[Path]:
    """Overlay actual traffic vs last-step reconstruction (unscaled vehicle counts)."""
    plt = _require_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "recon_val" not in annotated_scores.columns:
        return []

    selected = _select_top_uids(annotated_scores, max_series=max_series)
    paths: list[Path] = []

    eval_ds = annotated_scores.loc[
        annotated_scores["split"] == eval_split,
        "ds",
    ]
    eval_min = pd.to_datetime(eval_ds.min(), utc=True) if not eval_ds.empty else None
    eval_max = pd.to_datetime(eval_ds.max(), utc=True) if not eval_ds.empty else None
    plot_min, plot_max = _focus_plot_range(
        eval_min, eval_max, zoom_to_eval=zoom_to_eval
    )

    for col in target_columns:
        uid = col[len("y__") :] if col.startswith("y__") else col
        if uid not in selected:
            continue

        series = frame[col].sort_index()
        recon_df = annotated_scores[
            (annotated_scores["unique_id"] == uid)
            & (annotated_scores["split"] == eval_split)
        ][["ds", "recon_val", "actual_val"]].drop_duplicates(subset=["ds"]).set_index("ds").sort_index()

        ser_base = _mask_inactive_buckets(
            _trim_to_active_span(series.sort_index(), frame, col),
            frame,
            col,
        )

        def _draw(fig_name_suffix: str, xlim_zoom: bool) -> Path:
            recon_plot = recon_df["recon_val"]
            ser_plot = ser_base
            bounds = None
            if xlim_zoom and plot_min is not None and plot_max is not None:
                bounds = _align_eval_bounds_to_index(
                    ser_plot.index, plot_min, plot_max, pad_frac=0.02
                )
                if bounds is not None:
                    lo, hi = bounds
                    ser_plot = ser_plot.loc[lo:hi]
                    recon_plot = recon_plot.loc[lo:hi]

            fig, ax = plt.subplots(figsize=_RECON_FIGSIZE)
            ax.plot(
                ser_plot.index,
                ser_plot.values,
                color="tab:blue",
                linewidth=1.2,
                label="actual",
            )
            if not recon_plot.empty:
                ax.plot(
                    recon_plot.index,
                    recon_plot.values,
                    color="tab:orange",
                    linewidth=1.2,
                    alpha=0.8,
                    label="reconstructed (last step)",
                )

            if xlim_zoom and bounds is not None:
                lo, hi = bounds
                ax.set_xlim(lo, hi)

            ax.set_title(
                f"{uid} — reconstruction vs actual (TimesNet, split={eval_split}"
                f"{fig_name_suffix})"
            )
            ax.set_xlabel("time (UTC)")
            ax.set_ylabel("vehicle count")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()

            suffix = "" if not fig_name_suffix else fig_name_suffix
            out = output_dir / f"recon_{_safe(uid)}{suffix}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=120)
            plt.close(fig)
            return out

        paths.append(_draw(fig_name_suffix="", xlim_zoom=zoom_to_eval))
        if include_full_span_copy:
            paths.append(_draw(fig_name_suffix="_full", xlim_zoom=False))

    return paths


def plot_train_log(log: dict, output_path: Path) -> Path | None:
    history = log.get("history") or []
    if not history:
        return None
    plt = _require_matplotlib()
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(epochs, [r["train_loss"] for r in history], label="train", linewidth=1.5)
    ax.plot(epochs, [r["val_loss"] for r in history], label="val", linewidth=1.5)
    hp = log.get("hparams") or {}
    loss_label = "RMSE reconstruction" if hp.get("use_rmse_loss", True) else "MSE reconstruction"
    ax.set_xlabel("epoch")
    ax.set_ylabel(loss_label)
    ax.set_title("TimesNet training curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Class-level summary: stacked bar + line + anomaly zones
# ---------------------------------------------------------------------------

# Fixed color palette for vehicle classes (visually distinct, colorblind-friendly).
_CLASS_COLORS: dict[int, str] = {
    1: "#999999",   # unknown-1 – grey 
    2: "#56B4E9",   # car+trailer – sky blue
    3: "#D55E00",   # truck – vermillion
    5: "#CC79A7",   # bus – pink
    7: "#009E73",   # car – teal
    8: "#F0E442",   # truck+trailer – yellow
    9: "#0072B2",   # tractor+trailer – blue
    10: "#E69F00",  # motorcycle – amber
    11: "#882255",  # LCV – wine
}

_FALLBACK_COLOR = "#AAAAAA"


def _parse_cls(uid: str) -> int | None:
    """Extract vehicle class id from uid like '24005-cze-l1__cls8'."""
    if "__cls" not in uid:
        return None
    try:
        return int(uid.rsplit("__cls", 1)[1])
    except (ValueError, IndexError):
        return None


def _location_from_uid(uid: str) -> str:
    """Strip '__clsN' suffix to get the base location id."""
    if "__cls" in uid:
        return uid.rsplit("__cls", 1)[0]
    return uid


def _cls_label(cls_id: int) -> str:
    from models.TimesNet.config import VEHICLE_CLASS_LABELS
    return VEHICLE_CLASS_LABELS.get(cls_id, f"cls{cls_id}")


def plot_location_class_summary(
    frame: pd.DataFrame,
    annotated_scores: pd.DataFrame,
    target_columns: list[str],
    output_dir: Path,
    eval_split: str = "test",
    anomaly_zone_threshold: int = 1,
    zoom_to_eval: bool = True,
    max_locations: int | None = 12,
) -> list[Path]:
    """Per-location stacked-bar chart with anomaly zones/markers.

    For each location that has class-level targets (``y__<loc>__cls<N>``):

    * **Top subplot (actual)**: line = hourly total, stacked bars = per-class
      breakdown, anomaly indicators overlaid.
    * **Bottom subplot (reconstructed)**: same layout using last-step reconstructions.

    Anomaly indicators use two modes:
      * *Zone* (``axvspan``): when ≥ ``anomaly_zone_threshold`` classes fire
        in the same hour.  Semi-transparent red background.
      * *Point + label*: when 1–(threshold-1) classes fire.  Scatter markers
        with a small text label naming the anomalous class(es).
    """
    plt = _require_matplotlib()
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group target columns by base location
    loc_to_cols: dict[str, list[tuple[str, int]]] = {}
    for col in target_columns:
        uid = col[len("y__"):] if col.startswith("y__") else col
        cls_id = _parse_cls(uid)
        if cls_id is None:
            continue
        loc = _location_from_uid(uid)
        loc_to_cols.setdefault(loc, []).append((col, cls_id))

    if not loc_to_cols:
        return []

    # Sort classes within each location by total count (heaviest on bottom)
    for loc in loc_to_cols:
        loc_to_cols[loc].sort(
            key=lambda x: frame[x[0]].sum(), reverse=True
        )

    # If max_locations specified, pick locations with most anomalies
    if max_locations and len(loc_to_cols) > max_locations:
        anom_df = annotated_scores[annotated_scores["is_anomaly"]].copy()
        anom_df["_loc"] = anom_df["unique_id"].map(_location_from_uid)
        loc_counts = anom_df.groupby("_loc").size().sort_values(ascending=False)
        keep = set(loc_counts.head(max_locations).index)
        loc_to_cols = {k: v for k, v in loc_to_cols.items() if k in keep}

    eval_scores = annotated_scores[annotated_scores["split"] == eval_split]
    eval_min = pd.to_datetime(eval_scores["ds"].min(), utc=True) if not eval_scores.empty else None
    eval_max = pd.to_datetime(eval_scores["ds"].max(), utc=True) if not eval_scores.empty else None
    plot_min, plot_max = _focus_plot_range(
        eval_min, eval_max, zoom_to_eval=zoom_to_eval
    )

    paths: list[Path] = []

    for loc, col_cls_pairs in sorted(loc_to_cols.items()):
        # Collect actual values per class
        cls_ids = [c for _, c in col_cls_pairs]
        cls_cols = [c for c, _ in col_cls_pairs]
        cls_actual = frame[cls_cols].copy()
        cls_actual.columns = cls_ids

        # Reconstructed values from annotated_scores for this location's classes
        loc_uids = [
            col[len("y__"):] if col.startswith("y__") else col
            for col in cls_cols
        ]
        recon_data: dict[int, pd.Series] = {}
        val_col = "recon_val" if "recon_val" in annotated_scores.columns else None
        for uid, cls_id in zip(loc_uids, cls_ids):
            mask = (
                (annotated_scores["unique_id"] == uid)
                & (annotated_scores["split"] == eval_split)
            )
            if val_col is None:
                continue
            chunk = annotated_scores.loc[mask, ["ds", val_col]].set_index("ds")[val_col]
            if not chunk.empty:
                recon_data[cls_id] = chunk

        # Anomaly info per hour: which classes are anomalous
        anom_per_hour: dict[pd.Timestamp, list[int]] = {}
        for uid, cls_id in zip(loc_uids, cls_ids):
            mask = (
                (annotated_scores["unique_id"] == uid)
                & (annotated_scores["split"] == eval_split)
                & (annotated_scores["is_anomaly"])
            )
            for ds_val in annotated_scores.loc[mask, "ds"]:
                anom_per_hour.setdefault(ds_val, []).append(cls_id)

        if zoom_to_eval and plot_min is not None and plot_max is not None:
            bounds = _align_eval_bounds_to_index(
                cls_actual.index, plot_min, plot_max, pad_frac=0.02
            )
            if bounds is not None:
                lo, hi = bounds
                cls_actual = cls_actual.loc[lo:hi]

        if cls_actual.empty:
            continue

        ds_index = cls_actual.index
        n_hours = len(ds_index)

        # Determine bar width (in datetime units)
        if n_hours > 1:
            median_gap = pd.Series(ds_index).diff().dropna().median()
            bar_width = median_gap * 0.85
        else:
            bar_width = pd.Timedelta(hours=0.85)

        # Sort classes by total volume (heaviest first → bottom of stack)
        sorted_cls = sorted(cls_ids, key=lambda c: cls_actual[c].sum(), reverse=True)

        fig, (ax_act, ax_pred) = plt.subplots(
            2, 1, figsize=(16, 8), sharex=True,
            gridspec_kw={"hspace": 0.12}, layout="constrained",
        )

        # --- Top subplot: Actual ---
        bottom = np.zeros(n_hours, dtype=np.float64)
        handles_legend = []
        for cls_id in sorted_cls:
            vals = cls_actual[cls_id].to_numpy(dtype=np.float64)
            color = _CLASS_COLORS.get(cls_id, _FALLBACK_COLOR)
            bars = ax_act.bar(
                ds_index, vals, bottom=bottom,
                width=bar_width, color=color, edgecolor="none",
                alpha=0.75, label=_cls_label(cls_id),
            )
            handles_legend.append(bars)
            bottom += vals

        # Total line
        total_actual = cls_actual[sorted_cls].sum(axis=1)
        ax_act.plot(
            ds_index, total_actual.values,
            color="black", linewidth=1.3, alpha=0.9, label="total",
        )

        # Anomaly indicators on actual
        _draw_anomaly_indicators(
            ax_act, ds_index, total_actual, anom_per_hour,
            bar_width, anomaly_zone_threshold,
        )

        ax_act.set_title(f"{loc} — actual (class breakdown)", fontsize=12)
        ax_act.set_ylabel("vehicle count")
        ax_act.grid(True, alpha=0.2, axis="y")
        ax_act.legend(
            loc="upper left", fontsize=7, ncol=min(len(sorted_cls) + 1, 6),
            framealpha=0.8,
        )

        # --- Bottom subplot: Reconstructed ---
        if recon_data:
            recon_frame = pd.DataFrame(index=ds_index)
            for cls_id in sorted_cls:
                if cls_id in recon_data:
                    recon_frame[cls_id] = recon_data[cls_id].reindex(ds_index).fillna(0.0)
                else:
                    recon_frame[cls_id] = 0.0
            recon_frame = recon_frame.clip(lower=0.0)

            bottom_p = np.zeros(len(ds_index), dtype=np.float64)
            for cls_id in sorted_cls:
                vals_p = recon_frame[cls_id].to_numpy(dtype=np.float64)
                color = _CLASS_COLORS.get(cls_id, _FALLBACK_COLOR)
                ax_pred.bar(
                    ds_index, vals_p, bottom=bottom_p,
                    width=bar_width, color=color, edgecolor="none",
                    alpha=0.75,
                )
                bottom_p += vals_p

            total_recon = recon_frame[sorted_cls].sum(axis=1)
            ax_pred.plot(
                ds_index, total_recon.values,
                color="black", linewidth=1.3, alpha=0.9, label="total reconstructed",
            )
            ax_pred.legend(loc="upper left", fontsize=7, framealpha=0.8)
        else:
            ax_pred.text(
                0.5, 0.5, "no reconstructions available for eval split",
                ha="center", va="center", transform=ax_pred.transAxes,
                fontsize=11, color="gray",
            )

        ax_pred.set_title(f"{loc} — reconstructed (TimesNet)", fontsize=12)
        ax_pred.set_ylabel("vehicle count")
        ax_pred.set_xlabel("time (UTC)")
        ax_pred.grid(True, alpha=0.2, axis="y")
        fig.autofmt_xdate()

        out = output_dir / f"class_summary_{_safe(loc)}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    return paths


def _draw_anomaly_indicators(
    ax,
    ds_index: pd.DatetimeIndex,
    total_values: pd.Series,
    anom_per_hour: dict,
    bar_width: pd.Timedelta,
    zone_threshold: int,
) -> None:
    """Draw anomaly markers on an axis: zones for dense anomaly clusters, points for sparse."""
    import numpy as np

    if not anom_per_hour:
        return

    half_bar = bar_width / 2

    valid_ts = [t for t in sorted(anom_per_hour.keys()) if t in ds_index]
    if not valid_ts:
        return

    clusters = _group_consecutive(valid_ts, bar_width)
    
    zone_spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    point_hours: list[tuple[pd.Timestamp, list[int]]] = []

    for chunk in clusters:
        num_hours = len(chunk)
        total_anomalies = sum(len(anom_per_hour[t]) for t in chunk)
        if total_anomalies >= num_hours * zone_threshold:
            zone_spans.append((chunk[0], chunk[-1]))
        else:
            for t in chunk:
                point_hours.append((t, anom_per_hour[t]))

    # Draw zones
    if zone_spans:
        for span_lo, span_hi in zone_spans:
            ax.axvspan(
                span_lo - half_bar, span_hi + half_bar,
                alpha=0.13, color="red", zorder=0,
                label="_nolegend_",
            )
        # Add a single legend entry for zones
        ax.axvspan(
            zone_spans[0][0], zone_spans[0][0],
            alpha=0.13, color="red", label=f"anomaly zone (avg ≥{zone_threshold} cls/h)",
        )

    # Draw individual points
    if point_hours:
        xs = [h[0] for h in point_hours]
        ys = [float(total_values.get(h[0], 0)) for h in point_hours]
        ax.scatter(
            xs, ys,
            s=50, facecolors="none", edgecolors="tab:red",
            linewidths=1.5, zorder=7,
            label="anomaly (few cls)",
        )
        # Annotate with class names (only if not too many)
        if len(point_hours) <= 80:
            for ds_val, cls_list in point_hours:
                y_val = float(total_values.get(ds_val, 0))
                label_text = ", ".join(_cls_label(c) for c in sorted(cls_list))
                if len(label_text) > 18:
                    label_text = ", ".join(f"c{c}" for c in sorted(cls_list))
                ax.annotate(
                    label_text,
                    xy=(ds_val, y_val),
                    xytext=(0, 8),
                    textcoords="offset points",
                    fontsize=5.5,
                    color="darkred",
                    ha="center",
                    va="bottom",
                    rotation=45,
                    alpha=0.85,
                )


def _group_consecutive(
    timestamps: list[pd.Timestamp],
    gap: pd.Timedelta,
) -> list[list[pd.Timestamp]]:
    """Group timestamps into clusters where adjacent items are within gap*1.5."""
    if not timestamps:
        return []
    ts = sorted(timestamps)
    clusters: list[list[pd.Timestamp]] = []
    current_cluster = [ts[0]]
    for t in ts[1:]:
        if t - current_cluster[-1] <= gap * 1.5:
            current_cluster.append(t)
        else:
            clusters.append(current_cluster)
            current_cluster = [t]
    clusters.append(current_cluster)
    return clusters

