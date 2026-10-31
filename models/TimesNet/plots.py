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


# When eval spans more than this many days, zoom plots to the trailing window.
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
