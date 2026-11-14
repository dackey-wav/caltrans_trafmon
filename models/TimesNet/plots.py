"""Forecast plots and an HTML index that collects them.

Time axes use America/Los_Angeles. Actual values at gap or imputed steps are
not drawn, so the lines show only real observations.
"""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.utils.constants import LOCAL_TZ, steps_per_day  # noqa: E402

CHANNEL_TITLES = {
    "flow": "Flow, veh per step",
    "speed": "Speed, mph",
    "occupancy": "Occupancy, fraction",
}
METHOD_STYLE = {
    "model": {"color": "tab:red", "lw": 1.8},
    "persistence": {"color": "tab:gray", "lw": 1.0, "ls": ":"},
    "daily": {"color": "tab:green", "lw": 1.0, "ls": "--"},
    "weekly": {"color": "tab:blue", "lw": 1.0, "ls": "--"},
}
SCHOOL_BREAK_PREFIX = "is_school_break_"


def _local(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_convert(LOCAL_TZ) if idx.tz is not None else idx


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _date_axis(ax, span_days: float) -> None:
    tz = _local(pd.DatetimeIndex([pd.Timestamp.now(tz="UTC")])).tz
    if span_days <= 3:
        step = 3 if span_days <= 1.2 else 6
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, step), tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M", tz=tz))
    elif span_days <= 10:
        ax.xaxis.set_major_locator(mdates.DayLocator(tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %m-%d", tz=tz))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d", tz=tz))
    ax.grid(alpha=0.3)


def _shade_flags(ax, index: pd.DatetimeIndex, flags: pd.DataFrame | None) -> None:
    """Shade holidays and school breaks inside the visible range."""
    if flags is None or flags.empty:
        return
    colors = {"is_holiday": "orange", "school": "purple"}
    groups = {
        "is_holiday": flags["is_holiday"] if "is_holiday" in flags else None,
        "school": flags[[c for c in flags.columns if c.startswith(SCHOOL_BREAK_PREFIX)]].max(axis=1)
        if any(c.startswith(SCHOOL_BREAK_PREFIX) for c in flags.columns)
        else None,
    }
    for key, series in groups.items():
        if series is None:
            continue
        on = series.to_numpy() > 0
        if not on.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2], strict=False):
            ax.axvspan(index[a], index[min(b, len(index) - 1)], color=colors[key], alpha=0.08, lw=0)


def plot_train_log(log: dict, out_path: Path) -> Path:
    history = log.get("history", [])
    epochs = [h["epoch"] for h in history]
    loss_name = log.get("hparams", {}).get("loss", "loss")
    channels = list(history[0].get("val_mae", {}).keys()) if history else []
    fig, axes = plt.subplots(1, 1 + len(channels), figsize=(4.5 * (1 + len(channels)), 3.6))
    axes = np.atleast_1d(axes)
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    if log.get("best_epoch"):
        axes[0].axvline(log["best_epoch"], color="k", ls=":", lw=1, label="best")
    axes[0].set_title(f"Masked {loss_name} on scaled targets")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for ax, ch in zip(axes[1:], channels, strict=False):
        ax.plot(epochs, [h["val_mae"][ch] for h in history], color="tab:red")
        ax.set_title(f"Val MAE, {CHANNEL_TITLES.get(ch, ch)}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_error_by_lead(metrics: dict, out_path: Path) -> Path:
    channels = metrics["channels"]
    lead_h = np.asarray(metrics["lead_minutes"], dtype=float) / 60.0
    fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 3.8), squeeze=False)
    for ax, ch in zip(axes[0], channels, strict=False):
        for method, style in METHOD_STYLE.items():
            vals = metrics["by_lead"][method][ch]["mae"]
            ax.plot(lead_h, [np.nan if v is None else v for v in vals], label=method, **style)
        ax.set_title(f"MAE by lead, {CHANNEL_TITLES.get(ch, ch)}")
        ax.set_xlabel("lead, hours")
        ax.grid(alpha=0.3)
    axes[0][0].legend()
    fig.suptitle(f"{metrics['split']} {metrics['split_ds'][0][:10]} .. {metrics['split_ds'][1][:10]}")
    fig.tight_layout()
    return _save(fig, out_path)


def _masked_actual(frame: pd.DataFrame, target: str, active: np.ndarray) -> np.ndarray:
    vals = frame[target].to_numpy(dtype=float).copy()
    vals[~active] = np.nan
    return vals


def plot_period(
    frame: pd.DataFrame,
    active: np.ndarray,
    target_columns: list[str],
    channels: list[str],
    predictions: pd.DataFrame,
    leads: list[tuple[int, str]],
    row_lo: int,
    row_hi: int,
    weekly_lag: int,
    title: str,
    out_path: Path,
) -> Path:
    """Actual vs model at chosen leads and the weekly baseline, one row per channel."""
    sl = slice(row_lo, row_hi)
    idx = _local(frame.index[sl])
    flag_cols = [c for c in frame.columns if c == "is_holiday" or c.startswith(SCHOOL_BREAK_PREFIX)]
    flags = frame[flag_cols].iloc[sl] if flag_cols else None
    lead_colors = ["tab:red", "tab:orange", "tab:purple", "tab:brown"]
    fig, axes = plt.subplots(len(channels), 1, figsize=(15, 3.0 * len(channels)), sharex=True, squeeze=False)
    for j, (ch, tcol) in enumerate(zip(channels, target_columns, strict=False)):
        ax = axes[j][0]
        _shade_flags(ax, idx, flags)
        actual = _masked_actual(frame, tcol, active[:, j])
        ax.plot(idx, actual[sl], color="black", lw=1.2, label="actual")
        if row_lo - weekly_lag >= 0:
            weekly = frame[tcol].to_numpy(dtype=float)[row_lo - weekly_lag : row_hi - weekly_lag]
            ax.plot(idx, weekly, color="tab:blue", lw=0.9, ls="--", alpha=0.7, label="weekly baseline")
        sub = predictions[predictions["channel"] == ch]
        for (lead, label), color in zip(leads, lead_colors, strict=False):
            p = sub[sub["lead_steps"] == lead].set_index("ds")["y_pred"]
            p = p.reindex(frame.index[sl])
            ax.plot(idx, p.to_numpy(), color=color, lw=1.1, alpha=0.9, label=f"model {label}")
        ax.set_ylabel(CHANNEL_TITLES.get(ch, ch))
        span_days = (row_hi - row_lo) / max(1, weekly_lag / 7)
        _date_axis(ax, span_days)
    axes[0][0].legend(loc="upper left", ncol=5, fontsize=8)
    axes[0][0].set_title(title + "  (orange band: holiday, purple band: school break)")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_daily_overview(
    frame: pd.DataFrame,
    active: np.ndarray,
    target_columns: list[str],
    channels: list[str],
    predictions: pd.DataFrame,
    lead: int,
    lead_label: str,
    row_lo: int,
    row_hi: int,
    out_path: Path,
) -> Path:
    """Daily mean of actual and of the model at one lead across the whole split."""
    sl = slice(row_lo, row_hi)
    local = _local(frame.index[sl])
    fig, axes = plt.subplots(len(channels), 1, figsize=(15, 2.8 * len(channels)), sharex=True, squeeze=False)
    for j, (ch, tcol) in enumerate(zip(channels, target_columns, strict=False)):
        ax = axes[j][0]
        actual = pd.Series(_masked_actual(frame, tcol, active[:, j])[sl], index=local)
        sub = predictions[(predictions["channel"] == ch) & (predictions["lead_steps"] == lead)]
        pred = sub.set_index("ds")["y_pred"].reindex(frame.index[sl])
        pred.index = local
        both = actual.notna() & pred.notna()
        a_day = actual.where(both).groupby(local.date).mean()
        p_day = pred.where(both).groupby(local.date).mean()
        n_day = both.groupby(local.date).sum()
        full = n_day >= 0.5 * n_day.max()
        days = pd.to_datetime(a_day.index[full.to_numpy()])
        ax.plot(days, a_day[full].to_numpy(), color="black", lw=1.2, marker=".", label="actual")
        ax.plot(days, p_day[full].to_numpy(), color="tab:red", lw=1.1, marker=".", label=f"model {lead_label}")
        ax.set_ylabel(CHANNEL_TITLES.get(ch, ch))
        ax.grid(alpha=0.3)
    axes[0][0].legend(loc="upper left")
    axes[0][0].set_title("Daily means over steps that have both an observation and a forecast")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_example(
    frame: pd.DataFrame,
    active: np.ndarray,
    target_columns: list[str],
    channels: list[str],
    example: dict,
    history_steps: int,
    steps_day: int,
    out_path: Path,
) -> Path:
    """History, actual future, model forecast and weekly baseline for one origin."""
    origin = int(example["origin_row"])
    horizon = example["pred"].shape[0]
    lo = max(0, origin - history_steps)
    hi = min(len(frame), origin + horizon)
    idx = _local(frame.index[lo:hi])
    fut_idx = _local(frame.index[origin : origin + horizon])
    fig, axes = plt.subplots(len(channels), 1, figsize=(13, 2.8 * len(channels)), sharex=True, squeeze=False)
    for j, (ch, tcol) in enumerate(zip(channels, target_columns, strict=False)):
        ax = axes[j][0]
        actual = _masked_actual(frame, tcol, active[:, j])[lo:hi]
        ax.plot(idx, actual, color="black", lw=1.1, label="actual")
        ax.plot(fut_idx, example["weekly"][:, j], color="tab:blue", lw=0.9, ls="--", label="weekly baseline")
        ax.plot(fut_idx, example["pred"][:, j], color="tab:red", lw=1.6, label="forecast")
        ax.axvline(fut_idx[0], color="k", ls=":", lw=1)
        ax.set_ylabel(CHANNEL_TITLES.get(ch, ch))
        _date_axis(ax, (hi - lo) / steps_day)
    axes[0][0].legend(loc="upper left", ncol=3, fontsize=8)
    axes[0][0].set_title(f"Forecast from {fut_idx[0]:%Y-%m-%d %a %H:%M} (Los Angeles)")
    fig.tight_layout()
    return _save(fig, out_path)


def pick_weeks(
    frame: pd.DataFrame,
    active: np.ndarray,
    predictions: pd.DataFrame,
    channel: str,
    lead: int,
    row_lo: int,
    row_hi: int,
    freq: str,
    min_coverage: float = 0.8,
) -> list[tuple[str, int, int]]:
    """First, last and worst well-observed week of the split, plus a school-break week.

    Candidate weeks start at every day boundary of the split. Only weeks where at
    least ``min_coverage`` of steps are real observations qualify, so sensor
    outages do not crowd out the views.
    """
    spd = steps_per_day(freq)
    week = 7 * spd
    if row_hi - row_lo <= week:
        return [("whole split", row_lo, row_hi)]

    starts = np.arange(row_lo, row_hi - week + 1, spd)
    obs = active.any(axis=1).astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(obs)])
    coverage = (csum[starts + week] - csum[starts]) / week
    good = starts[coverage >= min_coverage]
    if good.size == 0:
        good = starts[[int(np.argmax(coverage))]]

    weeks = [("first week", int(good[0]), int(good[0]) + week)]
    weeks.append(("last week", int(good[-1]), int(good[-1]) + week))

    sub = predictions[(predictions["channel"] == channel) & (predictions["lead_steps"] == lead)]
    err = (sub["y_pred"] - sub["y_true"]).abs().where(sub["active"])
    err.index = pd.DatetimeIndex(sub["ds"])
    err = err.reindex(frame.index).to_numpy()
    err_csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(err))])
    err_cnt = np.concatenate([[0], np.cumsum(np.isfinite(err))])
    n = err_cnt[good + week] - err_cnt[good]
    mean_err = np.where(n > 0, (err_csum[good + week] - err_csum[good]) / np.maximum(n, 1), -1.0)
    if len(good) >= 2 and mean_err.max() > 0:
        w = int(good[int(np.argmax(mean_err))])
        weeks.append(("worst week", w, w + week))

    school_cols = [c for c in frame.columns if c.startswith(SCHOOL_BREAK_PREFIX)]
    if school_cols:
        on = (frame[school_cols].max(axis=1).to_numpy() > 0).astype(np.int64)
        on_csum = np.concatenate([[0], np.cumsum(on)])
        share = (on_csum[good + week] - on_csum[good]) / week
        if share.max() > 0:
            w = int(good[int(np.argmax(share))])
            weeks.append(("school break week", w, w + week))

    seen: set[int] = set()
    out = []
    for label, a, b in weeks:
        if a in seen:
            continue
        seen.add(a)
        out.append((label, a, b))
    return out


def write_index_html(
    out_dir: Path,
    title: str,
    sections: list[tuple[str, list[Path]]],
    summary_text: str,
    table_rows: list[list[str]] | None = None,
    table_header: list[str] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:sans-serif;margin:24px;max-width:1600px}"
        "img{max-width:100%;border:1px solid #ddd;margin:6px 0}"
        "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}"
        "th{background:#f3f3f3}pre{background:#f7f7f7;padding:10px}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<pre>{html.escape(summary_text)}</pre>",
    ]
    if table_rows and table_header:
        parts.append("<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in table_header) + "</tr>")
        for row in table_rows:
            parts.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>")
        parts.append("</table>")
    for heading, images in sections:
        if not images:
            continue
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        for img in images:
            rel = Path(img).relative_to(out_dir).as_posix()
            parts.append(f"<div><a href='{rel}'><img src='{rel}' loading='lazy'></a></div>")
    parts.append("</body></html>")
    path = out_dir / "index.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
