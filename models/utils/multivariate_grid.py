"""Masking helpers for joint multivariate anomaly models (TimesNet, TimesNet).

Locations can start/end at different times on a shared timeline. A timestep is
**active** only when that location has an observed bucket (not merely between
first/last measurement). Calendar hours outside the span and internal gaps are
inactive (excluded from loss/scoring/plots). Gaps may be forward-filled in the
stored target tensor for forward pass only; loss still uses the active mask.

Train/val/test splits are assigned chronologically *per location* on active
timesteps only (last ``test_fraction`` for test, etc.).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

ACTIVE_PREFIX = "active__"


def active_column_for(target_column: str) -> str:
    if not target_column.startswith("y__"):
        raise ValueError(f"expected y__ target column, got {target_column!r}")
    return f"{ACTIVE_PREFIX}{target_column[len('y__'):]}"


def list_active_columns(target_columns: Sequence[str]) -> list[str]:
    return [active_column_for(c) for c in target_columns]


def build_active_mask_frame(
    long_df: pd.DataFrame,
    location_ids: Sequence[str],
    full_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Boolean mask per location on ``full_index`` (True = station was operating)."""
    bounds = (
        long_df.groupby("location_id", sort=False)["ds"]
        .agg(["min", "max"])
    )
    active = pd.DataFrame(False, index=full_index, columns=list(location_ids))
    for loc in location_ids:
        if loc not in bounds.index:
            continue
        lo = bounds.at[loc, "min"]
        hi = bounds.at[loc, "max"]
        if pd.isna(lo) or pd.isna(hi):
            continue
        active.loc[lo:hi, loc] = True
    return active


def attach_active_columns(
    wide_y: pd.DataFrame,
    long_df: pd.DataFrame,
    full_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, int, int]:
    """Add ``active__*`` columns; return active df, inactive_cells, gap_cells."""
    location_ids = [c[len("y__") :] for c in wide_y.columns]
    active_raw = build_active_mask_frame(long_df, location_ids, full_index)
    reindexed_y = wide_y.copy()
    reindexed_y.index = full_index
    observed = pd.DataFrame(
        {loc: reindexed_y[f"y__{loc}"].notna() for loc in location_ids},
        index=full_index,
    )
    gap_mask = active_raw & ~observed
    inactive_mask = ~active_raw
    active_effective = active_raw & observed
    active_cols = pd.DataFrame(
        {
            active_column_for(f"y__{loc}"): active_effective[loc].astype(np.float32).values
            for loc in location_ids
        },
        index=full_index,
    )
    return (
        active_cols,
        int(inactive_mask.values.sum()),
        int(gap_mask.values.sum()),
    )


def fill_gap_values_in_span(
    wide_y: pd.DataFrame,
    long_df: pd.DataFrame,
    full_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Forward/back-fill missing buckets inside each location's [min, max] span.

    Used for model input only; ``active__*`` remains False on gap timesteps so
    loss and scoring ignore them.
    """
    location_ids = [c[len("y__") :] for c in wide_y.columns]
    active_raw = build_active_mask_frame(long_df, location_ids, full_index)
    out = wide_y.copy()
    out.index = full_index
    for loc in location_ids:
        col = f"y__{loc}"
        span = active_raw[loc]
        if not span.any():
            continue
        block = out.loc[span, col]
        out.loc[span, col] = block.ffill().bfill()
    return out


def per_location_split_masks(
    active: np.ndarray,
    *,
    val_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-target chronological train/val/test masks (shape T x F)."""
    if active.ndim != 2:
        raise ValueError(f"active must be 2-D, got shape={active.shape}")
    t_total, n_feat = active.shape
    train = np.zeros((t_total, n_feat), dtype=bool)
    val = np.zeros((t_total, n_feat), dtype=bool)
    test = np.zeros((t_total, n_feat), dtype=bool)

    for j in range(n_feat):
        idx = np.flatnonzero(active[:, j])
        n = len(idx)
        if n == 0:
            continue
        n_test = int(n * test_fraction)
        n_val = int(n * val_fraction)
        if n_test <= 0 and test_fraction > 0 and n >= 2:
            n_test = 1
        if n_val + n_test >= n:
            n_val = max(0, min(n_val, n - 1))
            n_test = max(0, n - n_val - 1)
        n_train = n - n_val - n_test
        if n_train <= 0:
            n_train = max(1, n - n_test)
            n_val = 0
        train[idx[:n_train], j] = True
        if n_val > 0:
            val[idx[n_train : n_train + n_val], j] = True
        if n_test > 0:
            test[idx[n_train + n_val :], j] = True

    return train, val, test


def window_start_indices(end_mask: np.ndarray, window: int) -> np.ndarray:
    """Window start rows where the window's last timestep satisfies ``end_mask``."""
    if end_mask.ndim != 1:
        raise ValueError("end_mask must be 1-D")
    t_total = len(end_mask)
    last = t_total - window + 1
    if last <= 0:
        return np.array([], dtype=np.int64)
    end_idx = np.arange(window - 1, window - 1 + last)
    return np.nonzero(end_mask[end_idx])[0].astype(np.int64)


def fit_robust_scaler_masked(
    values: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Per-column robust scaler fit only where ``mask`` is True."""
    if values.ndim != 2 or mask.ndim != 2:
        raise ValueError(f"values and mask must be 2-D; got {values.shape}, {mask.shape}")
    if values.shape != mask.shape:
        raise ValueError(f"shape mismatch: values={values.shape} mask={mask.shape}")

    n_cols = values.shape[1]
    median = np.zeros(n_cols, dtype=np.float32)
    scale = np.ones(n_cols, dtype=np.float32)

    for j in range(n_cols):
        sel = mask[:, j]
        if not np.any(sel):
            continue
        col = values[sel, j].astype(np.float64)
        med = float(np.median(col))
        mad = float(np.median(np.abs(col - med)))
        sc = 1.4826 * mad
        if sc <= eps:
            sc = float(np.std(col))
        if sc <= eps:
            sc = 1.0
        median[j] = med
        scale[j] = sc

    return {"median": median.astype(np.float32), "scale": scale.astype(np.float32)}


def extract_active_mask(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
) -> np.ndarray:
    """Return (T, F) active mask; infer from targets if legacy dataset."""
    active_cols = list_active_columns(target_columns)
    if all(c in frame.columns for c in active_cols):
        return frame[active_cols].to_numpy(dtype=bool)

    print(
        "[multivariate_grid] WARN: dataset has no active__ columns; "
        "inferring active span from first/last non-zero target (re-run prepare)."
    )
    y = frame[list(target_columns)].to_numpy(dtype=np.float32)
    t_total, n_feat = y.shape
    active = np.zeros((t_total, n_feat), dtype=bool)
    for j in range(n_feat):
        nz = np.flatnonzero(y[:, j] > 0)
        if len(nz) == 0:
            continue
        active[nz[0] : nz[-1] + 1, j] = True
    return active


def split_label_for_timestep(
    t_global: int,
    j: int,
    *,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    active_mask: np.ndarray,
) -> str:
    if not active_mask[t_global, j]:
        return "inactive"
    if test_mask[t_global, j]:
        return "test"
    if val_mask[t_global, j]:
        return "val"
    if train_mask[t_global, j]:
        return "train"
    return "unknown"


def mask_windows_for_targets(
    cell_mask: np.ndarray,
    target_idx: np.ndarray,
    window: int,
) -> np.ndarray:
    """Sliding-window mask (n_windows, window, n_targets)."""
    sel = cell_mask[:, target_idx].astype(np.float32)
    t_total = sel.shape[0]
    last = t_total - window + 1
    if last <= 0:
        return np.zeros((0, window, len(target_idx)), dtype=np.float32)
    w = np.lib.stride_tricks.sliding_window_view(sel, window, axis=0)[:last]
    return np.transpose(w, (0, 2, 1)).copy()


def subset_windows(
    xs: np.ndarray,
    x_tgt: np.ndarray,
    y_next: np.ndarray,
    x_mark: np.ndarray | None,
    loss_mask: np.ndarray | None,
    starts: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Return (xs, x_tgt, y_next, x_mark, loss_mask) for the selected window starts."""
    if starts.size == 0:
        return (
            np.zeros((0, *xs.shape[1:]), dtype=np.float32),
            np.zeros((0, *x_tgt.shape[1:]), dtype=np.float32),
            np.zeros((0, *y_next.shape[1:]), dtype=np.float32),
            None if x_mark is None else np.zeros((0, *x_mark.shape[1:]), dtype=np.float32),
            None if loss_mask is None else np.zeros((0, *loss_mask.shape[1:]), dtype=np.float32),
        )
    return (
        xs[starts],
        x_tgt[starts],
        y_next[starts],
        x_mark[starts] if x_mark is not None else None,
        loss_mask[starts] if loss_mask is not None else None,
    )


def summarize_split_bounds(mask: np.ndarray) -> tuple[int, int]:
    idx = np.flatnonzero(mask.any(axis=1))
    if len(idx) == 0:
        return (0, 0)
    return (int(idx[0]), int(idx[-1] + 1))
