"""Forecast windows gathered on the compute device from the base (T, C) arrays.

The whole series fits in device memory (tens of MB even at 5min), so batches
are built by index arithmetic on the device. No DataLoader workers and no
shared memory are involved.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class DeviceWindowBatches:
    """Iterate (x, y, y_active, marks, y_lag, lag_active) batches.

    x: (B, window, C) past targets; y and y_active: (B, pred_len, C) future
    targets and their activity mask; marks: (B, window + pred_len, M).
    y_lag and lag_active repeat y and y_active shifted back by ``lag_steps``;
    both are zeros where that shift falls before the start of the series.
    """

    def __init__(
        self,
        values: np.ndarray,
        active: np.ndarray,
        marks: np.ndarray,
        starts: np.ndarray,
        window: int,
        pred_len: int,
        batch_size: int,
        *,
        shuffle: bool,
        device: torch.device,
        lag_steps: int = 0,
    ) -> None:
        if values.ndim != 2 or active.shape != values.shape:
            raise ValueError(f"values/active shape mismatch: {values.shape} vs {active.shape}")
        if marks.shape[0] != values.shape[0]:
            raise ValueError(f"marks rows {marks.shape[0]} != values rows {values.shape[0]}")
        self.values = torch.tensor(np.ascontiguousarray(values), dtype=torch.float32, device=device)
        self.active = torch.tensor(np.ascontiguousarray(active), dtype=torch.float32, device=device)
        self.marks = torch.tensor(np.ascontiguousarray(marks), dtype=torch.float32, device=device)
        self.starts = torch.as_tensor(np.asarray(starts, dtype=np.int64), device=device)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.lag_steps = int(lag_steps)
        self.off_x = torch.arange(window, device=device)
        self.off_y = torch.arange(window, window + pred_len, device=device)
        self.off_m = torch.arange(window + pred_len, device=device)

    def __len__(self) -> int:
        return math.ceil(len(self.starts) / self.batch_size)

    def __iter__(self):
        n = len(self.starts)
        device = self.starts.device
        order = torch.randperm(n, device=device) if self.shuffle else torch.arange(n, device=device)
        for i in range(0, n, self.batch_size):
            s = self.starts[order[i : i + self.batch_size]][:, None]
            y_at = s + self.off_y
            lag_at = y_at - self.lag_steps
            in_series = lag_at >= 0
            y_lag = self.values[lag_at.clamp(min=0)]
            lag_active = self.active[lag_at.clamp(min=0)] * in_series.unsqueeze(-1)
            yield (
                self.values[s + self.off_x],
                self.values[y_at],
                self.active[y_at],
                self.marks[s + self.off_m],
                y_lag,
                lag_active,
            )


def forecast_window_starts(
    n_rows: int,
    window: int,
    pred_len: int,
    *,
    target_lo: int,
    target_hi: int,
    active: np.ndarray | None = None,
    stride: int = 1,
) -> np.ndarray:
    """Window starts whose whole horizon lies in rows [target_lo, target_hi).

    The input part may reach back into earlier rows; that is past data and does
    not leak. With ``active`` given, horizons without any observed cell are
    dropped because they carry no loss.
    """
    s_min = max(0, target_lo - window)
    s_max = min(target_hi, n_rows) - window - pred_len
    if s_max < s_min:
        return np.array([], dtype=np.int64)
    starts = np.arange(s_min, s_max + 1, max(1, int(stride)), dtype=np.int64)
    if active is None or starts.size == 0:
        return starts
    any_active = active.any(axis=1).astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(any_active)])
    lo = starts + window
    hits = csum[lo + pred_len] - csum[lo]
    return starts[hits > 0]
