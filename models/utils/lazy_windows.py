"""On-demand sliding windows for multivariate time-series training.

Materializing all windows for long 15-min runs can exceed host RAM (e.g. 74
targets × 79k steps × window 192). This module slices windows from the base
(T, F) array inside ``Dataset.__getitem__`` so peak memory stays O(T·F + batch).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LazyWindowDataset(Dataset):
    """Slice ``values[start:start+window]`` on each access."""

    def __init__(
        self,
        values: np.ndarray,
        starts: np.ndarray,
        target_idx: np.ndarray,
        window: int,
        *,
        time_marks: np.ndarray | None = None,
        cell_mask: np.ndarray | None = None,
        pred_horizon: int = 0,
    ) -> None:
        if values.ndim != 2:
            raise ValueError(f"values must be 2-D, got shape={values.shape}")
        self.values = values
        self.starts = np.asarray(starts, dtype=np.int64)
        self.target_idx = np.asarray(target_idx, dtype=np.int64)
        self.window = int(window)
        self.time_marks = time_marks
        self.cell_mask = cell_mask
        self.pred_horizon = int(pred_horizon)
        self.n_targets = int(len(self.target_idx))

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, ...]:
        start = int(self.starts[idx])
        end = start + self.window
        x = self.values[start:end]
        x_tgt = x[:, self.target_idx]
        if self.pred_horizon > 0:
            y_next = self.values[end : end + self.pred_horizon][:, self.target_idx]
        else:
            y_next = np.zeros((1, self.n_targets), dtype=np.float32)
        x_mark = self.time_marks[start:end] if self.time_marks is not None else None
        lmask = self.cell_mask[start:end] if self.cell_mask is not None else None
        return x, x_tgt, y_next, lmask, x_mark


def collate_window_batch(
    batch: Sequence[tuple[np.ndarray, ...]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    xs, x_tgts, y_nexts, lmasks, x_marks = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0))
    x_tgt = torch.from_numpy(np.stack(x_tgts, axis=0))
    y_next = torch.from_numpy(np.stack(y_nexts, axis=0))
    lmask = None
    if lmasks[0] is not None:
        lmask = torch.from_numpy(np.stack(lmasks, axis=0))
    x_mark = None
    if x_marks[0] is not None:
        x_mark = torch.from_numpy(np.stack(x_marks, axis=0))
    return x, x_tgt, y_next, lmask, x_mark


def all_window_starts(n_rows: int, window: int, *, pred_horizon: int = 0) -> np.ndarray:
    """Start indices for every valid sliding window on a length-``n_rows`` series."""
    if pred_horizon > 0:
        last = n_rows - window - pred_horizon + 1
    else:
        last = n_rows - window + 1
    if last <= 0:
        return np.array([], dtype=np.int64)
    return np.arange(last, dtype=np.int64)


def lazy_window_loader(
    values: np.ndarray,
    starts: np.ndarray,
    target_idx: np.ndarray,
    window: int,
    batch_size: int,
    shuffle: bool,
    *,
    time_marks: np.ndarray | None = None,
    cell_mask: np.ndarray | None = None,
    pred_horizon: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    ds = LazyWindowDataset(
        values=values,
        starts=starts,
        target_idx=target_idx,
        window=window,
        time_marks=time_marks,
        cell_mask=cell_mask,
        pred_horizon=pred_horizon,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        collate_fn=collate_window_batch,
    )
