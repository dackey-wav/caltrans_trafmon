"""Run a trained TimesNet experiment on a live window of 15-minute buckets.

The inputs are assembled exactly like in training (models/TimesNet): robust
scaled targets, gaps forward-filled, clock marks plus the calendar flags the
experiment kept, divided by the stored mark scale. Only ``absolute`` target
experiments are supported; ``weekly_residual`` would also need last week's
values for the whole horizon.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from models.TimesNet.train_timesnet import FeaturePlan, basic_time_marks, load_model
from models.utils.constants import step_minutes
from models.utils.multivariate_data_prep import _calendar_columns, load_scaler

CALENDAR_PAD = pd.Timedelta(days=8)


@dataclass(frozen=True)
class ExperimentInfo:
    path: Path
    freq: str
    window: int
    pred_len: int
    channels: list[str]


class LiveForecaster:
    def __init__(self, experiment_dir: Path | str, *, threads: int = 2):
        self.dir = Path(experiment_dir)
        log = json.loads((self.dir / "train_log.json").read_text(encoding="utf-8"))
        hp = log["hparams"]
        self.plan = FeaturePlan.from_dict(log["feature_plan"])
        if self.plan.target_mode != "absolute":
            raise ValueError(f"{self.dir.name}: only absolute-target experiments are supported")
        self.scaler, columns = load_scaler(self.dir / "scaler.json")
        if columns != self.plan.target_columns:
            raise ValueError("scaler columns do not match the feature plan")
        torch.set_num_threads(threads)
        self.model = load_model(self.dir / "model.pt").eval()
        self.info = ExperimentInfo(
            path=self.dir,
            freq=hp["freq"],
            window=int(hp["window"]),
            pred_len=int(hp["pred_len"]),
            channels=self.plan.channels,
        )
        self.step = pd.Timedelta(minutes=step_minutes(self.info.freq))

    def marks(self, index: pd.DatetimeIndex) -> np.ndarray:
        marks = basic_time_marks(index, self.plan.basic_marks)
        if not self.plan.mark_columns:
            return marks
        # Holiday neighbours (pre/post, bridge, long weekend) look at adjacent
        # days, so compute the calendar on a padded range and cut it back.
        padded = pd.date_range(index[0] - CALENDAR_PAD, index[-1] + CALENDAR_PAD, freq=self.step)
        ctx = _calendar_columns(padded, self.plan.mark_columns).reindex(index)
        missing = set(self.plan.mark_columns) - set(ctx.columns)
        if missing:
            raise ValueError(f"cannot rebuild calendar columns {sorted(missing)}")
        ctx = ctx[self.plan.mark_columns].to_numpy(dtype=np.float32)
        ctx = np.nan_to_num(ctx / np.asarray(self.plan.mark_scale, dtype=np.float32))
        return np.concatenate([marks, ctx], axis=1).astype(np.float32)

    def forecast(self, history: pd.DataFrame) -> pd.DataFrame:
        """``history``: UTC index on the model grid, one column per channel, NaN = missing.

        Returns ``pred_len`` rows starting one step after the last history row.
        """
        window = self.info.window
        if len(history) < window:
            raise ValueError(f"need {window} buckets of history, have {len(history)}")
        hist = history[self.info.channels].iloc[-window:]
        expected = pd.date_range(hist.index[0], periods=window, freq=self.step)
        if not hist.index.equals(expected):
            raise ValueError("history is not a contiguous grid")
        values = hist.ffill().bfill().to_numpy(dtype=np.float32)
        med, sc = self.scaler["median"], self.scaler["scale"]
        values = np.where(np.isfinite(values), values, med)
        x = (values - med) / sc

        future = pd.date_range(hist.index[-1] + self.step, periods=self.info.pred_len, freq=self.step)
        marks = self.marks(hist.index.append(future))
        with torch.no_grad():
            out = self.model(torch.from_numpy(x[None]), torch.from_numpy(marks[None]))
        pred = np.maximum(out[0].float().numpy() * sc + med, 0.0)
        return pd.DataFrame(pred, index=future, columns=self.info.channels)
