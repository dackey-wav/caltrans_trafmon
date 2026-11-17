"""Pure stream logic of the predictor node, kept free of MQTT and torch for testing.

5-minute measurements arrive in order (QoS 1 may redeliver). They are folded
into 15-minute buckets the same way the training query does it (sum flow,
average speed and occupancy, active only when every 5-minute row is 100 %
observed). Each closed bucket is compared with every stored forecast that
covers it.
"""

from __future__ import annotations

import math
import pickle
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PACIFIC = ZoneInfo("America/Los_Angeles")
STREAM_RESET_AFTER = pd.Timedelta(hours=1)
RECENT_BUCKETS = 192  # two days of 15-minute buckets
CHANNELS = ("flow", "speed", "occupancy")
SOURCE = {"flow": "flow_veh_5min", "speed": "speed_mph", "occupancy": "occupancy_fraction"}
LEAD_BUCKETS = (("0-1h", 1, 4), ("1-4h", 5, 16), ("4-24h", 17, 96))
REPORTED_LEADS = {"15min": 1, "1h": 4, "4h": 16, "24h": 96}


def parse_message_time(data: dict) -> pd.Timestamp:
    if data.get("ts_utc"):
        return pd.Timestamp(data["ts_utc"]).tz_convert(UTC)
    local = pd.Timestamp(data["timestamp"])
    return local.tz_localize(PACIFIC, ambiguous=True, nonexistent="shift_forward").tz_convert(UTC)


@dataclass
class Bucket:
    start: pd.Timestamp
    rows: dict[pd.Timestamp, dict] = field(default_factory=dict)

    def add(self, ts: pd.Timestamp, data: dict) -> None:
        self.rows[ts] = data

    def close(self, slots: int) -> tuple[dict[str, float], bool]:
        rows = list(self.rows.values())
        complete = len(rows) == slots and all(
            r.get("observed_percent") == 100 and all(_finite(r.get(SOURCE[c])) for c in CHANNELS)
            for r in rows
        )
        if not complete:
            return {c: math.nan for c in CHANNELS}, False
        return {
            "flow": float(sum(r[SOURCE["flow"]] for r in rows)),
            "speed": float(np.mean([r[SOURCE["speed"]] for r in rows])),
            "occupancy": float(np.mean([r[SOURCE["occupancy"]] for r in rows])),
        }, True


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


class RunningError:
    """MAE of the model and of the weekly naive baseline per lead bucket and channel."""

    def __init__(self):
        self.abs = {(b, c): 0.0 for b, *_ in LEAD_BUCKETS for c in CHANNELS}
        self.weekly_abs = dict(self.abs)
        self.n = {(b, c): 0 for b, *_ in LEAD_BUCKETS for c in CHANNELS}
        self.lead1_abs = {c: 0.0 for c in CHANNELS}
        self.lead1_n = 0

    @staticmethod
    def bucket_of(lead: int) -> str:
        for name, lo, hi in LEAD_BUCKETS:
            if lo <= lead <= hi:
                return name
        raise ValueError(lead)

    def add(self, lead: int, pred: dict, actual: dict, weekly: dict | None) -> None:
        if weekly is None:
            return  # score model and baseline on the same cells only
        name = self.bucket_of(lead)
        for c in CHANNELS:
            self.abs[(name, c)] += abs(pred[c] - actual[c])
            self.weekly_abs[(name, c)] += abs(weekly[c] - actual[c])
            self.n[(name, c)] += 1

    def add_lead1(self, pred: dict, actual: dict) -> None:
        for c in CHANNELS:
            self.lead1_abs[c] += abs(pred[c] - actual[c])
        self.lead1_n += 1

    def lead1_mae(self, channel: str) -> float | None:
        return self.lead1_abs[channel] / self.lead1_n if self.lead1_n else None

    def summary(self) -> dict:
        out = {}
        for name, *_ in LEAD_BUCKETS:
            row = {}
            for c in CHANNELS:
                n = self.n[(name, c)]
                if n:
                    mae = self.abs[(name, c)] / n
                    base = self.weekly_abs[(name, c)] / n
                    row[c] = {
                        "n": n,
                        "mae": round(mae, 4),
                        "weekly_mae": round(base, 4),
                        "skill_vs_weekly": round(1 - mae / base, 4) if base else None,
                    }
            out[name] = row
        return out


class StreamState:
    """History of closed buckets, open bucket, stored forecasts, running errors."""

    def __init__(self, *, freq_minutes: int = 15, sample_minutes: int = 5, keep_days: int = 8,
                 anomaly_z: float = 6.0, anomaly_min_n: int = 96):
        self.step = pd.Timedelta(minutes=freq_minutes)
        self.slots = freq_minutes // sample_minutes
        self.keep = int(pd.Timedelta(days=keep_days) / self.step)
        self.week = int(pd.Timedelta(days=7) / self.step)
        self.anomaly_z = anomaly_z
        self.anomaly_min_n = anomaly_min_n
        self.duplicates = 0
        self.received = 0
        self.resets = 0
        self.session: str | None = None
        self.reset()

    def reset(self) -> None:
        self.history: OrderedDict[pd.Timestamp, tuple[dict, bool]] = OrderedDict()
        self.open: Bucket | None = None
        self.last_ts: pd.Timestamp | None = None
        self.forecasts: OrderedDict[pd.Timestamp, pd.DataFrame] = OrderedDict()
        self.errors = RunningError()
        # Compact evaluated buckets for dashboards; part of the snapshot so a
        # restarted predictor still serves the full two days.
        self.recent: deque = deque(maxlen=RECENT_BUCKETS)

    # -- persistence ----------------------------------------------------
    def save(self, path: Path) -> None:
        """Snapshot so a restarted predictor does not need two days of warm-up."""
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self.__dict__, f)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, **kwargs) -> StreamState:
        state = cls(**kwargs)
        with path.open("rb") as f:
            saved = pickle.load(f)
        if saved.get("step") != state.step:
            raise ValueError("snapshot was made for another bucket size")
        state.__dict__.update({k: v for k, v in saved.items() if k not in ("anomaly_z", "anomaly_min_n")})
        return state

    # -- ingest ---------------------------------------------------------
    def ingest(self, data: dict) -> list[pd.Timestamp]:
        """Add one 5-minute message; return the buckets it closed (oldest first)."""
        ts = parse_message_time(data)
        session = data.get("session")
        if session and session != self.session:
            # A new replay session (dashboard reset): start from scratch.
            if self.session is not None:
                self.resets += 1
            self.reset()
            self.session = session
        elif self.last_ts is not None and ts < self.last_ts - STREAM_RESET_AFTER:
            # The source started over (e.g. the replay was restarted), not a late retry.
            self.reset()
            self.resets += 1
        if self.last_ts is not None and ts <= self.last_ts:
            self.duplicates += 1
            return []
        self.received += 1
        self.last_ts = ts
        start = ts.floor(self.step)
        closed: list[pd.Timestamp] = []
        if self.open is not None and start > self.open.start:
            closed += self._close_until(start)
        if self.open is None:
            closed += self._fill_gaps_before(start)
            self.open = Bucket(start)
        self.open.add(ts, data)
        if ts + pd.Timedelta(minutes=5) >= start + self.step:  # last slot of the bucket
            closed += self._close_until(start + self.step)
        return closed

    def _close_until(self, next_start: pd.Timestamp) -> list[pd.Timestamp]:
        closed = []
        if self.open is not None:
            values, active = self.open.close(self.slots)
            self._store(self.open.start, values, active)
            closed.append(self.open.start)
            self.open = None
            closed += self._fill_gaps_before(next_start)
        return closed

    def _fill_gaps_before(self, start: pd.Timestamp) -> list[pd.Timestamp]:
        """Store whole buckets that got no message at all, so the grid stays contiguous."""
        if not self.history:
            return []
        gap = next(reversed(self.history)) + self.step
        if start - gap > self.keep * self.step:  # long outage: older history is useless
            self.history.clear()
            return []
        closed = []
        while gap < start:
            self._store(gap, {c: math.nan for c in CHANNELS}, False)
            closed.append(gap)
            gap += self.step
        return closed

    def _store(self, start: pd.Timestamp, values: dict, active: bool) -> None:
        self.history[start] = (values, active)
        while len(self.history) > self.keep:
            self.history.popitem(last=False)

    # -- model window ---------------------------------------------------
    def window(self, length: int) -> pd.DataFrame | None:
        if len(self.history) < length:
            return None
        items = list(self.history.items())[-length:]
        frame = pd.DataFrame(
            [values for _, (values, _) in items],
            index=pd.DatetimeIndex([k for k, _ in items]),
            columns=list(CHANNELS),
        )
        return frame

    def active_count(self, length: int) -> int:
        return sum(active for _, active in list(self.history.values())[-length:])

    def add_forecast(self, forecast: pd.DataFrame) -> None:
        self.forecasts[forecast.index[0]] = forecast
        horizon_start = forecast.index[0] - len(forecast) * self.step
        for origin in list(self.forecasts):
            if origin < horizon_start:
                del self.forecasts[origin]

    # -- evaluation -----------------------------------------------------
    def evaluate(self, start: pd.Timestamp) -> dict:
        values, active = self.history[start]
        week_key = start - self.week * self.step
        weekly = None
        if week_key in self.history and self.history[week_key][1]:
            weekly = self.history[week_key][0]
        preds: dict[int, dict] = {}
        for first, fc in self.forecasts.items():
            if start in fc.index:
                lead = int((start - first) / self.step) + 1
                preds[lead] = {c: float(fc.at[start, c]) for c in CHANNELS}

        result = {
            "bucket_start": _local_iso(start),
            "bucket_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "active": active,
            "actual": {c: _round(values[c]) for c in CHANNELS} if active else None,
            "weekly_baseline": {c: _round(weekly[c]) for c in CHANNELS} if weekly else None,
            "predictions": {
                name: {c: _round(preds[lead][c]) for c in CHANNELS}
                for name, lead in REPORTED_LEADS.items() if lead in preds
            },
            "n_forecasts_covering": len(preds),
            "anomaly": None,
        }
        if not active:
            return result
        for lead, pred in preds.items():
            self.errors.add(lead, pred, values, weekly)
        if 1 in preds:
            result["anomaly"] = self._anomaly(preds[1], values)
            self.errors.add_lead1(preds[1], values)
        return result

    def _anomaly(self, pred: dict, actual: dict) -> dict:
        """Flag a 15-minute-ahead miss far larger than this stream's usual error."""
        out = {"flag": False, "z": {}}
        if self.errors.lead1_n < self.anomaly_min_n:
            out["warming_up"] = True
            return out
        for c in CHANNELS:
            mae = self.errors.lead1_mae(c)
            z = abs(actual[c] - pred[c]) / mae if mae else 0.0
            out["z"][c] = round(z, 2)
            if z > self.anomaly_z:
                out["flag"] = True
        return out


def _round(value: float) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), 4)


def _local_iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert(PACIFIC).isoformat()


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
