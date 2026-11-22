"""Predictor node: subscribe to a station's 5-minute feed, forecast, compare with the facts.

For every closed 15-minute bucket it
  1. publishes an evaluation (actual vs. the forecasts made 15 min, 1 h, 4 h
     and 24 h earlier, weekly baseline, deviation flag),
  2. runs TimesNet on the last ``window`` buckets and publishes the next 24 h
     as a retained forecast.
It also keeps retained ``history`` (last two days of evaluations) and
``metrics`` topics so a dashboard that opens mid-stream can draw at once.

MQTT callbacks only enqueue messages; the main thread does the model work,
so a slow forecast never blocks keep-alives. A measurement is acknowledged
(PUBACK) only after it has been processed.

Commands on ``traffic/pems/predictor/control``:
    {"cmd": "outage", "seconds": 30}   leave the broker for a while; the broker
                                        queues the feed in the persistent session

Environment:
    STATION_ID, MODEL_DIR (TimesNet experiment folder), OUTPUT_DIR (JSONL logs)
    ANOMALY_Z (default 6), MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD
"""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import threading
import time
from collections import deque
from pathlib import Path

import pandas as pd

from app.iot import topics
from app.iot.forecaster import LiveForecaster
from app.iot.mqtt_common import connect_client, go_offline, go_online
from app.iot.stream_state import CHANNELS, REPORTED_LEADS, StreamState, _local_iso, now_iso

LOG = logging.getLogger("predictor")
# Forecasts during the warm-up burst are only needed where they can still be
# scored against live data: the last day before the session start.
WARMUP_FORECAST_SPAN = pd.Timedelta(hours=26)
PUBLISH_EVERY_S = 1.0


class JsonlLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self.file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.file.flush()


def _vec(values: dict | None) -> list | None:
    return None if values is None else [values[c] for c in CHANNELS]


def _stats(values) -> dict | None:
    if not values:
        return None
    v = sorted(values)
    return {"n": len(v), "mean": round(sum(v) / len(v), 2), "p50": round(v[len(v) // 2], 2),
            "p95": round(v[max(0, int(len(v) * 0.95) - 1)], 2), "max": round(v[-1], 2)}


class Predictor:
    def __init__(self, client, station_id: str, forecaster: LiveForecaster, out_dir: Path,
                 anomaly_z: float):
        self.client = client
        self.station_id = station_id
        self.fc = forecaster
        self.out_dir = out_dir
        self.state_path = out_dir / f"state_{station_id}.pkl"
        kwargs = {"freq_minutes": int(self.fc.step / pd.Timedelta(minutes=1)), "anomaly_z": anomaly_z}
        self.state = StreamState(**kwargs)
        if self.state_path.exists() and os.getenv("RESUME_STATE", "1") == "1":
            try:
                self.state = StreamState.load(self.state_path, **kwargs)
                LOG.info("resumed state: session %s, %d buckets, last measurement %s",
                         self.state.session, len(self.state.history), self.state.last_ts)
            except Exception:
                LOG.exception("could not load %s; starting empty", self.state_path)
        self.eval_log = JsonlLog(out_dir / f"evaluations_{station_id}.jsonl")
        self.forecast_log = JsonlLog(out_dir / f"forecasts_{station_id}.jsonl")
        self.session_start: pd.Timestamp | None = None
        self.new_session()
        self.dirty = True
        self.last_publish = 0.0

    def new_session(self) -> None:
        self.latency_ms: deque = deque(maxlen=2000)
        self.inference_ms: deque = deque(maxlen=2000)
        self.forecasts_made = 0
        self.anomalies = 0
        self.invalid = 0
        self.errors = 0
        self.last_bucket: pd.Timestamp | None = None

    # -- stream ---------------------------------------------------------
    def handle(self, payload: bytes, received_ms: float) -> None:
        try:
            data = json.loads(payload)
            if data.get("station_id") != self.station_id or data.get("schema_version") != 1:
                raise ValueError("wrong station or schema")
            session_before = self.state.session
            closed = self.state.ingest(data)
        except (ValueError, KeyError, TypeError) as exc:
            self.invalid += 1
            LOG.warning("invalid measurement discarded: %s", exc)
            return
        if self.state.session != session_before:
            LOG.info("new replay session %s", self.state.session)
            self.new_session()
            self.dirty = True
        if data.get("session_start"):
            self.session_start = pd.Timestamp(data["session_start"])
        if "sent_at_ms" in data:
            self.latency_ms.append(received_ms - data["sent_at_ms"])
        for bucket in closed:
            try:
                self.on_bucket(bucket)
            except Exception:
                # One bad bucket must not take the node down (and into a restart loop).
                self.errors += 1
                LOG.exception("processing bucket %s failed", bucket)

    def on_bucket(self, bucket: pd.Timestamp) -> None:
        result = self.state.evaluate(bucket)
        result["station_id"] = self.station_id
        result["session"] = self.state.session
        result["evaluated_at"] = now_iso()
        if result["anomaly"] and result["anomaly"]["flag"]:
            self.anomalies += 1
        self.state.recent.append({
            "t": result["bucket_start"],
            "a": _vec(result["actual"]),
            "w": _vec(result["weekly_baseline"]),
            "p": {k: _vec(v) for k, v in result["predictions"].items()},
            "x": bool(result["anomaly"] and result["anomaly"]["flag"]),
        })
        self.last_bucket = bucket
        self.dirty = True
        self.publish(topics.evaluation(self.station_id), result)
        self.eval_log.write(result)

        if self.session_start is not None and bucket + self.fc.step < self.session_start - WARMUP_FORECAST_SPAN:
            return
        window = self.state.window(self.fc.info.window)
        if window is None:
            return
        if self.state.active_count(self.fc.info.window) < self.fc.info.window // 2:
            LOG.warning("skip forecast at %s: under half of the window is observed", bucket)
            return
        t0 = time.perf_counter()
        forecast = self.fc.forecast(window)
        self.inference_ms.append((time.perf_counter() - t0) * 1000)
        self.forecasts_made += 1
        self.state.add_forecast(forecast)
        message = {
            "station_id": self.station_id,
            "session": self.state.session,
            "model": self.fc.info.path.name,
            "issued_at": now_iso(),
            "data_until": _local_iso(bucket + self.fc.step),
            "step_minutes": int(self.fc.step / pd.Timedelta(minutes=1)),
            "units": {"flow": "veh/15min", "speed": "mph", "occupancy": "fraction"},
            "inference_ms": round(self.inference_ms[-1], 1),
            "horizon": [
                {"ts": _local_iso(ts), **{c: round(float(row[c]), 4) for c in CHANNELS}}
                for ts, row in forecast.iterrows()
            ],
        }
        self.publish(topics.forecast(self.station_id), message, retain=True)
        self.forecast_log.write({k: v for k, v in message.items() if k != "horizon"}
                                | {"horizon": message["horizon"][:4]})

    # -- outputs --------------------------------------------------------
    def publish(self, topic: str, message: dict, retain: bool = False) -> None:
        self.client.publish(topic, json.dumps(message, separators=(",", ":")), qos=1, retain=retain)

    def publish_dashboard(self, force: bool = False) -> None:
        """Retained history + metrics, at most once a second (bursts arrive much faster)."""
        if not self.dirty or (not force and time.monotonic() - self.last_publish < PUBLISH_EVERY_S):
            return
        self.dirty = False
        self.last_publish = time.monotonic()
        self.publish(topics.history(self.station_id), {
            "session": self.state.session,
            "channels": list(CHANNELS),
            "leads": list(REPORTED_LEADS),
            "buckets": list(self.state.recent),
        }, retain=True)
        self.publish(topics.metrics(self.station_id), self.summary(), retain=True)

    def summary(self) -> dict:
        return {
            "station_id": self.station_id,
            "session": self.state.session,
            "model": self.fc.info.path.name,
            "last_bucket": _local_iso(self.last_bucket) if self.last_bucket is not None else None,
            "messages_received": self.state.received,
            "duplicates_ignored": self.state.duplicates,
            "invalid": self.invalid,
            "processing_errors": self.errors,
            "buckets": len(self.state.history),
            "forecasts_made": self.forecasts_made,
            "deviations": self.anomalies,
            "mqtt_latency_ms": _stats(self.latency_ms),
            "inference_ms": _stats(self.inference_ms),
            "mae_by_lead": self.state.errors.summary(),
            "written_at": now_iso(),
        }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    station_id = os.environ["STATION_ID"].strip()
    out_dir = Path(os.getenv("OUTPUT_DIR", "artifacts/iot"))
    forecaster = LiveForecaster(os.environ["MODEL_DIR"])
    LOG.info("model %s: freq=%s window=%d horizon=%d", forecaster.info.path.name,
             forecaster.info.freq, forecaster.info.window, forecaster.info.pred_len)

    inbox: queue.Queue = queue.Queue()
    commands: queue.Queue = queue.Queue()
    status_topic = topics.predictor_status()
    offline = json.dumps({"state": "offline"})
    control_topic = topics.predictor_control()

    def on_message(c, _u, m):
        if m.topic == control_topic:
            # Ack at once: an unacknowledged command would be redelivered after
            # the simulated outage and start it again, forever.
            c.ack(m.mid, m.qos)
            commands.put(m.payload)
        else:
            inbox.put((m.payload, time.time() * 1000, m.mid, m.qos))

    # Fixed client ID + persistent session: the broker queues QoS 1 messages
    # while the predictor is away. PUBACK is sent only after processing, so
    # whatever sits in the inbox when the node dies is redelivered.
    client = connect_client(
        f"predictor-{station_id}",
        will=(status_topic, offline),
        birth=(status_topic, lambda: json.dumps({
            "state": "online", "station_id": station_id, "model": forecaster.info.path.name,
            "since": now_iso(),
        })),
        subscriptions=[(topics.measurement(station_id), 1), (control_topic, 1)],
        clean_session=False,
        manual_ack=True,
        on_message=on_message,
    )
    predictor = Predictor(client, station_id, forecaster, out_dir,
                          anomaly_z=float(os.getenv("ANOMALY_Z", "6")))

    def simulate_outage(raw: bytes) -> None:
        try:
            cmd = json.loads(raw)
            if cmd.get("cmd") != "outage":
                raise ValueError(f"unknown command {cmd.get('cmd')!r}")
            seconds = min(max(float(cmd.get("seconds", 30)), 3.0), 600.0)
        except (ValueError, TypeError, AttributeError) as exc:
            LOG.warning("bad control message %r: %s", raw[:200], exc)
            return
        until_ms = int((time.time() + seconds) * 1000)
        client.publish(status_topic, json.dumps({
            "state": "offline", "reason": "simulated network outage", "until_ms": until_ms,
        }), qos=1, retain=True).wait_for_publish(timeout=5)
        LOG.warning("simulated outage for %.0fs", seconds)
        go_offline(client)
        # Unacknowledged messages stay with the broker and come back after reconnect.
        while not inbox.empty():
            inbox.get_nowait()
        time.sleep(seconds)
        go_online(client)
        LOG.info("back online; the broker now delivers what it queued")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    summary_path = out_dir / f"summary_{station_id}.json"
    last_save = 0.0
    try:
        while not stop.is_set():
            try:
                simulate_outage(commands.get_nowait())
            except queue.Empty:
                pass
            try:
                payload, received, mid, qos = inbox.get(timeout=0.2)
                predictor.handle(payload, received)
                client.ack(mid, qos)
            except queue.Empty:
                pass
            predictor.publish_dashboard(force=inbox.empty())
            if time.time() - last_save > 10:
                summary_path.write_text(json.dumps(predictor.summary(), indent=2), encoding="utf-8")
                predictor.state.save(predictor.state_path)
                last_save = time.time()
    finally:
        summary_path.write_text(json.dumps(predictor.summary(), indent=2), encoding="utf-8")
        predictor.state.save(predictor.state_path)
        # A clean stop says so; a crash leaves the reason-less Last Will instead.
        stopped = json.dumps({"state": "offline", "reason": "stopped"})
        client.publish(status_topic, stopped, qos=1, retain=True).wait_for_publish(timeout=5)
        client.loop_stop()
        client.disconnect()
        LOG.info("stopped; summary in %s", summary_path)


if __name__ == "__main__":
    main()
