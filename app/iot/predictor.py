"""Predictor node: subscribe to a station's 5-minute feed, forecast, compare with the facts.

For every closed 15-minute bucket it
  1. publishes an evaluation (actual vs. the forecasts made 15 min, 1 h, 4 h
     and 24 h earlier, weekly baseline, running MAE, anomaly flag),
  2. runs TimesNet on the last ``window`` buckets and publishes the next 24 h
     as a retained forecast.

MQTT callbacks only enqueue messages; one worker thread does the model work,
so a slow forecast never blocks keep-alives or acknowledgements.

Environment:
    STATION_ID, MODEL_DIR (TimesNet experiment folder), OUTPUT_DIR (JSONL logs)
    ANOMALY_Z (default 6: a 15-min-ahead miss over 6x the running MAE), MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD
"""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import threading
import time
from pathlib import Path

from app.iot import topics
from app.iot.forecaster import LiveForecaster
from app.iot.mqtt_common import connect_client
from app.iot.stream_state import CHANNELS, StreamState, _local_iso, now_iso

LOG = logging.getLogger("predictor")


class JsonlLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self.file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.file.flush()


class Predictor:
    def __init__(self, client, station_id: str, forecaster: LiveForecaster, out_dir: Path,
                 anomaly_z: float):
        self.client = client
        self.station_id = station_id
        self.fc = forecaster
        self.out_dir = out_dir
        self.state_path = out_dir / f"state_{station_id}.pkl"
        kwargs = {"freq_minutes": int(self.fc.step / self._minute()), "anomaly_z": anomaly_z}
        self.state = StreamState(**kwargs)
        if self.state_path.exists() and os.getenv("RESUME_STATE", "1") == "1":
            try:
                self.state = StreamState.load(self.state_path, **kwargs)
                LOG.info("resumed state: %d buckets, last measurement %s",
                         len(self.state.history), self.state.last_ts)
            except Exception:
                LOG.exception("could not load %s; starting empty", self.state_path)
        self.eval_log = JsonlLog(out_dir / f"evaluations_{station_id}.jsonl")
        self.forecast_log = JsonlLog(out_dir / f"forecasts_{station_id}.jsonl")
        self.latency_ms: list[float] = []
        self.inference_ms: list[float] = []
        self.anomalies = 0
        self.invalid = 0
        self.errors = 0

    @staticmethod
    def _minute():
        import pandas as pd

        return pd.Timedelta(minutes=1)

    def handle(self, payload: bytes, received_ms: float) -> None:
        try:
            data = json.loads(payload)
            if data.get("station_id") != self.station_id or data.get("schema_version") != 1:
                raise ValueError("wrong station or schema")
            closed = self.state.ingest(data)
        except (ValueError, KeyError, TypeError) as exc:
            self.invalid += 1
            LOG.warning("invalid measurement discarded: %s", exc)
            return
        if "sent_at_ms" in data:
            self.latency_ms.append(received_ms - data["sent_at_ms"])
        for bucket in closed:
            try:
                self.on_bucket(bucket)
            except Exception:
                # One bad bucket must not take the node down (and into a restart loop).
                self.errors += 1
                LOG.exception("processing bucket %s failed", bucket)

    def on_bucket(self, bucket) -> None:
        result = self.state.evaluate(bucket)
        result["station_id"] = self.station_id
        result["evaluated_at"] = now_iso()
        if result["anomaly"] and result["anomaly"]["flag"]:
            self.anomalies += 1
            LOG.warning("ANOMALY %s z=%s actual=%s pred15=%s", result["bucket_start"],
                        result["anomaly"]["z"], result["actual"], result["predictions"].get("15min"))
        if result["predictions"] or result["active"]:
            self.publish(topics.evaluation(self.station_id), result)
            self.eval_log.write(result)

        window = self.state.window(self.fc.info.window)
        if window is None:
            if len(self.state.history) % 48 == 0:
                LOG.info("warming up: %d/%d buckets", len(self.state.history), self.fc.info.window)
            return
        if self.state.active_count(self.fc.info.window) < self.fc.info.window // 2:
            LOG.warning("skip forecast at %s: under half of the window is observed", bucket)
            return
        t0 = time.perf_counter()
        forecast = self.fc.forecast(window)
        self.inference_ms.append((time.perf_counter() - t0) * 1000)
        self.state.add_forecast(forecast)
        message = {
            "station_id": self.station_id,
            "model": self.fc.info.path.name,
            "issued_at": now_iso(),
            "data_until": _local_iso(bucket + self.fc.step),
            "step_minutes": int(self.fc.step / self._minute()),
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

    def publish(self, topic: str, message: dict, retain: bool = False) -> None:
        self.client.publish(topic, json.dumps(message, separators=(",", ":")), qos=1, retain=retain)

    def summary(self) -> dict:
        def stats(values):
            if not values:
                return None
            v = sorted(values)
            return {"n": len(v), "mean": round(sum(v) / len(v), 2), "p50": round(v[len(v) // 2], 2),
                    "p95": round(v[int(len(v) * 0.95) - 1 if len(v) > 1 else 0], 2), "max": round(v[-1], 2)}

        return {
            "station_id": self.station_id,
            "model": str(self.fc.info.path),
            "messages_received": self.state.received,
            "duplicates_ignored": self.state.duplicates,
            "stream_resets": self.state.resets,
            "invalid": self.invalid,
            "processing_errors": self.errors,
            "buckets": len(self.state.history),
            "forecasts_made": len(self.inference_ms),
            "anomalies": self.anomalies,
            "mqtt_latency_ms": stats(self.latency_ms),
            "inference_ms": stats(self.inference_ms),
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
    status_topic = topics.predictor_status()
    offline = json.dumps({"state": "offline"})
    # Fixed client ID + persistent session: the broker queues QoS 1 messages
    # while the predictor restarts, so no measurement is lost.
    # PUBACK is sent only after the worker has processed a message, so whatever
    # sits in the inbox when the container dies is redelivered by the broker.
    online = json.dumps({
        "state": "online", "station_id": station_id, "model": forecaster.info.path.name,
        "since": now_iso(),
    })
    client = connect_client(
        f"predictor-{station_id}",
        will=(status_topic, offline),
        birth=(status_topic, online),
        subscriptions=[(topics.measurement(station_id), 1)],
        clean_session=False,
        manual_ack=True,
        on_message=lambda c, u, m: inbox.put((m.payload, time.time() * 1000, m.mid, m.qos)),
    )
    predictor = Predictor(client, station_id, forecaster, out_dir,
                          anomaly_z=float(os.getenv("ANOMALY_Z", "6")))

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    summary_path = out_dir / f"summary_{station_id}.json"
    last_summary = 0.0
    try:
        while not stop.is_set():
            try:
                payload, received, mid, qos = inbox.get(timeout=1)
                predictor.handle(payload, received)
                client.ack(mid, qos)
            except queue.Empty:
                pass
            if time.time() - last_summary > 10:
                summary_path.write_text(json.dumps(predictor.summary(), indent=2), encoding="utf-8")
                predictor.state.save(predictor.state_path)
                last_summary = time.time()
    finally:
        summary_path.write_text(json.dumps(predictor.summary(), indent=2), encoding="utf-8")
        predictor.state.save(predictor.state_path)
        client.publish(status_topic, offline, qos=1, retain=True).wait_for_publish(timeout=5)
        client.loop_stop()
        client.disconnect()
        LOG.info("stopped; summary in %s", summary_path)


if __name__ == "__main__":
    main()
