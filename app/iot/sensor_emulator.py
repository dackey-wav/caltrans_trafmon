"""Sensor node: replay a downloaded PeMS station CSV as a live 5-minute MQTT feed.

The CSV is the ``processed/observations.csv(.gz)`` file written by
pems-data-downloader (``timestamp,station_id,speed_mph,flow_veh_5min,
occupancy_fraction,observed_percent,...``). Rows are published in time order on
``traffic/pems/stations/<id>/5min`` with QoS 1, using the same JSON schema as
the live FTP publisher in ``app.live``. Simulated time runs ``REPLAY_SPEED``
times faster than wall time.

Environment:
    STATION_ID              detector ID to replay (required)
    REPLAY_CSV              path to the CSV or CSV.GZ file (required)
    REPLAY_START            first local date/time streamed at normal pace
    REPLAY_END              last local date/time (optional)
    REPLAY_WARMUP_HOURS     history before REPLAY_START sent as a fast burst (default 48)
    REPLAY_SPEED            simulated seconds per wall second (default 300: 5 min -> 1 s)
    REPLAY_DROP_RATE        probability of losing a message, to test gap handling (default 0)
    REPLAY_WAIT_PREDICTOR   seconds to wait for the predictor to come online (default 60, 0 = no wait)
    MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from app.iot import topics
from app.iot.mqtt_common import connect_client

LOG = logging.getLogger("sensor")
PACIFIC = ZoneInfo("America/Los_Angeles")
VALUE_COLUMNS = ("speed_mph", "flow_veh_5min", "occupancy_fraction", "observed_percent")


@dataclass(frozen=True)
class Row:
    ts: datetime  # aware, UTC
    values: dict[str, float | None]

    def payload(self, station_id: str, seq: int) -> dict:
        local = self.ts.astimezone(PACIFIC)
        return {
            "schema_version": 1,
            "source": "pems",
            "origin": "replay",
            "station_id": station_id,
            "timestamp": local.strftime("%Y-%m-%d %H:%M:%S"),
            "ts_utc": self.ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seq": seq,
            "sent_at_ms": int(time.time() * 1000),
            **self.values,
        }


def _number(text: str | None) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    value = float(text)
    return value if value == value else None  # NaN -> None


def _parse_ts(text: str) -> datetime:
    ts = datetime.fromisoformat(text.strip())
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=PACIFIC)
    return ts.astimezone(UTC)


def _parse_local(text: str | None) -> datetime | None:
    if not text:
        return None
    ts = datetime.fromisoformat(text)
    return (ts if ts.tzinfo else ts.replace(tzinfo=PACIFIC)).astimezone(UTC)


def load_rows(path: Path, station_id: str) -> list[Row]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    missing = {"timestamp", "station_id", *VALUE_COLUMNS} - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{path} is not a processed PeMS observations CSV; missing {sorted(missing)}")
    rows = [
        Row(_parse_ts(r["timestamp"]), {c: _number(r[c]) for c in VALUE_COLUMNS})
        for r in reader
        if r["station_id"].strip() == station_id
    ]
    if not rows:
        raise ValueError(f"station {station_id} not found in {path}")
    rows.sort(key=lambda r: r.ts)
    repeats = sum(a.ts == b.ts for a, b in zip(rows, rows[1:], strict=False))
    if repeats:
        # Kept on purpose: PeMS archives occasionally repeat rows, and the
        # receiving node has to cope with that just like with QoS 1 redelivery.
        LOG.warning("%d repeated timestamps in the source; they are sent as-is", repeats)
    return rows


class Replay:
    """Publishes rows; a control topic can pause, resume or change the speed."""

    def __init__(self, client, station_id: str, speed: float, drop_rate: float, seed: int = 7):
        self.client = client
        self.station_id = station_id
        self.speed = speed
        self.drop_rate = drop_rate
        self.random = random.Random(seed)
        self.running = threading.Event()
        self.running.set()
        self.seq = 0
        self.sent = 0
        self.dropped = 0

    def on_control(self, payload: bytes) -> None:
        try:
            cmd = json.loads(payload)
            name = cmd.get("cmd")
            if name == "pause":
                self.running.clear()
            elif name == "resume":
                self.running.set()
            elif name == "speed":
                self.speed = max(1.0, float(cmd["value"]))
            else:
                raise ValueError(f"unknown command {name!r}")
            LOG.info("control: %s (speed=%s, running=%s)", cmd, self.speed, self.running.is_set())
        except (ValueError, KeyError, TypeError) as exc:
            LOG.warning("bad control message %r: %s", payload[:200], exc)

    def publish(self, row: Row) -> None:
        self.seq += 1
        if self.drop_rate and self.random.random() < self.drop_rate:
            self.dropped += 1
            return
        if not self.client.is_connected():
            # Store-and-forward: hold the stream until the broker is back
            # instead of piling messages into the client queue.
            LOG.warning("broker unreachable; holding stream at seq=%d", self.seq)
            while not self.client.is_connected():
                time.sleep(0.5)
            LOG.info("broker reachable again; resuming")
        info = self.client.publish(
            topics.measurement(self.station_id),
            json.dumps(row.payload(self.station_id, self.seq), separators=(",", ":")),
            qos=1,
        )
        try:
            info.wait_for_publish(timeout=30)
        except RuntimeError:
            # Connection dropped after publish(): paho keeps the QoS 1 message
            # queued and retransmits it after reconnecting.
            LOG.warning("connection lost while publishing seq=%d; paho will retransmit", self.seq)
        self.sent += 1

    def run(self, warmup: list[Row], live: list[Row]) -> None:
        LOG.info("warm-up burst: %d rows", len(warmup))
        for row in warmup:
            self.publish(row)
        LOG.info("streaming %d rows at %.0fx (%s .. %s)", len(live), self.speed,
                 live[0].ts.astimezone(PACIFIC) if live else "-",
                 live[-1].ts.astimezone(PACIFIC) if live else "-")
        prev = warmup[-1].ts if warmup else None
        for i, row in enumerate(live):
            self.running.wait()
            if prev is not None:
                gap = (row.ts - prev).total_seconds()
                time.sleep(min(max(gap, 0.0) / self.speed, 10.0))
            prev = row.ts
            self.publish(row)
            if i % 288 == 0:
                LOG.info("sim time %s  sent=%d dropped=%d",
                         row.ts.astimezone(PACIFIC).strftime("%Y-%m-%d %H:%M"), self.sent, self.dropped)


def wait_for_predictor(client, timeout: float) -> None:
    if timeout <= 0:
        return
    ready = threading.Event()

    def on_status(_client, _userdata, message):
        try:
            if json.loads(message.payload).get("state") == "online":
                ready.set()
        except ValueError:
            pass

    client.message_callback_add(topics.predictor_status(), on_status)
    client.subscribe(topics.predictor_status(), qos=1)
    if ready.wait(timeout):
        LOG.info("predictor is online")
    else:
        LOG.warning("predictor not online after %.0fs; streaming anyway", timeout)
    client.unsubscribe(topics.predictor_status())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    station_id = os.environ["STATION_ID"].strip()
    rows = load_rows(Path(os.environ["REPLAY_CSV"]), station_id)
    start = _parse_local(os.getenv("REPLAY_START")) or rows[0].ts
    end = _parse_local(os.getenv("REPLAY_END"))
    warmup_s = float(os.getenv("REPLAY_WARMUP_HOURS", "48")) * 3600
    warmup = [r for r in rows if 0 < (start - r.ts).total_seconds() <= warmup_s]
    live = [r for r in rows if r.ts >= start and (end is None or r.ts <= end)]
    if not live:
        raise ValueError("no rows in the requested replay range")

    status_topic = topics.station_status(station_id)
    offline = json.dumps({"state": "offline", "station_id": station_id})
    online = json.dumps({
        "state": "online", "station_id": station_id, "mode": "replay",
        "since": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    client = connect_client(
        f"sensor-{station_id}",
        will=(status_topic, offline),
        birth=(status_topic, online),
        subscriptions=[(topics.control(station_id), 1)],
    )
    replay = Replay(
        client,
        station_id,
        speed=float(os.getenv("REPLAY_SPEED", "300")),
        drop_rate=float(os.getenv("REPLAY_DROP_RATE", "0")),
    )
    client.message_callback_add(topics.control(station_id), lambda c, u, m: replay.on_control(m.payload))

    wait_for_predictor(client, float(os.getenv("REPLAY_WAIT_PREDICTOR", "60")))
    try:
        replay.run(warmup, live)
        LOG.info("replay finished: sent=%d dropped=%d", replay.sent, replay.dropped)
    finally:
        try:
            client.publish(status_topic, offline, qos=1, retain=True).wait_for_publish(timeout=5)
        except RuntimeError:
            pass  # broker gone: the Last Will publishes "offline" instead
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
