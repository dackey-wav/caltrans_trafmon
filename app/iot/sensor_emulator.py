"""Sensor node: replay a downloaded PeMS station CSV as a live 5-minute MQTT feed.

The CSV is the ``processed/observations.csv(.gz)`` file written by
pems-data-downloader (``timestamp,station_id,speed_mph,flow_veh_5min,
occupancy_fraction,observed_percent,latitude,longitude,...``). Rows are
published on ``traffic/pems/stations/<id>/5min`` with QoS 1, using the JSON
schema of the live FTP publisher in ``app.live``.

A simulated clock runs ``speed`` times faster than wall time; a row is sent
when the clock passes its timestamp. Each replay is a *session*: a reset sends
``warmup`` hours of history as a fast burst (the predictor needs two days of
input and a week for its baseline), then streams from ``start``.

Commands on ``.../control`` (JSON):
    {"cmd": "pause"} | {"cmd": "resume"} | {"cmd": "speed", "value": 600}
    {"cmd": "reset", "start": "2026-09-28T06:00", "speed": 300, "paused": false}
    {"cmd": "outage", "seconds": 30, "mode": "buffer" | "drop"}
    {"cmd": "loss", "value": 0.05}

The retained ``.../replay`` topic carries the clock (``sim_time`` at
``wall_time_ms``, ``speed``) so a dashboard can run the same clock locally.

Environment:
    STATION_ID, REPLAY_CSV          station and CSV (required)
    REPLAY_START                    first streamed local time (default: 8 days into the data)
    REPLAY_END                      last local time (default: end of data)
    REPLAY_WARMUP_HOURS             history burst before start (default 192)
    REPLAY_SPEED                    simulated seconds per wall second (default 300)
    REPLAY_DROP_RATE                probability of losing a message (default 0)
    REPLAY_AUTOSTART                1 = start streaming at boot, 0 = wait for a reset (default 1)
    REPLAY_WAIT_PREDICTOR           seconds to wait for the predictor at boot (default 60)
    MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from app.iot import topics
from app.iot.mqtt_common import connect_client, go_offline, go_online

LOG = logging.getLogger("sensor")
PACIFIC = ZoneInfo("America/Los_Angeles")
VALUE_COLUMNS = ("speed_mph", "flow_veh_5min", "occupancy_fraction", "observed_percent")
META_COLUMNS = ("latitude", "longitude", "freeway", "direction", "lane_type")
MAX_SPEED = 3600.0
STATE_EVERY_S = 1.0


@dataclass(frozen=True)
class Row:
    ts: datetime  # aware, UTC
    values: dict[str, float | None]


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


def parse_local(text: str | None) -> datetime | None:
    if not text:
        return None
    ts = datetime.fromisoformat(text)
    return (ts if ts.tzinfo else ts.replace(tzinfo=PACIFIC)).astimezone(UTC)


def iso_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_rows(path: Path, station_id: str) -> tuple[list[Row], dict]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    missing = {"timestamp", "station_id", *VALUE_COLUMNS} - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{path} is not a processed PeMS observations CSV; missing {sorted(missing)}")
    rows, meta = [], {}
    for r in reader:
        if r["station_id"].strip() != station_id:
            continue
        if not meta:
            meta = {c: (r.get(c) or "").strip() for c in META_COLUMNS}
        rows.append(Row(_parse_ts(r["timestamp"]), {c: _number(r[c]) for c in VALUE_COLUMNS}))
    if not rows:
        raise ValueError(f"station {station_id} not found in {path}")
    rows.sort(key=lambda r: r.ts)
    repeats = sum(a.ts == b.ts for a, b in zip(rows, rows[1:], strict=False))
    if repeats:
        # Kept on purpose: PeMS archives occasionally repeat rows, and the
        # receiving node has to cope with that just like with QoS 1 redelivery.
        LOG.warning("%d repeated timestamps in the source; they are sent as-is", repeats)
    for key in ("latitude", "longitude"):
        meta[key] = _number(meta.get(key))
    return rows, meta


class SimClock:
    """Simulated time = anchor + (wall elapsed) * speed while running."""

    def __init__(self, sim: datetime, speed: float):
        self.anchor_sim = sim
        self.anchor_wall = time.monotonic()
        self.speed = speed
        self.running = False

    def now(self) -> datetime:
        if not self.running:
            return self.anchor_sim
        return self.anchor_sim + timedelta(seconds=(time.monotonic() - self.anchor_wall) * self.speed)

    def _reanchor(self) -> None:
        self.anchor_sim = self.now()
        self.anchor_wall = time.monotonic()

    def set_speed(self, speed: float) -> None:
        self._reanchor()
        self.speed = speed

    def pause(self) -> None:
        self._reanchor()
        self.running = False

    def resume(self) -> None:
        self.anchor_wall = time.monotonic()
        self.running = True

    def jump(self, sim: datetime) -> None:
        self.anchor_sim = sim
        self.anchor_wall = time.monotonic()


class Replay:
    def __init__(self, client, station_id: str, rows: list[Row], meta: dict, *,
                 speed: float, drop_rate: float, warmup_hours: float, seed: int = 7):
        self.client = client
        self.station_id = station_id
        self.rows = rows
        self.meta = meta
        self.warmup = timedelta(hours=warmup_hours)
        self.drop_rate = drop_rate
        self.random = random.Random(seed)
        self.commands: queue.Queue = queue.Queue()
        self.clock = SimClock(rows[0].ts, speed)
        self.phase = "idle"
        self.session = ""
        self.start = self.end = rows[0].ts
        self.live: list[Row] = []
        self.idx = 0
        self.seq = 0
        self.sent = self.dropped = 0
        self.backlog: list[dict] = []
        self.outage: dict | None = None
        self.outage_until = 0.0
        self.last_state = 0.0

    # -- commands -------------------------------------------------------
    def on_control(self, payload: bytes) -> None:
        try:
            cmd = json.loads(payload)
            if not isinstance(cmd, dict) or "cmd" not in cmd:
                raise ValueError("expected {'cmd': ...}")
            self.commands.put(cmd)
        except ValueError as exc:
            LOG.warning("bad control message %r: %s", payload[:200], exc)

    def handle(self, cmd: dict) -> None:
        name = cmd.get("cmd")
        try:
            if name == "pause" and self.phase == "running":
                self.clock.pause()
                self.phase = "paused"
            elif name == "resume" and self.phase == "paused":
                self.clock.resume()
                self.phase = "running"
            elif name == "speed":
                self.clock.set_speed(min(MAX_SPEED, max(1.0, float(cmd["value"]))))
            elif name == "loss":
                self.drop_rate = min(0.9, max(0.0, float(cmd["value"])))
            elif name == "reset":
                self.reset(parse_local(cmd.get("start")) or self.start, cmd.get("speed"),
                           paused=bool(cmd.get("paused", False)), end=parse_local(cmd.get("end")))
            elif name == "outage":
                self.begin_outage(float(cmd.get("seconds", 30)), str(cmd.get("mode", "buffer")))
            else:
                raise ValueError(f"unknown or not applicable now: {name!r} (phase {self.phase})")
            LOG.info("control %s -> phase=%s speed=%.0f loss=%.2f", cmd, self.phase,
                     self.clock.speed, self.drop_rate)
        except (ValueError, KeyError, TypeError) as exc:
            LOG.warning("control %s rejected: %s", cmd, exc)
        self.publish_state(force=True)

    # -- session --------------------------------------------------------
    def reset(self, start: datetime, speed=None, *, paused: bool = False,
              end: datetime | None = None) -> None:
        first, last = self.rows[0].ts, self.rows[-1].ts
        start = min(max(start, first + self.warmup), last)
        start = start.replace(minute=start.minute - start.minute % 5, second=0, microsecond=0)
        self.session = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        self.start, self.end = start, min(end or last, last)
        warm = [r for r in self.rows if start - self.warmup <= r.ts < start]
        self.live = [r for r in self.rows if start <= r.ts <= self.end]
        self.idx = 0
        self.backlog = []
        if speed:
            self.clock.speed = min(MAX_SPEED, max(1.0, float(speed)))
        self.clock.jump(start)
        self.clock.running = False
        self.phase = "warmup"
        self.publish_state(force=True)
        LOG.info("session %s: warm-up %d rows, then %d rows from %s",
                 self.session, len(warm), len(self.live), start.astimezone(PACIFIC))
        for row in warm:
            self.emit(row, loss=False)
        if paused:
            self.phase = "paused"
        else:
            self.clock.resume()
            self.phase = "running"

    # -- outage ---------------------------------------------------------
    def begin_outage(self, seconds: float, mode: str) -> None:
        if mode not in ("buffer", "drop"):
            raise ValueError("outage mode must be buffer or drop")
        if self.outage:
            raise ValueError("already offline")
        seconds = min(max(seconds, 3.0), 600.0)
        self.outage = {"mode": mode, "seconds": seconds,
                       "until_ms": int((time.time() + seconds) * 1000), "lost": 0}
        self.outage_until = time.monotonic() + seconds
        self.publish_state(force=True)
        self.client.publish(topics.station_status(self.station_id), json.dumps({
            "state": "offline", "station_id": self.station_id,
            "reason": f"simulated network outage ({mode})", "until_ms": self.outage["until_ms"],
        }), qos=1, retain=True).wait_for_publish(timeout=5)
        LOG.warning("simulated outage for %.0fs (%s)", seconds, mode)
        go_offline(self.client)

    def end_outage(self) -> None:
        LOG.info("outage over: reconnecting, %d buffered, %d lost",
                 len(self.backlog), self.outage["lost"] if self.outage else 0)
        self.outage = None
        go_online(self.client)
        self.publish_state(force=True)  # shows the backlog before it is flushed
        self.flush()
        self.publish_state(force=True)

    # -- publishing -----------------------------------------------------
    def message(self, row: Row) -> dict:
        self.seq += 1
        return {
            "schema_version": 1,
            "source": "pems",
            "origin": "replay",
            "station_id": self.station_id,
            "timestamp": row.ts.astimezone(PACIFIC).strftime("%Y-%m-%d %H:%M:%S"),
            "ts_utc": iso_utc(row.ts),
            "session": self.session,
            "session_start": iso_utc(self.start),
            "seq": self.seq,
            **row.values,
        }

    def emit(self, row: Row, *, loss: bool = True) -> None:
        if loss and self.drop_rate and self.random.random() < self.drop_rate:
            self.seq += 1
            self.dropped += 1
            return
        msg = self.message(row)
        if self.outage and self.outage["mode"] == "drop":
            self.outage["lost"] += 1
            self.dropped += 1
            return
        if self.outage or not self.client.is_connected():
            self.backlog.append(msg)  # store-and-forward until the broker is reachable
            return
        self.send(msg)

    def send(self, msg: dict) -> None:
        msg["sent_at_ms"] = int(time.time() * 1000)
        info = self.client.publish(topics.measurement(self.station_id),
                                   json.dumps(msg, separators=(",", ":")), qos=1)
        try:
            info.wait_for_publish(timeout=30)
        except RuntimeError:
            # Connection dropped after publish(): paho keeps the QoS 1 message
            # queued and retransmits it after reconnecting.
            LOG.warning("connection lost while publishing seq=%d; paho will retransmit", msg["seq"])
        self.sent += 1

    def flush(self) -> None:
        backlog, self.backlog = self.backlog, []
        if backlog:
            LOG.info("sending %d buffered measurements", len(backlog))
        for msg in backlog:
            self.send(msg)

    def state(self) -> dict:
        return {
            "station_id": self.station_id,
            "station": self.meta,
            "session": self.session,
            "phase": self.phase,
            "speed": self.clock.speed,
            "sim_time": iso_utc(self.clock.now()),
            "wall_time_ms": int(time.time() * 1000),
            "start": iso_utc(self.start),
            "end": iso_utc(self.end),
            "data_from": iso_utc(self.rows[0].ts + self.warmup),
            "data_to": iso_utc(self.rows[-1].ts),
            "progress": round(self.idx / len(self.live), 4) if self.live else 0,
            "sent": self.sent,
            "dropped": self.dropped,
            "drop_rate": self.drop_rate,
            "backlog": len(self.backlog),
            "outage": self.outage,
        }

    def publish_state(self, force: bool = False) -> None:
        if not force and time.monotonic() - self.last_state < STATE_EVERY_S:
            return
        self.last_state = time.monotonic()
        if self.client.is_connected() and not (self.outage and not force):
            self.client.publish(topics.replay(self.station_id), json.dumps(self.state()),
                                qos=1, retain=True)

    # -- main loop ------------------------------------------------------
    def serve_forever(self) -> None:
        while True:
            timeout = 0.5
            if self.phase == "running" and self.idx < len(self.live):
                ahead = (self.live[self.idx].ts - self.clock.now()).total_seconds()
                timeout = min(0.25, max(0.0, ahead / self.clock.speed))
            try:
                self.handle(self.commands.get(timeout=timeout))
            except queue.Empty:
                pass
            if self.outage and time.monotonic() >= self.outage_until:
                self.end_outage()
            elif self.backlog and not self.outage and self.client.is_connected():
                self.flush()
            if self.phase == "running" or (self.outage and self.clock.running):
                now = self.clock.now()
                while self.idx < len(self.live) and self.live[self.idx].ts <= now:
                    self.emit(self.live[self.idx])
                    self.idx += 1
                if self.idx >= len(self.live) and self.phase == "running":
                    self.clock.jump(self.end)
                    self.clock.running = False
                    self.phase = "finished"
                    LOG.info("session %s finished: sent=%d dropped=%d", self.session, self.sent, self.dropped)
                    self.publish_state(force=True)
            self.publish_state()


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
    client.message_callback_remove(topics.predictor_status())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    station_id = os.environ["STATION_ID"].strip()
    rows, meta = load_rows(Path(os.environ["REPLAY_CSV"]), station_id)
    warmup_hours = float(os.getenv("REPLAY_WARMUP_HOURS", "192"))

    status_topic = topics.station_status(station_id)
    offline = json.dumps({"state": "offline", "station_id": station_id})
    replay_holder: dict = {}
    client = connect_client(
        f"sensor-{station_id}",
        will=(status_topic, offline),
        birth=(status_topic, lambda: json.dumps({
            "state": "online", "station_id": station_id, "mode": "replay",
            "since": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })),
        subscriptions=[(topics.control(station_id), 1)],
        on_message=lambda c, u, m: replay_holder["r"].on_control(m.payload) if "r" in replay_holder else None,
    )
    replay = Replay(client, station_id, rows, meta,
                    speed=float(os.getenv("REPLAY_SPEED", "300")),
                    drop_rate=float(os.getenv("REPLAY_DROP_RATE", "0")),
                    warmup_hours=warmup_hours)
    replay_holder["r"] = replay
    replay.publish_state(force=True)

    if os.getenv("REPLAY_AUTOSTART", "1") == "1":
        wait_for_predictor(client, float(os.getenv("REPLAY_WAIT_PREDICTOR", "60")))
        start = parse_local(os.getenv("REPLAY_START")) or rows[0].ts + timedelta(hours=warmup_hours)
        replay.reset(start, end=parse_local(os.getenv("REPLAY_END")))
    replay.serve_forever()


if __name__ == "__main__":
    main()
