"""Consume verified station measurements into the TimesNet source table."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from urllib.request import Request, urlopen

from app.live.feed import Measurement, validate_station_id, validate_values

LOG = logging.getLogger(__name__)


def validate_message(payload: bytes, station_id: str) -> Measurement:
    data = json.loads(payload)
    if data.get("schema_version") != 1 or data.get("source") != "pems":
        raise ValueError("unsupported measurement schema/source")
    if data.get("station_id") != station_id:
        raise ValueError("wrong station")
    row = Measurement(**{key: data[key] for key in Measurement.__dataclass_fields__})
    datetime.strptime(row.timestamp, "%Y-%m-%d %H:%M:%S")
    validate_values(row.speed_mph, row.flow_veh_5min, row.occupancy_fraction, row.observed_percent)
    if row.observed_percent != 100:
        raise ValueError("measurement is not fully observed")
    return row


def insert_row(row: Measurement) -> None:
    # JSONEachRow keeps the ClickHouse timestamp conversion explicit. The table
    # deduplicates retries by timestamp via ReplacingMergeTree.
    query = "INSERT INTO traffic.traffic_5min (ts,speed_mph,flow_veh_5min,occupancy_fraction,observed_percent,ingest_version) FORMAT JSONEachRow"
    record = {
        "ts": row.timestamp,
        "speed_mph": row.speed_mph,
        "flow_veh_5min": row.flow_veh_5min,
        "occupancy_fraction": row.occupancy_fraction,
        "observed_percent": row.observed_percent,
        "ingest_version": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
    }
    from urllib.parse import urlencode

    url = f"http://{os.getenv('DB_HOST', 'localhost')}:{os.getenv('PORT', '8123')}/?{urlencode({'query': query, 'database': 'traffic'})}"
    request = Request(url, data=(json.dumps(record) + "\n").encode(), method="POST")
    user = os.environ["CLICKHOUSE_USER"]
    password = os.environ["CLICKHOUSE_PASSWORD"]
    request.add_header("X-ClickHouse-User", user)
    request.add_header("X-ClickHouse-Key", password)
    with urlopen(request, timeout=15) as response:
        response.read()


def main() -> None:
    import paho.mqtt.client as mqtt

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    station_id = validate_station_id(os.environ["PEMS_STATION_ID"])
    if not os.getenv("CLICKHOUSE_USER") or not os.getenv("CLICKHOUSE_PASSWORD"):
        raise ValueError("live ingestion requires CLICKHOUSE_USER and CLICKHOUSE_PASSWORD")
    topic = f"traffic/pems/stations/{station_id}/5min"

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe(topic, qos=1)
            LOG.info("subscribed %s", topic)
        else:
            LOG.error("MQTT connection rejected: %s", reason_code)

    def on_message(client, userdata, message):
        try:
            row = validate_message(message.payload, station_id)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            LOG.exception("invalid measurement discarded")
            client.ack(message.mid, message.qos)
            return
        try:
            insert_row(row)
        except Exception:
            LOG.exception("ClickHouse write failed; retry on reconnect")
            client.disconnect()  # no ACK: broker may redeliver with a persistent session
            return
        client.ack(message.mid, message.qos)
        LOG.info("stored station=%s timestamp=%s", station_id, row.timestamp)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"trafmon-ingest-{station_id}", clean_session=False, manual_ack=True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(os.getenv("MQTT_HOST", "localhost"), int(os.getenv("MQTT_PORT", "1883")))
    client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()
