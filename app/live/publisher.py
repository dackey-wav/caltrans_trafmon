"""Poll the approved PeMS FTP feed or a normalized HTTPS source and publish rows."""

from __future__ import annotations

import json
import logging
import os
import time
from ftplib import FTP, FTP_TLS
from datetime import UTC, datetime
from urllib.request import Request, urlopen

from app.live.feed import FTP_FILE, Measurement, fresh_complete, parse_csv, parse_ftp_5min, validate_station_id

LOG = logging.getLogger(__name__)


def fetch_csv(url: str, cookie: str | None) -> str:
    headers = {"Accept": "text/csv", "User-Agent": "caltrans-trafmon/0.1"}
    if cookie:
        headers["Cookie"] = cookie
    with urlopen(Request(url, headers=headers), timeout=20) as response:
        if response.status != 200:
            raise RuntimeError(f"source returned HTTP {response.status}")
        return response.read().decode("utf-8-sig")


def fetch_ftp_5min(host: str, user: str, password: str, district: int, *, tls: bool, after_filename: str | None = None) -> tuple[str, bytes | None]:
    """Fetch only the newest documented 5-minute aggregate from one district."""
    if not user or not password:
        raise ValueError("PeMS FTP username and password are required")
    if not 1 <= district <= 12:
        raise ValueError("invalid PeMS district")
    ftp = FTP_TLS(timeout=30) if tls else FTP(timeout=30)
    with ftp:
        ftp.connect(host, 21)
        ftp.login(user, password)
        if tls:
            ftp.prot_p()
        ftp.cwd(f"D{district}/Data/5min")
        filenames = [name.rsplit("/", 1)[-1] for name in ftp.nlst()]
        matches = [name for name in filenames if FTP_FILE.fullmatch(name)]
        if not matches:
            raise ValueError("PeMS FTP directory has no 5-minute aggregate files")
        filename = max(matches)
        if after_filename is not None and filename <= after_filename:
            return filename, None
        chunks: list[bytes] = []
        ftp.retrbinary(f"RETR {filename}", chunks.append)
        return filename, b"".join(chunks)


def main() -> None:
    import paho.mqtt.client as mqtt

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    station_id = validate_station_id(os.environ["PEMS_STATION_ID"])
    source_mode = os.getenv("PEMS_SOURCE_MODE", "ftp")
    if source_mode not in {"ftp", "https_csv"}:
        raise ValueError("PEMS_SOURCE_MODE must be ftp or https_csv")
    source_url = os.getenv("PEMS_SOURCE_URL", "")
    if source_mode == "https_csv" and not source_url.startswith("https://"):
        raise ValueError("PEMS_SOURCE_URL must be an HTTPS station CSV feed")
    if source_mode == "ftp" and (not os.getenv("PEMS_FTP_USER") or not os.getenv("PEMS_FTP_PASSWORD")):
        raise ValueError("a separate PeMS FTP username and password are required")
    topic = f"traffic/pems/stations/{station_id}/5min"
    interval = max(30, int(os.getenv("PEMS_POLL_SECONDS", "300")))
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(os.getenv("MQTT_HOST", "localhost"), int(os.getenv("MQTT_PORT", "1883")))
    client.loop_start()
    last_published: str | None = None
    last_downloaded: str | None = None
    pending_rows: list[Measurement] = []
    try:
        while True:
            try:
                if not pending_rows:
                    if source_mode == "ftp":
                        filename, body = fetch_ftp_5min(
                            os.getenv("PEMS_FTP_HOST", "pems.dot.ca.gov"),
                            os.getenv("PEMS_FTP_USER", ""),
                            os.getenv("PEMS_FTP_PASSWORD", ""),
                            int(os.getenv("PEMS_DISTRICT", "11")),
                            tls=os.getenv("PEMS_FTP_TLS", "1") == "1",
                            after_filename=last_downloaded,
                        )
                        if body is None:
                            time.sleep(interval)
                            continue
                        last_downloaded = filename
                        pending_rows = parse_ftp_5min(body, filename, station_id)
                    else:
                        pending_rows = parse_csv(fetch_csv(source_url, os.getenv("PEMS_HTTP_COOKIE")), station_id)
                for row in fresh_complete(pending_rows, now=datetime.now(UTC)):
                    if last_published is not None and row.timestamp <= last_published:
                        continue
                    info = client.publish(topic, json.dumps(row.payload(), separators=(",", ":")), qos=1)
                    info.wait_for_publish(timeout=20)
                    if not info.is_published():
                        raise TimeoutError("MQTT publish was not acknowledged")
                    last_published = row.timestamp
                    LOG.info("published station=%s timestamp=%s", station_id, row.timestamp)
                pending_rows = []
            except Exception:
                LOG.exception("station poll failed; will retry")
            time.sleep(interval)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
