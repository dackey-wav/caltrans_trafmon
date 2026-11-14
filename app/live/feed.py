"""Validate a PeMS station CSV export before publishing its measurements."""

from __future__ import annotations

import csv
import gzip
import io
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
REQUIRED = ("station_id", "timestamp", "speed_mph", "flow_veh_5min", "occupancy_fraction", "observed_percent")
FTP_FILE = re.compile(r"5minagg_([0-9]{14})\.txt\.gz$")


@dataclass(frozen=True)
class Measurement:
    station_id: str
    timestamp: str  # California local time, matching traffic.traffic_5min
    speed_mph: float
    flow_veh_5min: float
    occupancy_fraction: float
    observed_percent: float

    def payload(self) -> dict:
        return {"schema_version": 1, "source": "pems", **asdict(self)}


def parse_csv(body: str, station_id: str) -> list[Measurement]:
    """Accept a normalized CSV export; reject login pages and malformed rows."""
    reader = csv.DictReader(io.StringIO(body.lstrip("\ufeff")))
    if not reader.fieldnames:
        raise ValueError("empty feed")
    fields = {field.strip().lower(): field for field in reader.fieldnames}
    missing = set(REQUIRED) - fields.keys()
    if missing:
        raise ValueError(f"feed is not station CSV; missing {', '.join(sorted(missing))}")
    station_key = fields["station_id"]
    result = []
    for row in reader:
        if (row.get(station_key) or "").strip() != station_id:
            continue
        try:
            timestamp = datetime.fromisoformat((row[fields["timestamp"]] or "").strip())
            if timestamp.tzinfo is not None:
                timestamp = timestamp.astimezone(PACIFIC).replace(tzinfo=None)
            if timestamp.second or timestamp.minute % 5:
                raise ValueError("timestamp is not on a 5-minute boundary")
            speed = float(row[fields["speed_mph"]])
            flow = float(row[fields["flow_veh_5min"]])
            occupancy = float(row[fields["occupancy_fraction"]])
            observed = float(row[fields["observed_percent"]])
            validate_values(speed, flow, occupancy, observed)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"invalid row {reader.line_num}: {exc}") from exc
        result.append(Measurement(station_id, timestamp.strftime("%Y-%m-%d %H:%M:%S"), speed, flow, occupancy, observed))
    if not result:
        raise ValueError(f"no rows for station {station_id}")
    return sorted(result, key=lambda item: item.timestamp)


def parse_ftp_5min(body: bytes, filename: str, station_id: str) -> list[Measurement]:
    """Parse PeMS's documented district-wide 5-minute FTP aggregate file."""
    match = FTP_FILE.fullmatch(filename)
    if not match:
        raise ValueError("unexpected PeMS 5-minute filename")
    timestamp = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    if timestamp.second or timestamp.minute % 5:
        raise ValueError("FTP filename is not on a 5-minute boundary")
    try:
        rows = csv.reader(io.StringIO(gzip.decompress(body).decode("utf-8-sig")))
        for line_number, row in enumerate(rows, 1):
            if not row or (line_number == 1 and row[0].strip().upper() == "VDS_ID"):
                continue
            if row[0].strip() != station_id:
                continue
            if len(row) != 11:
                raise ValueError(f"station row {line_number} has {len(row)} columns, expected 11")
            flow, occupancy, speed, observed = (float(row[index]) for index in (1, 2, 3, 10))
            validate_values(speed, flow, occupancy, observed)
            return [Measurement(station_id, timestamp.strftime("%Y-%m-%d %H:%M:%S"), speed, flow, occupancy, observed)]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"invalid PeMS FTP file: {exc}") from exc
    raise ValueError(f"station {station_id} missing from PeMS FTP file")


def validate_station_id(station_id: str) -> str:
    if not re.fullmatch(r"[0-9]{1,12}", station_id):
        raise ValueError("PEMS_STATION_ID must be a numeric detector station ID")
    return station_id


def validate_values(speed: float, flow: float, occupancy: float, observed: float) -> None:
    values = (speed, flow, occupancy, observed)
    if not all(isinstance(value, (float, int)) and math.isfinite(value) for value in values):
        raise ValueError("non-finite or non-numeric measurement")
    if not (0 <= speed <= 120 and flow >= 0 and 0 <= occupancy <= 1 and 0 <= observed <= 100):
        raise ValueError("measurement outside valid range")


def fresh_complete(rows: list[Measurement], *, now: datetime, max_age_minutes: int = 20) -> list[Measurement]:
    """Only publish fully observed, recent buckets; no fabricated zeroes."""
    local_now = now.astimezone(PACIFIC).replace(tzinfo=None)
    cutoff = local_now - timedelta(minutes=max_age_minutes)
    return [row for row in rows if cutoff <= datetime.fromisoformat(row.timestamp) <= local_now and row.observed_percent == 100]
