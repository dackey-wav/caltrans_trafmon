"""Comma-separated 5-minute station aggregate CSV."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime

import app.core.config as cfg
from app.importer.importer_csv.pipeline import progress_line

REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "speed_mph",
    "flow_veh_5min",
    "occupancy_fraction",
    "observed_percent",
)
STAGE_COLUMNS: tuple[str, ...] = (
    "ts",
    "speed_mph",
    "flow_veh_5min",
    "occupancy_fraction",
    "observed_percent",
)
MEASURE_COLUMNS: tuple[str, ...] = STAGE_COLUMNS[1:]
_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
)


def _header_key(name: str) -> str:
    return name.strip().lstrip("\ufeff").lower()


def _header_lookup(fieldnames: list[str] | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in fieldnames or []:
        key = _header_key(name)
        if key:
            lookup[key] = name
    return lookup


def parse_timestamp(value: str) -> str | None:
    """Return a naive California clock time as YYYY-MM-DD HH:MM:SS."""

    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
    for fmt in _TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return None


def parse_measure(value: str) -> str | None:
    """Return a canonical float token, '' when blank, or None when invalid."""

    text = value.strip().replace(",", ".")
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return None
    return format(number, ".10g")


def validate_sample(path: str) -> None:
    """Reject a sample whose comma-separated header lacks the series columns."""

    with open(path, encoding="utf-8-sig", newline="") as src:
        header = next(csv.reader(src, delimiter=","), [])
    if not header or (len(header) == 1 and "\t" in header[0]):
        raise ValueError("expected a comma-separated header")
    keys = {_header_key(cell) for cell in header}
    missing = [column for column in REQUIRED_COLUMNS if column not in keys]
    if missing:
        raise ValueError(f"header is missing: {', '.join(missing)}")


def _build_tsv(
    files: list[dict],
    output_path: str,
    progress_every: int,
) -> tuple[int, int, dict[str, int]]:
    rows_written = 0
    bad_rows = 0
    per_file_inserted: dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("\t".join(STAGE_COLUMNS) + "\n")
        for index, task in enumerate(files, start=1):
            inserted_for_file = 0
            with open(task["path"], encoding="utf-8-sig", newline="") as src:
                reader = csv.DictReader(src, delimiter=",")
                lookup = _header_lookup(reader.fieldnames)
                missing = [column for column in REQUIRED_COLUMNS if column not in lookup]
                if missing:
                    raise ValueError(
                        f"{task['path']} header is missing: {', '.join(missing)}"
                    )
                for row in reader:
                    if not any((cell or "").strip() for cell in row.values()):
                        continue
                    timestamp = parse_timestamp(row.get(lookup["timestamp"], "") or "")
                    if timestamp is None:
                        bad_rows += 1
                        continue
                    measures: list[str] = []
                    invalid = False
                    for column in MEASURE_COLUMNS:
                        parsed = parse_measure(row.get(lookup[column], "") or "")
                        if parsed is None:
                            invalid = True
                            break
                        measures.append(parsed)
                    if invalid:
                        bad_rows += 1
                        continue
                    out.write("\t".join([timestamp, *measures]) + "\n")
                    inserted_for_file += 1

            per_file_inserted[task["file_key"]] = inserted_for_file
            rows_written += inserted_for_file
            if progress_every > 0 and (index % progress_every == 0 or index == len(files)):
                print(
                    f"{progress_line(index, len(files), 'build-tsv')} "
                    f"rows_buffered={rows_written} bad_rows={bad_rows}"
                )

    return rows_written, bad_rows, per_file_inserted


@dataclass(frozen=True)
class Station5MinAdapter:
    format_id: str = "station-5min"
    stage_columns: tuple[str, ...] = STAGE_COLUMNS

    def stage_table_fq(self) -> str:
        table = os.getenv("CSV_STATION_5MIN_STAGE_TABLE", "traffic_5min_stage")
        return f"{cfg.DB_NAME}.{table}"

    def destination_table_fq(self) -> str:
        return f"{cfg.DB_NAME}.traffic_5min"

    def build_tsv(
        self,
        files: list[dict],
        output_path: str,
        progress_every: int,
    ) -> tuple[int, int, dict[str, int]]:
        return _build_tsv(files, output_path, progress_every)

    def transform_sql(self) -> str:
        stage = self.stage_table_fq()
        destination = self.destination_table_fq()
        return f"""
        INSERT INTO {destination} (
            ts, speed_mph, flow_veh_5min, occupancy_fraction, observed_percent, ingest_version
        )
        SELECT
            toDateTime64(ts, 0, 'America/Los_Angeles') AS ts,
            toFloat32OrNull(nullIf(speed_mph, '')) AS speed_mph,
            toFloat32OrNull(nullIf(flow_veh_5min, '')) AS flow_veh_5min,
            toFloat32OrNull(nullIf(occupancy_fraction, '')) AS occupancy_fraction,
            toFloat32OrNull(nullIf(observed_percent, '')) AS observed_percent,
            now64(3, 'UTC') AS ingest_version
        FROM {stage}
        """

    def optimize_sql(self) -> str:
        return f"OPTIMIZE TABLE {self.destination_table_fq()} FINAL"


ADAPTER = Station5MinAdapter()
