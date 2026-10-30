"""Preflight for the single 5-minute station CSV import."""

from __future__ import annotations

import csv
import os
from collections.abc import Callable
from dataclasses import dataclass

from app.importer.importer_csv.formats.station_5min import (
    ADAPTER,
    validate_sample,
)
from app.importer.importer_csv.pipeline import (
    DEFAULT_STATE_PATH,
    StagedAdapter,
    discover_csv_files,
    run_staged_import,
)


class PreflightError(ValueError):
    """Input validation failed before database access."""


@dataclass(frozen=True)
class FormatSpec:
    format_id: str
    destination_description: str
    destination_resolver: Callable[[], str]
    default_state_path: str
    runner: Callable[..., None]
    adapter: StagedAdapter
    sample_validator: Callable[[str], None]


FORMAT = FormatSpec(
    format_id="station-5min",
    destination_description="5-minute station series",
    destination_resolver=ADAPTER.destination_table_fq,
    default_state_path=DEFAULT_STATE_PATH,
    runner=run_staged_import,
    adapter=ADAPTER,
    sample_validator=validate_sample,
)


def get_format() -> FormatSpec:
    """Return the only registered import."""
    
    return FORMAT


def preflight(
    input_path: str | os.PathLike[str],
    spec: FormatSpec | None = None,
    *,
    sample_size: int = 3,
) -> list[str]:
    """Validate input discovery and sampled CSV shape without database access."""

    resolved = spec or FORMAT
    path = os.fspath(input_path)
    if not os.path.exists(path):
        raise PreflightError(f"input does not exist: {path}")

    files = discover_csv_files(path)
    if not files:
        raise PreflightError(f"input contains no CSV files: {path}")

    for sample in files[:sample_size]:
        try:
            resolved.sample_validator(sample)
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            raise PreflightError(
                f"{sample} does not match format {resolved.format_id}: {exc}"
            ) from exc
    return files
