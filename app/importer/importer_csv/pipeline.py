"""
Common helpers for CSV native bulk importers.

The CSV import flow is:

1) Scan files under `data_dir`, filter via importer state.
2) Build one combined TSVWithNames file from queued CSVs.
3) Upload the TSV into a staging table via ClickHouse `raw_insert`.
4) Run one `INSERT ... SELECT` transform from the staging table into the
   final destination table (typing, hashing, time conversion happen on the
   ClickHouse side, in C++).
5) Update on-disk state (resume/skip) only after a successful transform.

This module owns discovery, state serialization, TSV upload, and the shared
staging lifecycle. Format modules provide source parsing, file context, and
transform SQL.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Protocol

DEFAULT_STATE_FLUSH_EVERY = int(os.getenv("CSV_STATE_FLUSH_EVERY", "100"))
DEFAULT_PROGRESS_EVERY_FILES = int(os.getenv("CSV_NATIVE_PROGRESS_EVERY_FILES", "200"))

RUNTIME_DIR = os.path.join("app", "importer", "importer_csv", "runtime")
DEFAULT_STATE_PATH = os.path.join(RUNTIME_DIR, "state_traffic_5min.json")


def quote_ident(name: str) -> str:
    """Quote a SQL identifier with backticks (safe for `-`, `+`, etc.)."""

    return f"`{name.replace('`', '``')}`"


def _file_fingerprint(path: str) -> dict:
    """Build a file fingerprint used for resume/skip checks."""

    stat = os.stat(path)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _state_file_key(path: str) -> str:
    """Build a stable state key independent from the chosen `data_dir`."""

    abs_path = os.path.abspath(path)
    norm = os.path.normcase(os.path.normpath(abs_path))
    return norm.replace("\\", "/")


def _is_already_imported(
    state_files: dict,
    state_key: str,
    legacy_keys: Iterable[str],
    fingerprint: dict,
) -> bool:
    """Check current and legacy state keys for the same file fingerprint."""

    if state_files.get(state_key) == fingerprint:
        return True
    for key in legacy_keys:
        if key and state_files.get(key) == fingerprint:
            return True
    return False


def _load_state(path: str) -> dict:
    """Load importer state from JSON, returning a default on missing/invalid file."""

    if not os.path.exists(path):
        return {"files": {}, "last_run_at": None, "total_inserted": 0}

    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return {"files": {}, "last_run_at": None, "total_inserted": 0}

    if "files" not in state or not isinstance(state["files"], dict):
        state["files"] = {}
    return state


def _save_state(path: str, state: dict) -> None:
    """Persist importer state as compact JSON via atomic replace.

    On Windows, replacing a file can fail with `WinError 5` when another
    process has an active handle on the destination. We retry a few times and
    then fall back to in-place rewrite of the same file path.
    """

    state_dir = os.path.dirname(path)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)

    payload = json.dumps(state, ensure_ascii=False, separators=(",", ":"))

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())

    for _ in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            sleep(0.05)

    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())

    try:
        os.remove(tmp_path)
    except OSError:
        pass


def _flush_state_if_needed(
    state_path: str,
    state: dict,
    dirty_updates: int,
    flush_every: int,
    force: bool = False,
) -> int:
    """Flush state every N updates, or immediately when forced."""

    if flush_every <= 0:
        flush_every = 1
    if not force and dirty_updates < flush_every:
        return dirty_updates
    _save_state(state_path, state)
    return 0


def progress_line(current: int, total: int, label: str) -> str:
    """Render a lightweight textual progress indicator."""

    if total <= 0:
        return f"[progress] {label}: {current}"
    pct = (current / total) * 100.0
    return f"[progress] {label}: {current}/{total} ({pct:.1f}%)"


def discover_csv_files(input_path: str | os.PathLike[str]) -> list[str]:
    """Return one CSV input or all CSV files below a directory."""

    path = Path(input_path)
    if path.is_file():
        return [str(path)] if path.suffix.lower() == ".csv" else []
    if path.is_dir():
        return [
            str(candidate)
            for candidate in sorted(path.rglob("*"))
            if candidate.is_file() and candidate.suffix.lower() == ".csv"
        ]
    return []


def scan_files(
    input_path: str,
    state: dict,
    state_path: str,
    resume: bool,
    state_flush_every: int = DEFAULT_STATE_FLUSH_EVERY,
    progress_every: int = 500,
) -> tuple[list[dict], int, int]:
    """Discover `.csv` inputs and split them into queued/skipped.

    Returns (queued_tasks, total_files, skipped_files).
    Side effect: refreshes legacy state keys to the canonical form.
    """

    files = discover_csv_files(input_path)
    if not files:
        return [], 0, 0

    input_abs = os.path.abspath(input_path)
    data_dir_abs = input_abs if os.path.isdir(input_abs) else os.path.dirname(input_abs)
    queued: list[dict] = []
    skipped = 0
    dirty_updates = 0

    for index, path in enumerate(files, start=1):
        rel_path = os.path.relpath(path, data_dir_abs)
        display_key = rel_path.replace("\\", "/")
        file_key = _state_file_key(path)
        fingerprint = _file_fingerprint(path)

        if resume:
            legacy_keys = (display_key, os.path.basename(path))
            if _is_already_imported(state["files"], file_key, legacy_keys, fingerprint):
                if state["files"].get(file_key) != fingerprint:
                    state["files"][file_key] = fingerprint
                    dirty_updates += 1
                    dirty_updates = _flush_state_if_needed(
                        state_path, state, dirty_updates, state_flush_every
                    )
                skipped += 1
                continue

        queued.append(
            {
                "path": path,
                "display_key": display_key,
                "file_key": file_key,
                "fingerprint": fingerprint,
            }
        )

        if progress_every > 0 and (index % progress_every == 0 or index == len(files)):
            print(
                f"{progress_line(index, len(files), 'scan-files')} "
                f"queued={len(queued)} skipped={skipped}"
            )

    if dirty_updates:
        _flush_state_if_needed(state_path, state, dirty_updates, state_flush_every, force=True)

    return queued, len(files), skipped


class StagedAdapter(Protocol):
    """Format adapter consumed by the shared staged importer."""

    format_id: str
    stage_columns: tuple[str, ...]

    def stage_table_fq(self) -> str: ...

    def destination_table_fq(self) -> str: ...

    def build_tsv(
        self,
        files: list[dict],
        output_path: str,
        progress_every: int,
    ) -> tuple[int, int, dict[str, int]]: ...

    def transform_sql(self) -> str: ...


def run_staged_import(
    adapter: StagedAdapter,
    input_path: str,
    state_path: str,
    resume: bool = True,
    state_flush_every: int = DEFAULT_STATE_FLUSH_EVERY,
    progress_every: int = DEFAULT_PROGRESS_EVERY_FILES,
) -> None:
    """Run state, staging upload, transform, and state update for an adapter."""

    from app.core.db import ClickHouseDB

    state = _load_state(state_path)
    queued, total_files, skipped_files = scan_files(
        input_path,
        state,
        state_path,
        resume=resume,
        state_flush_every=state_flush_every,
    )
    adapter_name = f"{adapter.format_id}-native"

    if not queued:
        print(
            f"\n[summary] adapter={adapter_name} files_total={total_files} files_queued=0 "
            f"files_skipped={skipped_files} inserted_total=0 bad_rows_total=0"
        )
        return

    started = datetime.now(UTC)
    stage_table = adapter.stage_table_fq()
    destination_table = adapter.destination_table_fq()
    columns_sql = ",\n    ".join(
        f"{quote_ident(column)} String" for column in adapter.stage_columns
    )

    with ClickHouseDB.from_config() as db:
        db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {stage_table} (
                {columns_sql}
            )
            ENGINE = MergeTree
            ORDER BY tuple()
            """
        )
        db.execute(f"TRUNCATE TABLE {stage_table}")

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            suffix=f"_{adapter.format_id}_stage.tsv",
        ) as tmp:
            temp_path = tmp.name

        try:
            print(f"[stage] building combined TSV for {len(queued)} files...")
            rows_written, bad_rows, inserted_per_file = adapter.build_tsv(
                queued, temp_path, progress_every
            )
            print(f"[stage] combined TSV ready: rows={rows_written} bad_rows={bad_rows}")

            print("[stage] uploading TSV to ClickHouse staging table...")
            with open(temp_path, "rb") as inp:
                db.client.raw_insert(
                    table=stage_table,
                    insert_block=inp,
                    fmt="TSVWithNames",
                )
            print("[stage] upload complete")

            print(f"[stage] transforming staging -> {destination_table} ...")
            db.execute(adapter.transform_sql())
            optimize_sql = getattr(adapter, "optimize_sql", None)
            if optimize_sql is not None:
                db.execute(optimize_sql())
            print("[stage] transform complete")
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            db.execute(f"TRUNCATE TABLE {stage_table}")

    total_inserted = apply_state_updates(
        queued,
        inserted_per_file,
        state,
        state_path,
        state_flush_every=state_flush_every,
        progress_every=progress_every,
    )
    elapsed = (datetime.now(UTC) - started).total_seconds()
    rps = total_inserted / elapsed if elapsed > 0 else 0.0
    print(
        f"\n[summary] adapter={adapter_name} files_total={total_files} "
        f"files_queued={len(queued)} files_skipped={skipped_files} "
        f"inserted_total={total_inserted} bad_rows_total={bad_rows} "
        f"elapsed_s={elapsed:.2f} rps={rps:.1f}"
    )


def build_combined_tsv(
    files: list[dict],
    output_path: str,
    raw_columns: tuple[str, ...],
    stage_columns: tuple[str, ...],
    file_context: Callable[[str], dict[str, str]] | None = None,
    progress_every: int = DEFAULT_PROGRESS_EVERY_FILES,
) -> tuple[int, int, dict[str, int]]:
    """Build one combined TSVWithNames file for the staging insert.

    `raw_columns` is the column list of the source CSV files.
    `stage_columns` is the column list expected by the staging table; columns
    in `stage_columns` that are not in `raw_columns` are filled per-file via
    `file_context(path)`.

    Returns (rows_written, bad_rows, inserted_per_file).
    """

    extra_columns = tuple(c for c in stage_columns if c not in raw_columns)
    expected_tabs = len(raw_columns) - 1
    header = "\t".join(stage_columns)

    rows_written = 0
    bad_rows = 0
    per_file_inserted: dict[str, int] = {}
    total_files = len(files)

    with open(output_path, "w", encoding="utf-8", newline="\n") as out:
        out.write(header + "\n")

        for idx, task in enumerate(files, start=1):
            inserted_for_file = 0
            ctx_values = ""
            if extra_columns:
                ctx = file_context(task["path"]) if file_context else {}
                ctx_values = "\t".join(str(ctx.get(c, "") or "") for c in extra_columns) + "\t"

            with open(task["path"], encoding="utf-8") as src:
                first = True
                for raw_line in src:
                    line = raw_line.rstrip("\r\n")
                    if first:
                        first = False
                        continue
                    if not line:
                        continue
                    if line.count("\t") != expected_tabs:
                        bad_rows += 1
                        continue

                    out.write(f"{ctx_values}{line}\n")
                    inserted_for_file += 1

            per_file_inserted[task["file_key"]] = inserted_for_file
            rows_written += inserted_for_file

            if progress_every > 0 and (idx % progress_every == 0 or idx == total_files):
                print(
                    f"{progress_line(idx, total_files, 'build-tsv')} "
                    f"rows_buffered={rows_written} bad_rows={bad_rows}"
                )

    return rows_written, bad_rows, per_file_inserted


def apply_state_updates(
    queued: list[dict],
    inserted_per_file: dict[str, int],
    state: dict,
    state_path: str,
    state_flush_every: int = DEFAULT_STATE_FLUSH_EVERY,
    progress_every: int = DEFAULT_PROGRESS_EVERY_FILES,
) -> int:
    """Persist successful imports into the state file with rolling progress.

    Returns the total number of inserted rows (sum of `inserted_per_file`).
    """

    total_inserted = 0
    dirty_updates = 0
    total_files = len(queued)
    now_iso = datetime.now(UTC).isoformat()

    for index, task in enumerate(queued, start=1):
        inserted = inserted_per_file.get(task["file_key"], 0)
        total_inserted += inserted

        state["files"][task["file_key"]] = task["fingerprint"]
        state["last_run_at"] = now_iso
        state["total_inserted"] = int(state.get("total_inserted", 0)) + int(inserted)
        dirty_updates += 1
        dirty_updates = _flush_state_if_needed(state_path, state, dirty_updates, state_flush_every)

        if progress_every > 0 and (index % progress_every == 0 or index == total_files):
            print(
                f"{progress_line(index, total_files, 'state-update')} "
                f"inserted_total={total_inserted}"
            )

    _flush_state_if_needed(state_path, state, dirty_updates, state_flush_every, force=True)
    return total_inserted
