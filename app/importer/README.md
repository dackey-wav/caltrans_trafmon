# app/importer

`importer_csv/` loads one comma-separated 5-minute station file into ClickHouse:

```bash
python -m app import-csv PATH/TO/file.csv
```

`INPUT` may name one CSV file or a directory. Directory discovery is recursive
and case-insensitive for the `.csv` suffix. There is one format. Columns that
stay constant for the whole file (station, freeway, coordinates, direction,
lane type) are dropped. `timestamp` is read as California local time
(`America/Los_Angeles`) and stored in `traffic.traffic_5min`.

| Target table | Adapter |
|--------------|---------|
| `traffic_5min` | `formats/station_5min.py` |

`formats/registry.py` owns the immutable format registry, destination mapping,
default state path, adapter, and sample validator. Preflight finishes before
the importer creates a ClickHouse client.

### Architecture: ClickHouse-native bulk load

Every import goes through one path:

1. **Scan + state filter** (`pipeline.scan_files`). Walk the input
   recursively, fingerprint each `.csv` (size + mtime) and skip files
   already recorded in the state file.
2. **Build one TSV stream** (`pipeline.build_combined_tsv` or a
   format-aware builder that remaps source headers to a canonical column
   order). The stream is `TSVWithNames`: a header row followed by all data
   rows, tab-delimited.
3. **Bulk-upload to a staging table** via `client.raw_insert(...)`.
   Staging tables are plain `MergeTree` with columns typed `String` —
   letting ClickHouse parse text in C++ later.
4. **Transform with one `INSERT ... SELECT`** from staging into the final
   table. Parsing, casts, timezone conversion, hashing, and derived columns
   happen on the ClickHouse side.
5. **Update state** (`pipeline.apply_state_updates`) only after a
   successful transform. The staging table is then truncated.

See `pipeline.py` and `formats/` for format-specific details.

### State file

State keys are absolute, normalized paths. Use `--no-resume` to force a
full re-import of matching files.

### Tuning knobs (env)

| Variable | Purpose | Default |
|----------|---------|---------|
| `CSV_STATE_FLUSH_EVERY` | Persist state every N file updates. | 100 |
| `CSV_NATIVE_PROGRESS_EVERY_FILES` | Print progress lines every N files. | 200 |
