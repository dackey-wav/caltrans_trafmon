# app/core

Shared infrastructure used by every other module in `app/`:

- `config.py`        — loads + validates secrets and env vars (Pydantic).
- `db.py`            — single ClickHouse client wrapper.
- `init/init.sql`    — initial database schema (auto-applied by Docker).
- `init/apply_init.sh` — apply schema manually when the data volume already exists.

## Secrets and environment variables

All settings come from `.env` (template: `.env.example`). They are loaded
once on import via `app.core.config` and validated with Pydantic. Missing
or invalid values raise `RuntimeError` at startup, which is the desired
behavior - we don't want a half-configured importer to silently use empty
strings.

| Variable             | Purpose |
|----------------------|---------|
| `CLICKHOUSE_DB`      | Target database name. Blank falls back to `traffic`. |
| `CLICKHOUSE_USER`    | ClickHouse user. |
| `CLICKHOUSE_PASSWORD`| ClickHouse password. |
| `DB_HOST`            | ClickHouse host. `localhost` from the host machine, `clickhouse` inside Compose. |
| `PORT`               | ClickHouse HTTP port. Blank falls back to `8123`. |

There is no automated way to keep these descriptions in sync with `.env` —
when a new variable is added, update `.env.example`, this table, and
`Settings` in `config.py` together.

## `db.py` — ClickHouse client

`ClickHouseDB` is a thin wrapper over `clickhouse_connect`. The goal is
deliberately small: one place to construct a client, raise predictable
exception types, and offer the few patterns we actually use.

### Why a wrapper

We hit three problems with raw `clickhouse_connect`:

1. The driver raises a flat `ClickHouseError` for very different things
   (network, SQL syntax, server). Callers ended up wrapping every call
   anyway. The wrapper translates them once into `ConnectionError`,
   `ValueError`, or `RuntimeError`.
2. Every module was importing `cfg` and reconstructing host/port/db, with
   subtle drift. `ClickHouseDB.from_config()` is the single source of truth.
3. We needed a context-manager API (`with ClickHouseDB.from_config() as db:`)
   so workers and one-off scripts could not leak HTTP connections.

### Using it

```python
from app.core.db import ClickHouseDB

with ClickHouseDB.from_config() as db:
    cols, rows = db.query(
        "SELECT count() FROM traffic.traffic_5min"
    )
    db.execute("OPTIMIZE TABLE traffic.traffic_5min FINAL")
    db.insert_many(
        table="traffic.traffic_5min",
        columns=["ts", "speed_mph", "flow_veh_5min", "occupancy_fraction", "observed_percent"],
        data=[(...), (...)],
    )
```

Method choice cheat sheet:

| Method        | When to use it |
|---------------|----------------|
| `query(sql)`  | 90% of reads — returns `(columns, rows)`. |
| `query_raw(sql)` | When you need the driver's `result_set`/`first_row`/`summary`. |
| `execute(sql)`| DDL/DML and any non-`SELECT` statement (CREATE, INSERT…SELECT, TRUNCATE, OPTIMIZE). |
| `insert_many(table, columns, data)` | Row-tuple bulk insert for small application batches. |
| `client.raw_insert(table, insert_block, fmt="TSVWithNames")` | Bulk format upload (CSV native importer uses this). |

For massive bulk loads we **don't** use `insert_many` — it pays for the
Python row tuple representation. Instead the CSV importer streams a TSV
file straight into a staging table via `client.raw_insert(...)`. See
`app/importer/README.md`.

## `init/init.sql` — schema

This file is mounted into the ClickHouse container's
`/docker-entrypoint-initdb.d/`, so it runs **once** on first start (when
the data volume is empty). ClickHouse skips init scripts when it finds an
existing database directory in the volume.

To apply schema on openSUSE/Linux after a skipped init:

1. Fresh volume: `docker compose down -v && docker compose up -d clickhouse`
2. Re-run on every start after editing SQL: set `CLICKHOUSE_ALWAYS_RUN_INITDB_SCRIPTS=1` in `.env`

### Tables

#### `traffic.traffic_5min`

One row per 5-minute timestamp for the station file.

```sql
ENGINE = ReplacingMergeTree(ingest_version)
PARTITION BY toYYYYMM(ts)
ORDER BY (ts)
```

Stored columns: `ts`, `speed_mph`, `flow_veh_5min`, `occupancy_fraction`,
`observed_percent`, `ingest_version`. `ts` is `DateTime64(0, 'America/Los_Angeles')`.
The importer runs `OPTIMIZE TABLE ... FINAL` after a load so a re-import of the
same timestamp keeps the latest `ingest_version`.

### Editing the schema

The file is **idempotent only on a clean DB**. If you change a column,
the `CREATE TABLE IF NOT EXISTS` will silently keep the old definition
on existing volumes. Either:

1. Drop the volume, restart Docker (preferred during development), or
2. Apply the change as `ALTER TABLE ...` manually and update `init.sql`
   to match for fresh installs.
