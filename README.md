# Caltrans Trafmon

Independent copy of Traffic Monitor focused on TimesNet. The source repository is unchanged.

## Included
- React/Vite frontend: traffic charts, station map and TimesNet run viewer.
- FastAPI backend, CSV importer, ClickHouse schema and Docker Compose.
- TimesNet reconstruction pipeline, shared windowing/training utilities and related tests.
- Generic forecasting viewer retained for later integration; TimesNet forecasting is not implemented yet.
- Optional live PeMS 5-minute FTP → MQTT → ClickHouse bridge; see [app/live/README.md](app/live/README.md).

## Current limitations
The importer and TimesNet prepare step use the single-station `traffic.traffic_5min` table. The web API still reads the old per-vehicle feature tables, so its charts do not show this series yet. Calendar features and domain filters are still the inherited ones. Map fixtures are empty. Trained weights and secrets are not included. TimesNet reconstructs input windows; it does not forecast future traffic. PeMS station 1108617 and nearby CCTV C009 are documented in [app/live/README.md](app/live/README.md). A separate PeMS FTP account and ClickHouse credentials are required before the MQTT bridge can run against real data.

## Setup
Copy `.env.example` to `.env` and set `CLICKHOUSE_DB=traffic` plus a ClickHouse user and password.
`docker compose up -d clickhouse api frontend` starts the database and the web app. Init SQL runs on the first start of an empty volume.
Import from the host: `python -m app import-csv PATH/TO/file.csv` with `DB_HOST=localhost` and `PORT=8123`.
GPU training shell: `docker compose --profile ml-gpu up -d dev-gpu`, then `docker compose --profile ml-gpu exec dev-gpu bash`.
CLI: `python -m app --help`, `python -m app timesnet --help`.
Frontend checks: `npm ci`, `npm run build`, `npm test`, `npm run lint` in `app/frontend`.
Python checks: `uv run pytest -m "not integration_ch and not ml_cpu"`.

No `.git` history or remote is copied. Initialize a new repository when ready.
