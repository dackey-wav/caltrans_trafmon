# Caltrans Trafmon

Independent copy of Traffic Monitor focused on TimesNet. The source repository is unchanged.

## Included
- React/Vite frontend: traffic charts, station map and TimesNet run viewer.
- FastAPI backend, CSV importer, ClickHouse schema and Docker Compose.
- TimesNet reconstruction pipeline, shared windowing/training utilities and related tests.
- Generic forecasting viewer retained for later integration; TimesNet forecasting is not implemented yet.

## Current limitations
The importer loads one 5-minute station CSV into `traffic.traffic_5min`. The web API and TimesNet prepare step still read the old per-vehicle feature tables, so charts and training do not see this series yet. Calendar features and domain filters are still the inherited ones. Map fixtures are empty. Trained weights and secrets are not included. TimesNet reconstructs input windows; it does not forecast future traffic.

## Setup
Copy `.env.example` to `.env` and set `CLICKHOUSE_DB=traffic` plus a ClickHouse user and password.
`docker compose up -d clickhouse api frontend` starts the database and the web app. Init SQL runs on the first start of an empty volume.
Import from the host: `python -m app import-csv PATH/TO/file.csv` with `DB_HOST=localhost` and `PORT=8123`.
GPU training shell: `docker compose --profile ml-gpu up -d dev-gpu`, then `docker compose --profile ml-gpu exec dev-gpu bash`.
CLI: `python -m app --help`, `python -m app timesnet --help`.
Frontend checks: `npm ci`, `npm run build`, `npm test`, `npm run lint` in `app/frontend`.
Python checks: `uv run pytest -m "not integration_ch and not ml_cpu"`.

No `.git` history or remote is copied. Initialize a new repository when ready.
