# Caltrans Trafmon

Traffic forecasting for one Caltrans PeMS detector, demonstrated as a small IoT
system: a sensor node streams 5-minute measurements over MQTT, a predictor node
forecasts the next 24 hours with TimesNet and checks every forecast against
what really happened, and a browser dashboard watches and controls both.

## Parts
- `app/iot` — sensor emulator (replays a downloaded station CSV), predictor
  node, Mosquitto config. See [app/iot/README.md](app/iot/README.md).
- `app/frontend` — React/Vite dashboard. The *Live IoT system* page is an MQTT
  client over WebSockets (simulated clock, speed, reset, device status,
  simulated outages, actual vs. forecast). The *Station & model* page shows data
  quality and the model's offline test results.
- `app/api` — FastAPI with four read-only routes (`/api/health`, `/api/config`,
  `/api/station`, `/api/model`) for facts that do not change while streaming.
- `models/TimesNet` — `prepare → train → evaluate → compare` forecasting
  pipeline; see [models/TimesNet/README.md](models/TimesNet/README.md).
- `app/importer`, `app/core` — CSV import into ClickHouse `traffic.traffic_5min`,
  the table the TimesNet prepare step reads.
- `app/live` — optional real-time PeMS FTP → MQTT → ClickHouse bridge (needs a
  separate PeMS FTP account); see [app/live/README.md](app/live/README.md).

## Run the demo
```bash
docker compose --profile iot up -d --build mqtt predictor sensor-emulator api frontend
```
Open http://localhost:5173. The replayed station CSV lives in `data/stations/`
(ignored by git) and the model in `artifacts/forecast/timesnet/`; both are
selected with `IOT_*` variables (see `docker-compose.yml`).

## Other tasks
- Database: copy `.env.example` to `.env`, set a ClickHouse user and password,
  `docker compose up -d clickhouse`. Import: `python -m app import-csv PATH/TO/file.csv`
  with `DB_HOST=localhost` and `PORT=8123`.
- GPU training shell: `docker compose --profile ml-gpu up -d dev-gpu`, then
  `docker compose --profile ml-gpu exec dev-gpu bash`. CLI: `python -m app timesnet --help`.
- Frontend checks in `app/frontend`: `npm ci`, `npm run build`, `npm test`, `npm run lint`.
- Python checks: `uv run pytest -m "not integration_ch and not ml_cpu"`.

## Limitations
- The deployed model was trained on station 400237 (District 4, US-101 N).
  Station 1108617 (District 11, I-5 N at Old Town, San Diego) has far fewer
  imputed rows and will replace it after retraining; the school-calendar
  features are still the Morgan Hill ones.
- "Large deviation" marks are a heuristic (15-minute-ahead miss above 6× the
  running error); there is no incident ground truth to validate them.
- Trained weights, downloaded data and secrets are not in git.
