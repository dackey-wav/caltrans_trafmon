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

## Station and model
- **Station:** PeMS VDS 1108617, I-5 northbound at Old Town, San Diego
  (District 11). 5-minute data 2024-11-01 → 2026-11-16, 97.7 % directly
  observed (not imputed by PeMS).
- **Calendar features:** California holidays plus local flags in
  [models/utils/local_calendar.py](models/utils/local_calendar.py): San Diego
  Unified school breaks, UC San Diego and University of San Diego terms,
  Padres home games, Del Mar fair/racing days, Comic-Con (sources in the file).
- **Model:** TimesNet, 2 days of 15-minute history → next 24 h of flow, speed
  and occupancy (`sd15/experiments/w192_compact_lr1e-4`). Test period
  2026-07-28 → 2026-11-16, MAE against "same time last week":

  | | flow, veh/h | speed, mph | occupancy |
  |---|---|---|---|
  | TimesNet | 316 | 1.41 | 0.84 % |
  | same time last week | 361 | 1.73 | 1.00 % |
  | improvement | 12 % | 18 % | 16 % |

## Run the demo
```bash
docker compose --profile iot up -d --build mqtt predictor sensor-emulator api frontend
```
Open http://localhost:5173. The replayed station CSV lives in `data/stations/`
(ignored by git) and the model in `artifacts/forecast/timesnet/`; both are
selected with `IOT_*` variables (see `docker-compose.yml`). The replay starts
at 2026-09-28 06:00 PT (test period) at ×300; use *New simulation* to pick
another moment.

## Reproduce the data and the model
1. Download with [pems-data-downloader](https://github.com/wcc961129/pems-data-downloader):
   `pems-data fetch --district 11 --station-id 1108617 --start 2024-11-01 --end 2026-11-16 --output …`
   (split the range around days PeMS does not list, e.g. 2025-11-28/29).
   Put `processed/observations.csv.gz` into `data/stations/1108617_5min.csv.gz`.
2. Import the unpacked CSV: `python -m app import-csv data/stations/1108617_5min.csv`.
3. Train in the GPU shell:
   `python -m app timesnet prepare --run-id sd15 --freq 15min`, then
   `python -m app timesnet train --run-id sd15 --name w192_compact_lr1e-4 --window 192 --pred-len 96 --mark-set compact --learning-rate 0.0001 --patience 5`.

## Other tasks
- Database: copy `.env.example` to `.env`, set a ClickHouse user and password,
  `docker compose up -d clickhouse`. Import: `python -m app import-csv PATH/TO/file.csv`
  with `DB_HOST=localhost` and `PORT=8123`.
- GPU training shell: `docker compose --profile ml-gpu up -d dev-gpu`, then
  `docker compose --profile ml-gpu exec dev-gpu bash`. CLI: `python -m app timesnet --help`.
- Frontend checks in `app/frontend`: `npm ci`, `npm run build`, `npm test`, `npm run lint`.
- Python checks: `uv run pytest -m "not integration_ch and not ml_cpu"`.

## Limitations
- The live feed is a replay of historical data. The real-time PeMS FTP bridge
  in `app/live` needs a separate PeMS FTP account and has not been run.
- The two-machine setup is documented but was tested on one PC (Docker);
  the broker allows anonymous clients until a password file is enabled.
- One model configuration was trained for the new station; no
  hyper-parameter sweep was repeated for it.
- "Large deviation" marks are a heuristic (15-minute-ahead miss above 6× the
  running error); there is no incident ground truth to validate them.
- Local calendar flags cover 2024-11 → 2026-12; later dates need new entries.
- Trained weights, downloaded data and secrets are not in git.
