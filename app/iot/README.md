# IoT demo: sensor node → MQTT → predictor node → dashboard

Three MQTT clients exchange data only through a broker (Mosquitto):

| Node | Code | Role |
|---|---|---|
| sensor (device 1) | `app/iot/sensor_emulator.py` | replays a downloaded PeMS station CSV as a live 5-minute feed on a simulated clock |
| predictor (device 2) | `app/iot/predictor.py` | folds the feed into 15-minute buckets, runs TimesNet, publishes a 24 h forecast and checks every past forecast against the facts |
| dashboard | `app/frontend/src/live` | browser client over WebSockets: shows both nodes and sends them commands |
| broker | `mqtt` service, `app/iot/mosquitto.conf` | MQTT on 1883, MQTT over WebSockets on 9001 |

The live PeMS FTP bridge in `app/live` publishes the same `5min` message, so
the replay emulator and a real feed are interchangeable sources.

## Topics

All under `traffic/pems/`. `<id>` is the station ID.

| Topic | From → to | QoS / retain | Payload |
|---|---|---|---|
| `stations/<id>/5min` | sensor → predictor, dashboard | 1 | one 5-minute measurement (`app.live.feed.Measurement` schema + `ts_utc`, `session`, `seq`, `sent_at_ms`) |
| `stations/<id>/replay` | sensor → dashboard | 1, retained | session, phase, `speed`, `sim_time` at `wall_time_ms` (the dashboard runs the same clock), sent/lost/buffered counts |
| `stations/<id>/control` | dashboard → sensor | 1 | `pause`, `resume`, `speed`, `reset` (start, speed), `outage` (seconds, `buffer`/`drop`), `loss` (probability) |
| `stations/<id>/forecast` | predictor → dashboard | 1, retained | next 96 × 15 min of flow (veh/15 min), speed (mph), occupancy (fraction) |
| `stations/<id>/evaluation` | predictor → dashboard | 1 | one bucket: actual vs. forecasts made 15 min / 1 h / 4 h / 24 h before, last week's value, deviation flag |
| `stations/<id>/history` | predictor → dashboard | 1, retained | last two days of evaluations, so a dashboard opened mid-stream draws at once |
| `stations/<id>/metrics` | predictor → dashboard | 1, retained | counters, MQTT latency, inference time, MAE per lead |
| `predictor/control` | dashboard → predictor | 1 | `outage` (seconds) |
| `stations/<id>/status`, `predictor/status` | nodes → all | 1, retained | `online` on every connect; `offline` with a reason on a clean stop or simulated outage, without one as Last Will after a crash |

## Simulation

- Simulated time runs `speed` times faster than wall time (×1 … ×3600). A
  reset starts a new *session*: the sensor sends the 8 days before the start
  as a burst (TimesNet needs 2 days of input, the baseline a week), then
  streams from the start. The predictor clears its state when the session
  changes and only forecasts the last ~26 h of the burst, so it is ready in a
  few seconds.
- The dashboard shows the station's local time (PT) from the sensor's clock,
  the raw 5-minute readings it receives itself, the scored 15-minute actuals,
  the forecast made N hours earlier and the current 24 h forecast.

## Failure scenarios (buttons on the dashboard)

| Scenario | What happens | What to look at |
|---|---|---|
| Sensor outage · buffer | Sensor disconnects, keeps measuring into a local buffer, reconnects after 30 s and sends the backlog | sensor goes red with a countdown; raw line pauses; backlog flushed; no gap in the actuals |
| Sensor outage · lose data | Readings taken while offline are dropped | 30 s × speed of missing data; those buckets are not scored |
| Message loss 5–20 % | Random single messages lost | incomplete buckets become gaps; forecasts continue |
| Predictor outage | Predictor disconnects for 30 s; the broker keeps the feed in its persistent session | "Behind sensor" grows, then the queued messages arrive and it catches up |
| `docker compose kill predictor`, then `start` | Real crash: Last Will marks it offline; state resumes from the snapshot; unacknowledged messages are redelivered | |
| `docker compose restart mqtt` | Broker restart | all three clients reconnect; the sensor buffers meanwhile |

## Reliability details

- QoS 1 everywhere. The predictor uses a fixed client ID with a persistent
  session and acknowledges a measurement only after processing it.
  Redelivered or repeated measurements are dropped by timestamp.
- Lost messages and fully missing buckets become inactive steps, as in
  training; a forecast needs at least half of its 2-day window observed.
- The predictor snapshots its state (including the dashboard history) every
  10 s to `artifacts/iot/state_<id>.pkl` and resumes from it.

## Run

```bash
docker compose --profile iot up -d --build mqtt predictor sensor-emulator api frontend
```

Open http://localhost:5173. Defaults replay station 400237 from
`data/stations/400237_5min.csv.gz` (the `processed/observations.csv.gz` of
pems-data-downloader) from 2025-06-02 06:00 at ×300. Override with `IOT_*`
variables in `.env` or the shell (see `docker-compose.yml`).

Watch or drive it from a terminal as another client:

```bash
docker compose exec mqtt mosquitto_sub -t 'traffic/pems/#' -v
docker compose exec mqtt mosquitto_pub -t traffic/pems/stations/400237/control -q 1 -m '{"cmd":"speed","value":1200}'
```

The predictor also writes `artifacts/iot/evaluations_<id>.jsonl`,
`forecasts_<id>.jsonl` and `summary_<id>.json`.

## Two machines

1. Broker + predictor on machine B: `docker compose --profile iot up -d mqtt predictor`.
   Before opening ports 1883/9001 to the LAN, enable a password file in
   `app/iot/mosquitto.conf`.
2. Sensor on machine A: set `IOT_MQTT_HOST=<IP of B>` (and `IOT_MQTT_USERNAME`,
   `IOT_MQTT_PASSWORD`), then `docker compose --profile iot up -d --no-deps sensor-emulator`.
3. Dashboard anywhere: set `IOT_MQTT_WS_URL=ws://<IP of B>:9001` for the API.

## Limits

- The bundled model was trained on station 400237 (District 4) and must be
  retrained before it forecasts another station.
- A "large deviation" is a 15-minute-ahead miss larger than `ANOMALY_Z`
  (default 6) times the running MAE after one day of scored buckets. There is
  no incident ground truth, so it is off by default in the dashboard.
