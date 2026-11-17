# IoT demo: sensor node → MQTT → predictor node

Two logical devices exchange data only through an MQTT broker (Mosquitto):

| Node | Code | Role |
|---|---|---|
| sensor | `app/iot/sensor_emulator.py` | replays a downloaded PeMS station CSV as a live 5-minute feed |
| predictor | `app/iot/predictor.py` | folds the feed into 15-minute buckets, runs TimesNet, publishes a 24 h forecast and checks every past forecast against the facts |
| broker | `mqtt` service, `app/iot/mosquitto.conf` | MQTT on 1883, MQTT over WebSockets on 9001 (for a browser dashboard) |

The live PeMS FTP bridge in `app/live` publishes the same `5min` message, so
the replay emulator and a real feed are interchangeable sources.

## Topics

| Topic | Direction | QoS / retain | Payload |
|---|---|---|---|
| `traffic/pems/stations/<id>/5min` | sensor → predictor | 1 | one 5-minute measurement (`app.live.feed.Measurement` schema + `ts_utc`, `seq`, `sent_at_ms`) |
| `traffic/pems/stations/<id>/control` | anyone → sensor | 1 | `{"cmd":"pause"}`, `{"cmd":"resume"}`, `{"cmd":"speed","value":600}` |
| `traffic/pems/stations/<id>/forecast` | predictor → dashboard | 1, retained | next 96 × 15 min of flow (veh/15 min), speed (mph), occupancy (fraction) |
| `traffic/pems/stations/<id>/evaluation` | predictor → dashboard | 1 | actual bucket vs. forecasts made 15 min / 1 h / 4 h / 24 h before, weekly baseline, anomaly flag |
| `traffic/pems/stations/<id>/status`, `traffic/pems/predictor/status` | nodes → all | 1, retained | `online` on every connect, `offline` as Last Will |

## Reliability

- QoS 1 everywhere. The predictor uses a fixed client ID with a persistent
  session (`clean_session=False`) and acknowledges a message only after it has
  processed it, so the broker keeps and redelivers whatever the predictor
  missed while it was down.
- Redelivered or repeated measurements are dropped by timestamp. Lost
  messages and fully missing buckets become gaps (inactive steps), exactly as
  in training; a forecast needs at least half of its 2-day window observed.
- The predictor snapshots its state every 10 s to `artifacts/iot/state_<id>.pkl`
  and resumes from it, so a restart needs no new 2-day warm-up. A jump back in
  time by more than an hour (replay restarted) resets the state.
- The sensor holds the stream while the broker is unreachable and continues
  after reconnecting. Status topics are re-announced on every connect.

Tested on one PC (Docker): 5 % message loss, `docker kill` of the predictor
mid-stream and a broker restart. No processing errors, forecasts resumed
right after each fault.

## Run on one PC

```bash
docker compose --profile iot up -d --build mqtt predictor
docker compose --profile iot up sensor-emulator
```

Defaults replay station 400237 from `data/stations/400237_5min.csv.gz` (the
`processed/observations.csv.gz` of pems-data-downloader), starting
2025-06-02 (test split of the bundled model) at 300× speed after an 8-day
warm-up burst. Override with `IOT_*` variables (see `docker-compose.yml`), e.g.
`IOT_REPLAY_SPEED=1500 IOT_REPLAY_END=2025-06-08T23:59`.

Watch the traffic as a third client:

```bash
docker compose exec mqtt mosquitto_sub -t 'traffic/pems/#' -v
docker compose exec mqtt mosquitto_pub -t traffic/pems/stations/400237/control -q 1 -m '{"cmd":"speed","value":1200}'
```

The predictor writes `artifacts/iot/evaluations_<id>.jsonl`,
`forecasts_<id>.jsonl` and `summary_<id>.json` (MAE per lead bucket against
the weekly baseline, MQTT latency, inference time).

## Run on two machines

1. Broker + predictor on machine B: `docker compose --profile iot up -d mqtt predictor`.
   Before opening port 1883 to the LAN, enable a password file in
   `app/iot/mosquitto.conf`.
2. Sensor on machine A: set `IOT_MQTT_HOST=<IP of B>` (and `IOT_MQTT_USERNAME`,
   `IOT_MQTT_PASSWORD`), then `docker compose --profile iot up --no-deps sensor-emulator`.

## Limits

- The bundled model was trained on station 400237 (District 4). It must be
  retrained before it forecasts another station; its school-calendar flags
  are Morgan Hill specific.
- Anomaly = a 15-minute-ahead miss larger than `ANOMALY_Z` (default 6) times
  the stream's running MAE, after one day of scored buckets. It flags
  unusual traffic, not detector faults.
