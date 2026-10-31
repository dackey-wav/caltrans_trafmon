# TimesNet

Joint multivariate anomaly-detection pipeline (`prepare -> train -> eval`).

## GPU vs CPU

Optional: `TIMESNET_DEVICE=cuda` or `TIMESNET_DEVICE=cpu`.

## Workflow

```bash
python -m app.cli timesnet prepare --date-from 2026-03-01 --date-to 2026-04-01
python -m app.cli timesnet train --run-id <run_id>
python -m app.cli timesnet eval --run-id <run_id> --threshold-profile balanced
```

Scoring uses **window-mean reconstruction RMSE** on target columns (same objective as training).
Re-run `eval` after scoring fixes; retrain only if you change model hyperparameters.

## Threshold profiles

Default eval uses a **combined statistical gate** (percentile `q=0.999` AND
robust-z `>= 6.0`) plus an hourly intensity gate:

- traffic intensity: `|actual - expected| / expected >= 20%` on flow, speed, or occupancy

Profiles (override robust-z only):

- `balanced`: robust-z `threshold=6.0`
- `strict`: percentile `quantile=0.9995`
- `sensitive`: robust-z `threshold=5.0`

`anomalies.json` — domain-filtered output (plots + map).  
`anomalies_stat.json` — combined statistical gate only (debug).

## Artifacts

`artifacts/anomaly/timesnet/{run_id}/` — `dataset.parquet`, `model.pt`, `scores.parquet`, `anomalies.json`, `comparison.json`, `plots/`.

## Compare runs

```bash
python -m models.TimesNet.compare_runs --runs run_a,run_b
```

## Regression harness

1. Smoke: 3 locations, 7–14 days, `freq=1h`
2. Hourly all locations
3. 15-min stress (short range)
## Per-location timelines (missing data)

`prepare` writes `active__<location_id>` columns: a timestep is **active** only
when that location has an observed bucket in ClickHouse. Hours outside the
first/last span and **internal gaps** (e.g. a multi-month outage) are excluded
from training loss, scoring, and plots. Gaps are forward-filled in the target
tensor for the forward pass only; they are not treated as zero traffic.

Train/val/test are assigned **per location** on active hours only (default
70/15/15). After changing masking logic, re-run `prepare`, then `train`, then
`eval`.

