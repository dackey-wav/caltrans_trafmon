# Static map data (`app/core/data`)

- **`dataset_a_stations_geo.json`** — synthetic WGS84 points per measurement station for `dataset-a`. Used by `GET /api/map?dataset=dataset-a`.
- **`dataset_a_map_anomalies.json`** — optional `by_station` map: station id (e.g. `"10001"`) → list of `{ id?, ds_utc?, summary?, severity?, location_ids? }`. Empty object shows badge `0` on all pins.

Regenerate neutral fixtures from repo root:

```bash
python scripts/generate_synthetic_fixtures.py
```

## Updating anomalies from a model run

1. **API (backend):** `POST /api/models/local/{model_id}/map-push?run_id=<id>&dataset=dataset-a&write=true` — aggregates anomalies from that run into `dataset_a_map_anomalies.json` (includes `source_run_id`).
