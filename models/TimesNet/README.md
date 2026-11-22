# TimesNet

Traffic forecaster for one Caltrans station: `prepare -> train -> evaluate -> compare`.

The model reads the last `window` steps of flow, speed and occupancy and
predicts the next `pred_len` steps of all three. Time marks (month, day,
weekday, hour, minute in Los Angeles time plus calendar flags such as holidays
and school breaks) cover the input window and the horizon, so the model knows
in advance that tomorrow is a holiday.

## Workflow

```bash
python -m app timesnet prepare --freq 15min --run-id sd15
python -m app timesnet train --run-id sd15 --name base
python -m app timesnet sweep --run-id sd15 --grid "window=192,672;d_model=64,128"
python -m app timesnet compare --run-id sd15
```

`train` evaluates on the test split right after training. Re-run the scoring
and plots on their own with `timesnet evaluate --run-id sd15 --name base`.

Defaults: window 2 days, horizon 1 day (15min: 192 and 96 steps, 5min: 576
and 288), Huber loss on robust-scaled targets, bf16 autocast on GPU.
`--target weekly_residual` trains the correction to the same time last week
and adds that week back at prediction time. The week-ago values of the whole
horizon are fed in as inputs, since they are known when the forecast is made.
`TIMESNET_DEVICE=cuda|cpu` overrides device selection.

## Splits and masking

Rows split chronologically 70/15/15 on whole days. The scaler and the list of
calendar marks come from train rows only; a calendar column that stays constant
in train is dropped. Steps with `observed_percent < 100` count as inactive:
the loss and all metrics skip them, and plots leave gaps there.

## Evaluation

Every test row serves as a forecast origin. The model competes with three
baselines on the same cells:

- persistence: the last value repeated over the horizon
- daily: the value 24 hours before each target step
- weekly: the value 7 days before each target step

`forecast_metrics_test.json` holds MAE and RMSE per channel overall, per lead
step and per lead bucket (0-1h, 1-4h, 4-24h), flow in veh/h, skill against
each baseline, and for 5min runs scores on 15-minute blocks so that 5min and
15min experiments compare on equal terms.

## Artifacts

```
artifacts/forecast/timesnet/{run_id}/
  dataset.parquet, dataset_meta.json
  experiments/{name}/
    model.pt, scaler.json, train_log.json
    forecast_metrics_test.json, forecast_test.parquet
    index.html                   report: summary, table, all plots
    plots/train_curve.png        loss and val MAE per channel
    plots/test_error_by_lead.png
    plots/test_daily_overview.png
    plots/test_first_week.png, test_last_week.png, test_worst_week.png,
    plots/test_school_break_week.png
    plots/test_example_NN.png    history plus 24h forecast from local midnight
artifacts/forecast/timesnet/compare_test.csv
```

Open `index.html` in a browser to see one experiment. `train_log.json` records
`peak_gpu_gb`, epoch time and parameter count for sizing larger runs.
