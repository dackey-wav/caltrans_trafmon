from __future__ import annotations

from fastapi import APIRouter, Query

from app.api import models_local

router = APIRouter()


@router.get("/api/models/local")
def get_models_local():
    """List local artifact runs grouped by model (TimesNet)."""
    return models_local.list_local_models()


@router.get("/api/models/local/forecast/{model_id}/run")
def get_models_local_forecast_run(
    model_id: str,
    run_id: str = Query(..., description="Run path, e.g. run_v1"),
):
    """Manifest for one forecasting run (macro metrics, scopes, anomalies)."""
    return models_local.get_forecast_run_manifest(run_id, model_id)


@router.get("/api/models/local/forecast/{model_id}/series")
def get_models_local_forecast_series(
    model_id: str,
    run_id: str = Query(..., description="Run path"),
    scope: str | None = Query(
        None, description="Prediction scope: model id or baseline (default: model)"
    ),
):
    """Per-series MAE/RMSE/sMAPE table with residual anomaly counts."""
    return models_local.get_forecast_series(run_id, model_id, scope=scope)


@router.get("/api/models/local/forecast/{model_id}/chart")
def get_models_local_forecast_chart(
    model_id: str,
    run_id: str = Query(..., description="Run path"),
    unique_id: str = Query(..., description="Series id from per_series"),
    scope: str | None = Query(None, description="Prediction scope (default: model)"),
    max_points: int = Query(2500, ge=50, le=8000),
    window: str | None = Query(
        "full",
        description="Chart window: full | test | tail7d | tail30d (filters before downsampling)",
    ),
):
    """Actual vs forecast timeline (+ baseline overlay, residual anomalies)."""
    return models_local.get_forecast_chart(
        run_id,
        model_id,
        unique_id,
        scope=scope,
        max_points=max_points,
        window=window,
    )


@router.get("/api/models/local/{model_id}/run")
def get_models_local_run(
    model_id: str,
    run_id: str = Query(..., description="Run path, e.g. v5/weather_calendar"),
):
    """Manifest for one anomaly run (comparison, metrics, dataset meta)."""
    return models_local.get_timesnet_run_manifest(run_id, model_id=model_id)


@router.get("/api/models/local/{model_id}/series")
def get_models_local_series(
    model_id: str,
    run_id: str = Query(..., description="Run path, e.g. v5/weather_calendar"),
    mode: str | None = Query(
        "combined",
        description="Anomaly mode: percentile | combined",
    ),
):
    """Per-series summary table for a run."""
    return models_local.get_timesnet_run_series(run_id, mode=mode, model_id=model_id)


@router.get("/api/models/local/{model_id}/scores")
def get_models_local_scores(
    model_id: str,
    run_id: str = Query(..., description="Run path, e.g. v5/weather_calendar"),
    unique_id: str = Query(..., description="Target series id from anomalies per_series"),
    max_points: int = Query(2500, ge=50, le=8000),
    split: str | None = Query(
        None,
        description="Chronological split filter (omit for full timeline with split_ranges)",
    ),
    window: str | None = Query(
        "full",
        description="Chart window: full | test | tail7d | tail30d (filters before downsampling)",
    ),
    mode: str | None = Query(
        "combined",
        description="Anomaly mode: percentile | combined",
    ),
):
    """Downsampled score timeline for ECharts (annotated with thresholds)."""
    return models_local.get_timesnet_series_scores(
        run_id,
        unique_id,
        max_points=max_points,
        split=split or None,
        window=window,
        mode=mode,
        model_id=model_id,
    )


@router.get("/api/models/local/{model_id}/location-chart")
def get_models_local_location_chart(
    model_id: str,
    run_id: str = Query(..., description="Run path, e.g. v5/weather_calendar"),
    location_id: str = Query(
        ...,
        description="Base location id (lane), e.g. 24005-cze-l1 — all __clsN series",
    ),
    max_points: int = Query(2500, ge=50, le=8000),
    split: str | None = Query(
        None,
        description="Split filter (omit for full timeline with split_ranges)",
    ),
    window: str | None = Query(
        "full",
        description="Chart window: full | test | tail7d | tail30d (filters before downsampling)",
    ),
    mode: str | None = Query(
        "combined",
        description="Anomaly mode: percentile | combined",
    ),
):
    """Per-location stacked actual/predicted class chart with anomaly markers."""
    return models_local.get_timesnet_location_chart(
        run_id,
        location_id,
        max_points=max_points,
        split=split or None,
        window=window,
        mode=mode,
        model_id=model_id,
    )


@router.get("/api/models/local/{model_id}/anomalies")
def get_models_local_anomalies(
    model_id: str,
    run_id: str = Query(..., description="Run path, e.g. v5/weather_calendar"),
    unique_id: str | None = Query(None),
    limit: int = Query(200, ge=1, le=5000),
    mode: str | None = Query(
        "combined",
        description="Anomaly mode: percentile | combined",
    ),
    kind: str | None = Query(
        None,
        description="Filter by anomaly_kind: macro | micro",
    ),
):
    """Flagged anomaly records from anomalies JSON for the selected mode."""
    return models_local.get_timesnet_series_anomalies(
        run_id,
        unique_id=unique_id,
        limit=limit,
        mode=mode,
        kind=kind,
        model_id=model_id,
    )






@router.get("/api/models/local/{model_id}/gt-events")
def get_models_local_gt_events(
    model_id: str,
    location_prefix: str | None = Query(
        None,
        description="Filter by location prefix or gt_location_key (e.g. 24008-wis)",
    ),
    date_from: str | None = Query(None, description="Inclusive date_local YYYY-MM-DD"),
    date_to: str | None = Query(None, description="Inclusive date_local YYYY-MM-DD"),
):
    """Confirmed operational events from models/data/gt_anomalies/gt_events.csv."""
    return models_local.get_gt_events(
        location_prefix=location_prefix,
        date_from=date_from,
        date_to=date_to,
    )


@router.get("/api/models/local/{model_id}/gt-eval")
def get_models_local_gt_eval(
    model_id: str,
    run_id: str = Query(..., description="Run path with gt_eval.json"),
):
    """Ground-truth hit/miss report produced by eval --gt-eval."""
    return models_local.get_gt_eval(run_id, model_id=model_id)


@router.get("/api/models/local/{model_id}/compare")
def get_models_local_compare(
    model_id: str,
    runs: str = Query(..., description="Comma-separated run ids (2–4), first = baseline"),
):
    """Compare eval summaries across runs (delta anomaly rate vs baseline)."""
    run_ids = [run_id.strip() for run_id in runs.split(",") if run_id.strip()]
    return models_local.compare_timesnet_runs(run_ids, model_id=model_id)


@router.post("/api/models/local/{model_id}/map-push")
def post_models_local_map_push(
    model_id: str,
    run_id: str = Query(..., description="Run path to export, e.g. v5/weather_calendar"),
    dataset: str = Query("dataset-a", description="Map dataset (dataset-a only for now)"),
    write: bool = Query(True, description="Write dataset_a_map_anomalies.json on disk"),
    mode: str | None = Query(
        "combined",
        description="Anomaly mode for export: percentile | combined",
    ),
):
    """Aggregate run anomalies by station and push to static map config."""
    return models_local.push_run_anomalies_to_map(
        run_id, dataset=dataset, write=write, mode=mode, model_id=model_id
    )
