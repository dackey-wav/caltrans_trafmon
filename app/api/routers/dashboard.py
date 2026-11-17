"""Station data coverage and the deployed model's offline test results."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api")

REPO = Path(__file__).resolve().parents[3]
EXPERIMENTS = REPO / "artifacts" / "forecast" / "timesnet"
LOCAL_TZ = "America/Los_Angeles"


def _station_id() -> str:
    return os.getenv("IOT_STATION_ID", "400237")


def _replay_file() -> Path:
    data_dir = Path(os.getenv("IOT_DATA_DIR", "data/stations"))
    if not data_dir.is_absolute():
        data_dir = REPO / data_dir
    return data_dir / os.getenv("IOT_REPLAY_FILE", f"{_station_id()}_5min.csv.gz")


def _model_dir() -> Path:
    name = os.getenv(
        "IOT_MODEL", "ds15/experiments/sweep_window192_mark_setcompact_learning_rate0.0001"
    )
    return EXPERIMENTS / name


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/config")
def config() -> dict:
    """What the browser needs to join the MQTT side."""
    return {
        "station_id": _station_id(),
        "mqtt_ws_url": os.getenv("IOT_MQTT_WS_URL") or None,
        "topic_root": "traffic/pems",
        "model": _model_dir().name,
    }


@lru_cache(maxsize=4)
def _station_summary(path: str, mtime: float) -> dict:
    df = pd.read_csv(path)
    df = df[df["station_id"].astype(str) == _station_id()]
    if df.empty:
        raise HTTPException(404, f"station {_station_id()} not in {Path(path).name}")
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(LOCAL_TZ)
    observed = df["observed_percent"].fillna(0) >= 100
    month = ts.dt.strftime("%Y-%m")
    monthly = (
        pd.DataFrame({"month": month, "observed": observed})
        .groupby("month")
        .agg(rows=("observed", "size"), observed=("observed", "mean"))
        .reset_index()
    )
    first = df.iloc[0]
    return {
        "station_id": _station_id(),
        "file": Path(path).name,
        "latitude": float(first.get("latitude")) if "latitude" in df else None,
        "longitude": float(first.get("longitude")) if "longitude" in df else None,
        "freeway": str(first.get("freeway", "")),
        "direction": str(first.get("direction", "")),
        "lane_type": str(first.get("lane_type", "")),
        "rows": int(len(df)),
        "first": ts.iloc[0].isoformat(),
        "last": ts.iloc[-1].isoformat(),
        "observed_share": round(float(observed.mean()), 4),
        "monthly": [
            {"month": r.month, "rows": int(r.rows), "observed_share": round(float(r.observed), 4)}
            for r in monthly.itertuples()
        ],
    }


@router.get("/station")
def station() -> dict:
    path = _replay_file()
    if not path.exists():
        raise HTTPException(404, f"replay file {path.name} not found")
    return _station_summary(str(path), path.stat().st_mtime)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found for model {path.parent.name}")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/model")
def model() -> dict:
    """Training setup and test-split errors of the model the predictor runs."""
    exp = _model_dir()
    log = _read_json(exp / "train_log.json")
    metrics = _read_json(exp / "forecast_metrics_test.json")
    hp = log["hparams"]
    methods = list(metrics["overall"])
    return {
        "name": exp.name,
        "run_id": log.get("run_id"),
        "hparams": {k: hp.get(k) for k in (
            "freq", "window", "pred_len", "d_model", "e_layers", "learning_rate",
            "batch_size", "loss", "mark_set", "target",
        )},
        "parameters": log.get("n_parameters"),
        "best_epoch": log.get("best_epoch"),
        "epochs_run": log.get("epochs_run"),
        "train_seconds": log.get("train_seconds"),
        "device": log.get("device"),
        "split_ds": log.get("split_ds"),
        "marks": log.get("feature_plan", {}).get("mark_columns", []),
        "test": {
            "units": {"flow": "veh/15min", "speed": "mph", "occupancy": "fraction"},
            "methods": methods,
            "n_origins": metrics.get("n_origins"),
            "overall": {m: {c: v["mae"] for c, v in metrics["overall"][m].items()} for m in methods},
            "by_lead_bucket": {
                bucket: {m: {c: v["mae"] for c, v in per_method.items()}
                         for m, per_method in methods_.items()}
                for bucket, methods_ in metrics["by_lead_bucket"].items()
            },
            "lead_minutes": metrics["lead_minutes"],
            "by_lead": {m: {c: v["mae"] for c, v in metrics["by_lead"][m].items()} for m in methods},
            "skill_vs_weekly": metrics.get("skill_vs_weekly"),
        },
    }
