"""MQTT topic layout shared by the sensor emulator, the predictor and the dashboard."""

from __future__ import annotations

ROOT = "traffic/pems"


def measurement(station_id: str) -> str:
    """5-minute measurements. Same topic the live FTP publisher uses."""
    return f"{ROOT}/stations/{station_id}/5min"


def station_status(station_id: str) -> str:
    """Retained online/offline state of the sensor node (Last Will = offline)."""
    return f"{ROOT}/stations/{station_id}/status"


def control(station_id: str) -> str:
    """Commands to the sensor node: pause, resume, speed."""
    return f"{ROOT}/stations/{station_id}/control"


def forecast(station_id: str) -> str:
    """Latest 24 h forecast, retained so a late subscriber sees it at once."""
    return f"{ROOT}/stations/{station_id}/forecast"


def evaluation(station_id: str) -> str:
    """Forecast vs. actual for every completed 15-minute bucket."""
    return f"{ROOT}/stations/{station_id}/evaluation"


def predictor_status() -> str:
    return f"{ROOT}/predictor/status"
