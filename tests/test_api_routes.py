"""Offline contract tests for the FastAPI application and route inventory."""

from __future__ import annotations

from app.api.main import app
from app.api.models_local import MODEL_REGISTRY
from fastapi import FastAPI
from fastapi.testclient import TestClient

EXPECTED_API_ROUTES = {
    ("GET", "/api/map"),
    ("GET", "/api/models/local"),
    ("GET", "/api/models/local/forecast/{model_id}/chart"),
    ("GET", "/api/models/local/forecast/{model_id}/run"),
    ("GET", "/api/models/local/forecast/{model_id}/series"),
    ("GET", "/api/models/local/{model_id}/anomalies"),
    ("GET", "/api/models/local/{model_id}/compare"),
    ("GET", "/api/models/local/{model_id}/gt-eval"),
    ("GET", "/api/models/local/{model_id}/gt-events"),
    ("GET", "/api/models/local/{model_id}/location-chart"),
    ("POST", "/api/models/local/{model_id}/map-push"),
    ("GET", "/api/models/local/{model_id}/run"),
    ("GET", "/api/models/local/{model_id}/scores"),
    ("GET", "/api/models/local/{model_id}/series"),
    ("GET", "/api/datasets"),
    ("GET", "/api/traffic"),
    ("GET", "/api/traffic/classes"),
}


def test_app_imports_without_database_access():
    assert isinstance(app, FastAPI)
    assert app.title == "Traffic Monitor API"


def test_exact_api_route_inventory():
    routes = {
        (method, route.path)
        for route in app.routes
        if route.path.startswith("/api/")
        for method in route.methods
    }

    assert routes == EXPECTED_API_ROUTES
    assert all(path != "/api/stations" for _, path in routes)


def test_removed_routes_return_404_without_database_access():
    client = TestClient(app)

    assert client.get("/api/stations").status_code == 404
    assert client.get("/api/demo/mtad").status_code == 404
    assert client.get("/api/demo/timesnet").status_code == 404


def test_model_registry_contains_only_current_models():
    assert set(MODEL_REGISTRY) == {"timesnet"}
