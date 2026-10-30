"""Shared pytest fixtures (ClickHouse integration)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_SQL = REPO_ROOT / "app" / "core" / "init" / "init.sql"

INTEGRATION_TABLES = (
    "traffic_5min",
    "traffic_5min_stage",
)


def _clickhouse_env_ready() -> bool:
    host = os.getenv("DB_HOST", "").strip()
    db = os.getenv("CLICKHOUSE_DB", "").strip()
    return bool(host and db)


def _split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    buf: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buf.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(buf).rstrip().rstrip(";"))
            buf = []
    if buf:
        statements.append("\n".join(buf))
    return statements


@pytest.fixture(scope="session")
def clickhouse_integration_available() -> bool:
    return _clickhouse_env_ready()


@pytest.fixture(scope="session")
def clickhouse_session(clickhouse_integration_available: bool):
    if not clickhouse_integration_available:
        pytest.skip("Set DB_HOST and CLICKHOUSE_DB to run integration_ch tests")

    import clickhouse_connect

    host = os.environ["DB_HOST"]
    port = int(os.getenv("PORT", "8123"))
    user = os.getenv("CLICKHOUSE_USER", "default")
    password = os.getenv("CLICKHOUSE_PASSWORD", "")
    database = os.environ["CLICKHOUSE_DB"]

    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=user,
        password=password,
        database=database,
    )
    try:
        client.command("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ClickHouse not reachable at {host}:{port}: {exc}")

    sql = INIT_SQL.read_text(encoding="utf-8")
    for stmt in _split_sql_statements(sql):
        client.command(stmt)

    yield {
        "client": client,
        "database": database,
        "host": host,
        "port": port,
        "user": user,
        "password": password,
    }
    client.close()


@pytest.fixture
def clickhouse_db(clickhouse_session, monkeypatch):
    """Apply schema once per session; truncate data tables before each test."""

    import app.core.config as cfg

    session = clickhouse_session
    client = session["client"]
    db_name = session["database"]

    monkeypatch.setattr(cfg, "HOST", session["host"])
    monkeypatch.setattr(cfg, "PORT", session["port"])
    monkeypatch.setattr(cfg, "DB_USER", session["user"])
    monkeypatch.setattr(cfg, "DB_PASS", session["password"])
    monkeypatch.setattr(cfg, "DB_NAME", db_name)

    for table in INTEGRATION_TABLES:
        client.command(f"TRUNCATE TABLE IF EXISTS {db_name}.{table}")

    from app.core.db import ClickHouseDB

    with ClickHouseDB.from_config() as db:
        yield db
