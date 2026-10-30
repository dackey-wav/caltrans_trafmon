"""
Loads and validates application configuration from environment variables.
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

load_dotenv()


class Settings(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    # ClickHouse. Blank database, host and port fall back to the local compose setup.
    DB_HOST: str = Field(default="localhost")
    CLICKHOUSE_DB: str = Field(default="traffic")
    PORT: int = Field(default=8123, gt=0)
    CLICKHOUSE_USER: str = Field(default="")
    CLICKHOUSE_PASSWORD: str = Field(default="")


def _or_default(value: str | None, fallback: str) -> str:
    text = (value or "").strip()
    return text or fallback


def _load_settings() -> Settings:
    raw = {
        "DB_HOST": _or_default(os.getenv("DB_HOST"), "localhost"),
        "CLICKHOUSE_DB": _or_default(os.getenv("CLICKHOUSE_DB"), "traffic"),
        "PORT": _or_default(os.getenv("PORT"), "8123"),
        "CLICKHOUSE_USER": os.getenv("CLICKHOUSE_USER", ""),
        "CLICKHOUSE_PASSWORD": os.getenv("CLICKHOUSE_PASSWORD", ""),
    }
    try:
        return Settings(**raw)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid environment configuration: {exc}") from exc


settings = _load_settings()

HOST = settings.DB_HOST
DB_NAME = settings.CLICKHOUSE_DB
PORT = settings.PORT
DB_USER = settings.CLICKHOUSE_USER
DB_PASS = settings.CLICKHOUSE_PASSWORD
