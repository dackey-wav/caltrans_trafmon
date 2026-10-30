"""
Abstraction layer for interacting with ClickHouse, including connection handling,
query execution and data insertion.
"""

from collections.abc import Sequence
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver.exceptions import ClickHouseError, OperationalError, ProgrammingError

import app.core.config as cfg


class ClickHouseDB:
    def __init__(
        self,
        host: str = cfg.HOST,
        port: int = cfg.PORT,
        username: str = cfg.DB_USER,
        password: str = cfg.DB_PASS,
        database: str = cfg.DB_NAME,
    ):
        try:
            self.client = clickhouse_connect.get_client(
                host=host, port=port, username=username, password=password, database=database
            )
        except OperationalError as e:
            raise ConnectionError(f"Network/connection error to ClickHouse: {e}") from e
        except ClickHouseError as e:
            raise RuntimeError(f"ClickHouse server error: {e}") from e

    @classmethod
    def from_config(cls) -> "ClickHouseDB":
        """
        Manage the connection with ClickHouse DB
        using parameters from config file.
        """

        return cls(
            host=cfg.HOST,
            port=cfg.PORT,
            username=cfg.DB_USER,
            password=cfg.DB_PASS,
            database=cfg.DB_NAME,
        )

    def __enter__(self) -> "ClickHouseDB":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def query_raw(
        self, sql: str, parameters: dict[str, Any] | None = None
    ) -> tuple[list[str], list[Sequence[Any]]]:
        """
        Low-level access to the ClickHouse driver result object:
        Useful when you need first_row, result_set, summary, and specific driver behavior.

        :param sql: SQL query
        :param parameters: Dictionary of named parameters used in SQL placeholders (for example: {"limit": 100}).
        """

        try:
            return self.client.query(sql, parameters=parameters)
        except ProgrammingError as e:
            raise ValueError(f"SQL syntax or parameter error: {e}") from e
        except ClickHouseError as e:
            raise RuntimeError(f"ClickHouse query error: {e}") from e

    def query(
        self, sql: str, parameters: dict[str, Any] | None = None
    ) -> tuple[list[str], list[Sequence[Any]]]:
        """
        A convenient wrapper for 90% of cases:
        Returns only (column_names, result_rows) in a predictable format.

        :param sql: SQL query
        :param parameters: Dictionary of named parameters used in SQL placeholders (for example: {"limit": 100}).
        """
        result = self.query_raw(sql, parameters=parameters)
        return result.column_names, result.result_rows

    def execute(self, sql: str, parameters: dict[str, Any] | None = None) -> None:
        """
        Execute a DDL/DML query.

        :param sql: SQL query
        :param parameters: Dictionary of named parameters used in SQL placeholders (for example: {"limit": 100}).

        comment: In ClickHouse, UPDATE/DELETE are background operations
                 that rewrite partitions and can be expensive.
        """
        try:
            self.client.command(sql, parameters=parameters)
        except ProgrammingError as e:
            raise ValueError(f"SQL syntax or parameter error: {e}") from e
        except ClickHouseError as e:
            raise RuntimeError(f"ClickHouse execution error: {e}") from e

    def insert_many(
        self, table: str, columns: Sequence[str], data: Sequence[Sequence[Any]]
    ) -> None:
        """
        Bulk data insertion.

        :param table: table name
        :param columns: list of columns
        :param data: list of tuples/lists of values
        """
        if not data:
            return

        expected_len = len(columns)
        for idx, row in enumerate(data):
            if len(row) != expected_len:
                raise ValueError(
                    f"Row {idx} length {len(row)} does not match columns length {expected_len}"
                )

        try:
            self.client.insert(table, data, column_names=columns)
        except ClickHouseError as e:
            raise RuntimeError(f"Bulk insert error: {e}") from e

    def close(self) -> None:
        """
        Close the connection with ClickHouse DB.

        comment: Important: It is recommended to use a context manager.
        If you don't need one, remember to close the database connection using `close()`.
        """

        try:
            self.client.close()
        except ClickHouseError as e:
            raise RuntimeError(f"ClickHouse close error: {e}") from e
        except OperationalError as e:
            raise ConnectionError(f"Network error while closing connection: {e}") from e
