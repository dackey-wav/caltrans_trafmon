"""Dataset feature table names; prefer app.core.datasets."""

from __future__ import annotations

from app.core.datasets import (
    FEATURE_TABLE_DATASET_A,
    FEATURE_TABLE_DATASET_B,
    FEATURE_TABLES_BY_DATASET,
    feature_table_for,
)

__all__ = [
    "FEATURE_TABLE_DATASET_A",
    "FEATURE_TABLE_DATASET_B",
    "FEATURE_TABLES_BY_DATASET",
    "feature_table_for",
]
