"""Neutral dataset registry for feature tables and dashboard metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

FEATURE_TABLE_DATASET_A = "vehicletrace_feature_dataset_a"
FEATURE_TABLE_DATASET_B = "vehicletrace_feature_dataset_b"


@dataclass(frozen=True)
class Dataset:
    id: str
    label: str
    feature_table: str
    default_tz: str = "UTC"


_DATASETS: dict[str, Dataset] = {
    "dataset-a": Dataset(
        id="dataset-a",
        label="Dataset A",
        feature_table=FEATURE_TABLE_DATASET_A,
        default_tz="UTC",
    ),
    "dataset-b": Dataset(
        id="dataset-b",
        label="Dataset B",
        feature_table=FEATURE_TABLE_DATASET_B,
        default_tz="UTC",
    ),
}

DATASETS: Mapping[str, Dataset] = MappingProxyType(_DATASETS)

FEATURE_TABLES_BY_DATASET: Mapping[str, str] = MappingProxyType(
    {dataset_id: ds.feature_table for dataset_id, ds in _DATASETS.items()}
)


def get_dataset(dataset_id: str) -> Dataset:
    """Resolve a registered dataset id."""

    try:
        return _DATASETS[dataset_id]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset: {dataset_id!r}") from exc


def list_datasets() -> list[Dataset]:
    """Return registered datasets in stable id order."""

    return [_DATASETS[dataset_id] for dataset_id in sorted(_DATASETS)]


def feature_table_for(dataset_id: str) -> str:
    """ClickHouse feature table name for a dataset id."""

    return get_dataset(dataset_id).feature_table
