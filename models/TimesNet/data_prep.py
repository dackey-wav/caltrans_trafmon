"""Dataset preparation for TimesNet — delegates to the shared station frame."""

from __future__ import annotations

from models.utils import multivariate_data_prep as _dp

TimesNetDatasetMeta = _dp.MultivariateDatasetMeta

build_multivariate_frame = _dp.build_multivariate_frame
fit_robust_scaler = _dp.fit_robust_scaler
apply_scaler = _dp.apply_scaler
load_dataset = _dp.load_dataset
load_meta = _dp.load_meta
load_scaler = _dp.load_scaler
save_dataset = _dp.save_dataset
save_meta = _dp.save_meta
save_scaler = _dp.save_scaler

__all__ = [
    "TimesNetDatasetMeta",
    "apply_scaler",
    "build_multivariate_frame",
    "fit_robust_scaler",
    "load_dataset",
    "load_meta",
    "load_scaler",
    "save_dataset",
    "save_meta",
    "save_scaler",
]
