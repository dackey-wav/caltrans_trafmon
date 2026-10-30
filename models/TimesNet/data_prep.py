"""Dataset preparation for TimesNet — delegates to shared multivariate pipeline."""

from __future__ import annotations

from functools import partial

from models.TimesNet import config as tc
from models.utils import multivariate_data_prep as _dp

TimesNetDatasetMeta = _dp.MultivariateDatasetMeta

_build = partial(_dp.build_multivariate_frame, feature_table=tc.FEATURE_TABLE)
_build_class = partial(_dp.build_multivariate_class_frame, feature_table=tc.FEATURE_TABLE)

build_multivariate_frame = _build
build_multivariate_class_frame = _build_class
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
    "build_multivariate_class_frame",
    "build_multivariate_frame",
    "fit_robust_scaler",
    "load_dataset",
    "load_meta",
    "load_scaler",
    "save_dataset",
    "save_meta",
    "save_scaler",
]
