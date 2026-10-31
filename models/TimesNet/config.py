"""Configuration defaults and path helpers for the TimesNet anomaly pipeline."""

from __future__ import annotations

from pathlib import Path

from models.utils import artifacts as art
from models.utils.runner_common import threshold_preset as _threshold_preset

FEATURE_TABLE = "traffic_5min"

DEFAULT_FREQ = "15min"

DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 15
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
DEFAULT_SEED = 42
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_LOG_EVERY = 1
DEFAULT_USE_RMSE_LOSS = True

DEFAULT_D_MODEL = 64
DEFAULT_D_FF = 128
DEFAULT_E_LAYERS = 2
DEFAULT_TOP_K = 3
DEFAULT_NUM_KERNELS = 6
DEFAULT_DROPOUT = 0.1

DEFAULT_GPU_NUM_WORKERS = 4
DEFAULT_CPU_NUM_WORKERS = 0
DEFAULT_GPU_PIN_MEMORY = True
DEFAULT_CPU_PIN_MEMORY = False
DEFAULT_PERSISTENT_WORKERS = True
DEFAULT_USE_AMP = False
DEFAULT_MATMUL_PRECISION = "high"
DEFAULT_OOM_RETRY_ENABLED = True
DEFAULT_OOM_RETRY_MIN_BATCH = 32

DEFAULT_SCORE_SMOOTHING_SPAN: int | None = None

DEFAULT_LR_SCHEDULER_ENABLED = True
DEFAULT_LR_SCHEDULER_PATIENCE = 5
DEFAULT_LR_SCHEDULER_FACTOR = 0.5
DEFAULT_LR_SCHEDULER_MIN_LR = 1e-6

DEFAULT_THRESHOLD_METHOD = "robust_z"
# Raised from 3.5 — on dense 15-min runs z=3.5 flagged a large share of points.
DEFAULT_ANOMALY_THRESHOLD = 6.0
DEFAULT_THRESHOLD_QUANTILE = 0.999
THRESHOLD_PRESETS: dict[str, dict[str, float | str]] = {
    "balanced": {"method": "robust_z", "threshold": 6.0},
    "strict": {"method": "percentile", "quantile": 0.9995},
    "sensitive": {"method": "robust_z", "threshold": 5.0},
}

# Default eval policy: percentile gate + robust-z confirmation (see annotate_combined).
EVAL_PERCENTILE_QUANTILE = 0.999
EVAL_COMBINED_ROBUST_Z = 6.0
EVAL_COMBINED_ALERT_Z = 8.0

# Domain gate (60-minute aggregates; see domain_filter.py).
DOMAIN_AGG_FREQ = "1h"
MIN_TRAFFIC_INTENSITY_PCT = 0.20
MIN_HOURLY_VOLUME = 5.0

PROFILE_15MIN_GPU = {
    "freq": "15min",
    "window": 192,
    "batch_size": 256,
    "d_model": 64,
    "patience": 5,
    "epochs": 25,
    "num_workers": 4,
    "pin_memory": True,
    "use_amp": True,
}

ARTIFACTS_ROOT = art.repo_artifacts_base() / "anomaly" / "timesnet"

default_run_id = art.default_run_id
run_dir = lambda run_id: art.run_dir(ARTIFACTS_ROOT, run_id)
plots_dir = lambda run_id: art.plots_dir(ARTIFACTS_ROOT, run_id)
anomaly_plots_dir = lambda run_id: art.anomaly_plots_dir(ARTIFACTS_ROOT, run_id)
reconstruction_plots_dir = lambda run_id: art.reconstruction_plots_dir(ARTIFACTS_ROOT, run_id)
dataset_path = lambda run_id: art.dataset_path(ARTIFACTS_ROOT, run_id)
dataset_meta_path = lambda run_id: art.dataset_meta_path(ARTIFACTS_ROOT, run_id)
scaler_path = lambda run_id: art.scaler_path(ARTIFACTS_ROOT, run_id)
model_path = lambda run_id: art.model_path(ARTIFACTS_ROOT, run_id)
train_log_path = lambda run_id: art.train_log_path(ARTIFACTS_ROOT, run_id)
scores_path = lambda run_id: art.scores_path(ARTIFACTS_ROOT, run_id)
thresholds_path = lambda run_id: art.thresholds_path(ARTIFACTS_ROOT, run_id)
anomalies_path = lambda run_id: art.anomalies_path(ARTIFACTS_ROOT, run_id)
metrics_path = lambda run_id: art.metrics_path(ARTIFACTS_ROOT, run_id)
comparison_path = lambda run_id: art.comparison_path(ARTIFACTS_ROOT, run_id)


def anomalies_stat_path(run_id: str) -> Path:
    return art.run_dir(ARTIFACTS_ROOT, run_id) / "anomalies_stat.json"


def thresholds_percentile_path(run_id: str) -> Path:
    return art.run_dir(ARTIFACTS_ROOT, run_id) / "thresholds_percentile.json"


def thresholds_robust_path(run_id: str) -> Path:
    return art.run_dir(ARTIFACTS_ROOT, run_id) / "thresholds_robust.json"


def threshold_preset(name: str) -> dict[str, float | str]:
    return _threshold_preset(name, THRESHOLD_PRESETS)


def embed_freq_for(freq: str) -> str:
    """Map dataset freq to TimesNet TimeFeatureEmbedding freq code."""
    if freq == "1h":
        return "h"
    if freq in ("15min", "5min"):
        return "t"
    raise ValueError(f"Unsupported freq: {freq!r}")


