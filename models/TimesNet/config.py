"""Configuration defaults and path helpers for the TimesNet forecasting pipeline.

Layout under ``ARTIFACTS_ROOT``:

    {run_id}/dataset.parquet, dataset_meta.json        written by prepare
    {run_id}/experiments/{name}/model.pt, ...          one folder per train
"""

from __future__ import annotations

from pathlib import Path

from models.utils import artifacts as art

FEATURE_TABLE = "traffic_5min"

DEFAULT_FREQ = "15min"

DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 60
DEFAULT_PATIENCE = 10
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
DEFAULT_SEED = 42
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_LOG_EVERY = 1
SUPPORTED_LOSSES = ("huber", "mse", "mae")
DEFAULT_LOSS = "huber"

DEFAULT_D_MODEL = 64
DEFAULT_D_FF = 128
DEFAULT_E_LAYERS = 2
DEFAULT_TOP_K = 3
DEFAULT_NUM_KERNELS = 6
DEFAULT_DROPOUT = 0.1

DEFAULT_USE_AMP = True
DEFAULT_MATMUL_PRECISION = "high"
DEFAULT_OOM_RETRY_ENABLED = True
DEFAULT_OOM_RETRY_MIN_BATCH = 16

DEFAULT_LR_SCHEDULER_ENABLED = True
DEFAULT_LR_SCHEDULER_PATIENCE = 3
DEFAULT_LR_SCHEDULER_FACTOR = 0.5
DEFAULT_LR_SCHEDULER_MIN_LR = 1e-6

DEFAULT_EVAL_BATCH_SIZE = 512

# Read by app/api/models_local.py to browse artifacts from older runs. The
# TimesNet CLI no longer uses them.
EVAL_COMBINED_ROBUST_Z = 6.0
EVAL_COMBINED_ALERT_Z = 8.0

ARTIFACTS_ROOT = art.repo_artifacts_base() / "forecast" / "timesnet"

default_run_id = art.default_run_id


def run_dir(run_id: str) -> Path:
    return art.run_dir(ARTIFACTS_ROOT, run_id)


def dataset_path(run_id: str) -> Path:
    return art.dataset_path(ARTIFACTS_ROOT, run_id)


def dataset_meta_path(run_id: str) -> Path:
    return art.dataset_meta_path(ARTIFACTS_ROOT, run_id)


def experiments_root(run_id: str) -> Path:
    return ARTIFACTS_ROOT / run_id / "experiments"


def experiment_dir(run_id: str, name: str) -> Path:
    path = experiments_root(run_id) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path(run_id: str, name: str) -> Path:
    return experiment_dir(run_id, name) / "model.pt"


def scaler_path(run_id: str, name: str) -> Path:
    return experiment_dir(run_id, name) / "scaler.json"


def train_log_path(run_id: str, name: str) -> Path:
    return experiment_dir(run_id, name) / "train_log.json"


def metrics_path(run_id: str, name: str, split: str = "test") -> Path:
    return experiment_dir(run_id, name) / f"forecast_metrics_{split}.json"


def predictions_path(run_id: str, name: str, split: str = "test") -> Path:
    return experiment_dir(run_id, name) / f"forecast_{split}.parquet"


def plots_dir(run_id: str, name: str) -> Path:
    path = experiment_dir(run_id, name) / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_experiment_name(freq: str, window: int, pred_len: int, d_model: int) -> str:
    return f"{freq}_w{window}_h{pred_len}_d{d_model}_{art.default_run_id()}"
