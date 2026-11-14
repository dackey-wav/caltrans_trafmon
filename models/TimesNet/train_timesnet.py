"""Forecast training loop, feature assembly and model save/load for TimesNet."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TimesNet.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_D_FF,
    DEFAULT_D_MODEL,
    DEFAULT_DROPOUT,
    DEFAULT_E_LAYERS,
    DEFAULT_EPOCHS,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_EVERY,
    DEFAULT_LOSS,
    DEFAULT_LR_SCHEDULER_ENABLED,
    DEFAULT_LR_SCHEDULER_FACTOR,
    DEFAULT_LR_SCHEDULER_MIN_LR,
    DEFAULT_LR_SCHEDULER_PATIENCE,
    DEFAULT_MATMUL_PRECISION,
    DEFAULT_NUM_KERNELS,
    DEFAULT_OOM_RETRY_ENABLED,
    DEFAULT_OOM_RETRY_MIN_BATCH,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_TEST_FRACTION,
    DEFAULT_TOP_K,
    DEFAULT_USE_AMP,
    DEFAULT_VAL_FRACTION,
    DEFAULT_WEIGHT_DECAY,
    SUPPORTED_LOSSES,
)
from models.TimesNet.model import Model, TimesNetConfig, count_parameters
from models.utils.constants import LOCAL_TZ, steps_per_day, steps_per_hour
from models.utils.lazy_windows import DeviceWindowBatches, forecast_window_starts
from models.utils.multivariate_grid import extract_active_mask, fit_robust_scaler_masked
from models.utils.training import cuda_capability_line as _cuda_capability_line
from models.utils.training import resolve_device as _resolve_device_impl

BASIC_MARK_COLUMNS = ("month", "day", "weekday", "hour", "minute")
MARK_SETS = ("full", "compact")
TARGET_MODES = ("absolute", "weekly_residual")
# "compact" keeps the weekly clock and event flags but drops features that
# pin down the calendar date, which let the model memorize training days.
COMPACT_BASIC_MARKS = ("weekday", "hour", "minute")
COMPACT_DROP_PREFIXES = ("month_", "weekofyear_", "dom_", "is_month_")


@dataclass
class SplitRows:
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]

    def rows(self, split: str) -> tuple[int, int]:
        return tuple(getattr(self, split))

    def to_dict(self) -> dict:
        return {k: list(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict) -> SplitRows:
        return cls(**{k: tuple(int(x) for x in data[k]) for k in ("train", "val", "test")})


@dataclass
class FeaturePlan:
    """Which columns feed the model and how marks are scaled."""

    target_columns: list[str]
    mark_columns: list[str]
    mark_scale: list[float]
    dropped_constant_columns: list[str]
    basic_marks: list[str] = field(default_factory=lambda: list(BASIC_MARK_COLUMNS))
    mark_set: str = "full"
    target_mode: str = "absolute"
    lag_steps: int = 0

    @property
    def channels(self) -> list[str]:
        return [c[3:] if c.startswith("y__") else c for c in self.target_columns]

    @property
    def n_mark(self) -> int:
        # Residual mode appends the lagged targets and their activity flags.
        extra = 2 * len(self.target_columns) if self.target_mode == "weekly_residual" else 0
        return len(self.basic_marks) + len(self.mark_columns) + extra

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> FeaturePlan:
        return cls(
            target_columns=list(data["target_columns"]),
            mark_columns=list(data["mark_columns"]),
            mark_scale=[float(x) for x in data["mark_scale"]],
            dropped_constant_columns=list(data.get("dropped_constant_columns", [])),
            basic_marks=list(data.get("basic_marks", BASIC_MARK_COLUMNS)),
            mark_set=str(data.get("mark_set", "full")),
            target_mode=str(data.get("target_mode", "absolute")),
            lag_steps=int(data.get("lag_steps", 0)),
        )


@dataclass
class ModelInputs:
    scaled: np.ndarray
    active: np.ndarray
    marks: np.ndarray


@dataclass
class TrainResult:
    model: Model
    log: dict
    scaler: dict[str, np.ndarray]
    plan: FeaturePlan
    splits: SplitRows


def _resolve_device() -> torch.device:
    return _resolve_device_impl(
        primary_env="TIMESNET_DEVICE",
        fallback_env="NCAR_TIMESNET_DEVICE",
        model_label="TimesNet",
    )


def amp_dtype(device: torch.device, enabled: bool) -> torch.dtype | None:
    if not enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def chronological_split_rows(
    n_rows: int,
    freq: str,
    val_fraction: float,
    test_fraction: float,
) -> SplitRows:
    """Train, then val, then test, cut on whole days when the series is long enough."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction}")
    if val_fraction + test_fraction >= 0.9:
        raise ValueError(f"val+test must leave >=10% train; got {val_fraction + test_fraction}")

    test_lo = n_rows - int(n_rows * test_fraction)
    val_lo = test_lo - int(n_rows * val_fraction)
    spd = steps_per_day(freq)
    if n_rows >= 10 * spd:
        test_lo = (test_lo // spd) * spd
        val_lo = (val_lo // spd) * spd
    return SplitRows(train=(0, val_lo), val=(val_lo, test_lo), test=(test_lo, n_rows))


def basic_time_marks(
    index: pd.DatetimeIndex,
    names: Sequence[str] = BASIC_MARK_COLUMNS,
) -> np.ndarray:
    """Local-time clock marks (subset of month, day, weekday, hour, minute) in [-0.5, 0.5]."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(LOCAL_TZ)
    available = {
        "month": lambda: (idx.month - 1) / 11.0 - 0.5,
        "day": lambda: (idx.day - 1) / 30.0 - 0.5,
        "weekday": lambda: idx.day_of_week / 6.0 - 0.5,
        "hour": lambda: idx.hour / 23.0 - 0.5,
        "minute": lambda: idx.minute / 59.0 - 0.5,
    }
    marks = np.stack([np.asarray(available[n](), dtype=np.float64) for n in names], axis=1)
    return marks.astype(np.float32)


def plan_features(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
    context_columns: Sequence[str],
    train_rows: tuple[int, int],
    mark_set: str = "full",
) -> FeaturePlan:
    """Keep calendar columns that vary inside train; scale each into about [-1, 1]."""
    if mark_set not in MARK_SETS:
        raise ValueError(f"Unsupported mark_set {mark_set!r}; expected one of {MARK_SETS}")
    lo, hi = train_rows
    keep: list[str] = []
    scale: list[float] = []
    dropped: list[str] = []
    for col in context_columns:
        if col not in frame.columns:
            continue
        if mark_set == "compact" and col.startswith(COMPACT_DROP_PREFIXES):
            continue
        values = frame[col].to_numpy(dtype=np.float64)[lo:hi]
        values = values[np.isfinite(values)]
        if values.size == 0 or float(np.nanstd(values)) == 0.0:
            dropped.append(col)
            continue
        keep.append(col)
        scale.append(max(1.0, float(np.max(np.abs(values)))))
    return FeaturePlan(
        target_columns=list(target_columns),
        mark_columns=keep,
        mark_scale=scale,
        dropped_constant_columns=dropped,
        basic_marks=list(COMPACT_BASIC_MARKS if mark_set == "compact" else BASIC_MARK_COLUMNS),
        mark_set=mark_set,
    )


def fit_target_scaler(
    frame: pd.DataFrame,
    plan: FeaturePlan,
    train_rows: tuple[int, int],
) -> dict[str, np.ndarray]:
    y = frame[plan.target_columns].to_numpy(dtype=np.float32)
    active = extract_active_mask(frame, plan.target_columns)
    fit_mask = np.zeros_like(active)
    lo, hi = train_rows
    fit_mask[lo:hi] = active[lo:hi]
    return fit_robust_scaler_masked(y, fit_mask)


def build_model_inputs(
    frame: pd.DataFrame,
    plan: FeaturePlan,
    scaler: dict[str, np.ndarray],
) -> ModelInputs:
    y = frame[plan.target_columns].to_numpy(dtype=np.float32)
    scaled = (y - scaler["median"]) / scaler["scale"]
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    active = extract_active_mask(frame, plan.target_columns)
    marks = basic_time_marks(frame.index, plan.basic_marks)
    if plan.mark_columns:
        ctx = frame[plan.mark_columns].to_numpy(dtype=np.float32)
        ctx = np.nan_to_num(ctx / np.asarray(plan.mark_scale, dtype=np.float32))
        marks = np.concatenate([marks, ctx.astype(np.float32)], axis=1)
    if plan.target_mode == "weekly_residual":
        lagged = np.zeros_like(scaled)
        lag_active = np.zeros_like(active)
        lagged[plan.lag_steps :] = scaled[: -plan.lag_steps]
        lag_active[plan.lag_steps :] = active[: -plan.lag_steps]
        marks = np.concatenate([marks, lagged, lag_active], axis=1)
    return ModelInputs(scaled=scaled, active=active, marks=marks)


def masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    pred = pred.float()
    if kind == "mse":
        err = (pred - target) ** 2
    elif kind == "mae":
        err = (pred - target).abs()
    else:
        err = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0)
    m = mask.float()
    return (err * m).sum() / m.sum().clamp_min(1.0)


def _throughput(samples: int, elapsed_s: float) -> float:
    return float(samples / elapsed_s) if elapsed_s > 0 else 0.0


def _peak_gpu_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)


def train_forecaster(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
    context_columns: Sequence[str],
    *,
    freq: str,
    window: int,
    pred_len: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    patience: int = DEFAULT_PATIENCE,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    d_model: int = DEFAULT_D_MODEL,
    d_ff: int = DEFAULT_D_FF,
    e_layers: int = DEFAULT_E_LAYERS,
    top_k: int = DEFAULT_TOP_K,
    num_kernels: int = DEFAULT_NUM_KERNELS,
    dropout: float = DEFAULT_DROPOUT,
    seed: int = DEFAULT_SEED,
    loss: str = DEFAULT_LOSS,
    mark_set: str = "full",
    target_mode: str = "absolute",
    train_stride: int = 1,
    val_stride: int | None = None,
    use_amp: bool = DEFAULT_USE_AMP,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    matmul_precision: str = DEFAULT_MATMUL_PRECISION,
    log_every: int = DEFAULT_LOG_EVERY,
    oom_retry_enabled: bool = DEFAULT_OOM_RETRY_ENABLED,
    oom_retry_min_batch: int = DEFAULT_OOM_RETRY_MIN_BATCH,
    lr_scheduler_enabled: bool = DEFAULT_LR_SCHEDULER_ENABLED,
    lr_scheduler_patience: int = DEFAULT_LR_SCHEDULER_PATIENCE,
    lr_scheduler_factor: float = DEFAULT_LR_SCHEDULER_FACTOR,
    lr_scheduler_min_lr: float = DEFAULT_LR_SCHEDULER_MIN_LR,
) -> TrainResult:
    if loss not in SUPPORTED_LOSSES:
        raise ValueError(f"Unsupported loss {loss!r}; expected one of {SUPPORTED_LOSSES}")
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unsupported target_mode {target_mode!r}; expected one of {TARGET_MODES}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(str(matmul_precision))
        except Exception:
            pass

    n_rows = len(frame)
    splits = chronological_split_rows(n_rows, freq, val_fraction, test_fraction)
    plan = plan_features(frame, target_columns, context_columns, splits.train, mark_set)
    plan.target_mode = target_mode
    plan.lag_steps = 7 * steps_per_day(freq) if target_mode == "weekly_residual" else 0
    scaler = fit_target_scaler(frame, plan, splits.train)
    inputs = build_model_inputs(frame, plan, scaler)
    channels = plan.channels
    target_scale = scaler["scale"].astype(np.float32)

    if val_stride is None:
        val_stride = steps_per_hour(freq)
    start_active = inputs.active
    if plan.lag_steps:
        lagged_active = np.zeros_like(inputs.active)
        lagged_active[plan.lag_steps :] = inputs.active[: -plan.lag_steps]
        start_active = inputs.active & lagged_active
    starts_tr = forecast_window_starts(
        n_rows, window, pred_len,
        target_lo=splits.train[0], target_hi=splits.train[1],
        active=start_active, stride=train_stride,
    )
    starts_va = forecast_window_starts(
        n_rows, window, pred_len,
        target_lo=splits.val[0], target_hi=splits.val[1],
        active=start_active, stride=val_stride,
    )
    if len(starts_tr) == 0 or len(starts_va) == 0:
        raise RuntimeError(
            f"Not enough rows for window={window} pred_len={pred_len}: "
            f"rows={n_rows} splits={splits.to_dict()} train_windows={len(starts_tr)} "
            f"val_windows={len(starts_va)}. Prepare a longer date range or shrink window/pred_len."
        )

    device = _resolve_device()
    dtype = amp_dtype(device, use_amp)
    if device.type != "cuda":
        print(f"[timesnet:train] WARN: running on CPU ({_cuda_capability_line()}). Training will be slow.")
    else:
        print(f"[timesnet:train] {_cuda_capability_line()}")
    if plan.dropped_constant_columns:
        print(f"[timesnet:train] constant in train, not used: {plan.dropped_constant_columns}")

    cfg = TimesNetConfig(
        seq_len=window,
        pred_len=pred_len,
        enc_in=len(channels),
        c_out=len(channels),
        n_mark=plan.n_mark,
        e_layers=e_layers,
        d_model=d_model,
        d_ff=d_ff,
        top_k=top_k,
        num_kernels=num_kernels,
        dropout=dropout,
        residual=(target_mode == "weekly_residual"),
    )

    def _fit_with_batch(cur_batch_size: int) -> tuple[Model, dict]:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = Model(cfg).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        grad_scaler = (
            torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
        )
        scheduler = None
        if lr_scheduler_enabled:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim,
                mode="min",
                factor=lr_scheduler_factor,
                patience=lr_scheduler_patience,
                min_lr=lr_scheduler_min_lr,
            )
        train_loader = DeviceWindowBatches(
            inputs.scaled, inputs.active, inputs.marks, starts_tr, window, pred_len,
            cur_batch_size, shuffle=True, device=device, lag_steps=plan.lag_steps,
        )
        val_loader = DeviceWindowBatches(
            inputs.scaled, inputs.active, inputs.marks, starts_va, window, pred_len,
            cur_batch_size, shuffle=False, device=device, lag_steps=plan.lag_steps,
        )
        residual = target_mode == "weekly_residual"

        best_state: dict | None = None
        best_val = float("inf")
        best_epoch = 0
        epochs_without_improve = 0
        history: list[dict] = []
        run_started = time.time()
        print(
            f"[timesnet:train] device={device} amp={dtype} loss={loss} target={target_mode} freq={freq} "
            f"window={window} pred_len={pred_len} channels={channels} n_mark={cfg.n_mark} "
            f"params={count_parameters(model):,} train_windows={len(starts_tr)} "
            f"val_windows={len(starts_va)} batch={cur_batch_size}"
        )

        for epoch in range(1, epochs + 1):
            epoch_started = time.time()
            model.train()
            tot_tr = 0.0
            n_tr = 0
            for x, y, ymask, mark, y_lag, lag_mask in train_loader:
                target, mask = (y - y_lag, ymask * lag_mask) if residual else (y, ymask)
                optim.zero_grad(set_to_none=True)
                with autocast(device, dtype):
                    pred = model(x, mark)
                batch_loss = masked_loss(pred, target, mask, loss)
                if not torch.isfinite(batch_loss):
                    raise RuntimeError("Non-finite loss detected during training")
                if grad_scaler is not None:
                    grad_scaler.scale(batch_loss).backward()
                    grad_scaler.unscale_(optim)
                    if grad_clip_norm and grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    grad_scaler.step(optim)
                    grad_scaler.update()
                else:
                    batch_loss.backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    optim.step()
                bs = x.size(0)
                tot_tr += float(batch_loss.item()) * bs
                n_tr += bs
            tr_loss = tot_tr / max(n_tr, 1)

            model.eval()
            tot_va = 0.0
            n_va = 0
            abs_sum = np.zeros(len(channels), dtype=np.float64)
            cnt = np.zeros(len(channels), dtype=np.float64)
            with torch.no_grad():
                for x, y, ymask, mark, y_lag, lag_mask in val_loader:
                    target, mask = (y - y_lag, ymask * lag_mask) if residual else (y, ymask)
                    with autocast(device, dtype):
                        pred = model(x, mark)
                    batch_loss = masked_loss(pred, target, mask, loss)
                    bs = x.size(0)
                    tot_va += float(batch_loss.item()) * bs
                    n_va += bs
                    err = ((pred.float() - target).abs() * mask).sum(dim=(0, 1)).cpu().numpy()
                    abs_sum += err
                    cnt += mask.sum(dim=(0, 1)).cpu().numpy()
            va_loss = tot_va / max(n_va, 1)
            val_mae = {
                ch: float(abs_sum[j] / max(cnt[j], 1.0) * target_scale[j])
                for j, ch in enumerate(channels)
            }

            current_lr = optim.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step(va_loss)
            epoch_elapsed = time.time() - epoch_started
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "val_loss": va_loss,
                    "val_mae": val_mae,
                    "lr": current_lr,
                    "epoch_s": epoch_elapsed,
                    "samples_per_s": _throughput(n_tr + n_va, epoch_elapsed),
                    "peak_gpu_gb": _peak_gpu_gb(device),
                }
            )
            if va_loss < best_val - 1e-6:
                best_val = va_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epoch % max(1, log_every) == 0:
                elapsed_total = time.time() - run_started
                eta = (elapsed_total / epoch) * (epochs - epoch)
                mae_txt = " ".join(f"{k}={v:.3f}" for k, v in val_mae.items())
                peak = history[-1]["peak_gpu_gb"]
                print(
                    f"[timesnet:train] epoch={epoch:03d}/{epochs} train={tr_loss:.5f} "
                    f"val={va_loss:.5f} val_mae[{mae_txt}] lr={current_lr:.2e} "
                    f"epoch_s={epoch_elapsed:.1f}"
                    + (f" peak_gpu_gb={peak:.2f}" if peak is not None else "")
                    + f" eta_s={eta:.0f}"
                )
            if epochs_without_improve >= patience:
                print(f"[timesnet:train] early_stop at epoch={epoch} (patience={patience})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        log = {
            "task": "forecast",
            "history": history,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "epochs_run": history[-1]["epoch"] if history else 0,
            "hparams": {
                "freq": freq,
                "window": window,
                "pred_len": pred_len,
                "epochs": epochs,
                "batch_size": cur_batch_size,
                "learning_rate": learning_rate,
                "patience": patience,
                "val_fraction": val_fraction,
                "test_fraction": test_fraction,
                "d_model": d_model,
                "d_ff": d_ff,
                "e_layers": e_layers,
                "top_k": top_k,
                "num_kernels": num_kernels,
                "dropout": dropout,
                "seed": seed,
                "loss": loss,
                "mark_set": mark_set,
                "target_mode": target_mode,
                "train_stride": train_stride,
                "val_stride": val_stride,
                "use_amp": str(dtype) if dtype is not None else False,
                "weight_decay": weight_decay,
                "grad_clip_norm": grad_clip_norm,
                "lr_scheduler_enabled": bool(lr_scheduler_enabled),
            },
            "splits": splits.to_dict(),
            "split_ds": {
                name: [str(frame.index[lo]), str(frame.index[hi - 1])]
                for name, (lo, hi) in asdict(splits).items()
                if hi > lo
            },
            "feature_plan": plan.to_dict(),
            "n_parameters": count_parameters(model),
            "peak_gpu_gb": _peak_gpu_gb(device),
            "device": str(device),
            "train_windows": int(len(starts_tr)),
            "val_windows": int(len(starts_va)),
        }
        return model, log

    cur_batch = int(batch_size)
    while True:
        try:
            model, log = _fit_with_batch(cur_batch)
            return TrainResult(model=model, log=log, scaler=scaler, plan=plan, splits=splits)
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            msg = str(exc).lower()
            is_oom = "out of memory" in msg
            if not (oom_retry_enabled and is_oom and cur_batch > oom_retry_min_batch):
                raise
            next_batch = max(oom_retry_min_batch, cur_batch // 2)
            if next_batch == cur_batch:
                raise
            print(f"[timesnet:train] OOM with batch={cur_batch}; retrying with batch={next_batch}")
            cur_batch = next_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def save_model(model: Model, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.configs.to_dict()}, path)
    return path


def load_model(path: Path) -> Model:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = TimesNetConfig(**payload["config"])
    model = Model(cfg)
    model.load_state_dict(payload["state_dict"])
    return model


def save_train_log(log: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    return path


def load_train_log(path: Path) -> dict:
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)
