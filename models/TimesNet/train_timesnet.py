"""Training, scoring and threshold calibration for TimesNet."""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.TimesNet.config import (
    DEFAULT_ANOMALY_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_D_FF,
    DEFAULT_D_MODEL,
    DEFAULT_DROPOUT,
    DEFAULT_E_LAYERS,
    DEFAULT_EPOCHS,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_EVERY,
    DEFAULT_LR_SCHEDULER_ENABLED,
    DEFAULT_LR_SCHEDULER_FACTOR,
    DEFAULT_LR_SCHEDULER_MIN_LR,
    DEFAULT_LR_SCHEDULER_PATIENCE,
    DEFAULT_MATMUL_PRECISION,
    DEFAULT_NUM_KERNELS,
    DEFAULT_OOM_RETRY_ENABLED,
    DEFAULT_OOM_RETRY_MIN_BATCH,
    DEFAULT_PATIENCE,
    DEFAULT_SCORE_SMOOTHING_SPAN,
    DEFAULT_SEED,
    DEFAULT_TEST_FRACTION,
    DEFAULT_THRESHOLD_METHOD,
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_TOP_K,
    DEFAULT_USE_AMP,
    DEFAULT_USE_RMSE_LOSS,
    DEFAULT_VAL_FRACTION,
    DEFAULT_WEIGHT_DECAY,
    SUPPORTED_THRESHOLD_METHODS,
    embed_freq_for,
)
from models.TimesNet.model import Model, TimesNetConfig, build_timesnet_config
from models.utils.lazy_windows import all_window_starts, lazy_window_loader
from models.utils.multivariate_grid import (
    extract_active_mask,
    per_location_split_masks,
    split_label_for_timestep,
    summarize_split_bounds,
    window_start_indices,
)
from models.utils.training import cuda_capability_line as _cuda_capability_line
from models.utils.training import resolve_device as _resolve_device_impl


@dataclass
class TrainSplits:
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]
    window: int


def _resolve_device() -> torch.device:
    return _resolve_device_impl(
        primary_env="TIMESNET_DEVICE",
        fallback_env="NCAR_TIMESNET_DEVICE",
        model_label="TimesNet",
    )


def _grad_scaler(*, enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, *, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except (TypeError, AttributeError):
        return torch.cuda.amp.autocast(dtype=torch.float16)


def _time_marks_from_index(index: pd.DatetimeIndex, data_freq: str) -> np.ndarray:
    """Build TimeFeatureEmbedding inputs: month, day, weekday, hour [, minute]."""
    embed_freq = embed_freq_for(data_freq)
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    month = (idx.month - 1) / 11.0
    day = (idx.day - 1) / 30.0
    weekday = idx.dayofweek / 6.0
    hour = idx.hour / 23.0
    if embed_freq == "h":
        marks = np.stack([month, day, weekday, hour], axis=1)
    else:
        minute = idx.minute / 59.0
        marks = np.stack([month, day, weekday, hour, minute], axis=1)
    return marks.astype(np.float32)


def _chronological_splits(
    n_rows: int,
    val_fraction: float,
    test_fraction: float,
) -> TrainSplits:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction}")
    if val_fraction + test_fraction >= 0.9:
        raise ValueError(f"val+test must leave >=10% train; got {val_fraction + test_fraction}")

    n_test = int(n_rows * test_fraction)
    n_val = int(n_rows * val_fraction)
    n_train = n_rows - n_val - n_test
    return TrainSplits(
        train=(0, n_train),
        val=(n_train, n_train + n_val),
        test=(n_train + n_val, n_rows),
        window=0,
    )


def _throughput(samples: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return float(samples / elapsed_s)


def _recon_loss(
    recon: torch.Tensor,
    x_tgt: torch.Tensor,
    *,
    use_rmse_loss: bool,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if loss_mask is None:
        mse = nn.functional.mse_loss(recon, x_tgt)
        return torch.sqrt(mse) if use_rmse_loss else mse
    return _masked_recon_loss(recon, x_tgt, loss_mask, use_rmse_loss=use_rmse_loss)


def _masked_recon_loss(
    recon: torch.Tensor,
    x_tgt: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    use_rmse_loss: bool,
) -> torch.Tensor:
    diff2 = (recon - x_tgt) ** 2
    weighted = diff2 * loss_mask.float()
    denom = loss_mask.float().sum().clamp_min(1.0)
    mse = weighted.sum() / denom
    return torch.sqrt(mse) if use_rmse_loss else mse


def _forward_model(
    model: Model,
    x: torch.Tensor,
    x_mark: torch.Tensor | None,
) -> torch.Tensor:
    mark = x_mark if x_mark is not None else None
    return model(x, mark, None, None)


def _iter_timesnet_batches(
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Yield (x, x_tgt, lmask, x_mark) moved to ``device``."""
    for batch in loader:
        x, x_tgt, _, lmask, x_mark = batch
        x = x.to(device, non_blocking=True)
        x_tgt = x_tgt.to(device, non_blocking=True)
        if lmask is not None:
            lmask = lmask.to(device, non_blocking=True)
        if x_mark is not None:
            x_mark = x_mark.to(device, non_blocking=True)
        yield x, x_tgt, lmask, x_mark


def train_timesnet(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
    context_columns: Sequence[str],
    scaler: dict[str, np.ndarray],
    *,
    window: int,
    data_freq: str,
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
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    use_amp: bool = DEFAULT_USE_AMP,
    use_rmse_loss: bool = DEFAULT_USE_RMSE_LOSS,
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
) -> tuple[Model, dict, TrainSplits]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(str(matmul_precision))
        except Exception:
            pass

    column_order = list(target_columns) + list(context_columns)
    missing = [c for c in column_order if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing required columns: {missing}")

    values = frame[column_order].to_numpy(dtype=np.float32)
    scaled = ((values - scaler["median"]) / scaler["scale"]).astype(np.float32)
    if not np.isfinite(scaled).all():
        bad_cols = [column_order[i] for i in range(len(column_order)) if not np.isfinite(scaled[:, i]).all()]
        print(
            f"[timesnet:train] WARN: non-finite values after scaling in columns {bad_cols}; "
            "filling with 0. Re-run prepare after fixing data_prep if this persists."
        )
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    n_features = scaled.shape[1]
    target_idx = np.arange(len(target_columns))
    time_marks = _time_marks_from_index(frame.index, data_freq)

    active = extract_active_mask(frame, target_columns)
    train_m, val_m, test_m = per_location_split_masks(
        active, val_fraction=val_fraction, test_fraction=test_fraction
    )
    splits = TrainSplits(
        train=summarize_split_bounds(train_m),
        val=summarize_split_bounds(val_m),
        test=summarize_split_bounds(test_m),
        window=window,
    )

    starts_tr = window_start_indices((active & train_m).any(axis=1), window)
    starts_va = window_start_indices((active & val_m).any(axis=1), window)
    mask_tr = (active & train_m)[:, target_idx]
    mask_va = (active & val_m)[:, target_idx]

    device = _resolve_device()
    amp_enabled = bool(use_amp and device.type == "cuda")
    if device.type != "cuda":
        print(
            f"[timesnet:train] WARN: running on CPU ({_cuda_capability_line()}). "
            "Training will be slow."
        )
    else:
        print(f"[timesnet:train] {_cuda_capability_line()}")

    cfg = build_timesnet_config(
        n_features=n_features,
        window=window,
        data_freq=data_freq,
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        top_k=top_k,
        num_kernels=num_kernels,
        dropout=dropout,
    )

    def _fit_with_batch(cur_batch_size: int) -> tuple[Model, dict]:
        model = Model(cfg).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scaler_amp = _grad_scaler(enabled=True) if amp_enabled else None

        scheduler = None
        if lr_scheduler_enabled:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim,
                mode="min",
                factor=lr_scheduler_factor,
                patience=lr_scheduler_patience,
                min_lr=lr_scheduler_min_lr,
            )

        train_loader = lazy_window_loader(
            scaled,
            starts_tr,
            target_idx,
            window,
            cur_batch_size,
            shuffle=True,
            time_marks=time_marks,
            cell_mask=mask_tr,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        val_loader = lazy_window_loader(
            scaled,
            starts_va,
            target_idx,
            window,
            cur_batch_size,
            shuffle=False,
            time_marks=time_marks,
            cell_mask=mask_va,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        best_state: dict | None = None
        best_val = float("inf")
        epochs_without_improve = 0
        history: list[dict] = []
        run_started = time.time()

        loss_mode = "rmse" if use_rmse_loss else "mse"
        print(
            f"[timesnet:train] device={device} amp={amp_enabled} loss={loss_mode} "
            f"window={window} n_features={n_features} n_targets={len(target_columns)} "
            f"train_windows={len(starts_tr)} val_windows={len(starts_va)} batch={cur_batch_size} "
            f"x_mark={'yes' if time_marks is not None else 'no'} lazy_windows=True per_location_splits=True"
        )

        for epoch in range(1, epochs + 1):
            epoch_started = time.time()
            model.train()
            tot_tr = 0.0
            n_tr = 0
            for x, x_tgt, lmask, x_mark in _iter_timesnet_batches(train_loader, device):
                optim.zero_grad(set_to_none=True)
                with _autocast(device, enabled=amp_enabled):
                    recon = _forward_model(model, x, x_mark)
                    loss = _recon_loss(
                        recon[:, :, target_idx],
                        x_tgt,
                        use_rmse_loss=use_rmse_loss,
                        loss_mask=lmask,
                    )

                if scaler_amp is not None:
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optim)
                    if grad_clip_norm and grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler_amp.step(optim)
                    scaler_amp.update()
                else:
                    loss.backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    optim.step()

                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss detected during training")

                bs = x.size(0)
                tot_tr += float(loss.item()) * bs
                n_tr += bs

            tr_loss = tot_tr / max(n_tr, 1)

            model.eval()
            tot_va = 0.0
            n_va = 0
            with torch.no_grad():
                for x, x_tgt, lmask, x_mark in _iter_timesnet_batches(val_loader, device):
                    with _autocast(device, enabled=amp_enabled):
                        recon = _forward_model(model, x, x_mark)
                        loss = _recon_loss(
                            recon[:, :, target_idx],
                            x_tgt,
                            use_rmse_loss=use_rmse_loss,
                            loss_mask=lmask,
                        )
                    bs = x.size(0)
                    tot_va += float(loss.item()) * bs
                    n_va += bs

            va_loss = tot_va / max(n_va, 1)
            current_lr = optim.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step(va_loss)

            epoch_elapsed = time.time() - epoch_started
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "val_loss": va_loss,
                    "lr": current_lr,
                    "epoch_s": epoch_elapsed,
                    "samples_per_s": _throughput(n_tr + n_va, epoch_elapsed),
                }
            )

            if va_loss < best_val - 1e-6:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epoch % max(1, log_every) == 0:
                elapsed_total = time.time() - run_started
                done = epoch
                eta = (elapsed_total / done) * (epochs - done) if done > 0 else 0.0
                print(
                    f"[timesnet:train] epoch={epoch:03d}/{epochs} "
                    f"train={tr_loss:.6f} val={va_loss:.6f} lr={current_lr:.2e} "
                    f"epoch_s={epoch_elapsed:.2f} samples/s={history[-1]['samples_per_s']:.1f} "
                    f"eta_s={eta:.1f}"
                )

            if epochs_without_improve >= patience:
                print(f"[timesnet:train] early_stop at epoch={epoch} (patience={patience})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        log = {
            "history": history,
            "best_val_loss": best_val,
            "epochs_run": history[-1]["epoch"] if history else 0,
            "hparams": {
                "window": window,
                "data_freq": data_freq,
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
                "num_workers": num_workers,
                "pin_memory": bool(pin_memory),
                "persistent_workers": bool(persistent_workers),
                "use_amp": bool(amp_enabled),
                "use_rmse_loss": bool(use_rmse_loss),
                "weight_decay": weight_decay,
                "grad_clip_norm": grad_clip_norm,
                "matmul_precision": str(matmul_precision),
                "log_every": int(log_every),
                "lr_scheduler_enabled": bool(lr_scheduler_enabled),
                "lr_scheduler_patience": int(lr_scheduler_patience),
                "lr_scheduler_factor": float(lr_scheduler_factor),
            },
            "splits": {
                "train": list(splits.train),
                "val": list(splits.val),
                "test": list(splits.test),
            },
            "n_features": n_features,
            "n_targets": len(target_columns),
            "device": str(device),
        }
        return model, log

    cur_batch = int(batch_size)
    while True:
        try:
            model, log = _fit_with_batch(cur_batch)
            return model, log, splits
        except RuntimeError as exc:
            msg = str(exc).lower()
            is_oom = "out of memory" in msg or "cuda error: out of memory" in msg
            if not (oom_retry_enabled and is_oom and cur_batch > oom_retry_min_batch):
                raise
            next_batch = max(oom_retry_min_batch, cur_batch // 2)
            if next_batch == cur_batch:
                raise
            print(
                f"[timesnet:train] OOM with batch={cur_batch}; retrying with batch={next_batch}"
            )
            cur_batch = next_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _score_array_for_slice(
    model: Model,
    scaled_slice: np.ndarray,
    target_idx: np.ndarray,
    window: int,
    time_marks_slice: np.ndarray,
    batch_size: int,
    device: torch.device,
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    use_amp: bool,
    cell_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window-mean reconstruction RMSE per target (scaled space), aligned with training loss."""
    starts = all_window_starts(scaled_slice.shape[0], window, pred_horizon=0)
    loader = lazy_window_loader(
        scaled_slice,
        starts,
        target_idx,
        window,
        batch_size,
        shuffle=False,
        time_marks=time_marks_slice,
        cell_mask=cell_mask,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model.eval()
    all_scores: list[np.ndarray] = []
    all_recon: list[np.ndarray] = []
    all_actual: list[np.ndarray] = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.no_grad():
        for x, x_tgt_b, lmask, x_m in _iter_timesnet_batches(loader, device):
            with _autocast(device, enabled=amp_enabled):
                recon = _forward_model(model, x, x_m)
            recon_tgt = recon[:, :, target_idx]
            diff2 = (recon_tgt - x_tgt_b) ** 2
            if lmask is not None:
                lmask = lmask.to(device, non_blocking=True)
                denom = lmask.float().sum(dim=1).clamp_min(1.0)
                mse = (diff2 * lmask.float()).sum(dim=1) / denom
            else:
                mse = diff2.mean(dim=1)
            per_feat = torch.sqrt(mse).detach().cpu().numpy()
            recon_last = recon_tgt[:, -1, :].detach().cpu()
            actual_last = x_tgt_b[:, -1, :].detach().cpu()
            all_scores.append(per_feat)
            all_recon.append(recon_last.numpy())
            all_actual.append(actual_last.numpy())

    return (
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_recon, axis=0),
        np.concatenate(all_actual, axis=0),
    )


def compute_scores(
    model: Model,
    frame: pd.DataFrame,
    target_columns: Sequence[str],
    context_columns: Sequence[str],
    scaler: dict[str, np.ndarray],
    splits: TrainSplits,
    *,
    data_freq: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    use_amp: bool = DEFAULT_USE_AMP,
    score_smoothing_span: int | None = DEFAULT_SCORE_SMOOTHING_SPAN,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> pd.DataFrame:
    column_order = list(target_columns) + list(context_columns)
    values = frame[column_order].to_numpy(dtype=np.float32)
    scaled = ((values - scaler["median"]) / scaler["scale"]).astype(np.float32)
    target_idx = np.arange(len(target_columns))
    time_marks = _time_marks_from_index(frame.index, data_freq)

    device = _resolve_device()
    model = model.to(device)

    window = splits.window
    if scaled.shape[0] < window:
        raise ValueError("Dataset shorter than window; cannot score.")

    active = extract_active_mask(frame, target_columns)
    train_m, val_m, test_m = per_location_split_masks(
        active,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    score, recon_scaled, actual_scaled = _score_array_for_slice(
        model=model,
        scaled_slice=scaled,
        target_idx=target_idx,
        window=window,
        time_marks_slice=time_marks,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        use_amp=use_amp,
        cell_mask=active[:, target_idx],
    )

    aligned_ds = frame.index.to_numpy()[(window - 1) : (window - 1) + score.shape[0]]
    target_names = [c[len("y__") :] if c.startswith("y__") else c for c in target_columns]
    target_scale = scaler["scale"][: len(target_columns)].astype(np.float32)
    target_median = scaler["median"][: len(target_columns)].astype(np.float32)

    rows: list[dict] = []
    for i in range(score.shape[0]):
        t_global = (window - 1) + i
        ds_val = aligned_ds[i]
        for j, uid in enumerate(target_names):
            if not active[t_global, j]:
                continue
            split = split_label_for_timestep(
                t_global,
                j,
                train_mask=train_m,
                val_mask=val_m,
                test_mask=test_m,
                active_mask=active,
            )
            rows.append(
                {
                    "ds": pd.Timestamp(ds_val),
                    "unique_id": uid,
                    "recon_err": float(score[i, j]),
                    "pred_err": 0.0,
                    "score": float(score[i, j]),
                    "recon_val": float(recon_scaled[i, j] * target_scale[j] + target_median[j]),
                    "actual_val": float(actual_scaled[i, j] * target_scale[j] + target_median[j]),
                    "split": split,
                }
            )

    out = pd.DataFrame(rows)
    out["ds"] = pd.to_datetime(out["ds"], utc=True)

    if score_smoothing_span is not None and score_smoothing_span > 1:
        out["score"] = (
            out.groupby("unique_id", sort=False)["score"]
            .transform(lambda s: s.ewm(span=score_smoothing_span, adjust=True).mean())
            .astype(np.float32)
        )

    return out


def calibrate_thresholds(
    scores: pd.DataFrame,
    method: str = DEFAULT_THRESHOLD_METHOD,
    threshold: float = DEFAULT_ANOMALY_THRESHOLD,
    quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    calibration_split: str = "train",
) -> dict:
    if method not in SUPPORTED_THRESHOLD_METHODS:
        raise ValueError(f"Unsupported threshold method: {method!r}")

    calib = scores[scores["split"] == calibration_split]
    if calib.empty:
        raise ValueError(f"No rows in calibration split={calibration_split!r}")

    per_target: dict[str, dict] = {}
    for uid, group in calib.groupby("unique_id", sort=True):
        s = group["score"].to_numpy(dtype=np.float64)
        if method == "robust_z":
            med = float(np.median(s))
            mad = float(np.median(np.abs(s - med)))
            scale = 1.4826 * mad
            if scale <= 1e-6:
                std = float(np.std(s))
                scale = std if std > 1e-6 else 1.0
            cutoff = med + threshold * scale
            per_target[str(uid)] = {
                "method": "robust_z",
                "median": med,
                "scale": scale,
                "threshold": float(threshold),
                "cutoff": float(cutoff),
            }
        else:
            cutoff = float(np.quantile(s, quantile))
            per_target[str(uid)] = {
                "method": "percentile",
                "quantile": float(quantile),
                "cutoff": cutoff,
            }

    return {
        "method": method,
        "calibration_split": calibration_split,
        "per_target": per_target,
    }


def _apply_threshold_columns(
    out: pd.DataFrame,
    thresholds: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_target = thresholds["per_target"]
    method = thresholds["method"]
    cutoff_col = np.zeros(len(out), dtype=np.float64)
    z_col = np.zeros(len(out), dtype=np.float64)
    for uid, info in per_target.items():
        mask = (out["unique_id"] == uid).to_numpy()
        if not mask.any():
            continue
        s = out.loc[mask, "score"].to_numpy(dtype=np.float64)
        cutoff_col[mask] = info["cutoff"]
        if method == "robust_z":
            z_col[mask] = (s - info["median"]) / info["scale"]
        else:
            denom = max(info["cutoff"], 1e-6)
            z_col[mask] = s / denom
    is_anom = out["score"].to_numpy(dtype=np.float64) > cutoff_col
    return cutoff_col, z_col, is_anom


def annotate_anomalies(scores: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    from models.TimesNet import config as tc

    out = scores.copy()
    cutoff_col, z_col, is_anom = _apply_threshold_columns(out, thresholds)
    out["cutoff"] = cutoff_col.astype(np.float32)
    out["z"] = z_col.astype(np.float32)
    out["is_anomaly"] = is_anom

    method = thresholds["method"]
    abs_z = np.abs(z_col)
    if method == "percentile":
        alert_z = tc.EVAL_COMBINED_ALERT_Z
        severity = np.where(
            ~is_anom,
            "",
            np.where(abs_z >= alert_z, "alert", "warning"),
        )
    else:
        severity = np.where(
            ~is_anom,
            "",
            np.where(abs_z >= (DEFAULT_ANOMALY_THRESHOLD + 1.5), "alert", "warning"),
        )
    out["severity"] = severity
    return out


def annotate_combined(
    scores: pd.DataFrame,
    thresholds_percentile: dict,
    thresholds_robust: dict,
    *,
    robust_z_confirm: float | None = None,
    alert_z: float | None = None,
) -> pd.DataFrame:
    """Percentile gate AND robust-z confirmation."""
    from models.TimesNet import config as tc

    z_confirm = float(
        robust_z_confirm if robust_z_confirm is not None else tc.EVAL_COMBINED_ROBUST_Z
    )
    z_alert = float(alert_z if alert_z is not None else tc.EVAL_COMBINED_ALERT_Z)

    out = scores.copy()
    pct_cutoff, _, is_pct = _apply_threshold_columns(out, thresholds_percentile)
    _, robust_z, _ = _apply_threshold_columns(out, thresholds_robust)

    is_combined = is_pct & (robust_z >= z_confirm)
    out["cutoff"] = pct_cutoff.astype(np.float32)
    out["z"] = robust_z.astype(np.float32)
    out["is_anomaly"] = is_combined
    out["severity"] = np.where(
        ~is_combined,
        "",
        np.where(np.abs(robust_z) >= z_alert, "alert", "warning"),
    )
    return out


def save_model(model: Model, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = model.configs
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "seq_len": cfg.seq_len,
                "enc_in": cfg.enc_in,
                "c_out": cfg.c_out,
                "d_model": cfg.d_model,
                "d_ff": cfg.d_ff,
                "e_layers": cfg.e_layers,
                "top_k": cfg.top_k,
                "num_kernels": cfg.num_kernels,
                "freq": cfg.freq,
                "dropout": cfg.dropout,
                "n_time_mark": cfg.n_time_mark,
            },
        },
        path,
    )
    return path


def load_model(path: Path, *, data_freq: str | None = None) -> Model:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    raw = payload["config"]
    freq = raw.get("freq", "h")
    if data_freq is not None:
        from models.TimesNet.config import embed_freq_for

        freq = embed_freq_for(data_freq)
    cfg = TimesNetConfig(
        seq_len=int(raw["seq_len"]),
        enc_in=int(raw["enc_in"]),
        c_out=int(raw["c_out"]),
        d_model=int(raw["d_model"]),
        d_ff=int(raw["d_ff"]),
        e_layers=int(raw["e_layers"]),
        top_k=int(raw["top_k"]),
        num_kernels=int(raw["num_kernels"]),
        freq=str(freq),
        dropout=float(raw["dropout"]),
        n_time_mark=int(raw.get("n_time_mark", 4 if freq == "h" else 5)),
    )
    model = Model(cfg)
    model.load_state_dict(payload["state_dict"])
    return model


def save_train_log(log: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    return path


def save_scores(scores: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(path, index=False)
    return path


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(Path(path))
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    return df


def save_thresholds(thresholds: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    return path


def load_thresholds(path: Path) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)
