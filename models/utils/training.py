"""Shared PyTorch training utilities."""

from __future__ import annotations

import os

import torch


def resolve_device(
    *,
    primary_env: str,
    fallback_env: str | None = None,
    model_label: str = "model",
) -> torch.device:
    """Pick compute device from env vars (e.g. TIMESNET_DEVICE, NCAR_TIMESNET_DEVICE)."""
    raw = (os.environ.get(primary_env) or (os.environ.get(fallback_env) if fallback_env else "") or "").strip().lower()
    if raw in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{primary_env}=cuda requested but torch.cuda.is_available() is False. "
                f"Use a CUDA-enabled PyTorch wheel or set {primary_env}=cpu."
            )
        return torch.device("cuda")
    if raw == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cuda_capability_line() -> str:
    if not torch.cuda.is_available():
        return f"torch={torch.__version__} cuda_available=False"
    try:
        name = torch.cuda.get_device_name(0)
        return f"torch={torch.__version__} cuda={torch.version.cuda} device0={name!r}"
    except Exception:
        return f"torch={torch.__version__} cuda={torch.version.cuda}"


def runtime_loader_defaults(
    device: str,
    num_workers: int | None,
    pin_memory: bool | None,
    *,
    default_gpu_num_workers: int,
    default_cpu_num_workers: int,
    default_gpu_pin_memory: bool,
    default_cpu_pin_memory: bool,
) -> tuple[int, bool]:
    is_gpu = device == "cuda"
    if num_workers is None:
        num_workers = default_gpu_num_workers if is_gpu else default_cpu_num_workers
    if pin_memory is None:
        pin_memory = default_gpu_pin_memory if is_gpu else default_cpu_pin_memory
    return int(num_workers), bool(pin_memory)
