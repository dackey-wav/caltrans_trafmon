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
