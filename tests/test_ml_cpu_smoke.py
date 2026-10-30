"""Minimal PyTorch CPU checks for CI ml_cpu job."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ml_cpu


def test_torch_cpu_available_and_matmul():
    import torch

    x = torch.randn(4, 8)
    y = torch.randn(8, 3)
    out = x @ y
    assert out.shape == (4, 3)
    assert torch.isfinite(out).all()


def test_torchvision_import():
    import torchvision  # noqa: F401
