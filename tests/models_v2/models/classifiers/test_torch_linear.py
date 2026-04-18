"""
Tests for TorchLinearClassifier (torch-native).

CPU-only. MLX shape-parity test is guarded by ``pytest.importorskip`` so it
cleanly skips on hosts without MLX (e.g. Linux CI).

The torch classifier module is loaded via ``importlib.util`` directly from its
source file to bypass the classifiers package ``__init__.py``, which imports
MLX-dependent siblings and therefore cannot be imported on hosts without MLX.
This matches the spec guidance that these torch classes are importable by
absolute path only -- no package-level re-export.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_torch_linear_module():
    src = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "chuk_lazarus"
        / "models_v2"
        / "models"
        / "classifiers"
        / "torch_linear.py"
    )
    spec = importlib.util.spec_from_file_location(
        "chuk_lazarus_torch_linear_standalone", src
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_torch_linear_mod = _load_torch_linear_module()
TorchLinearClassifier = _torch_linear_mod.TorchLinearClassifier


class TestTorchLinearClassifier:
    """Unit tests for ``TorchLinearClassifier``."""

    def test_basic_init(self) -> None:
        clf = TorchLinearClassifier(input_size=64, num_labels=2)
        assert isinstance(clf, torch.nn.Module)
        assert clf.fc.weight.shape == (2, 64)
        assert clf.fc.bias is not None
        assert clf.fc.bias.shape == (2,)

    def test_forward_shape_binary(self) -> None:
        clf = TorchLinearClassifier(input_size=32, num_labels=1)
        x = torch.randn(8, 32)
        logits = clf(x)
        assert logits.shape == (8, 1)

    def test_forward_shape_multiclass(self) -> None:
        clf = TorchLinearClassifier(input_size=128, num_labels=10)
        x = torch.randn(16, 128)
        logits = clf(x)
        assert logits.shape == (16, 10)

    def test_gradient_flow(self) -> None:
        clf = TorchLinearClassifier(input_size=16, num_labels=3)
        x = torch.randn(4, 16)
        targets = torch.tensor([0, 1, 2, 0])
        logits = clf(x)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        for name, param in clf.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"

    def test_no_bias(self) -> None:
        clf = TorchLinearClassifier(input_size=64, num_labels=3, bias=False)
        assert clf.fc.bias is None
        x = torch.randn(4, 64)
        logits = clf(x)
        assert logits.shape == (4, 3)

    def test_default_dtype_is_float32(self) -> None:
        clf = TorchLinearClassifier(input_size=8, num_labels=2)
        assert clf.fc.weight.dtype == torch.float32

    def test_shape_parity_with_mlx_twin(self) -> None:
        """Torch output shape matches the MLX twin for identical args."""
        mx = pytest.importorskip("mlx.core", exc_type=ImportError)
        from chuk_lazarus.models_v2.models.classifiers.linear import (
            LinearClassifier as MLXLinearClassifier,
        )

        torch_clf = TorchLinearClassifier(input_size=48, num_labels=5)
        mlx_clf = MLXLinearClassifier(input_size=48, num_labels=5)

        x_torch = torch.randn(6, 48)
        x_mlx = mx.random.normal((6, 48))

        torch_out = torch_clf(x_torch)
        mlx_out = mlx_clf(x_mlx)

        assert tuple(torch_out.shape) == tuple(mlx_out.shape) == (6, 5)
