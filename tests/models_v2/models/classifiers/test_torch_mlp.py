"""
Tests for TorchMLPClassifier (torch-native).

CPU-only. MLX shape-parity test is guarded by ``pytest.importorskip`` so it
cleanly skips on hosts without MLX (e.g. Linux CI).

The torch classifier module is loaded via ``importlib.util`` directly from its
source file to bypass the classifiers package ``__init__.py`` (which imports
MLX-dependent siblings). Matches the spec: torch classes are importable by
absolute path only -- no package re-export.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from chuk_lazarus.models_v2.core.enums import ActivationType


def _load_torch_mlp_module():
    src = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "chuk_lazarus"
        / "models_v2"
        / "models"
        / "classifiers"
        / "torch_mlp.py"
    )
    spec = importlib.util.spec_from_file_location(
        "chuk_lazarus_torch_mlp_standalone", src
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_torch_mlp_mod = _load_torch_mlp_module()
TorchMLPClassifier = _torch_mlp_mod.TorchMLPClassifier


class TestTorchMLPClassifier:
    """Unit tests for ``TorchMLPClassifier``."""

    def test_basic_init(self) -> None:
        clf = TorchMLPClassifier(input_size=64, hidden_size=32, num_labels=2)
        assert isinstance(clf, torch.nn.Module)
        # Structure: up(input->hidden) -> act -> down(hidden->input) -> classifier(input->num_labels)
        assert clf.up_proj.weight.shape == (32, 64)
        assert clf.down_proj.weight.shape == (64, 32)
        assert clf.classifier.weight.shape == (2, 64)

    def test_forward_shape_binary(self) -> None:
        clf = TorchMLPClassifier(input_size=32, hidden_size=16, num_labels=1)
        x = torch.randn(8, 32)
        logits = clf(x)
        assert logits.shape == (8, 1)

    def test_forward_shape_multiclass(self) -> None:
        clf = TorchMLPClassifier(input_size=128, hidden_size=64, num_labels=10)
        x = torch.randn(16, 128)
        logits = clf(x)
        assert logits.shape == (16, 10)

    def test_gradient_flow(self) -> None:
        clf = TorchMLPClassifier(input_size=32, hidden_size=16, num_labels=3)
        x = torch.randn(4, 32)
        targets = torch.tensor([0, 1, 2, 0])
        logits = clf(x)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        for name, param in clf.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"

    def test_no_bias(self) -> None:
        clf = TorchMLPClassifier(
            input_size=64, hidden_size=32, num_labels=3, bias=False
        )
        assert clf.up_proj.bias is None
        assert clf.down_proj.bias is None
        assert clf.classifier.bias is None
        x = torch.randn(4, 64)
        logits = clf(x)
        assert logits.shape == (4, 3)

    def test_activation_enum_variants(self) -> None:
        x = torch.randn(4, 64)
        for activation in [
            ActivationType.GELU,
            ActivationType.RELU,
            ActivationType.SILU,
        ]:
            clf = TorchMLPClassifier(
                input_size=64,
                hidden_size=32,
                num_labels=3,
                activation=activation,
            )
            logits = clf(x)
            assert logits.shape == (4, 3)

    def test_activation_string_variants(self) -> None:
        x = torch.randn(4, 64)
        for activation_str in ["gelu", "relu", "silu"]:
            clf = TorchMLPClassifier(
                input_size=64,
                hidden_size=32,
                num_labels=3,
                activation=activation_str,
            )
            logits = clf(x)
            assert logits.shape == (4, 3)

    def test_activation_case_insensitive_string(self) -> None:
        clf = TorchMLPClassifier(
            input_size=16, hidden_size=8, num_labels=2, activation="GELU"
        )
        logits = clf(torch.randn(2, 16))
        assert logits.shape == (2, 2)

    def test_activation_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            TorchMLPClassifier(
                input_size=16, hidden_size=8, num_labels=2, activation="tanh"
            )
        with pytest.raises(TypeError):
            TorchMLPClassifier(
                input_size=16,
                hidden_size=8,
                num_labels=2,
                activation=42,  # type: ignore[arg-type]
            )

    def test_default_dtype_is_float32(self) -> None:
        clf = TorchMLPClassifier(input_size=8, hidden_size=4, num_labels=2)
        assert clf.up_proj.weight.dtype == torch.float32
        assert clf.down_proj.weight.dtype == torch.float32
        assert clf.classifier.weight.dtype == torch.float32

    def test_shape_parity_with_mlx_twin(self) -> None:
        """Torch output shape matches the MLX twin for identical args."""
        mx = pytest.importorskip("mlx.core")
        from chuk_lazarus.models_v2.models.classifiers.mlp import (
            MLPClassifier as MLXMLPClassifier,
        )

        torch_clf = TorchMLPClassifier(
            input_size=48, hidden_size=24, num_labels=5, activation="gelu"
        )
        mlx_clf = MLXMLPClassifier(
            input_size=48,
            hidden_size=24,
            num_labels=5,
            activation=ActivationType.GELU,
        )

        x_torch = torch.randn(6, 48)
        x_mlx = mx.random.normal((6, 48))

        torch_out = torch_clf(x_torch)
        mlx_out = mlx_clf(x_mlx)

        assert tuple(torch_out.shape) == tuple(mlx_out.shape) == (6, 5)
