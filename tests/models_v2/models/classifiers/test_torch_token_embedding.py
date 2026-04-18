"""
Tests for TorchTokenEmbedding (torch-native).

CPU-only. MLX shape-parity test is guarded by ``pytest.importorskip`` so it
cleanly skips on hosts without MLX (e.g. Linux CI).

The torch module is loaded via ``importlib.util`` directly from its source
file to bypass the classifiers package ``__init__.py`` (which imports
MLX-dependent siblings). Matches the spec: torch classes are importable by
absolute path only -- no package re-export.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_torch_token_module():
    src = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "chuk_lazarus"
        / "models_v2"
        / "models"
        / "classifiers"
        / "torch_token_embedding.py"
    )
    spec = importlib.util.spec_from_file_location(
        "chuk_lazarus_torch_token_embedding_standalone", src
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_torch_token_mod = _load_torch_token_module()
TorchTokenEmbedding = _torch_token_mod.TorchTokenEmbedding


class TestTorchTokenEmbedding:
    """Unit tests for ``TorchTokenEmbedding``."""

    def test_basic_init(self) -> None:
        emb = TorchTokenEmbedding(vocab_size=100, hidden_size=16)
        assert isinstance(emb, torch.nn.Module)
        assert emb.embedding.weight.shape == (100, 16)
        assert emb.vocab_size == 100
        assert emb.hidden_size == 16

    def test_forward_shape(self) -> None:
        emb = TorchTokenEmbedding(vocab_size=50, hidden_size=8)
        input_ids = torch.randint(low=0, high=50, size=(3, 7))
        out = emb(input_ids)
        assert out.shape == (3, 7, 8)

    def test_gradient_flow(self) -> None:
        """Embedding weight receives gradient through forward()."""
        emb = TorchTokenEmbedding(vocab_size=32, hidden_size=12)
        input_ids = torch.randint(low=0, high=32, size=(2, 5))
        out = emb(input_ids)
        loss = out.pow(2).mean()
        loss.backward()
        grad = emb.embedding.weight.grad
        assert grad is not None, "embedding weight has no gradient"
        assert grad.shape == emb.embedding.weight.shape
        assert torch.isfinite(grad).all()
        # Some rows were selected so at least one entry must be non-zero.
        assert (grad.abs() > 0).any()

    def test_gradient_flow_through_tied_projection(self) -> None:
        """When tied, backprop through the output projection updates the shared weight."""
        emb = TorchTokenEmbedding(vocab_size=16, hidden_size=8, tie_weights=True)
        proj = emb.as_output_projection()
        hidden = torch.randn(2, 3, 8)
        logits = proj(hidden)
        logits.pow(2).mean().backward()
        grad = emb.embedding.weight.grad
        assert grad is not None
        assert grad.shape == emb.embedding.weight.shape
        assert torch.isfinite(grad).all()
        assert (grad.abs() > 0).any()

    def test_default_dtype_is_float32(self) -> None:
        emb = TorchTokenEmbedding(vocab_size=20, hidden_size=6)
        assert emb.embedding.weight.dtype == torch.float32

    def test_tie_weights_false_independent_storage(self) -> None:
        emb = TorchTokenEmbedding(vocab_size=40, hidden_size=10, tie_weights=False)
        proj = emb.as_output_projection()
        assert isinstance(proj, torch.nn.Linear)
        assert proj.weight.shape == (40, 10)
        # Independent parameter (different identity).
        assert proj.weight is not emb.embedding.weight
        # Storage should be independent.
        assert (
            proj.weight.data_ptr() != emb.embedding.weight.data_ptr()
        )
        # And different numerical values (random init).
        assert not torch.equal(proj.weight.detach(), emb.embedding.weight.detach())

    def test_tie_weights_true_shares_weight(self) -> None:
        emb = TorchTokenEmbedding(vocab_size=40, hidden_size=10, tie_weights=True)
        proj = emb.as_output_projection()
        assert isinstance(proj, torch.nn.Linear)
        assert proj.weight.shape == (40, 10)
        # Same Parameter object and same storage when tied.
        assert proj.weight is emb.embedding.weight
        assert proj.weight.data_ptr() == emb.embedding.weight.data_ptr()
        assert torch.equal(proj.weight.detach(), emb.embedding.weight.detach())

    def test_tie_weights_true_updates_shared(self) -> None:
        """Mutating the shared weight via one path is visible from the other."""
        emb = TorchTokenEmbedding(vocab_size=12, hidden_size=4, tie_weights=True)
        proj = emb.as_output_projection()
        with torch.no_grad():
            emb.embedding.weight.add_(1.0)
        assert torch.equal(proj.weight.detach(), emb.embedding.weight.detach())

    def test_as_output_projection_is_stable(self) -> None:
        """Repeated calls return the same Linear instance (stable identity)."""
        emb = TorchTokenEmbedding(vocab_size=8, hidden_size=4, tie_weights=False)
        proj_a = emb.as_output_projection()
        proj_b = emb.as_output_projection()
        assert proj_a is proj_b

    def test_shape_parity_with_mlx_twin(self) -> None:
        """Torch forward shape matches an equivalent MLX embedding for same args."""
        mx = pytest.importorskip("mlx.core")
        mlx_nn = pytest.importorskip("mlx.nn")

        torch_emb = TorchTokenEmbedding(vocab_size=32, hidden_size=8)
        mlx_emb = mlx_nn.Embedding(num_embeddings=32, dims=8)

        input_ids_torch = torch.randint(low=0, high=32, size=(2, 5))
        input_ids_mlx = mx.array(input_ids_torch.tolist())

        torch_out = torch_emb(input_ids_torch)
        mlx_out = mlx_emb(input_ids_mlx)

        assert tuple(torch_out.shape) == tuple(mlx_out.shape) == (2, 5, 8)
