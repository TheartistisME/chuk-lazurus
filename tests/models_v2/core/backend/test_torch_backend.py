"""
Tests for PyTorch backend implementation.

These tests only run if PyTorch is available.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from chuk_lazarus.models_v2.core.backend import BackendType

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.fixture
def torch_available():
    """Check if torch is available."""
    return _TORCH_AVAILABLE


class TestTorchBackendCreation:
    """Tests for TorchBackend creation."""

    def test_torch_backend_exists(self):
        """TorchBackend class must be importable from the backend package."""
        from chuk_lazarus.models_v2.core.backend import TorchBackend

        assert TorchBackend is not None

    def test_torch_backend_creation(self, torch_available):
        """Default TorchBackend constructor returns a backend of type TORCH."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        assert backend.name == BackendType.TORCH

    def test_torch_backend_cpu_device(self, torch_available):
        """``device="cpu"`` must always resolve to ``"cpu"`` regardless of CUDA state."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        assert backend.device == "cpu"

    def test_torch_backend_cpu_dtype_is_float32(self, torch_available):
        """CPU device must use ``torch.float32`` per the dtype policy."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        import torch

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        assert backend.dtype == torch.float32

    def test_torch_backend_mps_falls_back_to_cpu_when_unavailable(
        self, torch_available, monkeypatch
    ):
        """``device="mps"`` must fall back to ``"cpu"`` when MPS is not available."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        import torch

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        # Force mps.is_available -> False to exercise the fallback branch.
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        backend = TorchBackend(device="mps")
        assert backend.device == "cpu"
        assert backend.dtype == torch.float32

    def test_torch_backend_missing_torch_raises(self, monkeypatch):
        """If torch cannot be imported the constructor raises ImportError with guidance."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("forced missing torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        from chuk_lazarus.models_v2.core.backend.torch_backend import TorchBackend

        with pytest.raises(ImportError, match="PyTorch is required"):
            TorchBackend()


class TestTorchBackendDeviceFallback:
    """Tests for device fallback / resolution branches."""

    def test_cuda_falls_back_to_cpu_when_cuda_unavailable(
        self, torch_available, monkeypatch
    ):
        """``device="cuda"`` must silently fall back to ``"cpu"`` when CUDA is absent."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        import torch

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        # Patch cuda.is_available so the constructor sees no CUDA.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        backend = TorchBackend(device="cuda")
        assert backend.device == "cpu"
        # Dtype policy: CPU => float32.
        assert backend.dtype == torch.float32


class TestTorchBackendTensorCreation:
    """Tests for TorchBackend tensor creation."""

    def test_zeros(self, torch_available):
        """``zeros`` returns a tensor of the requested shape filled with zero."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        tensor = backend.zeros((2, 3))
        assert tensor.shape == (2, 3)
        assert np.allclose(backend.to_numpy(tensor), np.zeros((2, 3)))

    def test_ones(self, torch_available):
        """``ones`` returns a tensor of the requested shape filled with one."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        tensor = backend.ones((3, 4))
        assert tensor.shape == (3, 4)
        assert np.allclose(backend.to_numpy(tensor), np.ones((3, 4)))

    def test_randn(self, torch_available):
        """``randn`` returns a tensor of the requested shape with finite values."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        tensor = backend.randn((2, 5))
        assert tensor.shape == (2, 5)
        assert np.all(np.isfinite(backend.to_numpy(tensor)))

    def test_arange(self, torch_available):
        """``arange`` matches ``numpy.arange`` for the same start/end/step."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        tensor = backend.arange(0, 10, 2)
        result = backend.to_numpy(tensor)
        assert np.allclose(result, np.arange(0, 10, 2))

    def test_from_numpy_and_to_numpy_roundtrip(self, torch_available):
        """``from_numpy``/``to_numpy`` must round-trip the same values."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor = backend.from_numpy(arr)
        assert tensor.shape == (2, 2)
        assert np.allclose(backend.to_numpy(tensor), arr)


class TestTorchBackendOperations:
    """Tests for TorchBackend tensor operations."""

    def test_matmul(self, torch_available):
        """``matmul`` computes standard matrix multiplication."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        a = backend.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = backend.from_numpy(np.array([[5, 6], [7, 8]], dtype=np.float32))
        c = backend.matmul(a, b)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        assert np.allclose(backend.to_numpy(c), expected)

    def test_softmax_sums_to_one(self, torch_available):
        """``softmax`` along the last axis must sum to 1 for every row."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        x = backend.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        result = backend.softmax(x, axis=-1)
        result_np = backend.to_numpy(result)
        assert np.allclose(result_np.sum(), 1.0, atol=1e-5)


class TestTorchBackendActivations:
    """Tests for TorchBackend activations."""

    def test_relu_clips_negatives(self, torch_available):
        """``relu`` must zero negatives and preserve positives exactly."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        x = backend.from_numpy(np.array([-1.0, 0.0, 2.0], dtype=np.float32))
        result = backend.to_numpy(backend.relu(x))
        assert np.allclose(result, np.array([0.0, 0.0, 2.0], dtype=np.float32))

    def test_activations_preserve_shape(self, torch_available):
        """silu/gelu/tanh/sigmoid return tensors matching the input shape."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        x = backend.from_numpy(np.array([0.0, 1.0, 2.0], dtype=np.float32))

        assert backend.silu(x).shape == (3,)
        assert backend.gelu(x).shape == (3,)
        assert backend.tanh(x).shape == (3,)
        assert backend.sigmoid(x).shape == (3,)

    def test_sigmoid_values(self, torch_available):
        """``sigmoid(0)`` must equal 0.5 within float tolerance."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        x = backend.from_numpy(np.array([0.0], dtype=np.float32))
        result = backend.to_numpy(backend.sigmoid(x))
        assert np.allclose(result, np.array([0.5], dtype=np.float32), atol=1e-6)


class TestTorchBackendNormalization:
    """Tests for TorchBackend normalization."""

    def test_layer_norm_shape(self, torch_available):
        """``layer_norm`` preserves the input shape."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        x = backend.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        weight = backend.ones((4,))
        bias = backend.zeros((4,))

        result = backend.layer_norm(x, weight, bias, eps=1e-5)
        assert result.shape == (1, 4)

    def test_layer_norm_without_bias(self, torch_available):
        """``layer_norm`` must accept ``bias=None`` (passes through to torch)."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        x = backend.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        weight = backend.ones((4,))

        result = backend.layer_norm(x, weight, None, eps=1e-5)
        assert result.shape == (1, 4)

    def test_rms_norm(self, torch_available):
        """``rms_norm`` with unit weights normalises by the RMS along the last axis."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        x = backend.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        weight = backend.ones((4,))

        result = backend.rms_norm(x, weight, eps=1e-5)
        result_np = backend.to_numpy(result)

        expected_rms = float(np.sqrt(np.mean(np.array([1.0, 2.0, 3.0, 4.0]) ** 2) + 1e-5))
        expected = np.array([[1.0, 2.0, 3.0, 4.0]]) / expected_rms
        assert np.allclose(result_np, expected, atol=1e-5)


class TestTorchBackendShapeOperations:
    """Tests for TorchBackend shape operations."""

    def test_reshape(self, torch_available):
        """``reshape`` preserves values but changes shape to the requested one."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        x = backend.from_numpy(np.arange(24, dtype=np.float32).reshape(2, 3, 4))

        reshaped = backend.reshape(x, (6, 4))
        assert reshaped.shape == (6, 4)
        assert np.allclose(
            backend.to_numpy(reshaped),
            np.arange(24, dtype=np.float32).reshape(6, 4),
        )

    def test_transpose_permutes_axes(self, torch_available):
        """``transpose`` permutes axes according to the provided tuple."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        x = backend.from_numpy(np.arange(24, dtype=np.float32).reshape(2, 3, 4))

        transposed = backend.transpose(x, (0, 2, 1))
        assert transposed.shape == (2, 4, 3)

    def test_concat_split_roundtrip(self, torch_available):
        """Concatenating and then splitting by the same count recovers the parts."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        a = backend.ones((2, 3))
        b = backend.zeros((2, 3))

        concatenated = backend.concatenate([a, b], axis=0)
        assert concatenated.shape == (4, 3)

        parts = backend.split(concatenated, 2, axis=0)
        assert len(parts) == 2
        assert parts[0].shape == (2, 3)
        assert np.allclose(backend.to_numpy(parts[0]), np.ones((2, 3)))
        assert np.allclose(backend.to_numpy(parts[1]), np.zeros((2, 3)))


class TestTorchBackendAttention:
    """Tests for TorchBackend attention."""

    def test_scaled_dot_product_attention_shape(self, torch_available):
        """SDPA returns a tensor with the same shape as ``value``."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        batch, heads, seq, dim = 1, 2, 4, 8
        q = backend.randn((batch, heads, seq, dim))
        k = backend.randn((batch, heads, seq, dim))
        v = backend.randn((batch, heads, seq, dim))

        result = backend.scaled_dot_product_attention(q, k, v)
        assert result.shape == (batch, heads, seq, dim)

    def test_causal_mask_shape_and_upper_triangle(self, torch_available):
        """``create_causal_mask`` returns a square mask with -inf above the diagonal."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        mask = backend.create_causal_mask(5)
        assert mask.shape == (5, 5)
        mask_np = backend.to_numpy(mask)
        # Upper triangle (strictly above diagonal) must be -inf.
        for i in range(5):
            for j in range(i + 1, 5):
                assert mask_np[i, j] == float("-inf")
        # Diagonal + lower triangle must be finite (0).
        for i in range(5):
            for j in range(0, i + 1):
                assert mask_np[i, j] == 0.0


class TestTorchBackendGradients:
    """Tests for TorchBackend gradient operations."""

    def test_stop_gradient_detaches(self, torch_available):
        """``stop_gradient`` must return a tensor with ``requires_grad == False``."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        import torch

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        x = torch.randn(2, 3, requires_grad=True)
        result = backend.stop_gradient(x)
        assert result.shape == (2, 3)
        assert result.requires_grad is False

    def test_eval_noop(self, torch_available):
        """``eval`` is a no-op on the eager torch backend and must not raise."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        x = backend.randn((2, 3))
        backend.eval(x)  # Should not raise
        backend.eval()  # No-args form is also valid


class TestTorchBackendAuxiliary:
    """Tests for auxiliary TorchBackend helpers (array/save/load)."""

    def test_array_constructs_tensor(self, torch_available):
        """``array`` builds a tensor from a Python list with the expected shape."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        tensor = backend.array([[1.0, 2.0], [3.0, 4.0]])
        assert tensor.shape == (2, 2)
        assert np.allclose(
            backend.to_numpy(tensor),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

    def test_save_load_roundtrip(self, torch_available, tmp_path):
        """``save``/``load`` must round-trip a tensor dict to disk and back."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cpu")
        payload = {"w": backend.from_numpy(np.array([1.0, 2.0], dtype=np.float32))}
        path = tmp_path / "state.pt"
        backend.save(payload, str(path))

        loaded = backend.load(str(path))
        assert "w" in loaded
        assert np.allclose(backend.to_numpy(loaded["w"]), np.array([1.0, 2.0]))


class TestTorchBackendCuda:
    """CUDA-guarded tests for dtype policy and device resolution."""

    @pytest.mark.skipif(
        not _TORCH_AVAILABLE, reason="PyTorch not installed"
    )
    def test_cuda_dtype_bf16_on_sm80_plus(self):
        """On CUDA SM>=8 with check_sm=True the backend must use bfloat16."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        major, _ = torch.cuda.get_device_capability(0)
        if major < 8:
            pytest.skip("Requires SM>=8 GPU")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cuda", check_sm=True)
        assert backend.device.startswith("cuda")
        assert backend.dtype == torch.bfloat16

    @pytest.mark.skipif(
        not _TORCH_AVAILABLE, reason="PyTorch not installed"
    )
    def test_cuda_dtype_fp16_below_sm80(self):
        """On CUDA SM<8 the backend must fall back to float16."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            pytest.skip("Requires SM<8 GPU")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cuda", check_sm=True)
        assert backend.device.startswith("cuda")
        assert backend.dtype == torch.float16

    @pytest.mark.skipif(
        not _TORCH_AVAILABLE, reason="PyTorch not installed"
    )
    def test_cuda_dtype_fp16_when_check_sm_disabled(self):
        """With check_sm=False on CUDA, dtype must be float16 regardless of capability."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend(device="cuda", check_sm=False)
        assert backend.device.startswith("cuda")
        assert backend.dtype == torch.float16
