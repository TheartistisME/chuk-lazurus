"""
PyTorch backend implementation.

Supports CUDA, CPU, and MPS (Apple Silicon via PyTorch).
"""

from __future__ import annotations

from typing import Any

from .base import Backend
from .types import BackendType


class TorchBackend(Backend):
    """PyTorch backend implementation."""

    def __init__(self, device: str = "cuda", check_sm: bool = True):
        try:
            import torch

            self._torch = torch
            self._check_sm = check_sm
            self._device = self._resolve_device(device)
            self._dtype = self._resolve_dtype(check_sm)
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for TorchBackend. Install with: pip install torch"
            ) from e

    def _resolve_device(self, device: str) -> str:
        normalized = (device or "cuda").lower()
        if normalized.startswith("cuda"):
            return normalized if self._torch.cuda.is_available() else "cpu"
        if normalized == "mps":
            mps_backend = getattr(self._torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return "mps"
            return "cpu"
        return normalized

    def _resolve_dtype(self, check_sm: bool) -> Any:
        if self._device.startswith("cuda"):
            if check_sm:
                device_id = self._device if ":" in self._device else "cuda:0"
                major, _ = self._torch.cuda.get_device_capability(device_id)
                if major >= 8:
                    return self._torch.bfloat16
            return self._torch.float16
        return self._torch.float32

    @property
    def name(self) -> BackendType:
        return BackendType.TORCH

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> Any:
        return self._dtype

    def array(self, data: Any, dtype: Any = None) -> Any:
        return self._torch.as_tensor(data, dtype=dtype, device=self._device)

    def save(self, data: Any, path: str) -> None:
        self._torch.save(data, path)

    def load(self, path: str) -> Any:
        try:
            return self._torch.load(path, map_location=self._device, weights_only=False)
        except TypeError:
            return self._torch.load(path, map_location=self._device)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return self._torch.zeros(shape, dtype=dtype, device=self._device)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return self._torch.ones(shape, dtype=dtype, device=self._device)

    def randn(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return self._torch.randn(shape, dtype=dtype, device=self._device)

    def arange(self, start: int, end: int, step: int = 1, dtype: Any = None) -> Any:
        return self._torch.arange(start, end, step, dtype=dtype, device=self._device)

    def from_numpy(self, array: Any) -> Any:
        return self._torch.from_numpy(array).to(self._device)

    def to_numpy(self, tensor: Any) -> Any:
        return tensor.cpu().numpy()

    def matmul(self, a: Any, b: Any) -> Any:
        return self._torch.matmul(a, b)

    def softmax(self, x: Any, axis: int = -1) -> Any:
        return self._torch.softmax(x, dim=axis)

    def relu(self, x: Any) -> Any:
        return self._torch.relu(x)

    def silu(self, x: Any) -> Any:
        return self._torch.nn.functional.silu(x)

    def gelu(self, x: Any) -> Any:
        return self._torch.nn.functional.gelu(x)

    def tanh(self, x: Any) -> Any:
        return self._torch.tanh(x)

    def sigmoid(self, x: Any) -> Any:
        return self._torch.sigmoid(x)

    def layer_norm(self, x: Any, weight: Any, bias: Any | None, eps: float) -> Any:
        return self._torch.nn.functional.layer_norm(
            x, weight.shape, weight=weight, bias=bias, eps=eps
        )

    def rms_norm(self, x: Any, weight: Any, eps: float) -> Any:
        rms = self._torch.sqrt(self._torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return weight * (x / rms)

    def reshape(self, x: Any, shape: tuple[int, ...]) -> Any:
        return x.reshape(shape)

    def transpose(self, x: Any, axes: tuple[int, ...]) -> Any:
        return x.permute(axes)

    def concatenate(self, tensors: list[Any], axis: int = 0) -> Any:
        return self._torch.cat(tensors, dim=axis)

    def split(self, x: Any, num_splits: int, axis: int = 0) -> list[Any]:
        return list(self._torch.chunk(x, num_splits, dim=axis))

    def scaled_dot_product_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        mask: Any | None = None,
        scale: float | None = None,
    ) -> Any:
        return self._torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, scale=scale
        )

    def create_causal_mask(self, seq_len: int, dtype: Any = None) -> Any:
        mask = self._torch.triu(
            self._torch.ones(seq_len, seq_len, device=self._device),
            diagonal=1,
        )
        return mask.masked_fill(mask == 1, float("-inf"))

    def stop_gradient(self, x: Any) -> Any:
        return x.detach()

    def eval(self, *tensors: Any) -> None:
        # PyTorch is eager, no-op
        pass
