"""
Backend registry and management.

Provides global backend instance management with auto-detection.
"""

from __future__ import annotations

import logging

from .base import Backend
from .types import BackendType

logger = logging.getLogger(__name__)

# Global backend registry
_current_backend: Backend | None = None


def _auto_backend_type(device: str | None = None) -> BackendType:
    """Resolve the default backend while preserving existing platform semantics."""
    import platform

    system = platform.system()
    if device is not None:
        normalized = device.lower()
        if normalized == "mps":
            return BackendType.MLX if system == "Darwin" else BackendType.TORCH
        if normalized == "cpu" or normalized.startswith("cuda"):
            return BackendType.TORCH

    return BackendType.MLX if system == "Darwin" else BackendType.TORCH


def _matches_request(
    backend: Backend,
    backend_type: BackendType | None,
    device: str | None,
    check_sm: bool,
) -> bool:
    """Check whether a cached backend already satisfies the requested selector."""
    if backend_type is not None and backend.name != backend_type:
        return False

    if device is not None and backend.name == BackendType.TORCH and backend.device != device:
        return False

    if backend.name == BackendType.TORCH and getattr(backend, "_check_sm", True) != check_sm:
        return False

    return True


def _build_backend(
    backend_type: BackendType,
    *,
    device: str | None = None,
    check_sm: bool = True,
) -> Backend:
    """Instantiate a backend for the requested selector."""
    if backend_type == BackendType.MLX:
        from .mlx_backend import MLXBackend

        return MLXBackend()

    if backend_type == BackendType.TORCH:
        from .torch_backend import TorchBackend

        return TorchBackend(device=device or "cuda", check_sm=check_sm)

    raise ValueError(f"Unsupported backend: {backend_type}")


def get_backend(
    name: BackendType | str | None = None,
    device: str | None = None,
    check_sm: bool = True,
) -> Backend:
    """
    Get the current backend.

    Auto-detects the best backend based on platform when no selector is provided:
    - macOS: MLX (Apple Silicon optimized)
    - Other: PyTorch (CUDA/CPU)

    Args:
        name: Optional backend selector such as ``"mlx"`` or ``"torch"``.
        device: Optional device selector for Torch backends.
        check_sm: Whether CUDA capability should influence Torch defaults.
    """
    global _current_backend

    requested_type = BackendType(name) if name is not None else None
    if _current_backend is not None and _matches_request(
        _current_backend,
        requested_type,
        device,
        check_sm,
    ):
        return _current_backend

    backend_type = requested_type or _auto_backend_type(device)

    if requested_type is None and backend_type == BackendType.MLX:
        try:
            _current_backend = _build_backend(BackendType.MLX)
            logger.debug("Auto-selected MLX backend for macOS")
            return _current_backend
        except ImportError:
            backend_type = BackendType.TORCH
            logger.debug("Falling back to PyTorch backend (MLX not available)")

    _current_backend = _build_backend(backend_type, device=device, check_sm=check_sm)
    if requested_type is None:
        logger.debug("Auto-selected %s backend", _current_backend.name)
    else:
        logger.debug(
            "Selected %s backend with device=%s check_sm=%s",
            _current_backend.name,
            device,
            check_sm,
        )

    return _current_backend


def set_backend(backend: Backend | BackendType | str) -> None:
    """
    Set the current backend.

    Args:
        backend: Backend instance, BackendType enum, or string name
    """
    global _current_backend

    if isinstance(backend, Backend):
        _current_backend = backend
    elif isinstance(backend, (BackendType, str)):
        backend_type = BackendType(str(backend))
        _current_backend = _build_backend(backend_type)
    else:
        raise TypeError(f"Expected Backend, BackendType, or str, got {type(backend)}")

    logger.debug(f"Set backend to {_current_backend.name}")


def reset_backend() -> None:
    """Reset to default backend (auto-detection on next get_backend() call)."""
    global _current_backend
    _current_backend = None
    logger.debug("Reset backend to auto-detection")
