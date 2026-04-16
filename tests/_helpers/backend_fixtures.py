"""Backend-selection helpers shared across Epic 2 tests.

See ``docs/refactor/dual-backend-cuda-all-buckets/03-workstreams.md`` §EWS-0.
"""

from __future__ import annotations

import importlib
from functools import lru_cache


@lru_cache(maxsize=None)
def _module_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def mlx_available() -> bool:
    """Return True when ``mlx.core`` is importable on this host."""

    return _module_importable("mlx.core")


def torch_available() -> bool:
    """Return True when ``torch`` is importable on this host."""

    return _module_importable("torch")


def cuda_available() -> bool:
    """Return True when torch.cuda reports a usable device."""

    if not torch_available():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def available_backends() -> tuple[str, ...]:
    """Return the subset of ``("mlx", "torch")`` importable on this host."""

    found: list[str] = []
    if mlx_available():
        found.append("mlx")
    if torch_available():
        found.append("torch")
    return tuple(found)


def get_backend_or_skip(name: str, device: str | None = None):
    """Return a ``Backend`` instance for *name* or ``pytest.skip`` if unavailable.

    The import is intentionally local so importing this helper does not drag
    ``mlx`` / ``torch`` into modules that don't need them.
    """

    import pytest

    if name == "mlx" and not mlx_available():
        pytest.skip("mlx not importable on this host")
    if name == "torch" and not torch_available():
        pytest.skip("torch not importable on this host")

    from chuk_lazarus.models_v2.core.backend import get_backend

    return get_backend(name, device=device, check_sm=False)


__all__ = [
    "available_backends",
    "cuda_available",
    "get_backend_or_skip",
    "mlx_available",
    "torch_available",
]
