"""
Runtime backend resolution for inference.
"""

from __future__ import annotations

import importlib.util
import os
import platform

from .types import LazarusBackend

BACKEND_ENV_VAR = "LAZARUS_BACKEND"
CHUK_BACKEND_ENV_VAR = "CHUK_BACKEND"

_CHUK_TO_LAZARUS = {"torch": LazarusBackend.CUDA, "mlx": LazarusBackend.MLX}


def _from_chuk(value: str) -> LazarusBackend:
    """Map a CHUK_BACKEND value (torch|mlx) to a LazarusBackend."""
    key = value.strip().lower()
    if key not in _CHUK_TO_LAZARUS:
        raise RuntimeError(
            f"Unsupported CHUK_BACKEND value {value!r}. Expected one of: "
            f"{sorted(_CHUK_TO_LAZARUS)}."
        )
    return _CHUK_TO_LAZARUS[key]


def is_backend_available(backend: LazarusBackend) -> bool:
    """Return whether a backend can be used in the current environment."""
    if backend == LazarusBackend.MLX:
        if importlib.util.find_spec("mlx") is None:
            return False
        try:
            import mlx.core  # noqa: F401
        except Exception:
            return False
        return True

    if backend == LazarusBackend.CUDA:
        if importlib.util.find_spec("torch") is None:
            return False
        import torch

        return torch.cuda.is_available()

    return False


def detect_default_backend() -> LazarusBackend:
    """Pick the best available backend for the current machine."""
    system = platform.system()

    if system == "Darwin" and is_backend_available(LazarusBackend.MLX):
        return LazarusBackend.MLX

    if is_backend_available(LazarusBackend.CUDA):
        return LazarusBackend.CUDA

    if is_backend_available(LazarusBackend.MLX):
        return LazarusBackend.MLX

    raise RuntimeError(
        "No supported inference backend is available. Install MLX for Apple Silicon "
        "or PyTorch with CUDA support for Nvidia GPUs."
    )


def resolve_backend(requested: LazarusBackend | str | None = None) -> LazarusBackend:
    """Resolve the runtime backend from explicit config, env, or auto-detection.

    Precedence: explicit ``requested`` arg > ``LAZARUS_BACKEND`` env
    > ``CHUK_BACKEND`` env (mapped torch->cuda, mlx->mlx) > auto-detect.

    If both ``LAZARUS_BACKEND`` and ``CHUK_BACKEND`` are set and disagree after
    mapping, a :class:`RuntimeError` is raised.
    """
    if requested is not None:
        raw_backend: LazarusBackend | str | None = requested
    else:
        lazarus_env = os.environ.get(BACKEND_ENV_VAR)
        chuk_env = os.environ.get(CHUK_BACKEND_ENV_VAR)

        if lazarus_env and chuk_env:
            lazarus_val = LazarusBackend(lazarus_env)
            chuk_val = _from_chuk(chuk_env)
            if lazarus_val != chuk_val:
                raise RuntimeError(
                    f"Conflicting backend env vars: {BACKEND_ENV_VAR}={lazarus_env!r} "
                    f"vs {CHUK_BACKEND_ENV_VAR}={chuk_env!r} "
                    f"(maps to {chuk_val.value!r}). Unset one to resolve."
                )
            raw_backend = lazarus_val
        elif lazarus_env:
            raw_backend = lazarus_env
        elif chuk_env:
            raw_backend = _from_chuk(chuk_env)
        else:
            raw_backend = None

    backend = LazarusBackend(raw_backend) if raw_backend else detect_default_backend()

    if not is_backend_available(backend):
        if backend == LazarusBackend.CUDA:
            raise RuntimeError(
                "CUDA backend requested but no CUDA-enabled PyTorch device is available."
            )
        raise RuntimeError("MLX backend requested but MLX is not installed.")

    return backend
