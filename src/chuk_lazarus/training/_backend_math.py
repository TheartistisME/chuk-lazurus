"""Tiny dispatch helper for dual-backend math (EWS-9).

Provides an ``xp`` object (mlx.core or torch) selected from either:
- an input tensor's module, or
- ``CHUK_BACKEND`` env var, or
- MLX default (historical).

The exposed helpers wrap the small set of primitives the training losses +
utils need (log, exp, sigmoid, sum, mean, var, sqrt, clip, softmax,
log_softmax, arange, zeros, zeros_like, abs_, maximum, reshape, stack,
concatenate, argmax). Everything else is passed through via ``xp.<op>``.

Lazy: all framework imports are function-scope so this module is
lazy-import-clean under ``CHUK_BACKEND=torch``.
"""

from __future__ import annotations

import os
from typing import Any


def detect_backend(x: Any) -> str:
    """Return 'mlx' or 'torch' from a tensor-ish value."""
    mod = type(x).__module__ if x is not None else ""
    if mod.startswith("mlx"):
        return "mlx"
    if mod.startswith("torch"):
        return "torch"
    env = os.environ.get("CHUK_BACKEND", "").lower()
    if env in ("mlx", "torch"):
        return env
    return "mlx"


def xp_for(backend: str):
    """Return a numpy-like module for the backend.

    Exposes: log, exp, sigmoid, softmax, log_softmax, sum, mean, var, sqrt,
    clip, arange, zeros, zeros_like, abs, maximum, minimum, reshape, stack,
    concatenate, argmax, array, float32.
    """
    if backend == "torch":
        import torch

        class _TorchXP:
            float32 = torch.float32

            log = staticmethod(torch.log)
            exp = staticmethod(torch.exp)
            sigmoid = staticmethod(torch.sigmoid)
            sqrt = staticmethod(torch.sqrt)
            abs = staticmethod(torch.abs)
            maximum = staticmethod(torch.maximum)
            minimum = staticmethod(torch.minimum)
            arange = staticmethod(torch.arange)
            stack = staticmethod(torch.stack)
            argmax = staticmethod(torch.argmax)

            @staticmethod
            def softmax(x, axis=-1):
                return torch.softmax(x, dim=axis)

            @staticmethod
            def log_softmax(x, axis=-1):
                return torch.log_softmax(x, dim=axis)

            @staticmethod
            def sum(x, axis=None, keepdims=False):
                if axis is None:
                    return torch.sum(x)
                return torch.sum(x, dim=axis, keepdim=keepdims)

            @staticmethod
            def mean(x, axis=None, keepdims=False):
                if axis is None:
                    return torch.mean(x)
                return torch.mean(x, dim=axis, keepdim=keepdims)

            @staticmethod
            def var(x, axis=None, keepdims=False):
                if axis is None:
                    return torch.var(x, unbiased=False)
                return torch.var(x, dim=axis, unbiased=False, keepdim=keepdims)

            @staticmethod
            def clip(x, lo, hi):
                return torch.clamp(x, lo, hi)

            @staticmethod
            def zeros(shape, dtype=None):
                dtype = dtype or torch.float32
                return torch.zeros(shape, dtype=dtype)

            @staticmethod
            def zeros_like(x):
                return torch.zeros_like(x)

            @staticmethod
            def concatenate(xs, axis=0):
                return torch.cat(xs, dim=axis)

            @staticmethod
            def reshape(x, shape):
                return x.reshape(shape)

            @staticmethod
            def array(data, dtype=None):
                if dtype is None:
                    return torch.as_tensor(data)
                return torch.as_tensor(data, dtype=dtype)

        return _TorchXP()

    # mlx path
    import mlx.core as _mx

    class _MlxXP:
        float32 = _mx.float32

        log = staticmethod(_mx.log)
        exp = staticmethod(_mx.exp)
        sigmoid = staticmethod(_mx.sigmoid)
        softmax = staticmethod(_mx.softmax)
        log_softmax = staticmethod(_mx.log_softmax)
        sum = staticmethod(_mx.sum)
        mean = staticmethod(_mx.mean)
        var = staticmethod(_mx.var)
        sqrt = staticmethod(_mx.sqrt)
        clip = staticmethod(_mx.clip)
        arange = staticmethod(_mx.arange)
        zeros = staticmethod(_mx.zeros)
        zeros_like = staticmethod(_mx.zeros_like)
        abs = staticmethod(_mx.abs)
        maximum = staticmethod(_mx.maximum)
        minimum = staticmethod(_mx.minimum)
        reshape = staticmethod(_mx.reshape)
        stack = staticmethod(_mx.stack)
        concatenate = staticmethod(_mx.concatenate)
        argmax = staticmethod(_mx.argmax)
        array = staticmethod(_mx.array)

    return _MlxXP()
