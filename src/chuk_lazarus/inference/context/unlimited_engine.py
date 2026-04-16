"""Shim — canonical location: research/unlimited_engine.py

Lazy re-export (EWS-3): the underlying ``research/unlimited_engine`` imports
``mlx.core`` at module scope, so the shim must defer resolution until
attribute access.  Keeps ``import chuk_lazarus.inference.context.unlimited_engine``
lazy-import clean under ``CHUK_BACKEND=torch`` on hosts without MLX.
"""

from __future__ import annotations


def __getattr__(name):
    from .research import unlimited_engine as _impl

    value = getattr(_impl, name)
    globals()[name] = value
    return value


def __dir__():
    from .research import unlimited_engine as _impl

    return sorted(set(list(globals().keys()) + dir(_impl)))
