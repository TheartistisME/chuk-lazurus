"""
Reusable model components.

Top-level submodule imports are intentionally absent; submodules are
exposed lazily via PEP 562 module ``__getattr__`` so that importing this
package does not pull ``mlx.*`` under ``CHUK_BACKEND=torch``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from . import attention, embeddings, ffn, normalization, recurrent, ssm  # noqa: F401

__all__ = ["embeddings", "attention", "ffn", "normalization", "ssm", "recurrent"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
