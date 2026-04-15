"""
Loss functions for models_v2.

Pure math: CE, DPO loss, etc. Training loops live in src/chuk_lazarus/training/.

Top-level ``import mlx.*`` is intentionally absent: re-exports are resolved
lazily via PEP 562 module ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .loss import compute_lm_loss  # noqa: F401

__all__ = ["compute_lm_loss"]

_LAZY: dict[str, tuple[str, str]] = {
    "compute_lm_loss": (".loss", "compute_lm_loss"),
    "compute_cross_entropy_loss": (".loss", "compute_cross_entropy_loss"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name, __name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
