"""
Output heads for different tasks.

Top-level ``import mlx.*`` is intentionally absent: re-exports are resolved
lazily via PEP 562 module ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .base import Head, HeadOutput  # noqa: F401
    from .classifier import ClassifierHead, PoolerHead, create_classifier_head  # noqa: F401
    from .lm_head import LMHead, create_lm_head, cross_entropy_loss  # noqa: F401
    from .regression import RegressionHead, create_regression_head  # noqa: F401

__all__ = [
    "Head",
    "HeadOutput",
    "LMHead",
    "create_lm_head",
    "cross_entropy_loss",
    "ClassifierHead",
    "PoolerHead",
    "create_classifier_head",
    "RegressionHead",
    "create_regression_head",
]

_LAZY: dict[str, tuple[str, str]] = {
    "Head": (".base", "Head"),
    "HeadOutput": (".base", "HeadOutput"),
    "LMHead": (".lm_head", "LMHead"),
    "create_lm_head": (".lm_head", "create_lm_head"),
    "cross_entropy_loss": (".lm_head", "cross_entropy_loss"),
    "ClassifierHead": (".classifier", "ClassifierHead"),
    "PoolerHead": (".classifier", "PoolerHead"),
    "create_classifier_head": (".classifier", "create_classifier_head"),
    "RegressionHead": (".regression", "RegressionHead"),
    "create_regression_head": (".regression", "create_regression_head"),
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
