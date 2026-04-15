"""
Complete models combining backbone + head.

Top-level ``import mlx.*`` is intentionally absent: re-exports are resolved
lazily via PEP 562 module ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .base import Model, ModelOutput  # noqa: F401
    from .causal_lm import CausalLM, create_causal_lm  # noqa: F401
    from .classifiers import (  # noqa: F401
        LinearClassifier,
        MLPClassifier,
        SequenceClassifier,
        TokenClassifier,
        create_classifier,
    )

__all__ = [
    "Model",
    "ModelOutput",
    "CausalLM",
    "create_causal_lm",
    "LinearClassifier",
    "MLPClassifier",
    "SequenceClassifier",
    "TokenClassifier",
    "create_classifier",
]

_LAZY: dict[str, tuple[str, str]] = {
    "Model": (".base", "Model"),
    "ModelOutput": (".base", "ModelOutput"),
    "CausalLM": (".causal_lm", "CausalLM"),
    "create_causal_lm": (".causal_lm", "create_causal_lm"),
    "LinearClassifier": (".classifiers", "LinearClassifier"),
    "MLPClassifier": (".classifiers", "MLPClassifier"),
    "SequenceClassifier": (".classifiers", "SequenceClassifier"),
    "TokenClassifier": (".classifiers", "TokenClassifier"),
    "create_classifier": (".classifiers", "create_classifier"),
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
