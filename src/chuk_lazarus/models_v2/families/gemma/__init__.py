"""
Gemma 3 model family.

Top-level ``import mlx.*`` is intentionally absent: re-exports resolve
via PEP 562 module ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .config import GemmaConfig  # noqa: F401
    from .convert import (  # noqa: F401
        GEMMA_WEIGHT_MAP,
        convert_hf_weights,
        convert_mlx_community_weights,
        load_hf_config,
        load_weights,
    )
    from .model import (  # noqa: F401
        FunctionGemmaForCausalLM,
        GemmaAttention,
        GemmaBlock,
        GemmaForCausalLM,
        GemmaMLP,
        GemmaModel,
        GemmaRMSNorm,
    )

__all__ = [
    # Config
    "GemmaConfig",
    # Model components
    "GemmaRMSNorm",
    "GemmaMLP",
    "GemmaAttention",
    "GemmaBlock",
    "GemmaModel",
    # Full models
    "GemmaForCausalLM",
    "FunctionGemmaForCausalLM",
    # Loading utilities
    "load_hf_config",
    "load_weights",
    "convert_hf_weights",
    "convert_mlx_community_weights",
    "GEMMA_WEIGHT_MAP",
]

_LAZY: dict[str, tuple[str, str]] = {
    "GemmaConfig": (".config", "GemmaConfig"),
    "GemmaRMSNorm": (".model", "GemmaRMSNorm"),
    "GemmaMLP": (".model", "GemmaMLP"),
    "GemmaAttention": (".model", "GemmaAttention"),
    "GemmaBlock": (".model", "GemmaBlock"),
    "GemmaModel": (".model", "GemmaModel"),
    "GemmaForCausalLM": (".model", "GemmaForCausalLM"),
    "FunctionGemmaForCausalLM": (".model", "FunctionGemmaForCausalLM"),
    "load_hf_config": (".convert", "load_hf_config"),
    "load_weights": (".convert", "load_weights"),
    "convert_hf_weights": (".convert", "convert_hf_weights"),
    "convert_mlx_community_weights": (".convert", "convert_mlx_community_weights"),
    "GEMMA_WEIGHT_MAP": (".convert", "GEMMA_WEIGHT_MAP"),
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
