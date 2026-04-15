"""
Llama model family.

Top-level ``import mlx.*`` is intentionally absent: re-exports resolved
via PEP 562 module ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .config import LlamaConfig  # noqa: F401
    from .convert import (  # noqa: F401
        LLAMA_WEIGHT_MAP,
        convert_hf_weights,
        load_hf_config,
        load_weights,
    )
    from .model import LlamaBlock, LlamaForCausalLM, LlamaModel  # noqa: F401

__all__ = [
    "LlamaConfig",
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaBlock",
    "load_hf_config",
    "load_weights",
    "convert_hf_weights",
    "LLAMA_WEIGHT_MAP",
]

_LAZY: dict[str, tuple[str, str]] = {
    "LlamaConfig": (".config", "LlamaConfig"),
    "LlamaForCausalLM": (".model", "LlamaForCausalLM"),
    "LlamaModel": (".model", "LlamaModel"),
    "LlamaBlock": (".model", "LlamaBlock"),
    "load_hf_config": (".convert", "load_hf_config"),
    "load_weights": (".convert", "load_weights"),
    "convert_hf_weights": (".convert", "convert_hf_weights"),
    "LLAMA_WEIGHT_MAP": (".convert", "LLAMA_WEIGHT_MAP"),
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
