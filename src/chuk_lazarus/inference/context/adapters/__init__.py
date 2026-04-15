"""Model adapters implementing TransformerLayerProtocol / ModelBackboneProtocol.

Public symbols are resolved lazily via PEP 562 ``__getattr__`` so that
``import chuk_lazarus.inference.context.adapters`` completes under
``CHUK_BACKEND=torch`` without pulling ``libmlx.so`` until an adapter
class is actually accessed.  Adapter method bodies still invoke MLX ops
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .gemma_adapter import GemmaBackboneAdapter, GemmaLayerAdapter
    from .llama_adapter import LlamaBackboneAdapter, LlamaLayerAdapter


_LAZY: dict[str, tuple[str, str]] = {
    "GemmaBackboneAdapter": (".gemma_adapter", "GemmaBackboneAdapter"),
    "GemmaLayerAdapter": (".gemma_adapter", "GemmaLayerAdapter"),
    "LlamaBackboneAdapter": (".llama_adapter", "LlamaBackboneAdapter"),
    "LlamaLayerAdapter": (".llama_adapter", "LlamaLayerAdapter"),
}

__all__ = sorted(_LAZY.keys())


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name, __name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(globals().keys()) + __all__
