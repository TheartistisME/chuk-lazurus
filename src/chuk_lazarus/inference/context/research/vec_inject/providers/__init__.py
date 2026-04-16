"""Concrete VecInjectProvider implementations."""

from __future__ import annotations

_LAZY: dict[str, tuple[str, str]] = {
    "KNOWLEDGE_STORE_FILE": ("._index_format", "KNOWLEDGE_STORE_FILE"),
    "KV_ROUTE_FILE": ("._index_format", "KV_ROUTE_FILE"),
    "VEC_INJECT_FILE": ("._index_format", "VEC_INJECT_FILE"),
    "KnowledgeStoreKey": ("._index_format", "KnowledgeStoreKey"),
    "VecInjectMetaKey": ("._index_format", "VecInjectMetaKey"),
    "VecInjectWindowKey": ("._index_format", "VecInjectWindowKey"),
    "LocalVecInjectProvider": ("._local_file", "LocalVecInjectProvider"),
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
    return sorted(list(globals().keys()) + __all__)
