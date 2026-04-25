"""Vector injection (vec_inject) public surface."""

from __future__ import annotations

_LAZY: dict[str, tuple[str, str]] = {
    "vec_inject": ("._primitives", "vec_inject"),
    "vec_inject_all": ("._primitives", "vec_inject_all"),
    "VecInjectProvider": ("._protocol", "VecInjectProvider"),
    "SourceType": ("._types", "SourceType"),
    "VecInjectMatch": ("._types", "VecInjectMatch"),
    "VecInjectMeta": ("._types", "VecInjectMeta"),
    "VecInjectResult": ("._types", "VecInjectResult"),
    "KV_ROUTE_FILE": (".providers", "KV_ROUTE_FILE"),
    "VEC_INJECT_FILE": (".providers", "VEC_INJECT_FILE"),
    "LocalVecInjectProvider": (".providers", "LocalVecInjectProvider"),
    "VecInjectMetaKey": (".providers", "VecInjectMetaKey"),
    "VecInjectWindowKey": (".providers", "VecInjectWindowKey"),
    # axis-BC PROP K.5 + K.0
    "vec_inject_to_kv_direct": (".kv_direct_adapter", "vec_inject_to_kv_direct"),
    "inject_via_kv_direct": (".kv_direct_adapter", "inject_via_kv_direct"),
    "SlidingWindowLayerRefusedError": (
        ".kv_direct_adapter",
        "SlidingWindowLayerRefusedError",
    ),
    "assert_global_attention_layer": (
        ".kv_direct_adapter",
        "assert_global_attention_layer",
    ),
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
