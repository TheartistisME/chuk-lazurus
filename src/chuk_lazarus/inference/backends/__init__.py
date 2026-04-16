"""Inference backend runtime helpers."""

from __future__ import annotations

_LAZY: dict[str, tuple[str, str]] = {
    "InferenceRuntime": (".base", "InferenceRuntime"),
    "MLXInferenceRuntime": (".mlx_runtime", "MLXInferenceRuntime"),
    "prefetch_residual_state": (".prefetch", "prefetch_residual_state"),
    "BACKEND_ENV_VAR": (".registry", "BACKEND_ENV_VAR"),
    "detect_default_backend": (".registry", "detect_default_backend"),
    "is_backend_available": (".registry", "is_backend_available"),
    "resolve_backend": (".registry", "resolve_backend"),
    "load_residual_state": (".storage", "load_residual_state"),
    "save_residual_state": (".storage", "save_residual_state"),
    "TorchInferenceRuntime": (".torch_runtime", "TorchInferenceRuntime"),
    "LazarusBackend": (".types", "LazarusBackend"),
    "ResidualState": (".types", "ResidualState"),
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
