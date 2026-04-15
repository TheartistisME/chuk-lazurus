"""
Model family implementations.

Top-level family submodule imports are intentionally absent; every
submodule and registry re-export is resolved lazily via PEP 562 module
``__getattr__`` so that importing this package does not pull ``mlx.*``
under ``CHUK_BACKEND=torch``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from . import gemma, gpt2, granite, jamba, llama, llama4, mamba, qwen3, starcoder2  # noqa: F401
    from .registry import (  # noqa: F401
        FamilyInfo,
        IntrospectionHooks,
        ModelFamilyRegistry,
        ModelFamilyType,
        WeightConverter,
        detect_model_family,
        get_family_info,
        get_family_registry,
        list_model_families,
    )

_FAMILIES = (
    "gemma",
    "gpt2",
    "granite",
    "jamba",
    "llama",
    "llama4",
    "mamba",
    "qwen3",
    "starcoder2",
)

_REGISTRY: dict[str, tuple[str, str]] = {
    "ModelFamilyType": (".registry", "ModelFamilyType"),
    "ModelFamilyRegistry": (".registry", "ModelFamilyRegistry"),
    "FamilyInfo": (".registry", "FamilyInfo"),
    "WeightConverter": (".registry", "WeightConverter"),
    "IntrospectionHooks": (".registry", "IntrospectionHooks"),
    "detect_model_family": (".registry", "detect_model_family"),
    "get_family_info": (".registry", "get_family_info"),
    "get_family_registry": (".registry", "get_family_registry"),
    "list_model_families": (".registry", "list_model_families"),
}

__all__ = list(_FAMILIES) + list(_REGISTRY.keys())


def __getattr__(name: str):
    import importlib

    if name in _FAMILIES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name in _REGISTRY:
        mod_name, attr = _REGISTRY[name]
        mod = importlib.import_module(mod_name, __name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
