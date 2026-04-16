"""
Steering subpackage for activation steering.

The public API is resolved lazily so torch hosts can import
``chuk_lazarus.introspection.steering`` without constructing MLX-only
symbols such as ``SteeredGemmaMLP``.
"""

from __future__ import annotations

import importlib


_LAZY_GROUPS: dict[str, list[str]] = {
    ".config": ["LegacySteeringConfig", "SteeringConfig", "SteeringMode"],
    ".core": ["ActivationSteering"],
    ".hook": ["SteeringHook"],
    ".legacy": ["SteeredGemmaMLP", "ToolCallingSteering"],
    ".service": [
        "DirectionExtractionResult",
        "SteeringComparisonResult",
        "SteeringGenerationResult",
        "SteeringService",
        "SteeringServiceConfig",
    ],
    ".utils": ["compare_steering_effects", "format_functiongemma_prompt", "steer_model"],
}

_LAZY: dict[str, tuple[str, str]] = {
    name: (module, name)
    for module, names in _LAZY_GROUPS.items()
    for name in names
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
