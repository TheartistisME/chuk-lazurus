"""Introspect command parsers."""

from __future__ import annotations

from importlib import import_module

_REGISTRARS: tuple[tuple[str, str], ...] = (
    ("_analyze", "register_analyze_parsers"),
    ("_ablation", "register_ablation_parsers"),
    ("_layer", "register_layer_parsers"),
    ("_generation", "register_generation_parsers"),
    ("_steering", "register_steering_parsers"),
    ("_arithmetic", "register_arithmetic_parsers"),
    ("_probing", "register_probing_parsers"),
    ("_memory", "register_memory_parsers"),
    ("_directions", "register_directions_parsers"),
    ("_embedding", "register_embedding_parsers"),
    ("_patch", "register_patch_parsers"),
    ("_circuit", "register_circuit_parsers"),
    ("_moe", "register_moe_parsers"),
    ("_classifier", "register_classifier_parsers"),
)


def _load(module_name: str, attr_name: str):
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, attr_name)


def register_introspect_parsers(subparsers):
    """Register the introspect subcommand and all sub-subcommands."""
    introspect_parser = subparsers.add_parser(
        "introspect",
        help="Model introspection and logit lens analysis",
        description="Analyze model behavior using logit lens and attention visualization.",
    )
    introspect_subparsers = introspect_parser.add_subparsers(
        dest="introspect_command", help="Introspection commands"
    )

    for module_name, attr_name in _REGISTRARS:
        _load(module_name, attr_name)(introspect_subparsers)


__all__ = ["register_introspect_parsers"]
