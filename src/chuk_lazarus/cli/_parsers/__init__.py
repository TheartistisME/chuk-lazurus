"""Parser registration surface for CLI subcommands.

Keep package import cheap: importing ``chuk_lazarus.cli._parsers`` should not
eagerly import every parser bucket or the introspection tree.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from ..commands._base import BACKEND_CHOICES, add_backend_flags

if TYPE_CHECKING:  # pragma: no cover
    from argparse import _SubParsersAction

__all__ = [
    "BACKEND_CHOICES",
    "add_backend_flags",
    "register_agent_context_parsers",
    "register_bench_parser",
    "register_context_parsers",
    "register_data_parsers",
    "register_experiment_parsers",
    "register_gym_parsers",
    "register_infer_parser",
    "register_introspect_parsers",
    "register_knowledge_parsers",
    "register_tokenizer_parsers",
    "register_train_parsers",
]


def _load(module_name: str, attr_name: str):
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[attr_name] = value
    return value


def register_bench_parser(subparsers):
    return _load("_bench", "register_bench_parser")(subparsers)


def register_agent_context_parsers(subparsers):
    return _load("_agent_context", "register_agent_context_parsers")(subparsers)


def register_context_parsers(subparsers):
    return _load("_context", "register_context_parsers")(subparsers)


def register_data_parsers(subparsers):
    return _load("_data", "register_data_parsers")(subparsers)


def register_experiment_parsers(subparsers):
    return _load("_experiment", "register_experiment_parsers")(subparsers)


def register_gym_parsers(subparsers):
    return _load("_gym", "register_gym_parsers")(subparsers)


def register_infer_parser(subparsers):
    return _load("_infer", "register_infer_parser")(subparsers)


def register_introspect_parsers(subparsers):
    return _load("_introspect", "register_introspect_parsers")(subparsers)


def register_knowledge_parsers(subparsers):
    return _load("_knowledge", "register_knowledge_parsers")(subparsers)


def register_tokenizer_parsers(subparsers):
    return _load("_tokenizer", "register_tokenizer_parsers")(subparsers)


def register_train_parsers(subparsers):
    return _load("_train", "register_train_parsers")(subparsers)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
