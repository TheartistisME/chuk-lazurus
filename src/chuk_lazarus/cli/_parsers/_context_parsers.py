"""Orchestrator that re-composes the legacy ``register_context_parsers``."""

from ._context_generate import (
    register_context_calibrate_parser,
    register_context_generate_parser,
)
from ._context_prefill import register_context_prefill_parser


def register_context_parsers(subparsers) -> None:
    """Register the ``context`` subcommand tree."""

    context_parser = subparsers.add_parser(
        "context", help="KV context checkpoint tools (prefill / generate)"
    )
    ctx_subparsers = context_parser.add_subparsers(dest="ctx_command", help="Context commands")

    register_context_prefill_parser(ctx_subparsers)
    register_context_generate_parser(ctx_subparsers)
    register_context_calibrate_parser(ctx_subparsers)


__all__ = [
    "register_context_calibrate_parser",
    "register_context_generate_parser",
    "register_context_parsers",
    "register_context_prefill_parser",
]
