"""Infer ``run`` subcommand parser (post-EWS-0 split target).

The legacy ``_infer.py`` is reduced to a one-line shim that re-exports
``register_infer_parser`` from this module so downstream importers continue
to work.  Epic 2 EWS-1 owns this file going forward; EWS-0 seeded it and
this revision switches backend-flag registration to the shared helper
``add_backend_flags`` from ``cli.commands._base`` (single source of truth).
"""

import asyncio

from ..commands._base import add_backend_flags
from ..commands.infer import run_inference


def register_infer_parser(subparsers):
    """Register the infer subcommand."""
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model", required=True, help="Model name or path")
    infer_parser.add_argument("--adapter", help="LoRA adapter path")
    infer_parser.add_argument("--prompt", help="Single prompt")
    infer_parser.add_argument("--prompt-file", help="File with prompts (one per line)")
    infer_parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens")
    infer_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    infer_parser.add_argument(
        "--chat", action="store_true", help="Use chat template (for chat models)"
    )
    infer_parser.add_argument("--system", help="System prompt (only used with --chat)")
    infer_parser.add_argument(
        "--engine",
        choices=["standard", "kv_direct"],
        default="standard",
        help="Inference engine (default: standard)",
    )
    add_backend_flags(infer_parser)
    infer_parser.set_defaults(func=lambda args: asyncio.run(run_inference(args)))


__all__ = ["register_infer_parser"]
