"""``context prefill`` parser (split from legacy ``_context.py``).

EWS-0 carved this out verbatim.  Ownership transfers to EWS-2 at merge.
"""

import asyncio

from ..commands._base import add_backend_flags
from ..commands.context import context_prefill_cmd


def register_context_prefill_parser(ctx_subparsers) -> None:
    """Register the ``context prefill`` subcommand."""

    ctx_prefill = ctx_subparsers.add_parser(
        "prefill", help="Prefill a document into a windowed checkpoint library"
    )
    ctx_prefill.add_argument("--model", "-m", required=True, help="Model ID or local path")
    ctx_prefill.add_argument("--input", "-i", required=True, help="Input text file to prefill")
    ctx_prefill.add_argument("--checkpoint", "-c", required=True, help="Output library directory")
    ctx_prefill.add_argument(
        "--window-size",
        type=int,
        default=None,
        dest="window_size",
        help="Tokens per window (default: 8192)",
    )
    ctx_prefill.add_argument(
        "--max-tokens",
        type=int,
        dest="max_tokens",
        help="Truncate input to at most N tokens",
    )
    ctx_prefill.add_argument(
        "--name",
        help="Human-readable library name (defaults to input filename stem)",
    )
    ctx_prefill.add_argument(
        "--no-resume",
        action="store_true",
        dest="no_resume",
        help="Ignore existing partial library and start fresh",
    )
    ctx_prefill.add_argument(
        "--residual-mode",
        choices=["interval", "full", "none", "darkspace"],
        default="interval",
        dest="residual_mode",
        help="Residual extraction: interval (8 samples), full (every position), darkspace (frame bank), none (skip)",
    )
    ctx_prefill.add_argument(
        "--frame-bank",
        dest="frame_bank",
        help="Path to frame_bank.npz (required for --residual-mode darkspace)",
    )
    ctx_prefill.add_argument(
        "--store-pages",
        action="store_true",
        dest="store_pages",
        help="Store pre-RoPE K,V pages for instant page injection at generate time",
    )
    ctx_prefill.add_argument(
        "--store-kv-full",
        action="store_true",
        dest="store_kv_full",
        help="Save full KV cache per window for Mode 6 KV injection (~9MB/window for 4B)",
    )
    ctx_prefill.add_argument(
        "--phases",
        default="all",
        help=(
            "Comma-separated phases to run: windows, interval, compass, darkspace, pages, surprise, sparse, kvectors, kvectors_full, mode7, all. "
            "E.g. --phases windows to prefill only, --phases kvectors to extract K-vector routing index "
            "(sparse/interval sampling), --phases kvectors_full for 100%% position coverage (~256KB/window), "
            "--phases mode7 to calibrate Mode 7 probes. Default: all"
        ),
    )
    ctx_prefill.add_argument(
        "--compass-layer",
        type=int,
        default=None,
        dest="compass_layer",
        help="Explicit layer for compass extraction (default: auto ~77%% depth, e.g. 29 for retrieval-layer routing)",
    )
    ctx_prefill.add_argument(
        "--mode",
        default="standard",
        choices=["standard", "export"],
        help=(
            "Prefill mode: standard (default, writes KV checkpoints for fast replay) or "
            "export (streaming, no KV checkpoints — portable ~33 MB index for demo/distribution)"
        ),
    )
    add_backend_flags(ctx_prefill)
    ctx_prefill.set_defaults(func=lambda args: asyncio.run(context_prefill_cmd(args)))


__all__ = ["register_context_prefill_parser"]
