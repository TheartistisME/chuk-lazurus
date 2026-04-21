"""Minimal CLI for axis-4 ``SessionRetriever``.

Usage
-----
``python -m chuk_lazarus.session_retrieval.cli \\
    --checkpoint-root /path/to/_sessions/checkpoints \\
    --original-input-root /path/to/_sessions/records \\
    --mode topical \\
    --query "What did we decide?"``

Prints the :class:`QueryResult` as JSON on stdout, converting ``Path`` objects
to strings and dropping non-serialisable fields. This is a thin wrapper around
:meth:`SessionRetriever.from_checkpoint_root` and the three query methods;
library code itself never prints.

Primitive delegated to
----------------------
Entirely to :mod:`chuk_lazarus.session_retrieval.retriever`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from chuk_lazarus.session_retrieval.retriever import QueryResult, SessionRetriever


def _result_to_json_dict(result: QueryResult) -> dict:
    """Convert a :class:`QueryResult` to a JSON-serialisable dict."""
    data = asdict(result)
    # asdict() already handles primitives + lists + dicts; window_keywords and
    # strict_assertions are already JSON-safe. There are no Path fields on
    # QueryResult itself, so nothing further to convert.
    return data


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m chuk_lazarus.session_retrieval.cli",
        description=(
            "Query across per-session clause-aligned checkpoints and emit "
            "a JSON QueryResult."
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Root directory of per-session checkpoints (axis-3 output).",
    )
    parser.add_argument(
        "--original-input-root",
        type=Path,
        default=None,
        help=(
            "Optional root of axis-2 per-turn records "
            "(<original-input-root>/<session_id>/*.json)."
        ),
    )
    parser.add_argument(
        "--model",
        default="google/gemma-4-E2B-it",
        help="HuggingFace model id to load.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device (cuda recommended; cpu bypasses several strict checks).",
    )
    parser.add_argument(
        "--mode",
        choices=("exact", "topical", "entity"),
        required=True,
        help="Routing mode.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help=(
            "The query. For --mode exact this must be a dotted handle "
            "<session_id>.<turn_index>.<chunk_index>."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a shell-style exit code."""
    args = _parse_args(argv)

    retriever = SessionRetriever.from_checkpoint_root(
        args.checkpoint_root,
        model_id=args.model,
        device=args.device,
        original_input_root=args.original_input_root,
    )

    if args.mode == "exact":
        result = retriever.query_exact_id(args.query)
    elif args.mode == "topical":
        result = retriever.query_topical(args.query)
    else:  # args.mode == "entity"
        result = retriever.query_entity_mention(args.query)

    payload = _result_to_json_dict(result)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
