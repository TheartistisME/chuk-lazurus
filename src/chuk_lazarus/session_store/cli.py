"""Axis-3 session-store CLI.

Usage
-----
    python -m chuk_lazarus.session_store.cli \
        --session-id <id> \
        --input-dir <path> \
        --checkpoint-root <path> \
        [--model <path>] \
        [--window-size 512] \
        [--overlap-tokens 64] \
        [--device cuda] \
        [--force] \
        [--verify] \
        [--dry-run]

Exits with the subprocess returncode from the build script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chuk_lazarus.session_store.invoke import invoke_build
from chuk_lazarus.session_store.verify import (
    verify_checkpoint,
    verify_metadata_preservation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chuk_lazarus.session_store.cli",
        description=(
            "Axis-3 session-store CLI: invoke the clause-aligned build "
            "script for a single session (zero-modification subprocess "
            "wrapper)."
        ),
    )
    parser.add_argument("--session-id", required=True, help="Logical session ID")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of per-clause JSON files (axis-2 output)",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Parent dir for per-session checkpoints",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional override for base model path",
    )
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run post-build verification (skipped on failure or dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the argv that would be executed; no subprocess spawned",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    result = invoke_build(
        session_id=args.session_id,
        input_dir=args.input_dir,
        checkpoint_root=args.checkpoint_root,
        model=args.model,
        window_size=args.window_size,
        overlap_tokens=args.overlap_tokens,
        device=args.device,
        force=args.force,
        dry_run=args.dry_run,
    )

    summary: dict = {
        "session_id": result.session_id,
        "checkpoint": str(result.checkpoint),
        "returncode": result.returncode,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        summary["argv"] = result.stdout

    if args.verify and result.returncode == 0 and not args.dry_run:
        try:
            summary["verify_checkpoint"] = verify_checkpoint(result.checkpoint)
            summary["verify_metadata"] = verify_metadata_preservation(args.input_dir)
        except AssertionError as exc:
            summary["verify_error"] = str(exc)
            # Verification failure: surface via non-zero exit.
            print(json.dumps(summary, indent=2))
            return 2

    print(json.dumps(summary, indent=2))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
