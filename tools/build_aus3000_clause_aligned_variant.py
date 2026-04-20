#!/usr/bin/env python3
"""DEPRECATED shim — use tools/build_clause_aligned_store.py instead.

This file used to contain the aus3000-specific clause-aligned builder. It has
been replaced by a generic clause-aware builder. This shim preserves the old
invocation path by forwarding with aus3000 defaults filled in.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERIC_SCRIPT = REPO_ROOT / "tools" / "build_clause_aligned_store.py"

# aus3000-friendly defaults (previously hardcoded in this file)
DEFAULT_INPUT_DIR = (
    "/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/"
    "embeddings/AS_NZS_3000-2018"
)
DEFAULT_CHECKPOINT = (
    "/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/"
    "embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant"
)


def main() -> int:
    print(
        "[DEPRECATED] tools/build_aus3000_clause_aligned_variant.py is a shim.\n"
        "             Use tools/build_clause_aligned_store.py directly.\n"
        "             This shim fills in aus3000 defaults and delegates.",
        file=sys.stderr,
    )
    argv = sys.argv[1:]
    if "--input-dir" not in argv:
        argv.extend(["--input-dir", DEFAULT_INPUT_DIR])
    if "--checkpoint" not in argv:
        argv.extend(["--checkpoint", DEFAULT_CHECKPOINT])
    os.execvp(sys.executable, [sys.executable, str(GENERIC_SCRIPT), *argv])


if __name__ == "__main__":
    raise SystemExit(main())
