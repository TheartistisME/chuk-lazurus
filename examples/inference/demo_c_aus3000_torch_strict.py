#!/usr/bin/env python3
"""DEPRECATED shim — use examples/inference/demo_clause_aligned_strict.py instead.

This file used to contain the aus3000-specific strict-injection demo. It has
been replaced by a generic clause-aligned demo whose --store flag defaults to
the aus3000 clause-aligned variant, so the old invocation path still works.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERIC_SCRIPT = REPO_ROOT / "examples" / "inference" / "demo_clause_aligned_strict.py"


def main() -> int:
    print(
        "[DEPRECATED] examples/inference/demo_c_aus3000_torch_strict.py is a shim.\n"
        "             Use examples/inference/demo_clause_aligned_strict.py directly.\n"
        "             The generic demo already defaults --store to the aus3000 clause-aligned variant.",
        file=sys.stderr,
    )
    os.execvp(sys.executable, [sys.executable, str(GENERIC_SCRIPT), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
