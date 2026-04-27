#!/usr/bin/env python3
"""Compatibility wrapper for the coding memory eval."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "prod" / "evals" / "coding-eval" / "evaluate_memory_coding_task.py"
    runpy.run_path(str(target), run_name="__main__")
