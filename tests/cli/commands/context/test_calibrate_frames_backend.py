"""EWS-3 — ``calibrate_frames`` module-level lazy-import contract.

AST-level CI gate (``tests/ci/test_no_top_level_mlx.py``) enforces the
no-top-level-mlx rule.  This file asserts the functional contract that the
handler body imports mlx *inside* the coroutine, not at module scope.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "src/chuk_lazarus/cli/commands/context/calibrate_frames.py"
)


def test_calibrate_frames_no_top_level_mlx_import():
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                if node.module:
                    names = [node.module]
            for n in names:
                assert not n.startswith("mlx"), (
                    f"top-level mlx import at line {node.lineno}: {n}"
                )


def test_calibrate_frames_handler_imports_mlx_lazily():
    tree = ast.parse(SOURCE.read_text())
    handler = next(
        n
        for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef)
        and n.name == "context_calibrate_frames_cmd"
    )
    inner_mlx_imports = [
        stmt
        for stmt in ast.walk(handler)
        if isinstance(stmt, ast.Import)
        and any(alias.name.startswith("mlx") for alias in stmt.names)
    ]
    assert inner_mlx_imports, (
        "expected ``import mlx.core`` inside the handler body for Template B dispatch"
    )
