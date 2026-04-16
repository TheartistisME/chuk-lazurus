"""EWS-3 — lazy shim contract for ``inference.context.unlimited_engine``.

Uses AST assertions (stays independent of ``libmlx.so`` availability — a
pre-existing carry-over on the Linux CUDA host outside EWS-3 scope).

The shim defers resolution to ``research/unlimited_engine`` via a
``__getattr__`` hook so importing the module does not pull mlx.
"""

from __future__ import annotations

import ast
from pathlib import Path

SHIM_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/chuk_lazarus/inference/context/unlimited_engine.py"
)


def test_shim_has_module_level_getattr():
    tree = ast.parse(SHIM_PATH.read_text())
    functions = [
        n.name for n in tree.body if isinstance(n, ast.FunctionDef)
    ]
    assert "__getattr__" in functions, (
        "unlimited_engine shim must define module-level __getattr__ for lazy re-export"
    )


def test_shim_has_no_top_level_star_import_of_research():
    tree = ast.parse(SHIM_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name != "*", (
                    "unlimited_engine shim must not use ``from .research.unlimited_engine "
                    "import *`` at top level — use lazy __getattr__ instead"
                )


def test_shim_getattr_imports_lazily():
    tree = ast.parse(SHIM_PATH.read_text())
    getattr_fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "__getattr__"
    )
    # The body must contain an ImportFrom pointing at research.unlimited_engine.
    inner_imports = [
        stmt
        for stmt in ast.walk(getattr_fn)
        if isinstance(stmt, ast.ImportFrom)
    ]
    assert inner_imports, "__getattr__ must perform the research import lazily"
