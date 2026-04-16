"""EWS-5: ``introspection.virtual_expert`` must not top-level import MLX.

AST-based gate (virtual_expert.py re-exports from ``chuk_lazarus.inference``
which pulls in subtrees owned by EWS-1b; the subprocess-import test is
not meaningful here until that WS lands).
"""

from __future__ import annotations

import ast
from pathlib import Path

TARGET = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "chuk_lazarus"
    / "introspection"
    / "virtual_expert.py"
)


def _toplevel_mlx_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    hits: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "mlx" or alias.name.startswith("mlx."):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "mlx" or node.module.startswith("mlx.")):
                hits.append(node.module)
    return hits


def test_virtual_expert_has_no_toplevel_mlx_imports():
    assert _toplevel_mlx_imports(TARGET) == [], (
        "virtual_expert.py must defer MLX imports (use TYPE_CHECKING + lazy proxy)."
    )
