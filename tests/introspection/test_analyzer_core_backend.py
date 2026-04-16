"""EWS-5: ``introspection.analyzer.core`` must not top-level import MLX.

AST-based gate (transitive imports are covered by the per-bucket CI gate
``tests/ci/test_no_top_level_mlx.py``; this file focuses on the surface
owned by EWS-5).
"""

from __future__ import annotations

import ast
from pathlib import Path

TARGET = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "chuk_lazarus"
    / "introspection"
    / "analyzer"
    / "core.py"
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


def test_analyzer_core_has_no_toplevel_mlx_imports():
    assert _toplevel_mlx_imports(TARGET) == [], (
        "analyzer/core.py must defer MLX imports (use TYPE_CHECKING + lazy proxy)."
    )
