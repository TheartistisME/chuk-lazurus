"""EWS-3 — backend flag contract for ``context generate`` + ``calibrate-frames``.

Uses AST-level assertions to stay independent of ``mlx`` native-library
availability (the Linux CUDA test host has ``mlx==0.31.1`` but ships no
``libmlx.so`` — a pre-existing carry-over outside EWS-3 scope).

Covers:
1. Both subparsers call the shared ``add_backend_flags`` helper.
2. Both command handlers raise ``NotImplementedError`` when ``--backend torch``
   is supplied (Template C per spec §3 + §5.3, §10).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
PARSER_PATH = REPO_ROOT / "src/chuk_lazarus/cli/_parsers/_context_generate.py"
GENERATE_CMD_PATH = REPO_ROOT / "src/chuk_lazarus/cli/commands/context/generate/_cmd.py"
CALIBRATE_CMD_PATH = REPO_ROOT / "src/chuk_lazarus/cli/commands/context/calibrate_frames.py"


def _ast(path: Path) -> ast.AST:
    return ast.parse(path.read_text())


def _count_calls(tree: ast.AST, func_name: str) -> int:
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == func_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == func_name)
        )
    )


def test_parser_imports_add_backend_flags():
    tree = _ast(PARSER_PATH)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "add_backend_flags":
                    found = True
    assert found, "_context_generate.py must import add_backend_flags"


def test_parser_calls_add_backend_flags_for_both_subparsers():
    tree = _ast(PARSER_PATH)
    # generate + calibrate-frames registrars each call once.
    assert _count_calls(tree, "add_backend_flags") >= 2


def test_generate_cmd_torch_arm_raises_nie():
    tree = _ast(GENERATE_CMD_PATH)
    # Find the handler and confirm it raises NotImplementedError somewhere.
    handler = next(
        n
        for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "context_generate_cmd"
    )
    raises = [
        n
        for n in ast.walk(handler)
        if isinstance(n, ast.Raise)
        and isinstance(n.exc, ast.Call)
        and isinstance(n.exc.func, ast.Name)
        and n.exc.func.id == "NotImplementedError"
    ]
    assert raises, "context_generate_cmd must raise NotImplementedError on torch arm"


def test_generate_cmd_consults_get_backend():
    tree = _ast(GENERATE_CMD_PATH)
    handler = next(
        n
        for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "context_generate_cmd"
    )
    assert _count_calls(handler, "get_backend") >= 1


def test_calibrate_cmd_torch_arm_raises_nie():
    tree = _ast(CALIBRATE_CMD_PATH)
    handler = next(
        n
        for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef)
        and n.name == "context_calibrate_frames_cmd"
    )
    raises = [
        n
        for n in ast.walk(handler)
        if isinstance(n, ast.Raise)
        and isinstance(n.exc, ast.Call)
        and isinstance(n.exc.func, ast.Name)
        and n.exc.func.id == "NotImplementedError"
    ]
    assert raises, (
        "context_calibrate_frames_cmd must raise NotImplementedError on torch arm"
    )


def test_calibrate_cmd_consults_get_backend():
    tree = _ast(CALIBRATE_CMD_PATH)
    handler = next(
        n
        for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef)
        and n.name == "context_calibrate_frames_cmd"
    )
    assert _count_calls(handler, "get_backend") >= 1
