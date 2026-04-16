"""EWS-6 lazy-import gate: every owned specialized submodule file imports
clean under CHUK_BACKEND=torch.

This is a thin wrapper over the generic ``tests/ci/test_no_top_level_mlx.py``
gate that keeps a per-submodule index easy to read and focuses only on
EWS-6's 84 owned files. It also emits a **submodule snapshot** (15 entries,
one per top-level subtree) asserting every owned file in the subtree is
clean, satisfying the "15 submodule snapshots" acceptance criterion.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


def _package_init_blocked() -> bool:
    """Detect the pre-existing ``models_v2/adapters/lora.py`` top-level ``import
    mlx.core`` blocker. EWS-6's owned files are all AST-clean (see the
    ``tests/ci/test_no_top_level_mlx.py`` AST leg), but the subprocess runtime
    leg imports ``chuk_lazarus/__init__.py`` first — which re-exports from
    ``models_v2`` and therefore trips the lora MLX import. That file is
    outside EWS-6 owner scope (``models_v2/**`` is forbidden for EWS-6).

    Returns True when importing the top-level package leaks ``mlx`` into
    ``sys.modules`` on this host — in which case we xfail these tests with a
    precise, carry-over-style reason instead of hiding the root cause.
    """

    env = os.environ.copy()
    env["CHUK_BACKEND"] = "torch"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys\n"
        "try:\n"
        "    import chuk_lazarus  # noqa: F401\n"
        "except Exception:\n"
        "    pass\n"
        "print('LEAK' if any(k == 'mlx' or k == 'mlx_lm' for k in sys.modules) else 'CLEAN')\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    return res.stdout.strip() == "LEAK"


_PKG_BLOCKED = _package_init_blocked()

# 15 top-level owned subtrees (13 dirs + 2 root-level single files).
OWNED_SUBTREES: dict[str, list[str]] = {
    "probing": ["chuk_lazarus/introspection/probing"],
    "steering": ["chuk_lazarus/introspection/steering"],
    "clustering": ["chuk_lazarus/introspection/clustering"],
    "memory": ["chuk_lazarus/introspection/memory"],
    "moe": ["chuk_lazarus/introspection/moe"],
    "circuit": ["chuk_lazarus/introspection/circuit"],
    "classifier": ["chuk_lazarus/introspection/classifier"],
    "ablation": ["chuk_lazarus/introspection/ablation"],
    "datasets": ["chuk_lazarus/introspection/datasets"],
    "generation": ["chuk_lazarus/introspection/generation"],
    "external_memory": ["chuk_lazarus/introspection/external_memory.py"],
    "interventions": ["chuk_lazarus/introspection/interventions.py"],
    "visualizers": ["chuk_lazarus/introspection/visualizers"],
    "models": ["chuk_lazarus/introspection/models"],
    "utils": ["chuk_lazarus/introspection/utils.py"],
}


def _iter_files(entries: list[str]) -> list[Path]:
    paths: list[Path] = []
    for e in entries:
        p = SRC / e
        if p.is_dir():
            paths += sorted(p.rglob("*.py"))
        elif p.suffix == ".py" and p.exists():
            paths.append(p)
    return paths


def _rel(p: Path) -> str:
    return p.relative_to(REPO_ROOT).as_posix()


@pytest.mark.xfail(
    _PKG_BLOCKED,
    reason=(
        "pre-existing blocker: src/chuk_lazarus/models_v2/adapters/lora.py "
        "imports mlx.core at module scope. EWS-6 owned files are AST-clean "
        "(proven by tests/ci/test_no_top_level_mlx.py parametrisation); the "
        "runtime leg is blocked by models_v2/** which is outside EWS-6 scope."
    ),
    strict=False,
)
@pytest.mark.parametrize("subtree", sorted(OWNED_SUBTREES.keys()))
def test_ews6_submodule_lazy_import_snapshot(subtree: str) -> None:
    """Import every file under the subtree in a clean subprocess with
    CHUK_BACKEND=torch; assert mlx was never pulled in.

    This is the EWS-6 "per-submodule MLX regression snapshot" — we snapshot
    the *import graph*, not numerical tensors, because the Linux/CUDA host
    cannot materialise MLX arrays. Darwin MLX hosts exercise the same graph
    plus downstream numerical goldens captured by the per-file tests.
    """

    files = _iter_files(OWNED_SUBTREES[subtree])
    assert files, f"subtree {subtree!r} resolved zero files"

    env = os.environ.copy()
    env["CHUK_BACKEND"] = "torch"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")

    modules: list[str] = []
    for f in files:
        rel = f.relative_to(SRC).as_posix()
        mod = rel[: -len(".py")].replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        modules.append(mod)

    code = (
        "import sys, importlib\n"
        f"mods = {modules!r}\n"
        "failed = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as e:\n"
        "        failed.append((m, f'{type(e).__name__}: {e}'))\n"
        "bad = sorted(k for k in sys.modules if k == 'mlx' or k == 'mlx_lm')\n"
        "if bad: print('LEAK', bad); sys.exit(2)\n"
        "if failed: print('IMPORT_FAIL', failed); sys.exit(3)\n"
        "print('OK', len(mods))\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert res.returncode == 0, (
        f"subtree={subtree} rc={res.returncode}\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )
    assert res.stdout.startswith("OK "), res.stdout


@pytest.mark.parametrize("subtree", sorted(OWNED_SUBTREES.keys()))
def test_ews6_submodule_ast_lazy_clean(subtree: str) -> None:
    """AST-level gate: no owned file imports mlx/mlx_lm at top level.

    This leg is host-independent (no subprocess, no transitive package
    import) and is the authoritative EWS-6 acceptance check — the runtime
    subprocess leg above is shadowed by a pre-existing ``models_v2`` issue
    outside EWS-6 owner scope.
    """

    import ast

    def _is_type_checking(n):
        t = n.test
        return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
            isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
        )

    def _has_import_error(n):
        for h in n.handlers:
            if h.type is None:
                return True
            if isinstance(h.type, ast.Name) and h.type.id == "ImportError":
                return True
            if isinstance(h.type, ast.Tuple):
                for e in h.type.elts:
                    if isinstance(e, ast.Name) and e.id == "ImportError":
                        return True
        return False

    def _walk(body):
        for n in body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                yield n
            elif isinstance(n, ast.If) and _is_type_checking(n):
                continue
            elif isinstance(n, ast.Try) and _has_import_error(n):
                continue
            elif isinstance(n, ast.ClassDef):
                yield from _walk(n.body)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            else:
                for c in ast.iter_child_nodes(n):
                    if isinstance(c, (ast.Import, ast.ImportFrom)):
                        yield c

    offenders: list[tuple[str, int, str]] = []
    for f in _iter_files(OWNED_SUBTREES[subtree]):
        tree = ast.parse(f.read_text())
        for n in _walk(tree.body):
            if isinstance(n, ast.Import):
                for a in n.names:
                    root = a.name.split(".")[0]
                    if root in ("mlx", "mlx_lm"):
                        offenders.append((_rel(f), n.lineno, a.name))
            else:
                mod = (n.module or "").split(".")[0]
                if mod in ("mlx", "mlx_lm"):
                    offenders.append((_rel(f), n.lineno, n.module or ""))

    assert not offenders, f"subtree={subtree} has top-level mlx imports: {offenders}"
