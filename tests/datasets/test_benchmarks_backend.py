"""EWS-11 backend tests for `chuk_lazarus.datasets.benchmarks`.

See `docs/refactor/dual-backend-cuda-all-buckets/03-workstreams.md` §EWS-11.

`benchmarks.py` is a pure-Python fixture provider today.  The backend
contract keeps it free of top-level mlx imports and guarantees content-hash
stable output across invocations so downstream MLX snapshots remain stable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src/chuk_lazarus/datasets/benchmarks.py"
MODULE = "chuk_lazarus.datasets.benchmarks"


def test_no_top_level_mlx_ast() -> None:
    tree = ast.parse(SRC_PATH.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            assert (mod or "").split(".")[0] not in {"mlx", "mlx_lm"}


def test_lazy_import_under_torch_env() -> None:
    env = os.environ.copy()
    env["CHUK_BACKEND"] = "torch"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys, importlib\n"
        f"importlib.import_module({MODULE!r})\n"
        "assert 'mlx' not in sys.modules and 'mlx_lm' not in sys.modules\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


def _hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def test_benchmark_fixtures_stable() -> None:
    """Repeated calls return identical content (snapshot-safe)."""

    from chuk_lazarus.datasets.benchmarks import (
        load_expert_benchmark,
        load_expert_test_categories,
    )

    a = load_expert_benchmark()
    b = load_expert_benchmark()
    assert _hash(a) == _hash(b)
    assert a is not b  # mutation-safe deep copy

    ca = load_expert_test_categories()
    cb = load_expert_test_categories()
    assert _hash(ca) == _hash(cb)
    assert ca is not cb
