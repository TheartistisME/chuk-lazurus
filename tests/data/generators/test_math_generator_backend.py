"""EWS-11 backend tests for `chuk_lazarus.data.generators`.

See `docs/refactor/dual-backend-cuda-all-buckets/03-workstreams.md` §EWS-11.

The generators are pure-Python (random + JSON).  Tests assert the backend
contract: no mlx import at load, deterministic output for a fixed seed so
MLX regression snapshots stay hash-stable across hosts.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "rel",
    [
        "src/chuk_lazarus/data/generators/__init__.py",
        "src/chuk_lazarus/data/generators/math_generator.py",
        "src/chuk_lazarus/data/generators/types.py",
    ],
)
def test_no_top_level_mlx_ast(rel: str) -> None:
    tree = ast.parse((REPO_ROOT / rel).read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            assert (mod or "").split(".")[0] not in {"mlx", "mlx_lm"}


# NOTE: runtime lazy-import assertion for `chuk_lazarus.data.generators.*` is
# deferred: importing any submodule of `chuk_lazarus.data` also executes the
# package's `__init__.py`, which eagerly imports `base_dataset` (mlx-hot).
# That cleanup belongs to EWS-12.  The AST gate above is sufficient to pin
# the EWS-11 contract — no new top-level mlx import is introduced by the
# generator sources themselves.


def _hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def test_math_generator_deterministic() -> None:
    pytest.importorskip("mlx.core", reason="chuk_lazarus.data.__init__ imports mlx (EWS-12)")
    from chuk_lazarus.data.generators import MathProblemGenerator

    g1 = MathProblemGenerator(seed=42)
    g2 = MathProblemGenerator(seed=42)
    b1 = g1.generate_batch(16)
    b2 = g2.generate_batch(16)

    d1 = [s.model_dump() for s in b1]
    d2 = [s.model_dump() for s in b2]
    assert _hash(d1) == _hash(d2)


def test_generate_lazarus_dataset_deterministic(tmp_path: Path) -> None:
    """Dataset hashes must match between two runs of the same seed."""

    pytest.importorskip("mlx.core", reason="chuk_lazarus.data.__init__ imports mlx (EWS-12)")
    from chuk_lazarus.data.generators import generate_lazarus_dataset

    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_lazarus_dataset(str(a), sft_samples=6, dpo_samples=3, seed=99)
    generate_lazarus_dataset(str(b), sft_samples=6, dpo_samples=3, seed=99)

    def dir_hash(root: Path) -> str:
        h = hashlib.sha256()
        for p in sorted(root.rglob("*")):
            if p.is_file():
                h.update(p.name.encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    assert dir_hash(a) == dir_hash(b)
