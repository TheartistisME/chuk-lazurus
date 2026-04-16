"""EWS-11 backend tests for `train datagen` command.

See `docs/refactor/dual-backend-cuda-all-buckets/03-workstreams.md` §EWS-11.

The train datagen pipeline is pure-Python (no tensor ops), so the backend
contract reduces to:

* the module must import cleanly under ``CHUK_BACKEND=torch`` without pulling
  ``mlx`` / ``mlx_lm`` at module import;
* the command handler must accept a Namespace whether or not ``--backend`` /
  ``--device`` are present (the EWS-0 root parser injects them into the env);
* dataset generation must be deterministic across repeated runs so MLX
  regression fixtures stay snapshot-stable (content-hash comparison).
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE = "chuk_lazarus.cli.commands.train.datagen"


def test_no_top_level_mlx_ast() -> None:
    src = (REPO_ROOT / "src/chuk_lazarus/cli/commands/train/datagen.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            root = (mod or "").split(".")[0]
            assert root not in {"mlx", "mlx_lm"}


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


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_generate_data_cmd_accepts_backend_flags(backend: str, tmp_path: Path) -> None:
    """`generate_data_cmd` must tolerate Namespace carrying backend/device."""

    from chuk_lazarus.cli.commands.train.datagen import generate_data_cmd

    args = Namespace(
        type="math",
        output=str(tmp_path / "out"),
        sft_samples=4,
        dpo_samples=2,
        seed=7,
        backend=backend,
        device=None,
    )

    with patch(
        "chuk_lazarus.data.generators.generate_lazarus_dataset",
    ) as mock_gen:
        asyncio.run(generate_data_cmd(args))

    mock_gen.assert_called_once()


def _hash_dir(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_generate_data_deterministic(tmp_path: Path) -> None:
    """Two runs with the same seed must hash-identically (snapshot stability)."""

    pytest.importorskip("mlx.core", reason="chuk_lazarus.data.__init__ imports mlx (EWS-12)")
    from chuk_lazarus.cli.commands.train._types import DataGenConfig, DataGenType
    from chuk_lazarus.cli.commands.train.datagen import generate_data

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    for out in (out_a, out_b):
        cfg = DataGenConfig(
            type=DataGenType.MATH,
            output=out,
            sft_samples=8,
            dpo_samples=4,
            seed=123,
        )
        asyncio.run(generate_data(cfg))

    assert _hash_dir(out_a) == _hash_dir(out_b)


def test_module_importable_directly() -> None:
    # Sanity: module is importable in the running interpreter without raising.
    importlib.import_module(MODULE)
