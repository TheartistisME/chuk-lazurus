"""EWS-11 backend tests for `data batching generate` command.

See `docs/refactor/dual-backend-cuda-all-buckets/03-workstreams.md` §EWS-11.

`data_batch_generate` tokenises and writes NPZ batches via numpy; it does not
import mlx/torch tensor libs at module load.  These tests pin that contract
plus Namespace compatibility for the EWS-0 `--backend`/`--device` flags.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
SRC_PATH = REPO_ROOT / "src/chuk_lazarus/cli/commands/data/batching/generate.py"
MODULE = "chuk_lazarus.cli.commands.data.batching.generate"


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


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_generate_config_from_args_with_backend(tmp_path: Path, backend: str) -> None:
    from chuk_lazarus.cli.commands.data.batching._types import GenerateConfig

    args = Namespace(
        plan=str(tmp_path / "plan"),
        dataset=str(tmp_path / "ds.jsonl"),
        tokenizer="dummy",
        output=str(tmp_path / "out"),
        backend=backend,
        device=None,
    )
    cfg = GenerateConfig.from_args(args)
    assert cfg.output == Path(tmp_path / "out")
