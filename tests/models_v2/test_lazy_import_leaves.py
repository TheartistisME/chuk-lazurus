"""EWS-0.3c — regression tests for models_v2 lazy-mlx leaves.

These tests assert that under ``CHUK_BACKEND=torch`` the critical-path
public symbols reachable from ``chuk_lazarus`` and ``chuk_lazarus.models_v2``
can be resolved without importing ``mlx.core`` or ``mlx_lm`` at runtime.

The gate semantics mirror the `LoRALinear` façade pattern landed in
EWS-0.1: leaf classes expose a metaclass-backed façade that defers
``import mlx.*`` until actual instantiation / attribute dereferencing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = str(REPO_ROOT / "src")


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "CHUK_BACKEND": "torch",
        "PYTHONPATH": SRC_PATH,
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_llama_for_causal_lm_resolves_without_mlx_core() -> None:
    """The primary critical-path probe for EWS-0.3c."""
    code = (
        "import chuk_lazarus\n"
        "cls = chuk_lazarus.LlamaForCausalLM\n"
        "assert cls is not None\n"
        "import sys\n"
        "assert 'mlx.core' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    result = _run(code)
    assert result.returncode == 0, (
        f"probe failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


def test_models_v2_public_surfaces_resolve_without_mlx_core() -> None:
    """Second gate: Model/Block/compute_lm_loss resolve lazily."""
    code = (
        "import chuk_lazarus.models_v2 as m\n"
        "for n in ('LlamaForCausalLM', 'Model', 'Block', 'compute_lm_loss'):\n"
        "    v = getattr(m, n)\n"
        "    assert v is not None, n\n"
        "import sys\n"
        "assert 'mlx.core' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    result = _run(code)
    assert result.returncode == 0, (
        f"probe failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


@pytest.mark.parametrize(
    "package",
    [
        "chuk_lazarus.models_v2.blocks",
        "chuk_lazarus.models_v2.models",
        "chuk_lazarus.models_v2.heads",
        "chuk_lazarus.models_v2.components",
        "chuk_lazarus.models_v2.losses",
        "chuk_lazarus.models_v2.families",
        "chuk_lazarus.models_v2.families.llama",
    ],
)
def test_package_init_does_not_import_mlx_core(package: str) -> None:
    """Every package __init__ should be importable under torch without
    pulling mlx.core at module-load time."""
    code = (
        f"import importlib\n"
        f"importlib.import_module({package!r})\n"
        "import sys\n"
        "assert 'mlx.core' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    result = _run(code)
    assert result.returncode == 0, (
        f"{package}: {result.stdout}\n{result.stderr}"
    )
    assert "OK" in result.stdout
