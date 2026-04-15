"""EWS-0.3a: lazy-mlx gates for ``inference/context/adapters``.

Ensures that the adapter modules themselves (and their package ``__init__``)
can be imported under ``CHUK_BACKEND=torch`` without pulling ``mlx`` or
``libmlx.so`` — even though adapter method bodies still invoke MLX ops
when MLX is available.

The parent packages (``chuk_lazarus.inference`` / ``.inference.context``)
currently have their own mlx-hot eager imports owned by sibling
workstreams; those are not gated here.  This file scopes the assertion
to the adapter submodules by running each check in a fresh subprocess
that imports the adapter module directly via its dotted path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CHUK_BACKEND"] = "torch"
    env["PYTHONPATH"] = (
        str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


ADAPTER_MODULES = [
    "chuk_lazarus.inference.context.adapters.gemma_adapter",
    "chuk_lazarus.inference.context.adapters.llama_adapter",
]


@pytest.mark.parametrize("module", ADAPTER_MODULES)
def test_adapter_module_imports_without_mlx(module: str) -> None:
    """Importing the adapter module directly must not load mlx.*.

    We bypass the parent package ``__init__`` (which has sibling-owned
    mlx-hot eager imports) by loading the adapter file via its file path
    under a synthetic module name.  This isolates the adapter's own
    top-level imports — the only thing this gate is asserting.
    """
    rel = module.split(".")
    src_path = REPO_ROOT / "src" / Path(*rel).with_suffix(".py")
    assert src_path.is_file(), src_path

    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location({module!r}, {str(src_path)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "bad = [k for k in sys.modules if k == 'mlx' or k == 'mlx_lm' "
        "or k.startswith('mlx.') or k.startswith('mlx_lm.')]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    res = _run(code)
    assert res.returncode == 0, (
        f"{module}: unexpected mlx/mlx_lm load\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )


def test_adapters_package_lazy_reexport() -> None:
    """The ``adapters`` package ``__init__`` is lazy — importing the
    package alone (via file path to bypass siblings) must not touch
    mlx and must still expose the four adapter names via ``__all__``
    / ``__getattr__`` resolution once accessed.
    """
    pkg_init = (
        REPO_ROOT
        / "src/chuk_lazarus/inference/context/adapters/__init__.py"
    )
    code = (
        "import importlib.util, sys, types\n"
        # Seed placeholder parent packages so the relative imports used by
        # __getattr__ resolve without running their real __init__.
        "for name in (\n"
        "    'chuk_lazarus', 'chuk_lazarus.inference',\n"
        "    'chuk_lazarus.inference.context',\n"
        "):\n"
        "    if name not in sys.modules:\n"
        "        m = types.ModuleType(name)\n"
        "        m.__path__ = []  # mark as package\n"
        "        sys.modules[name] = m\n"
        "name = 'chuk_lazarus.inference.context.adapters'\n"
        f"spec = importlib.util.spec_from_file_location(name, {str(pkg_init)!r}, submodule_search_locations=[{str(pkg_init.parent)!r}])\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules[name] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "bad = [k for k in sys.modules if k == 'mlx' or k == 'mlx_lm' "
        "or k.startswith('mlx.') or k.startswith('mlx_lm.')]\n"
        "assert not bad, bad\n"
        "expected = {'GemmaBackboneAdapter', 'GemmaLayerAdapter', "
        "'LlamaBackboneAdapter', 'LlamaLayerAdapter'}\n"
        "assert expected.issubset(set(mod.__all__)), mod.__all__\n"
        "print('OK')\n"
    )
    res = _run(code)
    assert res.returncode == 0, (
        f"adapters package did not import lazily\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )
