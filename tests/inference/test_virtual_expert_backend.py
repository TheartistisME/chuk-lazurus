"""Focused backend/lazy-import coverage for ``inference.virtual_experts``."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VIRTUAL_EXPERTS_ROOT = REPO_ROOT / "src" / "chuk_lazarus" / "inference" / "virtual_experts"

OWNED_MODULES = [
    "chuk_lazarus.inference.virtual_experts",
    "chuk_lazarus.inference.virtual_experts.base",
    "chuk_lazarus.inference.virtual_experts.cot_rewriter",
    "chuk_lazarus.inference.virtual_experts.dense_wrapper",
    "chuk_lazarus.inference.virtual_experts.plugins.math",
    "chuk_lazarus.inference.virtual_experts.registry",
    "chuk_lazarus.inference.virtual_experts.router",
    "chuk_lazarus.inference.virtual_experts.wrapper",
]

OWNED_FILES = [
    VIRTUAL_EXPERTS_ROOT / "__init__.py",
    VIRTUAL_EXPERTS_ROOT / "_optional.py",
    VIRTUAL_EXPERTS_ROOT / "base.py",
    VIRTUAL_EXPERTS_ROOT / "cot_rewriter.py",
    VIRTUAL_EXPERTS_ROOT / "dense_wrapper.py",
    VIRTUAL_EXPERTS_ROOT / "plugins" / "math.py",
    VIRTUAL_EXPERTS_ROOT / "registry.py",
    VIRTUAL_EXPERTS_ROOT / "router.py",
    VIRTUAL_EXPERTS_ROOT / "wrapper.py",
]


def _isolated_import_code(body: str) -> str:
    return textwrap.dedent(
        f"""
        import importlib
        import importlib.abc
        import pathlib
        import sys
        import types

        repo_root = pathlib.Path({str(REPO_ROOT)!r})
        src_root = repo_root / "src"
        chuk_root = src_root / "chuk_lazarus"
        inference_root = chuk_root / "inference"

        sys.path.insert(0, str(src_root))

        def _package(name, path):
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            mod.__package__ = name
            return mod

        sys.modules["chuk_lazarus"] = _package("chuk_lazarus", chuk_root)
        sys.modules["chuk_lazarus.inference"] = _package(
            "chuk_lazarus.inference", inference_root
        )

        class _Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                blocked = {{
                    "chuk_virtual_expert",
                    "chuk_virtual_expert_time",
                    "chuk_virtual_expert_arithmetic",
                    "mlx",
                    "mlx.core",
                    "mlx.nn",
                    "mlx_lm",
                }}
                if fullname in blocked or fullname.startswith("mlx."):
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, _Blocker())

        {body}
        """
    )


def _run_isolated(body: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CHUK_BACKEND"] = "torch"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", _isolated_import_code(body)],
        env=env,
        capture_output=True,
        text=True,
    )


def _assert_no_toplevel_imports(path: Path) -> None:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".", 1)[0]
            assert root not in {"mlx", "chuk_virtual_expert"}, (
                f"{path.relative_to(REPO_ROOT)} has top-level import {name!r}"
            )


@pytest.mark.parametrize("path", OWNED_FILES)
def test_virtual_experts_files_have_no_toplevel_mlx_or_optional_dep(path: Path) -> None:
    _assert_no_toplevel_imports(path)


@pytest.mark.parametrize("module_name", OWNED_MODULES)
def test_virtual_experts_modules_import_cleanly_on_torch_hosts(module_name: str) -> None:
    res = _run_isolated(
        f"""
        import importlib
        import sys

        mod = importlib.import_module({module_name!r})
        assert mod.__name__ == {module_name!r}
        bad = sorted(
            name for name in sys.modules
            if name == "mlx" or name.startswith("mlx.") or name == "mlx_lm"
        )
        assert not bad, bad
        """
    )

    assert res.returncode == 0, (
        f"{module_name} is not lazy-import clean on torch hosts\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )


def test_virtual_experts_package_reexports_stay_lazy_without_optional_dep() -> None:
    res = _run_isolated(
        """
        import importlib
        import sys

        pkg = importlib.import_module("chuk_lazarus.inference.virtual_experts")
        assert pkg.VirtualExpertAction.none_action("x").expert == "none"
        assert pkg.MathExpertPlugin.__name__ == "MathExpert"
        assert pkg.VirtualMoEWrapper.__name__ == "VirtualMoEWrapper"
        assert pkg.VirtualDenseWrapper.__name__ == "VirtualDenseWrapper"
        assert "chuk_virtual_expert" not in sys.modules
        bad = sorted(
            name for name in sys.modules
            if name == "mlx" or name.startswith("mlx.") or name == "mlx_lm"
        )
        assert not bad, bad
        """
    )

    assert res.returncode == 0, (
        "virtual_experts package re-exports are not lazy/optional-safe\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )


def test_optional_dependency_fallback_supports_registry_and_math_plugin() -> None:
    res = _run_isolated(
        """
        import asyncio
        import importlib

        base = importlib.import_module("chuk_lazarus.inference.virtual_experts.base")
        registry_mod = importlib.import_module("chuk_lazarus.inference.virtual_experts.registry")

        assert base.CHUK_VIRTUAL_EXPERT_AVAILABLE is False

        registry = registry_mod.get_default_registry()
        plugin = registry.get("math")
        assert plugin is not None

        action = base.VirtualExpertAction(
            expert="math",
            operation="evaluate",
            parameters={"expression": "2 + 2"},
        )
        result = asyncio.run(plugin.execute(action))
        assert result.success is True
        assert result.data["result"] == 4
        """
    )

    assert res.returncode == 0, (
        "optional chuk_virtual_expert fallback is not working cleanly\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )
