"""Global pytest helpers for optional backend availability."""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path

import pytest


_MLX_IMPORT_PATTERNS = (
    "import mlx.core",
    "import mlx.nn",
    "from mlx import",
    "from mlx.core",
    "from mlx.nn",
)
_VIRTUAL_EXPERT_IMPORT_PATTERNS = (
    "import chuk_virtual_expert",
    "from chuk_virtual_expert",
)
_MLX_ONLY_TEST_DIRS = (
    "/tests/inference/context/",
    "/tests/inference/virtual_experts/",
)
_VIRTUAL_EXPERT_TEST_DIRS = ("/tests/inference/virtual_experts/",)


@lru_cache(maxsize=None)
def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def pytest_ignore_collect(collection_path: Path, config) -> bool:  # noqa: ARG001
    if collection_path.suffix != ".py":
        return False

    path_text = collection_path.as_posix()
    file_text = _read_text(collection_path)

    if not _module_available("mlx.core"):
        if any(test_dir in path_text for test_dir in _MLX_ONLY_TEST_DIRS):
            return True
        if any(pattern in file_text for pattern in _MLX_IMPORT_PATTERNS):
            return True

    if not _module_available("chuk_virtual_expert"):
        if any(test_dir in path_text for test_dir in _VIRTUAL_EXPERT_TEST_DIRS):
            return True
        if any(pattern in file_text for pattern in _VIRTUAL_EXPERT_IMPORT_PATTERNS):
            return True

    return False


@pytest.fixture(params=["mlx", "torch"])
def backend_env(request, monkeypatch):
    """Parametrized fixture that selects a backend via ``CHUK_BACKEND``.

    Skips the parametrization entry when the underlying framework is not
    importable so tests can opt into "run on whichever backend is available".
    Consumers should pull the backend instance via ``get_backend_or_skip``
    from :mod:`tests._helpers.backend_fixtures`.
    """

    name = request.param
    if name == "mlx" and not _module_available("mlx.core"):
        pytest.skip("mlx not importable on this host")
    if name == "torch" and not _module_available("torch"):
        pytest.skip("torch not importable on this host")
    monkeypatch.setenv("CHUK_BACKEND", name)
    return name


@pytest.fixture
def mlx_golden(tmp_path):
    """Provide an isolated directory for MLX regression snapshots per test.

    The shared helpers in ``tests/_helpers/mlx_snapshots.py`` use the repo's
    canonical snapshot directory; this fixture is for tests that want to
    capture + diff a snapshot in isolation without touching the shared cache.
    """

    root = tmp_path / "mlx_golden"
    root.mkdir()
    return root


@pytest.fixture(autouse=True)
def skip_mlx_gated_cases(request):
    if _module_available("mlx.core"):
        return

    path_text = str(getattr(request.node, "fspath", ""))
    test_name = request.node.name.lower()
    if path_text.endswith("tests/cli/test_main.py") and "introspect" in test_name:
        pytest.skip("Introspection CLI parser tests require MLX-backed introspection modules.")
