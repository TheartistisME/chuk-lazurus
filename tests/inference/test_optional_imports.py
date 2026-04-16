"""Regression tests for optional backend imports."""

from __future__ import annotations

import importlib
import sys


def _clear_modules(prefixes: tuple[str, ...]) -> None:
    for name in list(sys.modules):
        if name.startswith(prefixes):
            sys.modules.pop(name, None)


def test_context_package_import_is_lazy():
    _clear_modules(("chuk_lazarus.inference.context", "mlx"))

    module = importlib.import_module("chuk_lazarus.inference.context")

    assert module.__name__ == "chuk_lazarus.inference.context"
    assert "mlx" not in sys.modules
    assert "mlx.core" not in sys.modules


def test_inference_package_import_is_lazy():
    _clear_modules(("chuk_lazarus.inference", "mlx"))

    module = importlib.import_module("chuk_lazarus.inference")

    assert module.__name__ == "chuk_lazarus.inference"
    assert "mlx" not in sys.modules
    assert "mlx.core" not in sys.modules


def test_knowledge_package_import_is_lazy():
    _clear_modules(("chuk_lazarus.inference.context.knowledge", "mlx"))

    module = importlib.import_module("chuk_lazarus.inference.context.knowledge")

    assert module.__name__ == "chuk_lazarus.inference.context.knowledge"
    assert "mlx" not in sys.modules
    assert "mlx.core" not in sys.modules


def test_backends_package_import_is_lazy():
    _clear_modules(("chuk_lazarus.inference.backends", "mlx"))

    module = importlib.import_module("chuk_lazarus.inference.backends")

    assert module.__name__ == "chuk_lazarus.inference.backends"
    assert "mlx" not in sys.modules
    assert "mlx.core" not in sys.modules


def test_virtual_experts_package_import_is_lazy():
    _clear_modules(("chuk_lazarus.inference.virtual_experts", "chuk_virtual_expert"))

    module = importlib.import_module("chuk_lazarus.inference.virtual_experts")

    assert module.__name__ == "chuk_lazarus.inference.virtual_experts"
    assert "chuk_virtual_expert" not in sys.modules
