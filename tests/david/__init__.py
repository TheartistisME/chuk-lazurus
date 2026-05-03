from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def require_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == name or missing.startswith(f"{name}."):
            pytest.fail(
                f"Expected importable module {name!r} for the production david "
                "terminal agent surface.",
                pytrace=False,
            )
        raise


def require_attr(module: Any, name: str, purpose: str) -> Any:
    assert hasattr(module, name), (
        f"{module.__name__} must expose {name!r} for {purpose}."
    )
    return getattr(module, name)


def value_at(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def assert_path_field(config: Any, field_name: str, expected: Path) -> None:
    value = value_at(config, field_name)
    assert value is not None, f"runtime config missing {field_name!r}"
    assert Path(value).resolve() == expected.resolve()
