"""Lazy-import proxy for ``mlx`` modules used by inference."""

from __future__ import annotations

import importlib
from typing import Any


class _LazyMod:
    __slots__ = ("_name", "_mod")

    def __init__(self, name: str) -> None:
        self._name = name
        self._mod: Any = None

    def __getattr__(self, attr: str) -> Any:
        if self._mod is None:
            self._mod = importlib.import_module(self._name)
        return getattr(self._mod, attr)


mx = _LazyMod("mlx.core")
nn = _LazyMod("mlx.nn")

__all__ = ["mx", "nn"]
