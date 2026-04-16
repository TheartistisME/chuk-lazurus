"""Lazy-import proxy for mlx.core / mlx.nn / mlx.optimizers (EWS-10).

The per-trainer modules reference ``mx``/``nn``/``optim`` at module scope
for readability, but must stay lazy-import-clean under ``CHUK_BACKEND=torch``
(see ``tests/ci/test_no_top_level_mlx.py``).  This module exports proxy
objects whose first attribute access triggers the real import.
"""

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
optim = _LazyMod("mlx.optimizers")

__all__ = ["mx", "nn", "optim"]
