"""Public CLI package surface.

Keep package import cheap and backend-neutral.  Accessing ``app`` or ``main``
resolves the real CLI entrypoint lazily.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .main import app as app
    from .main import main as main

__all__ = ["app", "main"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(".main", __name__)
        globals().update({export: getattr(module, export) for export in __all__})
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
