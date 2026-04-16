"""
Lazarus inference server.

Provides an OpenAI-compatible (and extensible) HTTP API for local LLM inference.

Quick start::

    from chuk_lazarus.server.engine import ModelEngine
    from chuk_lazarus.server.app import create_app, Protocol

    engine = await ModelEngine.load("google/gemma-3-1b-it")
    app = create_app(engine, protocols=[Protocol.OPENAI])

Or via CLI::

    lazarus serve --model google/gemma-3-1b-it --port 8080
    lazarus-serve --model google/gemma-3-1b-it --port 8080
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .app import Protocol as Protocol
    from .app import create_app as create_app
    from .engine import ModelEngine as ModelEngine

__all__ = [
    "ModelEngine",
    "Protocol",
    "create_app",
]

_LAZY = {
    "ModelEngine": (".engine", "ModelEngine"),
    "Protocol": (".app", "Protocol"),
    "create_app": (".app", "create_app"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
