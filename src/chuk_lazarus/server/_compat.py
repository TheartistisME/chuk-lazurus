"""Server-only optional dependency helpers."""

from __future__ import annotations

from typing import NoReturn

SERVER_DEPENDENCY_MESSAGE = (
    "Server dependencies not installed.\n"
    "Install with:  pip install 'chuk-lazarus[server]'\n"
    "           or:  uv add 'chuk-lazarus[server]'"
)


def raise_server_dependency_error(exc: BaseException | None = None) -> NoReturn:
    """Raise a consistent import error for optional server extras."""

    if exc is None:
        raise ImportError(SERVER_DEPENDENCY_MESSAGE)
    raise ImportError(SERVER_DEPENDENCY_MESSAGE) from exc


class MissingOptionalDependencyProxy:
    """Import-safe placeholder for FastAPI-only exports."""

    def __init__(self, symbol_name: str) -> None:
        self._symbol_name = symbol_name

    def _fail(self) -> NoReturn:
        raise_server_dependency_error()

    def __call__(self, *args, **kwargs):
        self._fail()

    def __getattr__(self, name: str):
        self._fail()

    def __repr__(self) -> str:
        return f"<missing optional dependency proxy for {self._symbol_name}>"
