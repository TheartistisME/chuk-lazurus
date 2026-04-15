"""
Base block abstractions.

Top-level ``import mlx.*`` is intentionally absent: this module is listed
in ``tests/ci/test_no_top_level_mlx.py::BACKEND_IN_SCOPE``. The real
``mlx.nn.Module``-backed abstractions are built lazily inside
``_build()`` and surfaced via PEP 562 ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

__all__ = ["Block", "BlockOutput", "SequenceModule"]

_built: dict[str, type] = {}


def _build() -> dict[str, type]:
    if _built:
        return _built

    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from typing import Any

    import mlx.core as mx
    import mlx.nn as nn

    from ..core.enums import BlockType

    @dataclass
    class BlockOutput:
        """
        Output from a block forward pass.
        """

        hidden_states: mx.array
        cache: Any | None = None
        aux_loss: mx.array | None = None

    class Block(nn.Module, ABC):
        """
        Abstract base class for all blocks.
        """

        @property
        @abstractmethod
        def block_type(self) -> BlockType:
            """Return the type of this block."""

        @property
        @abstractmethod
        def hidden_size(self) -> int:
            """Return the hidden dimension."""

        @abstractmethod
        def __call__(
            self,
            x: mx.array,
            mask: mx.array | None = None,
            cache: Any | None = None,
        ) -> "BlockOutput":
            """Forward pass through the block."""

        def init_cache(self, batch_size: int, max_seq_len: int) -> Any:
            return None

    class SequenceModule(nn.Module, ABC):
        """
        Abstract base for sequence modeling modules.
        """

        @abstractmethod
        def __call__(
            self,
            x: mx.array,
            mask: mx.array | None = None,
            cache: Any | None = None,
        ) -> tuple[mx.array, Any | None]:
            """Process sequence."""

    _built["Block"] = Block
    _built["BlockOutput"] = BlockOutput
    _built["SequenceModule"] = SequenceModule
    return _built


def _make_facade(name: str) -> type:
    """Façade class whose instantiation triggers the real class build."""

    class _Meta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, _build()[name])

        def __subclasscheck__(cls, subclass):
            return issubclass(subclass, _build()[name])

        def __getattr__(cls, attr):
            return getattr(_build()[name], attr)

    class _Facade(metaclass=_Meta):
        __qualname__ = name

        def __new__(cls, *args, **kwargs):
            return _build()[name](*args, **kwargs)

    _Facade.__name__ = name
    return _Facade


_facades: dict[str, type] = {}


def __getattr__(name: str):
    if name in ("Block", "BlockOutput", "SequenceModule"):
        if name not in _facades:
            _facades[name] = _make_facade(name)
        return _facades[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
