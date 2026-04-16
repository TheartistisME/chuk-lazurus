"""
Base backbone abstractions.

Defines the common interface for all backbone types.

Top-level ``import mlx.*`` is intentionally absent: this module is listed
in ``tests/ci/test_no_top_level_mlx.py::BACKEND_IN_SCOPE``.  All MLX usage
is deferred via PEP 562 module ``__getattr__``; ``Backbone`` (an
``mlx.nn.Module`` subclass) and ``BackboneOutput`` (a dataclass annotated
with ``mx.array``) are constructed lazily on first attribute access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

__all__ = ["Backbone", "BackboneOutput"]

_built: dict[str, type] = {}


def _build() -> dict[str, type]:
    """Build the real ``mlx.nn.Module``-backed ``Backbone`` and
    ``BackboneOutput`` dataclass. Cached after first call."""
    if _built:
        return _built

    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from typing import Any

    import mlx.core as mx
    import mlx.nn as nn

    @dataclass
    class BackboneOutput:
        """
        Output from a backbone forward pass.

        Attributes:
            last_hidden_state: Final hidden states, shape (batch, seq_len, hidden_size)
            hidden_states: Optional tuple of all layer hidden states
            cache: Optional cache for inference
            aux_loss: Optional auxiliary loss (e.g., from MoE)
        """

        last_hidden_state: mx.array
        hidden_states: tuple[mx.array, ...] | None = None
        cache: list[Any] | None = None
        aux_loss: mx.array | None = None

    class Backbone(nn.Module, ABC):
        """
        Abstract base class for all backbones.

        A backbone is the core of a model, consisting of:
        - Input embeddings (token + position)
        - A stack of blocks (attention, SSM, RNN, etc.)
        - Final normalization

        It produces hidden states that can be processed by different heads.
        """

        @property
        @abstractmethod
        def hidden_size(self) -> int:
            """Return the hidden dimension."""

        @property
        @abstractmethod
        def num_layers(self) -> int:
            """Return the number of layers/blocks."""

        @property
        @abstractmethod
        def vocab_size(self) -> int:
            """Return vocabulary size."""

        @abstractmethod
        def __call__(
            self,
            input_ids: mx.array,
            attention_mask: mx.array | None = None,
            cache: list[Any] | None = None,
            output_hidden_states: bool = False,
        ) -> "BackboneOutput":
            """Forward pass through the backbone."""

        def init_cache(
            self,
            batch_size: int,
            max_seq_len: int,
        ) -> list[Any]:
            """Initialize cache for all layers."""
            return [None] * self.num_layers

        @property
        def embedding_scale(self) -> float | None:
            """Return embedding scale factor if this model uses one.

            Some models (e.g., Gemma) scale embeddings by sqrt(hidden_size).
            Override in subclasses that need embedding scaling.
            """
            return None

        def get_input_embeddings(self) -> nn.Module:
            """Return the input embedding layer."""
            raise NotImplementedError

        def set_input_embeddings(self, embeddings: nn.Module) -> None:
            """Set the input embedding layer."""
            raise NotImplementedError

    _built["Backbone"] = Backbone
    _built["BackboneOutput"] = BackboneOutput
    return _built


def __getattr__(name: str):
    if name in ("Backbone", "BackboneOutput"):
        return _build()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
