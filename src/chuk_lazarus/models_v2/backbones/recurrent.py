"""
Recurrent backbone.

RNN-based architecture using LSTM, GRU, or MinGRU.

Top-level ``import mlx.*`` is intentionally absent: this module is listed
in ``tests/ci/test_no_top_level_mlx.py::BACKEND_IN_SCOPE``. The real
``RecurrentBackbone`` is built lazily via PEP 562 module ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

__all__ = ["RecurrentBackbone", "create_recurrent_backbone"]

_built: dict[str, object] = {}


def _build() -> dict[str, object]:
    if _built:
        return _built

    from typing import Any

    import mlx.core as mx
    import mlx.nn as nn

    from ..blocks.recurrent import RecurrentBlockWrapper, RecurrentWithFFN
    from ..components.embeddings import create_token_embedding
    from ..components.normalization import RMSNorm
    from .base import Backbone, BackboneOutput

    class RecurrentBackbone(Backbone):
        """Recurrent backbone (LSTM/GRU/MinGRU)."""

        def __init__(
            self,
            vocab_size: int,
            d_model: int,
            num_layers: int,
            rnn_type: str = "mingru",
            with_ffn: bool = True,
            intermediate_size: int | None = None,
            bidirectional: bool = False,
            norm_eps: float = 1e-5,
        ):
            super().__init__()

            self._vocab_size = vocab_size
            self._hidden_size = d_model
            self._num_layers = num_layers
            self.rnn_type = rnn_type
            self.bidirectional = bidirectional

            self.embed_tokens = create_token_embedding(
                vocab_size=vocab_size,
                hidden_size=d_model,
            )

            if with_ffn:
                self.layers = [
                    RecurrentWithFFN(
                        d_model=d_model,
                        rnn_type=rnn_type,
                        num_layers=1,
                        intermediate_size=intermediate_size,
                        norm_eps=norm_eps,
                    )
                    for _ in range(num_layers)
                ]
            else:
                self.layers = [
                    RecurrentBlockWrapper(
                        d_model=d_model,
                        rnn_type=rnn_type,
                        num_layers=1,
                        bidirectional=bidirectional,
                        norm_eps=norm_eps,
                    )
                    for _ in range(num_layers)
                ]

            self.norm = RMSNorm(d_model, eps=norm_eps)

        @property
        def hidden_size(self) -> int:
            return self._hidden_size

        @property
        def num_layers(self) -> int:
            return self._num_layers

        @property
        def vocab_size(self) -> int:
            return self._vocab_size

        def __call__(
            self,
            input_ids: mx.array,
            attention_mask: mx.array | None = None,
            cache: list[Any] | None = None,
            output_hidden_states: bool = False,
        ) -> BackboneOutput:
            hidden_states = self.embed_tokens(input_ids)

            all_hidden_states = (hidden_states,) if output_hidden_states else None
            new_cache = []

            for i, layer in enumerate(self.layers):
                layer_cache = cache[i] if cache else None
                output = layer(hidden_states, cache=layer_cache)
                hidden_states = output.hidden_states
                new_cache.append(output.cache)

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = self.norm(hidden_states)

            return BackboneOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                cache=new_cache,
            )

        def init_cache(
            self,
            batch_size: int,
            max_seq_len: int,
        ) -> list[Any]:
            return [layer.init_cache(batch_size, max_seq_len) for layer in self.layers]

        def get_input_embeddings(self) -> nn.Module:
            return self.embed_tokens

        def set_input_embeddings(self, embeddings: nn.Module) -> None:
            self.embed_tokens = embeddings

    def create_recurrent_backbone(
        vocab_size: int,
        d_model: int,
        num_layers: int,
        rnn_type: str = "mingru",
        with_ffn: bool = True,
    ) -> RecurrentBackbone:
        """Factory function for RecurrentBackbone."""
        return RecurrentBackbone(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            rnn_type=rnn_type,
            with_ffn=with_ffn,
        )

    _built["RecurrentBackbone"] = RecurrentBackbone
    _built["create_recurrent_backbone"] = create_recurrent_backbone
    return _built


def __getattr__(name: str):
    if name in ("RecurrentBackbone", "create_recurrent_backbone"):
        return _build()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
