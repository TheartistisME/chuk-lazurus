"""
Hybrid backbone.

Combines different block types (attention, SSM, RNN) in a single backbone.

Top-level ``import mlx.*`` is intentionally absent: this module is listed
in ``tests/ci/test_no_top_level_mlx.py::BACKEND_IN_SCOPE``. The real
``HybridBackbone`` is built lazily via PEP 562 module ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

__all__ = ["HybridBackbone", "create_hybrid_backbone"]

_built: dict[str, object] = {}


def _build() -> dict[str, object]:
    if _built:
        return _built

    from typing import Any

    import mlx.core as mx
    import mlx.nn as nn

    from ..blocks import HybridBlock, MambaBlockWrapper, TransformerBlock
    from ..blocks.hybrid import AlternatingHybrid
    from ..components.embeddings import create_token_embedding
    from ..components.normalization import RMSNorm
    from ..core.enums import BlockType, HybridCombineMode, HybridMixStrategy
    from .base import Backbone, BackboneOutput

    class HybridBackbone(Backbone):
        """Hybrid backbone combining different block types."""

        def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            num_layers: int,
            num_heads: int,
            num_kv_heads: int | None = None,
            d_state: int = 16,
            d_conv: int = 4,
            intermediate_size: int | None = None,
            mix_strategy: HybridMixStrategy | str = HybridMixStrategy.ALTERNATING,
            attention_ratio: float = 0.5,
            norm_eps: float = 1e-5,
        ):
            super().__init__()

            self._vocab_size = vocab_size
            self._hidden_size = hidden_size
            self._num_layers = num_layers
            self.mix_strategy = (
                HybridMixStrategy(mix_strategy)
                if isinstance(mix_strategy, str)
                else mix_strategy
            )

            num_kv_heads = num_kv_heads or num_heads
            intermediate_size = intermediate_size or hidden_size * 4

            self.embed_tokens = create_token_embedding(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
            )

            self.layers = []
            self.layer_types = []

            for i in range(num_layers):
                if self.mix_strategy == HybridMixStrategy.ALTERNATING:
                    block = AlternatingHybrid(
                        d_model=hidden_size,
                        layer_idx=i,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        d_state=d_state,
                        d_conv=d_conv,
                        intermediate_size=intermediate_size,
                        use_attention_first=True,
                        norm_eps=norm_eps,
                    )
                    self.layer_types.append(
                        BlockType.TRANSFORMER if i % 2 == 0 else BlockType.MAMBA
                    )

                elif self.mix_strategy == HybridMixStrategy.INTERLEAVED:
                    attention_interval = (
                        int(1 / attention_ratio) if attention_ratio > 0 else num_layers + 1
                    )
                    use_attention = i % attention_interval == 0

                    if use_attention:
                        block = TransformerBlock(
                            hidden_size=hidden_size,
                            num_heads=num_heads,
                            num_kv_heads=num_kv_heads,
                            intermediate_size=intermediate_size,
                            norm_eps=norm_eps,
                        )
                        self.layer_types.append(BlockType.TRANSFORMER)
                    else:
                        block = MambaBlockWrapper(
                            d_model=hidden_size,
                            d_state=d_state,
                            d_conv=d_conv,
                            norm_eps=norm_eps,
                        )
                        self.layer_types.append(BlockType.MAMBA)

                elif self.mix_strategy == HybridMixStrategy.PARALLEL:
                    block = HybridBlock(
                        d_model=hidden_size,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        d_state=d_state,
                        d_conv=d_conv,
                        intermediate_size=intermediate_size,
                        combine_mode=HybridCombineMode.ADD,
                        norm_eps=norm_eps,
                    )
                    self.layer_types.append(BlockType.HYBRID)

                else:
                    raise ValueError(f"Unknown mix_strategy: {self.mix_strategy}")

                self.layers.append(block)

            self.norm = RMSNorm(hidden_size, eps=norm_eps)

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
            batch_size, seq_len = input_ids.shape

            hidden_states = self.embed_tokens(input_ids)

            if attention_mask is None:
                mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
                mask = mask.astype(hidden_states.dtype)
            else:
                mask = attention_mask

            all_hidden_states = (hidden_states,) if output_hidden_states else None
            new_cache = []

            for i, layer in enumerate(self.layers):
                layer_cache = cache[i] if cache else None
                layer_type = self.layer_types[i]

                if layer_type in (BlockType.TRANSFORMER, BlockType.HYBRID):
                    output = layer(hidden_states, mask=mask, cache=layer_cache)
                else:
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

    def create_hybrid_backbone(
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mix_strategy: "HybridMixStrategy | str" = "alternating",
    ) -> HybridBackbone:
        """Factory function for HybridBackbone."""
        return HybridBackbone(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mix_strategy=mix_strategy,
        )

    _built["HybridBackbone"] = HybridBackbone
    _built["create_hybrid_backbone"] = create_hybrid_backbone
    return _built


def __getattr__(name: str):
    if name in ("HybridBackbone", "create_hybrid_backbone"):
        return _build()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
