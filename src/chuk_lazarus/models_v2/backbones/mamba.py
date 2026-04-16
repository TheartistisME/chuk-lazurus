"""
Mamba backbone.

Pure Mamba architecture with SSM blocks instead of attention.

Top-level ``import mlx.*`` is intentionally absent: this module is listed
in ``tests/ci/test_no_top_level_mlx.py::BACKEND_IN_SCOPE``. The real
``MambaBackbone`` (subclass of ``Backbone``/``mlx.nn.Module``) is built
lazily via PEP 562 module ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

    from ..core.config import ModelConfig  # noqa: F401

__all__ = ["MambaBackbone", "create_mamba_backbone"]

_built: dict[str, object] = {}


def _build() -> dict[str, object]:
    if _built:
        return _built

    from typing import Any

    import mlx.core as mx
    import mlx.nn as nn

    from ..blocks import MambaBlockWrapper
    from ..components.embeddings import create_token_embedding
    from ..components.normalization import RMSNorm
    from ..core.config import ModelConfig, SSMConfig
    from .base import Backbone, BackboneOutput

    class MambaBackbone(Backbone):
        """Mamba backbone (pure SSM)."""

        def __init__(
            self,
            vocab_size: int,
            d_model: int,
            num_layers: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            norm_eps: float = 1e-5,
        ):
            super().__init__()

            self._vocab_size = vocab_size
            self._hidden_size = d_model
            self._num_layers = num_layers
            self.d_state = d_state
            self.d_conv = d_conv
            self.expand = expand

            self.embed_tokens = create_token_embedding(
                vocab_size=vocab_size,
                hidden_size=d_model,
            )

            self.layers = [
                MambaBlockWrapper(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
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
        ) -> list[tuple[mx.array, mx.array]]:
            return [layer.init_cache(batch_size, max_seq_len) for layer in self.layers]

        def get_input_embeddings(self) -> nn.Module:
            return self.embed_tokens

        def set_input_embeddings(self, embeddings: nn.Module) -> None:
            self.embed_tokens = embeddings

        @classmethod
        def from_config(cls, config: ModelConfig) -> "MambaBackbone":
            ssm_config = config.ssm_config or SSMConfig()
            return cls(
                vocab_size=config.vocab_size,
                d_model=config.hidden_size,
                num_layers=config.num_hidden_layers,
                d_state=ssm_config.state_size,
                d_conv=ssm_config.conv_kernel,
                expand=ssm_config.expand,
                norm_eps=config.rms_norm_eps,
            )

    def create_mamba_backbone(
        vocab_size: int,
        d_model: int,
        num_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> MambaBackbone:
        """Factory function for MambaBackbone."""
        return MambaBackbone(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    _built["MambaBackbone"] = MambaBackbone
    _built["create_mamba_backbone"] = create_mamba_backbone
    return _built


def __getattr__(name: str):
    if name in ("MambaBackbone", "create_mamba_backbone"):
        return _build()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
