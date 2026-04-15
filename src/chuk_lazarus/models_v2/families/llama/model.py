"""
Llama model implementation.

Uses the composable architecture from models_v2.

Top-level ``import mlx.*`` is intentionally absent: this module is listed
in ``tests/ci/test_no_top_level_mlx.py::BACKEND_IN_SCOPE``. The real
``mlx.nn.Module``-backed classes (``LlamaBlock``, ``LlamaModel``,
``LlamaForCausalLM``) are constructed lazily on first instantiation via
the ``LoRALinear``-style façade pattern; the public symbols themselves
resolve via PEP 562 module ``__getattr__`` without importing ``mlx``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

__all__ = ["LlamaBlock", "LlamaModel", "LlamaForCausalLM"]

_built: dict[str, type] = {}


def _build() -> dict[str, type]:
    """Build the real ``mlx.nn.Module``-backed Llama classes. Cached."""
    if _built:
        return _built

    from typing import Any

    import mlx.core as mx
    import mlx.nn as nn

    from ...backbones.base import Backbone, BackboneOutput
    from ...blocks.base import Block, BlockOutput
    from ...components.attention import GroupedQueryAttention, SlidingWindowAttention
    from ...components.embeddings import create_token_embedding
    from ...components.ffn import SwiGLU
    from ...components.normalization import RMSNorm
    from ...core.config import FFNConfig
    from ...core.registry import register_model
    from ...heads import LMHead
    from ...models.base import Model, ModelOutput
    from ..constants import HFArchitecture, HFModelType
    from .config import LlamaConfig

    class LlamaBlock(Block):
        """
        Llama transformer block.

        Standard pre-norm transformer with:
        - RMSNorm
        - GQA or sliding window attention
        - SwiGLU FFN
        """

        def __init__(
            self,
            config: LlamaConfig,
            layer_idx: int = 0,
        ):
            super().__init__()

            self._hidden_size = config.hidden_size
            self.layer_idx = layer_idx

            self.input_layernorm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )

            attn_config = config.to_attention_config()

            if config.sliding_window:
                self.self_attn = SlidingWindowAttention(attn_config)
            else:
                self.self_attn = GroupedQueryAttention(attn_config)

            self.post_attention_layernorm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )

            ffn_config = FFNConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )
            self.mlp = SwiGLU(ffn_config)

        @property
        def block_type(self):
            from ...core.enums import BlockType

            return BlockType.TRANSFORMER

        @property
        def hidden_size(self) -> int:
            return self._hidden_size

        def __call__(
            self,
            x: mx.array,
            mask: mx.array | None = None,
            cache: tuple[mx.array, mx.array] | None = None,
        ) -> BlockOutput:
            residual = x
            x = self.input_layernorm(x)
            x, new_cache = self.self_attn(x, mask=mask, cache=cache)
            x = residual + x

            residual = x
            x = self.post_attention_layernorm(x)
            x = self.mlp(x)
            x = residual + x

            return BlockOutput(hidden_states=x, cache=new_cache)

    class LlamaModel(Backbone):
        """
        Llama backbone (without LM head).
        """

        def __init__(self, config: LlamaConfig):
            super().__init__()

            self.config = config
            self._vocab_size = config.vocab_size
            self._hidden_size = config.hidden_size
            self._num_layers = config.num_hidden_layers

            self.embed_tokens = create_token_embedding(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
            )

            self.layers = [
                LlamaBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)
            ]

            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
                output = layer(hidden_states, mask=mask, cache=layer_cache)
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

        def get_input_embeddings(self) -> nn.Module:
            return self.embed_tokens

        def set_input_embeddings(self, embeddings: nn.Module) -> None:
            self.embed_tokens = embeddings

    @register_model(
        model_type=HFModelType.LLAMA,
        architectures=[
            HFArchitecture.LLAMA_FOR_CAUSAL_LM,
            HFArchitecture.MISTRAL_FOR_CAUSAL_LM,
        ],
    )
    class LlamaForCausalLM(Model):
        """
        Llama for causal language modeling.
        """

        def __init__(self, config: LlamaConfig):
            super().__init__()

            self._config = config

            self.model = LlamaModel(config)

            if config.tie_word_embeddings:
                self.lm_head = LMHead(
                    hidden_size=config.hidden_size,
                    vocab_size=config.vocab_size,
                    tied_embeddings=self.model.embed_tokens,
                )
            else:
                self.lm_head = LMHead(
                    hidden_size=config.hidden_size,
                    vocab_size=config.vocab_size,
                )

        @property
        def config(self) -> LlamaConfig:
            return self._config

        @property
        def backbone(self) -> nn.Module:
            return self.model

        def __call__(
            self,
            input_ids: mx.array,
            attention_mask: mx.array | None = None,
            labels: mx.array | None = None,
            cache: list[Any] | None = None,
            output_hidden_states: bool = False,
        ) -> ModelOutput:
            backbone_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache=cache,
                output_hidden_states=output_hidden_states,
            )

            head_output = self.lm_head(
                hidden_states=backbone_output.last_hidden_state,
                labels=labels,
            )

            return ModelOutput(
                loss=head_output.loss,
                logits=head_output.logits,
                hidden_states=backbone_output.hidden_states,
                cache=backbone_output.cache,
            )

        def generate(
            self,
            input_ids: mx.array,
            max_new_tokens: int = 100,
            temperature: float = 1.0,
            top_k: int | None = None,
            top_p: float | None = None,
            repetition_penalty: float = 1.0,
            stop_tokens: list[int] | None = None,
        ) -> mx.array:
            """Generate text autoregressively."""
            stop_tokens_set = set(stop_tokens or [])

            output = self(input_ids)
            mx.eval(output.logits)
            cache = output.cache

            generated_tokens = [input_ids]

            for _ in range(max_new_tokens):
                logits = output.logits[:, -1, :]

                if repetition_penalty != 1.0:
                    all_tokens = mx.concatenate(generated_tokens, axis=1)
                    unique_tokens = set(all_tokens.flatten().tolist())
                    vocab_size = logits.shape[-1]
                    token_indices = mx.array(
                        [t for t in unique_tokens if t < vocab_size]
                    )
                    if token_indices.size > 0:
                        mask = mx.zeros((vocab_size,))
                        for tok in token_indices.tolist():
                            mask = mask.at[tok].add(1.0)
                        penalty_mask = mx.where(mask > 0, repetition_penalty, 1.0)
                        logits = logits / penalty_mask

                if temperature == 0.0:
                    next_token = mx.argmax(logits, axis=-1)
                    next_token = mx.expand_dims(next_token, axis=-1)
                else:
                    if temperature != 1.0:
                        logits = logits / temperature

                    if top_k is not None and top_k > 0:
                        top_k_values = mx.topk(logits, k=min(top_k, logits.shape[-1]))
                        min_val = top_k_values[:, -1:]
                        logits = mx.where(logits < min_val, float("-inf"), logits)

                    probs = mx.softmax(logits, axis=-1)
                    next_token = mx.random.categorical(mx.log(probs + 1e-10))
                    next_token = mx.expand_dims(next_token, axis=-1)

                mx.eval(next_token)

                generated_tokens.append(next_token)

                next_token_val = int(next_token[0, 0])
                if next_token_val in stop_tokens_set:
                    break

                output = self(next_token, cache=cache)
                mx.eval(output.logits)
                cache = output.cache

            return mx.concatenate(generated_tokens, axis=1)

        @classmethod
        def from_config(cls, config: LlamaConfig):
            return cls(config)

        @classmethod
        async def from_pretrained_async(
            cls,
            model_path: str,
            config: LlamaConfig | None = None,
        ):
            """Load pretrained model."""
            import json
            from pathlib import Path

            path = Path(model_path)

            if config is None:
                config_path = path / "config.json"
                with open(config_path) as f:
                    config_data = json.load(f)
                config = LlamaConfig(**config_data)

            model = cls(config)

            from mlx.utils import tree_unflatten

            from .convert import convert_hf_weights

            weights_path = path / "model.safetensors"
            if weights_path.exists():
                raw_weights = mx.load(str(weights_path))
                weights = convert_hf_weights(raw_weights)
                nested_weights = tree_unflatten(list(weights.items()))
                model.update(nested_weights)
            else:
                index_path = path / "model.safetensors.index.json"
                if index_path.exists():
                    import json

                    with open(index_path) as f:
                        index = json.load(f)
                    shard_files = set(index.get("weight_map", {}).values())
                    raw_weights = {}
                    for shard_file in sorted(shard_files):
                        shard_weights = mx.load(str(path / shard_file))
                        raw_weights.update(shard_weights)
                    weights = convert_hf_weights(raw_weights)
                    nested_weights = tree_unflatten(list(weights.items()))
                    model.update(nested_weights)

            return model

    _built["LlamaBlock"] = LlamaBlock
    _built["LlamaModel"] = LlamaModel
    _built["LlamaForCausalLM"] = LlamaForCausalLM
    return _built


def _make_facade(name: str) -> type:
    """Build a façade class whose instantiation triggers the real
    ``mlx.nn.Module``-backed implementation. Isinstance/subclass checks
    also route to the real class via the metaclass.
    """

    class _Meta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, _build()[name])

        def __subclasscheck__(cls, subclass):
            return issubclass(subclass, _build()[name])

        def __getattr__(cls, attr):  # e.g. classmethods like from_config
            return getattr(_build()[name], attr)

    class _Facade(metaclass=_Meta):
        __qualname__ = name

        def __new__(cls, *args, **kwargs):
            real = _build()[name]
            return real(*args, **kwargs)

    _Facade.__name__ = name
    return _Facade


_facades: dict[str, type] = {}


def __getattr__(attr: str):
    if attr in ("LlamaBlock", "LlamaModel", "LlamaForCausalLM"):
        if attr not in _facades:
            _facades[attr] = _make_facade(attr)
        return _facades[attr]
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")
