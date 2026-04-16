"""
MLX-backed inference runtime.
"""

from __future__ import annotations

from typing import Any

from .base import InferenceRuntime
from .types import LazarusBackend, ResidualState


class MLXInferenceRuntime(InferenceRuntime):
    """Preserve the legacy MLX generation path behind the runtime interface."""

    @property
    def backend(self) -> LazarusBackend:
        return LazarusBackend.MLX

    def generate(self, prompt: str, config):
        from ..generation import generate

        return generate(self._model, self._tokenizer, prompt, config)

    def extract_residual_state(self, prompt: str, layer_index: int) -> ResidualState:
        import mlx.core as mx

        from ..context import make_kv_generator

        input_ids = self._tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        kv_generator = make_kv_generator(self._model)
        hidden = kv_generator.prefill_to_layer(
            input_ids,
            target_layer=layer_index,
            sample_positions=[input_ids.shape[1] - 1],
        )
        residual = mx.stop_gradient(hidden[:, 0, :])
        return ResidualState(
            backend=LazarusBackend.MLX,
            layer_index=layer_index,
            tensor=residual,
            sequence_length=int(input_ids.shape[1]),
            hidden_size=int(residual.shape[-1]),
            dtype=str(residual.dtype),
            device="mps",
        )

    def generate_with_residual(self, prompt: str, residual_state: ResidualState, config):
        raise NotImplementedError(
            "MLX residual injection continues to use the existing context injection modules. "
            "The runtime interface only wraps extraction for now."
        )

    def clear_cache(self) -> None:
        # MLX uses unified memory, so there is no explicit cache clearing path here.
        return None
