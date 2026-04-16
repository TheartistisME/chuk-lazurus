"""
Legacy steering code for FunctionGemma.

This module contains backwards-compatible code for FunctionGemma
experiments. For new code, use ActivationSteering instead.

EWS-6 dual-backend note: ``SteeredGemmaMLP`` subclasses ``mlx.nn.Module``.
Defining it at module scope would force MLX on every import. It is built
lazily via module-level ``__getattr__`` (PEP 562). ``ToolCallingSteering``
is MLX-only at runtime but its class body no longer touches MLX symbols,
so the module imports clean under ``CHUK_BACKEND=torch``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from chuk_lazarus.introspection._backend_dispatch import lazy_mx as mx  # EWS-6 lazy

from .config import LegacySteeringConfig, SteeringMode

if TYPE_CHECKING:  # pragma: no cover
    import mlx.nn as nn  # noqa: F401


def _build_SteeredGemmaMLP():
    import mlx.core as _mx  # noqa: F401
    import mlx.nn as _nn

    class SteeredGemmaMLP(_nn.Module):
        """Gemma MLP wrapper that applies steering during forward pass."""

        def __init__(
            self,
            original_mlp,
            config: LegacySteeringConfig,
            layer_idx: int,
            control_layer: int = 11,
            gate_layer: int = 12,
            kill_switch_neuron: int = 230,
        ):
            super().__init__()
            self.original_mlp = original_mlp
            self.config = config
            self.layer_idx = layer_idx
            self.control_layer = control_layer
            self.gate_layer = gate_layer
            self.kill_switch_neuron = kill_switch_neuron

        def __call__(self, x):
            gate = self.original_mlp.gate_proj(x)
            up = self.original_mlp.up_proj(x)
            mlp_hidden = _nn.gelu_approx(gate) * up

            if self.layer_idx == self.control_layer:
                mlp_hidden = self._apply_control_steering(mlp_hidden)
            if self.layer_idx == self.gate_layer:
                mlp_hidden = self._apply_gate_steering(mlp_hidden)

            return self.original_mlp.down_proj(mlp_hidden)

        def _apply_control_steering(self, mlp_hidden):
            if self.config.mode == SteeringMode.NORMAL:
                return mlp_hidden

            batch_size, seq_len, hidden_size = mlp_hidden.shape
            modification = _mx.zeros((hidden_size,))

            scale = self.config.neuron_boost_scale
            if self.config.mode in [SteeringMode.BOOST_TOOL, SteeringMode.SUPPRESS_TOOL]:
                scale *= 0.3

            if self.config.mode in [SteeringMode.FORCE_TOOL, SteeringMode.BOOST_TOOL]:
                for neuron in self.config.tool_promoters:
                    if neuron < hidden_size:
                        modification = modification.at[neuron].add(scale)
                for neuron in self.config.tool_suppressors:
                    if neuron < hidden_size:
                        modification = modification.at[neuron].add(-scale * 0.5)
            else:
                for neuron in self.config.tool_promoters:
                    if neuron < hidden_size:
                        modification = modification.at[neuron].add(-scale)
                for neuron in self.config.tool_suppressors:
                    if neuron < hidden_size:
                        modification = modification.at[neuron].add(scale * 0.5)

            position_mask = _mx.zeros((seq_len,))
            position_mask = position_mask.at[-1].add(1.0)
            modification = modification.reshape(1, 1, hidden_size)
            position_mask = position_mask.reshape(1, seq_len, 1)

            return mlp_hidden + modification * position_mask

        def _apply_gate_steering(self, mlp_hidden):
            batch_size, seq_len, hidden_size = mlp_hidden.shape

            if self.config.use_kill_switch:
                mask = _mx.ones((hidden_size,))
                mask = mask.at[self.kill_switch_neuron].add(-1.0)
                position_mask = _mx.zeros((seq_len,))
                position_mask = position_mask.at[-1].add(1.0)
                mask_broadcast = mask.reshape(1, 1, hidden_size)
                position_broadcast = position_mask.reshape(1, seq_len, 1)
                mlp_hidden = mlp_hidden * (1 - position_broadcast + position_broadcast * mask_broadcast)

            if self.config.kill_switch_boost != 0:
                modification = _mx.zeros((hidden_size,))
                modification = modification.at[self.kill_switch_neuron].add(
                    self.config.kill_switch_boost
                )
                position_mask = _mx.zeros((seq_len,))
                position_mask = position_mask.at[-1].add(1.0)
                modification = modification.reshape(1, 1, hidden_size)
                position_mask = position_mask.reshape(1, seq_len, 1)
                mlp_hidden = mlp_hidden + modification * position_mask

            return mlp_hidden

    return SteeredGemmaMLP


class ToolCallingSteering:
    """Tool-calling specific steering (for FunctionGemma).

    Kept for backwards compatibility. For new code, use ActivationSteering.
    """

    CONTROL_LAYER = 11
    GATE_LAYER = 12
    KILL_SWITCH_NEURON = 230
    TOOL_PROMOTERS = [803, 2036, 831, 436, 969]
    TOOL_SUPPRESSORS = [1347, 1237, 821, 217, 543]

    def __init__(self, model: Any, tokenizer: Any, config: Any = None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = config
        self._original_mlps: dict[int, Any] = {}

    @classmethod
    def from_pretrained(cls, model_id: str) -> ToolCallingSteering:
        from ..ablation import AblationStudy

        study = AblationStudy.from_pretrained(model_id)
        return cls(study.adapter.model, study.adapter.tokenizer, study.adapter.config)

    def _install_steering(self, config: LegacySteeringConfig):
        SteeredGemmaMLP = _build_SteeredGemmaMLP()
        layers = self.model.model.layers
        for layer_idx in [self.CONTROL_LAYER, self.GATE_LAYER]:
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                self._original_mlps[layer_idx] = layer.mlp
                layer.mlp = SteeredGemmaMLP(
                    layer.mlp,
                    config,
                    layer_idx,
                    self.CONTROL_LAYER,
                    self.GATE_LAYER,
                    self.KILL_SWITCH_NEURON,
                )

    def _uninstall_steering(self):
        layers = self.model.model.layers
        for layer_idx, original_mlp in self._original_mlps.items():
            if layer_idx < len(layers):
                layers[layer_idx].mlp = original_mlp
        self._original_mlps.clear()

    def generate(
        self,
        prompt: str,
        mode: str = "normal",
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        config = LegacySteeringConfig(mode=SteeringMode(mode), **kwargs)
        tokens = self.tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()
        input_ids = mx.array([tokens])

        self._install_steering(config)
        try:
            stop_tokens = [self.tokenizer.eos_token_id]
            end_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if end_turn_id != self.tokenizer.unk_token_id:
                stop_tokens.append(end_turn_id)

            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_tokens=stop_tokens,
            )
            new_tokens = generated[0, len(tokens):].tolist()
            return self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        finally:
            self._uninstall_steering()

    def predict(self, prompt: str, mode: str = "normal", **kwargs) -> dict:
        config = LegacySteeringConfig(mode=SteeringMode(mode), **kwargs)
        tokens = self.tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()
        input_ids = mx.array([tokens])

        self._install_steering(config)
        try:
            output = self.model(input_ids)
            logits = output.logits[0, -1, :]
            probs = mx.softmax(logits, axis=-1)
            top_indices = mx.argsort(probs)[-5:][::-1].tolist()

            results = []
            for idx in top_indices:
                prob = float(probs[idx])
                try:
                    token = self.tokenizer.decode([idx])
                except Exception:
                    token = f"[{idx}]"
                results.append((token, prob))

            top_token = results[0][0]
            tool_indicators = ["[", "{", "<", "function", "tool", "call", "Function"]
            tool_likely = any(ind in top_token for ind in tool_indicators)

            return {
                "prompt": prompt,
                "mode": mode,
                "top_tokens": results,
                "tool_likely": tool_likely,
            }
        finally:
            self._uninstall_steering()

    def compare_modes(self, prompt: str) -> dict:
        results = {}
        for mode in SteeringMode:
            results[mode.value] = self.predict(prompt, mode=mode.value)
        return results


def __getattr__(name: str):
    if name == "SteeredGemmaMLP":
        cls = _build_SteeredGemmaMLP()
        globals()["SteeredGemmaMLP"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
