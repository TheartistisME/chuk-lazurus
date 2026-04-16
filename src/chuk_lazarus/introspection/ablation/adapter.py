"""
Model adapter for accessing model internals across architectures.

This module provides a unified interface for accessing layers,
getting/setting component weights, and running generation across
different model architectures.
"""

from __future__ import annotations

from typing import Any

from chuk_lazarus.introspection._backend_dispatch import lazy_mx as mx, lazy_nn as nn  # EWS-6 lazy


class ModelAdapter:
    """
    Adapter for accessing model internals across different architectures.

    Provides a unified interface for:
    - Accessing layers
    - Getting/setting component weights
    - Running generation
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Any,
        runtime: Any | None = None,
        pipeline: Any | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.runtime = runtime
        self.pipeline = pipeline
        self._detect_architecture()

    def _detect_architecture(self):
        """Detect model architecture and set accessors."""
        candidates = [
            ("model", "language_model", "layers"),
            ("model", "language_model", "model", "layers"),
            ("language_model", "model", "layers"),
            ("language_model", "layers"),
            ("model", "layers"),
            ("transformer", "h"),
            ("layers",),
        ]

        for path in candidates:
            target = self.model
            for step in path[:-1]:
                target = getattr(target, step, None)
                if target is None:
                    break
            if target is None:
                continue

            layers = getattr(target, path[-1], None)
            if layers is None:
                continue
            self._layers = layers
            self._backbone = target
            return

        raise ValueError(
            "Cannot detect model architecture. Expected model.model.layers, "
            "model.language_model.model.layers, model.language_model.layers, "
            "model.layers, or model.transformer.h"
        )

    def _runtime_backend_name(self) -> str | None:
        backend = getattr(self.runtime, "backend", None)
        if backend is None:
            return None
        name = str(backend).lower()
        if name == "cuda":
            return "torch"
        if name == "mlx":
            return "mlx"
        return name

    @staticmethod
    def _is_torch_tensor(value: Any) -> bool:
        return type(value).__module__.startswith("torch")

    def clone_weight(self, weight: Any) -> Any:
        if self._is_torch_tensor(weight):
            return weight.detach().clone()
        return mx.array(weight)

    def zeros_like(self, weight: Any) -> Any:
        if self._is_torch_tensor(weight):
            import torch

            return torch.zeros_like(weight)
        return mx.zeros_like(weight)

    def _assign_weight(self, module: Any, attr: str, weight: Any) -> None:
        current = getattr(module, attr)
        if self._is_torch_tensor(current):
            import torch

            source = weight.detach() if isinstance(weight, torch.Tensor) else torch.as_tensor(weight)
            source = source.to(device=current.device, dtype=current.dtype, non_blocking=True)
            with torch.no_grad():
                target = current.data if hasattr(current, "data") else current
                target.copy_(source)
            return

        setattr(module, attr, weight)
        mx.eval(weight)

    @property
    def num_layers(self) -> int:
        """Number of transformer/SSM layers."""
        return len(self._layers)

    @property
    def hidden_size(self) -> int:
        """Hidden dimension size."""
        if hasattr(self.config, "hidden_size"):
            return self.config.hidden_size
        elif hasattr(self.config, "d_model"):
            return self.config.d_model
        raise ValueError("Cannot determine hidden size from config")

    def get_layer(self, idx: int) -> nn.Module:
        """Get layer by index."""
        return self._layers[idx]

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses Mixture of Experts."""
        layer = self.get_layer(layer_idx)
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            # Check for MoE patterns
            if hasattr(mlp, "router") or hasattr(mlp, "experts"):
                return True
            # Check class name
            if "MoE" in type(mlp).__name__:
                return True
        return False

    def get_mlp_down_weight(self, layer_idx: int) -> Any:
        """Get MLP down projection weight.

        For MoE layers, returns the router weight instead.
        """
        layer = self.get_layer(layer_idx)

        # Check for MoE first
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            # MoE: return router weight (zeroing this effectively disables MLP)
            if hasattr(mlp, "router"):
                if hasattr(mlp.router, "weight"):
                    return mlp.router.weight
            # Dense MLP patterns
            if hasattr(mlp, "down_proj"):
                return mlp.down_proj.weight
            elif hasattr(mlp, "c_proj"):  # GPT-2 style
                return mlp.c_proj.weight
            elif hasattr(mlp, "w2"):  # Some Llama variants
                return mlp.w2.weight
        elif hasattr(layer, "feed_forward"):
            ff = layer.feed_forward
            if hasattr(ff, "down_proj"):
                return ff.down_proj.weight
            elif hasattr(ff, "w2"):
                return ff.w2.weight

        raise ValueError(f"Cannot find MLP down projection in layer {layer_idx}")

    def set_mlp_down_weight(self, layer_idx: int, weight: Any):
        """Set MLP down projection weight.

        For MoE layers, sets the router weight instead.
        """
        layer = self.get_layer(layer_idx)

        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            # MoE: set router weight
            if hasattr(mlp, "router"):
                if hasattr(mlp.router, "weight"):
                    self._assign_weight(mlp.router, "weight", weight)
                    return
            # Dense MLP patterns
            if hasattr(mlp, "down_proj"):
                self._assign_weight(mlp.down_proj, "weight", weight)
            elif hasattr(mlp, "c_proj"):
                self._assign_weight(mlp.c_proj, "weight", weight)
            elif hasattr(mlp, "w2"):
                self._assign_weight(mlp.w2, "weight", weight)
            else:
                raise ValueError(f"Cannot find MLP down projection in layer {layer_idx}")
        elif hasattr(layer, "feed_forward"):
            ff = layer.feed_forward
            if hasattr(ff, "down_proj"):
                self._assign_weight(ff.down_proj, "weight", weight)
            elif hasattr(ff, "w2"):
                self._assign_weight(ff.w2, "weight", weight)
            else:
                raise ValueError(f"Cannot find MLP down projection in layer {layer_idx}")
        else:
            raise ValueError(f"Cannot find MLP in layer {layer_idx}")

    def get_attn_o_weight(self, layer_idx: int) -> Any:
        """Get attention output projection weight."""
        layer = self.get_layer(layer_idx)

        # Try common patterns
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            if hasattr(attn, "o_proj"):
                return attn.o_proj.weight
            elif hasattr(attn, "out_proj"):
                return attn.out_proj.weight
        elif hasattr(layer, "attention"):
            attn = layer.attention
            if hasattr(attn, "o_proj"):
                return attn.o_proj.weight
            elif hasattr(attn, "wo"):  # Llama style
                return attn.wo.weight

        raise ValueError(f"Cannot find attention output projection in layer {layer_idx}")

    def set_attn_o_weight(self, layer_idx: int, weight: Any):
        """Set attention output projection weight."""
        layer = self.get_layer(layer_idx)

        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            if hasattr(attn, "o_proj"):
                self._assign_weight(attn.o_proj, "weight", weight)
            elif hasattr(attn, "out_proj"):
                self._assign_weight(attn.out_proj, "weight", weight)
            else:
                raise ValueError(f"Cannot find attention output projection in layer {layer_idx}")
        elif hasattr(layer, "attention"):
            attn = layer.attention
            if hasattr(attn, "o_proj"):
                self._assign_weight(attn.o_proj, "weight", weight)
            elif hasattr(attn, "wo"):
                self._assign_weight(attn.wo, "weight", weight)
            else:
                raise ValueError(f"Cannot find attention output projection in layer {layer_idx}")
        else:
            raise ValueError(f"Cannot find attention in layer {layer_idx}")

    def generate(
        self,
        prompt_or_input_ids: Any,
        max_new_tokens: int = 60,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from a prompt or pre-tokenized MLX input IDs."""
        if isinstance(prompt_or_input_ids, str):
            prompt = prompt_or_input_ids
            if self.runtime is not None:
                from ...inference import GenerationConfig

                result = self.runtime.generate(
                    prompt,
                    GenerationConfig(
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    ),
                )
                return result.text

            input_ids = mx.array(self.tokenizer.encode(prompt, return_tensors="np"))
        else:
            input_ids = prompt_or_input_ids
            if self._runtime_backend_name() == "torch":
                raise TypeError("Torch-backed ablation generation requires a prompt string.")

        # Use model's generate method if available
        if hasattr(self.model, "generate"):
            stop_tokens = []
            if self.tokenizer.eos_token_id is not None:
                stop_tokens = [self.tokenizer.eos_token_id]

            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_tokens=stop_tokens,
            )
            output_ids = generated[0, input_ids.shape[1] :].tolist()
            return self.tokenizer.decode(output_ids, skip_special_tokens=False)

        # Fallback: manual generation
        return self._manual_generate(input_ids, max_new_tokens, temperature)

    def _manual_generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Manual autoregressive generation."""
        generated = input_ids

        for _ in range(max_new_tokens):
            output = self.model(generated)

            # Handle different output types
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Get last token logits
            next_logits = logits[0, -1, :]

            # Sample
            if temperature == 0:
                next_token = mx.argmax(next_logits, axis=-1, keepdims=True)
            else:
                probs = mx.softmax(next_logits / temperature)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))
                next_token = mx.expand_dims(next_token, axis=0)

            generated = mx.concatenate([generated, next_token[None, :]], axis=1)

            # Check for EOS
            if self.tokenizer.eos_token_id and int(next_token[0]) == self.tokenizer.eos_token_id:
                break

        output_ids = generated[0, input_ids.shape[1] :].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=False)
