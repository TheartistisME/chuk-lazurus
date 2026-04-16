"""
LoRA (Low-Rank Adaptation) implementation.

LoRA adds trainable low-rank decomposition matrices to frozen weights:
    output = base_output + (x @ A @ B) * scaling

This allows efficient fine-tuning with minimal additional parameters.

Top-level ``import mlx.*`` is intentionally absent: this module is listed
in ``tests/ci/test_no_top_level_mlx.py::BACKEND_IN_SCOPE``.  All MLX usage
is deferred into the methods/functions that need it, and the real
``mlx.nn.Module``-backed ``LoRALinear`` is constructed lazily on first
instantiation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


_LoRALinearImpl = None


def _build_lora_linear_cls():
    """Build and cache the real ``mlx.nn.Module``-backed LoRALinear class."""
    global _LoRALinearImpl
    if _LoRALinearImpl is not None:
        return _LoRALinearImpl

    import mlx.core as mx
    import mlx.nn as nn

    class _LoRALinear(nn.Module):
        """
        LoRA adapter layer.

        Wraps a frozen linear layer and adds low-rank adaptation.
        """

        def __init__(
            self,
            base_layer,
            rank: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0,
        ):
            import mlx.core as mx
            import mlx.nn as nn  # noqa: F401

            super().__init__()
            self.base_layer = base_layer
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            self.dropout_rate = dropout

            in_features = base_layer.weight.shape[1]
            out_features = base_layer.weight.shape[0]

            # Initialize A with small random values, B with zeros
            # so that initial LoRA output is zero.
            self.lora_A = mx.random.normal((in_features, rank)) * 0.01
            self.lora_B = mx.zeros((rank, out_features))

            self.base_layer.freeze()
            self._training = False

        def __call__(self, x):
            import mlx.core as mx
            import mlx.nn as nn  # noqa: F401

            base_output = self.base_layer(x)

            lora_input = x
            if self.dropout_rate > 0 and self.training:
                mask = mx.random.bernoulli(1 - self.dropout_rate, x.shape)
                lora_input = x * mask / (1 - self.dropout_rate)

            lora_output = (lora_input @ self.lora_A @ self.lora_B) * self.scaling
            return base_output + lora_output

        def merge_weights(self):
            """Merge LoRA weights into base layer for inference."""
            import mlx.core as mx  # noqa: F401
            import mlx.nn as nn

            merged_weight = (
                self.base_layer.weight + (self.lora_B.T @ self.lora_A.T) * self.scaling
            )
            has_bias = hasattr(self.base_layer, "bias") and self.base_layer.bias is not None
            merged = nn.Linear(
                self.base_layer.weight.shape[1],
                self.base_layer.weight.shape[0],
                bias=has_bias,
            )
            merged.weight = merged_weight
            if has_bias:
                merged.bias = self.base_layer.bias
            return merged

        @property
        def training(self) -> bool:
            if not hasattr(self, "_training"):
                self._training = False
            return self._training

        @training.setter
        def training(self, value: bool):
            self._training = value

    _LoRALinearImpl = _LoRALinear
    return _LoRALinear


class _LoRALinearMeta(type):
    """Metaclass so that ``isinstance(x, LoRALinear)`` works against the lazy real class."""

    def __instancecheck__(cls, instance):
        real = _build_lora_linear_cls()
        return isinstance(instance, real)

    def __subclasscheck__(cls, subclass):
        real = _build_lora_linear_cls()
        return issubclass(subclass, real)


class LoRALinear(metaclass=_LoRALinearMeta):
    """
    Public façade for LoRALinear.  Instantiating this class produces an
    instance of the real ``mlx.nn.Module``-backed implementation.  No mlx
    import is performed until the class is actually used.

    Example:
        >>> base = nn.Linear(768, 768)
        >>> lora = LoRALinear(base, rank=8, alpha=16.0)
        >>> output = lora(x)
    """

    def __new__(cls, *args, **kwargs):
        real = _build_lora_linear_cls()
        return real(*args, **kwargs)


def apply_lora(model, config: LoRAConfig) -> dict[str, LoRALinear]:
    """
    Apply LoRA adapters to target modules in a model.
    """
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn

    lora_layers: dict[str, LoRALinear] = {}

    inner_model = model.model if hasattr(model, "model") else model

    layers = None
    if hasattr(inner_model, "layers"):
        layers = inner_model.layers
    elif hasattr(inner_model, "model") and hasattr(inner_model.model, "layers"):
        layers = inner_model.model.layers
    elif hasattr(inner_model, "backbone") and hasattr(inner_model.backbone, "layers"):
        layers = inner_model.backbone.layers

    if layers is None:
        logger.warning("Could not find transformer layers for LoRA")
        return lora_layers

    model.freeze()

    for layer_idx, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            for proj_name in config.target_modules:
                if proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if hasattr(attn, proj_name):
                        proj = getattr(attn, proj_name)
                        if isinstance(proj, nn.Linear):
                            lora = LoRALinear(
                                proj,
                                rank=config.rank,
                                alpha=config.alpha,
                                dropout=config.dropout,
                            )
                            setattr(attn, proj_name, lora)
                            lora_layers[f"layers.{layer_idx}.self_attn.{proj_name}"] = lora

        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            for proj_name in config.target_modules:
                if proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    if hasattr(mlp, proj_name):
                        proj = getattr(mlp, proj_name)
                        if isinstance(proj, nn.Linear):
                            lora = LoRALinear(
                                proj,
                                rank=config.rank,
                                alpha=config.alpha,
                                dropout=config.dropout,
                            )
                            setattr(mlp, proj_name, lora)
                            lora_layers[f"layers.{layer_idx}.mlp.{proj_name}"] = lora

    logger.info(f"Applied LoRA to {len(lora_layers)} layers")
    return lora_layers


def merge_lora_weights(model, lora_layers: dict[str, LoRALinear]):
    """
    Merge all LoRA weights into base model for efficient inference.
    """
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

    inner_model = model.model if hasattr(model, "model") else model
    layers = None

    if hasattr(inner_model, "layers"):
        layers = inner_model.layers
    elif hasattr(inner_model, "backbone") and hasattr(inner_model.backbone, "layers"):
        layers = inner_model.backbone.layers

    if layers is None:
        logger.warning("Could not find layers to merge")
        return

    for name, lora_layer in lora_layers.items():
        parts = name.split(".")
        layer_idx = int(parts[1])
        module_name = parts[2]
        proj_name = parts[3]

        layer = layers[layer_idx]
        module = getattr(layer, module_name)

        merged = lora_layer.merge_weights()
        setattr(module, proj_name, merged)

    logger.info(f"Merged {len(lora_layers)} LoRA layers")


def count_lora_parameters(lora_layers: dict[str, LoRALinear]) -> int:
    """Count total trainable LoRA parameters."""
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

    total = 0
    for layer in lora_layers.values():
        total += layer.lora_A.size + layer.lora_B.size
    return total
