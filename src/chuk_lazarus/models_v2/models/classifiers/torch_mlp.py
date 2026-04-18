"""
Torch-native MLP classifier.

Two-stage feed-forward classifier mirroring the MLX ``MLPClassifier`` twin:

    Linear(input_size -> hidden_size)
      -> activation
      -> Linear(hidden_size -> input_size)
      -> Linear(input_size -> num_labels)

The inner ``input -> hidden -> input`` block matches the chuk-mlx MLP
component used by the MLX twin. This is a torch-native implementation against
:mod:`torch.nn`; it is NOT an MLX port. No device placement is performed
inside the module -- the caller owns ``.to(device)``. Default dtype is
``torch.float32``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from chuk_lazarus.models_v2.core.enums import ActivationType

_ACTIVATION_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
}


def _resolve_activation(
    activation: ActivationType | str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve ``ActivationType`` or string alias to a functional activation."""
    if isinstance(activation, ActivationType):
        key = activation.value
    elif isinstance(activation, str):
        key = activation.lower()
    else:
        raise TypeError(
            f"activation must be ActivationType or str, got {type(activation).__name__}"
        )
    if key not in _ACTIVATION_FNS:
        raise ValueError(
            f"Unsupported activation '{key}'. Supported: {sorted(_ACTIVATION_FNS)}"
        )
    return _ACTIVATION_FNS[key]


class TorchMLPClassifier(nn.Module):
    """
    Torch-native MLP classifier.

    Args:
        input_size: Input feature dimension.
        hidden_size: Intermediate dimension for the MLP block.
        num_labels: Number of output classes (default 1 for binary logits).
        activation: ``ActivationType`` enum or lowercase string alias
            (``"gelu"``, ``"relu"``, ``"silu"``).
        bias: Whether the linear layers use learnable biases.

    Shape:
        - Input: ``(batch, input_size)``
        - Output: ``(batch, num_labels)``
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_labels: int = 1,
        activation: ActivationType | str = "gelu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_labels = int(num_labels)
        self.bias_enabled = bool(bias)
        self._activation_fn = _resolve_activation(activation)

        self.up_proj: nn.Linear = nn.Linear(
            self.input_size, self.hidden_size, bias=self.bias_enabled, dtype=torch.float32
        )
        self.down_proj: nn.Linear = nn.Linear(
            self.hidden_size, self.input_size, bias=self.bias_enabled, dtype=torch.float32
        )
        self.classifier: nn.Linear = nn.Linear(
            self.input_size, self.num_labels, bias=self.bias_enabled, dtype=torch.float32
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features, shape ``(batch, input_size)``.

        Returns:
            Logits of shape ``(batch, num_labels)``.
        """
        h = self.up_proj(x)
        h = self._activation_fn(h)
        h = self.down_proj(h)
        return self.classifier(h)
