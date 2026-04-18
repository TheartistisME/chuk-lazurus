"""
Torch-native linear classifier.

Single-layer linear classifier for binary or multi-class classification on
feature vectors. Mirrors the constructor signature of the MLX
``LinearClassifier`` twin so downstream tools can swap backends by selecting
the appropriate module at import time.

This is a torch-native implementation written against :mod:`torch.nn`; it is
NOT an MLX port. No device placement happens inside the module -- the caller
is responsible for ``.to(device)``. Default dtype is ``torch.float32``.
"""

from __future__ import annotations

import torch
from torch import nn


class TorchLinearClassifier(nn.Module):
    """
    Simple torch-native linear classifier.

    Args:
        input_size: Input feature dimension.
        num_labels: Number of output classes (default 1 for binary logits).
        bias: Whether the linear layer uses a learnable bias.

    Shape:
        - Input: ``(batch, input_size)``
        - Output: ``(batch, num_labels)``

    Example:
        >>> clf = TorchLinearClassifier(input_size=768, num_labels=2)
        >>> x = torch.randn(32, 768)
        >>> logits = clf(x)
        >>> logits.shape
        torch.Size([32, 2])
    """

    def __init__(
        self,
        input_size: int,
        num_labels: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.num_labels = int(num_labels)
        self.bias_enabled = bool(bias)
        self.fc: nn.Linear = nn.Linear(
            in_features=self.input_size,
            out_features=self.num_labels,
            bias=self.bias_enabled,
            dtype=torch.float32,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features, shape ``(batch, input_size)``.

        Returns:
            Logits of shape ``(batch, num_labels)``.
        """
        return self.fc(x)
