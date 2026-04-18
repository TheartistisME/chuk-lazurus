"""Torch-native training subpackage.

Isolated from the MLX training package: importing this subpackage never loads
``mlx.*``. The MLX training package does not re-export anything from here.
"""

from __future__ import annotations

from chuk_lazarus.training.torch.torch_base_trainer import (
    TorchBaseTrainer,
    TorchTrainerConfig,
)
from chuk_lazarus.training.torch.torch_classification_trainer import (
    TorchClassificationTrainer,
)

__all__ = [
    "TorchBaseTrainer",
    "TorchClassificationTrainer",
    "TorchTrainerConfig",
]
