"""
Backend-neutral runtime types for inference.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LazarusBackend(str, Enum):
    """Runtime backends supported by the inference pipeline."""

    MLX = "mlx"
    CUDA = "cuda"

    def __str__(self) -> str:
        return self.value


class ResidualState(BaseModel):
    """Serializable residual state captured from a model layer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: LazarusBackend = Field(..., description="Backend that produced the state")
    layer_index: int = Field(..., ge=0, description="Layer index used for capture/injection")
    tensor: Any = Field(..., description="Captured tensor on CPU memory unless noted otherwise")
    sequence_length: int = Field(..., ge=1, description="Sequence length observed at capture time")
    hidden_size: int = Field(..., ge=1, description="Residual width")
    dtype: str = Field(..., description="Tensor dtype name")
    device: str = Field(..., description="Original device used during capture")
