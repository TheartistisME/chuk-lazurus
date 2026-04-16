"""
Abstract runtime interface for inference backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .types import LazarusBackend, ResidualState

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ..generation import GenerationConfig, GenerationResult


class InferenceRuntime(ABC):
    """Backend-specific inference operations used by UnifiedPipeline."""

    def __init__(self, model, tokenizer: PreTrainedTokenizer):
        self._model = model
        self._tokenizer = tokenizer

    @property
    @abstractmethod
    def backend(self) -> LazarusBackend:
        """Backend identifier."""

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate a response for a prompt."""

    @abstractmethod
    def extract_residual_state(self, prompt: str, layer_index: int) -> ResidualState:
        """Capture a residual state for a prompt."""

    @abstractmethod
    def generate_with_residual(
        self,
        prompt: str,
        residual_state: ResidualState,
        config: GenerationConfig,
    ) -> GenerationResult:
        """Generate with an injected residual state."""

    @abstractmethod
    def clear_cache(self) -> None:
        """Release backend-specific caches if needed."""
