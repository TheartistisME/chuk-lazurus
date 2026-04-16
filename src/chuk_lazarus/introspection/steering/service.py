"""Service layer for steering CLI commands.

This module provides the SteeringService class that wraps ActivationSteering
to provide a simple interface for CLI commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import SteeringConfig
from .core import ActivationSteering


def _load_pipeline(model_id: str, *, backend: str | None = None, device: str | None = None):
    from ...inference import UnifiedPipeline, UnifiedPipelineConfig

    return UnifiedPipeline.from_pretrained(
        model_id,
        pipeline_config=UnifiedPipelineConfig(
            backend_name=backend,
            device=device,
        ),
        verbose=False,
    )


def _get_num_layers(model: Any, config: Any) -> int:
    if hasattr(config, "num_hidden_layers"):
        return int(config.num_hidden_layers)
    if hasattr(config, "num_layers"):
        return int(config.num_layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "layers"):
        return len(model.layers)
    return 32


def _get_hidden_size(model: Any, config: Any) -> int:
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "d_model"):
        return int(config.d_model)
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    if hasattr(model, "model") and hasattr(model.model, "hidden_size"):
        return int(model.model.hidden_size)
    return 4096


def _runtime_backend_name(runtime: Any) -> str | None:
    backend = getattr(runtime, "backend", None)
    if backend is None:
        return None
    name = str(backend).lower()
    if name == "cuda":
        return "torch"
    if name == "mlx":
        return "mlx"
    return name


def _vector_to_numpy(vector: Any) -> np.ndarray:
    if isinstance(vector, np.ndarray):
        array = vector
    elif type(vector).__module__.startswith("torch"):
        array = vector.detach().cpu().numpy()
    else:
        array = np.asarray(vector)
    return array.astype(np.float32, copy=False).reshape(-1)


def _extract_hidden_vector(runtime: Any, prompt: str, *, layer: int) -> np.ndarray:
    residual_state = runtime.extract_residual_state(prompt, layer_index=layer)
    tensor = residual_state.tensor
    array = _vector_to_numpy(tensor)
    return array.reshape(-1)


def _normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(direction)
    return direction / (norm + 1e-8)


def _steer_residual_state(base_state: Any, direction: np.ndarray, coefficient: float):
    from ...inference.backends.types import ResidualState

    base_tensor = base_state.tensor
    if not type(base_tensor).__module__.startswith("torch"):
        raise RuntimeError("Residual steering injection currently requires the torch runtime.")

    import torch

    base_array = _vector_to_numpy(base_tensor)
    base_norm = float(np.sqrt(np.mean(base_array * base_array)))
    steering = _normalize_direction(direction) * (coefficient * base_norm)

    update = torch.from_numpy(steering)
    steered_tensor = base_tensor.detach().cpu().clone()
    update = update.to(dtype=steered_tensor.dtype)

    if steered_tensor.ndim == 1:
        steered_tensor = steered_tensor + update
    else:
        steered_tensor = steered_tensor.clone()
        steered_tensor[..., :] = steered_tensor[..., :] + update

    return ResidualState(
        backend=base_state.backend,
        layer_index=base_state.layer_index,
        tensor=steered_tensor,
        sequence_length=base_state.sequence_length,
        hidden_size=base_state.hidden_size,
        dtype=base_state.dtype,
        device=base_state.device,
    )


class SteeringServiceConfig(BaseModel):
    """Configuration for SteeringService."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    layer: int | None = Field(default=None, description="Layer for steering")
    coefficient: float = Field(default=1.0, description="Steering coefficient")
    max_tokens: int = Field(default=100, description="Max tokens to generate")
    temperature: float = Field(default=0.0, description="Generation temperature")


class DirectionExtractionResult(BaseModel):
    """Result of direction extraction."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    direction: Any = Field(..., description="Direction vector")
    layer: int = Field(..., description="Layer index")
    norm: float = Field(..., description="Direction norm")
    cosine_similarity: float = Field(
        ..., description="Cosine similarity between positive and negative"
    )
    separation: float = Field(..., description="1 - cosine similarity")
    positive_prompt: str = Field(..., description="Positive prompt")
    negative_prompt: str = Field(..., description="Negative prompt")


class SteeringGenerationResult(BaseModel):
    """Result of steering generation."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(..., description="Input prompt")
    output: str = Field(..., description="Generated output")
    layer: int = Field(..., description="Steering layer")
    coefficient: float = Field(..., description="Steering coefficient")


class SteeringComparisonResult(BaseModel):
    """Result of steering comparison."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(..., description="Input prompt")
    results: dict[float, str] = Field(..., description="Coefficient -> output mapping")


class SteeringService:
    """Service class for steering operations.

    Provides a high-level interface for CLI commands to run steering
    without needing to understand the internal architecture.
    """

    Config = SteeringServiceConfig

    @classmethod
    async def extract_direction(
        cls,
        model: str,
        positive_prompt: str,
        negative_prompt: str,
        layer: int | None = None,
        backend: str | None = None,
        device: str | None = None,
    ) -> DirectionExtractionResult:
        """Extract steering direction from contrastive prompts.

        Args:
            model: Model path or name.
            positive_prompt: Prompt for positive direction.
            negative_prompt: Prompt for negative direction.
            layer: Layer to extract from (default: middle layer).

        Returns:
            DirectionExtractionResult with direction and metadata.
        """
        pipeline = _load_pipeline(model, backend=backend, device=device)
        num_layers = _get_num_layers(pipeline.model, pipeline.config)

        # Determine layer
        target_layer = layer if layer is not None else num_layers // 2

        h_positive = _extract_hidden_vector(pipeline.runtime, positive_prompt, layer=target_layer)
        h_negative = _extract_hidden_vector(pipeline.runtime, negative_prompt, layer=target_layer)

        # Compute direction: positive - negative
        direction = h_positive - h_negative
        direction_np = np.asarray(direction, dtype=np.float32)

        # Compute statistics
        norm = float(np.linalg.norm(direction_np))
        cos_sim = float(
            np.dot(h_positive, h_negative)
            / ((np.linalg.norm(h_positive) * np.linalg.norm(h_negative)) + 1e-8)
        )

        return DirectionExtractionResult(
            direction=direction_np,
            layer=target_layer,
            norm=norm,
            cosine_similarity=cos_sim,
            separation=1 - cos_sim,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
        )

    @classmethod
    def save_direction(
        cls,
        result: DirectionExtractionResult,
        output_path: str | Path,
        model_id: str,
    ) -> None:
        """Save extracted direction to file.

        Args:
            result: Direction extraction result.
            output_path: Path to save to.
            model_id: Model identifier.
        """
        np.savez(
            output_path,
            direction=result.direction,
            layer=result.layer,
            positive_prompt=result.positive_prompt,
            negative_prompt=result.negative_prompt,
            model_id=model_id,
            norm=result.norm,
            cosine_similarity=result.cosine_similarity,
        )

    @classmethod
    def load_direction(cls, path: str | Path) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Load direction from file.

        Args:
            path: Path to direction file.

        Returns:
            Tuple of (direction, layer, metadata).
        """
        path = Path(path)

        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            direction = data["direction"]
            layer = int(data["layer"]) if "layer" in data else None

            metadata = {}
            if "positive_prompt" in data:
                metadata["positive_prompt"] = str(data["positive_prompt"])
            if "negative_prompt" in data:
                metadata["negative_prompt"] = str(data["negative_prompt"])
            if "norm" in data:
                metadata["norm"] = float(data["norm"])
            if "cosine_similarity" in data:
                metadata["cosine_similarity"] = float(data["cosine_similarity"])

            return direction, layer, metadata

        elif path.suffix == ".json":
            import json

            with open(path) as f:
                data = json.load(f)
            direction = np.array(data["direction"], dtype=np.float32)
            layer = data.get("layer")
            return direction, layer, data

        else:
            raise ValueError(f"Unsupported direction format: {path.suffix}")

    @classmethod
    def resolve_model_shape(
        cls,
        model: str,
        *,
        backend: str | None = None,
        device: str | None = None,
    ) -> tuple[int, int]:
        """Get ``(num_layers, hidden_size)`` for a model via UnifiedPipeline."""
        pipeline = _load_pipeline(model, backend=backend, device=device)
        return (
            _get_num_layers(pipeline.model, pipeline.config),
            _get_hidden_size(pipeline.model, pipeline.config),
        )

    @classmethod
    async def generate_with_steering(
        cls,
        model: str,
        prompts: list[str],
        direction: np.ndarray,
        layer: int,
        coefficient: float = 1.0,
        max_tokens: int = 100,
        temperature: float = 0.0,
        name: str = "custom",
        positive_label: str = "positive",
        negative_label: str = "negative",
        backend: str | None = None,
        device: str | None = None,
    ) -> list[SteeringGenerationResult]:
        """Generate text with steering applied.

        Args:
            model: Model path or name.
            prompts: Prompts to generate from.
            direction: Steering direction vector.
            layer: Layer to apply steering.
            coefficient: Steering coefficient.
            max_tokens: Max tokens to generate.
            temperature: Generation temperature.
            name: Direction name.
            positive_label: Positive direction label.
            negative_label: Negative direction label.

        Returns:
            List of generation results.
        """
        pipeline = _load_pipeline(model, backend=backend, device=device)
        steering_config = SteeringConfig(
            layers=[layer],
            coefficient=coefficient,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        backend_name = _runtime_backend_name(pipeline.runtime)

        results = []
        if backend_name == "torch":
            from ...inference.generation import GenerationConfig

            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

            for prompt in prompts:
                base_state = pipeline.runtime.extract_residual_state(prompt, layer_index=layer)
                steered_state = _steer_residual_state(base_state, direction, coefficient)
                output = pipeline.runtime.generate_with_residual(
                    prompt,
                    steered_state,
                    generation_config,
                )
                results.append(
                    SteeringGenerationResult(
                        prompt=prompt,
                        output=output.text,
                        layer=layer,
                        coefficient=coefficient,
                    )
                )
            return results

        steerer = ActivationSteering(
            pipeline.model,
            pipeline.tokenizer,
            model_id=model,
        )
        steerer.add_direction(
            layer=layer,
            direction=direction,
            name=name,
            positive_label=positive_label,
            negative_label=negative_label,
        )

        for prompt in prompts:
            output = steerer.generate(prompt, steering_config)
            results.append(
                SteeringGenerationResult(
                    prompt=prompt,
                    output=output,
                    layer=layer,
                    coefficient=coefficient,
                )
            )

        return results

    @classmethod
    async def compare_coefficients(
        cls,
        model: str,
        prompt: str,
        direction: np.ndarray,
        layer: int,
        coefficients: list[float],
        max_tokens: int = 100,
        temperature: float = 0.0,
        backend: str | None = None,
        device: str | None = None,
    ) -> SteeringComparisonResult:
        """Compare steering at different coefficients.

        Args:
            model: Model path or name.
            prompt: Prompt to generate from.
            direction: Steering direction vector.
            layer: Layer to apply steering.
            coefficients: Coefficients to compare.
            max_tokens: Max tokens to generate.
            temperature: Generation temperature.

        Returns:
            Comparison result with outputs for each coefficient.
        """
        pipeline = _load_pipeline(model, backend=backend, device=device)
        backend_name = _runtime_backend_name(pipeline.runtime)

        results = {}
        if backend_name == "torch":
            from ...inference.generation import GenerationConfig

            for coef in coefficients:
                base_state = pipeline.runtime.extract_residual_state(prompt, layer_index=layer)
                steered_state = _steer_residual_state(base_state, direction, coef)
                config = GenerationConfig(
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                results[coef] = pipeline.runtime.generate_with_residual(
                    prompt,
                    steered_state,
                    config,
                ).text

            return SteeringComparisonResult(prompt=prompt, results=results)

        steerer = ActivationSteering(
            pipeline.model,
            pipeline.tokenizer,
            model_id=model,
        )
        steerer.add_direction(layer=layer, direction=direction)

        for coef in coefficients:
            config = SteeringConfig(
                layers=[layer],
                coefficient=coef,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            results[coef] = steerer.generate(prompt, config, coefficient=coef)

        return SteeringComparisonResult(prompt=prompt, results=results)

    @classmethod
    def create_neuron_direction(
        cls,
        hidden_size: int,
        neuron_idx: int,
    ) -> np.ndarray:
        """Create a one-hot direction for single neuron steering.

        Args:
            hidden_size: Model hidden size.
            neuron_idx: Neuron index to steer.

        Returns:
            One-hot direction vector.
        """
        direction = np.zeros(hidden_size, dtype=np.float32)
        direction[neuron_idx] = 1.0
        return direction


__all__ = [
    "SteeringService",
    "SteeringServiceConfig",
    "DirectionExtractionResult",
    "SteeringGenerationResult",
    "SteeringComparisonResult",
]
