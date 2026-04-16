"""Service layer for neuron analysis CLI commands.

This module provides the NeuronAnalysisService class that provides
functionality for analyzing individual neuron activations.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from chuk_lazarus.introspection._backend_dispatch import lazy_mx as mx  # EWS-6 lazy

from ..hooks import CaptureConfig, ModelHooks, PositionSelection
from .core import ActivationSteering
from .service import (
    _extract_hidden_vector,
    _get_num_layers,
    _load_pipeline,
    _normalize_direction,
    _runtime_backend_name,
    _vector_to_numpy,
)


class NeuronActivationResult(BaseModel):
    """Result of neuron activation analysis."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    neuron_idx: int = Field(..., description="Neuron index")
    min_val: float = Field(..., description="Minimum activation")
    max_val: float = Field(..., description="Maximum activation")
    mean_val: float = Field(..., description="Mean activation")
    std_val: float = Field(..., description="Standard deviation")
    weight: float | None = Field(default=None, description="Weight from direction file")
    separation: float | None = Field(default=None, description="Separation score (auto-discover)")


class DiscoveredNeuron(BaseModel):
    """Result of auto-discovered discriminative neuron."""

    model_config = ConfigDict(frozen=True)

    idx: int = Field(..., description="Neuron index")
    separation: float = Field(..., description="Separation score")
    best_pair: tuple[str, str] | None = Field(default=None, description="Best label pair")
    overall_std: float = Field(..., description="Overall standard deviation")
    mean_range: float = Field(..., description="Range of group means")
    group_means: dict[str, float] = Field(default_factory=dict, description="Mean per label group")


class NeuronAnalysisServiceConfig(BaseModel):
    """Configuration for NeuronAnalysisService."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    layers: list[int] = Field(..., description="Layers to analyze")
    neurons: list[int] | None = Field(default=None, description="Specific neurons to analyze")
    top_k: int = Field(default=10, description="Top K neurons for auto-discovery")


class NeuronAnalysisService:
    """Service class for neuron analysis operations.

    Provides a high-level interface for CLI commands to run neuron analysis
    without needing to understand the internal architecture.
    """

    Config = NeuronAnalysisServiceConfig

    @classmethod
    def load_neurons_from_direction(
        cls,
        direction_path: str,
        top_k: int = 10,
    ) -> tuple[list[int], dict[int, float], dict[str, str]]:
        """Load top neurons from a saved direction file.

        Args:
            direction_path: Path to direction file.
            top_k: Number of top neurons to load.

        Returns:
            Tuple of (neuron_indices, neuron_weights, metadata).
        """
        data = np.load(direction_path, allow_pickle=True)
        direction = data["direction"]

        # Get top neurons by absolute weight
        top_indices = np.argsort(np.abs(direction))[-top_k:][::-1]
        neurons = [int(i) for i in top_indices]
        weights = {int(i): float(direction[i]) for i in top_indices}

        metadata = {}
        if "label_positive" in data:
            metadata["positive_label"] = str(data["label_positive"])
            metadata["negative_label"] = str(data["label_negative"])

        return neurons, weights, metadata

    @classmethod
    async def auto_discover_neurons(
        cls,
        model: str,
        prompts: list[str],
        labels: list[str],
        layer: int,
        top_k: int = 10,
        backend: str | None = None,
        device: str | None = None,
    ) -> list[DiscoveredNeuron]:
        """Auto-discover discriminative neurons based on label groups.

        Args:
            model: Model path or name.
            prompts: List of prompts.
            labels: Labels for each prompt.
            layer: Layer to analyze.
            top_k: Number of top neurons to return.

        Returns:
            List of discovered neurons sorted by separation score.
        """
        pipeline = _load_pipeline(model, backend=backend, device=device)

        full_activations = [
            _extract_hidden_vector(pipeline.runtime, prompt, layer=layer) for prompt in prompts
        ]

        full_activations = np.array(full_activations)
        num_neurons = full_activations.shape[1]

        # Group by label
        unique_labels = sorted(set(labels))
        label_groups = {lbl: [] for lbl in unique_labels}
        for i, lbl in enumerate(labels):
            label_groups[lbl].append(full_activations[i])
        for lbl in unique_labels:
            label_groups[lbl] = np.array(label_groups[lbl])

        # Calculate separation for each neuron
        single_sample = all(len(label_groups[lbl]) == 1 for lbl in unique_labels)
        neuron_scores = []

        for neuron_idx in range(num_neurons):
            group_means = []
            group_stds = []
            for lbl in unique_labels:
                vals = label_groups[lbl][:, neuron_idx]
                group_means.append(np.mean(vals))
                group_stds.append(np.std(vals))

            overall_std = np.std(full_activations[:, neuron_idx])

            # Find max pairwise separation
            max_separation = 0.0
            best_pair = None
            for i, lbl1 in enumerate(unique_labels):
                for j, lbl2 in enumerate(unique_labels):
                    if i >= j:
                        continue
                    mean_diff = abs(group_means[i] - group_means[j])

                    if single_sample:
                        separation = mean_diff / overall_std if overall_std > 1e-6 else 0.0
                    else:
                        pooled_std = np.sqrt((group_stds[i] ** 2 + group_stds[j] ** 2) / 2)
                        separation = mean_diff / pooled_std if pooled_std > 1e-6 else 0.0

                    if separation > max_separation:
                        max_separation = separation
                        best_pair = (lbl1, lbl2)

            mean_range = max(group_means) - min(group_means)

            neuron_scores.append(
                DiscoveredNeuron(
                    idx=neuron_idx,
                    separation=max_separation,
                    best_pair=best_pair,
                    overall_std=overall_std,
                    mean_range=mean_range,
                    group_means={lbl: group_means[i] for i, lbl in enumerate(unique_labels)},
                )
            )

        # Sort and return top-k
        neuron_scores.sort(key=lambda x: -x.separation)
        return neuron_scores[:top_k]

    @classmethod
    async def analyze_neurons(
        cls,
        model: str,
        prompts: list[str],
        neurons: list[int],
        layers: list[int],
        steer_config: dict[str, Any] | None = None,
        backend: str | None = None,
        device: str | None = None,
    ) -> dict[int, list[NeuronActivationResult]]:
        """Analyze neuron activations across prompts.

        Args:
            model: Model path or name.
            prompts: List of prompts to analyze.
            neurons: Neuron indices to analyze.
            layers: Layers to analyze.
            steer_config: Optional steering configuration.

        Returns:
            Dict mapping layer -> list of neuron results.
        """
        pipeline = _load_pipeline(model, backend=backend, device=device)
        backend_name = _runtime_backend_name(pipeline.runtime)
        target_layers = sorted({int(layer) for layer in layers})

        # Collect activations
        all_activations = {layer: [] for layer in target_layers}

        for prompt in prompts:
            if steer_config and backend_name == "torch":
                hidden_by_layer = cls._capture_hidden_states_torch(
                    pipeline,
                    prompt,
                    target_layers,
                    steer_config=steer_config,
                )
                for layer in target_layers:
                    all_activations[layer].append(hidden_by_layer[layer])
                continue

            if steer_config:
                hidden_by_layer = cls._capture_hidden_states_mlx(
                    pipeline,
                    prompt,
                    target_layers,
                    steer_config=steer_config,
                )
                for layer in target_layers:
                    all_activations[layer].append(hidden_by_layer[layer])
                continue

            for layer in target_layers:
                all_activations[layer].append(
                    _extract_hidden_vector(pipeline.runtime, prompt, layer=layer)
                )

        # Compute statistics per layer
        results = {}
        for layer in target_layers:
            activations = np.array(all_activations[layer])
            layer_results = []

            for neuron in neurons:
                vals = activations[:, neuron]
                layer_results.append(
                    NeuronActivationResult(
                        neuron_idx=neuron,
                        min_val=float(vals.min()),
                        max_val=float(vals.max()),
                        mean_val=float(vals.mean()),
                        std_val=float(vals.std()),
                    )
                )

            results[layer] = layer_results

        return results

    @staticmethod
    def _capture_hidden_states_mlx(
        pipeline: Any,
        prompt: str,
        layers: list[int],
        *,
        steer_config: dict[str, Any] | None = None,
    ) -> dict[int, np.ndarray]:
        """Capture hidden states on MLX, optionally with steering applied."""
        model_obj = pipeline.model
        tokenizer = pipeline.tokenizer
        config = pipeline.config

        steerer = None
        if steer_config:
            steerer = ActivationSteering(model_obj, tokenizer)
            steerer.add_direction(
                steer_config["layer"],
                steer_config["direction"],
            )
            steerer._wrap_layer(steer_config["layer"], steer_config["coefficient"])

        try:
            hooks = ModelHooks(model_obj, model_config=config)
            hooks.configure(
                CaptureConfig(
                    layers=layers,
                    capture_hidden_states=True,
                    positions=PositionSelection.LAST,
                )
            )
            input_ids = tokenizer.encode(prompt, return_tensors="np")
            hooks.forward(mx.array(input_ids))

            captured = {}
            for layer in layers:
                h = hooks.state.hidden_states[layer][0, 0, :]
                captured[layer] = np.array(h.astype(mx.float32), copy=False)
            return captured
        finally:
            if steerer:
                steerer._unwrap_layers()

    @staticmethod
    def _capture_hidden_states_torch(
        pipeline: Any,
        prompt: str,
        layers: list[int],
        *,
        steer_config: dict[str, Any] | None = None,
    ) -> dict[int, np.ndarray]:
        """Capture hidden states from a torch runtime, optionally with steering."""
        runtime = pipeline.runtime
        model = pipeline.model

        import torch

        resolve_layers = getattr(runtime, "_resolve_layers", None)
        if callable(resolve_layers):
            model_layers = resolve_layers()
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            model_layers = model.model.layers
        elif hasattr(model, "layers"):
            model_layers = model.layers
        else:
            raise ValueError("Cannot detect torch model layer structure")

        tokenize_prompt = getattr(runtime, "_tokenize_prompt", None)
        if callable(tokenize_prompt):
            model_inputs = tokenize_prompt(prompt)
        else:
            batch = pipeline.tokenizer(prompt, return_tensors="pt")
            model_inputs = {
                name: tensor.to(getattr(runtime, "_device", "cpu"), non_blocking=True)
                for name, tensor in batch.items()
            }

        def unwrap_hidden_states(output: Any):
            if hasattr(output, "hidden_states"):
                return output.hidden_states
            if isinstance(output, tuple):
                return output[0]
            return output

        def replace_hidden_states(output: Any, hidden_states: Any):
            if hasattr(output, "hidden_states"):
                try:
                    output.hidden_states = hidden_states
                    return output
                except Exception:
                    cloned = copy.copy(output)
                    cloned.hidden_states = hidden_states
                    return cloned
            if isinstance(output, tuple):
                return (hidden_states, *output[1:])
            return hidden_states

        handles = []
        captured: dict[int, np.ndarray] = {}

        if steer_config:
            direction = _normalize_direction(steer_config["direction"])
            steer_layer = int(steer_config["layer"])
            coefficient = float(steer_config["coefficient"])

            def steering_hook(_module, _args, output):
                hidden = unwrap_hidden_states(output)
                steering = torch.from_numpy(direction).to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                hidden_norm = torch.sqrt(torch.mean(hidden * hidden))
                updated = hidden.clone()
                updated = updated + steering.view(1, 1, -1) * (coefficient * hidden_norm)
                return replace_hidden_states(output, updated)

            handles.append(model_layers[steer_layer].register_forward_hook(steering_hook))

        for layer_idx in layers:
            def capture_hook(_module, _args, output, layer_idx=layer_idx):
                hidden = unwrap_hidden_states(output)
                captured[layer_idx] = _vector_to_numpy(hidden[:, -1, :])

            handles.append(model_layers[layer_idx].register_forward_hook(capture_hook))

        try:
            with torch.inference_mode():
                model(**model_inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        return captured


__all__ = [
    "NeuronAnalysisService",
    "NeuronAnalysisServiceConfig",
    "NeuronActivationResult",
    "DiscoveredNeuron",
]
