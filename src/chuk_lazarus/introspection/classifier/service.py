"""Classifier service for CLI commands.

This module provides services for multi-class classifier training on activations.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


def _normalize_layers(layers: list[int], *, num_layers: int) -> list[int]:
    normalized: list[int] = []
    max_index = max(num_layers - 1, 0)
    for layer in layers:
        clamped = max(0, min(int(layer), max_index))
        if clamped not in normalized:
            normalized.append(clamped)
    return normalized


def _extract_hidden_vector(runtime: Any, prompt: str, *, layer: int):
    import numpy as np

    residual_state = runtime.extract_residual_state(prompt, layer_index=layer)
    tensor = residual_state.tensor

    if isinstance(tensor, np.ndarray):
        array = tensor
    elif type(tensor).__module__.startswith("torch"):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)

    if array.ndim > 1:
        array = array[0]
    return array.astype(np.float32, copy=False).reshape(-1)


class ClassifierConfig(BaseModel):
    """Configuration for classifier training."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    categories: dict[str, list[str]] = Field(..., description="Category -> prompts mapping")
    layers: list[int] | None = Field(default=None, description="Target layers")
    all_layers: bool = Field(default=False, description="Use all layers")
    layer_depth_ratio: float | None = Field(default=None, description="Layer depth ratio")
    max_iter: int = Field(default=1000, description="Max iterations")
    random_seed: int = Field(default=42, description="Random seed")
    bar_width: int = Field(default=50, description="Display bar width")
    backend: str | None = Field(default=None, description="Runtime backend override")
    device: str | None = Field(default=None, description="Runtime device override")


class ClassifierResult(BaseModel):
    """Result of classifier training."""

    model_config = ConfigDict(frozen=True)

    layer_results: list[dict[str, Any]] = Field(default_factory=list)
    best_layer: int | None = Field(default=None)
    best_accuracy: float = Field(default=0.0)
    model_id: str = Field(default="")
    categories: list[str] = Field(default_factory=list)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CLASSIFIER TRAINING RESULTS",
            f"{'=' * 70}",
            f"Model: {self.model_id}",
            f"Categories: {', '.join(self.categories)}",
            "",
            f"{'Layer':<8} {'Accuracy':<12} {'F1-Macro':<12}",
            "-" * 40,
        ]

        for r in self.layer_results:
            lines.append(f"{r['layer']:<8} {r['accuracy']:<12.3f} {r.get('f1_macro', 0):<12.3f}")

        lines.extend(
            [
                "-" * 40,
                f"\nBest layer: {self.best_layer}",
                f"Best accuracy: {self.best_accuracy:.3f}",
            ]
        )

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save results to file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


class ClassifierService:
    """Service for classifier training."""

    @classmethod
    async def train_and_evaluate(cls, config: ClassifierConfig) -> ClassifierResult:
        """Train and evaluate multi-class classifiers.

        Uses logistic regression to train classifiers that can distinguish
        between multiple categories of prompts.
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder

        pipeline = _load_pipeline(
            config.model,
            backend=config.backend,
            device=config.device,
        )
        num_layers = _get_num_layers(pipeline.model, pipeline.config)

        # Determine target layers
        if config.all_layers:
            target_layers = list(range(num_layers))
        elif config.layers:
            target_layers = _normalize_layers(config.layers, num_layers=num_layers)
        elif config.layer_depth_ratio:
            target_layers = _normalize_layers(
                [int(num_layers * config.layer_depth_ratio)],
                num_layers=num_layers,
            )
        else:
            # Default: sample 8 evenly spaced layers
            target_layers = _normalize_layers(
                [int(i * num_layers / 8) for i in range(8)],
                num_layers=num_layers,
            )

        # Collect activations
        all_activations = {layer: [] for layer in target_layers}
        all_labels = []
        categories = list(config.categories.keys())

        for category, prompts in config.categories.items():
            for prompt in prompts:
                for layer in target_layers:
                    all_activations[layer].append(
                        _extract_hidden_vector(pipeline.runtime, prompt, layer=layer)
                    )
                all_labels.append(category)

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(all_labels)

        # Train classifiers at each target layer
        layer_results = []
        best_layer = None
        best_accuracy = 0.0

        for layer in target_layers:
            X = np.array(all_activations[layer])

            # Train logistic regression
            try:
                clf = LogisticRegression(
                    max_iter=config.max_iter,
                    random_state=config.random_seed,
                    multi_class="multinomial",
                )
            except TypeError:
                clf = LogisticRegression(
                    max_iter=config.max_iter,
                    random_state=config.random_seed,
                )

            # Cross-validation
            n_samples = len(y)
            cv_folds = min(5, n_samples)
            if cv_folds >= 2:
                cv_scores = cross_val_score(clf, X, y, cv=cv_folds)
                accuracy = float(np.mean(cv_scores))
            else:
                accuracy = 0.0

            # Fit on full data for F1
            clf.fit(X, y)
            y_pred = clf.predict(X)
            f1_macro = float(f1_score(y, y_pred, average="macro"))

            layer_results.append(
                {
                    "layer": layer,
                    "accuracy": accuracy,
                    "f1_macro": f1_macro,
                }
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layer = layer

        return ClassifierResult(
            layer_results=layer_results,
            best_layer=best_layer,
            best_accuracy=best_accuracy,
            model_id=config.model,
            categories=categories,
        )


__all__ = [
    "ClassifierConfig",
    "ClassifierResult",
    "ClassifierService",
]
