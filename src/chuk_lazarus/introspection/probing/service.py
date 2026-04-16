"""Probing services for CLI commands.

This module provides service classes for probing operations:
- MetacognitiveService: Detect strategy switches (direct vs chain-of-thought)
- UncertaintyService: Analyze model uncertainty using hidden state geometry
- ProbeService: Train linear probes for task classification
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


class MetacognitiveConfig(BaseModel):
    """Configuration for metacognitive analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    prompts: list[str] = Field(..., description="Prompts to analyze")
    decision_layer: int | None = Field(default=None, description="Decision layer")
    layer_depth_ratio: float | None = Field(default=None, description="Layer depth ratio")
    top_k: int = Field(default=5, description="Top-k predictions")
    use_raw: bool = Field(default=False, description="Use raw mode")


class MetacognitiveResult(BaseModel):
    """Result of metacognitive analysis."""

    model_config = ConfigDict(frozen=True)

    results: list[dict[str, Any]] = Field(default_factory=list)
    direct_count: int = Field(default=0)
    cot_count: int = Field(default=0)
    model_id: str = Field(default="")
    decision_layer: int = Field(default=0)
    direct_accuracy: float | None = Field(default=None)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 90}",
            "METACOGNITIVE ANALYSIS",
            f"{'=' * 90}",
            f"\nModel: {self.model_id}",
            f"Decision layer: {self.decision_layer}",
            "",
            f"{'Prompt':<25} {'Top Token':<12} {'Prob':>6} {'Strategy':<12} {'Digit?':<6} {'Match?':<6}",
            "-" * 90,
        ]

        for r in self.results:
            short_prompt = r["prompt"][:23] + ".." if len(r["prompt"]) > 25 else r["prompt"]
            digit_str = "Yes" if r["is_digit"] else "No"
            match_str = "Yes" if r["correct_start"] else ("N/A" if not r["is_digit"] else "No")
            lines.append(
                f"{short_prompt:<25} {r['decision_token']!r:<12} {r['decision_prob']:>5.1%} "
                f"{r['strategy']:<12} {digit_str:<6} {match_str:<6}"
            )

        total = len(self.results)
        lines.extend(
            [
                "-" * 90,
                "\nSummary:",
                (
                    f"  Direct computation: {self.direct_count}/{total} ({100 * self.direct_count / total:.0f}%)"
                    if total
                    else ""
                ),
                (
                    f"  Chain-of-thought: {self.cot_count}/{total} ({100 * self.cot_count / total:.0f}%)"
                    if total
                    else ""
                ),
            ]
        )

        if self.direct_accuracy is not None:
            lines.append(f"  Direct accuracy: {self.direct_accuracy:.0%}")

        return "\n".join(lines)


class MetacognitiveService:
    """Service for metacognitive analysis."""

    @classmethod
    async def analyze(cls, config: MetacognitiveConfig) -> MetacognitiveResult:
        """Analyze metacognitive strategy switches.

        Detects whether model will use direct computation or chain-of-thought
        by examining the token identity at the decision layer.
        """
        from ..analyzer import AnalysisConfig, LayerStrategy, ModelAnalyzer
        from ..utils import apply_chat_template, extract_expected_answer

        async with ModelAnalyzer.from_pretrained(config.model) as analyzer:
            info = analyzer.model_info
            tokenizer = analyzer._tokenizer

            # Determine decision layer
            if config.decision_layer is not None:
                decision_layer = config.decision_layer
            elif config.layer_depth_ratio is not None:
                decision_layer = int(info.num_layers * config.layer_depth_ratio)
            else:
                decision_layer = int(info.num_layers * 0.7)  # Default ~70%

            # Check chat template
            has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template

            # Configure to capture only decision layer
            analysis_config = AnalysisConfig(
                layer_strategy=LayerStrategy.SPECIFIC,
                capture_layers=[decision_layer],
                top_k=config.top_k,
            )

            results = []
            direct_count = 0
            cot_count = 0

            for prompt in config.prompts:
                # Apply chat template if available
                analysis_prompt = prompt
                if not config.use_raw and has_chat_template:
                    analysis_prompt = apply_chat_template(tokenizer, prompt)

                result = await analyzer.analyze(analysis_prompt, analysis_config)

                # Get prediction at decision layer
                layer_pred = None
                for lp in result.layer_predictions:
                    if lp.layer_idx == decision_layer:
                        layer_pred = lp
                        break

                if layer_pred is None:
                    continue

                top_token = layer_pred.top_token
                top_prob = layer_pred.probability

                # Detect strategy based on token identity
                is_digit = top_token.strip().isdigit()
                if is_digit:
                    strategy = "DIRECT"
                    direct_count += 1
                else:
                    strategy = "COT"
                    cot_count += 1

                # Check if it matches expected answer
                expected = extract_expected_answer(prompt)
                correct_start = False
                if expected and is_digit:
                    correct_start = expected.startswith(top_token.strip())

                results.append(
                    {
                        "prompt": prompt,
                        "expected": expected,
                        "decision_layer": decision_layer,
                        "decision_token": top_token,
                        "decision_prob": top_prob,
                        "strategy": strategy,
                        "is_digit": is_digit,
                        "correct_start": correct_start,
                        "final_token": result.predicted_token,
                        "final_prob": result.final_probability,
                    }
                )

            # Calculate direct accuracy
            direct_results = [r for r in results if r["strategy"] == "DIRECT"]
            direct_accuracy = None
            if direct_results:
                correct = sum(1 for r in direct_results if r["correct_start"])
                direct_accuracy = correct / len(direct_results)

            return MetacognitiveResult(
                results=results,
                direct_count=direct_count,
                cot_count=cot_count,
                model_id=config.model,
                decision_layer=decision_layer,
                direct_accuracy=direct_accuracy,
            )


class UncertaintyConfig(BaseModel):
    """Configuration for uncertainty analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    prompts: list[str] = Field(..., description="Prompts to analyze")
    working_prompts: list[str] = Field(
        default_factory=list, description="Working prompts for calibration"
    )
    broken_prompts: list[str] = Field(
        default_factory=list, description="Broken prompts for calibration"
    )
    layer: int | None = Field(default=None, description="Target layer")
    layer_depth_ratio: float | None = Field(default=None, description="Layer depth ratio")
    backend: str | None = Field(default=None, description="Runtime backend override")
    device: str | None = Field(default=None, description="Runtime device override")


class UncertaintyResult(BaseModel):
    """Result of uncertainty analysis."""

    model_config = ConfigDict(frozen=True)

    results: list[dict[str, Any]] = Field(default_factory=list)
    model_id: str = Field(default="")
    detection_layer: int = Field(default=0)
    separation: float = Field(default=0.0)
    confident_count: int = Field(default=0)
    uncertain_count: int = Field(default=0)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 80}",
            "UNCERTAINTY DETECTION RESULTS",
            f"{'=' * 80}",
            f"Model: {self.model_id}",
            f"Detection layer: {self.detection_layer}",
            f"Compute-Refusal separation: {self.separation:.0f}",
            "",
            f"{'Prompt':<30} {'Score':>8} {'Prediction':<12} {'->Compute':>10} {'->Refusal':>10}",
            "-" * 80,
        ]

        for r in self.results:
            lines.append(
                f"{r['prompt']:<30} {r['score']:>8.0f} {r['prediction']:<12} "
                f"{r['dist_to_compute']:>10.0f} {r['dist_to_refusal']:>10.0f}"
            )

        lines.extend(
            [
                "-" * 80,
                f"Summary: {self.confident_count} confident, {self.uncertain_count} uncertain",
            ]
        )

        return "\n".join(lines)


class UncertaintyService:
    """Service for uncertainty analysis."""

    @classmethod
    async def analyze(cls, config: UncertaintyConfig) -> UncertaintyResult:
        """Analyze model uncertainty using hidden state geometry.

        Uses hidden state distance to "compute center" vs "refusal center"
        to predict whether model is confident about an answer.
        """
        import numpy as np

        pipeline = _load_pipeline(
            config.model,
            backend=config.backend,
            device=config.device,
        )
        num_layers = _get_num_layers(pipeline.model, pipeline.config)

        # Determine detection layer
        if config.layer is not None:
            detection_layer = _normalize_layers([config.layer], num_layers=num_layers)[0]
        elif config.layer_depth_ratio is not None:
            detection_layer = _normalize_layers(
                [int(num_layers * config.layer_depth_ratio)],
                num_layers=num_layers,
            )[0]
        else:
            detection_layer = _normalize_layers([int(num_layers * 0.7)], num_layers=num_layers)[0]

        # Default calibration prompts
        working_prompts = config.working_prompts or [
            "100 - 37 = ",
            "50 + 25 = ",
            "10 * 10 = ",
            "200 - 50 = ",
            "25 * 4 = ",
        ]
        broken_prompts = config.broken_prompts or [
            "100 - 37 =",
            "50 + 25 =",
            "10 * 10 =",
            "200 - 50 =",
            "25 * 4 =",
        ]

        # Calibrate
        working_hiddens = [
            _extract_hidden_vector(pipeline.runtime, p, layer=detection_layer) for p in working_prompts
        ]
        broken_hiddens = [
            _extract_hidden_vector(pipeline.runtime, p, layer=detection_layer) for p in broken_prompts
        ]

        compute_center = np.mean(working_hiddens, axis=0)
        refusal_center = np.mean(broken_hiddens, axis=0)

        separation = float(np.linalg.norm(compute_center - refusal_center))

        # Run detection
        results = []
        confident_count = 0
        uncertain_count = 0

        for prompt in config.prompts:
            h = _extract_hidden_vector(pipeline.runtime, prompt, layer=detection_layer)

            dist_compute = float(np.linalg.norm(h - compute_center))
            dist_refusal = float(np.linalg.norm(h - refusal_center))

            score = dist_refusal - dist_compute
            prediction = "CONFIDENT" if score > 0 else "UNCERTAIN"

            if prediction == "CONFIDENT":
                confident_count += 1
            else:
                uncertain_count += 1

            results.append(
                {
                    "prompt": prompt,
                    "score": score,
                    "prediction": prediction,
                    "dist_to_compute": dist_compute,
                    "dist_to_refusal": dist_refusal,
                }
            )

        return UncertaintyResult(
            results=results,
            model_id=config.model,
            detection_layer=detection_layer,
            separation=separation,
            confident_count=confident_count,
            uncertain_count=uncertain_count,
        )


class ProbeConfig(BaseModel):
    """Configuration for probe training."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    positive_prompts: list[str] = Field(..., description="Positive class prompts")
    negative_prompts: list[str] = Field(..., description="Negative class prompts")
    positive_label: str = Field(default="Positive", description="Positive class label")
    negative_label: str = Field(default="Negative", description="Negative class label")
    layers: list[int] | None = Field(default=None, description="Target layers")
    all_layers: bool = Field(default=False, description="Use all layers")
    ridge_alpha: float = Field(default=1.0, description="Ridge regularization")
    logistic_max_iter: int = Field(default=1000, description="Max iterations")
    random_seed: int = Field(default=42, description="Random seed")
    cross_val_folds: int = Field(default=5, description="Cross-validation folds")
    backend: str | None = Field(default=None, description="Runtime backend override")
    device: str | None = Field(default=None, description="Runtime device override")


class ProbeResult(BaseModel):
    """Result of probe training."""

    model_config = ConfigDict(frozen=True)

    layer_results: list[dict[str, Any]] = Field(default_factory=list)
    best_layer: int | None = Field(default=None)
    best_accuracy: float = Field(default=0.0)
    model_id: str = Field(default="")
    positive_label: str = Field(default="")
    negative_label: str = Field(default="")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "PROBE TRAINING RESULTS",
            f"{'=' * 70}",
            f"Model: {self.model_id}",
            f"Classes: {self.positive_label} vs {self.negative_label}",
            "",
            f"{'Layer':<8} {'Accuracy':<12} {'F1':<10} {'AUC':<10}",
            "-" * 50,
        ]

        for r in self.layer_results:
            lines.append(
                f"{r['layer']:<8} {r['accuracy']:<12.3f} {r.get('f1', 0):<10.3f} {r.get('auc', 0):<10.3f}"
            )

        lines.extend(
            [
                "-" * 50,
                f"\nBest layer: {self.best_layer}",
                f"Best accuracy: {self.best_accuracy:.3f}",
            ]
        )

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save results to file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


class ProbeService:
    """Service for probe training."""

    @classmethod
    async def train_and_evaluate(cls, config: ProbeConfig) -> ProbeResult:
        """Train and evaluate linear probes on model activations.

        Uses logistic regression to find which layers can distinguish
        between two types of prompts.
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, roc_auc_score
        from sklearn.model_selection import cross_val_score

        pipeline = _load_pipeline(
            config.model,
            backend=config.backend,
            device=config.device,
        )
        num_layers = _get_num_layers(pipeline.model, pipeline.config)

        # Determine which layers to probe
        if config.all_layers:
            target_layers = list(range(num_layers))
        elif config.layers:
            target_layers = _normalize_layers(config.layers, num_layers=num_layers)
        else:
            # Default: sample 8 evenly spaced layers
            target_layers = _normalize_layers(
                [int(i * num_layers / 8) for i in range(8)],
                num_layers=num_layers,
            )

        # Collect activations at the layers we actually probe
        all_activations = {layer: [] for layer in target_layers}
        all_labels = []

        for prompt in config.positive_prompts:
            for layer in target_layers:
                all_activations[layer].append(
                    _extract_hidden_vector(pipeline.runtime, prompt, layer=layer)
                )
            all_labels.append(1)

        for prompt in config.negative_prompts:
            for layer in target_layers:
                all_activations[layer].append(
                    _extract_hidden_vector(pipeline.runtime, prompt, layer=layer)
                )
            all_labels.append(0)

        y = np.array(all_labels)

        # Train probes at each target layer
        layer_results = []
        best_layer = None
        best_accuracy = 0.0

        for layer in target_layers:
            X = np.array(all_activations[layer])

            # Train logistic regression with cross-validation
            clf = LogisticRegression(
                max_iter=config.logistic_max_iter,
                random_state=config.random_seed,
                C=1.0 / config.ridge_alpha,
            )

            # Cross-validation
            cv_scores = cross_val_score(clf, X, y, cv=min(config.cross_val_folds, len(y)))
            accuracy = float(np.mean(cv_scores))

            # Fit on full data for other metrics
            clf.fit(X, y)
            y_pred = clf.predict(X)
            y_proba = clf.predict_proba(X)[:, 1]

            f1 = float(f1_score(y, y_pred))
            try:
                auc = float(roc_auc_score(y, y_proba))
            except ValueError:
                auc = 0.0

            layer_results.append(
                {
                    "layer": layer,
                    "accuracy": accuracy,
                    "f1": f1,
                    "auc": auc,
                }
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layer = layer

        return ProbeResult(
            layer_results=layer_results,
            best_layer=best_layer,
            best_accuracy=best_accuracy,
            model_id=config.model,
            positive_label=config.positive_label,
            negative_label=config.negative_label,
        )


__all__ = [
    "MetacognitiveConfig",
    "MetacognitiveResult",
    "MetacognitiveService",
    "ProbeConfig",
    "ProbeResult",
    "ProbeService",
    "UncertaintyConfig",
    "UncertaintyResult",
    "UncertaintyService",
]
