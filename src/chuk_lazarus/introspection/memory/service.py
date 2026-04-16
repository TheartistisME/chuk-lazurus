"""Memory analysis service for CLI commands.

This module provides the MemoryAnalysisService class that handles
all business logic for memory analysis, keeping CLI commands thin.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...datasets import FactType
from .._backend_dispatch import backend_matmul, from_backend_tensor, to_backend_tensor


class MemoryAnalysisConfig(BaseModel):
    """Configuration for memory analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    facts: list[dict[str, Any]] = Field(..., description="Facts to analyze")
    fact_type: FactType = Field(..., description="Type of facts")
    layer: int | None = Field(default=None, description="Target layer")
    layer_depth_ratio: float | None = Field(default=None, description="Layer depth ratio")
    top_k: int = Field(default=10, description="Top-k predictions")
    classify: bool = Field(default=False, description="Classify memorization levels")
    backend: str | None = Field(default=None, description="Optional runtime backend selector")
    device: str | None = Field(default=None, description="Optional device override")

    # Memorization thresholds
    memorized_prob_threshold: float = Field(default=0.1)
    partial_prob_threshold: float = Field(default=0.01)
    weak_prob_threshold: float = Field(default=0.001)
    memorized_rank: int = Field(default=1)
    partial_rank: int = Field(default=5)
    weak_rank: int = Field(default=15)


class MemoryAnalysisResult(BaseModel):
    """Result of memory analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(..., description="Model ID")
    fact_type: str = Field(..., description="Fact type analyzed")
    layer: int = Field(..., description="Layer analyzed")
    num_facts: int = Field(..., description="Number of facts")

    # Accuracy metrics
    top1_accuracy: int = Field(default=0)
    top5_accuracy: int = Field(default=0)
    not_found: int = Field(default=0)

    # Attractor analysis
    attractors: list[dict[str, Any]] = Field(default_factory=list)

    # Category accuracy
    category_accuracy: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Classification results
    memorized: list[dict[str, Any]] = Field(default_factory=list)
    partial: list[dict[str, Any]] = Field(default_factory=list)
    weak: list[dict[str, Any]] = Field(default_factory=list)
    not_memorized: list[dict[str, Any]] = Field(default_factory=list)

    # Raw results
    results: list[dict[str, Any]] = Field(default_factory=list)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            f"MEMORY STRUCTURE ANALYSIS: {self.fact_type}",
            f"{'=' * 70}",
            "\n1. RETRIEVAL ACCURACY",
            f"   Top-1: {self.top1_accuracy}/{self.num_facts} ({100 * self.top1_accuracy / self.num_facts:.1f}%)",
            f"   Top-5: {self.top5_accuracy}/{self.num_facts} ({100 * self.top5_accuracy / self.num_facts:.1f}%)",
            f"   Not found: {self.not_found}/{self.num_facts} ({100 * self.not_found / self.num_facts:.1f}%)",
        ]

        if self.category_accuracy:
            lines.append("\n2. ACCURACY BY CATEGORY")
            for cat, metrics in sorted(self.category_accuracy.items()):
                lines.append(
                    f"   {cat}: {metrics['top1']}/{metrics['total']} top-1, "
                    f"avg_prob={metrics['avg_prob']:.3f}"
                )

        if self.attractors:
            lines.append("\n3. TOP ATTRACTOR NODES")
            for attr in self.attractors[:10]:
                lines.append(
                    f"   '{attr['answer']}': appears {attr['count']} times, "
                    f"avg_prob={attr['avg_prob']:.4f}"
                )

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    def save_plot(self, path: str | Path) -> None:
        """Save analysis plot."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Plot 1: Accuracy summary
            ax = axes[0, 0]
            labels = ["Top-1", "Top-5", "Not Found"]
            values = [self.top1_accuracy, self.top5_accuracy, self.not_found]
            ax.bar(labels, values)
            ax.set_ylabel("Count")
            ax.set_title("Retrieval Accuracy")

            # Plot 2: Category accuracy
            if self.category_accuracy:
                ax = axes[0, 1]
                cats = sorted(self.category_accuracy.keys())
                accs = [
                    self.category_accuracy[c]["top1"] / self.category_accuracy[c]["total"] * 100
                    for c in cats
                ]
                ax.bar(cats, accs)
                ax.set_ylabel("Top-1 Accuracy (%)")
                ax.set_title("Accuracy by Category")
                ax.tick_params(axis="x", rotation=45)

            # Plot 3: Top attractors
            if self.attractors:
                ax = axes[1, 0]
                answers = [a["answer"] for a in self.attractors[:10]]
                counts = [a["count"] for a in self.attractors[:10]]
                ax.barh(answers, counts)
                ax.set_xlabel("Co-activation Count")
                ax.set_title("Top Attractor Nodes")

            plt.suptitle(f"Memory Analysis: {self.fact_type} @ Layer {self.layer}")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()

        except ImportError:
            pass  # matplotlib not available


class MemoryAnalysisService:
    """Service class for memory analysis operations."""

    @staticmethod
    def _get_embed(model: Any) -> Any:
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        return model.embed_tokens

    @staticmethod
    def _get_norm(model: Any) -> Any | None:
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        if hasattr(model, "norm"):
            return model.norm
        return None

    @classmethod
    def _get_lm_head(cls, model: Any) -> Any | None:
        if hasattr(model, "lm_head") and model.lm_head is not None:
            return model.lm_head

        embed = cls._get_embed(model)
        if hasattr(embed, "as_linear"):
            return embed.as_linear

        return None

    @staticmethod
    def _resolve_target_layer(config: MemoryAnalysisConfig, num_layers: int) -> int:
        if config.layer is not None:
            return config.layer
        if config.layer_depth_ratio is not None:
            return int(num_layers * config.layer_depth_ratio)
        return int(num_layers * 0.8)

    @staticmethod
    def _reshape_hidden_for_head(hidden: Any, backend_name: str) -> Any:
        if backend_name == "torch":
            hidden = hidden.clone()
            if hidden.ndim == 1:
                return hidden.unsqueeze(0).unsqueeze(0)
            if hidden.ndim == 2:
                return hidden.unsqueeze(1)
            return hidden

        if backend_name == "mlx":
            import mlx.core as mx

            if hidden.ndim == 1:
                return hidden[None, None, :]
            if hidden.ndim == 2:
                return hidden[:, None, :]
            return hidden

        raise NotImplementedError(f"Unsupported backend: {backend_name}")

    @staticmethod
    def _unwrap_logits(outputs: Any) -> Any:
        return outputs.logits if hasattr(outputs, "logits") else outputs

    @staticmethod
    def _get_embedding_weight(embed: Any) -> Any:
        weight = getattr(embed, "weight", None)
        if weight is None:
            raise AttributeError("Cannot resolve embedding projection weight.")
        return getattr(weight, "weight", weight)

    @classmethod
    def _project_logits(cls, model: Any, hidden: Any, backend: Any) -> np.ndarray:
        backend_name = str(backend.name).lower()
        hidden_3d = cls._reshape_hidden_for_head(hidden, backend_name)
        norm = cls._get_norm(model)
        if norm is not None:
            hidden_3d = norm(hidden_3d)

        lm_head = cls._get_lm_head(model)
        if lm_head is not None:
            logits = cls._unwrap_logits(lm_head(hidden_3d))
        else:
            embed = cls._get_embed(model)
            embed_weight = cls._get_embedding_weight(embed)
            logits = backend_matmul(hidden_3d, embed_weight.T, backend=backend)

        return from_backend_tensor(logits, backend=backend)[0, -1, :]

    @staticmethod
    def _top_k_predictions(logits: np.ndarray, tokenizer: Any, k: int) -> list[dict[str, Any]]:
        vector = np.asarray(logits, dtype=np.float32)
        vector = vector - float(np.max(vector))
        probs = np.exp(vector)
        probs = probs / probs.sum()

        top_k = min(k, probs.shape[0])
        top_indices = np.argsort(probs)[-top_k:][::-1]

        return [
            {
                "token": tokenizer.decode([int(idx)]),
                "token_id": int(idx),
                "prob": float(probs[idx]),
            }
            for idx in top_indices
        ]

    @staticmethod
    def _load_pipeline(config: MemoryAnalysisConfig) -> tuple[Any, Any]:
        from ...inference import UnifiedPipeline, UnifiedPipelineConfig
        from ...models_v2.core.backend import get_backend

        resolved_backend = get_backend(name=config.backend, device=config.device)
        resolved_backend_name = str(resolved_backend.name).lower()
        resolved_device = (
            getattr(resolved_backend, "device", None) if resolved_backend_name == "torch" else None
        )

        pipeline_config = UnifiedPipelineConfig(
            backend_name=resolved_backend_name,
            device=resolved_device,
        )

        try:
            pipeline = UnifiedPipeline.from_pretrained(
                config.model,
                pipeline_config=pipeline_config,
                verbose=False,
            )
        except ValueError as exc:
            message = str(exc).lower()
            if "detect model family" in message or "supported yet" in message:
                raise ValueError(f"Unsupported model: {config.model}") from exc
            raise

        return pipeline, resolved_backend

    @classmethod
    async def analyze(cls, config: MemoryAnalysisConfig) -> MemoryAnalysisResult:
        """Analyze model's memory of facts.

        Args:
            config: Analysis configuration.

        Returns:
            MemoryAnalysisResult with analysis metrics.
        """
        pipeline, backend = cls._load_pipeline(config)
        model = pipeline.model
        tokenizer = pipeline.tokenizer
        model_config = pipeline.config

        # Determine target layer
        num_layers = getattr(model_config, "num_hidden_layers", 32)
        target_layer = cls._resolve_target_layer(config, num_layers)

        # Build answer vocabulary
        answer_vocab = {fact["answer"]: fact for fact in config.facts}

        # Analyze each fact
        results = []
        for fact in config.facts:
            query = fact["query"]
            correct_answer = fact["answer"]

            residual_state = pipeline.runtime.extract_residual_state(query, layer_index=target_layer)
            hidden = to_backend_tensor(residual_state.tensor, backend=backend)
            logits = cls._project_logits(model, hidden, backend)
            predictions = cls._top_k_predictions(logits, tokenizer, config.top_k)

            # Find correct answer rank
            correct_rank = None
            correct_prob = None
            for j, pred in enumerate(predictions):
                if pred["token"].strip() == correct_answer or correct_answer in pred["token"]:
                    correct_rank = j + 1
                    correct_prob = pred["prob"]
                    break

            # Categorize predictions
            neighborhood = {
                "correct_rank": correct_rank,
                "correct_prob": correct_prob,
                "same_category": [],
                "other_answers": [],
            }

            for pred in predictions:
                token = pred["token"].strip()
                if token == correct_answer:
                    continue
                if token in answer_vocab:
                    other_fact = answer_vocab[token]
                    if "category" in fact and fact.get("category") == other_fact.get("category"):
                        neighborhood["same_category"].append(
                            {
                                "answer": token,
                                "prob": pred["prob"],
                            }
                        )
                    else:
                        neighborhood["other_answers"].append(
                            {
                                "answer": token,
                                "prob": pred["prob"],
                            }
                        )

            results.append(
                {
                    **fact,
                    "predictions": predictions[:10],
                    "neighborhood": neighborhood,
                }
            )

        # Compute metrics
        top1 = sum(1 for r in results if r["neighborhood"]["correct_rank"] == 1)
        top5 = sum(
            1
            for r in results
            if r["neighborhood"]["correct_rank"] and r["neighborhood"]["correct_rank"] <= 5
        )
        not_found = sum(1 for r in results if r["neighborhood"]["correct_rank"] is None)

        # Category accuracy
        category_accuracy = {}
        if "category" in config.facts[0]:
            categories = list({f["category"] for f in config.facts})
            for cat in categories:
                cat_facts = [r for r in results if r.get("category") == cat]
                cat_top1 = sum(1 for r in cat_facts if r["neighborhood"]["correct_rank"] == 1)
                cat_avg_prob = np.mean([r["neighborhood"]["correct_prob"] or 0 for r in cat_facts])
                category_accuracy[cat] = {
                    "top1": cat_top1,
                    "total": len(cat_facts),
                    "avg_prob": float(cat_avg_prob),
                }

        # Attractor analysis
        answer_counts: dict[str, int] = defaultdict(int)
        answer_probs: dict[str, list[float]] = defaultdict(list)
        for r in results:
            for cat in ["same_category", "other_answers"]:
                for item in r["neighborhood"].get(cat, []):
                    answer_counts[item["answer"]] += 1
                    answer_probs[item["answer"]].append(item["prob"])

        attractors = [
            {
                "answer": answer,
                "count": count,
                "avg_prob": float(np.mean(answer_probs[answer])),
            }
            for answer, count in sorted(answer_counts.items(), key=lambda x: -x[1])[:20]
        ]

        return MemoryAnalysisResult(
            model_id=config.model,
            fact_type=config.fact_type.value,
            layer=target_layer,
            num_facts=len(config.facts),
            top1_accuracy=top1,
            top5_accuracy=top5,
            not_found=not_found,
            attractors=attractors,
            category_accuracy=category_accuracy,
            results=results,
        )


__all__ = [
    "MemoryAnalysisConfig",
    "MemoryAnalysisResult",
    "MemoryAnalysisService",
]
