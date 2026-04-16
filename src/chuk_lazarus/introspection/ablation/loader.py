"""Model loading utilities for ablation studies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapter import ModelAdapter


def load_model_for_ablation(
    model_id: str,
    *,
    backend: str | None = None,
    device: str | None = None,
) -> ModelAdapter:
    """Load a model and wrap it in a backend-aware ModelAdapter."""
    from ...inference import UnifiedPipeline, UnifiedPipelineConfig
    from .adapter import ModelAdapter

    pipeline = UnifiedPipeline.from_pretrained(
        model_id,
        pipeline_config=UnifiedPipelineConfig(
            backend_name=backend,
            device=device,
        ),
        verbose=False,
    )
    return ModelAdapter(
        pipeline.model,
        pipeline.tokenizer,
        pipeline.config,
        runtime=pipeline.runtime,
        pipeline=pipeline,
    )
