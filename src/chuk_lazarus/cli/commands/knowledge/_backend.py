"""Backend dispatch helpers for the maintained knowledge CLI surface."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

from ._common import _build_pipeline_config, _resolve_backend_device, knowledge_backend_env


@dataclass(frozen=True)
class KnowledgeBackendSelection:
    """Resolved backend choice for a knowledge CLI invocation."""

    requested: str | None
    effective: str
    device: str | None


def resolve_backend_selection(args: Namespace) -> KnowledgeBackendSelection:
    """Default the maintained knowledge surface to the existing MLX path."""
    requested, device = _resolve_backend_device(args)
    effective = requested or "mlx"
    return KnowledgeBackendSelection(requested=requested, effective=effective, device=device)


def load_pipeline(
    model_id: str,
    *,
    backend: str | None = None,
    device: str | None = None,
):
    """Load a UnifiedPipeline under the backend/device selection contract."""
    from ....inference import UnifiedPipeline

    pipeline_kwargs = {"verbose": False}
    pipeline_config = _build_pipeline_config(backend=backend, device=device)
    if pipeline_config is not None:
        pipeline_kwargs["pipeline_config"] = pipeline_config

    with knowledge_backend_env(backend=backend, device=device):
        return UnifiedPipeline.from_pretrained(model_id, **pipeline_kwargs)


def load_build_backend(backend_name: str):
    """Resolve the build entrypoint for the requested backend."""
    from ....inference.context import knowledge as knowledge_pkg

    if backend_name == "torch":
        return knowledge_pkg.build_knowledge_store_torch
    return knowledge_pkg.streaming_prefill


def load_store_backend(backend_name: str):
    """Resolve the store class for the requested backend."""
    from ....inference.context import knowledge as knowledge_pkg

    if backend_name == "torch":
        return knowledge_pkg.TorchKnowledgeStore
    return knowledge_pkg.KnowledgeStore


def require_runtime_backend(command_name: str, backend_name: str) -> None:
    """Guard torch query/chat until the runtime batch lands."""
    if backend_name == "torch":
        raise RuntimeError(
            f"`knowledge {command_name} --backend torch` is wired to the maintained CLI seam, "
            "but the torch query/chat runtime is outside batch chuk-lazurus-wbt.3."
        )
