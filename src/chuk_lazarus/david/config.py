"""Configuration objects for the David offline runtime core."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdapterSessionMetadata:
    """Compatibility scope supplied by a validated model adapter/session."""

    model_id: str = "offline-deterministic"
    tokenizer_id: str = "offline-tokenizer"
    model_revision: str = "local"
    adapter_family: str = "offline"
    hidden_size: int | None = None
    route_layer: int | None = None
    route_query_head: int | None = None
    boundary_layer: int | None = None
    kv_source_layer: int | None = None
    kv_target_layer: int | None = None
    insertion_family: str = "text_only"
    memory_family: str = "david-runtime"

    def scope(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "model_revision": self.model_revision,
            "adapter_family": self.adapter_family,
            "route_layer": self.route_layer,
            "route_query_head": self.route_query_head,
            "boundary_layer": self.boundary_layer,
            "kv_source_layer": self.kv_source_layer,
            "kv_target_layer": self.kv_target_layer,
            "insertion_family": self.insertion_family,
            "memory_family": self.memory_family,
        }


@dataclass(frozen=True)
class DavidConfig:
    """Runtime configuration for the David terminal-agent harness surface."""

    workspace_root: Path
    state_dir: Path | None = None
    session_id: str = "default"
    user_id: str = "default"
    adapter: AdapterSessionMetadata = field(default_factory=AdapterSessionMetadata)
    auto_jit_index: bool = False
    model_tool_protocol: bool = False
    max_route_tokens: int = 2048

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_root", Path(self.workspace_root).resolve())
        if self.state_dir is None:
            state_dir = self.workspace_root / ".david"
        else:
            state_dir = Path(self.state_dir).resolve()
        object.__setattr__(self, "state_dir", state_dir)

    @property
    def user_memory_path(self) -> Path:
        return self.state_dir / "memory" / f"user-{self.user_id}.jsonl"

    @property
    def task_memory_path(self) -> Path:
        return self.state_dir / "memory" / f"task-{self.session_id}.jsonl"

    @property
    def index_manifest_path(self) -> Path:
        safe_model = self.adapter.model_id.replace("/", "_").replace("\\", "_")
        return self.state_dir / "indexes" / f"{safe_model}-{self.adapter.model_revision}.json"

