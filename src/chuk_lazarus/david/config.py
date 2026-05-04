"""Configuration objects for the David offline runtime core."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any


def _env_optional_string(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_model_max_new_tokens() -> int:
    value = os.environ.get("DAVID_MODEL_MAX_NEW_TOKENS")
    if value is None or not value.strip():
        return 160
    try:
        parsed = int(value)
    except ValueError:
        return 160
    return max(1, parsed)


def _env_model_backend() -> str:
    value = os.environ.get("DAVID_MODEL_BACKEND")
    if value is None or not value.strip():
        return "transformers"
    return value.strip().lower()


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
    model_path: str | None = None
    validation_report_path: str | None = None
    require_validated_model: bool = True
    model_backend: str = field(default_factory=_env_model_backend)
    model_device: str | None = field(default_factory=lambda: _env_optional_string("DAVID_MODEL_DEVICE"))
    model_dtype: str | None = field(default_factory=lambda: _env_optional_string("DAVID_MODEL_DTYPE") or "auto")
    model_max_new_tokens: int = field(default_factory=_env_model_max_new_tokens)
    color: bool = True
    once: str | None = None
    verify_command: str | None = None
    command_timeout_seconds: int | None = None
    adapter: AdapterSessionMetadata = field(default_factory=AdapterSessionMetadata)
    auto_jit_index: bool = False
    model_tool_protocol: bool = False
    max_route_tokens: int = 2048

    @classmethod
    def from_values(
        cls,
        *,
        workspace_path: Path,
        model_path: str | None,
        validation_report_path: str | None,
        require_validated_model: bool,
        color: bool,
        once: str | None,
        verify_command: str | None,
        command_timeout_seconds: int | None,
        auto_jit_index: bool = False,
        model_device: str | None = None,
        model_dtype: str | None = None,
        model_max_new_tokens: int | None = None,
    ) -> "DavidConfig":
        kwargs: dict[str, Any] = {
            "workspace_root": workspace_path,
            "model_path": model_path,
            "validation_report_path": validation_report_path,
            "require_validated_model": require_validated_model,
            "color": color,
            "once": once,
            "verify_command": verify_command,
            "command_timeout_seconds": command_timeout_seconds,
            "auto_jit_index": auto_jit_index,
        }
        if model_device is not None:
            kwargs["model_device"] = model_device
        if model_dtype is not None:
            kwargs["model_dtype"] = model_dtype
        if model_max_new_tokens is not None:
            kwargs["model_max_new_tokens"] = model_max_new_tokens
        return cls(**kwargs)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_root", Path(self.workspace_root).resolve())
        if self.state_dir is None:
            state_dir = self.workspace_root / ".david"
        else:
            state_dir = Path(self.state_dir).resolve()
        object.__setattr__(self, "state_dir", state_dir)

    @property
    def workspace_path(self) -> Path:
        return self.workspace_root

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
