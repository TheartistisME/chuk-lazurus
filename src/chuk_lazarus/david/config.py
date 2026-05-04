"""Configuration objects for the David offline runtime core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_NAME = "david.workspace_config"
CONFIG_SCHEMA_VERSION = 1
LEGACY_CONFIG_SCHEMA_VERSION = 0

DEFAULT_MODEL_BACKEND = "transformers"
DEFAULT_MODEL_DTYPE = "auto"
DEFAULT_MODEL_MAX_NEW_TOKENS = 160
DEFAULT_VALIDATION_REPORT_PATH = ".david/model_validation/model_validation_report.json"
DEFAULT_MODEL_ATTESTATION_PATH = ".david/model_validation/model_attestation.json"
DEFAULT_WORKSPACE_PATHS = {
    "state_dir": ".david",
    "memory_dir": ".david/memory",
    "indexes_dir": ".david/indexes",
    "model_validation_dir": ".david/model_validation",
    "sessions_dir": ".david/sessions",
}


class DavidConfigSchemaError(ValueError):
    """Raised when an operator config cannot be trusted or migrated."""


def load_workspace_operator_config(config_path: str | Path) -> dict[str, Any]:
    """Load and normalize a ``.david/config.json`` operator config document."""

    path = Path(config_path)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DavidConfigSchemaError(
            f"invalid David config JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise DavidConfigSchemaError(f"could not read David config {path}: {exc}") from exc
    return coerce_workspace_operator_config(loaded)


def coerce_workspace_operator_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the current operator config shape for current or legacy payloads.

    Older/minimal configs are upgraded in memory by filling defaults and stamping
    migration/readiness metadata. Unknown top-level keys are preserved for
    forward-compatible operator notes.
    """

    if not isinstance(payload, Mapping):
        raise DavidConfigSchemaError("David config root must be a JSON object")

    config = dict(payload)
    source_schema_name = _optional_config_string(config.get("schema_name"))
    if source_schema_name is not None and source_schema_name != CONFIG_SCHEMA_NAME:
        raise DavidConfigSchemaError(f"unsupported David config schema_name: {source_schema_name}")

    source_schema_version = _coerce_schema_version(config.get("schema_version"))
    is_current_schema = (
        source_schema_name == CONFIG_SCHEMA_NAME
        and source_schema_version == CONFIG_SCHEMA_VERSION
    )

    config["schema_name"] = CONFIG_SCHEMA_NAME
    config["schema_version"] = CONFIG_SCHEMA_VERSION
    config["paths"] = _coerce_workspace_paths(config.get("paths"))
    config["model"] = _optional_config_string(config.get("model"))
    config["validation_report"] = (
        _optional_config_string(config.get("validation_report")) or DEFAULT_VALIDATION_REPORT_PATH
    )
    config["model_attestation"] = (
        _optional_config_string(config.get("model_attestation")) or DEFAULT_MODEL_ATTESTATION_PATH
    )
    config["model_backend"] = _clean_backend(config.get("model_backend"))
    config["model_device"] = _optional_config_string(config.get("model_device"))
    config["model_dtype"] = _optional_config_string(config.get("model_dtype")) or DEFAULT_MODEL_DTYPE
    config["model_max_new_tokens"] = _positive_int_or_default(
        config.get("model_max_new_tokens"),
        DEFAULT_MODEL_MAX_NEW_TOKENS,
    )
    config["auto_jit_index"] = _bool_or_default(config.get("auto_jit_index"), False)
    config["migration"] = _migration_block(source_schema_version, is_current_schema)
    config["readiness"] = _readiness_block(config, is_current_schema)
    return config


def _env_optional_string(name: str) -> str | None:
    return _optional_config_string(os.environ.get(name))


def _env_model_max_new_tokens() -> int:
    value = os.environ.get("DAVID_MODEL_MAX_NEW_TOKENS")
    return _positive_int_or_default(value, DEFAULT_MODEL_MAX_NEW_TOKENS)


def _env_model_backend() -> str:
    return _clean_backend(os.environ.get("DAVID_MODEL_BACKEND"))


def _coerce_schema_version(value: Any) -> int:
    if value is None:
        return LEGACY_CONFIG_SCHEMA_VERSION
    if isinstance(value, bool):
        raise DavidConfigSchemaError("David config schema_version must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DavidConfigSchemaError("David config schema_version must be an integer") from exc
    if parsed < 0:
        raise DavidConfigSchemaError("David config schema_version must be non-negative")
    if parsed > CONFIG_SCHEMA_VERSION:
        raise DavidConfigSchemaError(
            f"unsupported David config schema_version {parsed}; "
            f"current version is {CONFIG_SCHEMA_VERSION}"
        )
    return parsed


def _coerce_workspace_paths(value: Any) -> dict[str, str]:
    paths = dict(DEFAULT_WORKSPACE_PATHS)
    if not isinstance(value, Mapping):
        return paths
    for key, default in DEFAULT_WORKSPACE_PATHS.items():
        configured = _optional_config_string(value.get(key))
        paths[key] = configured or default
    return paths


def _migration_block(source_schema_version: int, is_current_schema: bool) -> dict[str, Any]:
    status = "current" if is_current_schema else "coerced_legacy"
    return {
        "schema_name": CONFIG_SCHEMA_NAME,
        "source_schema_version": source_schema_version,
        "target_schema_version": CONFIG_SCHEMA_VERSION,
        "status": status,
        "requires_migration": False,
        "applied": [] if is_current_schema else [f"legacy_v{source_schema_version}_operator_defaults"],
        "pending": [],
    }


def _readiness_block(config: Mapping[str, Any], is_current_schema: bool) -> dict[str, str]:
    return {
        "config_schema": "current" if is_current_schema else "coerced_legacy",
        "model_config": "configured" if config.get("model") else "missing",
        "model_validation": "required_before_boot",
        "workspace_index": "jit_on_boot_enabled" if config.get("auto_jit_index") else "manual",
        "auto_load_policy": "blocked_until_validation_ready",
    }


def _clean_backend(value: Any) -> str:
    return (_optional_config_string(value) or DEFAULT_MODEL_BACKEND).lower()


def _optional_config_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _bool_or_default(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _positive_int_or_default(value: Any, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


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
    model_path: str | None = field(default_factory=lambda: _env_optional_string("DAVID_MODEL"))
    validation_report_path: str | None = field(
        default_factory=lambda: _env_optional_string("DAVID_VALIDATION_REPORT")
    )
    model_attestation_path: str | None = field(
        default_factory=lambda: _env_optional_string("DAVID_MODEL_ATTESTATION")
    )
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
        model_backend: str | None = None,
        model_device: str | None = None,
        model_dtype: str | None = None,
        model_max_new_tokens: int | None = None,
        model_attestation_path: str | None = None,
    ) -> "DavidConfig":
        kwargs: dict[str, Any] = {
            "workspace_root": workspace_path,
            "require_validated_model": require_validated_model,
            "color": color,
            "once": once,
            "verify_command": verify_command,
            "command_timeout_seconds": command_timeout_seconds,
            "auto_jit_index": auto_jit_index,
        }
        if model_path is not None:
            kwargs["model_path"] = model_path
        if validation_report_path is not None:
            kwargs["validation_report_path"] = validation_report_path
        if model_attestation_path is not None:
            kwargs["model_attestation_path"] = model_attestation_path
        if model_backend is not None:
            kwargs["model_backend"] = model_backend.strip().lower()
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
