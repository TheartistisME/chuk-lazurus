"""Runtime configuration for the David terminal agent facade.

This module is intentionally backend-neutral. It describes how David should
boot the harness and local coding tools, but it never imports a model runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class DavidMode(str, Enum):
    """Top-level operating mode for the David runtime."""

    MODEL = "model"
    OFFLINE_SHELL = "offline_shell"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class DavidRuntimeConfig:
    """Configuration for the David product runtime.

    ``model_path`` opts into validated model boot. Leaving it unset keeps David
    in offline/no-model shell mode so local coding tools can still run.
    """

    workspace_path: str | Path = "."
    model_path: str | Path | None = None
    validation_report_path: str | Path | None = None
    require_validated_model: bool = True
    offline: bool = False
    dry_run: bool = False
    allow_no_model_shell: bool = True
    allow_shell: bool = True
    allow_custom_tools: bool = True
    memory_root: str | Path | None = None
    trace_root: str | Path | None = None
    custom_tools_root: str | Path | None = None
    tool_timeout_seconds: int = 30
    verification_timeout_seconds: int = 120
    max_tool_output_chars: int = 12000
    verification_commands: tuple[str, ...] = ()
    session_id: str | None = None

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None = None, **overrides: Any
    ) -> "DavidRuntimeConfig":
        """Build a config from CLI-style aliases or direct dataclass keys."""

        data: dict[str, Any] = {}
        no_model = False
        for source in (values or {}, overrides):
            for raw_key, value in source.items():
                key = _CONFIG_ALIASES.get(str(raw_key), str(raw_key))
                if key == "no_model":
                    no_model = bool(value)
                    continue
                if key == "allow_unvalidated":
                    data["require_validated_model"] = not bool(value)
                    continue
                if key == "no_shell":
                    data["allow_shell"] = not bool(value)
                    continue
                if key in _FIELD_NAMES:
                    data[key] = value

        if data.get("offline") is None:
            data["offline"] = False
        if no_model:
            data["offline"] = True
            data["model_path"] = None
        if "verification_commands" in data:
            data["verification_commands"] = _coerce_command_tuple(data["verification_commands"])
        return cls(**data)

    def with_overrides(self, **overrides: Any) -> "DavidRuntimeConfig":
        """Return a copy with CLI/runtime overrides applied."""

        if not overrides:
            return self
        return self.from_mapping(self.to_dict(), **overrides)

    @property
    def mode(self) -> DavidMode:
        if self.model_path is None:
            return DavidMode.OFFLINE_SHELL
        return DavidMode.MODEL

    @property
    def model_requested(self) -> bool:
        return self.mode is DavidMode.MODEL and self.model_path is not None

    @property
    def resolved_workspace_path(self) -> Path:
        return Path(self.workspace_path).expanduser().resolve()

    @property
    def resolved_model_path(self) -> Path | None:
        if self.model_path is None:
            return None
        return Path(self.model_path).expanduser().resolve()

    @property
    def resolved_validation_report_path(self) -> Path | None:
        if self.validation_report_path is None:
            return None
        return Path(self.validation_report_path).expanduser().resolve()

    @property
    def resolved_trace_root(self) -> Path:
        if self.trace_root is not None:
            return Path(self.trace_root).expanduser().resolve()
        if self.memory_root is not None:
            return Path(self.memory_root).expanduser().resolve() / "david" / "tool_traces"
        return self.resolved_workspace_path / ".chuk_lazarus" / "david" / "tool_traces"

    @property
    def resolved_custom_tools_root(self) -> Path | None:
        if self.custom_tools_root is None:
            return None
        return Path(self.custom_tools_root).expanduser().resolve()

    def validation_errors(self) -> tuple[str, ...]:
        """Return configuration errors without touching model backends."""

        errors: list[str] = []
        if not str(self.workspace_path).strip():
            errors.append("workspace_path is required")
        if self.tool_timeout_seconds <= 0:
            errors.append("tool_timeout_seconds must be positive")
        if self.verification_timeout_seconds <= 0:
            errors.append("verification_timeout_seconds must be positive")
        if self.max_tool_output_chars <= 0:
            errors.append("max_tool_output_chars must be positive")
        if self.mode is DavidMode.MODEL and self.model_path is None:
            errors.append("model_path is required for model mode")
        if self.mode is DavidMode.OFFLINE_SHELL and not self.allow_no_model_shell:
            errors.append("offline/no-model shell mode is disabled")
        return tuple(errors)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "workspace_path",
            "model_path",
            "validation_report_path",
            "memory_root",
            "trace_root",
            "custom_tools_root",
        ):
            if data[key] is not None:
                data[key] = str(data[key])
        data["mode"] = self.mode.value
        data["model_requested"] = self.model_requested
        return data


DavidConfig = DavidRuntimeConfig


_FIELD_NAMES = {field.name for field in fields(DavidRuntimeConfig)}
_CONFIG_ALIASES = {
    "workspace": "workspace_path",
    "cwd": "workspace_path",
    "model": "model_path",
    "validation_report": "validation_report_path",
    "report": "validation_report_path",
    "require_validated": "require_validated_model",
    "require_model_validation": "require_validated_model",
    "allow_unvalidated": "allow_unvalidated",
    "no_shell": "no_shell",
    "shell": "allow_shell",
    "memory": "memory_root",
    "memory_root": "memory_root",
    "no_model": "no_model",
    "shell_mode": "offline",
    "offline_shell": "offline",
    "tools_trace_root": "trace_root",
    "tool_trace_root": "trace_root",
    "custom_tool_root": "custom_tools_root",
    "timeout_seconds": "tool_timeout_seconds",
    "max_output_chars": "max_tool_output_chars",
    "verify_timeout_seconds": "verification_timeout_seconds",
    "verify_commands": "verification_commands",
}


def _coerce_command_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)
