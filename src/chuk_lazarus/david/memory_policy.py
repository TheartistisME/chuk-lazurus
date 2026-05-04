"""Writeback policy helpers for David memory artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


USER_MEMORY_METHODS = {"user_continuity", "temporal_recall"}
USER_POLICY_METADATA_KEYS = {
    "effective_at",
    "expires_at",
    "sensitivity",
    "source_method",
    "supersedes",
}


def parse_policy_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_supersedes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _has_nonempty_metadata(metadata: dict[str, Any], keys: set[str]) -> bool:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, "", [], {}, ()):
            return True
    return False


def looks_like_temporary_workspace_state(text: str, metadata: dict[str, Any]) -> bool:
    if bool(metadata.get("temporary_workspace_state")):
        return True
    if bool(metadata.get("transient_workspace_state")):
        return True
    if _has_nonempty_metadata(
        metadata,
        {
            "failing_tests",
            "patch_targets",
            "selected_paths",
            "selected_tests",
            "workspace_snapshot",
            "workspace_state",
        },
    ):
        return True
    lowered = text.lower()
    return "temporary workspace state" in lowered or "transient workspace state" in lowered


@dataclass(frozen=True)
class MemoryWritebackDecision:
    family: str
    metadata: dict[str, Any]
    refused_user_reason: str | None = None


class MemoryWritebackPolicy:
    def classify(self, *, method: str, text: str, metadata: dict[str, Any]) -> MemoryWritebackDecision:
        family = "user" if method in USER_MEMORY_METHODS else "task"
        refused_reason: str | None = None
        if family == "user" and looks_like_temporary_workspace_state(text, metadata):
            family = "task"
            refused_reason = "temporary_workspace_state"
        return MemoryWritebackDecision(
            family=family,
            metadata=self.metadata_for_family(
                family=family,
                method=method,
                metadata=metadata,
                refused_user_reason=refused_reason,
            ),
            refused_user_reason=refused_reason,
        )

    def metadata_for_family(
        self,
        *,
        family: str,
        method: str,
        metadata: dict[str, Any],
        refused_user_reason: str | None = None,
    ) -> dict[str, Any]:
        cleaned = {key: value for key, value in metadata.items() if key not in USER_POLICY_METADATA_KEYS}
        if family != "user":
            if refused_user_reason:
                cleaned["user_memory_refused_reason"] = refused_user_reason
            return cleaned
        cleaned.update(
            {
                "effective_at": metadata.get("effective_at"),
                "expires_at": metadata.get("expires_at"),
                "sensitivity": str(metadata.get("sensitivity") or "normal"),
                "source_method": str(metadata.get("source_method") or method),
                "supersedes": normalize_supersedes(metadata.get("supersedes")),
            }
        )
        return cleaned
