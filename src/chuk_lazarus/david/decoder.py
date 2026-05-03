"""Decoder constraints and durable prior scoping for David."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AdapterSessionMetadata
from .routing import RoutePacket


@dataclass(frozen=True)
class DecoderPlan:
    constraints: dict[str, Any]
    prior_scope: dict[str, Any]


class DecoderController:
    def plan(self, *, route: RoutePacket, adapter: AdapterSessionMetadata, session_id: str) -> DecoderPlan:
        constraints: dict[str, Any] = {"format": "deterministic_summary"}
        if route.method == "repo_patch":
            constraints.update(
                {
                    "bias": ["valid_file_paths", "existing_symbols", "known_imports", "patch_compatible_edits", "known_tests"],
                    "tool_calls": ["read", "write", "list", "run"],
                }
            )
        elif route.method == "verify":
            constraints.update({"commands": "path_safe_local_commands", "capture": ["returncode", "stdout", "stderr"]})
        elif route.method == "symbolic_multi_hop":
            constraints.update({"evidence": "complete_chain_required"})
        elif route.method == "temporal_recall":
            constraints.update({"ordinal": "exact_occurrence_required"})
        elif route.method == "user_continuity":
            constraints.update({"respect": ["recency", "staleness", "confirmed_preferences"]})
        prior_scope = adapter.scope() | {"session_id": session_id, "method": route.method}
        return DecoderPlan(constraints=constraints, prior_scope=prior_scope)

