"""Capability-based task detection and route packet construction."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


PRODUCTION_METHODS = {
    "repo_patch",
    "source_dependency",
    "symbolic_multi_hop",
    "temporal_recall",
    "user_continuity",
    "verify",
}


class MethodDetector:
    def detect(self, prompt: str) -> str:
        text = prompt.lower()
        if any(word in text for word in ("pytest", "verify", "test command", "quality gate")):
            return "verify"
        if any(word in text for word in ("patch", "fix", "bug", "edit", "write file", "repo")):
            return "repo_patch"
        if any(word in text for word in ("import", "dependency", "symbol", "call graph", "source")):
            return "source_dependency"
        if any(word in text for word in ("chain", "multi-hop", "depends on", "because")):
            return "symbolic_multi_hop"
        if any(word in text for word in ("latest", "first", "last time", "previous", "earliest", "when did")):
            return "temporal_recall"
        user_memory_markers = (
            "remember",
            "remind me",
            "follow up",
            "check back",
            "tomorrow",
            "deadline",
            "preference",
            "my ",
            "i prefer",
            "i am worried",
            "i'm worried",
        )
        if any(marker in text for marker in user_memory_markers):
            return "user_continuity"
        return "source_dependency"


@dataclass(frozen=True)
class RoutePacket:
    method: str
    selected_windows: list[str]
    memory_family: str
    session_id: str
    tier: str
    route_reason: str
    evidence: list[dict[str, Any]]
    token_cost: int
    activation_score: float = 0.0
    lexical_score: float = 0.0
    ordinal_score: float = 0.0
    recency_score: float = 0.0
    residual_available: bool = False
    kv_ready: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "selected_windows": self.selected_windows,
            "memory_family": self.memory_family,
            "session_id": self.session_id,
            "tier": self.tier,
            "route_reason": self.route_reason,
            "evidence": self.evidence,
            "token_cost": self.token_cost,
            "activation_score": self.activation_score,
            "lexical_score": self.lexical_score,
            "ordinal_score": self.ordinal_score,
            "recency_score": self.recency_score,
            "residual_available": self.residual_available,
            "kv_ready": self.kv_ready,
            "provenance": self.provenance,
        }


class CentralRouter:
    def route(self, *, method: str, prompt: str, session_id: str, evidence: list[dict[str, Any]], max_tokens: int) -> RoutePacket:
        windows = [item["text"] for item in evidence[:3]]
        lexical = min(1.0, sum(item.get("score", 0) for item in evidence) / 10.0)
        token_cost = min(max_tokens, sum(len(re.findall(r"\S+", window)) for window in windows))
        memory_family = "user" if method in {"user_continuity", "temporal_recall"} else "task"
        tier = "hot" if evidence else "cold"
        return RoutePacket(
            method=method,
            selected_windows=windows,
            memory_family=memory_family,
            session_id=session_id,
            tier=tier,
            route_reason=f"{method} methodology selected from production capability detector",
            evidence=evidence,
            token_cost=token_cost,
            lexical_score=lexical,
            ordinal_score=1.0 if method == "temporal_recall" and evidence else 0.0,
            recency_score=1.0 if evidence else 0.0,
            residual_available=False,
            kv_ready=False,
            provenance={"router": "david.central_router.offline", "prompt_chars": len(prompt)},
        )
