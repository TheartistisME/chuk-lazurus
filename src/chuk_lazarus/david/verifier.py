"""Verification layer for David runtime capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .tools import LocalTools


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    capability: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    command_result: dict[str, Any] | None = None
    reason: str = "ok"

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "capability": self.capability,
            "evidence": self.evidence,
            "command_result": self.command_result,
            "reason": self.reason,
        }


class Verifier:
    def __init__(self, tools: LocalTools) -> None:
        self.tools = tools

    def verify(self, *, capability: str, evidence: list[dict[str, Any]], command: Sequence[str] | None = None) -> VerificationResult:
        if command is not None:
            result = self.tools.run(command)
            return VerificationResult(
                ok=result["returncode"] == 0,
                capability=capability,
                evidence=evidence,
                command_result=result,
                reason="command passed" if result["returncode"] == 0 else "command failed",
            )
        if capability in {"temporal_recall", "symbolic_multi_hop", "repo_patch", "source_dependency"}:
            return VerificationResult(bool(evidence), capability, evidence, reason="evidence present" if evidence else "missing evidence")
        return VerificationResult(True, capability, evidence)

