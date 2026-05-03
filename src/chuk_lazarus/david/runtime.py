"""Offline David runtime pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .config import DavidConfig
from .decoder import DecoderController, DecoderPlan
from .indexing import IndexReadiness, WorkspaceIndex
from .materializer import MaterializedContext, Materializer
from .memory import JsonlMemoryStore, MemoryBank
from .routing import CentralRouter, MethodDetector, RoutePacket
from .tools import LocalTools
from .verifier import VerificationResult, Verifier


@dataclass(frozen=True)
class RuntimeResult:
    prompt: str
    method: str
    answer: str
    index: IndexReadiness
    route: RoutePacket
    materialized: MaterializedContext
    decoder: DecoderPlan
    verification: VerificationResult
    writeback: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "method": self.method,
            "answer": self.answer,
            "index": {
                "ready": self.index.ready,
                "required": self.index.required,
                "manifest_path": str(self.index.manifest_path),
                "reason": self.index.reason,
                "jit_plan": self.index.jit_plan,
            },
            "route": self.route.to_json(),
            "materialized": {
                "strategy": self.materialized.strategy,
                "text_context": self.materialized.text_context,
                "compatibility": self.materialized.compatibility,
                "refused": self.materialized.refused,
                "reason": self.materialized.reason,
            },
            "decoder": {
                "constraints": self.decoder.constraints,
                "prior_scope": self.decoder.prior_scope,
            },
            "verification": self.verification.to_json(),
            "writeback": self.writeback,
        }


class DavidRuntime:
    def __init__(self, config: DavidConfig) -> None:
        self.config = config
        self.index = WorkspaceIndex(config.workspace_root, config.index_manifest_path, config.adapter)
        self.memory = MemoryBank(
            JsonlMemoryStore(config.user_memory_path, "user"),
            JsonlMemoryStore(config.task_memory_path, "task"),
        )
        self.detector = MethodDetector()
        self.router = CentralRouter()
        self.materializer = Materializer()
        self.decoder = DecoderController()
        self.tools = LocalTools(config.workspace_root)
        self.verifier = Verifier(self.tools)

    @classmethod
    def create(cls, config: DavidConfig) -> "DavidRuntime":
        return cls(config)

    def run_once(self, prompt: str, *, verify_command: Sequence[str] | None = None) -> RuntimeResult:
        readiness = self.index.check()
        if readiness.required and self.config.auto_jit_index:
            self.index.jit()
            readiness = self.index.check()

        method = self.detector.detect(prompt)
        evidence = self._recall(method, prompt)
        route = self.router.route(
            method=method,
            prompt=prompt,
            session_id=self.config.session_id,
            evidence=evidence,
            max_tokens=self.config.max_route_tokens,
        )
        materialized = self.materializer.materialize(route, self.config.adapter)
        decoder = self.decoder.plan(route=route, adapter=self.config.adapter, session_id=self.config.session_id)
        verification = self.verifier.verify(capability=method, evidence=evidence, command=verify_command if method == "verify" else None)
        answer = self._answer(prompt, method, readiness, route, materialized, verification)
        artifact = self.memory.writeback(
            method=method,
            user_id=self.config.user_id,
            session_id=self.config.session_id,
            text=f"Prompt: {prompt}\nAnswer: {answer}",
            metadata={
                "provenance": "david.runtime.run_once",
                "route": route.to_json(),
                "verification": verification.to_json(),
                "decoder_prior_scope": decoder.prior_scope,
            },
        )
        return RuntimeResult(
            prompt=prompt,
            method=method,
            answer=answer,
            index=readiness,
            route=route,
            materialized=materialized,
            decoder=decoder,
            verification=verification,
            writeback=artifact.to_json(),
        )

    def _recall(self, method: str, prompt: str) -> list[dict[str, Any]]:
        if method == "symbolic_multi_hop":
            return self.memory.symbolic_chain(prompt)
        if method == "temporal_recall":
            latest = self.memory.user.temporal(prompt, ordinal="latest")
            return [latest] if latest else []
        return self.memory.recall_for_method(method, prompt)

    def _answer(
        self,
        prompt: str,
        method: str,
        readiness: IndexReadiness,
        route: RoutePacket,
        materialized: MaterializedContext,
        verification: VerificationResult,
    ) -> str:
        parts = [f"method={method}", f"tier={route.tier}", f"evidence={len(route.evidence)}"]
        if readiness.required:
            parts.append(f"jit_required={readiness.reason}")
        if materialized.refused:
            parts.append(f"materializer_refused={materialized.reason}")
        if method == "verify":
            parts.append(f"verification={'passed' if verification.ok else 'failed'}")
        if self.config.model_tool_protocol:
            parts.append("tool_protocol=available")
        parts.append(f"summary={prompt[:120]}")
        return "; ".join(parts)

