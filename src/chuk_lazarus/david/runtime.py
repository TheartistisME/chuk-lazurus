"""Offline David runtime pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from .config import DavidConfig
from .decoder import DecoderController, DecoderPlan
from .indexing import IndexReadiness, WorkspaceIndex
from .materializer import MaterializedContext, Materializer
from .memory import JsonlMemoryStore, MemoryBank
from .patch_routing import DOC_SUFFIXES, SOURCE_SUFFIXES, is_protected_path, route_patch_targets
from .patching import PatchApplyDiagnostic, apply_patch_candidate, validate_patch_candidate
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

    def readiness(self) -> dict[str, str]:
        index = self.index.check()
        return {
            "model validation": f"ready ({self.config.adapter.adapter_family}:{self.config.adapter.model_id})",
            "index": "ready" if index.ready else f"missing: {index.reason}",
            "memory": self._memory_readiness(),
            "workspace": str(self.config.workspace_root),
        }

    def memory_status(self) -> str:
        user_count = len(self.memory.user.all())
        task_count = len(self.memory.task.all())
        return (
            "memory:\n"
            f"- user artifact: {self.config.user_memory_path} ({user_count} records)\n"
            f"- task artifact: {self.config.task_memory_path} ({task_count} records)"
        )

    def index_status(self) -> str:
        readiness = self.index.check()
        if readiness.ready:
            return f"index: ready at {readiness.manifest_path}"
        return f"index: {readiness.reason}\nplan: {readiness.jit_plan}"

    def verify(self, command: str | None = None) -> str:
        command = command or getattr(self.config, "verify_command", None)
        if not command:
            result = self.run_once("Verify quality gate")
            return result.answer
        shell = self.run_shell(command)
        return shell

    def run_shell(self, command: str) -> str:
        completed = subprocess.run(
            command,
            cwd=self.config.workspace_root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=getattr(self.config, "command_timeout_seconds", None) or 30,
            check=False,
        )
        output = (completed.stdout + completed.stderr).strip() or "(no output)"
        return f"$ {command}\nrc={completed.returncode}\n{output}"

    def apply_patch(self, patch_text: str) -> PatchApplyDiagnostic:
        diagnostic = validate_patch_candidate(patch_text)
        if diagnostic.failures:
            return diagnostic
        return apply_patch_candidate(self.config.workspace_root, patch_text)

    def read_file(self, path: str) -> str:
        return self.tools.read(path)

    def write_file(self, spec: str) -> str:
        path, separator, content = spec.partition(" ")
        if not separator or not path:
            return "write: expected /write <path> <content>"
        written = self.tools.write(path, content)
        return f"wrote {written.relative_to(self.config.workspace_root)}"

    def _recall(self, method: str, prompt: str) -> list[dict[str, Any]]:
        if method == "symbolic_multi_hop":
            return self.memory.symbolic_chain(prompt)
        if method == "temporal_recall":
            latest = self.memory.user.temporal(prompt, ordinal="latest")
            return [latest] if latest else []
        evidence = self.memory.recall_for_method(method, prompt)
        if method in {"repo_patch", "source_dependency"}:
            evidence = [*evidence, *self._workspace_route_evidence(prompt)]
        return evidence

    def _workspace_route_evidence(self, prompt: str) -> list[dict[str, Any]]:
        files = self._workspace_text_files()
        if not files:
            return []
        plan = route_patch_targets(prompt, files=files, limit=8)
        evidence: list[dict[str, Any]] = []
        for ordinal, item in enumerate(plan.evidence):
            windows = plan.windows_by_path.get(item.path, ())
            text = "\n".join(str(window.get("text") or "")[:2_000] for window in windows)
            evidence.append(
                {
                    "artifact_id": f"workspace:{item.path}",
                    "family": "task",
                    "kind": "patch_target",
                    "path": item.path,
                    "text": f"{item.path}\n{text}".rstrip(),
                    "timestamp": "workspace-scan",
                    "score": item.score,
                    "ordinal": ordinal,
                    "provenance": "david.patch_routing",
                    "route_reason": item.reason,
                    "classification": item.kind,
                    "selected_tests": list(plan.selected_tests),
                }
            )
        return evidence

    def _workspace_text_files(self, *, max_files: int = 80, max_bytes: int = 12_000) -> dict[str, str]:
        root = self.config.workspace_root
        skipped_dirs = {
            ".git",
            ".david",
            ".chuk_lazarus",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "node_modules",
            ".venv",
            "venv",
            "dist",
            "build",
            "target",
        }
        allowed_suffixes = SOURCE_SUFFIXES | DOC_SUFFIXES
        files: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(name for name in dirnames if name not in skipped_dirs)
            for filename in sorted(filenames):
                if len(files) >= max_files:
                    return files
                path = Path(dirpath) / filename
                try:
                    rel = path.relative_to(root)
                except ValueError:
                    continue
                rel_parts = set(rel.parts)
                rel_text = rel.as_posix()
                if rel_parts & skipped_dirs or is_protected_path(rel_text):
                    continue
                if path.suffix.lower() not in allowed_suffixes:
                    continue
                try:
                    if path.stat().st_size > max_bytes:
                        text = path.read_text(encoding="utf-8", errors="ignore")[:max_bytes]
                    else:
                        text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                files[rel_text] = text
        return files

    def _memory_readiness(self) -> str:
        user_exists = self.config.user_memory_path.exists()
        task_exists = self.config.task_memory_path.exists()
        if user_exists and task_exists:
            return "ready"
        missing = []
        if not user_exists:
            missing.append("user")
        if not task_exists:
            missing.append("task")
        return "missing: " + ", ".join(missing)

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
