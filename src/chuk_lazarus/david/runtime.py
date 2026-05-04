"""Offline David runtime pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from .agent_loop import (
    ActionPayload,
    AgentLoopResult,
    AgentLoopState,
    parse_agent_action,
    render_model_action_prompt,
    run_agent_loop as run_agent_loop_core,
)
from .config import AdapterSessionMetadata, DavidConfig
from .decoder import DecoderController, DecoderPlan
from .decoder_prior_store import DecoderPriorProductStore, DecoderPriorScope
from .indexing import IndexReadiness, WorkspaceIndex
from .live_indexer import LiveIndexer, LiveIndexRefresh, load_live_index_state
from .materializer import MaterializedContext, Materializer
from .materialization_replay import ReplayConsumerInput
from .memory import JsonlMemoryStore, MemoryBank
from .model_attestation import ManualModelAttestationResult, verify_model_attestation
from .model_artifacts import is_vindex_artifact_path
from .model_backend import (
    ModelBackend,
    ModelBackendResult,
    OfflineModelBackend,
    TransformersCausalLMBackend,
    VindexArtifactBackend,
)
from .patch_routing import DOC_SUFFIXES, SOURCE_SUFFIXES, is_protected_path, route_patch_targets
from .patching import PatchApplyDiagnostic, apply_patch_candidate, validate_patch_candidate
from .central_router_adapter import CentralRouterAdapter
from .product_router import ProductRoutePacket, ProductRouter
from .resume import SessionSnapshot, load_session_snapshot, save_session_snapshot, summarize_result
from .routing import CentralRouter, MethodDetector, RoutePacket
from .source_index import SourceIndexManifest, load_source_index
from .steering import DecoderSteeringPolicy, build_decoder_logits_processor
from .tools import LocalTools
from .torch_backend import TorchRuntimeModelBackend
from .verifier import VerificationResult, Verifier

try:
    from chuk_lazarus.harness.boot import boot_harness
except ImportError:  # pragma: no cover - only used if harness package is unavailable.
    boot_harness = None


DAVID_RESPONSE_SENTINEL = "David response:"
DEFAULT_GENERATION_CONTEXT_CHARS = 8_192
_LEGACY_GENERATION_INSTRUCTIONS = (
    "Respond with the next concise coding-agent action.",
    "Respond with the next concise coding-agent action",
    "Respond with the next concise coding-",
)


def _materialization_replay_metadata(model_result: ModelBackendResult | None) -> dict[str, Any] | None:
    if model_result is None:
        return None
    replay = model_result.metadata.get("materialization_replay")
    return replay if isinstance(replay, dict) else None


def _clean_runtime_model_text(text: str) -> tuple[str, dict[str, Any]]:
    raw_text = text
    cleaned = text.strip()
    sentinel_index = cleaned.rfind(DAVID_RESPONSE_SENTINEL)
    if sentinel_index >= 0:
        cleaned = cleaned[sentinel_index + len(DAVID_RESPONSE_SENTINEL) :].strip()
    lines = [
        line
        for line in cleaned.splitlines()
        if line.strip() and line.strip() not in _LEGACY_GENERATION_INSTRUCTIONS
    ]
    cleaned = "\n".join(lines).strip()
    metadata = {
        "cleaned": cleaned != raw_text,
        "sentinel": DAVID_RESPONSE_SENTINEL,
    }
    if metadata["cleaned"]:
        metadata["raw_model_text"] = raw_text
    return cleaned, metadata


def _truncate_generation_context(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    marker = "\n[... routed context truncated ...]"
    keep = max(0, max_chars - len(marker))
    return text[:keep].rstrip() + marker, True


def _generation_context_char_budget(max_route_tokens: int) -> int:
    token_budget_chars = max(0, int(max_route_tokens)) * 4
    if token_budget_chars <= 0:
        return DEFAULT_GENERATION_CONTEXT_CHARS
    return min(DEFAULT_GENERATION_CONTEXT_CHARS, token_budget_chars)


def _render_evidence_context(evidence: Sequence[dict[str, Any]]) -> tuple[str, int]:
    blocks: list[str] = []
    included = 0
    for ordinal, item in enumerate(evidence, start=1):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        path = str(item.get("path") or item.get("artifact_id") or f"evidence-{ordinal}")
        kind = str(item.get("kind") or "evidence")
        blocks.append(f"[{ordinal}] {kind}: {path}\n{text}")
        included += 1
    return "\n\n".join(blocks), included


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
    product_route: ProductRoutePacket | None = None
    model_result: ModelBackendResult | None = None
    source_index: dict[str, Any] | None = None
    live_index_refresh: dict[str, Any] | None = None
    decoder_prior: dict[str, Any] | None = None
    resume_snapshot: dict[str, Any] | None = None
    harness_session: dict[str, Any] | None = None
    writeback_verification: VerificationResult | None = None

    def to_json(self) -> dict[str, Any]:
        data = {
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
                "materialization_plan": self.materialized.materialization_plan,
                "materialization_replay": _materialization_replay_metadata(self.model_result),
            },
            "decoder": {
                "constraints": self.decoder.constraints,
                "prior_scope": self.decoder.prior_scope,
            },
            "verification": self.verification.to_json(),
            "writeback": self.writeback,
        }
        if self.writeback_verification is not None:
            data["writeback_verification"] = self.writeback_verification.to_json()
        if self.product_route is not None:
            data["product_route"] = self.product_route.to_json()
        if self.model_result is not None:
            data["model_result"] = {
                "text": self.model_result.text,
                "backend": self.model_result.backend,
                "ok": self.model_result.ok,
                "error": self.model_result.error,
                "metadata": self.model_result.metadata,
            }
        if self.source_index is not None:
            data["source_index"] = self.source_index
        if self.live_index_refresh is not None:
            data["live_index_refresh"] = self.live_index_refresh
        if self.decoder_prior is not None:
            data["decoder_prior"] = self.decoder_prior
        if self.resume_snapshot is not None:
            data["resume_snapshot"] = self.resume_snapshot
        if self.harness_session is not None:
            data["harness_session"] = self.harness_session
        return data


@dataclass(frozen=True)
class RuntimeAgentLoopResult:
    prompt: str
    loop: AgentLoopResult
    writeback: dict[str, Any] | None = None
    resume_snapshot: dict[str, Any] | None = None

    @property
    def answer(self) -> str:
        return (
            f"agent_loop status={self.loop.status}; steps={self.loop.steps}; "
            f"verified={self.loop.verified}; reason={self.loop.reason}"
        )

    @property
    def ok(self) -> bool:
        return self.loop.ok

    def to_json(self) -> dict[str, Any]:
        data = {
            "prompt": self.prompt,
            "answer": self.answer,
            "loop": self.loop.to_dict(),
            "writeback": self.writeback,
            "resume_snapshot": self.resume_snapshot,
        }
        return data


class DavidRuntime:
    def __init__(self, config: DavidConfig) -> None:
        self.config = config
        self.boot_errors: list[str] = []
        self.product_router_errors: list[str] = []
        self.harness_session = self._boot_session()
        self.adapter = self._adapter_from_harness() or config.adapter
        self.model_attestation = self._verify_model_attestation()
        self.index = WorkspaceIndex(config.workspace_root, self._index_manifest_path(), self.adapter)
        self.memory = MemoryBank(
            JsonlMemoryStore(config.user_memory_path, "user"),
            JsonlMemoryStore(config.task_memory_path, "task"),
        )
        self.detector = MethodDetector()
        self.router = CentralRouter()
        product_router = self._product_central_router()
        self.product_router = ProductRouter(
            router=product_router or self.router,
            detector=self.detector,
            proof_router_available=product_router is not None,
        )
        self.materializer = Materializer()
        self.decoder = DecoderController()
        self.tools = LocalTools(config.workspace_root)
        self.verifier = Verifier(self.tools)
        self.backend = self._create_backend()
        self.source_index_path = config.state_dir / "indexes" / f"{self._adapter_file_stem()}-source.json"
        self.live_index_state_path = config.state_dir / "indexes" / f"{self._adapter_file_stem()}-live-source-state.json"
        self.latest_live_index_refresh: LiveIndexRefresh | None = None
        self.decoder_prior_path = config.state_dir / "decoder_priors.json"
        self.decoder_prior_store = DecoderPriorProductStore.load(self.decoder_prior_path)
        self.resume_path = config.state_dir / "resume.json"
        self.resume_snapshot = self._load_resume_snapshot()
        self.auto_jit_summary = self.jit_index() if self.config.auto_jit_index else None

    @classmethod
    def create(cls, config: DavidConfig) -> "DavidRuntime":
        return cls(config)

    def run_once(self, prompt: str, *, verify_command: Sequence[str] | None = None) -> RuntimeResult:
        readiness = self.index.check()
        if readiness.required and self.config.auto_jit_index:
            self.jit_index()
            readiness = self.index.check()
        source_index = self._ensure_source_index() if self.config.auto_jit_index else self._loaded_source_index()

        method = self.detector.detect(prompt)
        workspace_files = self._workspace_text_files() if method in {"repo_patch", "source_dependency"} else {}
        evidence = self._recall(method, prompt, workspace_files=workspace_files)
        route = self.router.route(
            method=method,
            prompt=prompt,
            session_id=self.config.session_id,
            evidence=evidence,
            max_tokens=self.config.max_route_tokens,
        )
        product_route = self.product_router.route(
            prompt,
            session_id=self.config.session_id,
            evidence=evidence,
            files=workspace_files or None,
            source_index=source_index,
            method=method,
            max_tokens=self.config.max_route_tokens,
        )
        replay_consumer = self._backend_replay_consumer()
        materialized = self.materializer.materialize(route, self.adapter, replay_consumer=replay_consumer)
        decoder = self.decoder.plan(route=route, adapter=self.adapter, session_id=self.config.session_id)
        model_result = self._generate(prompt, method, product_route, materialized, decoder, replay_consumer)
        verification = self.verifier.verify(
            capability=method,
            evidence=evidence,
            command=verify_command if method == "verify" else None,
            metadata=self._verification_metadata(
                route=route,
                materialized=materialized,
                decoder=decoder,
                product_route=product_route,
                model_result=model_result,
            ),
        )
        answer = model_result.text if self.config.model_path and model_result.ok and model_result.text else self._answer(
            prompt, method, readiness, route, materialized, verification, model_result
        )
        decoder_prior = self._update_decoder_prior(method, decoder, verification, model_result)
        writeback_metadata = {
            "provenance": "david.runtime.run_once",
            **self._verification_metadata(
                route=route,
                materialized=materialized,
                decoder=decoder,
                product_route=product_route,
                model_result=model_result,
                decoder_prior=decoder_prior,
            ),
            "verification": verification.to_json(),
            "harness_session_id": getattr(self.harness_session, "session_id", None),
            "live_index_refresh": self._latest_live_index_refresh_json(),
        }
        artifact = self.memory.writeback(
            method=method,
            user_id=self.config.user_id,
            session_id=self.config.session_id,
            text=f"Prompt: {prompt}\nAnswer: {answer}",
            metadata=writeback_metadata,
        )
        writeback_verification = self.verifier.verify(
            capability=method,
            evidence=evidence,
            metadata={
                **writeback_metadata,
                "writeback": artifact.to_json(),
            },
        )
        result = RuntimeResult(
            prompt=prompt,
            method=method,
            answer=answer,
            index=readiness,
            route=route,
            materialized=materialized,
            decoder=decoder,
            verification=verification,
            writeback=artifact.to_json(),
            product_route=product_route,
            model_result=model_result,
            source_index=source_index.to_json() if source_index is not None else None,
            live_index_refresh=self._latest_live_index_refresh_json(),
            decoder_prior=decoder_prior,
            harness_session=self._harness_session_json(),
            writeback_verification=writeback_verification,
        )
        snapshot = self._save_resume_snapshot(result)
        return RuntimeResult(
            **{
                **result.__dict__,
                "resume_snapshot": snapshot.to_json(),
            }
        )

    def run_agent_loop(
        self,
        prompt: str | Sequence[ActionPayload],
        *,
        max_steps: int = 8,
        persist: bool = True,
    ) -> RuntimeAgentLoopResult:
        requests = self._agent_loop_requests(prompt)
        original_prompt = prompt if isinstance(prompt, str) else repr(list(prompt))
        if requests is None:
            loop = run_agent_loop_core(
                self._model_driven_agent_step(original_prompt),
                self.tools,
                max_steps=max_steps,
                objective=original_prompt,
                mode="model_driven",
            )
        else:
            loop = run_agent_loop_core(
                requests,
                self.tools,
                max_steps=max_steps,
                objective=original_prompt,
                mode="explicit",
            )
        writeback: dict[str, Any] | None = None
        if persist:
            artifact = self.memory.writeback(
                method="agent_loop",
                user_id=self.config.user_id,
                session_id=self.config.session_id,
                text=f"Agent loop: {original_prompt}\nStatus: {loop.status}\nReason: {loop.reason}",
                metadata={
                    "provenance": "david.runtime.run_agent_loop",
                    "adapter": self.adapter.scope(),
                    "adapter_scope": self.adapter.scope(),
                    "loop": loop.to_dict(),
                    "harness_session_id": getattr(self.harness_session, "session_id", None),
                    "live_index_refresh": self._latest_live_index_refresh_json(),
                },
            )
            writeback = artifact.to_json()
        result = RuntimeAgentLoopResult(prompt=original_prompt, loop=loop, writeback=writeback)
        snapshot = self._save_resume_snapshot(result)
        return RuntimeAgentLoopResult(
            prompt=original_prompt,
            loop=loop,
            writeback=writeback,
            resume_snapshot=snapshot.to_json(),
        )

    def agent_loop(
        self,
        prompt: str | Sequence[ActionPayload],
        *,
        max_steps: int = 8,
        persist: bool = True,
    ) -> RuntimeAgentLoopResult:
        return self.run_agent_loop(prompt, max_steps=max_steps, persist=persist)

    def readiness(self) -> dict[str, str]:
        index = self.index.check()
        backend = self.backend.status()
        source = self._loaded_source_index()
        return {
            "model validation": self._model_validation_status(),
            "backend": f"{backend.name}: {'loaded' if backend.loaded else backend.reason}",
            "index": "ready" if index.ready else f"missing: {index.reason}",
            "source index": self._source_index_status(source),
            "memory": self._memory_readiness(),
            "resume": "ready" if self.resume_snapshot is not None else "missing",
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
        source = self._loaded_source_index()
        source_line = (
            f"source index: {self._source_index_status(source)} at {self.source_index_path}"
            if source is not None
            else f"source index: missing at {self.source_index_path}"
        )
        if readiness.ready:
            return f"index: ready at {readiness.manifest_path}\n{source_line}"
        return f"index: {readiness.reason}\nplan: {readiness.jit_plan}\n{source_line}"

    def verify(self, command: str | None = None) -> str:
        command = command or getattr(self.config, "verify_command", None)
        if not command:
            result = self.run_once("Verify quality gate")
            return result.answer
        shell = self.run_shell(command)
        return shell

    def jit_index(self) -> str:
        self.index.jit()
        readiness = self.index.check()
        refresh = self._refresh_source_index()
        return (
            f"index: {'ready' if readiness.ready else readiness.reason} at {readiness.manifest_path}\n"
            f"source index: {refresh.file_count} files at {self.source_index_path}\n"
            "source refresh: "
            f"changed={len(refresh.changed_paths)} indexed={len(refresh.indexed_paths)} "
            f"deleted={len(refresh.deleted_paths)} manifest_sha256={refresh.manifest_sha256}"
        )

    def build_index(self) -> str:
        return self.jit_index()

    def refresh_index(self) -> str:
        return self.jit_index()

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

    def _agent_loop_requests(self, prompt: str | Sequence[ActionPayload]) -> Sequence[ActionPayload] | None:
        if not isinstance(prompt, str):
            return prompt

        stripped = prompt.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        else:
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
            return [prompt]

        try:
            if parse_agent_action(stripped) is not None:
                return [prompt]
        except Exception:
            return [prompt]
        return None

    def _model_driven_agent_step(self, objective: str):
        action_context = self._model_action_workspace_context(objective)

        def next_action(state: AgentLoopState) -> ActionPayload:
            model_prompt = render_model_action_prompt(state)
            if action_context["included"]:
                model_prompt = (
                    f"{model_prompt}\n"
                    "Routed workspace context:\n"
                    f"{action_context['text']}\n"
                    "Use this bounded workspace context as file/path evidence when choosing read, write, or verify actions."
                )
            model_result = self.backend.generate(model_prompt, max_new_tokens=320)
            if not model_result.ok:
                raise RuntimeError(f"backend refused action generation: {model_result.error or 'unknown error'}")

            raw_text = model_result.text.strip()
            try:
                action = parse_agent_action(raw_text)
            except Exception:
                return raw_text
            if action is None:
                return raw_text
            payload = action.to_dict()
            payload["_model_provenance"] = {
                "backend": model_result.backend,
                "ok": model_result.ok,
                "error": model_result.error,
                "metadata": model_result.metadata,
                "prompt": model_prompt,
                "objective": objective,
                "workspace_context": action_context["metadata"],
            }
            payload["_model_text"] = raw_text
            return payload

        return next_action

    def _model_action_workspace_context(self, objective: str) -> dict[str, Any]:
        method = self.detector.detect(objective)
        budget = _generation_context_char_budget(self.config.max_route_tokens)
        metadata: dict[str, Any] = {
            "source": "none",
            "method": method,
            "context_char_count": 0,
            "context_char_budget": budget,
            "workspace_file_count": 0,
            "evidence_count_available": 0,
            "evidence_count_included": 0,
            "selected_paths": [],
            "omitted": False,
            "omitted_reason": None,
            "truncated": False,
        }
        if method not in {"repo_patch", "source_dependency"}:
            metadata.update({"omitted": True, "omitted_reason": "non_workspace_method"})
            return {"included": False, "text": "", "metadata": metadata}

        workspace_files = self._workspace_text_files(max_files=40, max_bytes=6_000)
        metadata["workspace_file_count"] = len(workspace_files)
        evidence = self._recall(method, objective, workspace_files=workspace_files)
        metadata["evidence_count_available"] = len(evidence)
        metadata["selected_paths"] = [
            str(item.get("path"))
            for item in evidence
            if item.get("path") and not is_protected_path(str(item.get("path")))
        ][:8]
        raw_context, evidence_included = _render_evidence_context(evidence[:8])
        metadata["evidence_count_included"] = evidence_included
        if not raw_context:
            metadata.update({"omitted": True, "omitted_reason": "no_context_available"})
            return {"included": False, "text": "", "metadata": metadata}

        text, truncated = _truncate_generation_context(raw_context, budget)
        metadata.update(
            {
                "source": "workspace_route_evidence",
                "context_char_count": len(text),
                "truncated": truncated,
            }
        )
        return {"included": True, "text": text, "metadata": metadata}

    def _recall(
        self,
        method: str,
        prompt: str,
        *,
        workspace_files: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        if method == "symbolic_multi_hop":
            return self.memory.symbolic_chain(prompt)
        if method == "temporal_recall":
            latest = self.memory.user.temporal(prompt, ordinal="latest")
            return [latest] if latest else []
        evidence = self.memory.recall_for_method(method, prompt)
        if method in {"repo_patch", "source_dependency"}:
            evidence = [*evidence, *self._workspace_route_evidence(prompt, workspace_files=workspace_files)]
        return evidence

    def _workspace_route_evidence(
        self,
        prompt: str,
        *,
        workspace_files: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        files = workspace_files if workspace_files is not None else self._workspace_text_files()
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

    def _boot_session(self) -> Any | None:
        if boot_harness is None:
            self.boot_errors.append("harness boot module unavailable")
            return None
        if not (self.config.model_path or self.config.validation_report_path):
            return None
        require_validated_model = self.config.require_validated_model
        if self.config.model_attestation_path and self.config.require_validated_model:
            require_validated_model = False
        try:
            return boot_harness(
                model_path=self.config.model_path or "offline-deterministic",
                workspace_path=str(self.config.workspace_root),
                validation_report_path=self.config.validation_report_path,
                require_validated_model=require_validated_model,
            )
        except Exception as exc:
            self.boot_errors.append(f"{type(exc).__name__}: {exc}")
            return None

    def _adapter_from_harness(self) -> AdapterSessionMetadata | None:
        session = self.harness_session
        model_adapter = getattr(session, "model_adapter", None)
        if model_adapter is None:
            return None
        return AdapterSessionMetadata(
            model_id=str(model_adapter.model_identity),
            tokenizer_id=str(model_adapter.tokenizer_identity),
            model_revision=str(model_adapter.model_revision or "validated"),
            adapter_family=str(model_adapter.adapter_family),
            hidden_size=model_adapter.hidden_size,
            route_layer=model_adapter.route_layer_candidate,
            route_query_head=model_adapter.route_query_head_candidate,
            boundary_layer=model_adapter.boundary_layer_candidate,
            kv_source_layer=model_adapter.kv_source_layer_candidate,
            kv_target_layer=model_adapter.kv_target_layer_candidate,
            insertion_family=str(model_adapter.insertion_family or "text_only"),
            memory_family="david-runtime",
        )

    def _create_backend(self) -> ModelBackend:
        if self.config.model_path and is_vindex_artifact_path(self.config.model_path):
            return VindexArtifactBackend(self.config.model_path)
        can_auto_load = bool(getattr(self.harness_session, "can_auto_load", False))
        can_standard_decode = bool(
            self.config.require_validated_model
            and (can_auto_load or self._manual_review_standard_decode_allowed())
        )
        if self.config.model_path and can_standard_decode and not self.boot_errors:
            backend_selector = str(self.config.model_backend or "transformers").strip().lower()
            if backend_selector in {"torch-runtime", "torch_runtime", "torch"}:
                return TorchRuntimeModelBackend(
                    self.config.model_path,
                    local_files_only=True,
                    device=self.config.model_device,
                    torch_dtype=self.config.model_dtype,
                )
            if backend_selector not in {"transformers", "transformers-causal-lm", "hf"}:
                self.boot_errors.append(f"unsupported model backend selector: {self.config.model_backend}")
                return OfflineModelBackend(prefix="david")
            return TransformersCausalLMBackend(
                self.config.model_path,
                local_files_only=True,
                device=self.config.model_device,
                torch_dtype=self.config.model_dtype,
            )
        return OfflineModelBackend(prefix="david")

    def _product_central_router(self) -> CentralRouterAdapter | None:
        try:
            return CentralRouterAdapter.from_stable_wrapper_if_available()
        except Exception as exc:
            self.product_router_errors.append(f"{type(exc).__name__}: {exc}")
            return None

    def _model_validation_status(self) -> str:
        if self.boot_errors:
            return "blocked: " + "; ".join(self.boot_errors)
        if self._manual_review_standard_decode_allowed():
            return f"manual_reviewed ({self.adapter.adapter_family}:{self.adapter.model_id}; standard_decode_only)"
        if self.harness_session is not None:
            status = getattr(self.harness_session, "validation_status", None) or "unknown"
            return f"{status} ({self.adapter.adapter_family}:{self.adapter.model_id})"
        return f"offline shell mode ({self.adapter.adapter_family}:{self.adapter.model_id})"

    def _verify_model_attestation(self) -> ManualModelAttestationResult | None:
        if not self.config.model_attestation_path:
            return None
        result = verify_model_attestation(
            self.config.model_attestation_path,
            validation_report_path=self.config.validation_report_path,
            model_path=self.config.model_path,
        )
        can_auto_load = bool(getattr(self.harness_session, "can_auto_load", False))
        if self.config.require_validated_model and not can_auto_load and not result.standard_decode_allowed:
            self.boot_errors.append(f"model attestation invalid: {result.reason}")
        return result

    def _manual_review_standard_decode_allowed(self) -> bool:
        return bool(
            self.config.require_validated_model
            and self.model_attestation is not None
            and self.model_attestation.standard_decode_allowed
        )

    def _index_manifest_path(self) -> Path:
        return self.config.state_dir / "indexes" / f"{self._adapter_file_stem()}.json"

    def _adapter_file_stem(self) -> str:
        safe_model = self.adapter.model_id.replace("/", "_").replace("\\", "_")
        safe_revision = self.adapter.model_revision.replace("/", "_").replace("\\", "_")
        return f"{safe_model}-{safe_revision}"

    def _loaded_source_index(self) -> SourceIndexManifest | None:
        try:
            return load_source_index(self.source_index_path)
        except (OSError, ValueError):
            return None

    def _ensure_source_index(self) -> SourceIndexManifest:
        existing = self._loaded_source_index()
        if existing is not None and existing.adapter_scope == self.adapter.scope():
            return existing
        refresh = self._refresh_source_index()
        manifest = self._loaded_source_index()
        if manifest is None:
            raise RuntimeError(f"live source index refresh did not write {refresh.source_index_path}")
        return manifest

    def _refresh_source_index(self) -> LiveIndexRefresh:
        refresh = LiveIndexer(
            self.config.workspace_root,
            self.adapter.scope(),
            state_path=self.live_index_state_path,
            source_index_path=self.source_index_path,
            max_files=120,
            max_file_bytes=128_000,
        ).refresh()
        self.latest_live_index_refresh = refresh
        return refresh

    def _latest_live_index_refresh_json(self) -> dict[str, Any] | None:
        if self.latest_live_index_refresh is None:
            return None
        return self.latest_live_index_refresh.to_session_refresh_handle()

    def _source_index_status(self, source: SourceIndexManifest | None = None) -> str:
        source = self._loaded_source_index() if source is None else source
        if source is None:
            return "missing"
        parts = [f"ready ({len(source.files)} files)"]
        if source.adapter_scope != self.adapter.scope():
            parts.append("stale: adapter scope mismatch")
        elif self.live_index_state_path.exists():
            try:
                state = load_live_index_state(self.live_index_state_path)
                parts.append(f"live indexed_at={state.indexed_at or 'unknown'}")
            except (OSError, ValueError):
                parts.append("live state unreadable")
        else:
            parts.append("live state missing")
        if self.latest_live_index_refresh is not None:
            parts.append(f"manifest_sha256={self.latest_live_index_refresh.manifest_sha256}")
        elif source.indexed_at:
            parts.append(f"source indexed_at={source.indexed_at}")
        if source.truncated:
            parts.append("truncated")
        return "; ".join(parts)

    def _load_resume_snapshot(self) -> SessionSnapshot | None:
        try:
            return load_session_snapshot(self.resume_path)
        except (OSError, ValueError):
            return None

    def _save_resume_snapshot(self, result: Any) -> SessionSnapshot:
        live_index_refresh = getattr(result, "live_index_refresh", None)
        summary_input: Any = (
            result.answer
            if live_index_refresh is None
            else {"answer": result.answer, "live_index_refresh": live_index_refresh}
        )
        snapshot = SessionSnapshot(
            session_id=self.config.session_id,
            workspace=str(self.config.workspace_root),
            adapter_scope=self.adapter.scope(),
            memory_paths={
                "user": str(self.config.user_memory_path),
                "task": str(self.config.task_memory_path),
                "decoder_prior": str(self.decoder_prior_path),
                "source_index": str(self.source_index_path),
                "live_index_state": str(self.live_index_state_path),
            },
            last_result_summary=summarize_result(summary_input),
        )
        self.resume_snapshot = save_session_snapshot(snapshot, self.resume_path)
        return self.resume_snapshot

    def _generate(
        self,
        prompt: str,
        method: str,
        product_route: ProductRoutePacket,
        materialized: MaterializedContext,
        decoder: DecoderPlan,
        replay_consumer: ReplayConsumerInput = None,
    ) -> ModelBackendResult:
        context = self._generation_context(product_route, materialized)
        context_block = ""
        if context["included"]:
            context_block = (
                "Routed context:\n"
                f"{context['text']}\n"
                "Use the routed context above as grounding evidence. If it is insufficient, say what is missing.\n"
            )
        generation_prompt = (
            "You are David, a terminal coding agent operating inside the user's workspace.\n"
            "Use the routed methodology and evidence to produce the smallest useful next answer.\n"
            f"Task: {prompt}\n"
            f"Methodology: {method}\n"
            f"Capability: {product_route.capability}\n"
            f"Evidence count: {len(product_route.evidence)}\n"
            f"{context_block}"
            "Answer only in the response slot below. Do not repeat this prompt, labels, or instructions.\n"
            f"{DAVID_RESPONSE_SENTINEL}"
        )
        steering = self._decoder_steering_processor(decoder)
        model_result = self.backend.generate(
            generation_prompt,
            max_new_tokens=self.config.model_max_new_tokens,
            logits_processor=steering["processors"],
            materialization_plan=materialized.materialization_plan,
            replay_consumer=replay_consumer,
        )
        cleaned_text, postprocess_metadata = _clean_runtime_model_text(model_result.text)
        return ModelBackendResult(
            text=cleaned_text,
            backend=model_result.backend,
            ok=model_result.ok,
            error=model_result.error,
            metadata={
                **model_result.metadata,
                "decoder_steering": steering["metadata"],
                "generation_context": context["metadata"],
                "answer_postprocess": postprocess_metadata,
            },
        )

    def _generation_context(
        self,
        product_route: ProductRoutePacket,
        materialized: MaterializedContext,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "context_char_count": 0,
            "context_char_budget": _generation_context_char_budget(self.config.max_route_tokens),
            "evidence_count_available": len(product_route.evidence),
            "evidence_count_included": 0,
            "omitted": False,
            "omitted_reason": None,
            "refused": materialized.refused,
            "truncated": False,
            "source": "none",
        }
        if materialized.refused:
            metadata.update({"omitted": True, "omitted_reason": "materializer_refused"})
            return {"included": False, "text": "", "metadata": metadata}

        raw_context = materialized.text_context.strip()
        if raw_context:
            metadata["source"] = "materialized.text_context"
            metadata["evidence_count_included"] = len(product_route.evidence)
        else:
            raw_context, evidence_included = _render_evidence_context(product_route.evidence)
            metadata["source"] = "product_route.evidence" if raw_context else "none"
            metadata["evidence_count_included"] = evidence_included

        if not raw_context:
            metadata.update({"omitted": True, "omitted_reason": "no_context_available"})
            return {"included": False, "text": "", "metadata": metadata}

        text, truncated = _truncate_generation_context(raw_context, int(metadata["context_char_budget"]))
        metadata["context_char_count"] = len(text)
        metadata["truncated"] = truncated
        return {"included": True, "text": text, "metadata": metadata}

    def _backend_replay_consumer(self) -> ReplayConsumerInput:
        reporter = getattr(self.backend, "replay_consumer_capabilities", None)
        if not callable(reporter):
            return None
        try:
            return reporter(self.adapter)
        except Exception:
            return None

    def _decoder_steering_processor(self, decoder: DecoderPlan) -> dict[str, Any]:
        steering = decoder.constraints.get("steering")
        metadata: dict[str, Any] = {
            "attempted": bool(steering),
            "applied": False,
            "processor_count": 0,
            "refused_reason": None,
        }
        if isinstance(steering, dict):
            metadata.update(
                {
                    "target_language": steering.get("target_language"),
                    "task_type": steering.get("task_type"),
                    "logit_lock": bool(steering.get("logit_lock")),
                    "forbidden_token_families": list(steering.get("forbidden_token_families") or ()),
                }
            )
        else:
            metadata["refused_reason"] = "decoder plan has no steering metadata"
            return {"processors": None, "metadata": metadata}

        if not isinstance(self.backend, TransformersCausalLMBackend):
            metadata["refused_reason"] = "backend does not expose tokenizer for live steering"
            return {"processors": None, "metadata": metadata}

        status = self.backend.load()
        if not status.loaded:
            metadata["refused_reason"] = f"backend not loaded: {status.reason}"
            return {"processors": None, "metadata": metadata}
        tokenizer = self.backend.tokenizer
        if tokenizer is None:
            metadata["refused_reason"] = "backend tokenizer unavailable"
            return {"processors": None, "metadata": metadata}

        alpha_bounds = steering.get("alpha_bounds") if isinstance(steering.get("alpha_bounds"), dict) else {}
        try:
            policy = DecoderSteeringPolicy(
                task_type=str(
                    steering.get("task_type")
                    or decoder.prior_scope.get("task_type")
                    or decoder.prior_scope.get("method")
                    or "unknown"
                ),
                target_language=str(steering.get("target_language") or "unknown"),
                forbidden_token_families=tuple(str(item) for item in steering.get("forbidden_token_families") or ()),
                alpha_min=float(alpha_bounds.get("min", 0.0)),
                alpha_max=float(alpha_bounds.get("max", 0.0)),
                logit_lock=bool(steering.get("logit_lock")),
                steering_version=str(
                    steering.get("policy")
                    or decoder.prior_scope.get("steering_version")
                    or "unknown"
                ),
            )
            processor = build_decoder_logits_processor(
                policy=policy,
                adapter=self.adapter,
                tokenizer=tokenizer,
                scope=decoder.prior_scope,
            )
        except Exception as exc:
            metadata["refused_reason"] = f"{type(exc).__name__}: {exc}"
            return {"processors": None, "metadata": metadata}

        metadata.update(
            {
                "applied": True,
                "processor_count": 1,
                "refused_reason": None,
                "forbidden_token_count": len(processor.forbidden_token_ids),
            }
        )
        return {"processors": [processor], "metadata": metadata}

    def _verification_metadata(
        self,
        *,
        route: RoutePacket,
        materialized: MaterializedContext,
        decoder: DecoderPlan,
        product_route: ProductRoutePacket,
        model_result: ModelBackendResult,
        decoder_prior: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        materialized_json = {
            "strategy": materialized.strategy,
            "text_context": materialized.text_context,
            "compatibility": (
                materialized.compatibility if route.evidence or materialized.strategy != "none" else None
            ),
            "refused": materialized.refused,
            "reason": materialized.reason,
            "materialization_plan": materialized.materialization_plan,
        }
        materialization_replay = _materialization_replay_metadata(model_result)
        if materialization_replay is not None:
            materialized_json["materialization_replay"] = materialization_replay
        return {
            "adapter": self.adapter.scope(),
            "adapter_scope": self.adapter.scope(),
            "route": route.to_json(),
            "route_evidence_chain": [
                {
                    "artifact_id": item.get("artifact_id"),
                    "path": item.get("path"),
                    "kind": item.get("kind"),
                    "ordinal": item.get("ordinal"),
                    "provenance": item.get("provenance"),
                    "route_reason": item.get("route_reason"),
                }
                for item in route.evidence
            ],
            "product_route": product_route.to_json(),
            "materialized": materialized_json,
            "materialization": materialized_json,
            "compatibility": (
                materialized.compatibility if route.evidence or materialized.strategy != "none" else None
            ),
            "decoder": {
                "constraints": decoder.constraints,
                "prior_scope": decoder.prior_scope,
            },
            "decoder_prior_scope": decoder.prior_scope,
            "decoder_prior": decoder_prior,
            "model_attestation": self._model_attestation_json(),
            "backend": {
                "name": model_result.backend,
                "ok": model_result.ok,
                "error": model_result.error,
                "metadata": model_result.metadata,
            },
        }

    def _update_decoder_prior(
        self,
        method: str,
        decoder: DecoderPlan,
        verification: VerificationResult,
        model_result: ModelBackendResult,
    ) -> dict[str, Any]:
        layer = self.adapter.kv_target_layer or self.adapter.boundary_layer or self.adapter.route_layer or 0
        scope = DecoderPriorScope(
            model_id=self.adapter.model_id,
            tokenizer_id=self.adapter.tokenizer_id,
            adapter_family=self.adapter.adapter_family,
            layer=int(layer),
            task_type=method,
            steering_version="david-decoder-v1",
            model_revision=self.adapter.model_revision,
            adapter_config_id=str(decoder.prior_scope.get("adapter_config_id") or ""),
            insertion_family=self.adapter.insertion_family,
        )
        record = self.decoder_prior_store.update(
            scope,
            accepted=verification.ok,
            steering_applied=self._live_steering_applied(model_result),
        )
        self.decoder_prior_store.save()
        return record.to_json()

    @staticmethod
    def _live_steering_applied(model_result: ModelBackendResult) -> bool:
        steering = model_result.metadata.get("decoder_steering")
        if not isinstance(steering, dict):
            return False
        return bool(steering.get("applied"))

    def _harness_session_json(self) -> dict[str, Any] | None:
        if self.harness_session is None:
            return None
        if hasattr(self.harness_session, "to_dict"):
            data = self.harness_session.to_dict()
            data["model_attestation"] = self._model_attestation_json()
            return data
        if hasattr(self.harness_session, "__dict__"):
            data = dict(self.harness_session.__dict__)
            data["model_attestation"] = self._model_attestation_json()
            return data
        return {"repr": repr(self.harness_session), "model_attestation": self._model_attestation_json()}

    def _model_attestation_json(self) -> dict[str, Any] | None:
        if self.model_attestation is None:
            return None
        return self.model_attestation.to_dict()

    def _answer(
        self,
        prompt: str,
        method: str,
        readiness: IndexReadiness,
        route: RoutePacket,
        materialized: MaterializedContext,
        verification: VerificationResult,
        model_result: ModelBackendResult | None = None,
    ) -> str:
        parts = [f"method={method}", f"tier={route.tier}", f"evidence={len(route.evidence)}"]
        if readiness.required:
            parts.append(f"jit_required={readiness.reason}")
        if materialized.refused:
            parts.append(f"materializer_refused={materialized.reason}")
        if method == "verify":
            parts.append(f"verification={'passed' if verification.ok else 'failed'}")
        if model_result is not None and not model_result.ok:
            parts.append(f"backend_blocked={model_result.error}")
        if self.config.model_tool_protocol:
            parts.append("tool_protocol=available")
        parts.append(f"summary={prompt[:120]}")
        return "; ".join(parts)
