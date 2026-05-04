"""Workspace index readiness and JIT plan support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .config import AdapterSessionMetadata


WORKSPACE_INDEX_SCHEMA = "david.workspace_index.v1"
READINESS_MANIFEST_SCHEMA = "david.index_readiness_manifest.v1"
READINESS_MANIFEST_VERSION = 1

ARTIFACT_FAMILY_LEXICAL_SOURCE = "lexical_source"
ARTIFACT_FAMILY_ACTIVATION_ROUTE = "activation_route"
ARTIFACT_FAMILY_DENSE_VECTOR = "dense_vector"
ARTIFACT_FAMILY_BOUNDARY_RESIDUAL = "boundary_residual"
ARTIFACT_FAMILY_RESIDUAL_STREAM = "residual_stream"
ARTIFACT_FAMILY_KV_CACHE = "kv_cache"
ARTIFACT_FAMILIES = (
    ARTIFACT_FAMILY_LEXICAL_SOURCE,
    ARTIFACT_FAMILY_ACTIVATION_ROUTE,
    ARTIFACT_FAMILY_DENSE_VECTOR,
    ARTIFACT_FAMILY_BOUNDARY_RESIDUAL,
    ARTIFACT_FAMILY_RESIDUAL_STREAM,
    ARTIFACT_FAMILY_KV_CACHE,
)
CAPTURE_REQUIRED_FAMILIES = tuple(
    family for family in ARTIFACT_FAMILIES if family != ARTIFACT_FAMILY_LEXICAL_SOURCE
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class IndexArtifactReadiness:
    family: str
    ready: bool = False
    required: bool = True
    artifact_ref: str | None = None
    manifest_ref: str | None = None
    sidecar_refs: dict[str, str] | None = None
    window_count: int = 0
    missing_window_ids: list[str] | None = None
    stale_window_ids: list[str] | None = None
    capture_action: str = "capture"
    status: str = "missing"

    def to_json(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "ready": self.ready,
            "required": self.required,
            "artifact_ref": self.artifact_ref,
            "manifest_ref": self.manifest_ref,
            "sidecar_refs": dict(self.sidecar_refs or {}),
            "window_count": self.window_count,
            "missing_window_ids": list(self.missing_window_ids or []),
            "stale_window_ids": list(self.stale_window_ids or []),
            "capture_action": self.capture_action,
            "status": self.status,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "IndexArtifactReadiness":
        return cls(
            family=str(data["family"]),
            ready=bool(data.get("ready")),
            required=bool(data.get("required", True)),
            artifact_ref=_optional_str(data.get("artifact_ref")),
            manifest_ref=_optional_str(data.get("manifest_ref")),
            sidecar_refs={str(key): str(value) for key, value in dict(data.get("sidecar_refs") or {}).items()},
            window_count=int(data.get("window_count") or 0),
            missing_window_ids=[str(item) for item in data.get("missing_window_ids", [])],
            stale_window_ids=[str(item) for item in data.get("stale_window_ids", [])],
            capture_action=str(data.get("capture_action") or "capture"),
            status=str(data.get("status") or "missing"),
        )


@dataclass(frozen=True)
class IndexReadinessManifest:
    workspace_root: str
    adapter_scope: dict[str, Any]
    artifact_families: dict[str, IndexArtifactReadiness]
    schema: str = READINESS_MANIFEST_SCHEMA
    version: int = READINESS_MANIFEST_VERSION
    indexed_at: str = ""
    provenance: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "indexed_at": self.indexed_at or utc_now(),
            "workspace_root": self.workspace_root,
            "adapter_scope": self.adapter_scope,
            "artifact_families": {
                family: readiness.to_json()
                for family, readiness in sorted(self.artifact_families.items())
            },
            "provenance": dict(self.provenance or {}),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "IndexReadinessManifest":
        if data.get("schema") != READINESS_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported readiness manifest schema: {data.get('schema')!r}")
        families = {
            str(family): IndexArtifactReadiness.from_json(dict(payload))
            for family, payload in dict(data.get("artifact_families") or {}).items()
        }
        return cls(
            workspace_root=str(data["workspace_root"]),
            adapter_scope=dict(data.get("adapter_scope") or {}),
            artifact_families=_with_default_families(families),
            version=int(data.get("version") or READINESS_MANIFEST_VERSION),
            indexed_at=str(data.get("indexed_at") or ""),
            provenance=dict(data.get("provenance") or {}),
        )

    @classmethod
    def create(
        cls,
        workspace_root: Path,
        adapter_scope: dict[str, Any],
        *,
        lexical_manifest_ref: str | None = None,
        lexical_window_count: int = 0,
        provenance: dict[str, Any] | None = None,
    ) -> "IndexReadinessManifest":
        families = _with_default_families({})
        families[ARTIFACT_FAMILY_LEXICAL_SOURCE] = IndexArtifactReadiness(
            family=ARTIFACT_FAMILY_LEXICAL_SOURCE,
            ready=lexical_manifest_ref is not None,
            required=True,
            manifest_ref=lexical_manifest_ref,
            window_count=lexical_window_count,
            capture_action="index_source",
            status="ready" if lexical_manifest_ref is not None else "missing",
        )
        return cls(
            workspace_root=str(Path(workspace_root).resolve()),
            adapter_scope=dict(adapter_scope),
            artifact_families=families,
            indexed_at=utc_now(),
            provenance=dict(provenance or {"source": "chuk_lazarus.david.indexing"}),
        )


@dataclass(frozen=True)
class IndexReadiness:
    ready: bool
    required: bool
    manifest_path: Path
    reason: str
    jit_plan: dict[str, Any] | None = None


class WorkspaceIndex:
    def __init__(self, workspace_root: Path, manifest_path: Path, adapter: AdapterSessionMetadata) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.manifest_path = Path(manifest_path)
        self.adapter = adapter

    def check(self) -> IndexReadiness:
        if not self.manifest_path.exists():
            return IndexReadiness(
                ready=False,
                required=True,
                manifest_path=self.manifest_path,
                reason="missing index manifest for adapter scope",
                jit_plan=self.plan(),
            )
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if data.get("adapter_scope") != self.adapter.scope():
            return IndexReadiness(
                ready=False,
                required=True,
                manifest_path=self.manifest_path,
                reason="index adapter scope mismatch",
                jit_plan=self.plan(),
            )
        return IndexReadiness(True, False, self.manifest_path, "index ready")

    def plan(self) -> dict[str, Any]:
        return {
            "action": "jit_index_workspace",
            "workspace_root": str(self.workspace_root),
            "manifest_path": str(self.manifest_path),
            "adapter_scope": self.adapter.scope(),
            "capture": ["activation_routes", "boundary_residuals", "metadata", "provenance"],
            "artifact_families": list(ARTIFACT_FAMILIES),
            "required_capture_families": list(CAPTURE_REQUIRED_FAMILIES),
        }

    def jit(self) -> dict[str, Any]:
        readiness = IndexReadinessManifest.create(
            self.workspace_root,
            self.adapter.scope(),
            lexical_manifest_ref=None,
            provenance={
                "source": "chuk_lazarus.david.indexing.WorkspaceIndex.jit",
                "legacy_schema": WORKSPACE_INDEX_SCHEMA,
            },
        )
        manifest = {
            **readiness.to_json(),
            "capture": self.plan()["capture"],
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest


def load_readiness_manifest(path: Path) -> IndexReadinessManifest:
    return IndexReadinessManifest.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def _with_default_families(
    families: dict[str, IndexArtifactReadiness],
) -> dict[str, IndexArtifactReadiness]:
    output = dict(families)
    for family in ARTIFACT_FAMILIES:
        output.setdefault(family, IndexArtifactReadiness(family=family))
    return output


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
