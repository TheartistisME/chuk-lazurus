"""Workspace index readiness and JIT plan support."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .config import AdapterSessionMetadata


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
        }

    def jit(self) -> dict[str, Any]:
        manifest = {
            "schema": "david.workspace_index.v1",
            "workspace_root": str(self.workspace_root),
            "adapter_scope": self.adapter.scope(),
            "capture": self.plan()["capture"],
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

