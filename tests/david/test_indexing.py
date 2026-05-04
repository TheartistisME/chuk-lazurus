from __future__ import annotations

from pathlib import Path

from chuk_lazarus.david.config import AdapterSessionMetadata
from chuk_lazarus.david.indexing import (
    ARTIFACT_FAMILIES,
    ARTIFACT_FAMILY_LEXICAL_SOURCE,
    READINESS_MANIFEST_SCHEMA,
    READINESS_MANIFEST_VERSION,
    WorkspaceIndex,
    load_readiness_manifest,
)


def test_workspace_jit_writes_versioned_readiness_manifest(tmp_path: Path) -> None:
    adapter = AdapterSessionMetadata(model_id="offline")
    manifest_path = tmp_path / ".david" / "indexes" / "workspace.json"
    workspace_index = WorkspaceIndex(tmp_path, manifest_path, adapter)

    missing = workspace_index.check()
    manifest = workspace_index.jit()
    ready = workspace_index.check()
    loaded = load_readiness_manifest(manifest_path)

    assert missing.ready is False
    assert missing.required is True
    assert ready.ready is True
    assert manifest["schema"] == READINESS_MANIFEST_SCHEMA
    assert manifest["version"] == READINESS_MANIFEST_VERSION
    assert set(manifest["artifact_families"]) == set(ARTIFACT_FAMILIES)
    assert loaded.adapter_scope == adapter.scope()
    assert loaded.artifact_families[ARTIFACT_FAMILY_LEXICAL_SOURCE].capture_action == "index_source"
    assert loaded.artifact_families[ARTIFACT_FAMILY_LEXICAL_SOURCE].status == "missing"
