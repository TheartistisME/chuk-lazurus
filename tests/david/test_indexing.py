from __future__ import annotations

import json
from pathlib import Path

from chuk_lazarus.david.config import AdapterSessionMetadata
from chuk_lazarus.david.indexing import (
    ARTIFACT_FAMILIES,
    ARTIFACT_FAMILY_BOUNDARY_RESIDUAL,
    ARTIFACT_FAMILY_LEXICAL_SOURCE,
    IndexArtifactReadiness,
    IndexReadinessManifest,
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


def test_readiness_manifest_round_trips_sidecar_catalog_refs(tmp_path: Path) -> None:
    adapter = AdapterSessionMetadata(model_id="offline")
    manifest_path = tmp_path / ".david" / "indexes" / "workspace.json"
    readiness = IndexReadinessManifest.create(tmp_path, adapter.scope())
    readiness.artifact_families[ARTIFACT_FAMILY_BOUNDARY_RESIDUAL] = IndexArtifactReadiness(
        family=ARTIFACT_FAMILY_BOUNDARY_RESIDUAL,
        ready=False,
        required=True,
        sidecar_refs={"hot-window-1": "sidecars/hot-window-1.boundary.json"},
        window_count=2,
        missing_window_ids=["hot-window-2"],
        stale_window_ids=["hot-window-1"],
        capture_action="recapture_hot_windows",
        status="stale",
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(readiness.to_json(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    loaded = load_readiness_manifest(manifest_path)
    boundary = loaded.artifact_families[ARTIFACT_FAMILY_BOUNDARY_RESIDUAL]

    assert boundary.sidecar_refs == {"hot-window-1": "sidecars/hot-window-1.boundary.json"}
    assert boundary.missing_window_ids == ["hot-window-2"]
    assert boundary.stale_window_ids == ["hot-window-1"]
    assert boundary.capture_action == "recapture_hot_windows"
