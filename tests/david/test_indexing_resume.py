from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tests.david import require_attr, require_module, value_at


def _scope() -> dict[str, str]:
    return {
        "model_identity": "gemma-e2b-test",
        "tokenizer_identity": "gemma-tokenizer-test",
        "adapter_config_id": "gemma-e2b-test:kv-direct",
        "adapter_family": "gemma",
    }


def test_missing_workspace_index_reports_planned_jit_actions_without_mutation(
    tmp_path: Path,
) -> None:
    indexing = require_module("chuk_lazarus.david.indexing")
    check_workspace_index = require_attr(
        indexing,
        "check_workspace_index",
        "David workspace JIT readiness checks",
    )

    workspace = tmp_path / "workspace"
    readiness = check_workspace_index(workspace, _scope())

    assert value_at(readiness, "ready") is False
    assert value_at(readiness, "jit_required") is True
    assert value_at(readiness, "state") == "missing"
    actions = set(value_at(readiness, "planned_actions", ()))
    assert "create_david_workspace_root" in actions
    assert "create_workspace_task_memory_root" in actions
    assert "jit_index_workspace" in actions
    assert "write_workspace_index_manifest" in actions
    assert not (workspace / ".chuk_lazarus").exists(), "readiness checks must not mutate"


def test_workspace_index_manifest_compatibility_and_model_scope_mismatch(
    tmp_path: Path,
) -> None:
    indexing = require_module("chuk_lazarus.david.indexing")
    write_manifest = require_attr(
        indexing,
        "write_workspace_index_manifest",
        "creating David workspace index manifests",
    )
    check_workspace_index = require_attr(
        indexing,
        "check_workspace_index",
        "checking manifest compatibility",
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest_path = write_manifest(workspace, **_scope())

    assert manifest_path == workspace / ".chuk_lazarus" / "david" / "index" / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["schema_name"] == indexing.WORKSPACE_INDEX_SCHEMA_NAME
    assert manifest_payload["provenance"]["benchmark_artifact"] is False

    ready = check_workspace_index(workspace, _scope())
    assert value_at(ready, "ready") is True
    assert value_at(ready, "jit_required") is False
    assert value_at(ready, "planned_actions") == ()

    mismatched_scope = {**_scope(), "adapter_config_id": "other-adapter"}
    mismatch = check_workspace_index(workspace, mismatched_scope)
    assert value_at(mismatch, "ready") is False
    assert value_at(mismatch, "state") == "incompatible"
    assert "adapter_config_id_mismatch" in value_at(mismatch, "mismatch_reasons", ())
    assert "refresh_workspace_index_for_model" in value_at(mismatch, "planned_actions", ())


def test_atomic_session_snapshot_write_and_load_latest(tmp_path: Path) -> None:
    resume = require_module("chuk_lazarus.david.resume")
    write_snapshot = require_attr(
        resume,
        "write_session_snapshot",
        "atomic David session snapshot writes",
    )
    load_latest = require_attr(
        resume,
        "load_latest_session_snapshot",
        "loading the latest resumable David session",
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot_path = write_snapshot(
        workspace,
        session_id="session/one",
        user_id="user-1",
        task_id="task-1",
        task_type="repo_patch",
        selected_methodology="patch_targeting",
        turn_index=3,
        **_scope(),
    )

    assert snapshot_path.name == "session_one.json"
    assert snapshot_path.exists()
    assert (snapshot_path.parent / resume.LATEST_SESSION_FILENAME).exists()
    assert not list(snapshot_path.parent.glob("*.tmp"))

    latest = load_latest(workspace)
    assert value_at(latest, "session_id") == "session/one"
    assert value_at(latest, "turn_index") == 3
    assert value_at(latest, "selected_methodology") == "patch_targeting"
    assert value_at(latest, "model_identity") == _scope()["model_identity"]


def test_user_and_task_memory_paths_are_separate_in_snapshots(tmp_path: Path) -> None:
    indexing = require_module("chuk_lazarus.david.indexing")
    resume = require_module("chuk_lazarus.david.resume")
    resolve_paths = require_attr(
        indexing,
        "resolve_memory_paths",
        "separate David user and task memory paths",
    )
    snapshot_from_session = require_attr(
        resume,
        "snapshot_from_session",
        "resumable session metadata path capture",
    )
    write_snapshot = require_attr(
        resume,
        "write_session_snapshot",
        "snapshot persistence under David product root",
    )

    workspace = tmp_path / "workspace"
    user_memory_root = tmp_path / "person_memory"
    paths = resolve_paths(workspace, user_memory_root=user_memory_root)

    assert Path(paths.user_memory_path) == user_memory_root.resolve()
    assert Path(paths.task_memory_path) == workspace.resolve() / ".chuk_lazarus" / "david" / "task_memory"
    assert Path(paths.user_memory_path) != Path(paths.task_memory_path)

    session = SimpleNamespace(
        session_id="resume-1",
        workspace_path=str(workspace),
        user_id="user-1",
        task_id="task-1",
        selected_methodology="dependency_routing",
        **_scope(),
    )
    snapshot = snapshot_from_session(session, user_memory_root=user_memory_root)
    assert Path(snapshot.user_memory_path) == user_memory_root.resolve()
    assert Path(snapshot.task_memory_path) == Path(paths.task_memory_path)

    benchmark_dir = workspace / "benchmarks" / "proof-rig"
    benchmark_dir.mkdir(parents=True)
    write_snapshot(workspace, snapshot)
    assert not (benchmark_dir / ".chuk_lazarus").exists()
