from __future__ import annotations

from pathlib import Path

from chuk_lazarus.david import DavidConfig, DavidRuntime
from chuk_lazarus.david.memory import MemoryArtifact


def test_user_and_task_memory_are_separate(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    runtime.memory.user.append(MemoryArtifact(family="user", text="User prefers concise answers", kind="preference"))
    runtime.memory.task.append(MemoryArtifact(family="task", text="Patch target is src/runtime.py", kind="repo_patch"))

    user_hits = runtime.memory.user.recall("preference concise")
    task_hits = runtime.memory.task.recall("patch target")

    assert user_hits[0]["family"] == "user"
    assert task_hits[0]["family"] == "task"
    assert "src/runtime.py" not in user_hits[0]["text"]


def test_temporal_recall_uses_user_memory_latest_match(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))
    runtime.memory.user.append(MemoryArtifact(family="user", text="Earlier deadline is Monday", kind="temporal_recall", timestamp="2026-01-01T00:00:00+00:00"))
    runtime.memory.user.append(MemoryArtifact(family="user", text="Latest deadline is Friday", kind="temporal_recall", timestamp="2026-01-02T00:00:00+00:00"))

    result = runtime.run_once("What is the latest deadline?")

    assert result.method == "temporal_recall"
    assert result.route.evidence
    assert "Latest deadline is Friday" in result.route.evidence[0]["text"]


def test_temporal_recall_writeback_uses_user_memory(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    artifact = runtime.memory.writeback(
        method="temporal_recall",
        user_id="user-1",
        session_id="session-1",
        text="Latest preference is compact diffs",
        metadata={"provenance": "test"},
    )

    assert artifact.family == "user"
    assert artifact.kind == "temporal_recall"
    assert runtime.memory.user.all() == [artifact]
    assert runtime.memory.task.all() == []
    assert artifact.metadata["sensitivity"] == "normal"
    assert artifact.metadata["source_method"] == "temporal_recall"
    assert artifact.metadata["supersedes"] == []


def test_repo_patch_and_source_dependency_writeback_use_task_memory(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    patch_artifact = runtime.memory.writeback(
        method="repo_patch",
        user_id="user-1",
        session_id="session-1",
        text="Patch target is src/runtime.py",
        metadata={
            "effective_at": "2026-01-01T00:00:00+00:00",
            "expires_at": "2026-01-02T00:00:00+00:00",
            "provenance": "test",
            "sensitivity": "private",
            "source_method": "temporal_recall",
            "supersedes": ["old-id"],
        },
    )
    source_artifact = runtime.memory.writeback(
        method="source_dependency",
        user_id="user-1",
        session_id="session-1",
        text="Source dependency runs through src/session.py",
        metadata={"provenance": "test"},
    )

    assert [artifact.family for artifact in runtime.memory.task.all()] == ["task", "task"]
    assert [artifact.kind for artifact in runtime.memory.task.all()] == ["repo_patch", "source_dependency"]
    assert runtime.memory.user.all() == []
    assert patch_artifact.family == "task"
    assert source_artifact.family == "task"
    assert "sensitivity" not in patch_artifact.metadata
    assert "effective_at" not in patch_artifact.metadata
    assert "expires_at" not in patch_artifact.metadata
    assert "supersedes" not in patch_artifact.metadata
    assert "source_method" not in patch_artifact.metadata


def test_user_memory_policy_tracks_active_and_stale_memories(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))
    old = MemoryArtifact(
        family="user",
        text="User prefers compact diffs",
        kind="preference",
        artifact_id="old-pref",
        metadata={
            "effective_at": "2026-01-01T00:00:00+00:00",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "sensitivity": "private",
            "source_method": "user_continuity",
            "supersedes": [],
        },
    )
    expired = MemoryArtifact(
        family="user",
        text="User temporarily prefers long diffs",
        kind="preference",
        artifact_id="expired-pref",
        metadata={"expires_at": "2026-01-02T00:00:00+00:00"},
    )
    latest = MemoryArtifact(
        family="user",
        text="User prefers summary diffs",
        kind="preference",
        artifact_id="latest-pref",
        metadata={
            "effective_at": "2026-01-03T00:00:00+00:00",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "sensitivity": "private",
            "source_method": "user_continuity",
            "supersedes": ["old-pref"],
        },
    )
    runtime.memory.user.append(old)
    runtime.memory.user.append(expired)
    runtime.memory.user.append(latest)

    active = runtime.memory.active_user_memories(now="2026-01-04T00:00:00+00:00")
    stale = runtime.memory.stale_user_memories(now="2026-01-04T00:00:00+00:00")
    recall = runtime.memory.user.recall("prefers diffs")

    assert [artifact.artifact_id for artifact in active] == ["latest-pref"]
    assert {artifact.artifact_id for artifact in stale} == {"old-pref", "expired-pref"}
    assert [item["artifact_id"] for item in recall] == ["latest-pref"]


def test_user_writeback_policy_refuses_temporary_workspace_state(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    artifact = runtime.memory.writeback(
        method="user_continuity",
        user_id="user-1",
        session_id="session-1",
        text="Temporary workspace state: selected file is src/runtime.py",
        metadata={
            "provenance": "test",
            "selected_paths": ["src/runtime.py"],
            "sensitivity": "private",
        },
    )

    assert artifact.family == "task"
    assert runtime.memory.user.all() == []
    assert runtime.memory.task.all() == [artifact]
    assert artifact.metadata["user_memory_refused_reason"] == "temporary_workspace_state"
    assert "sensitivity" not in artifact.metadata


def test_symbolic_recall_can_chain_task_and_user_evidence(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))
    runtime.memory.task.append(MemoryArtifact(family="task", text="A depends on B", kind="symbolic_multi_hop"))
    runtime.memory.user.append(MemoryArtifact(family="user", text="B depends on C", kind="symbolic_multi_hop"))

    result = runtime.run_once("Explain the chain where A depends on B because B depends on C")

    assert result.method == "symbolic_multi_hop"
    assert len(result.route.evidence) >= 2
    assert {item["family"] for item in result.route.evidence} == {"task", "user"}
