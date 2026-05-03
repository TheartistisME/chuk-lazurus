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


def test_symbolic_recall_can_chain_task_and_user_evidence(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))
    runtime.memory.task.append(MemoryArtifact(family="task", text="A depends on B", kind="symbolic_multi_hop"))
    runtime.memory.user.append(MemoryArtifact(family="user", text="B depends on C", kind="symbolic_multi_hop"))

    result = runtime.run_once("Explain the chain where A depends on B because B depends on C")

    assert result.method == "symbolic_multi_hop"
    assert len(result.route.evidence) >= 2
    assert {item["family"] for item in result.route.evidence} == {"task", "user"}

