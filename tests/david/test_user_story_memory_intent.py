from __future__ import annotations

import json
from pathlib import Path

from chuk_lazarus.david import DavidConfig, DavidRuntime
from chuk_lazarus.david.routing import MethodDetector


def test_reminder_user_story_routes_to_user_continuity() -> None:
    prompt = (
        "By the way, remind me tomorrow that we need to check the rate-limiting "
        "on the new socket handler. I'm worried about the AWS buffer limits."
    )

    assert MethodDetector().detect(prompt) == "user_continuity"


def test_reminder_user_story_writes_user_memory_not_task_memory(tmp_path: Path) -> None:
    prompt = (
        "By the way, remind me tomorrow that we need to check the rate-limiting "
        "on the new socket handler. I'm worried about the AWS buffer limits."
    )
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / ".david"))

    result = runtime.run_once(prompt)

    user_memory = tmp_path / ".david" / "memory" / "user-default.jsonl"
    task_memory = tmp_path / ".david" / "memory" / "task-default.jsonl"
    assert result.method == "user_continuity"
    assert result.route.memory_family == "user"
    assert result.writeback["family"] == "user"
    assert user_memory.exists()
    assert not task_memory.exists()
    record = json.loads(user_memory.read_text(encoding="utf-8").splitlines()[-1])
    assert record["kind"] == "user_continuity"
    assert "rate-limiting" in record["text"]


def test_active_code_work_still_wins_over_user_memory_language() -> None:
    prompt = "Fix the socket handler bug tomorrow by patching src/socket.js"

    assert MethodDetector().detect(prompt) == "repo_patch"


def test_existing_detector_capabilities_remain_stable() -> None:
    detector = MethodDetector()

    assert detector.detect("Run pytest as the quality gate") == "verify"
    assert detector.detect("Trace the import dependency path for session.py") == "source_dependency"
    assert detector.detect("Build a multi-hop chain because auth depends on cache") == "symbolic_multi_hop"
    assert detector.detect("What was the latest preference I mentioned?") == "temporal_recall"
