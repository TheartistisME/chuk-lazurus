from __future__ import annotations

import json
from pathlib import Path

import pytest

from chuk_lazarus.david.resume import (
    SessionSnapshot,
    compact_summary,
    default_resume_path,
    format_resume_snapshot,
    load_session_snapshot,
    save_session_snapshot,
    summarize_result,
)


def test_resume_snapshot_saves_to_default_workspace_path(tmp_path: Path) -> None:
    snapshot = SessionSnapshot(
        session_id="session-1",
        workspace=str(tmp_path),
        adapter_scope={"model_id": "offline"},
        memory_paths={"user": "memory/user.jsonl", "task": "memory/task.jsonl"},
        last_result_summary="patched src/service.py",
    )

    saved = save_session_snapshot(snapshot)
    loaded = load_session_snapshot(default_resume_path(tmp_path))

    assert saved == snapshot
    assert loaded == snapshot
    assert (tmp_path / ".david" / "resume.json").exists()


def test_resume_snapshot_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_session_snapshot(tmp_path / ".david" / "resume.json") is None


def test_resume_snapshot_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "resume.json"
    path.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_session_snapshot(path)


def test_summarize_result_uses_answer_from_json() -> None:
    class Result:
        def to_json(self) -> dict[str, str]:
            return {"answer": "alpha beta gamma"}

    assert summarize_result(Result(), max_chars=10) == "alpha beta"


def test_format_resume_snapshot_shows_startup_friendly_fields() -> None:
    snapshot = SessionSnapshot(
        session_id="session-2",
        workspace="C:/workspace/project",
        adapter_scope={},
        memory_paths={},
        updated_at="2026-05-04T01:02:03+00:00",
        last_result_summary=" ".join(["result"] * 80),
    )

    text = format_resume_snapshot(snapshot, max_summary_chars=40)

    assert "David resume" in text
    assert "session: session-2" in text
    assert "workspace: C:/workspace/project" in text
    assert "updated: 2026-05-04T01:02:03+00:00" in text
    assert "last result: result result result" in text
    assert len(text.split("last result: ", 1)[1]) <= 40


def test_format_resume_snapshot_handles_missing_snapshot() -> None:
    assert format_resume_snapshot(None) == "David resume\n- status: no saved session"


def test_compact_summary_collapses_whitespace_and_truncates() -> None:
    assert compact_summary("alpha\n\n beta   gamma", max_chars=14) == "alpha beta..."
