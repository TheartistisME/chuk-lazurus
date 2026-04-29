from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from chuk_lazarus.repl_agent_tools import (
    LocalCodingToolRunner,
    ToolCall,
    extract_tool_calls,
)


def test_extract_tool_calls_xml_and_fenced_json() -> None:
    text = """
    <tool_call>{"name":"read_file","args":{"path":"README.md"}}</tool_call>
    ```tool_call
    [
      {"name":"search","arguments":{"pattern":"needle"},"call_id":"fixed"},
      {"name":"list_dir","arguments":{"path":"."}}
    ]
    ```
    ```tool_call
    {not json}
    ```
    """

    calls = extract_tool_calls(text)

    assert [call.name for call in calls] == ["read_file", "search", "list_dir"]
    assert calls[0].arguments == {"path": "README.md"}
    assert calls[1].call_id == "fixed"
    assert calls[2].call_id


def test_read_file_is_line_numbered_and_traced(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    trace_root = tmp_path / "traces"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    runner = LocalCodingToolRunner(workspace, trace_root)

    result = runner.execute(
        ToolCall("read_file", {"path": "notes.txt"}),
        session_id="session-1",
        turn_index=7,
    )

    assert result.ok
    assert "1: alpha" in result.output
    trace_path = trace_root / "session-1.jsonl"
    assert trace_path.exists()
    record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["name"] == "read_file"
    assert record["turn_index"] == 7


def test_path_escape_is_rejected(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "outside.txt").write_text("secret\n", encoding="utf-8")
    runner = LocalCodingToolRunner(workspace, tmp_path / "traces")

    result = runner.execute(ToolCall("read_file", {"path": "../outside.txt"}))

    assert not result.ok
    assert "outside workspace" in result.error


def test_write_file_creates_file_and_traces(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    trace_root = tmp_path / "traces"
    workspace.mkdir()
    runner = LocalCodingToolRunner(workspace, trace_root)

    result = runner.execute(
        ToolCall("write_file", {"path": "created.txt", "content": "hello\n"}),
        session_id="session-1",
        turn_index=8,
    )

    assert result.ok, result.error
    assert (workspace / "created.txt").read_text(encoding="utf-8") == "hello\n"
    assert "wrote 6 bytes" in result.output
    trace_path = trace_root / "session-1.jsonl"
    record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["name"] == "write_file"
    assert record["metadata"]["mode"] == "create"


def test_write_file_refuses_existing_file_without_explicit_mode(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "existing.txt"
    target.write_text("original\n", encoding="utf-8")
    runner = LocalCodingToolRunner(workspace, tmp_path / "traces")

    result = runner.execute(ToolCall("write_file", {"path": "existing.txt", "content": "new\n"}))

    assert not result.ok
    assert "overwrite=true or append=true" in result.error
    assert target.read_text(encoding="utf-8") == "original\n"


def test_write_file_overwrites_with_expected_hash(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "guarded.txt"
    target.write_text("old\n", encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    runner = LocalCodingToolRunner(workspace, tmp_path / "traces")

    result = runner.execute(
        ToolCall(
            "write_file",
            {
                "path": "guarded.txt",
                "content": "new\n",
                "overwrite": True,
                "expected_sha256": digest,
            },
        )
    )

    assert result.ok, result.error
    assert target.read_text(encoding="utf-8") == "new\n"
    assert result.metadata["mode"] == "overwrite"


def test_write_file_rejects_bad_hash_and_path_escape(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "guarded.txt"
    target.write_text("old\n", encoding="utf-8")
    runner = LocalCodingToolRunner(workspace, tmp_path / "traces")

    bad_hash_result = runner.execute(
        ToolCall(
            "write_file",
            {
                "path": "guarded.txt",
                "content": "new\n",
                "overwrite": True,
                "expected_sha256": "0" * 64,
            },
        )
    )
    escape_result = runner.execute(
        ToolCall("write_file", {"path": "../escape.txt", "content": "nope\n"})
    )

    assert not bad_hash_result.ok
    assert "expected_sha256 did not match" in bad_hash_result.error
    assert target.read_text(encoding="utf-8") == "old\n"
    assert not escape_result.ok
    assert "outside workspace" in escape_result.error


def test_search_prefers_or_falls_back(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("needle here\n", encoding="utf-8")
    (workspace / "b.txt").write_text("nothing\n", encoding="utf-8")
    runner = LocalCodingToolRunner(workspace, tmp_path / "traces")

    result = runner.execute(ToolCall("search", {"pattern": "needle", "path": "."}))

    assert result.ok
    assert "needle here" in result.output


def test_shell_runs_in_workspace(tmp_path) -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash is not available")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = LocalCodingToolRunner(workspace, tmp_path / "traces")

    result = runner.execute(ToolCall("shell", {"command": "pwd"}))

    assert result.ok
    assert result.output.strip() == str(workspace.resolve())


def test_apply_patch_updates_file(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "file.txt"
    target.write_text("before\n", encoding="utf-8")
    runner = LocalCodingToolRunner(workspace, tmp_path / "traces")
    patch = """\
--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-before
+after
"""

    result = runner.execute(ToolCall("apply_patch", {"patch": patch}))

    assert result.ok, result.error
    assert target.read_text(encoding="utf-8") == "after\n"
