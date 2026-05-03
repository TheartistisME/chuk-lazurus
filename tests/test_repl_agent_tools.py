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


def test_extract_tool_calls_accepts_gemma_custom_tool_shapes() -> None:
    text = """
    ```tool_call
    save_custom_tool(
      "weather",
      {
        "runtime": "python",
        "description": "Mock weather.",
        "script": "print('sunny')"
      }
    )
    ```
    ```json
    {"tool_name":"echoer","description":"Echo.","script":"print('ok')"}
    ```
    ```json
    {"name":"weather","runtime":"python","description":"Weather.","content":"print('ok')"}
    ```
    ```tool_call
    save_custom_tool{"name":"tiny_weather","runtime":"python","content":"print('ok')"}
    ```
    """

    calls = extract_tool_calls(text)

    assert [call.name for call in calls] == [
        "save_custom_tool",
        "save_custom_tool",
        "save_custom_tool",
        "save_custom_tool",
    ]
    assert calls[0].arguments["name"] == "weather"
    assert calls[0].arguments["script"] == "print('sunny')"
    assert calls[1].arguments["name"] == "echoer"
    assert calls[2].arguments["name"] == "weather"
    assert calls[3].arguments["name"] == "tiny_weather"


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


def test_custom_tool_can_be_saved_used_and_loaded_by_later_runner(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    trace_root = tmp_path / "traces"
    custom_root = tmp_path / "custom_tools"
    workspace.mkdir()
    runner = LocalCodingToolRunner(workspace, trace_root, custom_tools_root=custom_root)
    content = (
        "import json\n"
        "import sys\n"
        "args = json.load(sys.stdin)\n"
        "city = args.get('city', 'unknown')\n"
        "print(f'weather for {city}: sunny')\n"
    )

    saved = runner.execute(
        ToolCall(
            "save_custom_tool",
            {
                "name": "weather",
                "description": "Return a toy weather forecast for a city.",
                "runtime": "python",
                "content": content,
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        )
    )
    used = runner.execute(ToolCall("weather", {"city": "Perth"}))
    later_runner = LocalCodingToolRunner(workspace, trace_root, custom_tools_root=custom_root)
    used_later = later_runner.execute(ToolCall("weather", {"city": "Tokyo"}))

    assert saved.ok, saved.error
    assert (custom_root / "weather.json").exists()
    assert (custom_root / "weather.py").exists()
    assert used.ok, used.error
    assert used.output.strip() == "weather for Perth: sunny"
    assert used.metadata["custom_tool"] is True
    assert used_later.ok, used_later.error
    assert used_later.output.strip() == "weather for Tokyo: sunny"


def test_custom_tool_management_lists_reads_and_rejects_shadowing(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = LocalCodingToolRunner(
        workspace,
        tmp_path / "traces",
        custom_tools_root=tmp_path / "custom_tools",
    )
    content = "import json, sys\nprint(json.load(sys.stdin).get('value', ''))\n"

    shadowed = runner.execute(
        ToolCall(
            "save_custom_tool",
            {
                "name": "read_file",
                "description": "Should not be allowed.",
                "content": content,
            },
        )
    )
    created = runner.execute(
        ToolCall(
            "save_custom_tool",
            {
                "name": "echo_value",
                "description": "Echo the value argument.",
                "content": content,
            },
        )
    )
    duplicate = runner.execute(
        ToolCall(
            "save_custom_tool",
            {
                "name": "echo_value",
                "description": "Echo again.",
                "content": content,
            },
        )
    )
    listed = runner.execute(ToolCall("list_custom_tools", {}))
    read = runner.execute(ToolCall("read_custom_tool", {"name": "echo_value"}))

    assert not shadowed.ok
    assert "cannot shadow built-in" in shadowed.error
    assert created.ok, created.error
    assert not duplicate.ok
    assert "overwrite=true" in duplicate.error
    assert listed.ok
    assert "echo_value" in listed.output
    assert read.ok, read.error
    assert "Echo the value argument" in read.output
    assert "json.load" in read.output


def test_save_custom_tool_accepts_nested_alias_shape_from_model(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    custom_root = tmp_path / "custom_tools"
    workspace.mkdir()
    runner = LocalCodingToolRunner(
        workspace,
        tmp_path / "traces",
        custom_tools_root=custom_root,
    )

    result = runner.execute(
        ToolCall(
            "save_custom_tool",
            {
                "weather": {
                    "runtime": "python",
                    "description": "Mock weather.",
                    "script": "print('sunny')",
                    "overwrite": True,
                }
            },
        )
    )

    assert result.ok, result.error
    assert (custom_root / "weather.py").read_text(encoding="utf-8") == "print('sunny')"
    assert "weather" in runner.custom_tools


def test_save_custom_tool_defaults_missing_description(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    custom_root = tmp_path / "custom_tools"
    workspace.mkdir()
    runner = LocalCodingToolRunner(
        workspace,
        tmp_path / "traces",
        custom_tools_root=custom_root,
    )

    result = runner.execute(
        ToolCall(
            "save_custom_tool",
            {
                "name": "weather",
                "runtime": "python",
                "content": "print('sunny')",
            },
        )
    )

    assert result.ok, result.error
    manifest = json.loads((custom_root / "weather.json").read_text(encoding="utf-8"))
    assert manifest["description"] == "Custom tool weather."


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
    shell_cwd = result.output.strip().replace("\\", "/")
    expected_cwd = str(workspace.resolve()).replace("\\", "/")
    assert shell_cwd == expected_cwd or shell_cwd.endswith(expected_cwd.removeprefix("C:"))


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
