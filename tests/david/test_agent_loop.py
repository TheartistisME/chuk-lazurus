from __future__ import annotations

from pathlib import Path
import sys

from chuk_lazarus.david.agent_loop import (
    parse_agent_action,
    plan_natural_language_fallback_actions,
    render_model_action_prompt,
    run_agent_loop,
    AgentLoopState,
)
from chuk_lazarus.david.tools import LocalTools


def test_parse_agent_action_from_fenced_model_text() -> None:
    action = parse_agent_action(
        """I will read the file.

```json
{"action": "read", "path": "src/example.py"}
```
"""
    )

    assert action is not None
    assert action.action == "read"
    assert action.path == "src/example.py"


def test_parse_agent_action_from_prose_wrapped_json_object() -> None:
    action = parse_agent_action(
        'I will inspect the target first.\n{"action": "read", "path": "src/example.py"}\nThen I will wait.'
    )

    assert action is not None
    assert action.action == "read"
    assert action.path == "src/example.py"


def test_parse_agent_action_from_sentinel_label() -> None:
    action = parse_agent_action('David action JSON: {"action": "plan", "reason": "inspect before editing"}')

    assert action is not None
    assert action.action == "plan"
    assert action.reason == "inspect before editing"


def test_parse_agent_action_refuses_multiple_json_objects() -> None:
    result = run_agent_loop(
        [
            (
                'David action JSON: {"action": "read", "path": "notes.txt"}\n'
                '{"action": "write", "path": "notes.txt", "content": "unsafe"}'
            )
        ],
        LocalTools(Path.cwd()),
    )

    assert result.status == "refused"
    assert "multiple action JSON objects" in result.reason


def test_parse_agent_action_ignores_malformed_prose_json() -> None:
    action = parse_agent_action('David action JSON: {"action": "read", "path": "src/example.py"')

    assert action is None


def test_parse_agent_action_refuses_command_string_in_prose_json(tmp_path: Path) -> None:
    result = run_agent_loop(
        ['David action JSON: {"action": "run", "command": "python -m pytest"}'],
        LocalTools(tmp_path),
    )

    assert result.status == "refused"
    assert "command must be a list of arguments" in result.reason


def test_agent_loop_reads_workspace_file(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("remember me\n", encoding="utf-8")

    result = run_agent_loop([{"action": "read", "path": "notes.txt"}], LocalTools(tmp_path))

    assert result.status == "no_action"
    assert result.trace[0].action == "read"
    assert result.trace[0].observation["content"] == "remember me\n"


def test_agent_loop_writes_workspace_file(tmp_path: Path) -> None:
    result = run_agent_loop(
        [{"action": "write", "path": "src/new.py", "content": "VALUE = 42\n"}],
        LocalTools(tmp_path),
    )

    assert result.status == "no_action"
    assert (tmp_path / "src" / "new.py").read_text(encoding="utf-8") == "VALUE = 42\n"
    assert result.trace[0].observation["bytes"] == len("VALUE = 42\n")


def test_agent_loop_refuses_protected_write_path(tmp_path: Path) -> None:
    target = tmp_path / "scripts" / "run_swebench_pro_parity.py"
    target.parent.mkdir()
    target.write_text("old\n", encoding="utf-8")

    result = run_agent_loop(
        [{"action": "write", "path": "scripts/run_swebench_pro_parity.py", "content": "new\n"}],
        LocalTools(tmp_path),
    )

    assert result.status == "refused"
    assert result.trace[0].action == "refuse"
    assert result.trace[0].observation["requested_action"] == "write"
    assert "protected proof-rig path: scripts/run_swebench_pro_parity.py" in result.reason
    assert target.read_text(encoding="utf-8") == "old\n"


def test_agent_loop_applies_strict_patch_then_verifies(tmp_path: Path) -> None:
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir()
    target.write_text("VALUE = 1\n", encoding="utf-8")

    result = run_agent_loop(
        [
            {
                "action": "patch",
                "content": """src/example.py
<<<< SEARCH
VALUE = 1
==== REPLACE
VALUE = 2
>>>>
""",
            },
            {
                "action": "verify",
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "raise SystemExit(0 if Path('src/example.py').read_text(encoding='utf-8') == 'VALUE = 2\\n' else 1)"
                    ),
                ],
            },
        ],
        LocalTools(tmp_path),
    )

    assert result.status == "verified"
    assert result.trace[0].action == "patch"
    assert result.trace[0].ok is True
    assert result.trace[0].observation["mode"] == "strict_search_replace"
    assert result.trace[0].observation["changed_paths"] == ["src/example.py"]
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_agent_loop_refuses_protected_patch_path(tmp_path: Path) -> None:
    target = tmp_path / "scripts" / "run_swebench_pro_parity.py"
    target.parent.mkdir()
    target.write_text("old\n", encoding="utf-8")

    result = run_agent_loop(
        [
            {
                "action": "patch",
                "content": """scripts/run_swebench_pro_parity.py
<<<< SEARCH
old
==== REPLACE
new
>>>>
""",
            },
        ],
        LocalTools(tmp_path),
    )

    assert result.status == "refused"
    assert result.trace[0].action == "patch"
    assert result.trace[0].ok is False
    assert result.trace[0].observation["protected_paths"] == ["scripts/run_swebench_pro_parity.py"]
    assert any("protected proof-rig" in failure for failure in result.trace[0].observation["failures"])
    assert target.read_text(encoding="utf-8") == "old\n"


def test_model_driven_patch_failure_can_repair_and_verify(tmp_path: Path) -> None:
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir()
    target.write_text("VALUE = 1\n", encoding="utf-8")
    seen_repairs: list[object] = []

    def model_step(state: AgentLoopState):
        seen_repairs.append(state.last_observation.get("repair"))
        if state.step == 1:
            return {
                "action": "patch",
                "content": """src/example.py
<<<< SEARCH
VALUE = 99
==== REPLACE
VALUE = 2
>>>>
""",
            }
        if state.step == 2:
            assert state.last_observation["repair"]["trigger_action"] == "patch"
            return {
                "action": "patch",
                "content": """src/example.py
<<<< SEARCH
VALUE = 1
==== REPLACE
VALUE = 2
>>>>
""",
            }
        return {
            "action": "verify",
            "command": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "raise SystemExit(0 if Path('src/example.py').read_text(encoding='utf-8') == 'VALUE = 2\\n' else 1)"
                ),
            ],
        }

    result = run_agent_loop(model_step, LocalTools(tmp_path), max_steps=3, mode="model_driven")

    assert result.status == "verified"
    assert [step.action for step in result.trace] == ["patch", "patch", "verify"]
    assert result.trace[0].ok is False
    assert result.trace[0].observation["repair"]["next_step"] == 2
    assert result.trace[0].provenance["repair"]["attempt"] == 1
    assert seen_repairs[1]["trigger_action"] == "patch"
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_model_driven_verify_failure_can_patch_repair_and_verify(tmp_path: Path) -> None:
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir()
    target.write_text("VALUE = 1\n", encoding="utf-8")
    verify_command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            "raise SystemExit(0 if Path('src/example.py').read_text(encoding='utf-8') == 'VALUE = 2\\n' else 1)"
        ),
    ]

    def model_step(state: AgentLoopState):
        if state.step == 1:
            return {"action": "verify", "command": verify_command}
        if state.step == 2:
            assert state.last_observation["returncode"] == 1
            assert state.last_observation["repair"]["trigger_action"] == "verify"
            return {
                "action": "patch",
                "content": """src/example.py
<<<< SEARCH
VALUE = 1
==== REPLACE
VALUE = 2
>>>>
""",
            }
        return {"action": "verify", "command": verify_command}

    result = run_agent_loop(model_step, LocalTools(tmp_path), max_steps=3, mode="model_driven")

    assert result.status == "verified"
    assert result.ok is True
    assert [step.action for step in result.trace] == ["verify", "patch", "verify"]
    assert result.trace[0].ok is False
    assert result.trace[0].observation["repair"]["next_step"] == 2
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_model_driven_repair_stops_at_max_steps_without_verified_status(tmp_path: Path) -> None:
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir()
    target.write_text("VALUE = 1\n", encoding="utf-8")

    def model_step(state: AgentLoopState):
        if state.step == 1:
            return {"action": "verify", "passed": False, "reason": "not fixed yet"}
        return {
            "action": "patch",
            "content": """src/example.py
<<<< SEARCH
VALUE = 1
==== REPLACE
VALUE = 2
>>>>
""",
        }

    result = run_agent_loop(model_step, LocalTools(tmp_path), max_steps=2, mode="model_driven")

    assert result.status == "max_steps"
    assert result.verified is False
    assert [step.action for step in result.trace] == ["verify", "patch"]
    assert result.trace[0].observation["repair"]["next_step"] == 2
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_agent_loop_refuses_empty_patch_content(tmp_path: Path) -> None:
    result = run_agent_loop([{"action": "patch", "content": "  \n"}], LocalTools(tmp_path))

    assert result.status == "refused"
    assert "patch requires non-empty content" in result.reason


def test_model_action_prompt_includes_patch_schema() -> None:
    prompt = render_model_action_prompt(
        AgentLoopState(
            workspace_root="/workspace",
            step=1,
            max_steps=3,
            objective="fix repo bug",
        )
    )

    assert "read -> patch -> verify" in prompt
    assert "patch|write" in prompt
    assert "command as an array" in prompt


def test_agent_loop_runs_shell_command(tmp_path: Path) -> None:
    result = run_agent_loop(
        [{"action": "run", "command": [sys.executable, "-c", "print('ok')"]}],
        LocalTools(tmp_path),
    )

    assert result.status == "no_action"
    assert result.trace[0].action == "run"
    assert result.trace[0].ok is True
    assert result.trace[0].observation["stdout"] == "ok\n"


def test_agent_loop_stops_on_verify_pass(tmp_path: Path) -> None:
    result = run_agent_loop(
        [
            {"action": "plan", "reason": "verify directly"},
            {"action": "verify", "passed": True, "reason": "tests passed"},
            {"action": "write", "path": "should_not_exist.txt", "content": "nope"},
        ],
        LocalTools(tmp_path),
    )

    assert result.status == "verified"
    assert result.ok is True
    assert result.steps == 2
    assert not (tmp_path / "should_not_exist.txt").exists()


def test_agent_loop_reports_verify_failure(tmp_path: Path) -> None:
    result = run_agent_loop(
        [{"action": "verify", "command": [sys.executable, "-c", "raise SystemExit(7)"]}],
        LocalTools(tmp_path),
    )

    assert result.status == "verify_failed"
    assert result.verified is False
    assert result.trace[0].action == "verify"
    assert result.trace[0].observation["returncode"] == 7


def test_agent_loop_refuses_unknown_action(tmp_path: Path) -> None:
    result = run_agent_loop([{"action": "delete", "path": "notes.txt"}], LocalTools(tmp_path))

    assert result.status == "refused"
    assert result.trace[0].action == "refuse"
    assert "unknown action" in result.reason


def test_agent_loop_enforces_max_step_cap(tmp_path: Path) -> None:
    result = run_agent_loop(
        [
            {"action": "plan", "reason": "one"},
            {"action": "plan", "reason": "two"},
            {"action": "verify", "passed": True},
        ],
        LocalTools(tmp_path),
        max_steps=2,
    )

    assert result.status == "max_steps"
    assert result.verified is False
    assert len(result.trace) == 2


def test_agent_loop_refuses_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")

    result = run_agent_loop([{"action": "read", "path": "../outside.txt"}], LocalTools(tmp_path))

    assert result.status == "refused"
    assert result.trace[0].action == "refuse"
    assert "escapes workspace" in result.reason


def test_agent_loop_feeds_observations_to_model_callback(tmp_path: Path) -> None:
    seen: list[object] = []

    def model_step(state):
        seen.append(state.last_observation.get("bytes"))
        if state.step == 1:
            return {"action": "write", "path": "note.txt", "content": "ok\n"}
        return {"action": "verify", "passed": state.last_observation.get("bytes") == 3}

    result = run_agent_loop(model_step, LocalTools(tmp_path), objective="write note")

    assert result.status == "verified"
    assert seen == [None, 3]
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "ok\n"


def test_agent_loop_refuses_model_step_exception(tmp_path: Path) -> None:
    def model_step(state):
        del state
        raise RuntimeError("backend offline")

    result = run_agent_loop(model_step, LocalTools(tmp_path))

    assert result.status == "refused"
    assert result.trace[0].action == "refuse"
    assert "backend offline" in result.reason


def test_natural_language_fallback_plans_write_and_verify() -> None:
    actions = plan_natural_language_fallback_actions("create a file named hello.txt that says hello")

    assert len(actions) == 2
    assert actions[0]["action"] == "write"
    assert actions[0]["path"] == "hello.txt"
    assert actions[0]["content"] == "hello"
    assert actions[1]["action"] == "verify"
    assert isinstance(actions[1]["command"], list)


def test_model_driven_agent_loop_falls_back_from_no_json_text(tmp_path: Path) -> None:
    def model_step(state):
        del state
        return "offline backend prose with no JSON"

    result = run_agent_loop(
        model_step,
        LocalTools(tmp_path),
        objective="create a file named hello.txt that says hello",
        mode="model_driven",
    )

    assert result.status == "verified"
    assert result.ok is True
    assert [step.action for step in result.trace] == ["write", "verify"]
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello"
    assert result.trace[0].provenance["raw"]["_fallback_provenance"]["reason"] == "model returned no usable action"


def test_model_driven_agent_loop_falls_back_from_provider_exception(tmp_path: Path) -> None:
    def model_step(state):
        del state
        raise RuntimeError("vindex artifact is not generative")

    result = run_agent_loop(
        model_step,
        LocalTools(tmp_path),
        objective='write file at "notes/today.txt" with content "ship David"',
        mode="model_driven",
    )

    assert result.status == "verified"
    assert (tmp_path / "notes" / "today.txt").read_text(encoding="utf-8") == "ship David"
    assert "provider failed" in result.trace[0].provenance["raw"]["_fallback_provenance"]["reason"]


def test_natural_language_fallback_is_model_driven_only(tmp_path: Path) -> None:
    result = run_agent_loop(
        ["create a file named hello.txt that says hello"],
        LocalTools(tmp_path),
        objective="create a file named hello.txt that says hello",
        mode="explicit",
    )

    assert result.status == "no_action"
    assert not (tmp_path / "hello.txt").exists()


def test_natural_language_fallback_returns_no_action_when_unsure(tmp_path: Path) -> None:
    def model_step(state):
        del state
        return ""

    result = run_agent_loop(
        model_step,
        LocalTools(tmp_path),
        objective="make things better in the repo",
        mode="model_driven",
    )

    assert result.status == "no_action"
    assert result.trace[0].action == "no_action"


def test_natural_language_fallback_rejects_unsafe_path() -> None:
    actions = plan_natural_language_fallback_actions("create a file named ../outside.txt that says nope")

    assert actions == ()


def test_natural_language_fallback_accepts_simple_test_command() -> None:
    actions = plan_natural_language_fallback_actions(
        "create a file named hello.txt that says hello then run python -m py_compile hello.txt"
    )

    assert len(actions) == 2
    assert actions[1]["action"] == "verify"
    assert actions[1]["command"][:3] == [sys.executable, "-m", "py_compile"]


def test_natural_language_fallback_rejects_shell_string_command() -> None:
    actions = plan_natural_language_fallback_actions(
        "create a file named hello.txt that says hello then run pytest; rm -rf ."
    )

    assert actions == ()
