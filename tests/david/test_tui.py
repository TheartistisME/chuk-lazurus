from __future__ import annotations

from io import StringIO

from chuk_lazarus.david.resume import SessionSnapshot
from chuk_lazarus.david.tui import DavidTui


class FakeIndex:
    def __init__(self) -> None:
        self.jit_calls = 0

    def jit(self) -> str:
        self.jit_calls += 1
        return "jit indexed"


class FakeRuntime:
    def __init__(self) -> None:
        self.index = FakeIndex()
        self.respond_calls: list[str] = []
        self.agent_loop_calls: list[str] = []
        self.resume_snapshot = SessionSnapshot(
            session_id="session-1",
            workspace="C:/workspace/project",
            adapter_scope={"model_id": "offline"},
            memory_paths={"task": "task.jsonl"},
            updated_at="2026-05-04T01:02:03+00:00",
            last_result_summary="patched the David TUI",
        )

    def readiness(self) -> dict[str, str]:
        return {
            "model validation": "ready",
            "index": "warm",
            "memory": "hot",
        }

    def respond(self, prompt: str) -> str:
        self.respond_calls.append(prompt)
        return f"prompt={prompt}"

    def memory_status(self) -> str:
        return "memory detail"

    def index_status(self) -> str:
        return "index detail"

    def verify(self, command: str | None = None) -> str:
        return f"verified {command or 'default'}"

    def run_shell(self, command: str) -> str:
        return f"ran {command}"

    def read_file(self, path: str) -> str:
        return f"read {path}"

    def write_file(self, spec: str) -> str:
        return f"wrote {spec}"

    def apply_patch(self, patch_text: str) -> str:
        return f"applied {patch_text}"

    def run_agent_loop(self, prompt: str) -> dict[str, object]:
        self.agent_loop_calls.append(prompt)
        return {
            "loop": {
                "status": "verified",
                "steps": 2,
                "verified": True,
                "reason": "verify passed",
                "trace": [
                    {
                        "step": 1,
                        "action": "write",
                        "ok": True,
                        "observation": {"path": "src/app.py", "bytes": 4},
                    },
                    {
                        "step": 2,
                        "action": "verify",
                        "ok": True,
                        "observation": {"passed": True},
                    },
                ],
            }
        }


class DegradedRuntime(FakeRuntime):
    def readiness(self) -> dict[str, str]:
        return {
            "model validation": "needs_review: manual attestation required",
            "backend": "offline: not loaded",
            "index": "missing: no workspace index for adapter",
            "source index": "ready (12 files); live state missing",
            "memory": "missing: user, task",
            "resume": "missing",
        }


class VerifyDiscoveryRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.verify_calls: list[str | None] = []

    def verify(self, command: str | None = None) -> dict[str, object]:
        self.verify_calls.append(command)
        if command is None:
            return {
                "ok": True,
                "capability": "verify",
                "reason": "candidate quality gates discovered",
                "checks": {
                    "candidate_quality_gates": {
                        "candidate_count": 1,
                        "selected_count": 0,
                        "commands_run": 0,
                        "run_reason": "candidate gates were discovered but not executed without --cmd",
                        "discovered_commands": ["python -m pytest tests/david/test_tui.py -q"],
                        "selected_commands": [],
                    }
                },
                "evidence": [
                    {
                        "kind": "quality_gate_candidate",
                        "name": "pytest-tui",
                        "display_command": "python -m pytest tests/david/test_tui.py -q",
                        "reason": "focused TUI test",
                    }
                ],
            }
        return {
            "ok": True,
            "reason": "command passed",
            "command_result": {
                "command": command,
                "returncode": 0,
                "stdout": "ok\n",
                "stderr": "",
            },
        }


class RepairRuntime(FakeRuntime):
    def run_agent_loop(self, prompt: str) -> dict[str, object]:
        self.agent_loop_calls.append(prompt)
        return {
            "loop": {
                "status": "verified",
                "steps": 3,
                "verified": True,
                "reason": "verify passed after repair",
                "trace": [
                    {
                        "step": 1,
                        "action": "verify",
                        "ok": False,
                        "observation": {"passed": False, "reason": "tests failed"},
                    },
                    {
                        "step": 2,
                        "action": "patch",
                        "ok": True,
                        "observation": {"path": "src/app.py", "bytes": 5},
                    },
                    {
                        "step": 3,
                        "action": "verify",
                        "ok": True,
                        "observation": {"passed": True},
                    },
                ],
            },
            "repair_summary": {
                "failure_count": 1,
                "repair_attempt_count": 1,
                "retry_count": 0,
                "recovered": True,
                "recovery_outcome": "verified_after_failure",
                "verification": {"outcome": "passed", "step": 3},
                "failed_steps": [{"step": 1, "action": "verify", "reason": "tests failed"}],
                "repair_steps": [{"step": 2, "action": "patch", "ok": True}],
            },
        }


def test_once_outputs_startup_readiness_and_prompt_response():
    output = StringIO()
    tui = DavidTui(FakeRuntime(), color=False, output_stream=output)

    rc = tui.run(once="hello")

    assert rc == 0
    text = output.getvalue()
    assert "David terminal agent" in text
    assert "model validation: ready [state=ready]" in text
    assert "index: warm [state=ready]" in text
    assert "memory: hot [state=ready]" in text
    assert "David resume" in text
    assert "session: session-1" in text
    assert "workspace: C:/workspace/project" in text
    assert "updated: 2026-05-04T01:02:03+00:00" in text
    assert "last result: patched the David TUI" in text
    assert "prompt=hello" in text


def test_plain_safe_write_prompt_runs_agent_loop():
    output = StringIO()
    runtime = FakeRuntime()
    tui = DavidTui(runtime, color=False, output_stream=output)

    rc = tui.run(once="create a file named hello.txt that says hello")

    assert rc == 0
    text = output.getvalue()
    assert "agent loop: verified steps=2 verified=True" in text
    assert "- 1: write ok=True path=src/app.py bytes=4" in text
    assert "repair summary" not in text
    assert runtime.agent_loop_calls == ["create a file named hello.txt that says hello"]
    assert runtime.respond_calls == []


def test_plain_arbitrary_prompt_stays_runtime_response():
    output = StringIO()
    runtime = FakeRuntime()
    tui = DavidTui(runtime, color=False, output_stream=output)

    rc = tui.run(once="summarize this repo")

    assert rc == 0
    assert "prompt=summarize this repo" in output.getvalue()
    assert runtime.respond_calls == ["summarize this repo"]
    assert runtime.agent_loop_calls == []


def test_slash_commands_cover_terminal_surface():
    tui = DavidTui(FakeRuntime(), color=False, output_stream=StringIO())

    assert "David startup readiness" in tui.dispatch("/status").text
    assert "David startup readiness" in tui.dispatch("/readiness").text
    assert "last result: patched the David TUI" in tui.dispatch("/resume").text
    assert tui.dispatch("/memory").text == "memory detail"
    assert tui.dispatch("/index").text == "index detail"
    assert "jit indexed" in tui.dispatch("/index jit").text
    assert tui.runtime.index.jit_calls == 1
    assert "verified default" in tui.dispatch("/verify").text
    assert "verified pytest" in tui.dispatch("/verify pytest").text
    assert tui.dispatch("/run pytest -q").text == "ran pytest -q"
    assert tui.dispatch("/shell pytest -q").text == "ran pytest -q"
    agent = tui.dispatch('/agent {"action": "verify", "passed": true}').text
    assert "agent loop: verified steps=2 verified=True" in agent
    assert "- 1: write ok=True path=src/app.py bytes=4" in agent
    assert "- 2: verify ok=True passed=True" in agent
    loop = tui.dispatch('/loop {"action": "verify", "passed": true}').text
    assert "agent loop: verified" in loop
    assert tui.dispatch("/read src/app.py").text == "read src/app.py"
    assert tui.dispatch("/write src/app.py pass").text == "wrote src/app.py pass"
    assert "new\nline" in tui.dispatch("/apply new\\nline").text
    assert "new\nline" in tui.dispatch("/patch new\\nline").text
    assert "/readiness" in tui.dispatch("/help").text
    assert "/resume" in tui.dispatch("/help").text
    assert "/agent" in tui.dispatch("/help").text

    exit_result = tui.dispatch("/quit")
    assert exit_result.should_exit is True
    assert "bye" in exit_result.text


def test_startup_readiness_labels_degraded_and_missing_states():
    output = StringIO()
    tui = DavidTui(DegradedRuntime(), color=False, output_stream=output)

    rc = tui.run(once="/status")

    assert rc == 0
    text = output.getvalue()
    assert "model validation: needs_review: manual attestation required [state=degraded]" in text
    assert "backend: offline: not loaded [state=degraded]" in text
    assert "index: missing: no workspace index for adapter [state=missing]" in text
    assert "source index: ready (12 files); live state missing [state=degraded]" in text
    assert "memory: missing: user, task [state=missing]" in text
    assert "resume: missing [state=missing]" in text


def test_verify_discovery_and_command_output_are_formatted():
    runtime = VerifyDiscoveryRuntime()
    tui = DavidTui(runtime, color=False, output_stream=StringIO())

    discovery = tui.dispatch("/verify").text
    command = tui.dispatch("/verify python -m py_compile src/chuk_lazarus/david/tui.py").text

    assert runtime.verify_calls == [None, "python -m py_compile src/chuk_lazarus/david/tui.py"]
    assert "David verification" in discovery
    assert "mode: candidate discovery" in discovery
    assert "result: passed" in discovery
    assert "selected commands:\n- none" in discovery
    assert "discovered candidate gates:\n- pytest-tui: python -m pytest tests/david/test_tui.py -q (focused TUI test)" in discovery
    assert "commands run:\n- none" in discovery
    assert "- candidate_count: 1" in discovery
    assert "David verification" in command
    assert "mode: command" in command
    assert "command: python -m py_compile src/chuk_lazarus/david/tui.py" in command
    assert "result: passed" in command
    assert "return code: 0" in command
    assert "stdout:\nok" in command


def test_agent_loop_output_includes_repair_summary_when_present():
    runtime = RepairRuntime()
    tui = DavidTui(runtime, color=False, output_stream=StringIO())

    result = tui.dispatch("/agent fix the failing TUI test").text

    assert "agent loop: verified steps=3 verified=True" in result
    assert "- 1: verify ok=False passed=False" in result
    assert "repair summary: failures=1 repairs=1 retries=0 recovery=verified_after_failure verification=passed" in result
    assert "- failed: step=1 action=verify reason=tests failed" in result
    assert "- repair: step=2 action=patch" in result
    assert runtime.agent_loop_calls == ["fix the failing TUI test"]
