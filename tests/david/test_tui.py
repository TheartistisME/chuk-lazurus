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


def test_once_outputs_startup_readiness_and_prompt_response():
    output = StringIO()
    tui = DavidTui(FakeRuntime(), color=False, output_stream=output)

    rc = tui.run(once="hello")

    assert rc == 0
    text = output.getvalue()
    assert "David terminal agent" in text
    assert "model validation: ready" in text
    assert "index: warm" in text
    assert "memory: hot" in text
    assert "David resume" in text
    assert "session: session-1" in text
    assert "workspace: C:/workspace/project" in text
    assert "updated: 2026-05-04T01:02:03+00:00" in text
    assert "last result: patched the David TUI" in text
    assert "prompt=hello" in text


def test_slash_commands_cover_terminal_surface():
    tui = DavidTui(FakeRuntime(), color=False, output_stream=StringIO())

    assert "David startup readiness" in tui.dispatch("/status").text
    assert "David startup readiness" in tui.dispatch("/readiness").text
    assert "last result: patched the David TUI" in tui.dispatch("/resume").text
    assert tui.dispatch("/memory").text == "memory detail"
    assert tui.dispatch("/index").text == "index detail"
    assert "jit indexed" in tui.dispatch("/index jit").text
    assert tui.runtime.index.jit_calls == 1
    assert tui.dispatch("/verify").text == "verified default"
    assert tui.dispatch("/verify pytest").text == "verified pytest"
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
