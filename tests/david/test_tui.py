from __future__ import annotations

from io import StringIO

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
    assert "prompt=hello" in text


def test_slash_commands_cover_terminal_surface():
    tui = DavidTui(FakeRuntime(), color=False, output_stream=StringIO())

    assert "David startup readiness" in tui.dispatch("/status").text
    assert "David startup readiness" in tui.dispatch("/readiness").text
    assert tui.dispatch("/memory").text == "memory detail"
    assert tui.dispatch("/index").text == "index detail"
    assert "jit indexed" in tui.dispatch("/index jit").text
    assert tui.runtime.index.jit_calls == 1
    assert tui.dispatch("/verify").text == "verified default"
    assert tui.dispatch("/verify pytest").text == "verified pytest"
    assert tui.dispatch("/run pytest -q").text == "ran pytest -q"
    assert tui.dispatch("/shell pytest -q").text == "ran pytest -q"
    assert tui.dispatch("/read src/app.py").text == "read src/app.py"
    assert tui.dispatch("/write src/app.py pass").text == "wrote src/app.py pass"
    assert "new\nline" in tui.dispatch("/apply new\\nline").text
    assert "new\nline" in tui.dispatch("/patch new\\nline").text
    assert "/readiness" in tui.dispatch("/help").text

    exit_result = tui.dispatch("/quit")
    assert exit_result.should_exit is True
    assert "bye" in exit_result.text
