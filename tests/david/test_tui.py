from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

from tests.david import require_attr, require_module


def _session(*, jit_required: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        session_id="tui-session",
        validation_status="accepted",
        can_auto_load=True,
        jit_required=jit_required,
        jit_actions=("jit_index_workspace",) if jit_required else (),
        index_readiness=SimpleNamespace(state="needs_jit" if jit_required else "ready"),
        user_memory=SimpleNamespace(state="ready"),
        task_memory=SimpleNamespace(state="ready"),
        warnings=(),
    )


def test_tui_status_and_tools_work_with_injected_streams() -> None:
    tui = require_module("chuk_lazarus.david.tui")
    run_tui = require_attr(tui, "run_tui", "stream-friendly terminal UI")

    class FakeRuntime:
        def __init__(self) -> None:
            self.session = _session(jit_required=True)

        def status(self) -> SimpleNamespace:
            return self.session

        def tools(self) -> list[str]:
            return ["apply_patch", "read_file", "search"]

        def respond(self, prompt: str) -> SimpleNamespace:  # pragma: no cover
            raise AssertionError(f"status/tools test should not run prompt: {prompt!r}")

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = run_tui(
        FakeRuntime(),
        input_stream=io.StringIO("/status\n/tools\n/quit\n"),
        output_stream=stdout,
        error_stream=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    output = stdout.getvalue().lower()
    assert "david" in output
    assert "accepted" in output
    assert "needs_jit" in output or "jit_index_workspace" in output
    assert "read_file" in output
    assert "search" in output
    for benchmark_name in ("mrcr", "ruler", "locobench", "swe-bench"):
        assert benchmark_name not in output


def test_tui_dispatches_prompt_and_prints_verification_summary() -> None:
    tui = require_module("chuk_lazarus.david.tui")
    run_tui = require_attr(tui, "run_tui", "prompt dispatch")

    class FakeRuntime:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def status(self) -> SimpleNamespace:
            return _session()

        def tools(self) -> list[str]:
            return ["read_file"]

        def respond(self, prompt: str) -> SimpleNamespace:
            self.prompts.append(prompt)
            return SimpleNamespace(
                answer="README.md contains alpha project.",
                verification_summary="verified: read_file trace captured",
                tool_results=[SimpleNamespace(name="read_file", ok=True)],
                events=[{"type": "verification", "status": "passed"}],
            )

    runtime = FakeRuntime()
    stdout = io.StringIO()

    exit_code = run_tui(
        runtime,
        input_stream=io.StringIO("Inspect README.md\n/quit\n"),
        output_stream=stdout,
        error_stream=io.StringIO(),
    )

    assert exit_code == 0
    assert runtime.prompts == ["Inspect README.md"]
    output = stdout.getvalue().lower()
    assert "alpha project" in output
    assert "verified" in output
