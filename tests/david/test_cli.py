from __future__ import annotations

import io
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests.david import REPO_ROOT, assert_path_field, require_attr, require_module, value_at


def _write_validation_report(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_name": "lazarus.model_config_validation_report",
                "schema_version": 1,
                "validation_status": "accepted",
                "confidence": "high",
                "auto_load_allowed": True,
                "selected_config": {
                    "adapter_config_id": "gemma-e2b-test",
                    "kv_source_layer": 21,
                    "kv_target_layer": 23,
                    "insertion_family": "kv_direct",
                },
                "source_report_summary": {
                    "model_identity": "gemma-e2b-test",
                    "tokenizer_identity": "gemma-tokenizer-test",
                    "adapter_family": "gemma",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _session(*, jit_required: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        session_id="session-test",
        validation_status="accepted",
        can_auto_load=True,
        jit_required=jit_required,
        jit_actions=("jit_index_workspace",) if jit_required else (),
        index_readiness=SimpleNamespace(state="needs_jit" if jit_required else "ready"),
        user_memory=SimpleNamespace(state="ready"),
        task_memory=SimpleNamespace(state="ready"),
        warnings=(),
    )


def test_project_exposes_david_console_script() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert re.search(
        r'(?m)^david\s*=\s*"chuk_lazarus\.david\.cli:main"\s*$',
        pyproject,
    ), "pyproject.toml should install a runnable `david` console script."


def test_cli_main_runs_single_prompt_with_injected_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("CHUK_LAZARUS_DAVID_OFFLINE", "1")
    workspace = tmp_path / "workspace"
    model = tmp_path / "model"
    workspace.mkdir()
    model.mkdir()
    report = _write_validation_report(tmp_path / "validation.json")

    cli = require_module("chuk_lazarus.david.cli")
    main = require_attr(cli, "main", "an injected, testable CLI entrypoint")

    created: dict[str, Any] = {}

    class FakeRuntime:
        def __init__(self, config: Any, streams: Any = None) -> None:
            self.config = config
            self.streams = streams
            self.initialized = False
            self.prompts: list[str] = []

        def initialize(self) -> SimpleNamespace:
            self.initialized = True
            return _session()

        def run_once(self, prompt: str) -> SimpleNamespace:
            assert self.initialized
            self.prompts.append(prompt)
            return SimpleNamespace(
                answer="I inspected README.md",
                verification_summary="no tests requested",
                exit_code=0,
                events=[{"type": "verification", "status": "not_requested"}],
            )

    def runtime_factory(config: Any, streams: Any = None) -> FakeRuntime:
        runtime = FakeRuntime(config, streams)
        created["config"] = config
        created["streams"] = streams
        created["runtime"] = runtime
        return runtime

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        [
            "--workspace",
            str(workspace),
            "--model",
            str(model),
            "--validation-report",
            str(report),
            "--offline",
            "--no-shell",
            "--once",
            "inspect README.md",
        ],
        stdin=io.StringIO(),
        stdout=stdout,
        stderr=stderr,
        runtime_factory=runtime_factory,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert_path_field(created["config"], "workspace_path", workspace)
    assert_path_field(created["config"], "model_path", model)
    assert_path_field(created["config"], "validation_report_path", report)
    assert value_at(created["config"], "offline") is True
    allow_shell = value_at(
        created["config"],
        "allow_shell",
        value_at(created["config"], "shell_enabled", True),
    )
    assert allow_shell is False
    assert created["runtime"].prompts == ["inspect README.md"]
    output = stdout.getvalue().lower()
    assert "accepted" in output
    assert "inspected readme.md" in output


def test_cli_doctor_reports_readiness_without_running_a_turn(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    workspace = tmp_path / "workspace"
    model = tmp_path / "model"
    workspace.mkdir()
    model.mkdir()
    report = _write_validation_report(tmp_path / "validation.json")

    cli = require_module("chuk_lazarus.david.cli")
    main = require_attr(cli, "main", "doctor command execution")

    class FakeRuntime:
        def __init__(self, config: Any, streams: Any = None) -> None:
            self.config = config
            self.streams = streams

        def initialize(self) -> SimpleNamespace:
            assert value_at(self.config, "offline") is True
            return _session(jit_required=True)

        def run_once(self, prompt: str) -> SimpleNamespace:  # pragma: no cover
            raise AssertionError(f"doctor must not run a prompt: {prompt!r}")

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        [
            "doctor",
            "--workspace",
            str(workspace),
            "--model",
            str(model),
            "--validation-report",
            str(report),
            "--offline",
        ],
        stdin=io.StringIO(),
        stdout=stdout,
        stderr=stderr,
        runtime_factory=lambda config, streams=None: FakeRuntime(config, streams),
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    output = stdout.getvalue().lower()
    assert "workspace" in output
    assert "accepted" in output
    assert "needs_jit" in output or "jit_index_workspace" in output


def test_cli_rejects_missing_workspace_before_runtime_factory(tmp_path: Path) -> None:
    missing_workspace = tmp_path / "missing"
    model = tmp_path / "model"
    model.mkdir()
    report = _write_validation_report(tmp_path / "validation.json")

    cli = require_module("chuk_lazarus.david.cli")
    main = require_attr(cli, "main", "preflight argument validation")

    def runtime_factory(config: Any, streams: Any = None) -> Any:  # pragma: no cover
        raise AssertionError("runtime_factory should not be called for a bad workspace")

    stderr = io.StringIO()
    exit_code = main(
        [
            "--workspace",
            str(missing_workspace),
            "--model",
            str(model),
            "--validation-report",
            str(report),
            "--offline",
            "--once",
            "hello",
        ],
        stdin=io.StringIO(),
        stdout=io.StringIO(),
        stderr=stderr,
        runtime_factory=runtime_factory,
    )

    assert exit_code != 0
    assert "workspace" in stderr.getvalue().lower()
