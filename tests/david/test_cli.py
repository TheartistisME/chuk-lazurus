from __future__ import annotations

from pathlib import Path

from chuk_lazarus.david import cli
from chuk_lazarus.david.model_validation import ModelCommandResult


class FakeRuntime:
    created_with: cli.DavidConfig | None = None

    def __init__(self, config: cli.DavidConfig) -> None:
        self.config = config

    @classmethod
    def create(cls, config: cli.DavidConfig) -> "FakeRuntime":
        cls.created_with = config
        return cls(config)

    def readiness(self) -> dict[str, str]:
        return {
            "model validation": "ready",
            "index": "ready",
            "memory": "ready",
        }

    def respond(self, prompt: str) -> str:
        return f"answered: {prompt}"


def test_main_code_once_uses_workspace_and_prompt(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)

    rc = cli.main(["code", str(tmp_path), "--once", "summarize this repo", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    workspace = getattr(
        FakeRuntime.created_with,
        "workspace_path",
        getattr(FakeRuntime.created_with, "workspace_root"),
    )
    assert workspace == Path(tmp_path).resolve()
    output = capsys.readouterr().out
    assert "David startup readiness" in output
    assert "model validation: ready" in output
    assert "answered: summarize this repo" in output


def test_main_discovers_model_validation_report(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    workspace = tmp_path / "workspace"
    model = tmp_path / "model"
    workspace.mkdir()
    model.mkdir()
    report = model / "validation_report.json"
    report.write_text("{}", encoding="utf-8")

    rc = cli.main(["code", str(workspace), "--model", str(model), "--once", "/status", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.created_with.validation_report_path == str(report)


def test_main_explicit_validation_report_wins_over_discovery(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    workspace = tmp_path / "workspace"
    model = tmp_path / "model"
    workspace.mkdir()
    model.mkdir()
    discovered = model / "validation_report.json"
    explicit = tmp_path / "explicit-report.json"
    discovered.write_text("{}", encoding="utf-8")
    explicit.write_text("{}", encoding="utf-8")

    rc = cli.main(
        [
            "code",
            str(workspace),
            "--model",
            str(model),
            "--validation-report",
            str(explicit),
            "--once",
            "/status",
            "--no-color",
        ]
    )

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.created_with.validation_report_path == str(explicit)


def test_main_missing_validation_report_fails_closed_before_runtime(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    workspace = tmp_path / "workspace"
    model = tmp_path / "model"
    workspace.mkdir()
    model.mkdir()

    rc = cli.main(["code", str(workspace), "--model", str(model), "--once", "/status", "--no-color"])

    assert rc == 2
    assert FakeRuntime.created_with is None
    output = capsys.readouterr().out
    assert "model validation: blocked: no boot-safe validation report found" in output
    assert str(model / "validation_report.json") in output
    assert "--validation-report <path>" in output


def test_parser_keeps_code_subcommand_and_once_option():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "code",
            ".",
            "--model",
            "google/gemma-e2b",
            "--validation-report",
            "report.json",
            "--allow-unvalidated",
            "--once",
            "/status",
            "--auto-jit-index",
        ]
    )

    assert args.command == "code"
    assert args.workspace == "."
    assert args.model == "google/gemma-e2b"
    assert args.validation_report == "report.json"
    assert args.allow_unvalidated is True
    assert args.once == "/status"
    assert args.auto_jit_index is True


def test_parser_adds_explicit_model_scan_command():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "model",
            "scan",
            "google/gemma-e2b",
            "--output",
            "scan.json",
        ]
    )

    assert args.command == "model"
    assert args.model_command == "scan"
    assert args.model == "google/gemma-e2b"
    assert args.output == "scan.json"


def test_parser_adds_explicit_model_validate_command():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "model",
            "validate",
            "scan.json",
            "--output",
            "validated.json",
            "--model",
            "local-model",
        ]
    )

    assert args.command == "model"
    assert args.model_command == "validate"
    assert args.report == "scan.json"
    assert args.output == "validated.json"
    assert args.model == "local-model"


def test_model_scan_command_invokes_wrapper_without_runtime(monkeypatch, capsys):
    FakeRuntime.created_with = None
    calls = []

    def fake_scan(**kwargs):
        calls.append(kwargs)
        return ModelCommandResult(
            returncode=0,
            command=("python", "David/get_model_config.py"),
            stdout="scanner output\n",
            stderr="",
        )

    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    monkeypatch.setattr(cli, "run_model_scan", fake_scan)

    rc = cli.main(["model", "scan", "m", "--output", "scan.json"])

    assert rc == 0
    assert FakeRuntime.created_with is None
    assert calls == [
        {
            "model": "m",
            "output": Path("scan.json"),
        }
    ]
    assert "scanner output" in capsys.readouterr().out


def test_model_validate_failure_is_readable(monkeypatch, capsys):
    def fake_validate(**kwargs):
        return ModelCommandResult(
            returncode=7,
            command=("python", "David/validate_model_config.py", "--config-report", "scan.json"),
            stdout="partial stdout",
            stderr="bad report",
        )

    monkeypatch.setattr(cli, "run_model_validate", fake_validate)

    rc = cli.main(["model", "validate", "scan.json", "--output", "validated.json"])

    assert rc == 7
    err = capsys.readouterr().err
    assert "Model validation failed (rc=7)" in err
    assert "command: python David/validate_model_config.py --config-report scan.json" in err
    assert "bad report" in err
    assert "partial stdout" in err
