from __future__ import annotations

from pathlib import Path

from chuk_lazarus.david import cli


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
