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


def test_parser_keeps_code_subcommand_and_once_option():
    parser = cli.build_parser()

    args = parser.parse_args(["code", ".", "--once", "/status"])

    assert args.command == "code"
    assert args.workspace == "."
    assert args.once == "/status"
