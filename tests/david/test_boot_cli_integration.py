from __future__ import annotations

from pathlib import Path

from chuk_lazarus.david import cli


class FakeRuntime:
    created_with: object | None = None

    @classmethod
    def create(cls, config: object) -> "FakeRuntime":
        cls.created_with = config
        return cls()

    def readiness(self) -> dict[str, str]:
        return {
            "model validation": "ready",
            "index": "ready",
            "memory": "ready",
        }

    def respond(self, prompt: str) -> str:
        return f"fake response: {prompt}"


def test_main_preserves_boot_config_fields_with_real_config(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)

    report_path = tmp_path / "validation.json"
    rc = cli.main(
        [
            "code",
            str(tmp_path),
            "--model",
            "google/gemma-e2b",
            "--validation-report",
            str(report_path),
            "--allow-unvalidated",
            "--once",
            "inspect boot config",
            "--verify-command",
            "python -m pytest",
            "--timeout",
            "17",
            "--no-color",
            "--auto-jit-index",
        ]
    )

    assert rc == 0
    config = FakeRuntime.created_with
    assert isinstance(config, cli.DavidConfig)
    assert config.workspace_root == Path(tmp_path).resolve()
    assert config.workspace_path == Path(tmp_path).resolve()
    assert config.model_path == "google/gemma-e2b"
    assert config.validation_report_path == str(report_path)
    assert config.require_validated_model is False
    assert config.once == "inspect boot config"
    assert config.verify_command == "python -m pytest"
    assert config.command_timeout_seconds == 17
    assert config.color is False
    assert config.auto_jit_index is True

    output = capsys.readouterr().out
    assert "fake response: inspect boot config" in output
