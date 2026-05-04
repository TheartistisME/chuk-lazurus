from __future__ import annotations

import json
from pathlib import Path

import pytest

from chuk_lazarus.david import cli
from chuk_lazarus.david.doctor import DavidDoctorReport, DoctorCheck
from chuk_lazarus.david.resume import SessionSnapshot, save_session_snapshot
from chuk_lazarus.david.model_validation import (
    AutoModelValidationResult,
    ModelCommandResult,
    ValidationReportDiscovery,
)
from chuk_lazarus.david.model_onboarding import ModelOnboardingPlan, ModelOnboardingResult


@pytest.fixture(autouse=True)
def clear_operator_env(monkeypatch):
    for name in (
        "DAVID_MODEL",
        "DAVID_VALIDATION_REPORT",
        "DAVID_MODEL_ATTESTATION",
        "DAVID_MODEL_BACKEND",
        "DAVID_MODEL_DEVICE",
        "DAVID_MODEL_DTYPE",
        "DAVID_MODEL_MAX_NEW_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


class FakeRuntime:
    created_with: cli.DavidConfig | None = None
    verified_command: str | None = None
    patch_verified: bool = False
    index_action: str | None = None

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

    def verify(self, command: str | None = None) -> str:
        type(self).verified_command = command
        if command == "fail":
            return "$ fail\nrc=7\nfailed"
        if command is not None:
            return f"$ {command}\nrc=0\nverified"
        return (
            "David verification\n"
            "mode: candidate discovery\n"
            "selected commands:\n"
            "- none\n"
            "discovered candidate gates:\n"
            "- python -m pytest -q\n"
            "commands run:\n"
            "- none\n"
            "reason: candidate gates were discovered but not executed without --cmd\n"
            "verification metadata:\n"
            "- ok: true"
        )

    def verify_patch(self) -> str:
        type(self).patch_verified = True
        type(self).verified_command = None
        return (
            "David verification\n"
            "mode: built-in patch verification\n"
            "selected commands:\n"
            "- python -m pytest -q\n"
            "discovered candidate gates:\n"
            "- python -m pytest -q\n"
            "commands run:\n"
            "- none\n"
            "reason: built-in verification selected candidate gates but did not execute them without --cmd\n"
            "verification metadata:\n"
            "- capability: repo_patch\n"
            "- ok: true"
        )

    def index_status(self) -> str:
        type(self).index_action = "status"
        return "index: ready at fake-manifest"

    def build_index(self) -> str:
        type(self).index_action = "build"
        return "index: built fake workspace"

    def refresh_index(self) -> str:
        type(self).index_action = "refresh"
        return "index: refreshed fake workspace"

    def memory_status(self) -> str:
        return "memory:\n- user artifact: fake-user.jsonl\n- task artifact: fake-task.jsonl"


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


def test_main_keeps_common_options_before_code_subcommand(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    report = tmp_path / "validation-report.json"
    monkeypatch.setenv("DAVID_MODEL", "env-model")
    monkeypatch.setenv("DAVID_VALIDATION_REPORT", "env-report.json")
    monkeypatch.setenv("DAVID_MODEL_ATTESTATION", "env-attestation.json")
    monkeypatch.setenv("DAVID_MODEL_BACKEND", "transformers")

    rc = cli.main(
        [
            "--model",
            "root-model",
            "--validation-report",
            str(report),
            "--model-attestation",
            "review.json",
            "--model-backend",
            "torch-runtime",
            "--model-device",
            "cuda:0",
            "--model-dtype",
            "bfloat16",
            "--model-max-new-tokens",
            "321",
            "--once",
            "/status",
            "--no-color",
            "code",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.created_with.model_path == "root-model"
    assert FakeRuntime.created_with.validation_report_path == str(report)
    assert FakeRuntime.created_with.model_attestation_path == "review.json"
    assert FakeRuntime.created_with.model_backend == "torch-runtime"
    assert FakeRuntime.created_with.model_device == "cuda:0"
    assert FakeRuntime.created_with.model_dtype == "bfloat16"
    assert FakeRuntime.created_with.model_max_new_tokens == 321
    assert FakeRuntime.created_with.once == "/status"
    assert FakeRuntime.created_with.color is False
    assert "model validation: ready" in capsys.readouterr().out


def test_main_code_once_uses_operator_env_defaults(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    monkeypatch.setenv("DAVID_MODEL", "env-model")
    monkeypatch.setenv("DAVID_VALIDATION_REPORT", "env-report.json")
    monkeypatch.setenv("DAVID_MODEL_ATTESTATION", "env-attestation.json")
    monkeypatch.setenv("DAVID_MODEL_BACKEND", "torch-runtime")
    monkeypatch.setenv("DAVID_MODEL_DEVICE", "cpu")
    monkeypatch.setenv("DAVID_MODEL_DTYPE", "float32")
    monkeypatch.setenv("DAVID_MODEL_MAX_NEW_TOKENS", "77")

    rc = cli.main(["code", str(tmp_path), "--once", "/status", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.created_with.model_path == "env-model"
    assert FakeRuntime.created_with.validation_report_path == "env-report.json"
    assert FakeRuntime.created_with.model_attestation_path == "env-attestation.json"
    assert FakeRuntime.created_with.model_backend == "torch-runtime"
    assert FakeRuntime.created_with.model_device == "cpu"
    assert FakeRuntime.created_with.model_dtype == "float32"
    assert FakeRuntime.created_with.model_max_new_tokens == 77
    assert "model validation: ready" in capsys.readouterr().out


def test_main_code_once_uses_workspace_config_defaults(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    workspace = tmp_path / "workspace"
    model = workspace / "models" / "gemma"
    workspace_config = workspace / ".david" / "config.json"
    workspace_config.parent.mkdir(parents=True)
    model.mkdir(parents=True)
    workspace_config.write_text(
        json.dumps(
            {
                "model": "models/gemma",
                "validation_report": "reports/validation.json",
                "model_attestation": "review/attestation.json",
                "model_backend": "torch-runtime",
                "model_device": "cpu",
                "model_dtype": "float32",
                "model_max_new_tokens": 88,
                "auto_jit_index": True,
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(["code", str(workspace), "--once", "/status", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.created_with.model_path == str(model)
    assert FakeRuntime.created_with.validation_report_path == str(workspace / "reports" / "validation.json")
    assert FakeRuntime.created_with.model_attestation_path == str(workspace / "review" / "attestation.json")
    assert FakeRuntime.created_with.model_backend == "torch-runtime"
    assert FakeRuntime.created_with.model_device == "cpu"
    assert FakeRuntime.created_with.model_dtype == "float32"
    assert FakeRuntime.created_with.model_max_new_tokens == 88
    assert FakeRuntime.created_with.auto_jit_index is True
    assert "model validation: ready" in capsys.readouterr().out


def test_main_workspace_config_preserves_hf_model_id(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    workspace_config = tmp_path / ".david" / "config.json"
    workspace_config.parent.mkdir(parents=True)
    workspace_config.write_text(
        json.dumps(
            {
                "model": "google/gemma-e2b",
                "validation_report": "validation.json",
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(["code", str(tmp_path), "--once", "/status", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.created_with.model_path == "google/gemma-e2b"


def test_main_cli_args_override_workspace_config(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    workspace_config = tmp_path / ".david" / "config.json"
    workspace_config.parent.mkdir(parents=True)
    workspace_config.write_text(
        json.dumps(
            {
                "model": "config-model",
                "validation_report": "config-report.json",
                "model_attestation": "config-attestation.json",
                "model_backend": "config-backend",
                "model_device": "cpu",
                "model_dtype": "float32",
                "model_max_new_tokens": 88,
                "auto_jit_index": True,
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "code",
            str(tmp_path),
            "--model",
            "cli-model",
            "--validation-report",
            "cli-report.json",
            "--model-attestation",
            "cli-attestation.json",
            "--model-backend",
            "cli-backend",
            "--model-device",
            "cuda:0",
            "--model-dtype",
            "bfloat16",
            "--model-max-new-tokens",
            "44",
            "--once",
            "/status",
            "--no-color",
        ]
    )

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.created_with.model_path == "cli-model"
    assert FakeRuntime.created_with.validation_report_path == "cli-report.json"
    assert FakeRuntime.created_with.model_attestation_path == "cli-attestation.json"
    assert FakeRuntime.created_with.model_backend == "cli-backend"
    assert FakeRuntime.created_with.model_device == "cuda:0"
    assert FakeRuntime.created_with.model_dtype == "bfloat16"
    assert FakeRuntime.created_with.model_max_new_tokens == 44
    assert FakeRuntime.created_with.auto_jit_index is True


def test_main_env_overrides_workspace_config(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    workspace_config = tmp_path / ".david" / "config.json"
    workspace_config.parent.mkdir(parents=True)
    workspace_config.write_text(
        json.dumps(
            {
                "model": "config-model",
                "validation_report": "config-report.json",
                "model_attestation": "config-attestation.json",
                "model_backend": "config-backend",
                "model_device": "cpu",
                "model_dtype": "float32",
                "model_max_new_tokens": 88,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DAVID_MODEL", "env-model")
    monkeypatch.setenv("DAVID_VALIDATION_REPORT", "env-report.json")
    monkeypatch.setenv("DAVID_MODEL_ATTESTATION", "env-attestation.json")
    monkeypatch.setenv("DAVID_MODEL_BACKEND", "env-backend")
    monkeypatch.setenv("DAVID_MODEL_DEVICE", "cuda")
    monkeypatch.setenv("DAVID_MODEL_DTYPE", "float16")
    monkeypatch.setenv("DAVID_MODEL_MAX_NEW_TOKENS", "99")

    rc = cli.main(["code", str(tmp_path), "--once", "/status", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.created_with.model_path == "env-model"
    assert FakeRuntime.created_with.validation_report_path == "env-report.json"
    assert FakeRuntime.created_with.model_attestation_path == "env-attestation.json"
    assert FakeRuntime.created_with.model_backend == "env-backend"
    assert FakeRuntime.created_with.model_device == "cuda"
    assert FakeRuntime.created_with.model_dtype == "float16"
    assert FakeRuntime.created_with.model_max_new_tokens == 99


def test_main_malformed_workspace_config_fails_closed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    workspace_config = tmp_path / ".david" / "config.json"
    workspace_config.parent.mkdir(parents=True)
    workspace_config.write_text('{"model": ', encoding="utf-8")

    rc = cli.main(["code", str(tmp_path), "--once", "/status", "--no-color"])

    assert rc == 2
    assert FakeRuntime.created_with is None
    err = capsys.readouterr().err
    assert "David workspace config error" in err
    assert str(workspace_config) in err
    assert "invalid JSON" in err


def test_main_workspace_config_model_without_report_fails_closed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    workspace = tmp_path / "workspace"
    model = tmp_path / "model"
    workspace_config = workspace / ".david" / "config.json"
    workspace_config.parent.mkdir(parents=True)
    model.mkdir(parents=True)
    workspace_config.write_text(json.dumps({"model": str(model)}), encoding="utf-8")

    rc = cli.main(["code", str(workspace), "--once", "/status", "--no-color"])

    assert rc == 2
    assert FakeRuntime.created_with is None
    output = capsys.readouterr().out
    assert "model validation: blocked: no boot-safe validation report found" in output
    assert str(model / "validation_report.json") in output


def test_main_keeps_common_options_after_workspace(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    report = tmp_path / "validation-report.json"

    rc = cli.main(
        [
            "code",
            str(tmp_path),
            "--model",
            "workspace-tail-model",
            "--validation-report",
            str(report),
            "--model-attestation",
            "tail-review.json",
            "--model-backend",
            "torch-runtime",
            "--once",
            "/status",
            "--timeout",
            "9",
            "--no-color",
            "--auto-jit-index",
            "--model-device",
            "cpu",
            "--model-dtype",
            "float32",
            "--model-max-new-tokens",
            "42",
        ]
    )

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.created_with.model_path == "workspace-tail-model"
    assert FakeRuntime.created_with.validation_report_path == str(report)
    assert FakeRuntime.created_with.model_attestation_path == "tail-review.json"
    assert FakeRuntime.created_with.model_backend == "torch-runtime"
    assert FakeRuntime.created_with.once == "/status"
    assert FakeRuntime.created_with.command_timeout_seconds == 9
    assert FakeRuntime.created_with.auto_jit_index is True
    assert FakeRuntime.created_with.model_device == "cpu"
    assert FakeRuntime.created_with.model_dtype == "float32"
    assert FakeRuntime.created_with.model_max_new_tokens == 42
    assert "index: ready" in capsys.readouterr().out


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


def test_main_auto_validates_model_into_workspace_before_boot(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    validation_report = workspace / ".david" / "model_validation" / "model_validation_report.json"
    calls = []

    def fake_auto_validate(**kwargs):
        calls.append(kwargs)
        return AutoModelValidationResult(
            scan_report_path=workspace / ".david" / "model" / "model_config_report.json",
            validation_report_path=validation_report,
            scan_result=ModelCommandResult(
                returncode=0,
                command=("python", "David/get_model_config.py"),
                stdout="",
                stderr="",
            ),
            validation_result=ModelCommandResult(
                returncode=0,
                command=("python", "David/validate_model_config.py"),
                stdout="",
                stderr="",
            ),
        )

    monkeypatch.setattr(cli, "run_auto_model_validation", fake_auto_validate)

    rc = cli.main(
        [
            "code",
            str(workspace),
            "--model",
            "google/gemma-e2b",
            "--auto-validate-model",
            "--once",
            "/status",
            "--no-color",
        ]
    )

    assert rc == 0
    assert calls == [{"model": "google/gemma-e2b", "workspace_path": workspace}]
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.created_with.validation_report_path == str(validation_report)


def test_main_auto_validation_scan_failure_blocks_runtime(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fake_auto_validate(**_kwargs):
        return AutoModelValidationResult(
            scan_report_path=workspace / ".david" / "model" / "model_config_report.json",
            validation_report_path=workspace / ".david" / "model_validation" / "model_validation_report.json",
            scan_result=ModelCommandResult(
                returncode=7,
                command=("python", "David/get_model_config.py"),
                stdout="partial scan",
                stderr="scan exploded",
            ),
            validation_result=None,
        )

    monkeypatch.setattr(cli, "run_auto_model_validation", fake_auto_validate)

    rc = cli.main(
        [
            "code",
            str(workspace),
            "--model",
            "google/gemma-e2b",
            "--auto-validate-model",
            "--once",
            "/status",
            "--no-color",
        ]
    )

    assert rc == 7
    assert FakeRuntime.created_with is None
    err = capsys.readouterr().err
    assert "Auto model scan failed (rc=7)" in err
    assert "scan exploded" in err


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
    assert "--auto-validate-model" in output


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
            "--model-attestation",
            "attestation.json",
            "--allow-unvalidated",
            "--once",
            "/status",
            "--auto-jit-index",
            "--auto-validate-model",
            "--model-backend",
            "torch-runtime",
            "--model-device",
            "cuda",
            "--model-dtype",
            "float16",
            "--model-max-new-tokens",
            "99",
        ]
    )

    assert args.command == "code"
    assert args.workspace == "."
    assert args.model == "google/gemma-e2b"
    assert args.validation_report == "report.json"
    assert args.model_attestation == "attestation.json"
    assert args.allow_unvalidated is True
    assert args.once == "/status"
    assert args.auto_jit_index is True
    assert args.auto_validate_model is True
    assert args.model_backend == "torch-runtime"
    assert args.model_device == "cuda"
    assert args.model_dtype == "float16"
    assert args.model_max_new_tokens == 99


def test_parser_adds_verify_subcommand():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "verify",
            ".",
            "--cmd",
            "python -c \"print('ok')\"",
            "--allow-unvalidated",
            "--no-color",
        ]
    )

    assert args.command == "verify"
    assert args.workspace == "."
    assert args.cmd == "python -c \"print('ok')\""
    assert args.allow_unvalidated is True
    assert args.no_color is True


def test_verify_help_says_discovery_does_not_execute_without_cmd(capsys):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["verify", "--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "Discover candidate quality gates and report them without running them." in output
    assert "only executes workspace verification when --cmd is supplied" in output
    assert "candidates are not executed without --cmd" in output


def test_parser_adds_index_memory_resume_subcommands():
    parser = cli.build_parser()

    index_args = parser.parse_args(["index", ".", "--build", "--allow-unvalidated", "--no-color"])
    memory_args = parser.parse_args(["memory", ".", "--allow-unvalidated"])
    resume_args = parser.parse_args(["resume", ".", "--allow-unvalidated"])
    capabilities_args = parser.parse_args(["capabilities"])

    assert index_args.command == "index"
    assert index_args.workspace == "."
    assert index_args.build is True
    assert memory_args.command == "memory"
    assert memory_args.workspace == "."
    assert resume_args.command == "resume"
    assert resume_args.workspace == "."
    assert capabilities_args.command == "capabilities"


def test_parser_adds_init_command():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "init",
            ".",
            "--force",
            "--model",
            "google/gemma-e2b",
            "--model-backend",
            "torch-runtime",
            "--model-device",
            "cpu",
            "--model-dtype",
            "float32",
            "--model-max-new-tokens",
            "55",
            "--auto-jit-index",
        ]
    )

    assert args.command == "init"
    assert args.workspace == "."
    assert args.force is True
    assert args.model == "google/gemma-e2b"
    assert args.model_backend == "torch-runtime"
    assert args.model_device == "cpu"
    assert args.model_dtype == "float32"
    assert args.model_max_new_tokens == 55
    assert args.auto_jit_index is True


def test_init_command_invokes_workspace_initializer_without_runtime(monkeypatch, tmp_path, capsys):
    FakeRuntime.created_with = None
    calls = []

    def fake_initialize(workspace_path, **kwargs):
        calls.append({"workspace_path": workspace_path, **kwargs})
        david_dir = tmp_path / ".david"
        return cli.WorkspaceInitResult(
            workspace_root=tmp_path.resolve(),
            david_dir=david_dir,
            config_path=david_dir / "config.json",
            next_steps_path=david_dir / "NEXT_STEPS.md",
            config={"model": kwargs["model"]},
            created_paths=(david_dir,),
            existing_paths=(david_dir / "memory",),
            overwritten_paths=(david_dir / "config.json",),
            next_steps_summary="next",
        )

    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    monkeypatch.setattr(cli, "initialize_workspace", fake_initialize)

    rc = cli.main(
        [
            "init",
            str(tmp_path),
            "--force",
            "--model",
            "google/gemma-e2b",
            "--model-backend",
            "torch-runtime",
            "--model-device",
            "cpu",
            "--model-dtype",
            "float32",
            "--model-max-new-tokens",
            "55",
            "--auto-jit-index",
        ]
    )

    assert rc == 0
    assert FakeRuntime.created_with is None
    assert calls == [
        {
            "workspace_path": Path(tmp_path),
            "force": True,
            "model": "google/gemma-e2b",
            "backend": "torch-runtime",
            "device": "cpu",
            "dtype": "float32",
            "max_new_tokens": 55,
            "auto_jit_index": True,
        }
    ]
    output = capsys.readouterr().out
    assert "David init" in output
    assert f"workspace: {tmp_path.resolve()}" in output
    assert "created paths:" in output
    assert "skipped paths:" in output
    assert "overwritten paths:" in output
    assert "david model onboard <model> --workspace <workspace>" in output


def test_init_command_explicit_args_override_operator_env(monkeypatch, tmp_path):
    calls = []

    def fake_initialize(workspace_path, **kwargs):
        calls.append({"workspace_path": workspace_path, **kwargs})
        david_dir = tmp_path / ".david"
        return cli.WorkspaceInitResult(
            workspace_root=tmp_path.resolve(),
            david_dir=david_dir,
            config_path=david_dir / "config.json",
            next_steps_path=david_dir / "NEXT_STEPS.md",
            config={},
            created_paths=(),
            existing_paths=(),
            overwritten_paths=(),
            next_steps_summary="",
        )

    monkeypatch.setenv("DAVID_MODEL", "env-model")
    monkeypatch.setenv("DAVID_MODEL_BACKEND", "env-backend")
    monkeypatch.setattr(cli, "initialize_workspace", fake_initialize)

    rc = cli.main(
        [
            "--model",
            "root-model",
            "init",
            str(tmp_path),
            "--model-backend",
            "cli-backend",
        ]
    )

    assert rc == 0
    assert calls[0]["model"] == "root-model"
    assert calls[0]["backend"] == "cli-backend"


def test_capabilities_command_prints_truthful_status_without_runtime(monkeypatch, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None

    rc = cli.main(["capabilities"])

    assert rc == 0
    assert FakeRuntime.created_with is None
    output = capsys.readouterr().out
    assert "David capabilities" in output
    assert "WIRED:" in output
    assert "CLI/TUI terminal surface" in output
    assert "boot harness and startup readiness gates" in output
    assert "validation and model attestation checks" in output
    assert "torch standard decode path guarded by validation/attestation" in output
    assert "doctor readiness surface with WSL path checks when available" in output
    assert "capability router surface" in output
    assert "materializer metadata and compatibility guards" in output
    assert "residual-sidecar catalog/readiness and replay metadata checks" in output
    assert "decoder prior store scope checks and steering metadata" in output
    assert "quality gate discovery surface that does not run candidates without --cmd" in output
    assert "structured verifier checks for route, evidence, repair, sidecar, and decoder-prior metadata" in output
    assert "bounded agent loop read/patch/verify shell" in output
    assert "GUARDED/PARTIAL:" in output
    assert "model-driven repo autonomy remains guarded" in output
    assert "repo patching is partially wired through guarded read -> patch -> verify" in output
    assert "quality gate execution requires explicit --cmd" in output
    assert "residual-sidecar tensor replay is opt-in and fail-closed behind manifest/scope evidence" in output
    assert "decoder prior live steering requires a compatible loaded backend/tokenizer and scoped prior" in output
    assert "KV/direct tensor replay remains guarded research/runtime plumbing, not a production auto-load path" in output
    assert "TODO:" in output
    assert "real Gemma CUDA smoke against a validated model" in output
    assert "full central router product wiring" in output
    assert "real activation/residual/KV indexes" in output
    assert "adapter-safe production KV-direct replay/materialization" in output
    assert "broader live steering backend coverage" in output
    assert "full behavioral verifier evidence beyond structured metadata checks" in output
    assert "broad multi-step autonomous repo patching" in output
    assert "tensor replay wired" not in output.lower()
    assert "production kv-direct replay wired" not in output.lower()


def test_verify_command_prints_output_and_returns_command_rc(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    FakeRuntime.verified_command = None
    FakeRuntime.patch_verified = False

    rc = cli.main(["verify", str(tmp_path), "--cmd", "echo ok", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.verified_command == "echo ok"
    assert FakeRuntime.patch_verified is False
    output = capsys.readouterr().out
    assert "$ echo ok" in output
    assert "rc=0" in output
    assert "verified" in output


def test_verify_command_returns_nonzero_command_rc(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    FakeRuntime.verified_command = None
    FakeRuntime.patch_verified = False

    rc = cli.main(["verify", str(tmp_path), "--cmd", "fail", "--allow-unvalidated", "--no-color"])

    assert rc == 7
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.verified_command == "fail"
    assert FakeRuntime.patch_verified is False
    assert "rc=7" in capsys.readouterr().out


def test_verify_patch_runs_builtin_verifier_without_tui(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    FakeRuntime.verified_command = "stale"
    FakeRuntime.patch_verified = False

    rc = cli.main(["verify", str(tmp_path), "--patch", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.verified_command is None
    assert FakeRuntime.patch_verified is True
    output = capsys.readouterr().out
    assert "mode: built-in patch verification" in output
    assert "selected commands:" in output
    assert "discovered candidate gates:" in output
    assert "commands run:" in output
    assert "verification metadata:" in output
    assert "capability: repo_patch" in output


def test_index_status_command_calls_runtime_without_tui(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    FakeRuntime.index_action = None

    rc = cli.main(["index", str(tmp_path), "--status", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.index_action == "status"
    assert "index: ready at fake-manifest" in capsys.readouterr().out


def test_index_build_command_calls_runtime_build_without_tui(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    FakeRuntime.index_action = None

    rc = cli.main(["index", str(tmp_path), "--build", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.index_action == "build"
    assert "index: built fake workspace" in capsys.readouterr().out


def test_index_refresh_command_calls_runtime_refresh_without_tui(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    FakeRuntime.index_action = None

    rc = cli.main(["index", str(tmp_path), "--refresh", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    assert FakeRuntime.index_action == "refresh"
    assert "index: refreshed fake workspace" in capsys.readouterr().out


def test_memory_command_prints_user_and_task_artifacts(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None

    rc = cli.main(["memory", str(tmp_path), "--allow-unvalidated", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    output = capsys.readouterr().out
    assert "user artifact: fake-user.jsonl" in output
    assert "task artifact: fake-task.jsonl" in output


def test_resume_command_prints_no_saved_session_without_tui(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None

    rc = cli.main(["resume", str(tmp_path), "--allow-unvalidated", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    output = capsys.readouterr().out
    assert "David resume" in output
    assert "status: no saved session" in output


def test_resume_command_prints_valid_saved_session_without_tui(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    save_session_snapshot(
        SessionSnapshot(
            session_id="session-cli",
            workspace=str(tmp_path),
            adapter_scope={"model_id": "offline"},
            memory_paths={},
            updated_at="2026-05-04T01:02:03+00:00",
            last_result_summary="finished safely",
        )
    )

    rc = cli.main(["resume", str(tmp_path), "--allow-unvalidated", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    output = capsys.readouterr().out
    assert "David resume" in output
    assert "session: session-cli" in output
    assert f"workspace: {tmp_path}" in output
    assert "last result: finished safely" in output
    assert "no saved session" not in output
    assert "corrupt saved session" not in output


def test_resume_command_reports_corrupt_saved_session_without_tui(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    resume_path = tmp_path / ".david" / "resume.json"
    resume_path.parent.mkdir()
    resume_path.write_text('{"schema": ', encoding="utf-8")

    rc = cli.main(["resume", str(tmp_path), "--allow-unvalidated", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    output = capsys.readouterr().out
    assert "David resume" in output
    assert "status: corrupt saved session" in output
    assert str(resume_path) in output
    assert "invalid JSON at line 1, column 12" in output


def test_resume_command_reports_unreadable_saved_session_without_tui(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    FakeRuntime.created_with = None
    resume_path = tmp_path / ".david" / "resume.json"
    resume_path.mkdir(parents=True)

    rc = cli.main(["resume", str(tmp_path), "--allow-unvalidated", "--no-color"])

    assert rc == 0
    assert FakeRuntime.created_with is not None
    output = capsys.readouterr().out
    assert "David resume" in output
    assert "status: unreadable saved session" in output
    assert str(resume_path) in output


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


def test_parser_adds_model_onboard_command():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "model",
            "onboard",
            "google/gemma-e2b",
            "--workspace",
            ".",
            "--execute",
        ]
    )

    assert args.command == "model"
    assert args.model_command == "onboard"
    assert args.model == "google/gemma-e2b"
    assert args.workspace == "."
    assert args.execute is True


def test_parser_adds_doctor_command():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "doctor",
            "--model",
            "google/gemma-e2b",
            "--workspace",
            ".",
            "--model-attestation",
            "review.json",
            "--auto-validate-model",
        ]
    )

    assert args.command == "doctor"
    assert args.model == "google/gemma-e2b"
    assert args.workspace == "."
    assert args.model_attestation == "review.json"
    assert args.auto_validate_model is True


def test_parser_preserves_root_doctor_common_options():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "--model",
            "root-model",
            "--validation-report",
            "root-report.json",
            "--model-attestation",
            "root-attestation.json",
            "doctor",
            "--workspace",
            ".",
        ]
    )

    assert args.command == "doctor"
    assert args.model == "root-model"
    assert args.validation_report == "root-report.json"
    assert args.model_attestation == "root-attestation.json"


def test_doctor_command_prints_report_without_runtime(monkeypatch, tmp_path, capsys):
    FakeRuntime.created_with = None
    calls = []

    def fake_doctor(**kwargs):
        calls.append(kwargs)
        return DavidDoctorReport(
            model=kwargs["model"],
            workspace_path=tmp_path.resolve(),
            checks=(
                DoctorCheck("HF snapshot", "ready", "cache complete"),
                DoctorCheck("validation report", "missing", "no report"),
            ),
            validation_discovery=ValidationReportDiscovery(path=None, checked_paths=(tmp_path / "missing.json",)),
        )

    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    monkeypatch.setattr(cli, "run_doctor", fake_doctor)

    rc = cli.main(
        [
            "doctor",
            "--model",
            "google/gemma-e2b",
            "--workspace",
            str(tmp_path),
            "--validation-report",
            "report.json",
            "--model-attestation",
            "attestation.json",
            "--auto-validate-model",
        ]
    )

    assert rc == 2
    assert FakeRuntime.created_with is None
    assert calls == [
        {
            "model": "google/gemma-e2b",
            "workspace_path": Path(tmp_path),
            "validation_report": "report.json",
            "attestation_path": "attestation.json",
            "auto_validate_model": True,
        }
    ]
    output = capsys.readouterr().out
    assert "David model doctor" in output
    assert "HF snapshot: ready: cache complete" in output
    assert "validation report: missing: no report" in output


def test_doctor_command_uses_workspace_config_defaults(monkeypatch, tmp_path, capsys):
    FakeRuntime.created_with = None
    calls = []
    workspace = tmp_path / "workspace"
    model = workspace / "models" / "gemma"
    workspace_config = workspace / ".david" / "config.json"
    workspace_config.parent.mkdir(parents=True)
    model.mkdir(parents=True)
    workspace_config.write_text(
        json.dumps(
            {
                "model": "models/gemma",
                "validation_report": "reports/validation.json",
                "model_attestation": "review/attestation.json",
            }
        ),
        encoding="utf-8",
    )

    def fake_doctor(**kwargs):
        calls.append(kwargs)
        return DavidDoctorReport(
            model=kwargs["model"],
            workspace_path=workspace.resolve(),
            checks=(DoctorCheck("validation report", "ready", "config default"),),
            validation_discovery=ValidationReportDiscovery(
                path=Path(kwargs["validation_report"]),
                checked_paths=(),
            ),
        )

    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    monkeypatch.setattr(cli, "run_doctor", fake_doctor)

    rc = cli.main(["doctor", "--workspace", str(workspace)])

    assert rc == 0
    assert FakeRuntime.created_with is None
    assert calls == [
        {
            "model": str(model),
            "workspace_path": workspace,
            "validation_report": str(workspace / "reports" / "validation.json"),
            "attestation_path": str(workspace / "review" / "attestation.json"),
            "auto_validate_model": False,
        }
    ]
    assert "validation report: ready: config default" in capsys.readouterr().out


def test_doctor_command_cli_flags_override_workspace_config(monkeypatch, tmp_path):
    FakeRuntime.created_with = None
    calls = []
    workspace_config = tmp_path / ".david" / "config.json"
    workspace_config.parent.mkdir(parents=True)
    workspace_config.write_text(
        json.dumps(
            {
                "model": "config-model",
                "validation_report": "config-report.json",
                "model_attestation": "config-attestation.json",
            }
        ),
        encoding="utf-8",
    )

    def fake_doctor(**kwargs):
        calls.append(kwargs)
        return DavidDoctorReport(
            model=kwargs["model"],
            workspace_path=tmp_path.resolve(),
            checks=(DoctorCheck("validation report", "ready", "explicit"),),
            validation_discovery=ValidationReportDiscovery(
                path=Path(kwargs["validation_report"]),
                checked_paths=(),
            ),
        )

    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    monkeypatch.setattr(cli, "run_doctor", fake_doctor)

    rc = cli.main(
        [
            "doctor",
            "--workspace",
            str(tmp_path),
            "--model",
            "cli-model",
            "--validation-report",
            "cli-report.json",
            "--model-attestation",
            "cli-attestation.json",
        ]
    )

    assert rc == 0
    assert FakeRuntime.created_with is None
    assert calls == [
        {
            "model": "cli-model",
            "workspace_path": tmp_path,
            "validation_report": "cli-report.json",
            "attestation_path": "cli-attestation.json",
            "auto_validate_model": False,
        }
    ]


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


def test_model_onboard_plan_command_invokes_wrapper_without_runtime(monkeypatch, tmp_path, capsys):
    FakeRuntime.created_with = None
    calls = []
    plan = ModelOnboardingPlan(
        model="google/gemma-e2b",
        workspace_path=tmp_path.resolve(),
        scan_report_path=tmp_path / ".david" / "model_validation" / "model_config_report.json",
        validation_report_path=tmp_path / ".david" / "model_validation" / "model_validation_report.json",
        next_actions=("run again with --execute when ready",),
    )

    def fake_onboard(**kwargs):
        calls.append(kwargs)
        return ModelOnboardingResult(
            plan=plan,
            execute=False,
            status="planned",
            summary="planned only",
            next_actions=plan.next_actions,
        )

    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    monkeypatch.setattr(cli, "onboard_model", fake_onboard)

    rc = cli.main(["model", "onboard", "google/gemma-e2b", "--workspace", str(tmp_path)])

    assert rc == 0
    assert FakeRuntime.created_with is None
    assert calls == [
        {
            "model": "google/gemma-e2b",
            "workspace_path": Path(tmp_path),
            "execute": False,
        }
    ]
    output = capsys.readouterr().out
    assert "David model onboarding" in output
    assert "status: planned" in output
    assert "execute: false" in output
    assert "attestation: not created or accepted by this command" in output
    assert "run again with --execute when ready" in output


def test_model_onboard_execute_command_reports_scan_validate_without_attestation(monkeypatch, tmp_path, capsys):
    calls = []
    plan = ModelOnboardingPlan(
        model="local-model",
        workspace_path=tmp_path.resolve(),
        scan_report_path=tmp_path / ".david" / "model_validation" / "model_config_report.json",
        validation_report_path=tmp_path / ".david" / "model_validation" / "model_validation_report.json",
        next_actions=(),
    )

    def fake_onboard(**kwargs):
        calls.append(kwargs)
        return ModelOnboardingResult(
            plan=plan,
            execute=True,
            status="needs_review",
            summary="needs_review: fail closed",
            next_actions=("create manual attestation only after review",),
            scan_result=ModelCommandResult(
                returncode=0,
                command=("python", "David/get_model_config.py"),
                stdout="",
                stderr="",
            ),
            validation_result=ModelCommandResult(
                returncode=0,
                command=("python", "David/validate_model_config.py"),
                stdout="",
                stderr="",
            ),
        )

    monkeypatch.setattr(cli, "onboard_model", fake_onboard)

    rc = cli.main(["model", "onboard", "local-model", "--workspace", str(tmp_path), "--execute"])

    assert rc == 0
    assert calls == [
        {
            "model": "local-model",
            "workspace_path": Path(tmp_path),
            "execute": True,
        }
    ]
    output = capsys.readouterr().out
    assert "status: needs_review" in output
    assert "execute: true" in output
    assert "scan rc: 0" in output
    assert "validation rc: 0" in output
    assert "attestation: not created or accepted by this command" in output


def test_model_onboard_rejected_returns_failure(monkeypatch, tmp_path, capsys):
    plan = ModelOnboardingPlan(
        model="bad-model",
        workspace_path=tmp_path.resolve(),
        scan_report_path=tmp_path / ".david" / "model_validation" / "model_config_report.json",
        validation_report_path=tmp_path / ".david" / "model_validation" / "model_validation_report.json",
        next_actions=(),
    )

    def fake_onboard(**_kwargs):
        return ModelOnboardingResult(
            plan=plan,
            execute=True,
            status="rejected",
            summary="rejected: fail closed",
            next_actions=("fix scanner evidence",),
            errors=("model scan failed with returncode 7",),
        )

    monkeypatch.setattr(cli, "onboard_model", fake_onboard)

    rc = cli.main(["model", "onboard", "bad-model", "--workspace", str(tmp_path), "--execute"])

    assert rc == 2
    output = capsys.readouterr().out
    assert "status: rejected" in output
    assert "model scan failed with returncode 7" in output


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
