from __future__ import annotations

import json
import sys
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
            "--model-device",
            "cuda:0",
            "--model-dtype",
            "bfloat16",
            "--model-max-new-tokens",
            "88",
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
    assert config.model_device == "cuda:0"
    assert config.model_dtype == "bfloat16"
    assert config.model_max_new_tokens == 88
    assert config.color is False
    assert config.auto_jit_index is True

    output = capsys.readouterr().out
    assert "fake response: inspect boot config" in output


def test_main_preserves_model_runtime_env_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DavidRuntime", FakeRuntime)
    monkeypatch.setenv("DAVID_MODEL_DEVICE", "cuda")
    monkeypatch.setenv("DAVID_MODEL_DTYPE", "float16")
    monkeypatch.setenv("DAVID_MODEL_MAX_NEW_TOKENS", "77")

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
            "/status",
            "--no-color",
        ]
    )

    assert rc == 0
    config = FakeRuntime.created_with
    assert isinstance(config, cli.DavidConfig)
    assert config.model_device == "cuda"
    assert config.model_dtype == "float16"
    assert config.model_max_new_tokens == 77


def test_main_status_uses_real_runtime_with_validation_report(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    model = tmp_path / "model"
    workspace.mkdir()
    model.mkdir()
    report_path = tmp_path / "validation.json"
    report_path.write_text(json.dumps(_validation_report()), encoding="utf-8")

    rc = cli.main(
        [
            "code",
            str(workspace),
            "--model",
            str(model),
            "--validation-report",
            str(report_path),
            "--once",
            "/status",
            "--no-color",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "David startup readiness" in output
    assert "model validation: accepted (gemma:gemma-cli-test)" in output
    assert "workspace:" in output


def test_main_rejected_validation_report_fails_closed_before_tui(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    model = tmp_path / "model"
    workspace.mkdir()
    model.mkdir()
    report_path = tmp_path / "needs-review.json"
    report_path.write_text(
        json.dumps(_validation_report(status="needs_review", auto_load_allowed=False)),
        encoding="utf-8",
    )
    command = f'"{sys.executable}" -c "print(\'should-not-run\')"'

    rc = cli.main(
        [
            "code",
            str(workspace),
            "--model",
            str(model),
            "--validation-report",
            str(report_path),
            "--once",
            f"/run {command}",
            "--no-color",
        ]
    )

    assert rc == 2
    output = capsys.readouterr().out
    assert "David startup readiness" in output
    assert "model validation: blocked:" in output
    assert "validation report" in output
    assert str(report_path) in output
    assert "--allow-unvalidated" in output
    assert "offline shell mode" in output
    assert "David terminal agent" not in output
    assert "should-not-run" not in output


def test_main_allow_unvalidated_opens_offline_shell_with_rejected_report(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    model = tmp_path / "model"
    workspace.mkdir()
    model.mkdir()
    report_path = tmp_path / "needs-review.json"
    report_path.write_text(
        json.dumps(_validation_report(status="needs_review", auto_load_allowed=False)),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "code",
            str(workspace),
            "--model",
            str(model),
            "--validation-report",
            str(report_path),
            "--allow-unvalidated",
            "--once",
            "/status",
            "--no-color",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "David terminal agent" in output
    assert "David startup readiness" in output
    assert "backend: offline-deterministic: loaded" in output


def _validation_report(
    *,
    status: str = "accepted",
    auto_load_allowed: bool = True,
) -> dict[str, object]:
    selected_config = {
        "adapter_config_id": "gemma-cli-layer-23",
        "route_layer": 11,
        "route_query_head": 3,
        "route_dimension": 2048,
        "boundary_layer": 17,
        "residual_capture_layer": 17,
        "kv_source_layer": 21,
        "kv_target_layer": 23,
        "injection_layer": 23,
        "projection_producer_layer": 21,
        "behavior_cache_layer": 21,
        "insertion_family": "kv_direct",
        "kv_layout": "bshd",
        "candidate_role": "behavioral",
    }
    return {
        "schema_name": "lazarus.model_config_validation_report",
        "schema_version": 1,
        "validation_status": status,
        "confidence": "high",
        "validation_level": "behavioral",
        "auto_load_allowed": auto_load_allowed,
        "harness_load_policy": "auto",
        "selected_config": selected_config,
        "source_report_summary": {
            "model_identity": "gemma-cli-test",
            "tokenizer_identity": "gemma-cli-tokenizer",
            "adapter_family": "gemma",
            "model_revision_or_hash": "cli-rev",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        },
        "model_identity_gate": {
            "model_identity": "gemma-cli-test",
            "tokenizer_identity": "gemma-cli-tokenizer",
            "adapter_family": "gemma",
            "model_revision_or_hash": "cli-rev",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        },
        "topology_gate": {"accepted": True},
        "projection_gate": {"ranked_candidates": []},
        "behavior_gate": {"accepted": True},
        "report_integrity": {"accepted": True},
        "provenance": {"loader_options": {"model": "gemma-cli-test"}},
        "warnings": [],
    }
