from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from chuk_lazarus.david import doctor
from chuk_lazarus.david import model_validation
from chuk_lazarus.david.model_validation import discover_validation_report, summarize_validation_report


def test_discover_validation_report_prefers_model_validation_report(tmp_path):
    workspace = tmp_path / "workspace"
    model = tmp_path / "model"
    workspace_report_dir = workspace / ".david"
    workspace_report_dir.mkdir(parents=True)
    model.mkdir()
    model_report = model / "validation_report.json"
    workspace_report = workspace_report_dir / "model_validation_report.json"
    model_report.write_text("{}", encoding="utf-8")
    workspace_report.write_text("{}", encoding="utf-8")

    discovery = discover_validation_report(model_path=str(model), workspace_path=workspace)

    assert discovery.path == model_report


def test_discover_validation_report_uses_workspace_report_when_model_report_missing(tmp_path):
    workspace = tmp_path / "workspace"
    model = tmp_path / "model"
    workspace_report_dir = workspace / ".david"
    workspace_report_dir.mkdir(parents=True)
    model.mkdir()
    workspace_report = workspace_report_dir / "model_validation_report.json"
    workspace_report.write_text("{}", encoding="utf-8")

    discovery = discover_validation_report(model_path=str(model), workspace_path=workspace)

    assert discovery.path == workspace_report


def test_discover_validation_report_uses_auto_workspace_validation_report(tmp_path):
    workspace = tmp_path / "workspace"
    workspace_report_dir = workspace / ".david" / "model_validation"
    workspace_report_dir.mkdir(parents=True)
    workspace_report = workspace_report_dir / "model_validation_report.json"
    workspace_report.write_text("{}", encoding="utf-8")

    discovery = discover_validation_report(model_path="google/gemma-e2b", workspace_path=workspace)

    assert discovery.path == workspace_report


def test_discover_validation_report_reports_checked_paths_when_missing(tmp_path):
    workspace = tmp_path / "workspace"
    model = tmp_path / "model"
    workspace.mkdir()
    model.mkdir()

    discovery = discover_validation_report(model_path=str(model), workspace_path=workspace)

    assert discovery.path is None
    assert model / "validation_report.json" in discovery.checked_paths
    assert model / "model_validation_report.json" in discovery.checked_paths
    assert workspace / ".david" / "model_validation_report.json" in discovery.checked_paths
    assert workspace / ".david" / "model_validation" / "model_validation_report.json" in discovery.checked_paths


def test_run_model_scan_invokes_standalone_getter(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    script = repo / "David" / "get_model_config.py"
    script.parent.mkdir(parents=True)
    script.write_text("# helper\n", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(model_validation.subprocess, "run", fake_run)

    result = model_validation.run_model_scan(
        model="model-id",
        output=Path("scan.json"),
        repo_root=repo,
        extra_args=("--inspect-only",),
    )

    assert result.returncode == 0
    assert calls == [
        (
            (
                sys.executable,
                str(script),
                "--model",
                "model-id",
                "--json-out",
                "scan.json",
                "--inspect-only",
            ),
            {
                "cwd": repo,
                "check": False,
                "capture_output": True,
                "text": True,
            },
        )
    ]


def test_run_model_validate_invokes_standalone_validator(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    script = repo / "David" / "validate_model_config.py"
    script.parent.mkdir(parents=True)
    script.write_text("# helper\n", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="validator notes")

    monkeypatch.setattr(model_validation.subprocess, "run", fake_run)

    result = model_validation.run_model_validate(
        report=Path("scan.json"),
        output=Path("validated.json"),
        model="model-id",
        repo_root=repo,
        extra_args=("--dry-run",),
    )

    assert result.returncode == 0
    assert result.stderr == "validator notes"
    assert calls[0][0] == (
        sys.executable,
        str(script),
        "--config-report",
        "scan.json",
        "--json-out",
        "validated.json",
        "--model",
        "model-id",
        "--dry-run",
    )


def test_run_auto_model_validation_writes_workspace_artifact_paths(monkeypatch, tmp_path):
    calls = []

    def fake_scan(**kwargs):
        calls.append(("scan", kwargs))
        return model_validation.ModelCommandResult(
            returncode=0,
            command=("scan",),
            stdout="",
            stderr="",
        )

    def fake_validate(**kwargs):
        calls.append(("validate", kwargs))
        return model_validation.ModelCommandResult(
            returncode=0,
            command=("validate",),
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(model_validation, "run_model_scan", fake_scan)
    monkeypatch.setattr(model_validation, "run_model_validate", fake_validate)

    result = model_validation.run_auto_model_validation(
        model="google/gemma-e2b",
        workspace_path=tmp_path,
    )

    assert result.returncode == 0
    assert result.scan_report_path == tmp_path / ".david" / "model" / "model_config_report.json"
    assert result.validation_report_path == (
        tmp_path / ".david" / "model_validation" / "model_validation_report.json"
    )
    assert result.scan_report_path.parent.exists()
    assert result.validation_report_path.parent.exists()
    assert calls == [
        (
            "scan",
            {
                "model": "google/gemma-e2b",
                "output": result.scan_report_path,
                "repo_root": None,
            },
        ),
        (
            "validate",
            {
                "report": result.scan_report_path,
                "output": result.validation_report_path,
                "model": "google/gemma-e2b",
                "repo_root": None,
            },
        ),
    ]


def test_run_auto_model_validation_stops_when_scan_fails(monkeypatch, tmp_path):
    def fake_scan(**_kwargs):
        return model_validation.ModelCommandResult(
            returncode=9,
            command=("scan",),
            stdout="",
            stderr="nope",
        )

    def fake_validate(**_kwargs):
        raise AssertionError("validator should not run after failed scan")

    monkeypatch.setattr(model_validation, "run_model_scan", fake_scan)
    monkeypatch.setattr(model_validation, "run_model_validate", fake_validate)

    result = model_validation.run_auto_model_validation(
        model="google/gemma-e2b",
        workspace_path=tmp_path,
    )

    assert result.returncode == 9
    assert result.validation_result is None


def test_summarize_validation_report_exposes_accepted_boot_fields(tmp_path):
    report_path = tmp_path / "accepted-validation.json"
    report_path.write_text(
        json.dumps(_validation_report_payload()),
        encoding="utf-8",
    )

    summary = summarize_validation_report(report_path)

    assert summary.path == report_path
    assert summary.readable is True
    assert summary.can_auto_load is True
    assert summary.validation_status == "accepted"
    assert summary.confidence == "high"
    assert summary.auto_load_allowed is True
    assert summary.harness_load_policy == "allow_auto_load"
    assert summary.selected_adapter_config_id == "gemma-e2b-l23"
    assert summary.selected_insertion_family == "kv_direct"
    assert summary.selected_config_layer_scope["route_layer"] == 11
    assert summary.selected_config_layer_scope["kv_source_layer"] == 21
    assert summary.selected_config_layer_scope["kv_target_layer"] == 23
    assert summary.rejection_reasons == ()
    assert summary.layer_scope_text() == (
        "route_layer=11, boundary_layer=17, residual_capture_layer=17, "
        "kv_source_layer=21, kv_target_layer=23, injection_layer=23"
    )


def test_summarize_validation_report_keeps_needs_review_fail_closed(tmp_path):
    report_path = tmp_path / "needs-review-validation.json"
    report = _validation_report_payload(
        validation_status="needs_review",
        confidence="low",
        auto_load_allowed=False,
        harness_load_policy="manual_or_validation_required",
        warnings=["behavioral KV validation was required but did not pass"],
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")

    summary = summarize_validation_report(report_path)

    assert summary.can_auto_load is False
    assert summary.validation_status == "needs_review"
    assert summary.confidence == "low"
    assert summary.auto_load_allowed is False
    assert summary.harness_load_policy == "manual_or_validation_required"
    assert "behavioral KV validation was required but did not pass" in summary.warnings
    assert "validation_status is 'needs_review', expected 'accepted'" in summary.rejection_reasons
    assert "auto_load_allowed is false" in summary.rejection_reasons


def test_doctor_surfaces_accepted_validation_report_summary(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    model = tmp_path / "gemma"
    workspace.mkdir()
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_text("", encoding="utf-8")
    report_path = model / "validation_report.json"
    report_path.write_text(json.dumps(_validation_report_payload()), encoding="utf-8")

    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "ready", "CUDA available"),
    )
    monkeypatch.setattr(
        doctor,
        "_wsl_tooling_check",
        lambda: doctor.DoctorCheck("WSL/tooling", "ready", "available"),
    )
    monkeypatch.setattr(
        doctor,
        "_auto_validate_check",
        lambda **_kwargs: doctor.DoctorCheck("--auto-validate-model", "ready", "not needed"),
    )
    _disable_wsl_probe_checks(monkeypatch)

    report = doctor.run_doctor(model=str(model), workspace_path=workspace)
    checks = {check.name: check for check in report.checks}
    formatted = doctor.format_doctor_report(report)

    assert checks["validation report"].status == "ready"
    assert checks[".vindex.ple artifact"].status == "missing"
    assert "validation_status=accepted" in checks["validation report"].detail
    assert report.validation_summary is not None
    assert report.validation_summary.can_auto_load is True
    assert report.ready is True
    assert "- validation report summary:" in formatted
    assert "  - validation_status: accepted" in formatted
    assert "  - selected_config_layer_scope: route_layer=11" in formatted
    assert "  - rejection_reasons: none" in formatted


def test_doctor_blocks_needs_review_validation_report_with_reasons(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    model = tmp_path / "gemma"
    workspace.mkdir()
    model.mkdir()
    report_path = tmp_path / "review-required.json"
    report_path.write_text(
        json.dumps(
            _validation_report_payload(
                validation_status="needs_review",
                confidence="low",
                auto_load_allowed=False,
                harness_load_policy="manual_or_validation_required",
                warnings=["validation score margin 0.020 is below threshold 0.100"],
            )
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "ready", "CUDA available"),
    )
    monkeypatch.setattr(
        doctor,
        "_wsl_tooling_check",
        lambda: doctor.DoctorCheck("WSL/tooling", "ready", "available"),
    )
    _disable_wsl_probe_checks(monkeypatch)

    report = doctor.run_doctor(
        model=str(model),
        workspace_path=workspace,
        validation_report=str(report_path),
    )
    checks = {check.name: check for check in report.checks}
    formatted = doctor.format_doctor_report(report)

    assert checks["validation report"].status == "blocked"
    assert "validation_status=needs_review" in checks["validation report"].detail
    assert "auto_load_allowed=False" in checks["validation report"].detail
    assert "validation_status is 'needs_review', expected 'accepted'" in formatted
    assert "auto_load_allowed is false" in formatted
    assert "validation score margin 0.020 is below threshold 0.100" in formatted
    assert report.ready is False


def test_doctor_ready_for_needs_review_validation_with_standard_decode_attestation(
    monkeypatch,
    tmp_path,
):
    workspace = tmp_path / "workspace"
    model = tmp_path / "gemma"
    workspace.mkdir()
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_text("", encoding="utf-8")
    report_path = model / "validation_report.json"
    report_path.write_text(
        json.dumps(
            _validation_report_payload(
                validation_status="needs_review",
                confidence="low",
                auto_load_allowed=False,
                harness_load_policy="manual_or_validation_required",
                warnings=["operator standard-decode attestation required"],
            )
        ),
        encoding="utf-8",
    )
    attestation_path = model / "model_attestation.json"
    attestation_path.write_text(json.dumps(_attestation_payload()), encoding="utf-8")

    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "ready", "CUDA available"),
    )
    monkeypatch.setattr(
        doctor,
        "_wsl_tooling_check",
        lambda: doctor.DoctorCheck("WSL/tooling", "ready", "available"),
    )
    monkeypatch.setattr(
        doctor,
        "_auto_validate_check",
        lambda **_kwargs: doctor.DoctorCheck("--auto-validate-model", "ready", "not needed"),
    )
    _disable_wsl_probe_checks(monkeypatch)

    report = doctor.run_doctor(model=str(model), workspace_path=workspace)
    checks = {check.name: check for check in report.checks}

    assert checks["validation report"].status == "blocked"
    assert checks["model attestation"].status == "ready"
    assert checks[".vindex.ple artifact"].status == "missing"
    assert report.attestation_summary is not None
    assert report.attestation_summary.standard_decode_allowed is True
    assert report.attestation_summary.standard_decode_only is True
    assert report.attestation_summary.rejection_reasons == ()
    assert report.ready is True


def test_doctor_does_not_hide_validation_failure_with_invalid_attestation(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    model = tmp_path / "gemma"
    workspace.mkdir()
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_text("", encoding="utf-8")
    report_path = model / "validation_report.json"
    report_path.write_text(
        json.dumps(
            _validation_report_payload(
                validation_status="needs_review",
                confidence="low",
                auto_load_allowed=False,
                harness_load_policy="manual_or_validation_required",
            )
        ),
        encoding="utf-8",
    )
    attestation_path = model / "model_attestation.json"
    attestation_payload = _attestation_payload()
    attestation_payload.pop("reviewer")
    attestation_path.write_text(json.dumps(attestation_payload), encoding="utf-8")

    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "ready", "CUDA available"),
    )
    monkeypatch.setattr(
        doctor,
        "_wsl_tooling_check",
        lambda: doctor.DoctorCheck("WSL/tooling", "ready", "available"),
    )
    monkeypatch.setattr(
        doctor,
        "_auto_validate_check",
        lambda **_kwargs: doctor.DoctorCheck("--auto-validate-model", "ready", "not needed"),
    )
    _disable_wsl_probe_checks(monkeypatch)

    report = doctor.run_doctor(model=str(model), workspace_path=workspace)
    checks = {check.name: check for check in report.checks}

    assert checks["validation report"].status == "blocked"
    assert checks["model attestation"].status == "review"
    assert report.attestation_summary is not None
    assert "reviewer is missing" in report.attestation_summary.rejection_reasons
    assert report.ready is False


def test_doctor_surfaces_manual_attestation_as_standard_decode_only(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    model = tmp_path / "gemma"
    workspace.mkdir()
    model.mkdir()
    attestation_dir = workspace / ".david"
    attestation_dir.mkdir()
    attestation_path = attestation_dir / "model_attestation.json"
    attestation_path.write_text(json.dumps(_attestation_payload()), encoding="utf-8")

    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "ready", "CUDA available"),
    )
    monkeypatch.setattr(
        doctor,
        "_wsl_tooling_check",
        lambda: doctor.DoctorCheck("WSL/tooling", "ready", "available"),
    )
    _disable_wsl_probe_checks(monkeypatch)

    report = doctor.run_doctor(model=str(model), workspace_path=workspace)
    checks = {check.name: check for check in report.checks}
    formatted = doctor.format_doctor_report(report)

    assert checks["validation report"].status == "missing"
    assert checks["model attestation"].status == "ready"
    assert str(attestation_path) in checks["model attestation"].detail
    assert "standard_decode_only=True" in checks["model attestation"].detail
    assert "tensor_replay_effective=False" in checks["model attestation"].detail
    assert "unsafe replay capabilities refused" in checks["model attestation"].detail
    assert report.attestation_summary is not None
    assert report.attestation_summary.standard_decode_only is True
    assert report.attestation_summary.selected_adapter_config_id == "gemma-e2b-reviewed"
    assert "- model attestation summary:" in formatted
    assert "  - attestation_status: manual_reviewed" in formatted
    assert "  - reviewer: jehma" in formatted
    assert "  - allowed_capabilities: standard_decode" in formatted
    assert "  - standard_decode_only: True" in formatted
    assert "  - tensor_replay_effective: False" in formatted
    assert "  - rejection_reasons: none" in formatted
    assert "- checked attestation paths:" in formatted
    assert report.ready is False


def test_doctor_refuses_unsafe_attestation_replay_capabilities(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    attestation_path = tmp_path / "reviewed-attestation.json"
    attestation_path.write_text(
        json.dumps(
            _attestation_payload(
                allowed_capabilities=[
                    "standard_decode",
                    "kv_direct_tensor_replay",
                    "residual_tensor_replay",
                ]
            )
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "ready", "CUDA available"),
    )
    monkeypatch.setattr(
        doctor,
        "_wsl_tooling_check",
        lambda: doctor.DoctorCheck("WSL/tooling", "ready", "available"),
    )
    _disable_wsl_probe_checks(monkeypatch)

    report = doctor.run_doctor(
        model=None,
        workspace_path=workspace,
        attestation_path=str(attestation_path),
    )
    checks = {check.name: check for check in report.checks}
    formatted = doctor.format_doctor_report(report)

    assert checks["model attestation"].status == "review"
    assert "refused_replay_capabilities=kv_direct_tensor_replay, residual_tensor_replay" in (
        checks["model attestation"].detail
    )
    assert "manual attestation permits standard decode only" in checks["model attestation"].detail
    assert report.attestation_summary is not None
    assert report.attestation_summary.standard_decode_allowed is True
    assert report.attestation_summary.tensor_replay_effective is False
    assert "  - allowed_capabilities: standard_decode, kv_direct_tensor_replay, residual_tensor_replay" in (
        formatted
    )
    assert "  - refused_replay_capabilities: kv_direct_tensor_replay, residual_tensor_replay" in formatted
    assert "  - rejection_reasons: none" in formatted


def test_run_model_scan_reports_missing_helper_without_subprocess(monkeypatch, tmp_path):
    def fake_run(*_args, **_kwargs):
        raise AssertionError("subprocess should not run when helper is missing")

    monkeypatch.setattr(model_validation.subprocess, "run", fake_run)

    result = model_validation.run_model_scan(
        model="model-id",
        output=Path("scan.json"),
        repo_root=tmp_path,
    )

    assert result.returncode == 2
    assert "Standalone David model helper not found" in result.stderr
    assert "get_model_config.py" in result.stderr


def test_run_model_validate_reports_start_failure(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    script = repo / "David" / "validate_model_config.py"
    script.parent.mkdir(parents=True)
    script.write_text("# helper\n", encoding="utf-8")

    def fake_run(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(model_validation.subprocess, "run", fake_run)

    result = model_validation.run_model_validate(
        report=Path("scan.json"),
        output=Path("validated.json"),
        repo_root=repo,
    )

    assert result.returncode == 2
    assert "Failed to start standalone David model helper: permission denied" in result.stderr


def test_doctor_distinguishes_local_hf_snapshot_vindex_and_missing_report(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    model = tmp_path / "gemma"
    workspace.mkdir()
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_text("", encoding="utf-8")
    _write_vindex_artifact(workspace)

    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "review", "CPU-only torch"),
    )
    monkeypatch.setattr(
        doctor,
        "_wsl_tooling_check",
        lambda: doctor.DoctorCheck("WSL/tooling", "review", "unavailable"),
    )
    _disable_wsl_probe_checks(monkeypatch)

    report = doctor.run_doctor(model=str(model), workspace_path=workspace)
    checks = {check.name: check for check in report.checks}

    assert checks["model location"].status == "ready"
    assert checks["HF snapshot"].status == "ready"
    assert checks[".vindex.ple artifact"].status == "ready"
    assert checks["validation report"].status == "missing"
    assert report.ready is False


def test_doctor_classifies_direct_vindex_artifact_without_hf_snapshot_noise(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = _write_vindex_artifact(tmp_path)

    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "review", "CPU-only torch"),
    )
    monkeypatch.setattr(
        doctor,
        "_wsl_tooling_check",
        lambda: doctor.DoctorCheck("WSL/tooling", "review", "unavailable"),
    )
    _disable_wsl_probe_checks(monkeypatch)

    report = doctor.run_doctor(model=str(artifact), workspace_path=workspace)
    checks = {check.name: check for check in report.checks}

    assert checks["model location"].status == "review"
    assert ".vindex.ple artifact path" in checks["model location"].detail
    assert "direct generation unsupported" in checks["model location"].detail
    assert "HF files are incomplete" not in checks["model location"].detail
    assert "missing config.json" not in checks["model location"].detail
    assert checks["HF snapshot"].status == "review"
    assert "not applicable for direct .vindex.ple artifact path" in checks["HF snapshot"].detail
    assert "missing config.json" not in checks["HF snapshot"].detail
    assert checks[".vindex.ple artifact"].status == "ready"
    assert "family=gemma4" in checks[".vindex.ple artifact"].detail
    assert "layers=35" in checks[".vindex.ple artifact"].detail
    assert "hidden_size=1536" in checks[".vindex.ple artifact"].detail
    assert "manifest_files=1" in checks[".vindex.ple artifact"].detail
    assert checks["--auto-validate-model"].status == "review"
    assert "not applicable for direct .vindex.ple artifact path" in checks["--auto-validate-model"].detail
    assert checks["validation report"].status == "missing"
    assert report.ready is False


def test_doctor_reports_hf_cache_snapshot_completeness(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    cache = tmp_path / "hf" / "hub"
    snapshot = cache / "models--google--gemma-e2b" / "snapshots" / "abc123"
    workspace.mkdir()
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.model").write_text("", encoding="utf-8")
    (snapshot / "model.safetensors").write_text("", encoding="utf-8")

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "ready", "CUDA available"),
    )
    monkeypatch.setattr(
        doctor,
        "_wsl_tooling_check",
        lambda: doctor.DoctorCheck("WSL/tooling", "ready", "available"),
    )
    _disable_wsl_probe_checks(monkeypatch)

    report = doctor.run_doctor(model="google/gemma-e2b", workspace_path=workspace)
    checks = {check.name: check for check in report.checks}

    assert checks["model location"].status == "review"
    assert checks["HF snapshot"].status == "ready"
    assert str(snapshot) in checks["HF snapshot"].detail


def test_doctor_reports_complete_wsl_hf_snapshot_for_hf_model_id(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        script = command[-1]
        if "model_cache =" in script:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "cache": "/home/jehma/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots",
                        "snapshot_count": 1,
                        "complete": True,
                        "snapshot": (
                            "/home/jehma/.cache/huggingface/hub/"
                            "models--google--gemma-4-E2B-it/snapshots/b4a60110"
                        ),
                        "detail": "config, tokenizer, and weights present",
                    }
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "missing": [],
                    "cuda_status": "ready",
                    "cuda_detail": "CUDA available; devices=1",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(doctor.shutil, "which", lambda item: "wsl.exe" if item == "wsl" else None)
    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty_hf"))
    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "review", "CPU-only torch"),
    )

    report = doctor.run_doctor(model="google/gemma-4-E2B-it", workspace_path=workspace)
    checks = {check.name: check for check in report.checks}

    assert checks["HF snapshot"].status == "missing"
    assert checks["WSL HF snapshot"].status == "ready"
    assert "/home/jehma/.cache/huggingface/hub/models--google--gemma-4-E2B-it" in (
        checks["WSL HF snapshot"].detail
    )
    assert checks["WSL Python packages"].status == "ready"
    assert "CUDA available" in checks["WSL Python packages"].detail
    assert calls[0][1]["timeout"] == doctor.DEFAULT_WSL_PROBE_TIMEOUT_SECONDS
    assert calls[1][1]["timeout"] == doctor.DEFAULT_WSL_PROBE_TIMEOUT_SECONDS


def test_doctor_checks_wsl_absolute_model_path_as_local_path(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "path": "/home/jehma/models/gemma",
                    "exists": True,
                    "is_dir": True,
                    "complete": True,
                    "detail": "config, tokenizer, and weights present",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(doctor.shutil, "which", lambda item: "wsl.exe" if item == "wsl" else None)
    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "review", "CPU-only torch"),
    )
    monkeypatch.setattr(
        doctor,
        "_wsl_tooling_check",
        lambda: doctor.DoctorCheck("WSL/tooling", "ready", "available"),
    )
    monkeypatch.setattr(
        doctor,
        "_wsl_python_packages_check",
        lambda: doctor.DoctorCheck("WSL Python packages", "review", "disabled"),
    )
    monkeypatch.setattr(
        doctor,
        "_auto_validate_check",
        lambda **_kwargs: doctor.DoctorCheck("--auto-validate-model", "review", "disabled"),
    )

    report = doctor.run_doctor(model="/home/jehma/models/gemma", workspace_path=workspace)
    checks = {check.name: check for check in report.checks}

    assert checks["model location"].status == "ready"
    assert "WSL local HF-style model directory" in checks["model location"].detail
    assert checks["HF snapshot"].status == "review"
    assert "not applicable for WSL local model path" in checks["HF snapshot"].detail
    assert checks["WSL HF snapshot"].status == "review"
    assert "not applicable; model is not an HF id" in checks["WSL HF snapshot"].detail
    assert calls[0][1]["timeout"] == doctor.DEFAULT_WSL_PROBE_TIMEOUT_SECONDS


def test_doctor_reports_missing_wsl_absolute_model_path_without_hf_cache_probe(
    monkeypatch,
    tmp_path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"path": "/mnt/c/models/missing-gemma", "exists": False}),
            stderr="",
        )

    monkeypatch.setattr(doctor.shutil, "which", lambda item: "wsl.exe" if item == "wsl" else None)
    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "review", "CPU-only torch"),
    )
    monkeypatch.setattr(
        doctor,
        "_wsl_python_packages_check",
        lambda: doctor.DoctorCheck("WSL Python packages", "review", "disabled"),
    )
    monkeypatch.setattr(
        doctor,
        "_auto_validate_check",
        lambda **_kwargs: doctor.DoctorCheck("--auto-validate-model", "review", "disabled"),
    )

    report = doctor.run_doctor(model="/mnt/c/models/missing-gemma", workspace_path=workspace)
    checks = {check.name: check for check in report.checks}

    assert checks["model location"].status == "missing"
    assert "WSL local model path does not exist" in checks["model location"].detail
    assert "HF model id" not in checks["model location"].detail
    assert checks["HF snapshot"].status == "review"
    assert "not applicable for WSL local model path" in checks["HF snapshot"].detail
    assert checks["WSL HF snapshot"].status == "review"
    assert "not applicable; model is not an HF id" in checks["WSL HF snapshot"].detail
    assert len(calls) == 1


def test_unix_absolute_model_paths_are_not_hf_model_ids():
    assert doctor._looks_like_hf_model_id("google/gemma-e2b") is True
    assert doctor._looks_like_hf_model_id("/home/jehma/models/gemma") is False
    assert doctor._looks_like_hf_model_id("/mnt/c/models/gemma") is False
    assert doctor._looks_like_hf_model_id("/opt/models/gemma") is False


def test_doctor_wsl_probe_timeout_degrades_to_review(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(doctor.shutil, "which", lambda item: "wsl.exe" if item == "wsl" else None)
    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "review", "CPU-only torch"),
    )

    report = doctor.run_doctor(model="google/gemma-4-E2B-it", workspace_path=workspace)
    checks = {check.name: check for check in report.checks}

    assert checks["WSL HF snapshot"].status == "review"
    assert "timed out after 20s" in checks["WSL HF snapshot"].detail
    assert checks["WSL Python packages"].status == "review"
    assert "timed out after 20s" in checks["WSL Python packages"].detail


def test_doctor_wsl_probe_timeout_env_override(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setenv(doctor.WSL_PROBE_TIMEOUT_ENV, "12.5")
    monkeypatch.setattr(doctor.shutil, "which", lambda item: "wsl.exe" if item == "wsl" else None)
    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "review", "CPU-only torch"),
    )

    report = doctor.run_doctor(model="google/gemma-4-E2B-it", workspace_path=workspace)
    checks = {check.name: check for check in report.checks}

    assert calls[0][1]["timeout"] == 12.5
    assert calls[1][1]["timeout"] == 12.5
    assert "timed out after 12.5s" in checks["WSL HF snapshot"].detail
    assert "timed out after 12.5s" in checks["WSL Python packages"].detail


def test_doctor_wsl_probe_timeout_env_invalid_uses_default(monkeypatch):
    monkeypatch.setenv(doctor.WSL_PROBE_TIMEOUT_ENV, "not-a-timeout")

    assert doctor._wsl_probe_timeout_seconds() == doctor.DEFAULT_WSL_PROBE_TIMEOUT_SECONDS

    monkeypatch.setenv(doctor.WSL_PROBE_TIMEOUT_ENV, "0")

    assert doctor._wsl_probe_timeout_seconds() == doctor.DEFAULT_WSL_PROBE_TIMEOUT_SECONDS


def test_doctor_wsl_python_missing_packages_stays_review(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fake_run(command, **_kwargs):
        script = command[-1]
        if "model_cache =" in script:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "cache": "/home/test/.cache/huggingface/hub/models--google--gemma-e2b/snapshots",
                        "snapshot_count": 0,
                        "complete": False,
                    }
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "missing": ["torch", "accelerate"],
                    "cuda_status": "not_checked",
                    "cuda_detail": "torch is not importable",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(doctor.shutil, "which", lambda item: "wsl.exe" if item == "wsl" else None)
    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    monkeypatch.setattr(doctor, "_optional_package_checks", lambda: ())
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_check",
        lambda: doctor.DoctorCheck("torch CUDA", "review", "CPU-only torch"),
    )

    report = doctor.run_doctor(model="google/gemma-e2b", workspace_path=workspace)
    checks = {check.name: check for check in report.checks}

    assert checks["WSL HF snapshot"].status == "review"
    assert checks["WSL Python packages"].status == "review"
    assert "missing packages: torch, accelerate" in checks["WSL Python packages"].detail


def test_torch_cuda_check_reports_missing_torch(monkeypatch):
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: None)

    check = doctor._torch_cuda_check()

    assert check.status == "missing"
    assert check.detail == "torch is not installed"


def test_torch_cuda_check_degrades_gracefully_on_cpu_only_torch(monkeypatch, tmp_path):
    report_path = tmp_path / "validation.json"
    payload = _validation_report_payload()
    payload["model"] = {"hidden_size": 2048, "num_layers": 2, "dtype": "float16"}
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    fake_torch = SimpleNamespace(
        float16=object(),
        bfloat16=object(),
        cuda=SimpleNamespace(
            is_available=lambda: False,
            is_bf16_supported=lambda: False,
        ),
    )
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: object() if name == "torch" else None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    check = doctor._torch_cuda_check(validation_report_path=report_path)

    assert check.name == "torch CUDA"
    assert check.status == "review"
    assert "CPU-only torch" in check.detail
    assert "cuda_available=False" in check.detail
    assert "selected_device=cpu" in check.detail
    assert "bfloat16_support=False" in check.detail
    assert "rough_model_memory_estimate=" in check.detail
    assert "estimate_evidence=hidden_size=2048" in check.detail


def test_torch_cuda_check_reports_cuda_devices_dtype_and_vram_feasibility(
    monkeypatch,
    tmp_path,
):
    report_path = tmp_path / "validation.json"
    payload = _validation_report_payload()
    payload["model"] = {
        "hidden_size": 1536,
        "num_layers": 2,
        "vocab_size": 1024,
        "dtype": "float16",
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    gib = 1024**3
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        current_device=lambda: 1,
        get_device_name=lambda index: ["Tiny GPU", "Ready GPU"][index],
        get_device_properties=lambda index: SimpleNamespace(
            total_memory=[2 * gib, 24 * gib][index]
        ),
        get_device_capability=lambda index: [(7, 5), (8, 9)][index],
        mem_get_info=lambda index: ([1 * gib, 20 * gib][index], [2 * gib, 24 * gib][index]),
        is_bf16_supported=lambda: True,
    )
    fake_torch = SimpleNamespace(float16=object(), bfloat16=object(), cuda=fake_cuda)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: object() if name == "torch" else None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    check = doctor._torch_cuda_check(validation_report_path=report_path)

    assert check.status == "ready"
    assert "CUDA available" in check.detail
    assert "devices=2" in check.detail
    assert "selected_device=cuda:1" in check.detail
    assert "selected_device_feasibility=likely_feasible_for_weights" in check.detail
    assert "float16_support=cuda" in check.detail
    assert "bfloat16_support=True" in check.detail
    assert "cuda:0, Tiny GPU, total=2048MiB, free=1024MiB, cc=7.5" in check.detail
    assert "cuda:1, Ready GPU, total=24576MiB, free=20480MiB, cc=8.9" in check.detail
    assert "estimate_source=validation_report_topology_rough" in check.detail


def test_torch_cuda_check_marks_low_vram_selected_device_for_review(monkeypatch, tmp_path):
    report_path = tmp_path / "validation.json"
    payload = _validation_report_payload()
    payload["model"] = {
        "parameter_count": 8_000_000_000,
        "dtype": "float16",
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    gib = 1024**3
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        current_device=lambda: 0,
        get_device_properties=lambda _index: SimpleNamespace(total_memory=8 * gib),
        mem_get_info=lambda _index: (6 * gib, 8 * gib),
    )
    fake_torch = SimpleNamespace(float16=object(), bfloat16=object(), cuda=fake_cuda)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: object() if name == "torch" else None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    check = doctor._torch_cuda_check(validation_report_path=report_path)

    assert check.status == "review"
    assert "selected_device_feasibility=blocked_vram_below_weight_estimate" in check.detail
    assert "estimate_source=validation_report_parameter_count" in check.detail


def _disable_wsl_probe_checks(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_wsl_hf_snapshot_check",
        lambda _model: doctor.DoctorCheck("WSL HF snapshot", "review", "disabled"),
    )
    monkeypatch.setattr(
        doctor,
        "_wsl_python_packages_check",
        lambda: doctor.DoctorCheck("WSL Python packages", "review", "disabled"),
    )


def _write_vindex_artifact(parent: Path) -> Path:
    artifact = parent / "gemma.vindex.ple"
    artifact.mkdir()
    (artifact / "index.json").write_text(
        json.dumps(
            {
                "family": "gemma4",
                "source": {"huggingface_repo": "google/gemma-4-E2B-it"},
                "hidden_size": 1536,
                "num_layers": 35,
                "vocab_size": 262144,
                "dtype": "f32",
            }
        ),
        encoding="utf-8",
    )
    (artifact / "tokenizer.json").write_text("{}", encoding="utf-8")
    (artifact / "weight_manifest.json").write_text(
        json.dumps([{"key": "layers.0.self_attn.q_proj.weight", "file": "attn_weights.bin"}]),
        encoding="utf-8",
    )
    (artifact / "attn_weights.bin").write_bytes(b"weights")
    return artifact


def _validation_report_payload(
    *,
    validation_status: str = "accepted",
    confidence: str = "high",
    auto_load_allowed: bool = True,
    harness_load_policy: str = "allow_auto_load",
    warnings: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_name": "lazarus.model_config_validation_report",
        "schema_version": 1,
        "validation_status": validation_status,
        "confidence": confidence,
        "validation_level": "topology_projection_behavior",
        "auto_load_allowed": auto_load_allowed,
        "harness_load_policy": harness_load_policy,
        "selected_config": {
            "adapter_config_id": "gemma-e2b-l23",
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
        },
        "warnings": warnings or [],
        "report_integrity": {"status": "ok", "warnings": []},
        "topology_gate": {"status": "ok", "warnings": []},
        "projection_gate": {"status": "ok", "warnings": []},
        "behavior_gate": {"status": "ok", "warnings": []},
    }


def _attestation_payload(
    *,
    allowed_capabilities: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_name": "david.manual_model_attestation",
        "schema_version": 1,
        "validation_report_sha256": "sha256:reviewed-report-digest",
        "model_identity": "google/gemma-4-E2B-it",
        "tokenizer_identity": "google/gemma-4-E2B-it",
        "model_revision_or_hash": "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf",
        "adapter_family": "gemma4",
        "reviewer": "jehma",
        "reviewed_at": "2026-05-04T07:30:00Z",
        "expires_at": "2026-06-04T07:30:00Z",
        "rationale": "Operator reviewed tied validator candidates for standard decode only.",
        "selected_config": {
            "adapter_config_id": "gemma-e2b-reviewed",
            "route_layer": 14,
            "boundary_layer": 23,
            "kv_source_layer": 28,
            "kv_target_layer": 29,
            "insertion_family": "kv_direct",
        },
        "allowed_capabilities": allowed_capabilities or ["standard_decode"],
        "refused_capabilities": [
            "kv_direct_tensor_replay",
            "residual_tensor_replay",
        ],
    }
