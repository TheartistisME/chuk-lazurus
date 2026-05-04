from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from chuk_lazarus.david import doctor
from chuk_lazarus.david import model_validation
from chuk_lazarus.david.model_validation import discover_validation_report


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
    (workspace / "gemma.vindex.ple").mkdir()

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

    report = doctor.run_doctor(model=str(model), workspace_path=workspace)
    checks = {check.name: check for check in report.checks}

    assert checks["model location"].status == "ready"
    assert checks["HF snapshot"].status == "ready"
    assert checks[".vindex.ple artifact"].status == "ready"
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

    report = doctor.run_doctor(model="google/gemma-e2b", workspace_path=workspace)
    checks = {check.name: check for check in report.checks}

    assert checks["model location"].status == "review"
    assert checks["HF snapshot"].status == "ready"
    assert str(snapshot) in checks["HF snapshot"].detail
