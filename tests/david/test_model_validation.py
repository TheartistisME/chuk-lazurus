from __future__ import annotations

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
