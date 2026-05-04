from __future__ import annotations

import json
from pathlib import Path

from chuk_lazarus.david.model_onboarding import (
    ACCEPTED,
    NEEDS_REVIEW,
    PLANNED,
    REJECTED,
    onboard_model,
    plan_model_onboarding,
)
from chuk_lazarus.david.model_validation import ModelCommandResult


def test_plan_model_onboarding_uses_workspace_auto_paths(tmp_path):
    plan = plan_model_onboarding(model="google/gemma-e2b", workspace_path=tmp_path)

    assert plan.model == "google/gemma-e2b"
    assert plan.workspace_path == tmp_path.resolve()
    assert plan.scan_report_path == tmp_path / ".david" / "model" / "model_config_report.json"
    assert plan.validation_report_path == (
        tmp_path / ".david" / "model_validation" / "model_validation_report.json"
    )
    assert any("Fail closed" in action for action in plan.next_actions)


def test_onboard_model_plan_only_does_not_call_helpers(tmp_path):
    def fail_scan(**_kwargs):
        raise AssertionError("scan should not run without execute=True")

    def fail_validate(**_kwargs):
        raise AssertionError("validate should not run without execute=True")

    result = onboard_model(
        model="google/gemma-e2b",
        workspace_path=tmp_path,
        scan_fn=fail_scan,
        validate_fn=fail_validate,
    )

    assert result.status == PLANNED
    assert result.execute is False
    assert result.scan_result is None
    assert result.validation_result is None
    assert "no scan" in result.summary


def test_onboard_model_executes_fake_scan_validate_and_accepts_report(tmp_path):
    calls = []

    def fake_scan(**kwargs):
        calls.append(("scan", kwargs))
        _write_json(kwargs["output"], {"schema_name": "scanner"})
        return _command_result(0, "scan")

    def fake_validate(**kwargs):
        calls.append(("validate", kwargs))
        _write_json(kwargs["output"], _validation_report_payload())
        return _command_result(0, "validate")

    result = onboard_model(
        model="google/gemma-e2b",
        workspace_path=tmp_path,
        execute=True,
        scan_fn=fake_scan,
        validate_fn=fake_validate,
    )

    assert result.status == ACCEPTED
    assert result.accepted is True
    assert result.validation_summary is not None
    assert result.validation_summary.can_auto_load is True
    assert result.errors == ()
    assert result.plan.scan_report_path.is_file()
    assert result.plan.validation_report_path.is_file()
    assert calls == [
        (
            "scan",
            {
                "model": "google/gemma-e2b",
                "output": result.plan.scan_report_path,
                "repo_root": None,
            },
        ),
        (
            "validate",
            {
                "report": result.plan.scan_report_path,
                "output": result.plan.validation_report_path,
                "model": "google/gemma-e2b",
                "repo_root": None,
            },
        ),
    ]


def test_onboard_model_needs_review_is_fail_closed_with_attestation_guidance(tmp_path):
    def fake_scan(**kwargs):
        _write_json(kwargs["output"], {"schema_name": "scanner"})
        return _command_result(0, "scan")

    def fake_validate(**kwargs):
        _write_json(
            kwargs["output"],
            _validation_report_payload(
                validation_status="needs_review",
                confidence="low",
                auto_load_allowed=False,
                harness_load_policy="manual_or_validation_required",
                warnings=["behavioral KV evidence required"],
            ),
        )
        return _command_result(0, "validate")

    result = onboard_model(
        model="google/gemma-e2b",
        workspace_path=tmp_path,
        execute=True,
        scan_fn=fake_scan,
        validate_fn=fake_validate,
    )

    actions = "\n".join(result.next_actions)

    assert result.status == NEEDS_REVIEW
    assert result.accepted is False
    assert result.needs_review is True
    assert result.errors == ()
    assert "fail closed" in result.summary
    assert "manual model attestation" in actions
    assert "standard_decode" in actions
    assert "kv_replay" in actions
    assert "behavioral KV evidence required" in actions


def test_onboard_model_rejects_when_scan_fails_and_skips_validator(tmp_path):
    def fake_scan(**_kwargs):
        return _command_result(7, "scan", stderr="scanner failed")

    def fake_validate(**_kwargs):
        raise AssertionError("validator should not run after failed scan")

    result = onboard_model(
        model="google/gemma-e2b",
        workspace_path=tmp_path,
        execute=True,
        scan_fn=fake_scan,
        validate_fn=fake_validate,
    )

    assert result.status == REJECTED
    assert result.rejected is True
    assert result.validation_result is None
    assert result.errors == ("model scan failed with returncode 7",)
    assert "Fail closed" in result.next_actions[0]


def test_onboard_model_rejects_unreadable_validation_output(tmp_path):
    def fake_scan(**kwargs):
        _write_json(kwargs["output"], {"schema_name": "scanner"})
        return _command_result(0, "scan")

    def fake_validate(**_kwargs):
        return _command_result(0, "validate")

    result = onboard_model(
        model="google/gemma-e2b",
        workspace_path=tmp_path,
        execute=True,
        scan_fn=fake_scan,
        validate_fn=fake_validate,
    )

    assert result.status == REJECTED
    assert result.validation_summary is not None
    assert result.validation_summary.readable is False
    assert result.errors
    assert "could not read report" in result.summary


def _command_result(returncode: int, name: str, stderr: str = "") -> ModelCommandResult:
    return ModelCommandResult(returncode=returncode, command=(name,), stdout="", stderr=stderr)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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
        "validation_status": validation_status,
        "confidence": confidence,
        "auto_load_allowed": auto_load_allowed,
        "harness_load_policy": harness_load_policy,
        "warnings": warnings or [],
        "selected_config": {
            "adapter_config_id": "gemma-e2b-l23",
            "route_layer": 11,
            "boundary_layer": 17,
            "residual_capture_layer": 17,
            "kv_source_layer": 21,
            "kv_target_layer": 23,
            "injection_layer": 23,
            "insertion_family": "kv_direct",
        },
    }
