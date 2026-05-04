"""Guided model onboarding plan/execution for David startup gates.

This module is intentionally CLI-free. It plans deterministic scanner and
validator artifact paths, can run injected scan/validate callables, and keeps
adapter loading fail-closed unless the validator proves the model is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from chuk_lazarus.david.model_validation import (
    ModelCommandResult,
    ValidationReportSummary,
    run_model_scan,
    run_model_validate,
    summarize_validation_report,
    workspace_auto_model_report_paths,
)


ACCEPTED = "accepted"
NEEDS_REVIEW = "needs_review"
REJECTED = "rejected"
PLANNED = "planned"
REVIEW_STATUSES = frozenset({NEEDS_REVIEW, "review_required"})

ScanCallable = Callable[..., ModelCommandResult]
ValidateCallable = Callable[..., ModelCommandResult]


@dataclass(frozen=True)
class ModelOnboardingPlan:
    """Deterministic scanner/validator artifact plan for one workspace."""

    model: str
    workspace_path: Path
    scan_report_path: Path
    validation_report_path: Path
    next_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "workspace_path": str(self.workspace_path),
            "scan_report_path": str(self.scan_report_path),
            "validation_report_path": str(self.validation_report_path),
            "next_actions": list(self.next_actions),
        }


@dataclass(frozen=True)
class ModelOnboardingResult:
    """Guided onboarding result with fail-closed startup status."""

    plan: ModelOnboardingPlan
    execute: bool
    status: str
    summary: str
    next_actions: tuple[str, ...]
    scan_result: ModelCommandResult | None = None
    validation_result: ModelCommandResult | None = None
    validation_summary: ValidationReportSummary | None = None
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def accepted(self) -> bool:
        return self.status == ACCEPTED

    @property
    def needs_review(self) -> bool:
        return self.status == NEEDS_REVIEW

    @property
    def rejected(self) -> bool:
        return self.status == REJECTED

    def to_dict(self) -> dict[str, object]:
        return {
            "execute": self.execute,
            "status": self.status,
            "summary": self.summary,
            "plan": self.plan.to_dict(),
            "next_actions": list(self.next_actions),
            "scan_returncode": self.scan_result.returncode if self.scan_result else None,
            "validation_returncode": self.validation_result.returncode
            if self.validation_result
            else None,
            "validation_status": self.validation_summary.validation_status
            if self.validation_summary
            else None,
            "auto_load_allowed": self.validation_summary.auto_load_allowed
            if self.validation_summary
            else None,
            "errors": list(self.errors),
        }


def plan_model_onboarding(
    *,
    model: str,
    workspace_path: str | Path,
) -> ModelOnboardingPlan:
    """Plan scanner/validator outputs without touching model weights."""

    workspace_root = Path(workspace_path).expanduser().resolve()
    scan_report_path, validation_report_path = workspace_auto_model_report_paths(workspace_root)
    return ModelOnboardingPlan(
        model=model,
        workspace_path=workspace_root,
        scan_report_path=scan_report_path,
        validation_report_path=validation_report_path,
        next_actions=(
            f"Write scanner measurement report to {scan_report_path}.",
            f"Write validator startup-gate report to {validation_report_path}.",
            "Run with execute=True only when the caller explicitly wants to scan and validate.",
            (
                "Fail closed: do not auto-load this model until validation_status is accepted "
                "and auto_load_allowed is true."
            ),
        ),
    )


def onboard_model(
    *,
    model: str,
    workspace_path: str | Path,
    execute: bool = False,
    repo_root: str | Path | None = None,
    scan_fn: ScanCallable = run_model_scan,
    validate_fn: ValidateCallable = run_model_validate,
) -> ModelOnboardingResult:
    """Plan or execute guided onboarding for a model path/id.

    Tests can inject ``scan_fn`` and ``validate_fn`` to avoid subprocesses or
    model loading. The default callables are the existing standalone helpers.
    """

    plan = plan_model_onboarding(model=model, workspace_path=workspace_path)
    if not execute:
        return ModelOnboardingResult(
            plan=plan,
            execute=False,
            status=PLANNED,
            summary=(
                "planned model onboarding only; no scan, validation, model load, or indexing ran"
            ),
            next_actions=plan.next_actions,
        )

    plan.scan_report_path.parent.mkdir(parents=True, exist_ok=True)
    plan.validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else None

    scan_result = scan_fn(model=model, output=plan.scan_report_path, repo_root=root)
    if scan_result.returncode != 0:
        errors = (f"model scan failed with returncode {scan_result.returncode}",)
        return ModelOnboardingResult(
            plan=plan,
            execute=True,
            status=REJECTED,
            summary="rejected: scan did not produce trusted measurement evidence",
            scan_result=scan_result,
            next_actions=_rejected_next_actions(plan, errors),
            errors=errors,
        )

    validation_result = validate_fn(
        report=plan.scan_report_path,
        output=plan.validation_report_path,
        model=model,
        repo_root=root,
    )
    if validation_result.returncode != 0:
        errors = (f"model validation failed with returncode {validation_result.returncode}",)
        return ModelOnboardingResult(
            plan=plan,
            execute=True,
            status=REJECTED,
            summary="rejected: validator did not accept the scanner report",
            scan_result=scan_result,
            validation_result=validation_result,
            next_actions=_rejected_next_actions(plan, errors),
            errors=errors,
        )

    validation_summary = summarize_validation_report(plan.validation_report_path)
    status = _status_from_validation_summary(validation_summary)
    summary = _summary_text(status, validation_summary)
    errors = tuple(validation_summary.rejection_reasons) if status == REJECTED else ()

    return ModelOnboardingResult(
        plan=plan,
        execute=True,
        status=status,
        summary=summary,
        scan_result=scan_result,
        validation_result=validation_result,
        validation_summary=validation_summary,
        next_actions=_next_actions_for_status(plan, status, validation_summary),
        errors=errors,
    )


def _status_from_validation_summary(summary: ValidationReportSummary) -> str:
    if summary.can_auto_load:
        return ACCEPTED
    if summary.readable and summary.validation_status in REVIEW_STATUSES:
        return NEEDS_REVIEW
    return REJECTED


def _summary_text(status: str, summary: ValidationReportSummary) -> str:
    if status == ACCEPTED:
        return "accepted: validator allows auto-load for the selected adapter config"
    if status == NEEDS_REVIEW:
        return (
            "needs_review: fail closed; manual attestation may permit standard_decode only "
            "after operator review"
        )
    reason = "; ".join(summary.rejection_reasons) if summary.rejection_reasons else "unknown"
    return f"rejected: fail closed; {reason}"


def _next_actions_for_status(
    plan: ModelOnboardingPlan,
    status: str,
    summary: ValidationReportSummary,
) -> tuple[str, ...]:
    if status == ACCEPTED:
        return (
            "Use the accepted validator report as the startup gate input.",
            (
                "Auto-load is allowed only for the selected adapter/model/tokenizer/layer "
                "scope in the report."
            ),
            "Keep scanner and validator reports with workspace provenance for later verification.",
        )
    if status == NEEDS_REVIEW:
        return _needs_review_next_actions(plan, summary)
    return _rejected_next_actions(plan, summary.rejection_reasons)


def _needs_review_next_actions(
    plan: ModelOnboardingPlan,
    summary: ValidationReportSummary,
) -> tuple[str, ...]:
    warnings = "; ".join(summary.warnings[:3]) if summary.warnings else "none"
    return (
        "Fail closed: do not auto-load this model for KV/residual materialization.",
        f"Review validator warnings/reasons in {plan.validation_report_path}; warnings={warnings}.",
        (
            "If an operator accepts the risk, create a manual model attestation bound to the "
            "validation report SHA256 and selected_config."
        ),
        (
            "Manual attestation guidance: allow only standard_decode; do not attest kv_replay, "
            "kv_direct, residual_replay, or tensor replay capabilities."
        ),
        (
            "After attestation, re-run startup checks so the attestation verifier can bind "
            "report digest, model identity, tokenizer identity, adapter family, and expiry."
        ),
    )


def _rejected_next_actions(
    plan: ModelOnboardingPlan,
    reasons: tuple[str, ...],
) -> tuple[str, ...]:
    reason_text = "; ".join(reasons) if reasons else "validator did not prove startup safety"
    return (
        f"Fail closed: reject auto-load for {plan.model}.",
        f"Fix scanner/validator evidence before retrying; reason={reason_text}.",
        (
            "Do not create a manual attestation for a rejected report; produce a fresh "
            "accepted or needs_review validation report first."
        ),
    )
