"""``python -m chuk_lazarus.cross_session_demo.cli`` entry point.

Orchestrates the end-to-end demo:

1. Run axes 1-3 per plan.
2. Measure tokens (>128k gate).
3. Construct a fresh SessionRetriever.
4. Run the three query paths.
5. Build + persist the VerificationReport.
6. Print a green/red summary and exit 0/1 accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from chuk_lazarus.cross_session_demo.pipeline import run_session
from chuk_lazarus.cross_session_demo.report import VerificationReport, build_report
from chuk_lazarus.cross_session_demo.session_generator import (
    DEFAULT_PLANS,
    SessionPlan,
)
from chuk_lazarus.cross_session_demo.token_budget import (
    GEMMA_SNAPSHOT_DEFAULT,
    measure_total_tokens,
)
from chuk_lazarus.cross_session_demo.verification import run_cross_session_queries

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class DemoResult:
    """Thin container returned by :func:`run_demo`."""

    report: VerificationReport
    sessions: list[dict]


def _load_plans(plans_json: Path | None) -> list[SessionPlan]:
    if plans_json is None:
        return list(DEFAULT_PLANS)
    raw = json.loads(Path(plans_json).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(
            f"--plans-json must be a list of plan objects, got {type(raw).__name__}"
        )
    return [SessionPlan.model_validate(item) for item in raw]


def run_demo(
    *,
    inputs_root: Path,
    checkpoints_root: Path,
    plans: list[SessionPlan] | None = None,
    device: str = "cuda",
    model_id: str = "google/gemma-4-E2B-it",
    tokenizer_path: str = GEMMA_SNAPSHOT_DEFAULT,
    skip_build: bool = False,
) -> DemoResult:
    """Top-level demo runner consumed by the CLI and the example wrapper."""
    # Deferred import: heavy CUDA/transformers pulls happen only when the
    # retriever actually constructs.
    from chuk_lazarus.session_retrieval import SessionRetriever

    selected_plans = plans if plans is not None else list(DEFAULT_PLANS)
    if len(selected_plans) < 5:
        raise RuntimeError(
            f"cross_session_demo requires >= 5 plans, got {len(selected_plans)}"
        )

    inputs_root = Path(inputs_root).resolve()
    checkpoints_root = Path(checkpoints_root).resolve()

    sessions: list[dict] = []
    if skip_build:
        # skip_build expects the inputs/checkpoints to already exist on disk.
        # This branch is only useful for post-hoc report regeneration and is
        # intentionally not exercised by the default CI run.
        raise RuntimeError(
            "--skip-build is not yet supported; run the full pipeline instead."
        )

    for plan in selected_plans:
        result = run_session(plan, inputs_root, checkpoints_root, device=device)
        sessions.append(result)

    total_tokens = measure_total_tokens(
        [s["full_text"] for s in sessions],
        tokenizer_path=tokenizer_path,
    )

    retriever = SessionRetriever.from_checkpoint_root(
        checkpoints_root,
        original_input_root=inputs_root,
        device=device,
        model_id=model_id,
    )

    executions = run_cross_session_queries(retriever, sessions)
    report = build_report(
        sessions=sessions,
        total_tokens=total_tokens,
        executions=executions,
        repo_root=_REPO_ROOT,
    )
    return DemoResult(report=report, sessions=sessions)


def _format_summary(report: VerificationReport) -> str:
    lines: list[str] = []
    lines.append(f"sessions                          = {report.num_sessions}")
    lines.append(f"total_tokens                      = {report.total_tokens}")
    lines.append(f"token_budget_met (>128000)        = {report.token_budget_met}")
    lines.append(f"all_six_strict_per_query_pass     = {report.all_six_strict_per_query_pass}")
    lines.append(f"verbatim_recall_per_query         = {report.verbatim_recall_per_query}")
    dirty = [p for p, v in report.zero_mod_upstream.items() if v != "clean"]
    lines.append(f"zero_mod_upstream dirty paths     = {dirty}")
    status = "PASS" if report.acceptance_passed else "FAIL"
    lines.append(f"\nACCEPTANCE: {status}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m chuk_lazarus.cross_session_demo.cli",
        description="Axis-5 cross-session demo / acceptance gate.",
    )
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--plans-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it")
    parser.add_argument("--tokenizer-path", default=GEMMA_SNAPSHOT_DEFAULT)
    parser.add_argument("--skip-build", action="store_true")

    args = parser.parse_args(argv)

    plans = _load_plans(args.plans_json)
    demo = run_demo(
        inputs_root=args.inputs_root,
        checkpoints_root=args.checkpoints_root,
        plans=plans,
        device=args.device,
        model_id=args.model_id,
        tokenizer_path=args.tokenizer_path,
        skip_build=args.skip_build,
    )

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        demo.report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print(_format_summary(demo.report))
    return 0 if demo.report.acceptance_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DemoResult", "main", "run_demo"]
