"""CLI entrypoint for IDDIA meta grading and improvement-agent helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .activation import evaluate_activation, load_policy
from .grader import DEFAULT_EXPECTED_CONCEPTS, grade_file
from .signoff import append_changelog, append_signoff, helper_context
from .spawner import default_vee_agent, default_vee_repo, spawn_improvement_agent
from .state import DEFAULT_META_ROOT
from .supervisor import supervise_spawn


class AgentHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Keep workflow examples readable while still showing defaults."""


def _write_stdout_utf8(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m IDDIA.meta",
        description="Grade IDDIA outputs, spawn optimizer agents, and verify proof signoffs.",
        formatter_class=AgentHelpFormatter,
        epilog="""Agent loop:
  1. Grade a package with fixed expected concepts and preferred chapters.
  2. If weak, spawn an optimizer from the grade.
  3. Supervise until completed, failed_no_signoff, failed_no_proof, or running.
  4. Independently rerun the claimed metric before accepting a proven result.
  5. Append changelog/signoff with proof claim, metric, evidence, and verdict.

Trust rule: tests prove implementation contracts; evals/meta-grades prove retrieval quality.""",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=DEFAULT_META_ROOT,
        help="Local ignored meta state root for grades, spawn requests, transcripts, and signoffs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    grade = subparsers.add_parser(
        "grade",
        help="Grade a JSON package/report against an explicit rubric",
        description=(
            "Score an IDDIA/tool JSON output with concept, chapter, noise, "
            "explanation, handoff, and actionability criteria."
        ),
        formatter_class=AgentHelpFormatter,
        epilog="""Agent notes:
  - Pass every required concept with repeated --expected-concept flags.
  - Pass chapter intent with repeated --preferred-chapter flags.
  - Grades write JSON under IDDIA/artifacts/meta/grades/.
  - score_gates cap false confidence when critical retrieval signals are missing.
  - Use --spawn or --policy only after the rubric is explicit.""",
    )
    grade.add_argument("input", type=Path)
    grade.add_argument(
        "--expected-concept",
        action="append",
        default=[],
        help="Required concept signal; repeat for each concept in the rubric.",
    )
    grade.add_argument(
        "--preferred-chapter",
        action="append",
        default=[],
        help="Preferred DDIA chapter/section direction; repeat when needed.",
    )
    grade.add_argument(
        "--criteria-config",
        type=Path,
        help="Optional JSON file with extra required_terms or required_fields criteria.",
    )
    grade.add_argument("--policy", type=Path, help="Activation policy JSON")
    grade.add_argument(
        "--objective",
        default="Improve IDDIA output quality based on the grade",
        help="Objective sent to a spawned optimizer when activation fires.",
    )
    grade.add_argument(
        "--spawn",
        action="store_true",
        help="Force a Vee/tmux optimizer spawn after grading.",
    )
    grade.add_argument("--vee-repo", type=Path, default=default_vee_repo(), help="Local Vee repo")
    grade.add_argument("--vee-agent", default=default_vee_agent(), help="Vee agent type/name")
    grade.add_argument("--tmux-session", default=None, help="tmux session for spawned agents")
    grade.add_argument("--tmux-window", default=None, help="tmux window for spawned agents")
    grade.add_argument("--worker-name", default=None, help="Stable worker name for the optimizer")

    spawn = subparsers.add_parser(
        "spawn",
        help="Spawn an optimizer from an existing grade JSON",
        description=(
            "Create or reuse a tmux target, launch a Vee worker, and send it a "
            "proof-before-signoff prompt based on a grade record."
        ),
        formatter_class=AgentHelpFormatter,
        epilog="""Agent notes:
  - The spawned agent must stay inside IDDIA unless instructed otherwise.
  - The prompt tells it to state a claim, use a fixed metric, run tests/evals,
    append changelog/signoff, and close itself.
  - Always run `python -m IDDIA.meta supervise <spawn-request>` after spawning.""",
    )
    spawn.add_argument("grade", type=Path)
    spawn.add_argument(
        "--objective",
        default="Improve IDDIA output quality based on the grade",
        help="Concrete improvement objective and proof target for the optimizer.",
    )
    spawn.add_argument("--vee-repo", type=Path, default=default_vee_repo(), help="Local Vee repo")
    spawn.add_argument("--vee-agent", default=default_vee_agent(), help="Vee agent type/name")
    spawn.add_argument("--tmux-session", default=None, help="tmux session for the worker pane")
    spawn.add_argument("--tmux-window", default=None, help="tmux window for the worker pane")
    spawn.add_argument("--worker-name", default=None, help="Stable worker name for supervision")

    supervise = subparsers.add_parser(
        "supervise",
        help="Check a spawned optimizer and validate fresh proof signoff",
        description=(
            "Capture tmux pane output, inspect fresh signoffs created after the "
            "spawn request, and write a status JSON."
        ),
        formatter_class=AgentHelpFormatter,
        epilog="""Statuses:
  running            pane is still active and no fresh proof signoff exists
  completed          fresh signoff has claim, metric, evidence, and verdict
  failed_no_signoff  pane is gone and no matching fresh signoff exists
  failed_no_proof    fresh signoff exists but proof fields are incomplete""",
    )
    supervise.add_argument("request", type=Path)
    supervise.add_argument(
        "--wait-seconds",
        type=float,
        default=0,
        help="Seconds to wait before checking tmux/signoff state.",
    )

    helper = subparsers.add_parser(
        "helper-context",
        help="Print changelog tail, latest signoff, and mandatory agent context",
        description=(
            "Print the local meta memory an optimizer should read before editing: "
            "last changelog lines, latest signoff, and required commands."
        ),
        formatter_class=AgentHelpFormatter,
    )
    helper.set_defaults(command="helper-context")

    changelog = subparsers.add_parser(
        "changelog",
        help="Append durable local meta notes",
        formatter_class=AgentHelpFormatter,
    )
    changelog_sub = changelog.add_subparsers(dest="changelog_command", required=True)
    changelog_append = changelog_sub.add_parser(
        "append",
        help="Append one measured note to the IDDIA meta changelog",
        formatter_class=AgentHelpFormatter,
    )
    changelog_append.add_argument("message")

    signoff = subparsers.add_parser(
        "signoff",
        help="Append an agent handoff with mandatory proof fields",
        formatter_class=AgentHelpFormatter,
    )
    signoff_sub = signoff.add_subparsers(dest="signoff_command", required=True)
    signoff_append = signoff_sub.add_parser(
        "append",
        help="Append a proof-bearing signoff for the next agent",
        description="Record what changed, why, how to understand it, and whether proof succeeded.",
        formatter_class=AgentHelpFormatter,
        epilog="""Proof verdicts:
  proven             fixed metric improved or requirement was fully met
  partially_proven   implementation improved but product metric is incomplete
  not_proven         change could not prove the intended claim""",
    )
    signoff_append.add_argument(
        "--file",
        action="append",
        default=[],
        dest="files",
        help="Modified file; repeat for each relevant path.",
    )
    signoff_append.add_argument("--objective", required=True, help="Agent objective")
    signoff_append.add_argument("--tldr", required=True, help="Short change summary")
    signoff_append.add_argument("--proof-claim", default="", help="Claim being proven")
    signoff_append.add_argument(
        "--proof-metric",
        default="",
        help="Fixed command/rubric used to prove the claim.",
    )
    signoff_append.add_argument(
        "--proof-evidence",
        action="append",
        default=[],
        help="Evidence line; repeat for before/after scores, tests, and evals.",
    )
    signoff_append.add_argument(
        "--proof-verdict",
        default="",
        help="One of proven, partially_proven, not_proven, or a precise local verdict.",
    )
    signoff_append.add_argument(
        "--dependency",
        action="append",
        default=[],
        dest="dependencies",
        help="Mandatory context/command/dependency for the next agent; repeat as needed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    state_root = Path(args.state_root)

    if args.command == "grade":
        expected = tuple(args.expected_concept or DEFAULT_EXPECTED_CONCEPTS)
        preferred = tuple(args.preferred_chapter or ())
        grade = grade_file(
            args.input,
            state_root=state_root,
            expected_concepts=expected,
            preferred_chapters=preferred,
            criteria_config_path=args.criteria_config,
        )
        print(f"grade={grade.path}")
        print(f"score={grade.payload['overall_score']:.2f}/5 {grade.payload['label']}")
        print(f"recommendation={grade.payload['recommendation']}")

        should_spawn = args.spawn
        reason = "forced by --spawn"
        if args.policy:
            decision = evaluate_activation(
                grade.payload,
                policy=load_policy(args.policy),
                state_root=state_root,
            )
            should_spawn = should_spawn or decision.should_spawn
            reason = decision.reason
            print(f"activation={decision.should_spawn} reason={decision.reason}")
        if should_spawn:
            result = spawn_improvement_agent(
                grade.payload,
                objective=f"{args.objective}. Activation reason: {reason}",
                state_root=state_root,
                repo_root=Path.cwd(),
                vee_repo=args.vee_repo,
                vee_agent=args.vee_agent,
                tmux_session=args.tmux_session,
                tmux_window=args.tmux_window,
                worker_name=args.worker_name,
            )
            print(f"spawn_status={result.status}")
            print(f"spawn_request={result.request_path}")
        return 0

    if args.command == "spawn":
        grade_record = json.loads(args.grade.read_text(encoding="utf-8"))
        result = spawn_improvement_agent(
            grade_record,
            objective=args.objective,
            state_root=state_root,
            repo_root=Path.cwd(),
            vee_repo=args.vee_repo,
            vee_agent=args.vee_agent,
            tmux_session=args.tmux_session,
            tmux_window=args.tmux_window,
            worker_name=args.worker_name,
        )
        print(f"spawn_status={result.status}")
        print(f"spawn_request={result.request_path}")
        return 0

    if args.command == "supervise":
        result = supervise_spawn(
            args.request,
            state_root=state_root,
            wait_seconds=args.wait_seconds,
        )
        print(f"supervise_status={result.status}")
        print(f"spawn_run={result.status_path}")
        if result.transcript_path:
            print(f"transcript={result.transcript_path}")
        if result.signoff_path:
            print(f"signoff={result.signoff_path}")
        return 0 if result.status in {"completed", "running"} else 2

    if args.command == "helper-context":
        _write_stdout_utf8(helper_context(state_root=state_root))
        return 0

    if args.command == "changelog" and args.changelog_command == "append":
        path = append_changelog(args.message, state_root=state_root)
        print(f"changelog={path}")
        return 0

    if args.command == "signoff" and args.signoff_command == "append":
        path = append_signoff(
            files_modified=tuple(args.files),
            agent_objective=args.objective,
            tldr=args.tldr,
            mandatory_dependencies=tuple(args.dependencies),
            proof_claim=args.proof_claim,
            proof_metric=args.proof_metric,
            proof_evidence=tuple(args.proof_evidence),
            proof_verdict=args.proof_verdict,
            state_root=state_root,
        )
        print(f"signoff={path}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
