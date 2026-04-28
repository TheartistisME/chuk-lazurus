"""CLI entrypoint for IDDIA meta grading and improvement-agent helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .activation import evaluate_activation, load_policy
from .grader import DEFAULT_EXPECTED_CONCEPTS, grade_file
from .signoff import append_changelog, append_signoff, helper_context
from .spawner import default_vee_repo, spawn_improvement_agent
from .state import DEFAULT_META_ROOT


def _write_stdout_utf8(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m IDDIA.meta",
        description="Grade IDDIA/tool outputs and optionally spawn improvement agents.",
    )
    parser.add_argument("--state-root", type=Path, default=DEFAULT_META_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    grade = subparsers.add_parser("grade", help="Grade a tool-output JSON package/report")
    grade.add_argument("input", type=Path)
    grade.add_argument("--expected-concept", action="append", default=[])
    grade.add_argument("--preferred-chapter", action="append", default=[])
    grade.add_argument("--criteria-config", type=Path)
    grade.add_argument("--policy", type=Path, help="Activation policy JSON")
    grade.add_argument("--objective", default="Improve IDDIA output quality based on the grade")
    grade.add_argument("--spawn", action="store_true", help="Force a spawn attempt after grading")
    grade.add_argument("--no-start", action="store_true", help="Spawn but do not start the vee agent")
    grade.add_argument("--vee-repo", type=Path, default=default_vee_repo())

    spawn = subparsers.add_parser("spawn", help="Spawn from an existing grade JSON")
    spawn.add_argument("grade", type=Path)
    spawn.add_argument("--objective", default="Improve IDDIA output quality based on the grade")
    spawn.add_argument("--no-start", action="store_true")
    spawn.add_argument("--vee-repo", type=Path, default=default_vee_repo())

    helper = subparsers.add_parser("helper-context", help="Print helper context for agents")
    helper.set_defaults(command="helper-context")

    changelog = subparsers.add_parser("changelog", help="Append to the IDDIA meta changelog")
    changelog_sub = changelog.add_subparsers(dest="changelog_command", required=True)
    changelog_append = changelog_sub.add_parser("append")
    changelog_append.add_argument("message")

    signoff = subparsers.add_parser("signoff", help="Append an IDDIA meta signoff")
    signoff_sub = signoff.add_subparsers(dest="signoff_command", required=True)
    signoff_append = signoff_sub.add_parser("append")
    signoff_append.add_argument("--file", action="append", default=[], dest="files")
    signoff_append.add_argument("--objective", required=True)
    signoff_append.add_argument("--tldr", required=True)
    signoff_append.add_argument(
        "--dependency",
        action="append",
        default=[],
        dest="dependencies",
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
                start=not args.no_start,
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
            start=not args.no_start,
        )
        print(f"spawn_status={result.status}")
        print(f"spawn_request={result.request_path}")
        return 0

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
            state_root=state_root,
        )
        print(f"signoff={path}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
