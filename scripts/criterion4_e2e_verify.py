#!/usr/bin/env python3
"""Criterion 4 end-to-end verification harness.

Runs the cross-session demo and validates acceptance criterion 4:
  - End-to-end conversational roundtrip: chat_loop → session_close → cross-session
    retrieval in a fresh empty-context session retrieves VERBATIM content from
    ≥1 prior session via an exact-ID probe, with coherent surrounding output.

Usage:
  python scripts/criterion4_e2e_verify.py [--output-root /tmp/csd-criterion4] [--reuse]

Exit codes:
  0 = CRITERION_4_PASS
  1 = CRITERION_4_FAIL
  2 = CRITERION_4_ERROR (build blocker)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="criterion4_e2e_verify.py",
        description="Criterion 4 acceptance verification"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/csd-criterion4"),
        help="Output directory for demo artifacts"
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Skip build phase; reuse existing checkpoints"
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).resolve()
    report_path = output_root / "report.json"
    inputs_root = output_root / "inputs"
    checkpoints_root = output_root / "checkpoints"

    # Run the demo if not reusing.
    if not args.reuse:
        inputs_root.mkdir(parents=True, exist_ok=True)
        checkpoints_root.mkdir(parents=True, exist_ok=True)

        from chuk_lazarus.cross_session_demo.cli import run_demo

        try:
            demo = run_demo(
                inputs_root=inputs_root,
                checkpoints_root=checkpoints_root,
                device="cuda",
            )
            report = demo.report
            # Persist report to JSON
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        except Exception as e:
            print(f"CRITERION_4_ERROR\nreason={str(e)}")
            return 2
    else:
        # Reuse mode: report must already exist.
        if not report_path.exists():
            print(f"CRITERION_4_ERROR\nreason=--reuse specified but {report_path} not found")
            return 2

    # Parse the report.
    try:
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"CRITERION_4_ERROR\nreason=Failed to load report JSON: {e}")
        return 2

    # Criterion 4 validation checks.
    verbatim_hits = report_data.get("verbatim_recall_per_query", [])
    all_strict_pass = report_data.get("all_six_strict_per_query_pass", False)
    query_executions = report_data.get("query_executions", [])

    # Validate: at least one exact-id query with verbatim hit.
    exact_id_with_verbatim = False
    first_match = None
    for execution in query_executions:
        if execution.get("mode") == "exact" and execution.get("verbatim_hit"):
            exact_id_with_verbatim = True
            first_match = execution
            break

    if not exact_id_with_verbatim:
        print("CRITERION_4_FAIL")
        print(f"reason=No exact-id query with verbatim_hit=true found")
        return 1

    # Validate: coherent output (non-empty, contains sentences).
    answer = first_match.get("generated_answer", "")
    answer_excerpt = answer[:500]
    is_coherent = bool(answer) and len(answer) > 20 and ("." in answer or "?" in answer)

    if not is_coherent:
        print("CRITERION_4_FAIL")
        print(f"reason=Generated answer not coherent: {answer_excerpt[:100]}")
        return 1

    # Validate: strict assertions fired.
    strict_assertions = first_match.get("strict_assertions", {})
    hook_fired = strict_assertions.get("hook_fired", False)

    if not hook_fired:
        print("CRITERION_4_FAIL")
        print("reason=Strict assertion hook_fired=false")
        return 1

    # All checks passed.
    print("CRITERION_4_PASS")
    planted_phrase = first_match.get("planted_phrase", "N/A")
    print(f"planted_phrase={planted_phrase}")
    print(f"answer_excerpt={answer_excerpt}")
    print(f"strict_assertions={strict_assertions}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
