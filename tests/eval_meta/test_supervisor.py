from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval_meta.context import latest_signoff, tail_lines
from scripts.eval_meta.supervisor import (
    build_prompt,
    decide_activation,
    grade_file,
    run_vee_spawn,
)


def test_grades_tradeguru_jsonl_and_flags_memory_regression(tmp_path: Path) -> None:
    path = tmp_path / "eval_results_sample.jsonl"
    rows = [
        {
            "condition": "memory_off_empty_store",
            "question_id": "q1",
            "score": {"total": 8, "max_total": 8},
        },
        {
            "condition": "memory_on_tradeguru_store",
            "question_id": "q1",
            "score": {"total": 4, "max_total": 8},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    grade = grade_file(path)

    assert grade.kind == "tradeguru_jsonl"
    assert grade.status == "needs_improvement"
    assert grade.percent == 75.0
    assert "memory_on_tradeguru_store" in grade.metrics["conditions"]
    assert any("underperformed" in finding for finding in grade.findings)


def test_grades_summary_json_with_warnings(tmp_path: Path) -> None:
    path = tmp_path / "eval_results.jsonl.summary.json"
    path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "memory_off_empty_store": {"mean_score": 8, "scores": [8, 8]},
                    "memory_on_tradeguru_store": {"mean_score": 6, "scores": [6, 6]},
                },
                "warnings": ["noise store reused"],
            }
        ),
        encoding="utf-8",
    )

    grade = grade_file(path)

    assert grade.kind == "summary_json"
    assert grade.status == "needs_improvement"
    assert any("Eval warning" in finding for finding in grade.findings)


def test_grades_generic_tool_output(tmp_path: Path) -> None:
    path = tmp_path / "tool.txt"
    path.write_text("ok\nWARNING: slow\nTraceback happened\nFAILED test_x\n", encoding="utf-8")

    grade = grade_file(path)

    assert grade.kind == "generic_text"
    assert grade.status == "needs_improvement"
    assert grade.metrics["error_markers"] >= 2
    assert grade.metrics["warning_markers"] >= 1


def test_activation_policy() -> None:
    grade = type("Gradeish", (), {"status": "needs_improvement"})()

    assert decide_activation("never", 10, grade).should_spawn is False
    assert decide_activation("always", 10, grade).should_spawn is True
    assert decide_activation("every:3", 6, grade).should_spawn is True
    assert decide_activation("every:3", 7, grade).should_spawn is False

    passing = type("Gradeish", (), {"status": "pass"})()
    assert decide_activation("always", 10, passing).should_spawn is False
    assert decide_activation("always", 10, passing, spawn_on_pass=True).should_spawn is True


def test_context_tail_and_latest_signoff(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("\n".join(f"line {number}" for number in range(150)), encoding="utf-8")
    assert tail_lines(changelog, 100).splitlines()[0] == "line 50"

    signoffs = tmp_path / "signoffs"
    signoffs.mkdir()
    record = {
        "files_modified": ["a.py"],
        "agent_objective": "fix eval",
        "tldr_of_change": "tightened scoring",
        "dependencies_mandatory_for_system_understanding": ["pytest"],
    }
    (signoffs / "001.json").write_text(json.dumps(record), encoding="utf-8")

    latest = latest_signoff(signoffs)

    assert latest["files_modified"] == ["a.py"]
    assert latest["agent_objective"] == "fix eval"


def test_prompt_contains_required_signoff_fields(tmp_path: Path) -> None:
    path = tmp_path / "tool.txt"
    path.write_text("all good\n", encoding="utf-8")
    grade = grade_file(path)

    prompt = build_prompt(grade, "change tail", {"raw_text": "prior signoff"})

    assert "change tail" in prompt
    assert "prior signoff" in prompt
    for field in (
        "files_modified",
        "agent_objective",
        "tldr_of_change",
        "dependencies_mandatory_for_system_understanding",
    ):
        assert field in prompt


def test_vee_spawn_is_dry_run_by_default() -> None:
    result = run_vee_spawn("prompt", spawn=False, vee_binary="missing-vee-for-test")

    assert result["dry_run"] is True
    assert result["spawned"] is False


def test_vee_spawn_requires_binary_when_requested() -> None:
    with pytest.raises(RuntimeError):
        run_vee_spawn("prompt", spawn=True, vee_binary="missing-vee-for-test")
