from __future__ import annotations

import json

from IDDIA.meta.activation import ActivationPolicy, evaluate_activation
from IDDIA.meta.grader import grade_payload
from IDDIA.meta.signoff import append_changelog, append_signoff, helper_context
from IDDIA.meta.spawner import spawn_improvement_agent


def sample_package() -> dict:
    return {
        "stage": "plan",
        "next_stage": "build",
        "task": "Plan a replayable event log projection",
        "next_steps": "Recommended next step: add deterministic rebuild checks",
        "hits": [
            {
                "chunk_id": "ddia-p0465-c00",
                "chapter_title": "Chapter 11: Stream Processing",
                "section_title": "Event Sourcing",
                "principle_tags": ["source-of-truth"],
                "concept_tags": ["event-log", "materialized-view", "schema"],
                "matched_tags": ["event-log", "materialized-view"],
                "matched_terms": ["replay", "schema"],
                "noise_flags": [],
                "why_this_hit": {"summary": "matched event log and schema"},
                "text": "This source text should not be copied into the grade record.",
            }
        ],
        "improvement_backlog": ["Add a harder crash-recovery scenario."],
        "files_modified": ["IDDIA/meta/grader.py"],
        "agent_objective": "Grade package quality",
        "tldr": "Added a grader",
        "mandatory_dependencies": ["stdlib only"],
    }


def test_meta_grader_scores_and_redacts_source_text(tmp_path):
    grade = grade_payload(
        sample_package(),
        state_root=tmp_path,
        expected_concepts=("event-log", "materialized-view", "schema"),
        preferred_chapters=("Stream Processing",),
    )

    assert grade.path.exists()
    assert grade.payload["overall_score"] >= 4.5
    assert grade.payload["label"] == "strong"
    assert "source text should not be copied" not in json.dumps(grade.payload)
    assert any(
        item["id"] == "mandatory_handoff_signoff" and item["passed"]
        for item in grade.payload["criteria"]
    )


def test_meta_grader_does_not_count_query_only_concepts_when_hits_exist(tmp_path):
    package = sample_package()
    package["task"] = "Plan manifest lineage and replay for schema migration"
    package["next_steps"] = "Add rollback manifests"
    package["hits"][0]["concept_tags"] = ["schema"]
    package["hits"][0]["matched_tags"] = ["schema"]
    package["hits"][0]["matched_terms"] = ["schema"]

    grade = grade_payload(
        package,
        state_root=tmp_path,
        expected_concepts=("schema", "provenance-lineage", "replay", "manifest"),
        preferred_chapters=("Stream Processing",),
    )
    concept_criterion = next(
        item for item in grade.payload["criteria"] if item["id"] == "expected_concept_coverage"
    )

    assert concept_criterion["details"]["matched"] == ("schema",)
    assert "replay" in concept_criterion["details"]["missing"]


def test_activation_policy_every_n_and_threshold(tmp_path):
    grade = {"grade_id": "grade-1", "overall_score": 4.7}
    policy = ActivationPolicy(mode="every_n", every_n=3)

    assert not evaluate_activation(grade, policy=policy, state_root=tmp_path).should_spawn
    assert not evaluate_activation(grade, policy=policy, state_root=tmp_path).should_spawn
    assert evaluate_activation(grade, policy=policy, state_root=tmp_path).should_spawn
    assert not evaluate_activation(grade, policy=policy, state_root=tmp_path).should_spawn

    weak_grade = {"grade_id": "grade-2", "overall_score": 3.6}
    threshold = ActivationPolicy(mode="threshold", target_score=4.0)
    assert evaluate_activation(weak_grade, policy=threshold, state_root=tmp_path).should_spawn


def test_pending_spawn_fallback_writes_request_and_prompt(tmp_path):
    grade = {
        "grade_id": "grade-weak",
        "overall_score": 2.5,
        "label": "weak",
        "recommendation": "Improve coverage.",
        "criteria": [{"id": "coverage", "score": 0.2, "notes": ["missing concept"]}],
    }

    result = spawn_improvement_agent(
        grade,
        objective="Improve IDDIA retrieval grading",
        state_root=tmp_path,
        force_pending=True,
    )

    assert result.status == "pending"
    assert result.request_path.exists()
    assert result.prompt_path.exists()
    request = json.loads(result.request_path.read_text(encoding="utf-8"))
    prompt = result.prompt_path.read_text(encoding="utf-8")
    assert request["status"] == "pending"
    assert "Improve IDDIA retrieval grading" in prompt
    assert "python -m IDDIA.meta helper-context" in prompt
    assert "vee agent kill" in prompt


def test_helper_context_includes_changelog_signoff_and_dependencies(tmp_path):
    append_changelog(
        "Added grading state.",
        state_root=tmp_path,
        timestamp="2026-04-28T00:00:00Z",
    )
    signoff_path = append_signoff(
        files_modified=("IDDIA/meta/grader.py",),
        agent_objective="Build meta system",
        tldr="Added grading and helper context",
        mandatory_dependencies=("stdlib only", "vee via WSL when available"),
        state_root=tmp_path,
        timestamp="2026-04-28T00:01:00Z",
    )

    output = helper_context(state_root=tmp_path)

    assert "Added grading state." in output
    assert "Build meta system" in output
    assert "stdlib only" in output
    assert "vee agent spawn <agent>" in output
    assert signoff_path.parent == tmp_path / "signoffs"
