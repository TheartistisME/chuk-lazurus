"""Extensible, source-snippet-safe grading for IDDIA and tool-output JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .state import (
    DEFAULT_META_ROOT,
    safe_slug,
    sha256_path,
    utc_now,
    utc_stamp,
    write_json,
)

TEXT_OMIT_KEYS = {"text", "snippet", "content", "source_text", "quote"}
DEFAULT_EXPECTED_CONCEPTS = (
    "source-of-truth",
    "derived-index",
    "schema",
    "durability",
    "consistency",
    "replay",
    "provenance-lineage",
)


@dataclass(frozen=True)
class GradeRecord:
    payload: dict[str, Any]
    path: Path


def normalize_signal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def signal_variants(value: str) -> set[str]:
    normalized = normalize_signal(value)
    variants = {normalized, normalized.replace("-", ""), normalized.replace("-", " ")}
    if normalized == "provenance-lineage":
        variants.update({"provenance", "lineage", "manifest"})
    if normalized == "derived-index":
        variants.update({"derived", "index", "cache", "materialized view"})
    if normalized == "source-of-truth":
        variants.update({"canonical", "authoritative", "system of record"})
    if normalized == "failure-recovery":
        variants.update({"failure", "recovery", "crash", "restart"})
    return {variant for variant in variants if variant}


def safe_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[omitted: grade records do not quote source text]"
                if key in TEXT_OMIT_KEYS
                else safe_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [safe_payload(item) for item in value]
    return value


def flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            if key in TEXT_OMIT_KEYS:
                continue
            parts.append(str(key))
            parts.append(flatten_text(item))
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def get_path(payload: Any, dotted_path: str) -> Any:
    current = payload
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return None
    return current


def payload_hits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hits = payload.get("hits")
    if isinstance(hits, list):
        return [hit for hit in hits if isinstance(hit, dict)]
    scenarios = payload.get("scenarios")
    if isinstance(scenarios, list):
        collected: list[dict[str, Any]] = []
        for scenario in scenarios:
            if isinstance(scenario, dict):
                collected.extend(payload_hits(scenario))
        return collected
    return []


def hit_signal_text(hit: dict[str, Any]) -> str:
    return flatten_text(
        {
            "chunk_id": hit.get("chunk_id"),
            "page": hit.get("page"),
            "score": hit.get("score"),
            "principle_tags": hit.get("principle_tags"),
            "stage_tags": hit.get("stage_tags"),
            "concept_tags": hit.get("concept_tags"),
            "matched_tags": hit.get("matched_tags"),
            "matched_terms": hit.get("matched_terms"),
            "chapter_title": hit.get("chapter_title"),
            "section_title": hit.get("section_title"),
            "why_this_hit": hit.get("why_this_hit"),
        }
    ).lower()


def concept_matched(concept: str, hits: list[dict[str, Any]], payload_text: str) -> bool:
    variants = signal_variants(concept)
    if hits:
        return any(any(variant in hit_signal_text(hit) for variant in variants) for hit in hits)
    if any(variant in payload_text for variant in variants):
        return True
    return False


def chapter_matched(chapter: str, hit: dict[str, Any]) -> bool:
    target = normalize_signal(chapter)
    location = normalize_signal(
        " ".join(str(hit.get(key, "")) for key in ("chapter_title", "section_title"))
    )
    return bool(target and target in location)


def label_score(score: float) -> str:
    if score >= 4.5:
        return "strong"
    if score >= 4.0:
        return "good"
    if score >= 3.0:
        return "mixed"
    return "weak"


def criterion(
    criterion_id: str,
    label: str,
    score: float,
    *,
    weight: float,
    notes: list[str],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = round(max(0.0, min(1.0, score)), 3)
    return {
        "id": criterion_id,
        "label": label,
        "score": score,
        "weight": weight,
        "passed": score >= 0.8,
        "notes": notes,
        "details": details or {},
    }


def default_criteria(
    payload: dict[str, Any],
    *,
    expected_concepts: tuple[str, ...],
    preferred_chapters: tuple[str, ...],
) -> list[dict[str, Any]]:
    hits = payload_hits(payload)
    text = flatten_text(payload).lower()
    matched = tuple(
        concept for concept in expected_concepts if concept_matched(concept, hits, text)
    )
    missing = tuple(concept for concept in expected_concepts if concept not in matched)
    concept_score = len(matched) / max(1, len(expected_concepts))

    criteria = [
        criterion(
            "expected_concept_coverage",
            "Expected concept coverage",
            concept_score,
            weight=1.4,
            notes=(
                ["matched expected concepts"]
                if not missing
                else [f"missing concepts: {', '.join(missing)}"]
            ),
            details={"matched": matched, "missing": missing, "expected": expected_concepts},
        )
    ]

    if preferred_chapters:
        preferred_hits = [
            str(hit.get("chunk_id", hit.get("page", "unknown")))
            for hit in hits
            if any(chapter_matched(chapter, hit) for chapter in preferred_chapters)
        ]
        chapter_score = min(1.0, len(preferred_hits) / max(1, min(2, len(hits) or 1)))
        notes = (
            [f"preferred chapter hits: {len(preferred_hits)}"]
            if preferred_hits
            else ["no preferred chapter hit"]
        )
    else:
        preferred_hits = []
        chapter_score = 1.0
        notes = ["no preferred chapters configured; criterion treated as satisfied"]
    criteria.append(
        criterion(
            "preferred_chapter_coverage",
            "Preferred chapter coverage",
            chapter_score,
            weight=0.9,
            notes=notes,
            details={"preferred_chapters": preferred_chapters, "hit_ids": preferred_hits},
        )
    )

    if hits:
        top_hit = hits[0]
        top_text = hit_signal_text(top_hit)
        top_has_concept = any(
            variant in top_text
            for concept in expected_concepts
            for variant in signal_variants(concept)
        )
        top_has_chapter = any(chapter_matched(chapter, top_hit) for chapter in preferred_chapters)
        score = 1.0 if top_has_concept or top_has_chapter else 0.25
        note = (
            "top hit carries expected signal" if score == 1.0 else "top hit lacks expected signal"
        )
    else:
        score = 0.0
        note = "no hits available"
    criteria.append(
        criterion(
            "top_hit_relevance",
            "Top-hit relevance",
            score,
            weight=1.0,
            notes=[note],
        )
    )

    noisy_hits = [hit for hit in hits if hit.get("noise_flags")]
    noise_score = 1.0 if not hits else 1.0 - (len(noisy_hits) / len(hits))
    criteria.append(
        criterion(
            "noise_flags",
            "Noise flags",
            noise_score,
            weight=0.8,
            notes=[f"{len(noisy_hits)} noisy hit(s) out of {len(hits)}"],
            details={"noisy_hit_ids": [hit.get("chunk_id") for hit in noisy_hits]},
        )
    )

    explained_hits = [
        hit
        for hit in hits
        if isinstance(hit.get("why_this_hit"), dict)
        and str(hit["why_this_hit"].get("summary", "")).strip()
    ]
    explanation_score = (
        len(explained_hits) / max(1, len(hits))
        if hits
        else (1.0 if any(term in text for term in ("why", "explanation", "rationale")) else 0.0)
    )
    criteria.append(
        criterion(
            "explanation_coverage",
            "Explanation and why coverage",
            explanation_score,
            weight=0.9,
            notes=[f"{len(explained_hits)} explained hit(s) out of {len(hits)}"],
        )
    )

    backlog_terms = ("improvement backlog", "recommendation", "recommended", "next steps")
    backlog_score = 1.0 if any(term in text for term in backlog_terms) else 0.0
    if isinstance(payload.get("improvement_backlog"), list) and payload["improvement_backlog"]:
        backlog_score = 1.0
    criteria.append(
        criterion(
            "improvement_backlog_clarity",
            "Improvement backlog clarity",
            backlog_score,
            weight=0.7,
            notes=[
                "clear improvement or next-step backlog present"
                if backlog_score
                else "no clear improvement backlog found"
            ],
        )
    )

    mandatory_fields = (
        "files_modified",
        "agent_objective",
        "tldr",
        "mandatory_dependencies",
    )
    present = [
        field for field in mandatory_fields if get_path(payload, field) not in (None, "", [])
    ]
    package_contract = [field for field in ("stage", "task", "next_stage") if payload.get(field)]
    handoff_score = max(len(present) / len(mandatory_fields), len(package_contract) / 3 * 0.6)
    criteria.append(
        criterion(
            "mandatory_handoff_signoff",
            "Mandatory handoff and signoff info",
            handoff_score,
            weight=0.8,
            notes=[
                f"signoff fields present: {', '.join(present) or 'none'}",
                f"package contract fields present: {', '.join(package_contract) or 'none'}",
            ],
            details={"required_signoff_fields": mandatory_fields},
        )
    )

    return criteria


def configured_criteria(
    payload: dict[str, Any], config: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not config:
        return []
    text = flatten_text(payload).lower()
    criteria: list[dict[str, Any]] = []
    for raw in config.get("criteria", []):
        if not isinstance(raw, dict):
            continue
        criterion_id = str(raw.get("id", "custom"))
        label = str(raw.get("label", criterion_id.replace("_", " ").title()))
        weight = float(raw.get("weight", 0.5))
        kind = str(raw.get("kind", "required_terms"))
        if kind == "required_fields":
            fields = tuple(str(path) for path in raw.get("paths", []))
            present = [path for path in fields if get_path(payload, path) not in (None, "", [])]
            score = len(present) / max(1, len(fields))
            notes = [f"present fields: {', '.join(present) or 'none'}"]
        else:
            terms = tuple(str(term).lower() for term in raw.get("terms", []))
            matched = [term for term in terms if term in text]
            score = len(matched) / max(1, len(terms))
            notes = [f"matched terms: {', '.join(matched) or 'none'}"]
        criteria.append(criterion(criterion_id, label, score, weight=weight, notes=notes))
    return criteria


def recommendation(criteria: list[dict[str, Any]]) -> str:
    weak = [item for item in criteria if float(item["score"]) < 0.8]
    if not weak:
        return "Grade is healthy. Keep adding harder regression samples before changing behavior."
    labels = ", ".join(item["label"] for item in weak[:3])
    return f"Improve the weakest grading areas first: {labels}."


def grade_payload(
    payload: dict[str, Any],
    *,
    input_path: Path | None = None,
    state_root: Path = DEFAULT_META_ROOT,
    expected_concepts: tuple[str, ...] = DEFAULT_EXPECTED_CONCEPTS,
    preferred_chapters: tuple[str, ...] = (),
    criteria_config: dict[str, Any] | None = None,
) -> GradeRecord:
    criteria = default_criteria(
        payload,
        expected_concepts=expected_concepts,
        preferred_chapters=preferred_chapters,
    )
    criteria.extend(configured_criteria(payload, criteria_config))
    total_weight = sum(float(item["weight"]) for item in criteria) or 1.0
    weighted = sum(float(item["score"]) * float(item["weight"]) for item in criteria)
    overall = round((weighted / total_weight) * 5.0, 2)
    grade_id_source = str(input_path or payload.get("id") or payload.get("task") or "tool-output")
    grade_id = f"{utc_stamp()}-{safe_slug(Path(grade_id_source).stem, 'grade')}"
    record = {
        "schema_version": 1,
        "grade_id": grade_id,
        "graded_at": utc_now(),
        "input_path": str(input_path) if input_path else None,
        "input_sha256": sha256_path(input_path) if input_path and input_path.exists() else None,
        "overall_score": overall,
        "label": label_score(overall),
        "criteria": criteria,
        "recommendation": recommendation(criteria),
        "safe_input_summary": safe_payload(payload),
    }
    output_path = state_root / "grades" / f"{grade_id}.json"
    write_json(output_path, record)
    return GradeRecord(payload=record, path=output_path)


def grade_file(
    input_path: Path,
    *,
    state_root: Path = DEFAULT_META_ROOT,
    expected_concepts: tuple[str, ...] = DEFAULT_EXPECTED_CONCEPTS,
    preferred_chapters: tuple[str, ...] = (),
    criteria_config_path: Path | None = None,
) -> GradeRecord:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    config = None
    if criteria_config_path:
        config = json.loads(criteria_config_path.read_text(encoding="utf-8"))
    return grade_payload(
        payload,
        input_path=input_path,
        state_root=state_root,
        expected_concepts=expected_concepts,
        preferred_chapters=preferred_chapters,
        criteria_config=config,
    )
