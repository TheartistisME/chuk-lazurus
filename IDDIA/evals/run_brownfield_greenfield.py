from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # Allow direct script execution from repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from IDDIA.core import DEFAULT_ARTIFACT_ROOT, STAGES, build_context_package

MODE_VALUES = {"brownfield", "greenfield"}
COMPLEXITY_VALUES = {"simple", "medium", "complex"}
DEFAULT_SCENARIOS_PATH = Path(__file__).with_name("brownfield_greenfield_scenarios.json")
DEFAULT_OUTPUT_ROOT = DEFAULT_ARTIFACT_ROOT / "evals" / "brownfield-greenfield"


@dataclass(frozen=True)
class Scenario:
    id: str
    mode: str
    complexity: str
    stage: str
    task: str
    next_steps: str
    expected_concepts: tuple[str, ...]
    preferred_chapters: tuple[str, ...]


@dataclass(frozen=True)
class Grade:
    value: float
    label: str
    matched_concepts: tuple[str, ...]
    missing_concepts: tuple[str, ...]
    preferred_chapter_hits: int
    noisy_hits: int
    explained_hits: int
    top_hit_has_expected_signal: bool
    notes: tuple[str, ...]


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def normalize_signal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def signal_variants(value: str) -> set[str]:
    normalized = normalize_signal(value)
    variants = {normalized}
    variants.add(normalized.replace("-", ""))
    variants.add(normalized.replace("-", " "))
    if normalized == "provenance-lineage":
        variants.update({"lineage", "provenance", "manifest"})
    if normalized == "failure-recovery":
        variants.update({"failure", "recovery", "fault-tolerance", "crash-recovery"})
    if normalized == "materialized-view":
        variants.update({"materialized", "derived-data", "projection", "index"})
    if normalized == "event-log":
        variants.update({"log", "event-sourcing", "append-only"})
    if normalized == "source-of-truth":
        variants.update({"canonical", "authoritative", "system-of-record"})
    return variants


def load_scenarios(path: Path = DEFAULT_SCENARIOS_PATH) -> list[Scenario]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path} has unsupported schema_version")

    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for raw in payload.get("scenarios", []):
        scenario = Scenario(
            id=str(raw["id"]),
            mode=str(raw["mode"]),
            complexity=str(raw["complexity"]),
            stage=str(raw["stage"]),
            task=str(raw["task"]),
            next_steps=str(raw["next_steps"]),
            expected_concepts=tuple(raw["expected_concepts"]),
            preferred_chapters=tuple(raw["preferred_chapters"]),
        )
        validate_scenario(scenario)
        if scenario.id in seen:
            raise ValueError(f"duplicate scenario id: {scenario.id}")
        seen.add(scenario.id)
        scenarios.append(scenario)
    if not scenarios:
        raise ValueError(f"{path} does not define any scenarios")
    return scenarios


def validate_scenario(scenario: Scenario) -> None:
    if scenario.mode not in MODE_VALUES:
        raise ValueError(f"{scenario.id}: invalid mode {scenario.mode!r}")
    if scenario.complexity not in COMPLEXITY_VALUES:
        raise ValueError(f"{scenario.id}: invalid complexity {scenario.complexity!r}")
    if scenario.stage not in STAGES:
        raise ValueError(f"{scenario.id}: invalid stage {scenario.stage!r}")
    if not scenario.task.strip():
        raise ValueError(f"{scenario.id}: task is empty")
    if not scenario.next_steps.strip():
        raise ValueError(f"{scenario.id}: next_steps is empty")
    if not scenario.expected_concepts:
        raise ValueError(f"{scenario.id}: expected_concepts is empty")
    if not scenario.preferred_chapters:
        raise ValueError(f"{scenario.id}: preferred_chapters is empty")


def hit_signal_text(hit: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "principle_tags",
        "concept_tags",
        "matched_tags",
        "matched_terms",
        "chapter_title",
        "section_title",
    ):
        value = hit.get(key, "")
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value))
    why = hit.get("why_this_hit") or {}
    if isinstance(why, dict):
        parts.extend(str(value) for value in why.values())
    return " ".join(parts).lower()


def concept_is_matched(concept: str, hits: list[dict[str, Any]]) -> bool:
    variants = signal_variants(concept)
    for hit in hits:
        text = hit_signal_text(hit)
        if any(variant and variant in text for variant in variants):
            return True
    return False


def chapter_is_matched(preferred_chapter: str, hit: dict[str, Any]) -> bool:
    target = normalize_signal(preferred_chapter)
    location = normalize_signal(
        " ".join(str(hit.get(key, "")) for key in ("chapter_title", "section_title"))
    )
    return target in location


def top_hit_has_expected_signal(
    scenario: Scenario,
    hits: list[dict[str, Any]],
    matched_concepts: tuple[str, ...],
) -> bool:
    if not hits:
        return False
    top_text = hit_signal_text(hits[0])
    concept_signal = any(
        variant in top_text for concept in matched_concepts for variant in signal_variants(concept)
    )
    chapter_signal = any(
        chapter_is_matched(chapter, hits[0]) for chapter in scenario.preferred_chapters
    )
    return concept_signal or chapter_signal


def label_grade(value: float) -> str:
    if value >= 4.5:
        return "strong"
    if value >= 4.0:
        return "good"
    if value >= 3.0:
        return "mixed"
    return "weak"


def grade_package(scenario: Scenario, package: dict[str, Any]) -> Grade:
    hits = list(package.get("hits", []))
    matched_concepts = tuple(
        concept for concept in scenario.expected_concepts if concept_is_matched(concept, hits)
    )
    missing_concepts = tuple(
        concept for concept in scenario.expected_concepts if concept not in matched_concepts
    )
    preferred_chapter_hits = sum(
        1
        for hit in hits
        if any(chapter_is_matched(chapter, hit) for chapter in scenario.preferred_chapters)
    )
    noisy_hits = sum(1 for hit in hits if hit.get("noise_flags"))
    explained_hits = sum(
        1
        for hit in hits
        if isinstance(hit.get("why_this_hit"), dict)
        and str(hit["why_this_hit"].get("summary", "")).strip()
    )
    top_expected = top_hit_has_expected_signal(scenario, hits, matched_concepts)

    concept_ratio = len(matched_concepts) / max(1, len(scenario.expected_concepts))
    chapter_ratio = min(1.0, preferred_chapter_hits / max(1, min(2, len(hits))))
    explanation_ratio = explained_hits / max(1, len(hits))
    noise_penalty = min(0.9, noisy_hits * 0.3)
    top_penalty = 0.35 if hits and not top_expected else 0.0
    empty_penalty = 1.0 if not hits else 0.0

    value = (
        1.0
        + (concept_ratio * 2.35)
        + (chapter_ratio * 1.1)
        + (explanation_ratio * 0.45)
        - noise_penalty
        - top_penalty
        - empty_penalty
    )
    value = round(max(1.0, min(5.0, value)), 1)

    notes: list[str] = []
    if missing_concepts:
        notes.append(f"missing concepts: {', '.join(missing_concepts)}")
    if preferred_chapter_hits == 0:
        notes.append("no preferred chapter hit")
    if noisy_hits:
        notes.append(f"{noisy_hits} noisy hit(s)")
    if hits and not top_expected:
        notes.append("top hit lacks expected signal")
    if not notes:
        notes.append("retrieval matched expected concepts and chapter direction")

    return Grade(
        value=value,
        label=label_grade(value),
        matched_concepts=matched_concepts,
        missing_concepts=missing_concepts,
        preferred_chapter_hits=preferred_chapter_hits,
        noisy_hits=noisy_hits,
        explained_hits=explained_hits,
        top_hit_has_expected_signal=top_expected,
        notes=tuple(notes),
    )


def safe_package_payload(package: dict[str, Any]) -> dict[str, Any]:
    safe = dict(package)
    safe["hits"] = []
    for hit in package.get("hits", []):
        copied = dict(hit)
        copied["text"] = "[omitted: eval reports do not quote source text]"
        safe["hits"].append(copied)
    return safe


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summarize_hit(hit: dict[str, Any]) -> str:
    tags = ", ".join(hit.get("concept_tags") or hit.get("principle_tags") or ["untagged"])
    why = hit.get("why_this_hit") or {}
    summary = why.get("summary", "not explained") if isinstance(why, dict) else "not explained"
    location = (
        " / ".join(
            part
            for part in (str(hit.get("chapter_title", "")), str(hit.get("section_title", "")))
            if part
        )
        or "not labeled"
    )
    return (
        f"{hit.get('chunk_id')} p{hit.get('page')} "
        f"score={float(hit.get('score', 0.0)):.3f} "
        f"location={location}; tags={tags}; why={summary}"
    )


def markdown_table_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("\n", " ") for value in values) + " |"


def render_report(
    *,
    scenarios: list[Scenario],
    packages: dict[str, dict[str, Any]],
    grades: dict[str, Grade],
    package_dir: Path,
) -> str:
    lines: list[str] = [
        "# IDDIA Brownfield/Greenfield Retrieval Eval",
        "",
        f"- Generated at: {datetime.now(UTC).replace(microsecond=0).isoformat()}",
        f"- Scenarios: {len(scenarios)}",
        f"- Package directory: `{package_dir}`",
        "- Source snippets: omitted from eval outputs to keep the report copyright-safe.",
        "",
        "## Summary",
        "",
    ]

    all_grades = [grades[scenario.id].value for scenario in scenarios]
    lines.extend(
        [
            f"- Average grade: {statistics.mean(all_grades):.2f}/5",
            f"- Strong or good (>=4.0): {sum(value >= 4.0 for value in all_grades)}/{len(all_grades)}",
            f"- Needs improvement (<4.0): {sum(value < 4.0 for value in all_grades)}/{len(all_grades)}",
            "",
        ]
    )

    for key, label in (("mode", "Mode"), ("complexity", "Complexity")):
        lines.append(f"### By {label}")
        lines.append("")
        grouped: dict[str, list[float]] = defaultdict(list)
        for scenario in scenarios:
            grouped[getattr(scenario, key)].append(grades[scenario.id].value)
        for group, values in sorted(grouped.items()):
            lines.append(
                f"- {group}: {statistics.mean(values):.2f}/5 across {len(values)} scenario(s)"
            )
        lines.append("")

    lines.extend(
        [
            "## Scenario Grades",
            "",
            markdown_table_row(
                [
                    "Scenario",
                    "Mode",
                    "Complexity",
                    "Stage",
                    "Grade",
                    "Matched concepts",
                    "Preferred hits",
                    "Top hit",
                ]
            ),
            "|---|---|---|---|---:|---|---:|---|",
        ]
    )
    for scenario in scenarios:
        package = packages[scenario.id]
        grade = grades[scenario.id]
        top_hit = package.get("hits", [{}])[0] if package.get("hits") else {}
        lines.append(
            markdown_table_row(
                [
                    scenario.id,
                    scenario.mode,
                    scenario.complexity,
                    scenario.stage,
                    f"{grade.value:.1f} {grade.label}",
                    ", ".join(grade.matched_concepts) or "none",
                    str(grade.preferred_chapter_hits),
                    f"{top_hit.get('page', 'n/a')} {top_hit.get('chapter_title', '')}".strip(),
                ]
            )
        )
    lines.append("")

    lines.extend(["## Per-Scenario Notes", ""])
    for scenario in scenarios:
        package = packages[scenario.id]
        grade = grades[scenario.id]
        lines.extend(
            [
                f"### {scenario.id}",
                "",
                f"- Mode/complexity/stage: {scenario.mode} / {scenario.complexity} / {scenario.stage}",
                f"- Grade: {grade.value:.1f}/5 ({grade.label})",
                f"- Task: {scenario.task}",
                f"- Next steps: {scenario.next_steps}",
                f"- Matched concepts: {', '.join(grade.matched_concepts) or 'none'}",
                f"- Missing concepts: {', '.join(grade.missing_concepts) or 'none'}",
                f"- Notes: {'; '.join(grade.notes)}",
                "- Top retrieved context:",
            ]
        )
        for hit in package.get("hits", [])[:3]:
            lines.append(f"  - {summarize_hit(hit)}")
        lines.append("")

    lines.extend(render_improvement_backlog(scenarios=scenarios, packages=packages, grades=grades))
    return "\n".join(lines).rstrip() + "\n"


def render_improvement_backlog(
    *,
    scenarios: list[Scenario],
    packages: dict[str, dict[str, Any]],
    grades: dict[str, Grade],
) -> list[str]:
    missing_counter: Counter[str] = Counter()
    weak_scenarios: list[str] = []
    noisy_counter: Counter[str] = Counter()
    chapter_misses: Counter[str] = Counter()

    for scenario in scenarios:
        grade = grades[scenario.id]
        missing_counter.update(grade.missing_concepts)
        if grade.value < 4.0:
            weak_scenarios.append(scenario.id)
        for hit in packages[scenario.id].get("hits", []):
            noisy_counter.update(hit.get("noise_flags") or [])
        if grade.preferred_chapter_hits == 0:
            chapter_misses.update(scenario.preferred_chapters)

    lines = ["## Improvement Backlog", ""]
    if weak_scenarios:
        lines.append("- Raise below-4 scenarios first: " + ", ".join(weak_scenarios))
    else:
        lines.append("- No scenario is below 4.0; keep broadening regression coverage.")

    if missing_counter:
        lines.append(
            "- Add or tune retrieval concepts for missing signals: "
            + ", ".join(f"{name} ({count})" for name, count in missing_counter.most_common())
        )
    if chapter_misses:
        lines.append(
            "- Add chapter-affinity tests for missed chapter targets: "
            + ", ".join(f"{name} ({count})" for name, count in chapter_misses.most_common())
        )
    if noisy_counter:
        lines.append(
            "- Tighten noise filtering for: "
            + ", ".join(f"{name} ({count})" for name, count in noisy_counter.most_common())
        )
    if not missing_counter and not chapter_misses and not noisy_counter:
        lines.append(
            "- Main next step: add harder adversarial prompts and assert stable top-hit ordering."
        )
    lines.append("")
    return lines


def run_eval(
    *,
    scenarios_path: Path = DEFAULT_SCENARIOS_PATH,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    top_k: int = 5,
) -> tuple[Path, Path, dict[str, Grade]]:
    scenarios = load_scenarios(scenarios_path)
    run_dir = output_root / utc_stamp()
    package_dir = run_dir / "packages"
    packages: dict[str, dict[str, Any]] = {}
    grades: dict[str, Grade] = {}

    for scenario in scenarios:
        rendered = build_context_package(
            task=scenario.task,
            stage=scenario.stage,
            next_steps=scenario.next_steps,
            artifact_root=artifact_root,
            top_k=top_k,
            output_format="json",
            max_snippet_chars=1,
        )
        package = safe_package_payload(json.loads(rendered))
        packages[scenario.id] = package
        grades[scenario.id] = grade_package(scenario, package)
        write_json(package_dir / f"{scenario.id}.json", package)

    report = render_report(
        scenarios=scenarios,
        packages=packages,
        grades=grades,
        package_dir=package_dir,
    )
    report_path = run_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    latest_report_path = output_root / "latest_report.md"
    latest_report_path.parent.mkdir(parents=True, exist_ok=True)
    latest_report_path.write_text(report, encoding="utf-8")
    return report_path, latest_report_path, grades


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run brownfield/greenfield retrieval scenarios against IDDIA."
    )
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_path, latest_report_path, grades = run_eval(
        scenarios_path=args.scenarios,
        artifact_root=args.artifact_root,
        output_root=args.output_root,
        top_k=args.top_k,
    )
    values = [grade.value for grade in grades.values()]
    print(f"Scenarios: {len(values)}")
    print(f"Average grade: {statistics.mean(values):.2f}/5")
    print(f"Good or strong: {sum(value >= 4.0 for value in values)}/{len(values)}")
    print(f"Report: {report_path}")
    print(f"Latest report: {latest_report_path}")
    for scenario_id, grade in grades.items():
        print(f"{scenario_id}: {grade.value:.1f}/5 {grade.label} - {'; '.join(grade.notes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
