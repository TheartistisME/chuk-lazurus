#!/usr/bin/env python3
"""AUS3000 benchmark evaluator and report generator.

Report mode reproduces the current repeated benchmark loop for comparison.
Strict gate modes use the clause-aligned gold set fixture and exact
window_metadata.json route scoring.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chuk_lazarus.cli.commands.context.generate._torch import _validate_torch_checkpoint
from chuk_lazarus.inference.context.knowledge.torch_query import (
    _load_torch_runtime,
    _prepare_store_response,
)
from chuk_lazarus.inference.context.knowledge.torch_store import TorchKnowledgeStore


DEFAULT_MODEL = Path(
    "/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/"
    "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
)
DEFAULT_CHECKPOINT = Path(
    "/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/"
    "AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant"
)
DEFAULT_CORPUS = Path(
    "/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/"
    "AS_NZS_3000-2018/as_nzs_3000_2018_lazarus_corpus.txt"
)
DEFAULT_OUTPUT = Path(
    "/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/"
    "AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_validation_report.txt"
)
DEFAULT_BENCHMARK_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "aus3000" / "benchmark" / "epic1_v1.json"

STRICT_MODES = (
    "store_evidence_gate",
    "route_gate",
    "grounding_gate",
    "ood_gate",
    "no_regression_gate",
    "single_pass_gate",
    "soak_gate",
)

STRICT_GATE_ORDER = ("store_evidence_gate", "single_pass_gate", "soak_gate")

INSUFFICIENCY_PATTERNS = (
    "insufficient",
    "not enough",
    "does not contain",
    "doesn't contain",
    "cannot answer",
    "can't answer",
    "not provided",
    "not specified",
    "not in the retrieved",
    "outside the retrieved",
    "outside the store",
    "outside the knowledge store",
    "the context does not",
    "the store context does not",
    "the retrieved context does not",
)

ELECTRICAL_BLEED_KEYWORDS = {
    "electrical",
    "installation",
    "installations",
    "switchboard",
    "switchboards",
    "rcd",
    "earthing",
    "earth",
    "fault",
    "socket",
    "circuit",
    "subcircuit",
    "men",
    "clause",
    "standard",
    "voltage",
    "conductors",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "under",
    "what",
    "with",
}


@dataclass(frozen=True)
class CorpusRecord:
    clause_id: str
    clause_title: str
    clause_content: str
    standard_id: str
    standard_title: str


@dataclass(frozen=True)
class EvalCase:
    name: str
    category: str
    prompt: str
    primary_clause_ids: tuple[str, ...] = ()
    support_clause_ids: tuple[str, ...] = ()
    required_answer_anchors: tuple[str, ...] = ()
    required_literal_groups: tuple[tuple[str, ...], ...] = ()
    expect_insufficient: bool = False
    expect_no_electrical_bleed: bool = False

    @property
    def expected_clause_ids(self) -> tuple[str, ...]:
        return self.primary_clause_ids + self.support_clause_ids


@dataclass(frozen=True)
class CaseEvaluation:
    case: EvalCase
    verdict: str
    route_pass: bool
    grounding_pass: bool
    ood_pass: bool
    effective_top_k: int
    routed_clause_ids: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class GateOutcome:
    mode: str
    executed: int
    pass_count: int
    fail_count: int
    review_count: int = 0
    route_pass_count: int = 0
    grounding_pass_count: int = 0
    ood_pass_count: int = 0
    regression_pass_count: int = 0
    overall_pass: bool = False
    details: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluator for the clause-aligned AUS3000 Gemma checkpoint."
    )
    parser.add_argument("--mode", choices=("report", *STRICT_MODES), default="report")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Base Gemma model path")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Torch sidecar checkpoint directory",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Plain-text corpus file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Plain-text report file")
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=30.0,
        help="How long to keep cycling the prompt suite in report/soak mode",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on total executed cases",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
        help="Maximum generated tokens per answer",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Report-mode store routing top-k")
    parser.add_argument(
        "--clear-cache-every",
        type=int,
        default=1,
        help="Call runtime.clear_cache() every N prompts while keeping weights loaded",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable tokenizer chat template when querying",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for runtime loading (default: cuda)",
    )
    parser.add_argument(
        "--benchmark-fixture",
        type=Path,
        default=DEFAULT_BENCHMARK_FIXTURE,
        help="Machine-readable Epic 1 gold-set fixture",
    )
    return parser


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9%/+. ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_literal_text(text: str) -> str:
    """Normalize literal phrases while preserving unit equivalences."""
    text = text.replace("Ω", "Ω")
    text = re.sub(r"(?i)\bM\s*Ω\b", "Mohm", text)
    text = re.sub(r"(?i)\bM\s*ohm\b", "Mohm", text)
    text = re.sub(r"(?i)\bmega\s+ohm\b", "megaohm", text)
    return normalize_text(text)


def significant_terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[a-z0-9%/+.]+", normalize_text(text)):
        if len(token) < 3 or token in STOPWORDS:
            continue
        terms.add(token)
    return terms


def excerpt(text: str, width: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= width:
        return compact
    return compact[: width - 3].rstrip() + "..."


def parse_corpus(path: Path) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    current: dict[str, str] = {}
    content_lines: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("clause_title: "):
            if current:
                current["clause_content"] = "\n".join(content_lines).strip()
                records.append(
                    CorpusRecord(
                        clause_id=current.get("clause_id", ""),
                        clause_title=current.get("clause_title", ""),
                        clause_content=current.get("clause_content", ""),
                        standard_id=current.get("standard_id", ""),
                        standard_title=current.get("standard_title", ""),
                    )
                )
            current = {"clause_title": raw_line.split(": ", 1)[1]}
            content_lines = []
            continue

        if raw_line.startswith("standard_id: "):
            current["standard_id"] = raw_line.split(": ", 1)[1]
            continue
        if raw_line.startswith("standard_title: "):
            current["standard_title"] = raw_line.split(": ", 1)[1]
            continue
        if raw_line.startswith("clause_id: "):
            current["clause_id"] = raw_line.split(": ", 1)[1]
            continue
        if raw_line.startswith("clause_content: "):
            content_lines = [raw_line.split(": ", 1)[1]]
            continue
        content_lines.append(raw_line)

    if current:
        current["clause_content"] = "\n".join(content_lines).strip()
        records.append(
            CorpusRecord(
                clause_id=current.get("clause_id", ""),
                clause_title=current.get("clause_title", ""),
                clause_content=current.get("clause_content", ""),
                standard_id=current.get("standard_id", ""),
                standard_title=current.get("standard_title", ""),
            )
        )

    return records


def load_benchmark_fixture(path: Path = DEFAULT_BENCHMARK_FIXTURE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_case_suite(path: Path = DEFAULT_BENCHMARK_FIXTURE) -> list[EvalCase]:
    fixture = load_benchmark_fixture(path)
    cases: list[EvalCase] = []
    for raw_case in fixture["cases"]:
        literal_groups = tuple(tuple(group) for group in raw_case.get("required_literal_groups", ()))
        cases.append(
            EvalCase(
                name=raw_case["name"],
                category=raw_case["category"],
                prompt=raw_case["prompt"],
                primary_clause_ids=tuple(raw_case.get("primary_clause_ids", ())),
                support_clause_ids=tuple(raw_case.get("support_clause_ids", ())),
                required_answer_anchors=tuple(raw_case.get("required_answer_anchors", ())),
                required_literal_groups=literal_groups,
                expect_insufficient=bool(raw_case.get("expect_insufficient", False)),
                expect_no_electrical_bleed=bool(raw_case.get("expect_no_electrical_bleed", False)),
            )
        )
    return cases


def case_by_name(cases: list[EvalCase]) -> dict[str, EvalCase]:
    return {case.name: case for case in cases}


def unique_primary_clause_ids(cases: list[EvalCase]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for case in cases:
        for clause_id in case.primary_clause_ids:
            if clause_id not in seen:
                seen.add(clause_id)
                ordered.append(clause_id)
    return tuple(ordered)


def case_effective_top_k(case: EvalCase) -> int:
    if case.expect_insufficient:
        return 1
    if case.category == "comparison":
        return len(case.primary_clause_ids)
    return 1


def expected_records_for_case(
    case: EvalCase,
    records_by_clause_id: dict[str, CorpusRecord],
) -> list[CorpusRecord]:
    clause_ids = case.expected_clause_ids
    if not clause_ids:
        clause_ids = case.primary_clause_ids
    return [records_by_clause_id[clause_id] for clause_id in clause_ids if clause_id in records_by_clause_id]


def load_window_metadata(store_path: Path) -> dict[int, dict[str, Any]]:
    metadata_path = store_path / "window_metadata.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {int(window_id): window_metadata for window_id, window_metadata in data.items()}


def window_clause_id(window_id: int, window_metadata: dict[int, dict[str, Any]]) -> str:
    return str(window_metadata.get(window_id, {}).get("clause_id", ""))


def routed_clause_ids(
    window_ids: list[int],
    window_metadata: dict[int, dict[str, Any]],
    *,
    top_k: int,
) -> tuple[str, ...]:
    return tuple(window_clause_id(window_id, window_metadata) for window_id in window_ids[:top_k])


def insufficiency_detected(answer: str) -> bool:
    normalized = normalize_text(answer)
    return any(normalize_text(pattern) in normalized for pattern in INSUFFICIENCY_PATTERNS)


def electrical_bleed_detected(answer: str) -> bool:
    normalized = normalize_text(answer)
    return any(keyword in normalized.split() for keyword in ELECTRICAL_BLEED_KEYWORDS)


def keyword_hits(answer: str, expected_keywords: tuple[str, ...]) -> list[str]:
    normalized_answer = normalize_text(answer)
    hits: list[str] = []
    for keyword in expected_keywords:
        if normalize_text(keyword) in normalized_answer:
            hits.append(keyword)
    return hits


def overlap_ratio(answer: str, source_text: str) -> float:
    answer_terms = significant_terms(answer)
    if not answer_terms:
        return 0.0
    source_terms = significant_terms(source_text)
    if not source_terms:
        return 0.0
    return len(answer_terms & source_terms) / len(answer_terms)


def _literal_groups_present(answer: str, literal_groups: tuple[tuple[str, ...], ...]) -> bool:
    normalized_answer = normalize_literal_text(answer)
    for group in literal_groups:
        if not any(normalize_literal_text(literal) in normalized_answer for literal in group):
            return False
    return True


def _route_expected_clause_ids(case: EvalCase) -> tuple[str, ...]:
    if case.category == "comparison":
        return case.primary_clause_ids
    if case.primary_clause_ids:
        return (case.primary_clause_ids[0],)
    return ()


def _route_case_pass(case: EvalCase, routed_ids: tuple[str, ...]) -> bool:
    expected = _route_expected_clause_ids(case)
    if not expected:
        return True
    return routed_ids[: len(expected)] == expected


def _score_in_domain_case_report(
    case: EvalCase,
    answer: str,
    expected_records: list[CorpusRecord],
    routed_ids: tuple[str, ...],
) -> tuple[str, list[str], bool, bool]:
    notes: list[str] = []
    route_pass = _route_case_pass(case, routed_ids)
    insufficient = insufficiency_detected(answer)
    if insufficient:
        notes.append("Model claimed the retrieved store context was insufficient.")

    source_text = "\n\n".join(
        f"{record.clause_id} {record.clause_title}\n{record.clause_content}" for record in expected_records
    )
    hits = keyword_hits(answer, case.required_answer_anchors)
    ratio = len(hits) / len(case.required_answer_anchors) if case.required_answer_anchors else 0.0
    overlap = overlap_ratio(answer, source_text)
    literal_ok = _literal_groups_present(answer, case.required_literal_groups)

    notes.append(f"Route clause ids: {', '.join(routed_ids) if routed_ids else 'none'}")
    notes.append(f"Route exact match: {'yes' if route_pass else 'no'}")
    notes.append(f"Anchor hits: {len(hits)}/{len(case.required_answer_anchors)}")
    if hits:
        notes.append("Matched anchors: " + ", ".join(hits))
    notes.append(f"Answer/source overlap ratio: {overlap:.2f}")
    notes.append(f"Required literal groups present: {'yes' if literal_ok else 'no'}")

    if route_pass and not insufficient and ratio >= 0.75 and literal_ok:
        return "PASS", notes, route_pass, True

    if route_pass and not insufficient and (ratio >= 0.5 or overlap >= 0.30):
        notes.append("Answer is partially grounded but not strict-pass clean.")
        return "REVIEW", notes, route_pass, False

    notes.append("Answer did not satisfy the report-mode grounding contract.")
    return "FAIL", notes, route_pass, False


def _score_in_domain_case_strict(
    case: EvalCase,
    answer: str,
    expected_records: list[CorpusRecord],
    routed_ids: tuple[str, ...],
) -> tuple[bool, bool, list[str]]:
    notes: list[str] = []
    route_pass = _route_case_pass(case, routed_ids)
    insufficient = insufficiency_detected(answer)
    source_text = "\n\n".join(
        f"{record.clause_id} {record.clause_title}\n{record.clause_content}" for record in expected_records
    )
    hits = keyword_hits(answer, case.required_answer_anchors)
    ratio = len(hits) / len(case.required_answer_anchors) if case.required_answer_anchors else 0.0
    overlap = overlap_ratio(answer, source_text)
    literal_ok = _literal_groups_present(answer, case.required_literal_groups)

    notes.append(f"Route exact match: {'yes' if route_pass else 'no'}")
    notes.append(f"Anchor hits: {len(hits)}/{len(case.required_answer_anchors)}")
    notes.append(f"Answer/source overlap ratio: {overlap:.2f}")
    notes.append(f"Required literal groups present: {'yes' if literal_ok else 'no'}")

    grounding_pass = route_pass and not insufficient and ratio >= 0.75 and literal_ok
    if not grounding_pass:
        if insufficient:
            notes.append("Strict mode rejected the answer because it claimed insufficiency.")
        if not route_pass:
            notes.append("Strict mode rejected the answer because exact route evidence was missing.")
        if ratio < 0.75:
            notes.append("Strict mode rejected the answer because too few anchors were present.")
        if not literal_ok:
            notes.append("Strict mode rejected the answer because a required literal group was missing.")

    return route_pass, grounding_pass, notes


def _score_ood_case_report(case: EvalCase, answer: str) -> tuple[str, list[str], bool]:
    notes: list[str] = []
    insufficient = insufficiency_detected(answer)
    electrical_bleed = electrical_bleed_detected(answer)

    if insufficient and not electrical_bleed:
        notes.append("Out-of-domain refusal detected.")
        return "PASS", notes, True

    if insufficient and electrical_bleed:
        notes.append("Refusal text included electrical bleedthrough and does not pass.")
        return "FAIL", notes, False

    if electrical_bleed:
        notes.append("Unexpected electrical/domain bleedthrough appeared in an out-of-domain answer.")
    else:
        notes.append("Model did not clearly state that the store context was insufficient.")
    return "FAIL", notes, False


def _score_ood_case_strict(case: EvalCase, answer: str) -> tuple[bool, list[str]]:
    notes: list[str] = []
    insufficient = insufficiency_detected(answer)
    electrical_bleed = electrical_bleed_detected(answer)

    if insufficient and not electrical_bleed:
        notes.append("Out-of-domain refusal detected.")
        return True, notes

    if insufficient and electrical_bleed:
        notes.append("Strict mode rejected refusal text because electrical bleed was present.")
        return False, notes

    if electrical_bleed:
        notes.append("Strict mode rejected answer because electrical bleed was present.")
    else:
        notes.append("Strict mode rejected answer because it did not clearly refuse.")
    return False, notes


def evaluate_report_case(
    case: EvalCase,
    answer: str,
    expected_records: list[CorpusRecord],
    routed_ids: tuple[str, ...],
) -> CaseEvaluation:
    if case.expect_insufficient:
        verdict, notes, ood_pass = _score_ood_case_report(case, answer)
        return CaseEvaluation(
            case=case,
            verdict=verdict,
            route_pass=False,
            grounding_pass=ood_pass,
            ood_pass=ood_pass,
            effective_top_k=0,
            routed_clause_ids=routed_ids,
            notes=tuple(notes),
        )

    verdict, notes, route_pass, grounding_pass = _score_in_domain_case_report(
        case, answer, expected_records, routed_ids
    )
    return CaseEvaluation(
        case=case,
        verdict=verdict,
        route_pass=route_pass,
        grounding_pass=grounding_pass,
        ood_pass=False,
        effective_top_k=case_effective_top_k(case),
        routed_clause_ids=routed_ids,
        notes=tuple(notes),
    )


def evaluate_strict_case(
    case: EvalCase,
    answer: str,
    expected_records: list[CorpusRecord],
    routed_ids: tuple[str, ...],
) -> CaseEvaluation:
    if case.expect_insufficient:
        ood_pass, notes = _score_ood_case_strict(case, answer)
        verdict = "PASS" if ood_pass else "FAIL"
        return CaseEvaluation(
            case=case,
            verdict=verdict,
            route_pass=True,
            grounding_pass=ood_pass,
            ood_pass=ood_pass,
            effective_top_k=0,
            routed_clause_ids=routed_ids,
            notes=tuple(notes),
        )

    route_pass, grounding_pass, notes = _score_in_domain_case_strict(case, answer, expected_records, routed_ids)
    verdict = "PASS" if (route_pass and grounding_pass) else "FAIL"
    return CaseEvaluation(
        case=case,
        verdict=verdict,
        route_pass=route_pass,
        grounding_pass=grounding_pass,
        ood_pass=False,
        effective_top_k=case_effective_top_k(case),
        routed_clause_ids=routed_ids,
        notes=tuple(notes),
    )


def write_header(
    handle,
    *,
    args,
    metadata: dict[str, Any],
    store_path: Path,
    records: list[CorpusRecord],
    cases: list[EvalCase],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    handle.write("AUS3000 Lazarus Variant Validation Report\n")
    handle.write("=" * 80 + "\n\n")
    handle.write(f"Started (UTC): {now}\n")
    handle.write(f"Mode: {args.mode}\n")
    handle.write(f"Model: {args.model}\n")
    handle.write(f"Checkpoint: {args.checkpoint}\n")
    handle.write(f"Store: {store_path}\n")
    handle.write(f"Corpus: {args.corpus}\n")
    handle.write(f"Output: {args.output}\n")
    handle.write(f"Benchmark fixture: {args.benchmark_fixture}\n")
    handle.write(f"Duration target (minutes): {args.duration_minutes}\n")
    handle.write(f"Max cases: {args.max_cases if args.max_cases is not None else 'unbounded'}\n")
    handle.write(f"Top-k: {args.top_k}\n")
    handle.write(f"Max new tokens: {args.max_new_tokens}\n")
    handle.write(f"Temperature: {args.temperature}\n")
    handle.write(f"No chat template: {args.no_chat_template}\n")
    handle.write(f"Corpus records parsed: {len(records)}\n")
    handle.write(f"Prompt suite size: {len(cases)}\n")
    handle.write(f"Checkpoint window count: {metadata.get('windowing', {}).get('num_windows', 'unknown')}\n")
    handle.write(f"Checkpoint token count: {metadata.get('windowing', {}).get('num_tokens', 'unknown')}\n")
    handle.write(f"Strict gate order: {' -> '.join(STRICT_GATE_ORDER)}\n")
    handle.write("\nMethod:\n")
    handle.write("- Report mode keeps the repeated prompt loop for baseline comparison.\n")
    handle.write("- Strict gate modes use exact clause metadata from window_metadata.json.\n")
    handle.write("- Strict counted modes use PASS/FAIL only.\n")
    handle.write("\n")
    handle.flush()


def _write_case_block(
    handle,
    *,
    case_number: int,
    round_number: int,
    case: EvalCase,
    evaluation: CaseEvaluation,
    response,
    elapsed_minutes: float,
    window_texts: list[str],
    expected_records: list[CorpusRecord],
) -> None:
    handle.write("-" * 80 + "\n")
    handle.write(f"Case #: {case_number}\n")
    handle.write(f"Round: {round_number}\n")
    handle.write(f"Case Name: {case.name}\n")
    handle.write(f"Category: {case.category}\n")
    handle.write(f"Prompt: {case.prompt}\n")
    handle.write(f"Verdict: {evaluation.verdict}\n")
    handle.write(f"Response Mode: {response.mode}\n")
    handle.write(f"Routing Mode: {response.routing_mode}\n")
    handle.write(f"Effective Top-k: {evaluation.effective_top_k}\n")
    handle.write(f"Routed Windows: {response.window_ids}\n")
    handle.write(f"Routed Clause IDs: {list(evaluation.routed_clause_ids)}\n")
    handle.write(f"Expansion Terms: {', '.join(response.expansion_terms[:15])}\n")
    handle.write(f"Residual Status: {response.residual_reason}\n")
    handle.write(f"Boundary Shape: {response.boundary_shape}\n")
    handle.write(f"Stats: {response.stats_summary}\n")
    handle.write(f"Elapsed Minutes: {elapsed_minutes:.2f}\n")
    handle.write("\nAnswer:\n")
    handle.write(response.text.rstrip() + "\n")

    if expected_records:
        handle.write("\nExpected Source Records:\n")
        for record in expected_records:
            handle.write(
                f"* {record.clause_id} | {record.clause_title} | "
                f"{excerpt(record.clause_content, width=320)}\n"
            )

    handle.write("\nRouted Window Previews:\n")
    if window_texts:
        for window_id, text in zip(response.window_ids[: len(window_texts)], window_texts, strict=False):
            handle.write(f"* window {window_id}: {excerpt(text, width=320)}\n")
    else:
        handle.write("* none\n")

    handle.write("\nValidation Notes:\n")
    for note in evaluation.notes:
        handle.write(f"* {note}\n")
    handle.write("\n")
    handle.flush()


def _write_gate_summary(handle, outcome: GateOutcome) -> None:
    handle.write("=" * 80 + "\n")
    handle.write("Summary\n")
    handle.write("=" * 80 + "\n")
    handle.write(f"Mode: {outcome.mode}\n")
    handle.write(f"Cases executed: {outcome.executed}\n")
    handle.write(f"PASS: {outcome.pass_count}\n")
    handle.write(f"FAIL: {outcome.fail_count}\n")
    if outcome.review_count:
        handle.write(f"REVIEW: {outcome.review_count}\n")
    if outcome.route_pass_count:
        handle.write(f"Route PASS: {outcome.route_pass_count}\n")
    if outcome.grounding_pass_count:
        handle.write(f"Grounding PASS: {outcome.grounding_pass_count}\n")
    if outcome.ood_pass_count:
        handle.write(f"OOD PASS: {outcome.ood_pass_count}\n")
    if outcome.regression_pass_count:
        handle.write(f"No regression PASS: {outcome.regression_pass_count}\n")
    handle.write(f"Overall PASS: {'yes' if outcome.overall_pass else 'no'}\n")
    for detail in outcome.details:
        handle.write(f"* {detail}\n")
    handle.flush()


def run_store_evidence_gate(
    fixture: dict[str, Any],
    window_metadata: dict[int, dict[str, Any]],
) -> GateOutcome:
    expected_ids = set(fixture["unique_primary_clause_ids"])
    present_ids = {str(record.get("clause_id", "")) for record in window_metadata.values()}
    missing = sorted(expected_ids - present_ids)
    extra = sorted(present_ids - expected_ids)
    details = (
        f"Expected primary clause ids: {len(expected_ids)}",
        f"Present primary clause ids: {len(expected_ids) - len(missing)}",
        f"Missing primary clause ids: {', '.join(missing) if missing else 'none'}",
        f"Extra primary clause ids: {', '.join(extra) if extra else 'none'}",
    )
    overall_pass = not missing
    return GateOutcome(
        mode="store_evidence_gate",
        executed=1,
        pass_count=1 if overall_pass else 0,
        fail_count=0 if overall_pass else 1,
        overall_pass=overall_pass,
        details=details,
    )


def _select_cases_for_mode(mode: str, cases: list[EvalCase], fixture: dict[str, Any]) -> list[EvalCase]:
    by_name = case_by_name(cases)
    if mode == "route_gate" or mode == "grounding_gate":
        return [case for case in cases if not case.expect_insufficient]
    if mode == "ood_gate":
        return [case for case in cases if case.expect_insufficient]
    if mode == "no_regression_gate":
        return [by_name[name] for name in fixture["no_regression_case_ids"]]
    return cases


def _is_strict_mode(mode: str) -> bool:
    return mode in STRICT_MODES


def _strict_gate_pass(mode: str, evaluation: CaseEvaluation) -> bool:
    if mode == "route_gate":
        return evaluation.route_pass
    if mode == "grounding_gate":
        return evaluation.grounding_pass
    if mode == "ood_gate":
        return evaluation.ood_pass
    return evaluation.verdict == "PASS"


def _write_case_result(
    handle,
    *,
    case_number: int,
    round_number: int,
    case: EvalCase,
    evaluation: CaseEvaluation,
    response,
    elapsed_minutes: float,
    window_texts: list[str],
    expected_records: list[CorpusRecord],
) -> None:
    _write_case_block(
        handle,
        case_number=case_number,
        round_number=round_number,
        case=case,
        evaluation=evaluation,
        response=response,
        elapsed_minutes=elapsed_minutes,
        window_texts=window_texts,
        expected_records=expected_records,
    )


def _run_report_mode(
    *,
    args,
    metadata: dict[str, Any],
    store_path: Path,
    records: list[CorpusRecord],
    cases: list[EvalCase],
) -> GateOutcome:
    records_by_clause_id = {record.clause_id: record for record in records}
    window_metadata = load_window_metadata(store_path)
    runtime, tokenizer = _load_torch_runtime(str(args.model), args.device)
    store = TorchKnowledgeStore.load(store_path)

    started = time.monotonic()
    deadline = started + max(args.duration_minutes, 0.0) * 60.0
    executed = 0
    pass_count = 0
    review_count = 0
    fail_count = 0

    with args.output.open("w", encoding="utf-8") as handle:
        write_header(handle, args=args, metadata=metadata, store_path=store_path, records=records, cases=cases)

        try:
            for index, case in enumerate(itertools.cycle(cases), start=1):
                if args.max_cases is not None and executed >= args.max_cases:
                    break
                if executed > 0 and time.monotonic() >= deadline:
                    break

                round_number = ((index - 1) // len(cases)) + 1
                expected_records = expected_records_for_case(case, records_by_clause_id)
                response = _prepare_store_response(
                    store=store,
                    runtime=runtime,
                    tokenizer=tokenizer,
                    question=case.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    no_chat_template=args.no_chat_template,
                )
                routed_ids = routed_clause_ids(response.window_ids, window_metadata, top_k=args.top_k)
                evaluation = evaluate_report_case(case, response.text, expected_records, routed_ids)

                executed += 1
                if evaluation.verdict == "PASS":
                    pass_count += 1
                elif evaluation.verdict == "REVIEW":
                    review_count += 1
                else:
                    fail_count += 1

                elapsed_minutes = (time.monotonic() - started) / 60.0
                print(
                    f"[{executed}] round={round_number} case={case.name} verdict={evaluation.verdict} "
                    f"elapsed={elapsed_minutes:.1f}m",
                    file=sys.stderr,
                    flush=True,
                )

                window_texts = [
                    store.get_window_text(window_id, tokenizer) for window_id in response.window_ids[: args.top_k]
                ]
                _write_case_result(
                    handle,
                    case_number=executed,
                    round_number=round_number,
                    case=case,
                    evaluation=evaluation,
                    response=response,
                    elapsed_minutes=elapsed_minutes,
                    window_texts=window_texts,
                    expected_records=expected_records,
                )

                if args.clear_cache_every > 0 and executed % args.clear_cache_every == 0:
                    runtime.clear_cache()
        finally:
            runtime.clear_cache()

        completed_at = datetime.now(timezone.utc).isoformat()
        actual_minutes = (time.monotonic() - started) / 60.0
        summary = GateOutcome(
            mode="report",
            executed=executed,
            pass_count=pass_count,
            fail_count=fail_count,
            review_count=review_count,
            overall_pass=fail_count == 0,
            details=(
                f"Completed (UTC): {completed_at}",
                f"Actual runtime minutes: {actual_minutes:.2f}",
                f"Pass rate: {pass_count / executed:.2%}" if executed else "Pass rate: n/a",
            ),
        )
        _write_gate_summary(handle, summary)

    print(f"Report written to {args.output}", file=sys.stderr)
    print(
        f"Summary: executed={executed} pass={pass_count} review={review_count} fail={fail_count}",
        file=sys.stderr,
    )
    return summary


def _run_strict_case(
    *,
    case: EvalCase,
    response,
    expected_records: list[CorpusRecord],
    window_metadata: dict[int, dict[str, Any]],
) -> CaseEvaluation:
    top_k = case_effective_top_k(case)
    routed_ids = routed_clause_ids(response.window_ids, window_metadata, top_k=top_k)
    return evaluate_strict_case(case, response.text, expected_records, routed_ids)


def _run_strict_mode(
    *,
    args,
    metadata: dict[str, Any],
    store_path: Path,
    records: list[CorpusRecord],
    cases: list[EvalCase],
) -> GateOutcome:
    fixture = load_benchmark_fixture(args.benchmark_fixture)
    selected_cases = _select_cases_for_mode(args.mode, cases, fixture)
    window_metadata = load_window_metadata(store_path)
    records_by_clause_id = {record.clause_id: record for record in records}

    if args.mode == "store_evidence_gate":
        outcome = run_store_evidence_gate(fixture, window_metadata)
        with args.output.open("w", encoding="utf-8") as handle:
            write_header(handle, args=args, metadata=metadata, store_path=store_path, records=records, cases=cases)
            _write_gate_summary(handle, outcome)
        print(f"Summary: mode={args.mode} pass={outcome.pass_count} fail={outcome.fail_count}", file=sys.stderr)
        return outcome

    started = time.monotonic()
    deadline = started + max(args.duration_minutes, 0.0) * 60.0
    executed = 0
    pass_count = 0
    fail_count = 0
    route_pass_count = 0
    grounding_pass_count = 0
    ood_pass_count = 0
    regression_pass_count = 0

    with args.output.open("w", encoding="utf-8") as handle:
        write_header(handle, args=args, metadata=metadata, store_path=store_path, records=records, cases=cases)
        handle.write("Startup: loading torch runtime and knowledge store\n\n")
        handle.flush()

        runtime, tokenizer = _load_torch_runtime(str(args.model), args.device)
        store = TorchKnowledgeStore.load(store_path)

        try:
            if args.mode == "soak_gate":
                case_iterable = enumerate(itertools.cycle(selected_cases), start=1)
            else:
                case_iterable = enumerate(selected_cases, start=1)

            for index, case in case_iterable:
                if args.max_cases is not None and executed >= args.max_cases:
                    break
                if args.mode == "soak_gate" and executed > 0 and time.monotonic() >= deadline:
                    break

                round_number = ((index - 1) // len(selected_cases)) + 1
                expected_records = expected_records_for_case(case, records_by_clause_id)
                query_top_k = case_effective_top_k(case)
                response = _prepare_store_response(
                    store=store,
                    runtime=runtime,
                    tokenizer=tokenizer,
                    question=case.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=query_top_k,
                    no_chat_template=args.no_chat_template,
                )
                evaluation = _run_strict_case(
                    case=case,
                    response=response,
                    expected_records=expected_records,
                    window_metadata=window_metadata,
                )

                executed += 1
                gate_pass = _strict_gate_pass(args.mode, evaluation)
                if gate_pass:
                    pass_count += 1
                else:
                    fail_count += 1

                if case.expect_insufficient and evaluation.ood_pass:
                    ood_pass_count += 1
                if not case.expect_insufficient and evaluation.route_pass:
                    route_pass_count += 1
                if not case.expect_insufficient and evaluation.grounding_pass:
                    grounding_pass_count += 1
                if case.name in fixture["no_regression_case_ids"] and evaluation.verdict == "PASS":
                    regression_pass_count += 1

                elapsed_minutes = (time.monotonic() - started) / 60.0
                print(
                    f"[{executed}] round={round_number} case={case.name} verdict={evaluation.verdict} "
                    f"elapsed={elapsed_minutes:.1f}m",
                    file=sys.stderr,
                    flush=True,
                )

                window_texts = [
                    store.get_window_text(window_id, tokenizer) for window_id in response.window_ids[: query_top_k]
                ]
                _write_case_result(
                    handle,
                    case_number=executed,
                    round_number=round_number,
                    case=case,
                    evaluation=evaluation,
                    response=response,
                    elapsed_minutes=elapsed_minutes,
                    window_texts=window_texts,
                    expected_records=expected_records,
                )

                if args.clear_cache_every > 0 and executed % args.clear_cache_every == 0:
                    runtime.clear_cache()
        finally:
            runtime.clear_cache()

        if args.mode == "soak_gate":
            overall_pass = executed > 0 and pass_count == executed
        else:
            overall_pass = executed == len(selected_cases) and pass_count == len(selected_cases)

        summary = GateOutcome(
            mode=args.mode,
            executed=executed,
            pass_count=pass_count,
            fail_count=fail_count,
            route_pass_count=route_pass_count,
            grounding_pass_count=grounding_pass_count,
            ood_pass_count=ood_pass_count,
            regression_pass_count=regression_pass_count,
            overall_pass=overall_pass,
            details=(
                f"Route gate cases: {route_pass_count}/{len(selected_cases) if selected_cases else 0}",
                f"Grounding gate cases: {grounding_pass_count}/{len(selected_cases) if selected_cases else 0}",
                f"OOD gate cases: {ood_pass_count}/{len(selected_cases) if selected_cases else 0}",
                f"No regression cases: {regression_pass_count}/{len(fixture['no_regression_case_ids'])}",
            ),
        )
        _write_gate_summary(handle, summary)

    print(
        f"Summary: mode={args.mode} executed={executed} pass={pass_count} fail={fail_count}",
        file=sys.stderr,
    )
    return summary


def main() -> int:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata, store_path = _validate_torch_checkpoint(args.checkpoint)
    records = parse_corpus(args.corpus)
    cases = build_case_suite(args.benchmark_fixture)

    if args.mode == "report":
        outcome = _run_report_mode(args=args, metadata=metadata, store_path=store_path, records=records, cases=cases)
        return 0 if outcome.overall_pass else 1

    outcome = _run_strict_mode(args=args, metadata=metadata, store_path=store_path, records=records, cases=cases)
    return 0 if outcome.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
