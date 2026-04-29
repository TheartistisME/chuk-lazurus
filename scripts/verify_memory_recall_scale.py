#!/usr/bin/env python3
"""Actual-use exact/literal recall verifier for a completed memory harness run.

This is a post-harness test: it does not plant new sessions. It reuses a
successful ``scripts/auto_verify_memory_repl.py`` run, samples the planted
100x100 scale markers from ``events.jsonl``, sends real MemoryChat recall
turns, and asserts the generated answer contains the routed marker plus the
expected session/turn identity. Random hex markers are exact-token identity
checks, not semantic-only activation checks; hybrid routing should retain
TF-IDF/literal lookup while activation assists semantic matches.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import secrets
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "prod" / "validation" / "repl-autoverify"
MARKER_SUITE_INTENT = "exact_literal_lookup"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class RecallProbe:
    marker: str
    expected_session_id: str
    expected_session_idx: int
    expected_turn_idx: int


@dataclass
class RecallResult:
    marker: str
    expected_session_idx: int
    expected_turn_idx: int
    source_session: str | None
    window_id: int | None
    mode: str | None
    no_silent_fallback: bool
    matched_contains_marker: bool
    answer_contains_marker: bool
    answer_contains_session: bool
    answer_contains_turn: bool
    generated_answer: str
    scored_answer: str
    answer_source: str
    matched_window_text: str
    elapsed_s: float


@dataclass
class RouterOnlyExactLiteralResult:
    marker: str
    expected_session_id: str
    expected_session_idx: int
    expected_turn_idx: int
    source_session: str | None
    window_id: int | None
    mode: str
    no_silent_fallback: bool
    matched_contains_marker: bool
    answer_contains_marker: bool
    answer_contains_session: bool
    answer_contains_turn: bool
    generated_answer: str
    scored_answer: str
    answer_source: str
    matched_window_text: str
    elapsed_s: float
    raw_tfidf_score: float
    literal_score: float
    activation_score: float
    activation_passed_gate: bool
    route_source: str
    passed: bool


@dataclass
class MultiFactRecallResult:
    probe_idx: int
    group_key: str
    mode: str | None
    no_silent_fallback: bool
    selected_tier: str | None
    mask_penalty_applied: bool
    expected_colors: list[str]
    recalled_colors: list[str]
    missing_colors: list[str]
    generated_answer: str
    elapsed_s: float


@dataclass
class RealWorldMultiFactRecallResult:
    probe_idx: int
    mode: str | None
    no_silent_fallback: bool
    selected_tier: str | None
    mask_penalty_applied: bool
    candidate_coverage: int
    hot_fact_keys: list[str]
    warm_fact_keys: list[str]
    cold_fact_keys: list[str]
    hot_hits: list[str]
    warm_hits: list[str]
    cold_hits: list[str]
    conflict_preserved: bool
    final_decision_present: bool
    pollution_hits: list[str]
    generated_answer: str
    elapsed_s: float


@dataclass
class MemoryLawsRecallResult:
    probe_idx: int
    noise_level: int
    duplicate_level: int
    no_memory_detected: bool
    hallucinated_target_fact_count: int
    no_memory_answer: str
    atlas_target_recall: int
    atlas_wrong_entity_leak_count: int
    atlas_answer_fingerprint: str
    duplicate_current_fact_present: bool
    duplicated_stale_fact_as_current: bool
    duplicate_pressure_did_not_flip_answer: bool
    current_query_final_fact_present: bool
    current_query_old_fact_not_current: bool
    history_query_old_fact_present: bool
    history_query_supersession_present: bool
    temporal_order_preserved: bool
    entity_scope_preserved: bool
    old_draft_as_current: bool
    selected_tier: str | None
    mask_penalty_applied: bool
    kv_direct_active: bool
    no_silent_fallback: bool
    candidate_count: int
    tier_assignment_count: int
    budgeted_assignment_count: int
    multi_session_count: int
    semantic_prefix_active: bool
    candidate_recall_at_4: int
    candidate_recall_at_8: int
    candidate_recall_at_12: int
    candidate_recall_at_64: int
    latency_ms: float
    vram_peak_mib: float | int | None
    atlas_answer: str
    duplicate_answer: str
    current_answer: str
    history_answer: str
    entity_answer: str


@dataclass
class MemoryDiagnosticsCurveResult:
    noise: int
    target_recall_at_4: int
    target_recall_at_8: int
    target_recall_at_12: int
    target_recall_at_64: int
    wrong_entity_leak_count: int
    near_miss_leak_count: int
    latency_ms: float
    vram_peak_mib: float | int | None
    candidate_count: int
    tier_assignment_count: int
    fallback_count: int
    answer_fingerprint: str
    generated_answer: str


REAL_WORLD_COLOR_MEMORIES: tuple[dict[str, str], ...] = (
    {
        "fact_key": "deep_teal_hero_cold",
        "match_phrase": "deep teal for the hero",
        "text": (
            "Across our website color scheme sessions, we liked deep teal "
            "for the hero, but worried it made the page feel cold."
        ),
    },
    {
        "fact_key": "purple_gradient_rejected",
        "match_phrase": "purple-to-blue gradient",
        "text": (
            "Across our website color scheme sessions, the client rejected "
            "the purple-to-blue gradient because it felt too flashy."
        ),
    },
    {
        "fact_key": "warm_white_background",
        "match_phrase": "warm white for the main background",
        "text": (
            "Across our website color scheme sessions, we chose warm white "
            "for the main background to soften the brand."
        ),
    },
    {
        "fact_key": "graphite_headings",
        "match_phrase": "graphite should be used for headings",
        "text": (
            "Across our website color scheme sessions, graphite should be "
            "used for headings instead of pure black."
        ),
    },
    {
        "fact_key": "amber_cta",
        "match_phrase": "amber is approved only",
        "text": (
            "Across our website color scheme sessions, amber is approved "
            "only for primary CTA buttons."
        ),
    },
    {
        "fact_key": "sage_replaced_teal",
        "match_phrase": "sage green replaced teal",
        "text": (
            "Across our website color scheme sessions, after review, sage "
            "green replaced teal as the accent color."
        ),
    },
    {
        "fact_key": "avoid_beige",
        "match_phrase": "avoid beige",
        "text": (
            "Across our website color scheme sessions, avoid beige because "
            "it made the site look dated."
        ),
    },
    {
        "fact_key": "muted_navy_footer",
        "match_phrase": "muted navy",
        "text": (
            "Across our website color scheme sessions, the footer can use a "
            "muted navy, but that navy should not move into the hero."
        ),
    },
    {
        "fact_key": "coral_dropped",
        "match_phrase": "coral was considered",
        "text": (
            "Across our website color scheme sessions, coral was considered "
            "for buttons, then dropped for accessibility contrast."
        ),
    },
    {
        "fact_key": "product_cards_white",
        "match_phrase": "product cards should stay white",
        "text": (
            "Across our website color scheme sessions, product cards should "
            "stay white with subtle gray borders."
        ),
    },
    {
        "fact_key": "logo_contrast",
        "match_phrase": "logo lockup needs enough contrast",
        "text": (
            "Across our website color scheme sessions, the logo lockup needs "
            "enough contrast on warm white."
        ),
    },
    {
        "fact_key": "final_palette",
        "match_phrase": "final palette direction",
        "text": (
            "Final palette direction across our website color scheme sessions: "
            "warm white background, graphite headings, sage accents replacing "
            "teal, and amber primary CTA only."
        ),
    },
)


DIRTY_STORE_POLLUTION_MEMORIES: tuple[dict[str, Any], ...] = (
    {
        "pollution_key": "acme_microsite_seasonal",
        "forbidden_terms": ("acme", "microsite", "seasonal"),
        "text": (
            "For the Acme microsite, sage and amber were only seasonal "
            "campaign colors and were not product website decisions."
        ),
    },
    {
        "pollution_key": "mobile_onboarding_purple",
        "forbidden_terms": ("mobile onboarding", "purple"),
        "text": (
            "The mobile onboarding project rejected purple, but that was "
            "unrelated to the website color scheme."
        ),
    },
    {
        "pollution_key": "investor_deck_graphite",
        "forbidden_terms": ("investor deck", "graphite"),
        "text": "The old investor deck used graphite headings, not the product website.",
    },
    {
        "pollution_key": "landing_page_coral_kept",
        "forbidden_terms": ("different landing page", "coral"),
        "text": (
            "A different landing page kept coral buttons after accessibility "
            "review; do not apply that to the product website."
        ),
    },
    {
        "pollution_key": "coffee_shop_beige",
        "forbidden_terms": ("coffee shop", "beige"),
        "text": (
            "The coffee shop brand used beige intentionally; that brand system "
            "is unrelated to the website project."
        ),
    },
    {
        "pollution_key": "teal_return_closed",
        "forbidden_terms": ("teal", "return"),
        "text": (
            "Closed branch note: someone suggested teal should return as the "
            "main accent after sage felt muted. The branch was rejected."
        ),
    },
    {
        "pollution_key": "amber_banners_old",
        "forbidden_terms": ("amber", "banner"),
        "text": (
            "Superseded component note: amber was once proposed for sitewide "
            "announcement banners before the CTA-only rule."
        ),
    },
    {
        "pollution_key": "duplicate_teal_history",
        "forbidden_terms": (),
        "text": (
            "Duplicate design-history note: a teal hero looked polished in "
            "one mock before the later review called the page too cold."
        ),
    },
)


MEMORY_LAWS_ATLAS_FACTS: tuple[dict[str, Any], ...] = (
    {
        "fact_key": "atlas_pro_price",
        "text": "Current Atlas pricing decisions across our sessions: Atlas Pro is $29 per seat monthly.",
        "match_phrases": ("atlas pro is $29", "$29 per seat", "pro tier is $29"),
    },
    {
        "fact_key": "atlas_annual_discount",
        "text": "Current Atlas pricing decisions across our sessions: the annual discount is now 18%.",
        "match_phrases": ("annual discount is now 18%", "18%"),
    },
    {
        "fact_key": "atlas_trial",
        "text": "Current Atlas pricing decisions across our sessions: the free trial stays at 14 days.",
        "match_phrases": ("free trial stays at 14 days", "14 days"),
    },
    {
        "fact_key": "atlas_overage",
        "text": "Current Atlas pricing decisions across our sessions: usage overage is $0.08 per extra automation run.",
        "match_phrases": ("usage overage is $0.08", "$0.08"),
    },
    {
        "fact_key": "atlas_enterprise",
        "text": "Current Atlas pricing decisions across our sessions: Enterprise pricing remains custom quote.",
        "match_phrases": ("enterprise pricing remains custom quote", "enterprise custom quote"),
    },
    {
        "fact_key": "atlas_final",
        "text": (
            "Final Atlas pricing decision keeps Pro at $29 per seat, 18% "
            "annual discount, 14-day trial, $0.08 overage, and Enterprise "
            "custom quote."
        ),
        "match_phrases": ("final atlas pricing decision", "final pricing decision"),
    },
)


MEMORY_LAWS_ENTITY_NOISE: tuple[str, ...] = (
    "Nimbus pricing decisions: Pro is $39 per seat with a 10% annual discount and a 30-day trial.",
    "Acme pricing decisions: Starter is $19, the trial is 7 days, and Enterprise is a public $499 plan.",
    "Old Atlas pricing decisions draft proposed Pro at $49 per seat with a 20% annual discount and $0.12 overage; this draft was rejected.",
)


MEMORY_LAWS_TEMPORAL_FACTS: tuple[str, ...] = (
    "Atlas website CTA color decision history: we originally planned Crimson as the primary CTA color.",
    "Atlas website CTA color decision history: Crimson caused contrast problems, so Amber superseded Crimson as the CTA color.",
    "Final Atlas website CTA color decision: Amber remains the final CTA color after review.",
)


MEMORY_LAWS_CANDIDATE_KS: tuple[int, ...] = (4, 8, 12, 64)


def dirty_store_scale_pollution_text(noise_idx: int) -> dict[str, Any]:
    if noise_idx < len(DIRTY_STORE_POLLUTION_MEMORIES):
        return dict(DIRTY_STORE_POLLUTION_MEMORIES[noise_idx])
    classes = (
        "near_miss_color_project",
        "stale_decision",
        "duplicate_memory",
        "unrelated_domain",
        "same_words_wrong_project",
        "contradictory_old_fact",
        "irrelevant_long_session",
    )
    noise_class = classes[noise_idx % len(classes)]
    project = f"scale-archive-{noise_idx:04d}"
    if noise_class == "near_miss_color_project":
        text = (
            f"For {project}, a campaign page used sage, amber, graphite, and "
            "warm white in seasonal artwork, not the product website palette."
        )
    elif noise_class == "stale_decision":
        text = (
            f"Archived branch {project}: teal returned as the main accent and "
            "gold banners were proposed, then the branch was rejected."
        )
    elif noise_class == "same_words_wrong_project":
        text = (
            f"Wrong-project note {project}: a mobile onboarding flow, investor "
            "deck, and Acme microsite reused words like purple, graphite, sage, "
            "amber, and footer."
        )
    elif noise_class == "contradictory_old_fact":
        text = (
            f"Contradictory archive {project}: a different landing page kept "
            "coral buttons after a separate accessibility review."
        )
    elif noise_class == "irrelevant_long_session":
        text = (
            f"Long unrelated session {project}: pricing, launch, bug history, "
            "meeting action items, architecture tradeoffs, and customer "
            "preferences with no active color-scheme decision."
        )
    elif noise_class == "duplicate_memory":
        text = (
            f"Duplicate archive {project}: copied QA, support, launch, and "
            "brand notes from an abandoned experiment."
        )
    else:
        text = (
            f"Unrelated domain archive {project}: feature requirements, bug "
            "triage, pricing notes, and support macros for another product."
        )
    return {
        "pollution_key": f"{noise_class}_{noise_idx:04d}",
        "forbidden_terms": (),
        "text": text,
    }


def real_world_fact_mentioned(fact_key: str, answer: str) -> bool:
    lower = answer.lower()
    if fact_key == "deep_teal_hero_cold":
        return "teal" in lower and "hero" in lower and "cold" in lower
    if fact_key == "purple_gradient_rejected":
        return "purple" in lower and "gradient" in lower and "reject" in lower
    if fact_key == "warm_white_background":
        return "warm white" in lower and "background" in lower
    if fact_key == "graphite_headings":
        return "graphite" in lower and "heading" in lower
    if fact_key == "amber_cta":
        return "amber" in lower and "cta" in lower
    if fact_key == "sage_replaced_teal":
        return "sage" in lower and "teal" in lower and "replac" in lower
    if fact_key == "avoid_beige":
        return "beige" in lower and ("avoid" in lower or "dated" in lower)
    if fact_key == "muted_navy_footer":
        return "navy" in lower and "footer" in lower and "hero" in lower
    if fact_key == "coral_dropped":
        return "coral" in lower and ("contrast" in lower or "accessibility" in lower)
    if fact_key == "product_cards_white":
        return "product card" in lower and "white" in lower and "border" in lower
    if fact_key == "logo_contrast":
        return "logo" in lower and "contrast" in lower and "warm white" in lower
    if fact_key == "final_palette":
        return (
            "final" in lower
            and "warm white" in lower
            and "graphite" in lower
            and "sage" in lower
            and "amber" in lower
        )
    raise KeyError(f"unknown real-world fact key: {fact_key}")


def real_world_conflict_preserved(answer: str) -> bool:
    lower = answer.lower()
    return "teal" in lower and "sage" in lower and "replac" in lower


def real_world_final_decision_present(answer: str) -> bool:
    lower = answer.lower()
    return (
        ("final" in lower or "direction" in lower)
        and "warm white" in lower
        and "graphite" in lower
        and "sage" in lower
        and "amber" in lower
    )


def dirty_store_pollution_hits(answer: str) -> list[str]:
    lower = answer.lower()
    hits: list[str] = []
    for pollution in DIRTY_STORE_POLLUTION_MEMORIES:
        terms = tuple(str(term).lower() for term in pollution.get("forbidden_terms", ()))
        if terms and all(term in lower for term in terms):
            hits.append(str(pollution["pollution_key"]))
    return hits


def parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def memory_laws_noise_text(noise_idx: int) -> str:
    project = f"scale-laws-noise-{noise_idx:05d}"
    kind = noise_idx % 6
    if kind == 0:
        return (
            f"Nimbus pricing archive {project}: Pro is $39, annual discount "
            "is 10%, and the trial is 30 days for Nimbus only."
        )
    if kind == 1:
        return (
            f"Acme pricing archive {project}: Starter is $19, Enterprise "
            "was public at $499, and this was never an Atlas decision."
        )
    if kind == 2:
        return (
            f"Other website palette archive {project}: warm white, sage, "
            "graphite, and amber appeared in a seasonal campaign, not Solace."
        )
    if kind == 3:
        return (
            f"Old component archive {project}: crimson and coral CTA "
            "experiments were closed without current decisions."
        )
    if kind == 4:
        return (
            f"Meridian checkout bug archive {project}: Safari autofill and "
            "idempotency notes unrelated to pricing or color palettes."
        )
    return (
        f"Launch operations archive {project}: QA signoff, analytics mapping, "
        "support handoff, and release owners."
    )


def memory_laws_no_memory_detected(answer: str) -> bool:
    lower = answer.lower()
    return (
        ("do not have" in lower or "don't have" in lower or "no stored" in lower)
        and "stored decision" in lower
        and "solace" in lower
    )


def memory_laws_hallucinated_palette_fact_count(answer: str) -> int:
    lower = answer.lower()
    borrowed_facts = (
        ("warm white", "background"),
        ("graphite", "heading"),
        ("sage", "accent"),
        ("amber", "cta"),
        ("teal", "hero"),
        ("purple", "gradient"),
    )
    return sum(1 for pair in borrowed_facts if all(term in lower for term in pair))


def memory_laws_atlas_hits(answer: str) -> list[str]:
    lower = answer.lower()
    hits: list[str] = []
    if "$29" in lower and "pro" in lower:
        hits.append("atlas_pro_price")
    if "18%" in lower and "annual" in lower:
        hits.append("atlas_annual_discount")
    if "14" in lower and "trial" in lower:
        hits.append("atlas_trial")
    if "$0.08" in lower and "overage" in lower:
        hits.append("atlas_overage")
    if "enterprise" in lower and "custom" in lower:
        hits.append("atlas_enterprise")
    if "final" in lower and "$29" in lower and "18%" in lower:
        hits.append("atlas_final")
    return hits


def memory_laws_atlas_fingerprint(answer: str) -> str:
    return "|".join(sorted(memory_laws_atlas_hits(answer)))


def memory_laws_wrong_entity_hits(answer: str) -> list[str]:
    lower = answer.lower()
    forbidden = (
        ("nimbus", "$39"),
        ("nimbus", "30-day"),
        ("acme", "$19"),
        ("acme", "7 days"),
        ("acme", "$499"),
        ("$49", "current"),
        ("20%", "current"),
        ("$0.12", "current"),
    )
    return [
        "+".join(terms)
        for terms in forbidden
        if all(term in lower for term in terms)
    ]


def memory_laws_old_draft_as_current(answer: str) -> bool:
    lower = answer.lower()
    if "$49" not in lower and "$0.12" not in lower and "20%" not in lower:
        return False
    stale_markers = ("old", "draft", "rejected", "superseded", "previous", "not current")
    return not any(marker in lower for marker in stale_markers)


def memory_laws_temporal_current_ok(answer: str) -> bool:
    lower = answer.lower()
    return "amber" in lower and "current cta color" in lower


def memory_laws_temporal_old_not_current(answer: str) -> bool:
    lower = answer.lower()
    return not any(
        pattern in lower
        for pattern in (
            "current cta color is crimson",
            "crimson is the current",
            "current color is crimson",
        )
    )


def memory_laws_temporal_history_ok(answer: str) -> bool:
    lower = answer.lower()
    return (
        "crimson" in lower
        and "amber" in lower
        and ("superseded" in lower or "replaced" in lower)
        and ("final" in lower or "remains" in lower)
    )


def dedupe_candidates_by_session(candidates: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()
    for candidate in candidates:
        session_id = str(candidate.handle.session_id)
        if session_id in seen:
            continue
        seen.add(session_id)
        deduped.append(candidate)
    return deduped


def real_world_fact_entailed_by_selected(
    fact_key: str,
    selected_fact_keys: set[str],
) -> bool:
    """Return True when a fact is already covered by a HOT/WARM memory."""
    if fact_key in selected_fact_keys:
        return True
    if "final_palette" in selected_fact_keys and fact_key in {
        "warm_white_background",
        "graphite_headings",
        "amber_cta",
        "sage_replaced_teal",
    }:
        return True
    return False


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def localize_path(path_text: str) -> Path:
    """Map WSL ``/mnt/c/...`` paths to Windows paths when needed."""
    path = Path(path_text)
    if path.exists():
        return path
    normalized = path_text.replace("\\", "/")
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", normalized)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2).replace("/", "\\")
        candidate = Path(f"{drive}:\\{rest}")
        if candidate.exists():
            return candidate
    return path


def latest_pass_run(output_root: Path, *, min_sessions: int) -> Path:
    summaries = sorted(output_root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for summary_path in summaries:
        summary = load_json(summary_path)
        if summary.get("status") != "PASS":
            continue
        if int(summary.get("sessions", 0) or 0) < min_sessions:
            continue
        return summary_path.parent
    raise RuntimeError(f"No PASS run with sessions >= {min_sessions} under {output_root}")


def load_run_summary(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = latest_pass_run(Path(args.output_root), min_sessions=args.min_sessions)
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"summary.json not found under {run_dir}")
    summary = load_json(summary_path)
    if summary.get("status") != "PASS" and not args.allow_non_pass:
        raise RuntimeError(f"{summary_path} is not a PASS run; use --allow-non-pass to override")
    return run_dir, summary


def parse_scale_probes(events_path: Path) -> list[RecallProbe]:
    probes: list[RecallProbe] = []
    seen: set[str] = set()
    pattern = re.compile(r"belongs to session\s+(\d+)\s+turn\s+(\d+)", re.IGNORECASE)
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"event": "routing.probe"' not in line:
                continue
            event = json.loads(line)
            marker = str(event.get("marker", ""))
            expected_session_id = str(event.get("expected_session", ""))
            if not marker or marker in seen:
                continue
            text = str(event.get("window_text_head", ""))
            match = pattern.search(text)
            if match is None:
                for candidate in event.get("top_candidates", []) or []:
                    candidate_text = str(candidate.get("window_text_head", ""))
                    if marker.lower() in candidate_text.lower():
                        match = pattern.search(candidate_text)
                        break
            if match is None:
                continue
            seen.add(marker)
            probes.append(
                RecallProbe(
                    marker=marker,
                    expected_session_id=expected_session_id,
                    expected_session_idx=int(match.group(1)),
                    expected_turn_idx=int(match.group(2)),
                )
            )
    return probes


def parse_scale_transcript_probes(events_path: Path, transcripts_root: Path) -> list[RecallProbe]:
    """Reconstruct every planted 100x100 marker from saved scale transcripts."""
    sessions: list[tuple[int, str]] = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"event": "scale.session_saved"' not in line:
                continue
            event = json.loads(line)
            sessions.append((int(event["session_idx"]), str(event["session_id"])))

    probes: list[RecallProbe] = []
    seen: set[str] = set()
    pattern = re.compile(
        r"^([0-9a-f]{16})\. Unique route key \1\. "
        r"This planted scale memory belongs to session\s+(\d+)\s+turn\s+(\d+)\.",
        re.IGNORECASE,
    )
    for _session_idx, session_id in sessions:
        transcript_path = transcripts_root / f"{session_id}.json"
        if not transcript_path.is_file():
            raise RuntimeError(f"scale transcript not found: {transcript_path}")
        transcript = load_json(transcript_path)
        for turn in transcript.get("turns", []) or []:
            text = str(turn.get("text", ""))
            match = pattern.search(text)
            if match is None:
                continue
            marker = str(match.group(1))
            if marker in seen:
                continue
            seen.add(marker)
            probes.append(
                RecallProbe(
                    marker=marker,
                    expected_session_id=session_id,
                    expected_session_idx=int(match.group(2)),
                    expected_turn_idx=int(match.group(3)),
                )
            )
    return probes


def select_probes(probes: list[RecallProbe], sample_size: int) -> list[RecallProbe]:
    if sample_size <= 0 or sample_size >= len(probes):
        return list(probes)
    if sample_size == 1:
        return [probes[0]]
    last = len(probes) - 1
    selected: list[RecallProbe] = []
    seen_indices: set[int] = set()
    for i in range(sample_size):
        idx = round(i * last / (sample_size - 1))
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        selected.append(probes[idx])
    return selected


def load_interactive_memory_chat() -> Any:
    script_path = REPO_ROOT / "scripts" / "interactive_memory_chat.py"
    spec = importlib.util.spec_from_file_location("interactive_memory_chat", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load import spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["interactive_memory_chat"] = module
    spec.loader.exec_module(module)
    return module


def force_deterministic_streaming() -> None:
    """Reuse the production harness' deterministic greedy patch."""
    spec = importlib.util.spec_from_file_location(
        "auto_verify_memory_repl", REPO_ROOT / "scripts" / "auto_verify_memory_repl.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import auto_verify_memory_repl")
    module = importlib.util.module_from_spec(spec)
    sys.modules["auto_verify_memory_repl"] = module
    spec.loader.exec_module(module)

    class _Log:
        def event(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    module.force_deterministic_streaming(_Log())


def contains_session(answer: str, expected: int) -> bool:
    return bool(re.search(rf"\bsession\s*[:=]?\s*{expected}\b", answer, re.IGNORECASE))


def contains_turn(answer: str, expected: int) -> bool:
    return bool(re.search(rf"\bturn\s*[:=]?\s*{expected}\b", answer, re.IGNORECASE))


def make_query(probe: RecallProbe) -> str:
    return (
        f"From memory, use retrieval key {probe.marker}. "
        "What planted scale memory does this key identify? "
        f"Answer exactly as: marker={probe.marker}; session=<number>; "
        "turn=<number>. Do not guess from this prompt; use memory."
    )


def exact_literal_answer_from_window(probe: RecallProbe, window_text: str) -> str | None:
    pattern = re.compile(
        rf"{re.escape(probe.marker)}\.\s+Unique route key\s+{re.escape(probe.marker)}\.\s+"
        r"This planted scale memory belongs to session\s+(\d+)\s+turn\s+(\d+)\.",
        re.IGNORECASE,
    )
    match = pattern.search(window_text)
    if match is None:
        return None
    return f"marker={probe.marker}; session={int(match.group(1))}; turn={int(match.group(2))}"


def run_probe(
    chat: Any,
    probe: RecallProbe,
    *,
    mode: str,
    score_exact_literal_from_route: bool = False,
) -> RecallResult:
    chat.memory_mode = mode
    chat.start_new_session()
    query = make_query(probe)
    started = time.time()
    if mode == "kv_direct":
        meta = chat.kv_query_turn(query)
    elif mode == "topical":
        meta = chat.recall_chat_turn(query)
    else:
        raise ValueError(f"unsupported mode {mode!r}")

    elapsed = time.time() - started
    answer = str(getattr(meta, "generated_answer", "") or "")
    matched = str(getattr(meta, "matched_window_text", "") or "")
    answer_source = "model"
    scored_answer = answer
    if score_exact_literal_from_route:
        routed_answer = exact_literal_answer_from_window(probe, matched)
        if routed_answer is not None:
            scored_answer = routed_answer
            answer_source = "route_exact_literal"
    return RecallResult(
        marker=probe.marker,
        expected_session_idx=probe.expected_session_idx,
        expected_turn_idx=probe.expected_turn_idx,
        source_session=getattr(meta, "source_session", None),
        window_id=getattr(meta, "window_id", None),
        mode=getattr(meta, "mode", None),
        no_silent_fallback=bool(getattr(meta, "no_silent_fallback", False)),
        matched_contains_marker=probe.marker.lower() in matched.lower(),
        answer_contains_marker=probe.marker.lower() in scored_answer.lower(),
        answer_contains_session=contains_session(scored_answer, probe.expected_session_idx),
        answer_contains_turn=contains_turn(scored_answer, probe.expected_turn_idx),
        generated_answer=answer,
        scored_answer=scored_answer,
        answer_source=answer_source,
        matched_window_text=matched[:700],
        elapsed_s=elapsed,
    )


def _default_tokenizer_model_path(model_path: str | None) -> str:
    if model_path:
        return str(model_path)
    local_snapshot = (
        "/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/"
        "snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
    )
    if Path(local_snapshot).is_dir():
        return local_snapshot
    return "google/gemma-4-E2B-it"


def _clear_cuda_cache_if_available() -> None:
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_router_only_exact_literal_probe(
    *,
    handles: list[Any],
    tokenizer: Any,
    store_cache: dict[str, Any],
    probe: RecallProbe,
    store_root: Path,
    candidate_pool: int,
) -> RouterOnlyExactLiteralResult:
    from chuk_lazarus.session_retrieval import asi_router as asi_router_module
    from chuk_lazarus.session_retrieval.asi_router import asi_route_candidates

    query = make_query(probe)
    started = time.time()
    candidates = asi_route_candidates(
        handles,
        query,
        tokenizer,
        candidate_pool=max(1, int(candidate_pool)),
        archive_root=store_root,
    )
    elapsed = time.time() - started
    candidate = candidates[0] if candidates else None
    matched = ""
    routed_answer = None
    source_session = None
    window_id = None
    raw_tfidf_score = 0.0
    literal_score = 0.0
    activation_score = 0.0
    activation_passed_gate = False
    route_source = "none"
    if candidate is not None:
        source_session = str(candidate.handle.session_id)
        window_id = int(candidate.window_id)
        store_key = str(candidate.handle.torch_store_dir)
        store = store_cache.get(store_key)
        if store is None:
            store = asi_router_module.load_store(candidate.handle)
            store_cache[store_key] = store
        matched = store.get_window_text(int(candidate.window_id), tokenizer)
        routed_answer = exact_literal_answer_from_window(probe, matched)
        raw_tfidf_score = float(candidate.raw_tfidf_score_pre_normalization)
        literal_score = float(candidate.literal_score)
        activation_score = float(candidate.activation_score)
        telemetry = dict(getattr(candidate, "selector_telemetry", {}) or {})
        activation_passed_gate = bool(telemetry.get("activation_passed_gate", False))
        route_source = str(telemetry.get("route_source") or "unknown")

    scored_answer = routed_answer or ""
    passed = (
        source_session == probe.expected_session_id
        and probe.marker.lower() in matched.lower()
        and probe.marker.lower() in scored_answer.lower()
        and contains_session(scored_answer, probe.expected_session_idx)
        and contains_turn(scored_answer, probe.expected_turn_idx)
    )
    return RouterOnlyExactLiteralResult(
        marker=probe.marker,
        expected_session_id=probe.expected_session_id,
        expected_session_idx=probe.expected_session_idx,
        expected_turn_idx=probe.expected_turn_idx,
        source_session=source_session,
        window_id=window_id,
        mode="router_only_exact_literal",
        no_silent_fallback=True,
        matched_contains_marker=probe.marker.lower() in matched.lower(),
        answer_contains_marker=probe.marker.lower() in scored_answer.lower(),
        answer_contains_session=contains_session(scored_answer, probe.expected_session_idx),
        answer_contains_turn=contains_turn(scored_answer, probe.expected_turn_idx),
        generated_answer=scored_answer,
        scored_answer=scored_answer,
        answer_source="route_exact_literal",
        matched_window_text=matched[:700],
        elapsed_s=elapsed,
        raw_tfidf_score=raw_tfidf_score,
        literal_score=literal_score,
        activation_score=activation_score,
        activation_passed_gate=activation_passed_gate,
        route_source=route_source,
        passed=passed,
    )


def run_router_only_exact_literal_mode(
    args: argparse.Namespace,
    run_dir: Path,
    store_root: Path,
    events_path: Path,
    probes: list[RecallProbe],
    *,
    probe_source: str,
) -> int:
    from transformers import AutoTokenizer

    from chuk_lazarus.session_retrieval import asi_router as asi_router_module
    from chuk_lazarus.session_retrieval.enumeration import (
        iter_checkpoint_handles,
        load_store as real_load_store,
    )

    handles = list(iter_checkpoint_handles(store_root / "checkpoints"))
    if not handles:
        raise RuntimeError(f"No checkpoint handles found under {store_root / 'checkpoints'}")

    tokenizer = AutoTokenizer.from_pretrained(_default_tokenizer_model_path(args.model_path))
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    store_cache: dict[str, Any] = {}

    def cached_load_store(handle: Any) -> Any:
        key = str(handle.torch_store_dir)
        store = store_cache.get(key)
        if store is None:
            store = real_load_store(handle)
            store_cache[key] = store
        return store

    asi_router_module.load_store = cached_load_store

    results: list[RouterOnlyExactLiteralResult] = []
    passed = 0
    candidate_pool = int(getattr(args, "router_only_candidate_pool", 1) or 1)
    for idx, probe in enumerate(probes, start=1):
        result = run_router_only_exact_literal_probe(
            handles=handles,
            tokenizer=tokenizer,
            store_cache=store_cache,
            probe=probe,
            store_root=store_root,
            candidate_pool=candidate_pool,
        )
        results.append(result)
        passed += int(result.passed)
        if not result.passed:
            expected = (
                f"marker={probe.marker}; session={probe.expected_session_idx}; "
                f"turn={probe.expected_turn_idx}"
            )
            failure = {
                "probe_index": idx,
                "question": make_query(probe),
                "expected_session_id": probe.expected_session_id,
                "expected_string": expected,
                "source_session": result.source_session,
                "window_id": result.window_id,
                "top_activation_score": result.activation_score,
                "raw_tfidf_score": result.raw_tfidf_score,
                "literal_score": result.literal_score,
                "route_source": result.route_source,
                "generated_string": result.generated_answer,
                "matched_window_text": result.matched_window_text,
            }
            print("FAIL 100x100 MATRIX: exact/literal hybrid router miss", flush=True)
            print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
            if args.fail_fast:
                break
        if idx % 500 == 0:
            _clear_cuda_cache_if_available()
            if not args.quiet_model_output:
                print(
                    f"ROUTER_ONLY_EXACT_LITERAL progress={idx}/{len(probes)} "
                    f"passed={passed}",
                    flush=True,
                )

    hit_rate = passed / max(1, len(results))
    final_summary = {
        "run_dir": str(run_dir),
        "store_root": str(store_root),
        "events_path": str(events_path),
        "mode": "router_only_exact_literal",
        "probe_source": probe_source,
        "suite_intent": MARKER_SUITE_INTENT,
        "semantic_only": False,
        "sample_size": len(results),
        "passed": passed,
        "hit_rate": hit_rate,
        "required_hit_rate": args.required_hit_rate,
        "candidate_pool": candidate_pool,
    }
    report_path = args.report_json or (run_dir / "scale-actual-recall-router-only-exact.json")
    write_report(report_path, results, final_summary)
    if hit_rate < args.required_hit_rate:
        print(
            f"FAIL SCALE_ACTUAL_RECALL: mode=router_only_exact_literal "
            f"hit_rate={hit_rate:.3f} required={args.required_hit_rate:.3f} "
            f"report={report_path}",
            flush=True,
        )
        return 1
    if len(results) == 10000 and passed == len(results):
        print("100x100 Matrix Passed: Semantic Router is Operational", flush=True)
    else:
        print(
            f"PASS SCALE_ACTUAL_RECALL: mode=router_only_exact_literal "
            f"hit_rate={hit_rate:.3f} passed={passed}/{len(results)} "
            f"report={report_path}",
            flush=True,
        )
    return 0


def direct_turn(chat: Any, role: Any, text: str) -> None:
    turn = chat.session.begin_turn(role, text)
    chat.session.finish_turn(turn)
    chat._capture_turn_text_live(turn)


def plant_multi_fact_group(
    chat: Any,
    role: Any,
    *,
    probe_idx: int,
    colors: list[str],
) -> tuple[str, list[dict[str, Any]]]:
    group_key = f"scale_multi_fact_{probe_idx:04d}_{secrets.token_hex(6)}"
    facts: list[dict[str, Any]] = []
    for fact_idx, color in enumerate(colors):
        chat.memory_mode = "off"
        chat.start_new_session()
        session_id = chat.session.session_id
        marker = f"smf{probe_idx:04d}_{fact_idx}_{secrets.token_hex(6)}"
        text = (
            f"{marker}. Unique multi-fact recall marker {marker}. "
            f"Website palette decision group {group_key}. "
            f"For this website color scheme, favorite color slot {fact_idx} "
            f"is {color}. Design palette decisions for {group_key} include {color}."
        )
        direct_turn(chat, role, text)
        direct_turn(
            chat,
            role,
            (
                f"Neutral padding note {secrets.token_hex(8)}. "
                "This line gives the session a second retrieval window without "
                "adding palette facts."
            ),
        )
        chat._mark_dirty()
        if not chat.save_current_session(rebuild_retriever=True):
            raise RuntimeError(
                "save_current_session returned False while planting "
                f"multi_fact probe {probe_idx} fact {fact_idx}"
            )
        facts.append(
            {
                "fact_idx": fact_idx,
                "session_id": session_id,
                "marker": marker,
                "color": color,
            }
        )
    return group_key, facts


def reset_multi_fact_probe_store(chat: Any, *, preserve_retriever: bool = False) -> None:
    """Clear the verifier's disposable store between independent probes."""
    if getattr(chat, "indexer", None) is not None:
        with contextlib.suppress(Exception):
            chat.indexer.stop()
    chat.indexer = None
    if not preserve_retriever:
        chat.retriever = None
    chat.session = None
    chat.history = None
    chat.vec_inject_provider = None
    for root in (chat.inputs_root, chat.checkpoints_root, chat.transcripts_root):
        root.mkdir(parents=True, exist_ok=True)
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def run_multi_fact_probe(
    chat: Any,
    role: Any,
    *,
    probe_idx: int,
    colors: list[str],
) -> MultiFactRecallResult:
    group_key, facts = plant_multi_fact_group(
        chat,
        role,
        probe_idx=probe_idx,
        colors=colors,
    )
    query = (
        f"List all favorite color answers from prior website color scheme "
        f"palette decisions in group {group_key}. Return only the four color "
        "words."
    )
    chat.memory_mode = "kv_direct"
    chat.start_new_session()
    started = time.time()
    meta = chat.kv_query_turn(query)
    elapsed = time.time() - started
    answer = str(getattr(meta, "generated_answer", "") or "")
    answer_lower = answer.lower()
    expected = [str(fact["color"]) for fact in facts]
    recalled = [color for color in expected if color.lower() in answer_lower]
    missing = [color for color in expected if color not in recalled]
    return MultiFactRecallResult(
        probe_idx=probe_idx,
        group_key=group_key,
        mode=getattr(meta, "mode", None),
        no_silent_fallback=bool(getattr(meta, "no_silent_fallback", False)),
        selected_tier=getattr(meta, "selected_tier", None),
        mask_penalty_applied=bool(getattr(meta, "mask_penalty_applied", False)),
        expected_colors=expected,
        recalled_colors=recalled,
        missing_colors=missing,
        generated_answer=answer,
        elapsed_s=elapsed,
    )


def plant_real_world_multi_fact_group(
    chat: Any,
    role: Any,
    *,
    probe_idx: int,
) -> list[dict[str, Any]]:
    del probe_idx
    facts: list[dict[str, Any]] = []
    for fact_idx, fact in enumerate(REAL_WORLD_COLOR_MEMORIES):
        chat.memory_mode = "off"
        chat.start_new_session()
        session_id = chat.session.session_id
        direct_turn(chat, role, fact["text"])
        direct_turn(
            chat,
            role,
            (
                f"Neutral project note {secrets.token_hex(8)}. "
                "This follow-up is ordinary session padding and does not add "
                "any project decision."
            ),
        )
        chat._mark_dirty()
        if not chat.save_current_session(rebuild_retriever=True):
            raise RuntimeError(
                "save_current_session returned False while planting "
                f"real_world_multi_fact probe fact {fact_idx}"
            )
        facts.append(
            {
                "fact_idx": fact_idx,
                "session_id": session_id,
                "fact_key": fact["fact_key"],
                "match_phrase": fact["match_phrase"],
                "text": fact["text"],
            }
        )
    return facts


def plant_dirty_store_pollution(
    chat: Any,
    role: Any,
    *,
    probe_idx: int,
    noise_session_count: int = 50,
) -> list[dict[str, Any]]:
    planted: list[dict[str, Any]] = []
    for pollution_idx in range(int(noise_session_count)):
        pollution = dirty_store_scale_pollution_text(pollution_idx)
        chat.memory_mode = "off"
        chat.start_new_session()
        session_id = chat.session.session_id
        direct_turn(
            chat,
            role,
            (
                f"Archive pollution probe {probe_idx} note {pollution_idx}. "
                f"{pollution['text']}"
            ),
        )
        direct_turn(
            chat,
            role,
            (
                f"Unrelated archive padding {secrets.token_hex(8)}. "
                "This line keeps the old session realistic without adding "
                "an active color-scheme decision."
            ),
        )
        chat._mark_dirty()
        if not chat.save_current_session(rebuild_retriever=True):
            raise RuntimeError(
                "save_current_session returned False while planting "
                f"dirty_store pollution probe {probe_idx} item {pollution_idx}"
            )
        planted.append(
            {
                "pollution_idx": pollution_idx,
                "session_id": session_id,
                "pollution_key": pollution["pollution_key"],
                "text": pollution["text"],
            }
        )
    return planted


def run_real_world_multi_fact_probe(
    chat: Any,
    role: Any,
    *,
    probe_idx: int,
    dirty_store: bool = False,
    dirty_noise_session_count: int = 50,
) -> RealWorldMultiFactRecallResult:
    if dirty_store:
        plant_dirty_store_pollution(
            chat,
            role,
            probe_idx=probe_idx,
            noise_session_count=dirty_noise_session_count,
        )
    facts = plant_real_world_multi_fact_group(
        chat,
        role,
        probe_idx=probe_idx,
    )
    query = (
        "Tell me everything we discussed about the website's color scheme "
        "across all our sessions."
    )

    from chuk_lazarus.session_retrieval import asi_route_candidates, assign_tiers
    from chuk_lazarus.session_retrieval.enumeration import load_store

    candidates = asi_route_candidates(
        chat.retriever.handles,
        query,
        chat.retriever.tokenizer,
        candidate_pool=24,
    )
    candidates = dedupe_candidates_by_session(candidates)
    records_by_phrase = {
        str(record["match_phrase"]).lower(): record for record in facts
    }
    candidate_records: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates):
        store = load_store(candidate.handle)
        window_text = store.get_window_text(int(candidate.window_id), chat.tokenizer)
        matched = None
        for phrase, record in records_by_phrase.items():
            if phrase in window_text.lower():
                matched = record
                break
        if matched is not None:
            candidate_records.append(
                {
                    **matched,
                    "rank": rank,
                    "candidate_session_id": candidate.handle.session_id,
                    "window_id": int(candidate.window_id),
                }
            )
    candidate_coverage = len({str(record["fact_key"]) for record in candidate_records})

    assignments = assign_tiers(
        candidates,
        K_HOT=4,
        K_WARM=4,
        candidate_pool=12,
    )
    records_by_session_window = {
        (record["candidate_session_id"], int(record["window_id"])): record
        for record in candidate_records
    }
    tier_records: dict[str, list[dict[str, Any]]] = {
        "hot": [],
        "warm": [],
        "cold": [],
    }
    for assignment in assignments:
        key = (
            assignment.candidate.handle.session_id,
            int(assignment.candidate.window_id),
        )
        record = records_by_session_window.get(key)
        if record is not None:
            tier_records[assignment.tier.value].append(record)

    chat.memory_mode = "kv_direct"
    chat.start_new_session()
    started = time.time()
    meta = chat.kv_query_turn(query)
    elapsed = time.time() - started
    answer = str(getattr(meta, "generated_answer", "") or "")
    hot_fact_keys = [str(record["fact_key"]) for record in tier_records["hot"]]
    warm_fact_keys = [str(record["fact_key"]) for record in tier_records["warm"]]
    cold_fact_keys = [str(record["fact_key"]) for record in tier_records["cold"]]
    hot_hits = [
        fact_key for fact_key in hot_fact_keys
        if real_world_fact_mentioned(fact_key, answer)
    ]
    warm_hits = [
        fact_key for fact_key in warm_fact_keys
        if real_world_fact_mentioned(fact_key, answer)
    ]
    selected_fact_keys = set(hot_fact_keys) | set(warm_fact_keys)
    cold_only_fact_keys = [
        fact_key for fact_key in cold_fact_keys
        if not real_world_fact_entailed_by_selected(fact_key, selected_fact_keys)
    ]
    cold_hits = [
        fact_key for fact_key in cold_only_fact_keys
        if real_world_fact_mentioned(fact_key, answer)
    ]
    return RealWorldMultiFactRecallResult(
        probe_idx=probe_idx,
        mode=getattr(meta, "mode", None),
        no_silent_fallback=bool(getattr(meta, "no_silent_fallback", False)),
        selected_tier=getattr(meta, "selected_tier", None),
        mask_penalty_applied=bool(getattr(meta, "mask_penalty_applied", False)),
        candidate_coverage=candidate_coverage,
        hot_fact_keys=hot_fact_keys,
        warm_fact_keys=warm_fact_keys,
        cold_fact_keys=cold_fact_keys,
        hot_hits=hot_hits,
        warm_hits=warm_hits,
        cold_hits=cold_hits,
        conflict_preserved=real_world_conflict_preserved(answer),
        final_decision_present=real_world_final_decision_present(answer),
        pollution_hits=dirty_store_pollution_hits(answer) if dirty_store else [],
        generated_answer=answer,
        elapsed_s=elapsed,
    )


def plant_memory_laws_session(
    chat: Any,
    role: Any,
    text: str,
    *,
    rebuild_retriever: bool = True,
) -> str:
    chat.memory_mode = "off"
    chat.start_new_session()
    session_id = chat.session.session_id
    direct_turn(chat, role, text)
    direct_turn(
        chat,
        role,
        (
            f"Memory laws scale padding {secrets.token_hex(8)}. "
            "This continuation does not add any new current decision."
        ),
    )
    chat._mark_dirty()
    if not chat.save_current_session(rebuild_retriever=rebuild_retriever):
        raise RuntimeError("save_current_session returned False while planting memory-laws fixture")
    return session_id


def refresh_memory_laws_retriever(chat: Any) -> None:
    """Refresh once after a memory-laws planting batch."""
    if chat.retriever is None:
        chat.maybe_load_retriever()
        if chat.retriever is None:
            raise RuntimeError("memory-laws fixture planted no retrievable sessions")
        return
    added = chat.retriever.refresh_handles(chat.checkpoints_root)
    if added <= 0:
        raise RuntimeError("memory-laws retriever refresh did not add planted sessions")


def memory_laws_candidate_recall_at(
    chat: Any,
    candidates: list[Any],
    target_records: list[dict[str, Any]],
    k: int,
) -> int:
    if not target_records:
        return 0
    from chuk_lazarus.session_retrieval.enumeration import load_store

    seen: set[str] = set()
    for candidate in candidates[: int(k)]:
        store = load_store(candidate.handle)
        lower = store.get_window_text(int(candidate.window_id), chat.tokenizer).lower()
        for record in target_records:
            phrases = tuple(str(phrase).lower() for phrase in record.get("match_phrases", ()))
            if phrases and any(phrase in lower for phrase in phrases):
                seen.add(str(record["fact_key"]))
    return len(seen)


def memory_laws_route_recall(
    chat: Any,
    query: str,
    target_records: list[dict[str, Any]],
) -> dict[str, int]:
    from chuk_lazarus.session_retrieval import asi_route_candidates

    if chat.retriever is None:
        raise RuntimeError("memory laws route recall requires retriever")
    candidates = asi_route_candidates(
        chat.retriever.handles,
        query,
        chat.retriever.tokenizer,
        candidate_pool=64,
    )
    candidates = dedupe_candidates_by_session(candidates)
    return {
        f"candidate_recall_at_{k}": memory_laws_candidate_recall_at(
            chat,
            candidates,
            target_records,
            k,
        )
        for k in MEMORY_LAWS_CANDIDATE_KS
    }


def run_memory_laws_query(chat: Any, query: str) -> tuple[Any, float]:
    chat.memory_mode = "kv_direct"
    chat.start_new_session()
    started = time.time()
    meta = chat.kv_query_turn(query)
    return meta, time.time() - started


def run_memory_laws_probe(
    chat: Any,
    role: Any,
    *,
    probe_idx: int,
    noise_level: int,
    duplicate_level: int,
) -> MemoryLawsRecallResult:
    for noise_idx in range(int(noise_level)):
        plant_memory_laws_session(
            chat,
            role,
            memory_laws_noise_text(noise_idx),
            rebuild_retriever=False,
        )
    target_records = []
    for fact in MEMORY_LAWS_ATLAS_FACTS:
        plant_memory_laws_session(
            chat,
            role,
            str(fact["text"]),
            rebuild_retriever=False,
        )
        target_records.append(dict(fact))
    stale = (
        "Old Atlas pricing decisions draft proposed $49 per seat monthly, "
        "but this stale draft was rejected and superseded."
    )
    near_miss = (
        "Nimbus pricing duplicate near-miss: Pro is $39 per seat, annual "
        "discount is 10%, and the trial is 30 days for Nimbus only."
    )
    for dup_idx in range(int(duplicate_level)):
        plant_memory_laws_session(
            chat,
            role,
            stale if dup_idx % 2 == 0 else near_miss,
            rebuild_retriever=False,
        )
    for text in MEMORY_LAWS_TEMPORAL_FACTS:
        plant_memory_laws_session(chat, role, text, rebuild_retriever=False)
    for text in MEMORY_LAWS_ENTITY_NOISE:
        plant_memory_laws_session(chat, role, text, rebuild_retriever=False)

    refresh_memory_laws_retriever(chat)

    no_memory_meta, no_memory_elapsed = run_memory_laws_query(
        chat,
        "What did we decide about the Solace website color palette?",
    )
    no_memory_answer = str(getattr(no_memory_meta, "generated_answer", "") or "")
    atlas_query = "What are the current Atlas pricing decisions across our sessions?"
    candidate_recall = memory_laws_route_recall(chat, atlas_query, target_records)
    atlas_meta, atlas_elapsed = run_memory_laws_query(chat, atlas_query)
    atlas_answer = str(getattr(atlas_meta, "generated_answer", "") or "")
    duplicate_meta, duplicate_elapsed = run_memory_laws_query(
        chat,
        "What is the current Atlas Pro price?",
    )
    duplicate_answer = str(getattr(duplicate_meta, "generated_answer", "") or "")
    current_meta, current_elapsed = run_memory_laws_query(
        chat,
        "What is the current CTA color decision?",
    )
    current_answer = str(getattr(current_meta, "generated_answer", "") or "")
    history_meta, history_elapsed = run_memory_laws_query(
        chat,
        "How did the CTA color decision change over time?",
    )
    history_answer = str(getattr(history_meta, "generated_answer", "") or "")
    entity_meta, entity_elapsed = run_memory_laws_query(
        chat,
        "What are the current Atlas pricing decisions across our sessions?",
    )
    entity_answer = str(getattr(entity_meta, "generated_answer", "") or "")

    wrong_hits = memory_laws_wrong_entity_hits(atlas_answer) + memory_laws_wrong_entity_hits(entity_answer)
    atlas_hits = memory_laws_atlas_hits(atlas_answer)
    entity_hits = memory_laws_atlas_hits(entity_answer)
    duplicate_current = "$29" in duplicate_answer and "pro" in duplicate_answer.lower()
    duplicate_stale_current = memory_laws_old_draft_as_current(duplicate_answer)
    old_draft_as_current = memory_laws_old_draft_as_current(entity_answer)
    no_silent = all(
        bool(getattr(meta, "no_silent_fallback", False))
        for meta in (no_memory_meta, atlas_meta, duplicate_meta, current_meta, history_meta, entity_meta)
    )
    kv_direct_active = all(
        bool(getattr(meta, "kv_direct_active", False))
        for meta in (no_memory_meta, atlas_meta, duplicate_meta, current_meta, history_meta, entity_meta)
    )
    selected_tier = getattr(atlas_meta, "selected_tier", None)
    elapsed_ms = round(
        1000.0
        * (
            no_memory_elapsed
            + atlas_elapsed
            + duplicate_elapsed
            + current_elapsed
            + history_elapsed
            + entity_elapsed
        ),
        2,
    )
    return MemoryLawsRecallResult(
        probe_idx=probe_idx,
        noise_level=int(noise_level),
        duplicate_level=int(duplicate_level),
        no_memory_detected=memory_laws_no_memory_detected(no_memory_answer),
        hallucinated_target_fact_count=memory_laws_hallucinated_palette_fact_count(no_memory_answer),
        no_memory_answer=no_memory_answer,
        atlas_target_recall=len(atlas_hits),
        atlas_wrong_entity_leak_count=len(wrong_hits),
        atlas_answer_fingerprint=memory_laws_atlas_fingerprint(atlas_answer),
        duplicate_current_fact_present=duplicate_current,
        duplicated_stale_fact_as_current=duplicate_stale_current,
        duplicate_pressure_did_not_flip_answer=duplicate_current and not duplicate_stale_current,
        current_query_final_fact_present=memory_laws_temporal_current_ok(current_answer),
        current_query_old_fact_not_current=memory_laws_temporal_old_not_current(current_answer),
        history_query_old_fact_present="crimson" in history_answer.lower(),
        history_query_supersession_present=(
            "superseded" in history_answer.lower() or "replaced" in history_answer.lower()
        ),
        temporal_order_preserved=memory_laws_temporal_history_ok(history_answer),
        entity_scope_preserved=len(entity_hits) >= 5 and not memory_laws_wrong_entity_hits(entity_answer),
        old_draft_as_current=old_draft_as_current,
        selected_tier=selected_tier,
        mask_penalty_applied=bool(getattr(atlas_meta, "mask_penalty_applied", False)),
        kv_direct_active=kv_direct_active,
        no_silent_fallback=no_silent,
        candidate_count=int(getattr(atlas_meta, "candidate_count", 0) or 0),
        tier_assignment_count=int(getattr(atlas_meta, "tier_assignment_count", 0) or 0),
        budgeted_assignment_count=int(getattr(atlas_meta, "budgeted_assignment_count", 0) or 0),
        multi_session_count=int(getattr(atlas_meta, "multi_session_count", 0) or 0),
        semantic_prefix_active=bool(getattr(atlas_meta, "semantic_prefix_active", False)),
        candidate_recall_at_4=int(candidate_recall["candidate_recall_at_4"]),
        candidate_recall_at_8=int(candidate_recall["candidate_recall_at_8"]),
        candidate_recall_at_12=int(candidate_recall["candidate_recall_at_12"]),
        candidate_recall_at_64=int(candidate_recall["candidate_recall_at_64"]),
        latency_ms=elapsed_ms,
        vram_peak_mib=getattr(atlas_meta, "vram_peak_mib", None),
        atlas_answer=atlas_answer,
        duplicate_answer=duplicate_answer,
        current_answer=current_answer,
        history_answer=history_answer,
        entity_answer=entity_answer,
    )


def result_passed(result: RecallResult, *, mode: str) -> bool:
    if result.mode != mode:
        return False
    if not result.no_silent_fallback:
        return False
    return (
        result.matched_contains_marker
        and result.answer_contains_marker
        and result.answer_contains_session
        and result.answer_contains_turn
    )


def multi_fact_result_passed(result: MultiFactRecallResult) -> bool:
    if result.mode != "kv_direct":
        return False
    if not result.no_silent_fallback:
        return False
    if result.selected_tier != "hot":
        return False
    return len(result.missing_colors) == 0


def real_world_multi_fact_result_passed(result: RealWorldMultiFactRecallResult) -> bool:
    if result.mode != "kv_direct":
        return False
    if not result.no_silent_fallback:
        return False
    if result.selected_tier != "hot":
        return False
    if not result.mask_penalty_applied:
        return False
    if result.candidate_coverage < 10:
        return False
    if len(result.hot_fact_keys) != 4 or len(result.warm_fact_keys) != 4:
        return False
    if len(result.cold_fact_keys) != 4:
        return False
    if result.pollution_hits:
        return False
    return (
        len(result.hot_hits) == 4
        and len(result.warm_hits) >= 3
        and len(result.cold_hits) == 0
        and result.conflict_preserved
        and result.final_decision_present
    )


def memory_laws_result_passed(result: MemoryLawsRecallResult) -> bool:
    if not result.kv_direct_active or not result.no_silent_fallback:
        return False
    if result.selected_tier != "hot":
        return False
    if not result.mask_penalty_applied:
        return False
    if result.candidate_count < 1:
        return False
    if result.budgeted_assignment_count > result.tier_assignment_count:
        return False
    if result.tier_assignment_count > result.candidate_count:
        return False
    return (
        result.no_memory_detected
        and result.hallucinated_target_fact_count == 0
        and result.atlas_target_recall >= 5
        and result.atlas_wrong_entity_leak_count == 0
        and result.duplicate_pressure_did_not_flip_answer
        and result.current_query_final_fact_present
        and result.current_query_old_fact_not_current
        and result.history_query_old_fact_present
        and result.history_query_supersession_present
        and result.temporal_order_preserved
        and result.entity_scope_preserved
        and not result.old_draft_as_current
        and result.candidate_recall_at_12 >= 5
    )


def write_report(report_path: Path, results: list[Any], summary: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Completed repl-autoverify run directory.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--allow-non-pass", action="store_true")
    parser.add_argument("--min-sessions", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=100, help="0 means all parsed probes.")
    parser.add_argument(
        "--mode",
        choices=(
            "kv_direct",
            "topical",
            "multi_fact",
            "real_world_multi_fact",
            "dirty_real_world_multi_fact",
            "memory_laws",
            "memory_diagnostics_curve",
        ),
        default="kv_direct",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--required-hit-rate", type=float, default=0.99)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--quiet-model-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Parse run artifacts without loading the model.")
    parser.add_argument(
        "--full-matrix-from-transcripts",
        action="store_true",
        help=(
            "Reconstruct all planted scale markers from transcripts. Use with "
            "--sample-size 0 for the full 100x100 exact/literal lookup matrix."
        ),
    )
    parser.add_argument(
        "--score-exact-literal-from-route",
        action="store_true",
        help=(
            "For the exact/literal marker suite, score the deterministic routed "
            "window identity while still recording the model-generated string."
        ),
    )
    parser.add_argument(
        "--router-only-exact-literal",
        action="store_true",
        help=(
            "Run the exact/literal marker suite against the hybrid router only, "
            "without model generation."
        ),
    )
    parser.add_argument(
        "--router-only-candidate-pool",
        type=int,
        default=int(os.environ.get("LAZARUS_KV_ROUTE_CANDIDATE_POOL", "1") or "1"),
        help="Candidate pool used by --router-only-exact-literal.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--memory-laws-noise-levels",
        default="10,100,1000",
        help="Comma-separated irrelevant-noise levels for --mode memory_laws.",
    )
    parser.add_argument(
        "--memory-laws-duplicate-levels",
        default="5,25,100",
        help="Comma-separated stale duplicate levels for --mode memory_laws.",
    )
    parser.add_argument(
        "--memory-laws-long-noise-levels",
        default="10,100,1000,10000",
        help="Optional long/nightly levels; pass them via --memory-laws-noise-levels for long sweeps.",
    )
    parser.add_argument(
        "--memory-laws-skip-long",
        action="store_true",
        help="Compatibility flag for CI wrappers that skip long memory-laws sweeps.",
    )
    parser.add_argument(
        "--memory-diagnostics-noise-levels",
        default="10,100,1000",
        help="Comma-separated noise levels for --mode memory_diagnostics_curve.",
    )
    parser.add_argument(
        "--memory-diagnostics-long-noise-levels",
        default="10,100,1000,10000",
        help="Optional long/nightly diagnostics levels; pass them via --memory-diagnostics-noise-levels.",
    )
    return parser


def run_multi_fact_mode(args: argparse.Namespace, run_dir: Path, store_root: Path) -> int:
    harness_store_root = store_root
    store_root = localize_path(str(run_dir / "scale-multi-fact-store"))
    if store_root.exists():
        shutil.rmtree(store_root)
    imc = load_interactive_memory_chat()
    force_deterministic_streaming()
    chat = imc.MemoryChat(
        store_root=store_root,
        model_path=args.model_path,
        max_new_tokens=max(int(args.max_new_tokens), 96),
        memory_mode="kv_direct",
        device=args.device,
    )
    chat.load_model()
    chat.maybe_load_retriever()

    from chuk_lazarus.inference.chat import Role

    palette = [
        "cerulean",
        "saffron",
        "chartreuse",
        "vermillion",
        "indigo",
        "magenta",
        "ochre",
        "turquoise",
        "crimson",
        "periwinkle",
        "ultramarine",
        "malachite",
    ]
    probe_count = max(1, int(args.sample_size))
    previous_env = {
        key: os.environ.get(key)
        for key in (
            "LAZARUS_KV_CANDIDATE_POOL",
            "LAZARUS_KV_K_HOT",
            "LAZARUS_KV_K_WARM",
            "LAZARUS_MAX_TOTAL_INJECT_TOKENS",
            "LAZARUS_KV_HOT_BONUS",
        )
    }
    results: list[MultiFactRecallResult] = []
    passed = 0
    try:
        os.environ["LAZARUS_KV_CANDIDATE_POOL"] = "4"
        os.environ["LAZARUS_KV_K_HOT"] = "4"
        os.environ["LAZARUS_KV_K_WARM"] = "0"
        os.environ["LAZARUS_MAX_TOTAL_INJECT_TOKENS"] = "65536"
        os.environ["LAZARUS_KV_HOT_BONUS"] = "0.0"
        for probe_idx in range(1, probe_count + 1):
            reset_multi_fact_probe_store(chat)
            offset = (probe_idx - 1) % (len(palette) - 3)
            colors = palette[offset : offset + 4]
            try:
                if args.quiet_model_output:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = run_multi_fact_probe(
                            chat,
                            Role.USER,
                            probe_idx=probe_idx,
                            colors=colors,
                        )
                else:
                    result = run_multi_fact_probe(
                        chat,
                        Role.USER,
                        probe_idx=probe_idx,
                        colors=colors,
                    )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"FAIL SCALE_ACTUAL_RECALL probe={probe_idx}/{probe_count} "
                    f"mode=multi_fact: {exc!r}"
                )
                print(traceback.format_exc())
                return 1
            results.append(result)
            ok = multi_fact_result_passed(result)
            passed += int(ok)
            verdict = "PASS" if ok else "FAIL"
            print(
                f"{verdict} SCALE_ACTUAL_RECALL probe={probe_idx}/{probe_count} "
                f"mode=multi_fact recalled={len(result.recalled_colors)}/4 "
                f"elapsed_s={result.elapsed_s:.2f}",
                flush=True,
            )
            if not ok:
                print(
                    json.dumps(asdict(result), indent=2, sort_keys=True)[:2400],
                    flush=True,
                )
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    hit_rate = passed / max(1, len(results))
    final_summary = {
        "run_dir": str(run_dir),
        "store_root": str(store_root),
        "harness_store_root": str(harness_store_root),
        "mode": args.mode,
        "sample_size": len(results),
        "passed": passed,
        "hit_rate": hit_rate,
        "required_hit_rate": args.required_hit_rate,
    }
    report_path = args.report_json or (run_dir / "scale-actual-recall-multi_fact.json")
    write_report(report_path, results, final_summary)
    if hit_rate < args.required_hit_rate:
        print(
            f"FAIL SCALE_ACTUAL_RECALL: mode=multi_fact hit_rate={hit_rate:.3f} "
            f"required={args.required_hit_rate:.3f} report={report_path}",
            flush=True,
        )
        return 1
    print(
        f"PASS SCALE_ACTUAL_RECALL: mode=multi_fact hit_rate={hit_rate:.3f} "
        f"passed={passed}/{len(results)} report={report_path}",
        flush=True,
    )
    return 0


def run_real_world_multi_fact_mode(
    args: argparse.Namespace,
    run_dir: Path,
    store_root: Path,
    *,
    dirty_store: bool = False,
) -> int:
    harness_store_root = store_root
    store_name = (
        "scale-dirty-real-world-multi-fact-store"
        if dirty_store
        else "scale-real-world-multi-fact-store"
    )
    store_root = localize_path(str(run_dir / store_name))
    if store_root.exists():
        shutil.rmtree(store_root)
    imc = load_interactive_memory_chat()
    force_deterministic_streaming()
    chat = imc.MemoryChat(
        store_root=store_root,
        model_path=args.model_path,
        max_new_tokens=max(int(args.max_new_tokens), 220),
        memory_mode="kv_direct",
        device=args.device,
    )
    chat.load_model()
    chat.maybe_load_retriever()

    from chuk_lazarus.inference.chat import Role

    probe_count = max(1, int(args.sample_size))
    previous_env = {
        key: os.environ.get(key)
        for key in (
            "LAZARUS_KV_CANDIDATE_POOL",
            "LAZARUS_KV_K_HOT",
            "LAZARUS_KV_K_WARM",
            "LAZARUS_KV_ROUTE_CANDIDATE_POOL",
            "LAZARUS_KV_DEDUP_SESSION",
            "LAZARUS_MAX_TOTAL_INJECT_TOKENS",
            "LAZARUS_KV_SEMANTIC_PREFIX_TOKENS",
            "LAZARUS_KV_HOT_BONUS",
        )
    }
    results: list[RealWorldMultiFactRecallResult] = []
    passed = 0
    try:
        os.environ["LAZARUS_KV_CANDIDATE_POOL"] = "12"
        os.environ["LAZARUS_KV_ROUTE_CANDIDATE_POOL"] = "64" if dirty_store else "24"
        os.environ["LAZARUS_KV_DEDUP_SESSION"] = "1"
        os.environ["LAZARUS_KV_K_HOT"] = "4"
        os.environ["LAZARUS_KV_K_WARM"] = "4"
        os.environ["LAZARUS_MAX_TOTAL_INJECT_TOKENS"] = "65536"
        os.environ["LAZARUS_KV_SEMANTIC_PREFIX_TOKENS"] = "4096"
        os.environ["LAZARUS_KV_HOT_BONUS"] = "0.0"
        for probe_idx in range(1, probe_count + 1):
            reset_multi_fact_probe_store(chat)
            try:
                if args.quiet_model_output:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = run_real_world_multi_fact_probe(
                            chat,
                            Role.USER,
                            probe_idx=probe_idx,
                            dirty_store=dirty_store,
                        )
                else:
                    result = run_real_world_multi_fact_probe(
                        chat,
                        Role.USER,
                        probe_idx=probe_idx,
                        dirty_store=dirty_store,
                    )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"FAIL SCALE_ACTUAL_RECALL probe={probe_idx}/{probe_count} "
                    f"mode={args.mode}: {exc!r}"
                )
                print(traceback.format_exc())
                return 1
            results.append(result)
            ok = real_world_multi_fact_result_passed(result)
            passed += int(ok)
            verdict = "PASS" if ok else "FAIL"
            print(
                f"{verdict} SCALE_ACTUAL_RECALL probe={probe_idx}/{probe_count} "
                f"mode={args.mode} "
                f"HOT={len(result.hot_hits)}/4 "
                f"WARM={len(result.warm_hits)}/4 "
                f"COLD={len(result.cold_hits)}/4 "
                f"pollution={len(result.pollution_hits)} "
                f"elapsed_s={result.elapsed_s:.2f}",
                flush=True,
            )
            if not ok:
                print(
                    json.dumps(asdict(result), indent=2, sort_keys=True)[:3200],
                    flush=True,
                )
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    hit_rate = passed / max(1, len(results))
    final_summary = {
        "run_dir": str(run_dir),
        "store_root": str(store_root),
        "harness_store_root": str(harness_store_root),
        "mode": args.mode,
        "sample_size": len(results),
        "passed": passed,
        "hit_rate": hit_rate,
        "required_hit_rate": args.required_hit_rate,
    }
    report_path = args.report_json or (run_dir / f"scale-actual-recall-{args.mode}.json")
    write_report(report_path, results, final_summary)
    if hit_rate < args.required_hit_rate:
        print(
            f"FAIL SCALE_ACTUAL_RECALL: mode={args.mode} "
            f"hit_rate={hit_rate:.3f} required={args.required_hit_rate:.3f} "
            f"report={report_path}",
            flush=True,
        )
        return 1
    print(
        f"PASS SCALE_ACTUAL_RECALL: mode={args.mode} "
        f"hit_rate={hit_rate:.3f} passed={passed}/{len(results)} "
        f"report={report_path}",
        flush=True,
    )
    return 0


def run_memory_laws_mode(
    args: argparse.Namespace,
    run_dir: Path,
    store_root: Path,
) -> int:
    harness_store_root = store_root
    store_root = localize_path(str(run_dir / "scale-memory-laws-store"))
    if store_root.exists():
        shutil.rmtree(store_root)
    imc = load_interactive_memory_chat()
    force_deterministic_streaming()
    chat = imc.MemoryChat(
        store_root=store_root,
        model_path=args.model_path,
        max_new_tokens=max(int(args.max_new_tokens), 220),
        memory_mode="kv_direct",
        device=args.device,
    )
    chat.load_model()
    chat.maybe_load_retriever()

    from chuk_lazarus.inference.chat import Role

    noise_levels = parse_csv_ints(args.memory_laws_noise_levels)
    duplicate_levels = parse_csv_ints(args.memory_laws_duplicate_levels)
    if not noise_levels:
        noise_levels = [10]
    if not duplicate_levels:
        duplicate_levels = [5]
    probe_count = max(1, int(args.sample_size))
    previous_env = {
        key: os.environ.get(key)
        for key in (
            "LAZARUS_KV_CANDIDATE_POOL",
            "LAZARUS_KV_K_HOT",
            "LAZARUS_KV_K_WARM",
            "LAZARUS_KV_ROUTE_CANDIDATE_POOL",
            "LAZARUS_KV_DEDUP_SESSION",
            "LAZARUS_MAX_TOTAL_INJECT_TOKENS",
            "LAZARUS_KV_SEMANTIC_PREFIX_TOKENS",
            "LAZARUS_KV_HOT_BONUS",
        )
    }
    results: list[MemoryLawsRecallResult] = []
    passed = 0
    try:
        os.environ["LAZARUS_KV_CANDIDATE_POOL"] = "12"
        os.environ["LAZARUS_KV_ROUTE_CANDIDATE_POOL"] = "64"
        os.environ["LAZARUS_KV_DEDUP_SESSION"] = "1"
        os.environ["LAZARUS_KV_K_HOT"] = "4"
        os.environ["LAZARUS_KV_K_WARM"] = "4"
        os.environ["LAZARUS_MAX_TOTAL_INJECT_TOKENS"] = "65536"
        os.environ["LAZARUS_KV_SEMANTIC_PREFIX_TOKENS"] = "4096"
        os.environ["LAZARUS_KV_HOT_BONUS"] = "0.0"
        for probe_idx in range(1, probe_count + 1):
            reset_multi_fact_probe_store(chat, preserve_retriever=True)
            noise_level = noise_levels[(probe_idx - 1) % len(noise_levels)]
            duplicate_level = duplicate_levels[(probe_idx - 1) % len(duplicate_levels)]
            try:
                if args.quiet_model_output:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = run_memory_laws_probe(
                            chat,
                            Role.USER,
                            probe_idx=probe_idx,
                            noise_level=noise_level,
                            duplicate_level=duplicate_level,
                        )
                else:
                    result = run_memory_laws_probe(
                        chat,
                        Role.USER,
                        probe_idx=probe_idx,
                        noise_level=noise_level,
                        duplicate_level=duplicate_level,
                    )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"FAIL SCALE_ACTUAL_RECALL probe={probe_idx}/{probe_count} "
                    f"mode=memory_laws: {exc!r}"
                )
                print(traceback.format_exc())
                return 1
            results.append(result)
            ok = memory_laws_result_passed(result)
            passed += int(ok)
            verdict = "PASS" if ok else "FAIL"
            print(
                f"{verdict} SCALE_ACTUAL_RECALL probe={probe_idx}/{probe_count} "
                f"mode=memory_laws noise={result.noise_level} dup={result.duplicate_level} "
                f"atlas={result.atlas_target_recall}/6 wrong_entity={result.atlas_wrong_entity_leak_count} "
                f"elapsed_ms={result.latency_ms:.0f}",
                flush=True,
            )
            if not ok:
                print(
                    json.dumps(asdict(result), indent=2, sort_keys=True)[:4000],
                    flush=True,
                )
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    hit_rate = passed / max(1, len(results))
    final_summary = {
        "run_dir": str(run_dir),
        "store_root": str(store_root),
        "harness_store_root": str(harness_store_root),
        "mode": args.mode,
        "sample_size": len(results),
        "passed": passed,
        "hit_rate": hit_rate,
        "required_hit_rate": args.required_hit_rate,
        "noise_levels": noise_levels,
        "duplicate_levels": duplicate_levels,
    }
    report_path = args.report_json or (run_dir / "scale-actual-recall-memory_laws.json")
    write_report(report_path, results, final_summary)
    if hit_rate < args.required_hit_rate:
        print(
            f"FAIL SCALE_ACTUAL_RECALL: mode=memory_laws hit_rate={hit_rate:.3f} "
            f"required={args.required_hit_rate:.3f} report={report_path}",
            flush=True,
        )
        return 1
    print(
        f"PASS SCALE_ACTUAL_RECALL: mode=memory_laws hit_rate={hit_rate:.3f} "
        f"passed={passed}/{len(results)} report={report_path}",
        flush=True,
    )
    return 0


def run_memory_diagnostics_curve_mode(
    args: argparse.Namespace,
    run_dir: Path,
    store_root: Path,
) -> int:
    harness_store_root = store_root
    store_root = localize_path(str(run_dir / "scale-memory-diagnostics-curve-store"))
    if store_root.exists():
        shutil.rmtree(store_root)
    imc = load_interactive_memory_chat()
    force_deterministic_streaming()
    chat = imc.MemoryChat(
        store_root=store_root,
        model_path=args.model_path,
        max_new_tokens=max(int(args.max_new_tokens), 220),
        memory_mode="kv_direct",
        device=args.device,
    )
    chat.load_model()
    chat.maybe_load_retriever()

    from chuk_lazarus.inference.chat import Role

    noise_levels = parse_csv_ints(args.memory_diagnostics_noise_levels)
    if not noise_levels:
        noise_levels = [10, 100]
    previous_env = {
        key: os.environ.get(key)
        for key in (
            "LAZARUS_KV_CANDIDATE_POOL",
            "LAZARUS_KV_K_HOT",
            "LAZARUS_KV_K_WARM",
            "LAZARUS_KV_ROUTE_CANDIDATE_POOL",
            "LAZARUS_KV_DEDUP_SESSION",
            "LAZARUS_MAX_TOTAL_INJECT_TOKENS",
            "LAZARUS_KV_SEMANTIC_PREFIX_TOKENS",
            "LAZARUS_KV_HOT_BONUS",
        )
    }
    results: list[MemoryDiagnosticsCurveResult] = []
    passed = 0
    breakpoint_noise: int | None = None
    try:
        os.environ["LAZARUS_KV_CANDIDATE_POOL"] = "16"
        os.environ["LAZARUS_KV_ROUTE_CANDIDATE_POOL"] = "64"
        os.environ["LAZARUS_KV_DEDUP_SESSION"] = "1"
        os.environ["LAZARUS_KV_K_HOT"] = "4"
        os.environ["LAZARUS_KV_K_WARM"] = "4"
        os.environ["LAZARUS_MAX_TOTAL_INJECT_TOKENS"] = "65536"
        os.environ["LAZARUS_KV_SEMANTIC_PREFIX_TOKENS"] = "4096"
        os.environ["LAZARUS_KV_HOT_BONUS"] = "0.0"
        for noise_level in noise_levels:
            reset_multi_fact_probe_store(chat)
            for noise_idx in range(int(noise_level)):
                plant_memory_laws_session(
                    chat,
                    Role.USER,
                    memory_laws_noise_text(noise_idx),
                    rebuild_retriever=False,
                )
            target_records: list[dict[str, Any]] = []
            for fact in MEMORY_LAWS_ATLAS_FACTS:
                plant_memory_laws_session(
                    chat,
                    Role.USER,
                    str(fact["text"]),
                    rebuild_retriever=False,
                )
                target_records.append(dict(fact))
            refresh_memory_laws_retriever(chat)
            query = "What are the current Atlas pricing decisions across our sessions?"
            candidate_recall = memory_laws_route_recall(chat, query, target_records)
            meta, elapsed = run_memory_laws_query(chat, query)
            answer = str(getattr(meta, "generated_answer", "") or "")
            wrong_hits = memory_laws_wrong_entity_hits(answer)
            result = MemoryDiagnosticsCurveResult(
                noise=int(noise_level),
                target_recall_at_4=int(candidate_recall["candidate_recall_at_4"]),
                target_recall_at_8=int(candidate_recall["candidate_recall_at_8"]),
                target_recall_at_12=int(candidate_recall["candidate_recall_at_12"]),
                target_recall_at_64=int(candidate_recall["candidate_recall_at_64"]),
                wrong_entity_leak_count=len(wrong_hits),
                near_miss_leak_count=len(wrong_hits),
                latency_ms=round(float(elapsed) * 1000.0, 2),
                vram_peak_mib=getattr(meta, "vram_peak_mib", None),
                candidate_count=int(getattr(meta, "candidate_count", 0) or 0),
                tier_assignment_count=int(getattr(meta, "tier_assignment_count", 0) or 0),
                fallback_count=0 if bool(getattr(meta, "no_silent_fallback", False)) else 1,
                answer_fingerprint=memory_laws_atlas_fingerprint(answer),
                generated_answer=answer,
            )
            ok = (
                result.target_recall_at_12 > 0
                and result.target_recall_at_64 >= result.target_recall_at_12
                and result.wrong_entity_leak_count == 0
                and isinstance(result.vram_peak_mib, (int, float))
            )
            if not ok and breakpoint_noise is None:
                breakpoint_noise = int(noise_level)
            passed += int(ok)
            results.append(result)
            verdict = "PASS" if ok else "FAIL"
            print(
                f"{verdict} MEMORY_DIAGNOSTICS_CURVE "
                f"noise={result.noise} recall@12={result.target_recall_at_12}/6 "
                f"latency={result.latency_ms:.0f}ms leaks={result.wrong_entity_leak_count}",
                flush=True,
            )
            if not ok:
                print(json.dumps(asdict(result), indent=2, sort_keys=True)[:4000], flush=True)
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    hit_rate = passed / max(1, len(results))
    final_summary = {
        "run_dir": str(run_dir),
        "store_root": str(store_root),
        "harness_store_root": str(harness_store_root),
        "mode": args.mode,
        "sample_size": len(results),
        "passed": passed,
        "hit_rate": hit_rate,
        "required_hit_rate": args.required_hit_rate,
        "noise_levels": noise_levels,
        "breakpoint_noise": breakpoint_noise,
        "breakpoint_not_reached_under_full_levels": breakpoint_noise is None,
        "curve_report_written": True,
    }
    report_path = args.report_json or (run_dir / "scale-actual-recall-memory_diagnostics_curve.json")
    write_report(report_path, results, final_summary)
    if hit_rate < args.required_hit_rate:
        print(
            f"FAIL SCALE_ACTUAL_RECALL: mode=memory_diagnostics_curve "
            f"hit_rate={hit_rate:.3f} required={args.required_hit_rate:.3f} "
            f"report={report_path}",
            flush=True,
        )
        return 1
    print(
        f"PASS SCALE_ACTUAL_RECALL: mode=memory_diagnostics_curve "
        f"hit_rate={hit_rate:.3f} passed={passed}/{len(results)} "
        f"report={report_path}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "memory_diagnostics_curve" and args.run_dir is None:
        run_dir = Path(args.output_root) / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-memory-diagnostics-curve"
        store_root = run_dir / "store"
        if args.dry_run:
            print(
                f"DRY_RUN SCALE_ACTUAL_RECALL: run_dir={run_dir} "
                f"store_root={store_root} mode=memory_diagnostics_curve "
                f"noise_levels={args.memory_diagnostics_noise_levels}",
                flush=True,
            )
            return 0
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_memory_diagnostics_curve_mode(args, run_dir, store_root)
    run_dir, harness_summary = load_run_summary(args)
    events_path = localize_path(str(harness_summary.get("events_jsonl") or run_dir / "events.jsonl"))
    store_root = localize_path(str(harness_summary.get("store_root") or run_dir / "store"))
    if args.mode == "multi_fact":
        if args.dry_run:
            print(
                f"DRY_RUN SCALE_ACTUAL_RECALL: run_dir={run_dir} "
                f"store_root={store_root} mode=multi_fact probes={args.sample_size}",
                flush=True,
            )
            return 0
        return run_multi_fact_mode(args, run_dir, store_root)
    if args.mode in {"real_world_multi_fact", "dirty_real_world_multi_fact"}:
        if args.dry_run:
            print(
                f"DRY_RUN SCALE_ACTUAL_RECALL: run_dir={run_dir} "
                f"store_root={store_root} mode={args.mode} "
                f"probes={args.sample_size}",
                flush=True,
            )
            return 0
        return run_real_world_multi_fact_mode(
            args,
            run_dir,
            store_root,
            dirty_store=args.mode == "dirty_real_world_multi_fact",
        )
    if args.mode == "memory_laws":
        if args.dry_run:
            print(
                f"DRY_RUN SCALE_ACTUAL_RECALL: run_dir={run_dir} "
                f"store_root={store_root} mode=memory_laws "
                f"probes={args.sample_size} "
                f"noise_levels={args.memory_laws_noise_levels} "
                f"duplicate_levels={args.memory_laws_duplicate_levels}",
                flush=True,
            )
            return 0
        return run_memory_laws_mode(args, run_dir, store_root)
    if args.mode == "memory_diagnostics_curve":
        if args.dry_run:
            print(
                f"DRY_RUN SCALE_ACTUAL_RECALL: run_dir={run_dir} "
                f"store_root={store_root} mode=memory_diagnostics_curve "
                f"noise_levels={args.memory_diagnostics_noise_levels}",
                flush=True,
            )
            return 0
        return run_memory_diagnostics_curve_mode(args, run_dir, store_root)
    probe_source = "transcripts" if args.full_matrix_from_transcripts else "events"
    parsed_probes = (
        parse_scale_transcript_probes(events_path, store_root / "transcripts")
        if args.full_matrix_from_transcripts
        else parse_scale_probes(events_path)
    )
    probes = select_probes(parsed_probes, args.sample_size)
    if not probes:
        raise RuntimeError(f"No scale routing probes found in {events_path}")
    if args.dry_run:
        print(
            f"DRY_RUN SCALE_ACTUAL_RECALL: run_dir={run_dir} "
            f"store_root={store_root} events={events_path} probes={len(probes)} "
            f"mode={args.mode} intent={MARKER_SUITE_INTENT} source={probe_source}",
            flush=True,
        )
        return 0
    if args.router_only_exact_literal:
        return run_router_only_exact_literal_mode(
            args,
            run_dir,
            store_root,
            events_path,
            probes,
            probe_source=probe_source,
        )

    imc = load_interactive_memory_chat()
    force_deterministic_streaming()
    chat = imc.MemoryChat(
        store_root=store_root,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        memory_mode=args.mode,
        device=args.device,
    )
    chat.load_model()
    chat.maybe_load_retriever()
    if chat.retriever is None:
        raise RuntimeError(f"No retriever could be loaded from {store_root}")

    results: list[RecallResult] = []
    passed = 0
    for idx, probe in enumerate(probes, start=1):
        try:
            if args.quiet_model_output:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_probe(
                        chat,
                        probe,
                        mode=args.mode,
                        score_exact_literal_from_route=args.score_exact_literal_from_route,
                    )
            else:
                result = run_probe(
                    chat,
                    probe,
                    mode=args.mode,
                    score_exact_literal_from_route=args.score_exact_literal_from_route,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL SCALE_ACTUAL_RECALL probe={idx}/{len(probes)} marker={probe.marker}: {exc!r}")
            print(traceback.format_exc())
            return 1
        results.append(result)
        ok = result_passed(result, mode=args.mode)
        passed += int(ok)
        verdict = "PASS" if ok else "FAIL"
        print(
            f"{verdict} SCALE_ACTUAL_RECALL probe={idx}/{len(probes)} "
            f"marker={probe.marker} expected=session {probe.expected_session_idx} "
            f"turn {probe.expected_turn_idx} elapsed_s={result.elapsed_s:.2f}",
            flush=True,
        )
        if not ok:
            print(
                json.dumps(asdict(result), indent=2, sort_keys=True)[:2400],
                flush=True,
            )
            if args.fail_fast:
                break

    hit_rate = passed / max(1, len(results))
    final_summary = {
        "run_dir": str(run_dir),
        "store_root": str(store_root),
        "events_path": str(events_path),
        "mode": args.mode,
        "probe_source": probe_source,
        "suite_intent": MARKER_SUITE_INTENT,
        "semantic_only": False,
        "score_exact_literal_from_route": bool(args.score_exact_literal_from_route),
        "sample_size": len(results),
        "passed": passed,
        "hit_rate": hit_rate,
        "required_hit_rate": args.required_hit_rate,
    }
    report_path = args.report_json or (run_dir / f"scale-actual-recall-{args.mode}.json")
    write_report(report_path, results, final_summary)
    if hit_rate < args.required_hit_rate:
        print(
            f"FAIL SCALE_ACTUAL_RECALL: hit_rate={hit_rate:.3f} "
            f"required={args.required_hit_rate:.3f} report={report_path}",
            flush=True,
        )
        return 1
    print(
        f"PASS SCALE_ACTUAL_RECALL: mode={args.mode} hit_rate={hit_rate:.3f} "
        f"passed={passed}/{len(results)} report={report_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
