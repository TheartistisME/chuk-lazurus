#!/usr/bin/env python3
"""Actual-use recall verifier for a completed infinite-memory harness run.

This is a post-harness test: it does not plant new sessions. It reuses a
successful ``scripts/auto_verify_memory_repl.py`` run, samples the planted
100x100 scale markers from ``events.jsonl``, sends real MemoryChat recall
turns, and asserts the generated answer contains the routed marker plus the
expected session/turn identity.
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
    matched_window_text: str
    elapsed_s: float


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


def run_probe(chat: Any, probe: RecallProbe, *, mode: str) -> RecallResult:
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
    return RecallResult(
        marker=probe.marker,
        expected_session_idx=probe.expected_session_idx,
        expected_turn_idx=probe.expected_turn_idx,
        source_session=getattr(meta, "source_session", None),
        window_id=getattr(meta, "window_id", None),
        mode=getattr(meta, "mode", None),
        no_silent_fallback=bool(getattr(meta, "no_silent_fallback", False)),
        matched_contains_marker=probe.marker.lower() in matched.lower(),
        answer_contains_marker=probe.marker.lower() in answer.lower(),
        answer_contains_session=contains_session(answer, probe.expected_session_idx),
        answer_contains_turn=contains_turn(answer, probe.expected_turn_idx),
        generated_answer=answer,
        matched_window_text=matched[:700],
        elapsed_s=elapsed,
    )


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


def reset_multi_fact_probe_store(chat: Any) -> None:
    """Clear the verifier's disposable store between independent probes."""
    if getattr(chat, "indexer", None) is not None:
        with contextlib.suppress(Exception):
            chat.indexer.stop()
    chat.indexer = None
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    probes = select_probes(parse_scale_probes(events_path), args.sample_size)
    if not probes:
        raise RuntimeError(f"No scale routing probes found in {events_path}")
    if args.dry_run:
        print(
            f"DRY_RUN SCALE_ACTUAL_RECALL: run_dir={run_dir} "
            f"store_root={store_root} events={events_path} probes={len(probes)} "
            f"mode={args.mode}",
            flush=True,
        )
        return 0

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
                    result = run_probe(chat, probe, mode=args.mode)
            else:
                result = run_probe(chat, probe, mode=args.mode)
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

    hit_rate = passed / max(1, len(results))
    final_summary = {
        "run_dir": str(run_dir),
        "store_root": str(store_root),
        "events_path": str(events_path),
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
