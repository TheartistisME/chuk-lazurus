#!/usr/bin/env python3
"""Operator automation for the interactive memory REPL.

This script is the "make it prove itself" harness for Lazarus chat memory.
It drives the same ``MemoryChat`` surface used by
``scripts/interactive_memory_chat.py`` and exits 0 only when every scripted
assertion passes. All stdout/stderr, structured events, environment details,
and summary state are written under ``prod/validation/repl-autoverify`` so a
later agent can inspect the run without guessing what happened.

The heavy path is Linux/CUDA only. On Windows this script is useful for
``--list-checks`` and syntax validation, not for live proof.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import platform
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "prod" / "validation" / "repl-autoverify"
TARGET_LAYER = 29

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


CHECKS: tuple[tuple[str, str], ...] = (
    ("PREFLIGHT", "Linux/CUDA, Gemma-4 layer table, and REPL import are valid."),
    ("LIVE_SAVE", "A MemoryChat session can live-index turns and save a loadable store."),
    ("ROUTING_SCALE", "ASI routing finds planted facts across many saved sessions."),
    ("TOPICAL_RECALL", "Topical residual injection recalls a planted fact."),
    ("PROBE_NO_MUTATION", "/query, /exact, and /entity probes do not mutate chat state."),
    ("KV_DIRECT_RECALL", "KV-direct recall runs without silent fallback."),
    ("MULTI_FACT_RECALL", "KV-direct tier policy recalls HOT/WARM facts across sessions and mutes COLD."),
    (
        "REAL_WORLD_MULTI_FACT_RECALL",
        "Natural website color-scheme memories synthesize HOT/WARM facts and mute COLD.",
    ),
    (
        "DIRTY_STORE_REAL_WORLD_RECALL",
        "Dirty-store routing recalls current target facts without near-miss leakage.",
    ),
    ("TOKEN_BUDGET", "The token-budget governor binds under pressure."),
    ("VRAM_BOUNDED", "KV-direct peak VRAM stays below the configured ceiling."),
    ("FALLBACK_TRUTH", "Forced KV-direct failure emits truthful no_silent_fallback=False."),
    (
        "INFINITE_TURN_LATENCY",
        "Residual-bounded active KV keeps same-session turn latency flat.",
    ),
    ("MEMORY_OFF", "memory_mode=off disables retrieval and uses plain chat only."),
    ("CRASH_GATE", "Incomplete and in-flight stores stay invisible to readers."),
    ("PARALLEL_WRITES", "Two simultaneous MemoryChat saves do not cross-corrupt stores."),
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return repr(value)


class HarnessFailure(RuntimeError):
    def __init__(self, check: str, reason: str, **fields: Any) -> None:
        super().__init__(reason)
        self.check = check
        self.reason = reason
        self.fields = fields


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass
class CheckRecord:
    check: str
    status: str
    evidence: str
    elapsed_s: float
    fields: dict[str, Any] = field(default_factory=dict)


class AuditLog:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.summary_path = run_dir / "summary.json"
        self.transcript_path = run_dir / "transcript.log"
        self.records: list[CheckRecord] = []
        self._event_lock = threading.Lock()
        self.transcript_handle: Any | None = None

    def open(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_handle = self.transcript_path.open("a", encoding="utf-8")
        self.event("run.start", run_dir=str(self.run_dir), repo_root=str(REPO_ROOT))

    def close(self) -> None:
        self.event("run.close")
        if self.transcript_handle is not None:
            self.transcript_handle.close()
            self.transcript_handle = None

    def event(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(record, sort_keys=True, default=json_default)
        with self._event_lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def pass_check(self, check: str, evidence: str, elapsed_s: float, **fields: Any) -> None:
        record = CheckRecord(check, "PASS", evidence, elapsed_s, dict(fields))
        self.records.append(record)
        self.event("check.pass", **asdict(record))
        print(f"PASS {check}: {evidence} elapsed_s={elapsed_s:.2f}", flush=True)

    def fail_check(self, exc: HarnessFailure, elapsed_s: float) -> None:
        record = CheckRecord(
            exc.check,
            "FAIL",
            exc.reason,
            elapsed_s,
            dict(exc.fields),
        )
        self.records.append(record)
        self.event("check.fail", **asdict(record))
        print(
            f"FAIL {exc.check}: {exc.reason} fields="
            f"{json.dumps(exc.fields, sort_keys=True, default=json_default)}",
            flush=True,
        )

    def write_summary(self, status: str, **extra: Any) -> None:
        summary = {
            "status": status,
            "repo_root": str(REPO_ROOT),
            "run_dir": str(self.run_dir),
            "events_jsonl": str(self.events_path),
            "transcript_log": str(self.transcript_path),
            "checks": [asdict(record) for record in self.records],
            **extra,
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=json_default),
            encoding="utf-8",
        )


def require(check: str, condition: bool, reason: str, **fields: Any) -> None:
    if not condition:
        raise HarnessFailure(check, reason, **fields)


def run_git_metadata(run_dir: Path, log: AuditLog) -> None:
    metadata: dict[str, Any] = {}
    commands = {
        "rev_parse_head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status_short": ["git", "status", "--short", "--untracked-files=no"],
    }
    for key, cmd in commands.items():
        try:
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            metadata[key] = {
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        except Exception as exc:  # noqa: BLE001
            metadata[key] = {"error": repr(exc)}
    out = run_dir / "git-metadata.json"
    out.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    log.event("metadata.git", path=str(out), head=metadata.get("rev_parse_head", {}))


def write_environment_metadata(run_dir: Path, log: AuditLog, args: argparse.Namespace) -> None:
    interesting_env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("LAZARUS_")
        or key in {
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "PYTHONPATH",
            "VIRTUAL_ENV",
        }
    }
    metadata = {
        "argv": sys.argv,
        "args": vars(args),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "env": interesting_env,
    }
    out = run_dir / "environment.json"
    out.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    log.event("metadata.environment", path=str(out))


def load_interactive_memory_chat() -> Any:
    path = REPO_ROOT / "scripts" / "interactive_memory_chat.py"
    spec = importlib.util.spec_from_file_location("interactive_memory_chat", path)
    if spec is None or spec.loader is None:
        raise HarnessFailure("PREFLIGHT", f"Could not load import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["interactive_memory_chat"] = module
    spec.loader.exec_module(module)
    return module


def force_deterministic_streaming(log: AuditLog) -> None:
    """Patch chat streaming to greedy decoding for reproducible automation."""

    from transformers import TextIteratorStreamer

    import chuk_lazarus.chat_loop.cli as chat_cli

    def deterministic_stream_assistant_reply(
        *,
        model: Any,
        tokenizer: Any,
        history: Any,
        windower: Any,
        session: Any,
        turn: Any,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        on_chunk: Any = None,
        on_text_delta: Any = None,
    ) -> str:
        del temperature
        prompt = chat_cli.format_history(tokenizer, history, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "streamer": streamer,
        }
        errors: list[BaseException] = []

        def _run_generate() -> None:
            try:
                model.generate(**generation_kwargs)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()
        accumulated: list[str] = []
        for text_delta in streamer:
            if not text_delta:
                continue
            accumulated.append(text_delta)
            if on_text_delta is not None:
                on_text_delta(text_delta)
            boundaries = windower.feed_text(text_delta)
            for boundary in boundaries:
                session.append_chunk(turn, boundary)
                if on_chunk is not None:
                    decoded = tokenizer.decode(
                        windower._token_ids[
                            boundary.start_token_offset : boundary.end_token_offset
                        ],
                        skip_special_tokens=True,
                    )
                    on_chunk(boundary, decoded)

        tail = windower.flush()
        if tail is not None:
            session.append_chunk(turn, tail)
            if on_chunk is not None:
                decoded = tokenizer.decode(
                    windower._token_ids[
                        tail.start_token_offset : tail.end_token_offset
                    ],
                    skip_special_tokens=True,
                )
                on_chunk(tail, decoded)
        session.finish_turn(turn)
        thread.join()
        if errors:
            raise errors[0]
        return "".join(accumulated)

    chat_cli.stream_assistant_reply = deterministic_stream_assistant_reply
    log.event("streaming.patch", mode="deterministic_greedy")


def parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


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


DIRTY_STORE_NEAR_MISS_MEMORIES: tuple[dict[str, Any], ...] = (
    {
        "noise_class": "near_miss_color_project",
        "pollution_key": "acme_microsite_seasonal",
        "forbidden_terms": ("acme", "microsite", "seasonal"),
        "text": (
            "For the Acme microsite, sage and amber were only seasonal "
            "campaign colors and were not product website decisions."
        ),
    },
    {
        "noise_class": "same_words_wrong_project",
        "pollution_key": "mobile_onboarding_purple",
        "forbidden_terms": ("mobile onboarding", "purple"),
        "text": (
            "The mobile onboarding project rejected purple, but that was "
            "unrelated to the website color scheme."
        ),
    },
    {
        "noise_class": "same_words_wrong_project",
        "pollution_key": "investor_deck_graphite",
        "forbidden_terms": ("investor deck", "graphite"),
        "text": (
            "The old investor deck used graphite headings, not the product "
            "website."
        ),
    },
    {
        "noise_class": "contradictory_old_fact",
        "pollution_key": "landing_page_coral_kept",
        "forbidden_terms": ("different landing page", "coral"),
        "text": (
            "A different landing page kept coral buttons after accessibility "
            "review; do not apply that to the product website."
        ),
    },
    {
        "noise_class": "unrelated_domain",
        "pollution_key": "coffee_shop_beige",
        "forbidden_terms": ("coffee shop", "beige"),
        "text": (
            "The coffee shop brand used beige intentionally; that brand system "
            "is unrelated to the website project."
        ),
    },
    {
        "noise_class": "stale_decision",
        "pollution_key": "teal_return_closed",
        "forbidden_terms": ("teal", "return"),
        "text": (
            "Closed branch note: someone suggested teal should return as the "
            "main accent after sage felt muted. The branch was rejected."
        ),
    },
    {
        "noise_class": "stale_decision",
        "pollution_key": "amber_banners_old",
        "forbidden_terms": ("amber", "banner"),
        "text": (
            "Superseded component note: amber was once proposed for sitewide "
            "announcement banners before the CTA-only rule."
        ),
    },
    {
        "noise_class": "duplicate_memory",
        "pollution_key": "duplicate_teal_history",
        "forbidden_terms": (),
        "text": (
            "Duplicate design-history note: a teal hero looked polished in "
            "one mock before the later review called the page too cold."
        ),
    },
    {
        "noise_class": "irrelevant_long_session",
        "pollution_key": "launch_ops_padding",
        "forbidden_terms": (),
        "text": (
            "Launch operations archive: QA signoff, analytics mapping, support "
            "handoff, and deployment checklist notes. This long session talks "
            "about timelines and owners, not active palette decisions."
        ),
    },
)


DIRTY_STORE_GENERIC_NOISE_CLASSES: tuple[str, ...] = (
    "near_miss_color_project",
    "stale_decision",
    "duplicate_memory",
    "unrelated_domain",
    "same_words_wrong_project",
    "contradictory_old_fact",
    "irrelevant_long_session",
)


DIRTY_STORE_DOMAIN_PROBES: tuple[dict[str, Any], ...] = (
    {
        "domain": "pricing_decisions",
        "query": "What are the current Atlas pricing decisions across our sessions?",
        "target_context": "Atlas pricing decisions",
        "facts": (
            {
                "fact_key": "atlas_pro_price",
                "text": "Current Atlas pricing decisions across our sessions: the Pro tier is $29 per seat monthly.",
            },
            {
                "fact_key": "atlas_annual_discount",
                "text": "Current Atlas pricing decisions across our sessions: the annual discount is now 18%.",
            },
            {
                "fact_key": "atlas_trial",
                "text": "Current Atlas pricing decisions across our sessions: the free trial stays at 14 days.",
            },
            {
                "fact_key": "atlas_overage",
                "text": "Current Atlas pricing decisions across our sessions: usage overage is $0.08 per extra automation run.",
            },
            {
                "fact_key": "atlas_enterprise",
                "text": "Current Atlas pricing decisions across our sessions: Enterprise pricing remains custom quote.",
            },
            {
                "fact_key": "atlas_final",
                "text": (
                    "Final pricing decision for Atlas pricing decisions: Pro "
                    "$29 per seat, 18% annual discount, 14-day trial, $0.08 "
                    "overage, and Enterprise custom quote."
                ),
            },
        ),
        "noise": (
            "Nimbus pricing page kept Pro at $39 per seat and a 10% annual discount.",
            "Acme billing moved the trial to 30 days for a partner bundle.",
            "Legacy Atlas draft proposed $49 per seat before the current decision.",
            "Support packaging notes mention refund windows but no Atlas price change.",
            "Mobile upsell research liked a $19 starter plan for another product.",
            "Old investor model assumed usage overage was $0.12, then got retired.",
            "Coffee shop loyalty pricing uses stamp cards and is unrelated.",
            "Nimbus Enterprise switched to a public $499 plan, not custom quote.",
            "Archived annual-plan copy used 20% before Atlas settled on 18%.",
            "A sandbox checkout page tested a 7-day trial for QA only.",
            "Partner reseller notes include volume discounts for non-Atlas SKUs.",
            "The pricing-table CSS audit discussed sticky headers only.",
        ),
        "forbidden_terms": (
            ("nimbus", "$39"),
            ("30 days", "partner"),
            ("$49", "legacy"),
            ("$19", "starter"),
            ("$0.12", "retired"),
            ("$499", "nimbus"),
            ("20%", "archived"),
            ("7-day", "trial"),
        ),
    },
    {
        "domain": "bug_history",
        "query": "What is the Meridian checkout bug history and current final status?",
        "target_context": "Meridian checkout bug history",
        "facts": (
            {
                "fact_key": "meridian_duplicate_charge",
                "text": "Current Meridian checkout bug history across our sessions: the incident was a duplicate charge on checkout.",
            },
            {
                "fact_key": "meridian_safari_root",
                "text": "Current Meridian checkout bug history across our sessions: Safari address autofill replayed the payment intent.",
            },
            {
                "fact_key": "meridian_idempotency",
                "text": "Current Meridian checkout bug history across our sessions: the fix added an idempotency guard around payment confirmation.",
            },
            {
                "fact_key": "meridian_refund_backfill",
                "text": "Current Meridian checkout bug history across our sessions: the refund queue was backfilled for affected orders.",
            },
            {
                "fact_key": "meridian_regression_test",
                "text": "Current Meridian checkout bug history across our sessions: a regression test now covers the Safari autofill path.",
            },
            {
                "fact_key": "meridian_final_fixed",
                "text": "Final bug status for Meridian checkout bug history: fixed, with checkout anomaly monitoring still on.",
            },
        ),
        "noise": (
            "Orion checkout bug history: Firefox coupon paste duplicated discounts.",
            "The Meridian mobile app had a login spinner bug unrelated to checkout.",
            "Archived hypothesis blamed Stripe webhooks before Safari autofill was confirmed.",
            "Legacy refund report for Vega required manual CSV import.",
            "A different storefront fixed duplicate cart items with a debounce.",
            "Orion final status stayed monitoring-only, not fixed.",
            "Feature flag cleanup notes mention checkout CSS class names only.",
            "Meridian search bug history involved empty filters, not payment intents.",
            "A coffee subscription app backfilled refunds for address typo orders.",
            "Old QA notes suggested removing Apple Pay, then rejected that plan.",
            "The support macro bug sent duplicate emails from another workspace.",
            "Archived incident review for Vega added a webhook retry cap.",
        ),
        "forbidden_terms": (
            ("orion", "firefox"),
            ("login spinner", "mobile app"),
            ("stripe webhooks", "hypothesis"),
            ("vega", "csv"),
            ("duplicate cart", "debounce"),
            ("monitoring-only", "orion"),
            ("empty filters", "search"),
            ("apple pay", "removing"),
            ("duplicate emails", "support"),
        ),
    },
)


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


def dirty_store_near_miss_hits(answer: str) -> list[str]:
    lower = answer.lower()
    hits: list[str] = []
    for memory in DIRTY_STORE_NEAR_MISS_MEMORIES:
        terms = tuple(str(term).lower() for term in memory.get("forbidden_terms", ()))
        if terms and all(term in lower for term in terms):
            hits.append(str(memory["pollution_key"]))
    return hits


def dirty_domain_wrong_project_hits(
    answer: str,
    probe: dict[str, Any],
) -> list[str]:
    lower = answer.lower()
    hits: list[str] = []
    for terms in probe.get("forbidden_terms", ()):
        lowered = tuple(str(term).lower() for term in terms)
        if lowered and all(term in lower for term in lowered):
            hits.append("+".join(lowered))
    return hits


def dirty_store_stale_fact_marked_or_excluded(answer: str) -> bool:
    lower = answer.lower()
    stale_pairs = (
        ("teal", "return"),
        ("amber", "banner"),
        ("coral", "kept"),
        ("beige", "intentionally"),
    )
    return not any(a in lower and b in lower for a, b in stale_pairs)


def dirty_domain_fact_mentioned(domain: str, fact_key: str, answer: str) -> bool:
    lower = answer.lower()
    if domain == "pricing_decisions":
        if fact_key == "atlas_pro_price":
            return "$29" in lower and "pro" in lower
        if fact_key == "atlas_annual_discount":
            return "18%" in lower and "annual" in lower
        if fact_key == "atlas_trial":
            return "14" in lower and "trial" in lower
        if fact_key == "atlas_overage":
            return "$0.08" in lower and "overage" in lower
        if fact_key == "atlas_enterprise":
            return "enterprise" in lower and "custom" in lower
        if fact_key == "atlas_final":
            return "final" in lower and "$29" in lower and "18%" in lower
    if domain == "bug_history":
        if fact_key == "meridian_duplicate_charge":
            return "duplicate" in lower and "charge" in lower
        if fact_key == "meridian_safari_root":
            return "safari" in lower and "autofill" in lower
        if fact_key == "meridian_idempotency":
            return "idempotency" in lower or "idempotent" in lower
        if fact_key == "meridian_refund_backfill":
            return "refund" in lower and "backfill" in lower
        if fact_key == "meridian_regression_test":
            return "regression" in lower and "test" in lower
        if fact_key == "meridian_final_fixed":
            return "final" in lower and "fixed" in lower
    raise KeyError(f"unknown dirty domain fact: {domain}/{fact_key}")


def dirty_domain_final_decision_present(domain: str, answer: str) -> bool:
    lower = answer.lower()
    if domain == "pricing_decisions":
        return "final" in lower and "$29" in lower and "18%" in lower
    if domain == "bug_history":
        return "final" in lower and "fixed" in lower
    return False


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


class ReplAutomation:
    def __init__(self, args: argparse.Namespace, log: AuditLog) -> None:
        self.args = args
        self.log = log
        self.imc: Any | None = None
        self.chat: Any | None = None
        self.role_cls: Any | None = None
        self.store_root = args.store_root or (log.run_dir / "store")
        self.primary_marker = secrets.token_hex(8)
        self.color_marker = secrets.token_hex(8)
        self.scale_markers: list[dict[str, Any]] = []
        self.real_world_records: list[dict[str, Any]] = []
        self.dirty_real_world_records: list[dict[str, Any]] = []
        self.dirty_noise_records: list[dict[str, Any]] = []
        self.multi_fact_records: list[dict[str, Any]] = []
        self.primary_clause: dict[str, Any] | None = None
        self.budget_observations: list[dict[str, Any]] = []
        self.log.event(
            "markers.seeded",
            policy="per-fact random hex; no shared run-id substring",
            primary_marker=self.primary_marker,
            color_marker=self.color_marker,
        )

    def run(self) -> None:
        self.run_check("PREFLIGHT", self.preflight)
        self.run_check("LIVE_SAVE", self.live_save)
        self.run_check("ROUTING_SCALE", self.routing_scale)
        self.run_check("TOPICAL_RECALL", self.topical_recall)
        self.run_check("PROBE_NO_MUTATION", self.probe_no_mutation)
        self.run_check("KV_DIRECT_RECALL", self.kv_direct_recall)
        self.run_check("MULTI_FACT_RECALL", self.multi_fact_recall)
        self.run_check(
            "REAL_WORLD_MULTI_FACT_RECALL",
            self.real_world_multi_fact_recall,
        )
        self.run_check(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            self.dirty_store_real_world_recall,
        )
        self.run_check("TOKEN_BUDGET", self.token_budget)
        self.run_check("VRAM_BOUNDED", self.vram_bounded)
        self.run_check("FALLBACK_TRUTH", self.fallback_truth)
        self.run_check("INFINITE_TURN_LATENCY", self.infinite_turn_latency)
        self.run_check("MEMORY_OFF", self.memory_off)
        self.run_check("CRASH_GATE", self.crash_gate)
        self.run_check("PARALLEL_WRITES", self.parallel_writes)

    def run_check(self, name: str, fn: Any) -> None:
        started = time.time()
        try:
            evidence = fn()
        except HarnessFailure as exc:
            self.log.fail_check(exc, time.time() - started)
            raise
        except Exception as exc:  # noqa: BLE001
            wrapped = HarnessFailure(
                name,
                f"Unhandled exception: {exc!r}",
                traceback=traceback.format_exc(),
            )
            self.log.fail_check(wrapped, time.time() - started)
            raise wrapped from exc
        self.log.pass_check(name, evidence, time.time() - started)

    def preflight(self) -> str:
        if self.args.list_checks:
            return "list-only"
        if not self.args.skip_env_check:
            require(
                "PREFLIGHT",
                platform.system() == "Linux",
                "automation must run on Linux/CUDA",
                platform=platform.system(),
            )
        import torch

        if not self.args.skip_env_check:
            require(
                "PREFLIGHT",
                torch.cuda.is_available(),
                "torch.cuda.is_available() is False",
            )
            total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
            require(
                "PREFLIGHT",
                total_mib >= self.args.min_vram_mib,
                "GPU VRAM below configured floor",
                total_mib=total_mib,
                min_vram_mib=self.args.min_vram_mib,
            )
            self.log.event(
                "cuda.device",
                name=torch.cuda.get_device_name(0),
                total_mib=round(total_mib, 2),
            )

        from chuk_lazarus.inference.context.knowledge.gemma4_e2b_it_layers import (
            GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS,
        )

        require(
            "PREFLIGHT",
            TARGET_LAYER in GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS,
            "Gemma-4 target layer 29 is not global attention",
            global_layers=sorted(GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS),
        )
        self.imc = load_interactive_memory_chat()
        from chuk_lazarus.inference.chat import Role

        self.role_cls = Role
        force_deterministic_streaming(self.log)
        run_git_metadata(self.log.run_dir, self.log)
        write_environment_metadata(self.log.run_dir, self.log, self.args)
        return "Linux/CUDA preflight and MemoryChat import passed"

    def make_chat(
        self,
        *,
        memory_mode: str,
        generation_engine: str = "standard",
        hot_budget_mib: int | None = None,
        session_cache_size: int | None = None,
        max_new_tokens: int | None = None,
    ) -> Any:
        assert self.imc is not None
        chat = self.imc.MemoryChat(
            store_root=self.store_root,
            model_path=self.args.model_path,
            max_new_tokens=(
                self.args.max_new_tokens
                if max_new_tokens is None
                else int(max_new_tokens)
            ),
            memory_mode=memory_mode,
            device=self.args.device,
            generation_engine=generation_engine,
            hot_budget_mib=hot_budget_mib,
            session_cache_size=session_cache_size,
        )
        return chat

    def load_main_chat(self) -> Any:
        if self.chat is None:
            chat = self.make_chat(memory_mode="topical")
            chat.load_model()
            chat.maybe_load_retriever()
            self.chat = chat
        return self.chat

    def live_save(self) -> str:
        chat = self.load_main_chat()
        chat.memory_mode = "topical"
        chat.start_new_session()
        session_id = chat.session.session_id
        meta = chat.plain_chat_turn(
            "Reply with exactly this sentence and nothing else: live save ready."
        )
        self.log.event(
            "live_save.turn",
            session_id=session_id,
            prompt="live save ready",
            generated_answer=getattr(meta, "generated_answer", ""),
            generated_tokens=getattr(meta, "generated_tokens", None),
        )
        planted_facts = [
            (
                self.primary_marker,
                "My dog is named Banjo. If asked later, answer Banjo.",
            ),
            (
                self.color_marker,
                "My favorite color is teal. If asked later, answer teal.",
            ),
        ]
        assert self.role_cls is not None
        for marker, fact_text in planted_facts:
            text = (
                f"{marker}. Unique recall key {marker}. "
                f"{fact_text} The retrieval key is {marker}."
            )
            self.direct_turn(chat, self.role_cls.USER, text)
            self.log.event(
                "live_save.direct_fact",
                session_id=session_id,
                marker=marker,
                text=text,
            )
        require(
            "LIVE_SAVE",
            chat.save_current_session(rebuild_retriever=True),
            "save_current_session returned False for planted REPL session",
        )
        manifest = chat.checkpoints_root / session_id / "torch_store" / "manifest.json"
        save_state = chat.checkpoints_root / session_id / "save-state.json"
        boundaries = chat.checkpoints_root / session_id / "torch_store" / "boundaries"
        require("LIVE_SAVE", manifest.exists(), "manifest.json missing", path=manifest)
        require("LIVE_SAVE", save_state.exists(), "save-state.json missing", path=save_state)
        boundary_count = len(list(boundaries.glob("window_*.npy")))
        require(
            "LIVE_SAVE",
            boundary_count > 0,
            "no boundary residual files were written",
            boundaries=boundaries,
        )
        self.primary_clause = self.find_clause_for_marker(session_id, self.primary_marker)
        require(
            "LIVE_SAVE",
            self.primary_clause is not None,
            "primary marker was not found in emitted AUS3000 clauses",
            marker=self.primary_marker,
        )
        return f"saved session {session_id} with {boundary_count} boundary file(s)"

    def direct_turn(self, chat: Any, role: Any, text: str) -> None:
        turn = chat.session.begin_turn(role, text)
        chat.session.finish_turn(turn)
        chat._capture_turn_text_live(turn)

    def plant_direct_session(self, session_idx: int, turns_per_session: int) -> str:
        chat = self.load_main_chat()
        assert self.role_cls is not None
        chat.memory_mode = "off"
        chat.start_new_session()
        session_id = chat.session.session_id
        for turn_idx in range(turns_per_session):
            marker = secrets.token_hex(8)
            text = (
                f"{marker}. Unique route key {marker}. "
                f"This planted scale memory belongs to session {session_idx} "
                f"turn {turn_idx}. The retrieval key is {marker}."
            )
            self.direct_turn(chat, self.role_cls.USER, text)
            if turn_idx == 0:
                self.scale_markers.append(
                    {
                        "marker": marker,
                        "session_idx": session_idx,
                        "turn_idx": turn_idx,
                        "session_id": session_id,
                    }
                )
        chat._mark_dirty()
        require(
            "ROUTING_SCALE",
            chat.save_current_session(rebuild_retriever=True),
            "save_current_session returned False for scale session",
            session_idx=session_idx,
            session_id=session_id,
        )
        return session_id

    def plant_budget_pressure_session(self, turns: int = 8) -> dict[str, Any]:
        """Create one high-volume session so TOKEN_BUDGET overflows deterministically."""
        chat = self.load_main_chat()
        assert self.role_cls is not None
        chat.memory_mode = "off"
        chat.start_new_session()
        session_id = chat.session.session_id
        marker = secrets.token_hex(8)
        for turn_idx in range(turns):
            text = (
                f"{marker}. Budget pressure route key {marker}. "
                f"This is budget pressure window {turn_idx}. "
                f"Use retrieval key {marker} for this whole session."
            )
            self.direct_turn(chat, self.role_cls.USER, text)
        chat._mark_dirty()
        require(
            "TOKEN_BUDGET",
            chat.save_current_session(rebuild_retriever=True),
            "save_current_session returned False for budget pressure session",
            session_id=session_id,
            marker=marker,
            turns=turns,
        )
        self.log.event(
            "token_budget.fixture",
            session_id=session_id,
            marker=marker,
            turns=turns,
        )
        return {"session_id": session_id, "marker": marker, "turns": turns}

    def routing_scale(self) -> str:
        chat = self.load_main_chat()
        self.scale_markers.clear()
        for session_idx in range(self.args.sessions):
            session_id = self.plant_direct_session(
                session_idx,
                self.args.turns_per_session,
            )
            self.log.event(
                "scale.session_saved",
                session_idx=session_idx,
                session_id=session_id,
                turns_per_session=self.args.turns_per_session,
            )
        require("ROUTING_SCALE", chat.retriever is not None, "retriever missing")

        from chuk_lazarus.session_retrieval import asi_route_candidates
        from chuk_lazarus.session_retrieval.enumeration import load_store

        hits = 0
        probes = self.scale_markers
        for marker_record in probes:
            query = str(marker_record["marker"])
            candidates = asi_route_candidates(
                chat.retriever.handles,
                query,
                chat.retriever.tokenizer,
                candidate_pool=max(parse_csv_ints(self.args.candidate_pools)),
            )
            require(
                "ROUTING_SCALE",
                bool(candidates),
                "ASI router returned no candidates",
                marker=marker_record["marker"],
            )
            top = candidates[0]
            store = load_store(top.handle)
            window_text = store.get_window_text(int(top.window_id), chat.tokenizer)
            top_candidates: list[dict[str, Any]] = []
            for rank, candidate in enumerate(candidates[:5], start=1):
                candidate_store = load_store(candidate.handle)
                candidate_text = candidate_store.get_window_text(
                    int(candidate.window_id),
                    chat.tokenizer,
                )
                top_candidates.append(
                    {
                        "rank": rank,
                        "session_id": candidate.handle.session_id,
                        "window_id": int(candidate.window_id),
                        "ucb1_score": float(candidate.ucb1_score),
                        "raw_router_score": float(candidate.raw_router_score),
                        "contains_marker": (
                            marker_record["marker"].lower()
                            in candidate_text.lower()
                        ),
                        "window_text_head": candidate_text[:180],
                    }
                )
            hit = (
                top.handle.session_id == marker_record["session_id"]
                and marker_record["marker"].lower() in window_text.lower()
            )
            hits += int(hit)
            cold_start_note = (
                "ucb1_score=Infinity is the router's cold-start exploration "
                "tie, not exact-match confidence"
                if math.isinf(float(top.ucb1_score))
                else ""
            )
            self.log.event(
                "routing.probe",
                marker=marker_record["marker"],
                expected_session=marker_record["session_id"],
                top_session=top.handle.session_id,
                top_window_id=int(top.window_id),
                top_score=float(top.ucb1_score),
                raw_router_score=float(top.raw_router_score),
                cold_start_note=cold_start_note,
                top_candidates=top_candidates,
                hit=hit,
                window_text_head=window_text[:240],
            )
        hit_rate = hits / max(1, len(probes))
        require(
            "ROUTING_SCALE",
            hit_rate >= self.args.routing_hit_threshold,
            "routing hit-rate below threshold",
            hit_rate=hit_rate,
            threshold=self.args.routing_hit_threshold,
            hits=hits,
            probes=len(probes),
        )
        return f"routing@1 hit_rate={hit_rate:.3f} over {len(probes)} planted sessions"

    def topical_recall(self) -> str:
        chat = self.load_main_chat()
        chat.memory_mode = "topical"
        chat.start_new_session()
        query = (
            f"What is my dog's name from memory marker {self.primary_marker}? "
            "Answer with the name only."
        )
        meta = chat.recall_chat_turn(query)
        strict = dict(getattr(meta, "strict_assertions", {}) or {})
        answer = str(getattr(meta, "generated_answer", "") or "")
        matched = str(getattr(meta, "matched_window_text", "") or "")
        require("TOPICAL_RECALL", meta.mode == "topical", "topical recall fell back", mode=meta.mode)
        require(
            "TOPICAL_RECALL",
            bool(getattr(meta, "no_silent_fallback", False)),
            "topical recall reported silent fallback",
            meta=self.meta_to_dict(meta),
        )
        require(
            "TOPICAL_RECALL",
            strict and all(bool(v) for v in strict.values()),
            "one or more strict assertions failed",
            strict_assertions=strict,
        )
        require(
            "TOPICAL_RECALL",
            self.primary_marker.lower() in matched.lower(),
            "routed window did not contain primary marker",
            matched_window_text=matched[:500],
        )
        require(
            "TOPICAL_RECALL",
            "banjo" in answer.lower(),
            "generated answer did not contain expected fact",
            answer=answer,
        )
        self.log.event("topical.meta", **self.meta_to_dict(meta))
        return "topical residual injection recalled Banjo with all strict assertions true"

    def probe_no_mutation(self) -> str:
        chat = self.load_main_chat()
        require("PROBE_NO_MUTATION", self.primary_clause is not None, "missing primary clause")
        assert self.primary_clause is not None
        before_turns = len(chat.session.turns)
        before_history = len(chat.history.messages)
        probes = [
            ("topical", f"Find {self.primary_marker}"),
            ("exact", self.primary_clause["clause_id"]),
            ("entity_mention", f"Banjo {self.primary_marker}"),
        ]
        for mode, query in probes:
            chat.probe(mode, query)
            after_turns = len(chat.session.turns)
            after_history = len(chat.history.messages)
            require(
                "PROBE_NO_MUTATION",
                after_turns == before_turns and after_history == before_history,
                "probe mutated chat state",
                mode=mode,
                query=query,
                before_turns=before_turns,
                after_turns=after_turns,
                before_history=before_history,
                after_history=after_history,
            )
            self.log.event("probe.no_mutation", mode=mode, query=query)
        return "topical/exact/entity probes left session and history unchanged"

    def kv_direct_recall(self) -> str:
        chat = self.load_main_chat()
        chat.memory_mode = "kv_direct"
        chat.start_new_session()
        query = (
            f"Use memory marker {self.primary_marker}. What is my dog's name? "
            "Answer with Banjo."
        )
        os.environ["LAZARUS_KV_CANDIDATE_POOL"] = str(
            max(parse_csv_ints(self.args.candidate_pools))
        )
        meta = chat.kv_query_turn(query)
        self.assert_kv_meta("KV_DIRECT_RECALL", meta)
        answer = str(getattr(meta, "generated_answer", "") or "")
        require(
            "KV_DIRECT_RECALL",
            "banjo" in answer.lower() or self.primary_marker.lower() in answer.lower(),
            "KV-direct answer did not contain expected memory",
            answer=answer,
            meta=self.meta_to_dict(meta),
        )
        self.log.event("kv_direct.meta", **self.meta_to_dict(meta))
        return "KV-direct recall active with no silent fallback"

    def plant_real_world_color_session(
        self,
        *,
        fact_idx: int,
        fact: dict[str, str],
    ) -> dict[str, Any]:
        chat = self.load_main_chat()
        assert self.role_cls is not None
        chat.memory_mode = "off"
        chat.start_new_session()
        session_id = chat.session.session_id
        self.direct_turn(chat, self.role_cls.USER, fact["text"])
        self.direct_turn(
            chat,
            self.role_cls.USER,
            (
                f"Neutral project note {secrets.token_hex(8)}. "
                "This follow-up is ordinary session padding and does not add "
                "any project decision."
            ),
        )
        chat._mark_dirty()
        require(
            "REAL_WORLD_MULTI_FACT_RECALL",
            chat.save_current_session(rebuild_retriever=True),
            "save_current_session returned False for real-world color session",
            fact_idx=fact_idx,
            session_id=session_id,
            fact_key=fact["fact_key"],
        )
        record = {
            "fact_idx": int(fact_idx),
            "session_id": session_id,
            "fact_key": fact["fact_key"],
            "match_phrase": fact["match_phrase"],
            "text": fact["text"],
        }
        self.log.event("real_world_multi_fact.session_saved", **record)
        return record

    def real_world_multi_fact_recall(self) -> str:
        chat = self.load_main_chat()
        require(
            "REAL_WORLD_MULTI_FACT_RECALL",
            chat.retriever is not None,
            "retriever missing",
        )
        self.real_world_records = [
            self.plant_real_world_color_session(fact_idx=idx, fact=dict(fact))
            for idx, fact in enumerate(REAL_WORLD_COLOR_MEMORIES)
        ]

        from chuk_lazarus.session_retrieval import asi_route_candidates, assign_tiers
        from chuk_lazarus.session_retrieval.enumeration import load_store

        query = (
            "Tell me everything we discussed about the website's color scheme "
            "across all our sessions."
        )
        candidates = asi_route_candidates(
            chat.retriever.handles,
            query,
            chat.retriever.tokenizer,
            candidate_pool=64,
        )
        candidates = dedupe_candidates_by_session(candidates)
        records_by_phrase = {
            str(record["match_phrase"]).lower(): record
            for record in self.real_world_records
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
        seen_fact_keys = {str(record["fact_key"]) for record in candidate_records}
        require(
            "REAL_WORLD_MULTI_FACT_RECALL",
            len(seen_fact_keys) >= 10,
            "ASI candidate_pool did not cover at least 10 of 12 natural memories",
            seen=sorted(seen_fact_keys),
            expected=sorted(record["fact_key"] for record in self.real_world_records),
        )

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
        require(
            "REAL_WORLD_MULTI_FACT_RECALL",
            all(len(tier_records[tier]) == 4 for tier in ("hot", "warm", "cold")),
            "top 12 tier split was not 4/4/4 for natural color memories",
            tier_records=tier_records,
        )

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
        previous_max_new_tokens = int(chat.max_new_tokens)
        try:
            os.environ["LAZARUS_KV_CANDIDATE_POOL"] = "12"
            os.environ["LAZARUS_KV_ROUTE_CANDIDATE_POOL"] = "64"
            os.environ["LAZARUS_KV_DEDUP_SESSION"] = "1"
            os.environ["LAZARUS_KV_K_HOT"] = "4"
            os.environ["LAZARUS_KV_K_WARM"] = "4"
            os.environ["LAZARUS_MAX_TOTAL_INJECT_TOKENS"] = "65536"
            os.environ["LAZARUS_KV_SEMANTIC_PREFIX_TOKENS"] = "4096"
            os.environ["LAZARUS_KV_HOT_BONUS"] = "0.0"
            chat.max_new_tokens = max(previous_max_new_tokens, 220)
            chat.memory_mode = "kv_direct"
            chat.start_new_session()
            meta = chat.kv_query_turn(query)
        finally:
            chat.max_new_tokens = previous_max_new_tokens
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assert_kv_meta("REAL_WORLD_MULTI_FACT_RECALL", meta)
        answer = str(getattr(meta, "generated_answer", "") or "")
        hot_hits = [
            record for record in tier_records["hot"]
            if real_world_fact_mentioned(str(record["fact_key"]), answer)
        ]
        warm_hits = [
            record for record in tier_records["warm"]
            if real_world_fact_mentioned(str(record["fact_key"]), answer)
        ]
        selected_fact_keys = {
            str(record["fact_key"])
            for tier in ("hot", "warm")
            for record in tier_records[tier]
        }
        cold_only_records = [
            record for record in tier_records["cold"]
            if not real_world_fact_entailed_by_selected(
                str(record["fact_key"]),
                selected_fact_keys,
            )
        ]
        cold_hits = [
            record for record in cold_only_records
            if real_world_fact_mentioned(str(record["fact_key"]), answer)
        ]
        conflict_preserved = real_world_conflict_preserved(answer)
        final_decision_present = real_world_final_decision_present(answer)
        require(
            "REAL_WORLD_MULTI_FACT_RECALL",
            len(hot_hits) == 4,
            "generated answer missed one or more HOT natural memories",
            hot_expected=tier_records["hot"],
            hot_hits=hot_hits,
            answer=answer,
            meta=self.meta_to_dict(meta),
        )
        require(
            "REAL_WORLD_MULTI_FACT_RECALL",
            len(warm_hits) >= 3,
            "generated answer recalled too few WARM natural memories",
            warm_expected=tier_records["warm"],
            warm_hits=warm_hits,
            answer=answer,
            meta=self.meta_to_dict(meta),
        )
        require(
            "REAL_WORLD_MULTI_FACT_RECALL",
            len(cold_hits) == 0,
            "generated answer mentioned COLD-only natural memories",
            cold_expected=tier_records["cold"],
            cold_only_expected=cold_only_records,
            cold_hits=cold_hits,
            answer=answer,
            meta=self.meta_to_dict(meta),
        )
        require(
            "REAL_WORLD_MULTI_FACT_RECALL",
            conflict_preserved,
            "generated answer did not preserve a color revision",
            answer=answer,
            tier_records=tier_records,
        )
        require(
            "REAL_WORLD_MULTI_FACT_RECALL",
            final_decision_present,
            "generated answer did not identify a final/current decision",
            answer=answer,
            tier_records=tier_records,
        )
        require(
            "REAL_WORLD_MULTI_FACT_RECALL",
            bool(getattr(meta, "mask_penalty_applied", False)),
            "axis-6 did not report mask_penalty_applied=True",
            meta=self.meta_to_dict(meta),
        )
        require(
            "REAL_WORLD_MULTI_FACT_RECALL",
            getattr(meta, "selected_tier", None) == "hot",
            "axis-6 did not report selected_tier=hot",
            meta=self.meta_to_dict(meta),
        )
        self.log.event(
            "real_world_multi_fact.meta",
            answer=answer,
            hot_hits=len(hot_hits),
            warm_hits=len(warm_hits),
            cold_hits=len(cold_hits),
            conflict_preserved=conflict_preserved,
            final_decision_present=final_decision_present,
            tier_records=tier_records,
            meta=self.meta_to_dict(meta),
        )
        return (
            "natural multi-fact recall HOT=4/4 WARM="
            f"{len(warm_hits)}/4 COLD={len(cold_hits)}/4 "
            f"conflict_preserved={conflict_preserved} "
            f"final_decision_present={final_decision_present} "
            f"selected_tier={getattr(meta, 'selected_tier', None)}"
        )

    def refresh_retriever_after_batch(self, check: str, chat: Any) -> None:
        if chat.retriever is None:
            chat.maybe_load_retriever()
        else:
            chat.retriever.refresh_handles(chat.checkpoints_root)
        require(check, chat.retriever is not None, "retriever missing after batch refresh")

    def dirty_store_noise_text(self, noise_idx: int) -> tuple[str, str, str]:
        if noise_idx < len(DIRTY_STORE_NEAR_MISS_MEMORIES):
            memory = DIRTY_STORE_NEAR_MISS_MEMORIES[noise_idx]
            return (
                str(memory["noise_class"]),
                str(memory["pollution_key"]),
                str(memory["text"]),
            )
        noise_class = DIRTY_STORE_GENERIC_NOISE_CLASSES[
            noise_idx % len(DIRTY_STORE_GENERIC_NOISE_CLASSES)
        ]
        project = f"archive-{noise_idx:04d}"
        if noise_class == "near_miss_color_project":
            text = (
                f"For {project}, the landing page color notes mention sage, amber, "
                "graphite, and warm white as campaign exploration terms, but not "
                "as product website color-scheme decisions."
            )
        elif noise_class == "stale_decision":
            text = (
                f"Archived design branch {project}: a teal hero, gold secondary "
                "buttons, and beige backgrounds were proposed before being closed."
            )
        elif noise_class == "duplicate_memory":
            text = (
                f"Duplicate archive {project}: pricing, launch, QA, and support "
                "owners were copied between meeting notes."
            )
        elif noise_class == "same_words_wrong_project":
            text = (
                f"Wrong-project note {project}: mobile onboarding, the investor "
                "deck, and an Acme microsite reused words like purple, graphite, "
                "sage, amber, and footer."
            )
        elif noise_class == "contradictory_old_fact":
            text = (
                f"Contradictory old fact {project}: an unrelated landing page kept "
                "coral buttons and a navy hero after a separate review."
            )
        elif noise_class == "irrelevant_long_session":
            text = (
                f"Long archive session {project}: roadmap grooming, analytics "
                "events, support macros, bug triage, launch dependencies, owner "
                "handoffs, and customer-call summaries with no current palette "
                "decision."
            )
        else:
            text = (
                f"Unrelated domain archive {project}: customer preferences, bug "
                "history, feature requirements, pricing notes, and architecture "
                "tradeoffs for other projects."
            )
        return noise_class, f"{noise_class}_{noise_idx:04d}", text

    def plant_dirty_store_noise_sessions(
        self,
        *,
        count: int,
        start_idx: int = 0,
    ) -> list[dict[str, Any]]:
        chat = self.load_main_chat()
        assert self.role_cls is not None
        records: list[dict[str, Any]] = []
        for offset in range(int(count)):
            noise_idx = int(start_idx + offset)
            noise_class, pollution_key, text = self.dirty_store_noise_text(noise_idx)
            chat.memory_mode = "off"
            chat.start_new_session()
            session_id = chat.session.session_id
            self.direct_turn(
                chat,
                self.role_cls.USER,
                (
                    f"Dirty store noise class {noise_class} item {noise_idx}. "
                    f"{text}"
                ),
            )
            if noise_class == "irrelevant_long_session":
                self.direct_turn(
                    chat,
                    self.role_cls.USER,
                    (
                        f"Long unrelated continuation {secrets.token_hex(6)}: "
                        "meeting actions, implementation owner swaps, QA notes, "
                        "sales handoff, abandoned experiments, stale preferences, "
                        "and unrelated project names."
                    ),
                )
            chat._mark_dirty()
            require(
                "DIRTY_STORE_REAL_WORLD_RECALL",
                chat.save_current_session(rebuild_retriever=False),
                "save_current_session returned False for dirty noise session",
                noise_idx=noise_idx,
                session_id=session_id,
                noise_class=noise_class,
            )
            record = {
                "noise_idx": noise_idx,
                "session_id": session_id,
                "noise_class": noise_class,
                "pollution_key": pollution_key,
                "text": text,
            }
            records.append(record)
            if offset < 20 or offset % 100 == 0:
                self.log.event("dirty_store.noise_session_saved", **record)
        return records

    def plant_dirty_store_target_sessions(self) -> list[dict[str, Any]]:
        chat = self.load_main_chat()
        assert self.role_cls is not None
        records: list[dict[str, Any]] = []
        for fact_idx, fact in enumerate(REAL_WORLD_COLOR_MEMORIES):
            chat.memory_mode = "off"
            chat.start_new_session()
            session_id = chat.session.session_id
            self.direct_turn(
                chat,
                self.role_cls.USER,
                (
                    "Current product website palette memory. "
                    f"{fact['text']}"
                ),
            )
            self.direct_turn(
                chat,
                self.role_cls.USER,
                (
                    f"Ordinary follow-up {secrets.token_hex(6)}. This session "
                    "is about the current website color scheme, not the Acme "
                    "microsite, mobile onboarding, investor deck, coffee shop, "
                    "or an old branch."
                ),
            )
            chat._mark_dirty()
            require(
                "DIRTY_STORE_REAL_WORLD_RECALL",
                chat.save_current_session(rebuild_retriever=False),
                "save_current_session returned False for dirty target session",
                fact_idx=fact_idx,
                session_id=session_id,
                fact_key=fact["fact_key"],
            )
            record = {
                "fact_idx": int(fact_idx),
                "session_id": session_id,
                "fact_key": fact["fact_key"],
                "match_phrase": fact["match_phrase"],
                "text": fact["text"],
            }
            records.append(record)
            self.log.event("dirty_store.target_session_saved", **record)
        return records

    def dirty_store_color_probe(self) -> dict[str, Any]:
        chat = self.load_main_chat()
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            chat.retriever is not None,
            "retriever missing",
        )

        from chuk_lazarus.session_retrieval import asi_route_candidates, assign_tiers
        from chuk_lazarus.session_retrieval.enumeration import load_store

        query = (
            "Tell me everything we discussed about the website's color scheme "
            "across all our sessions."
        )
        candidates = asi_route_candidates(
            chat.retriever.handles,
            query,
            chat.retriever.tokenizer,
            candidate_pool=48,
        )
        candidates = dedupe_candidates_by_session(candidates)
        records_by_phrase = {
            str(record["match_phrase"]).lower(): record
            for record in self.dirty_real_world_records
        }
        noise_by_text = {
            str(record["text"]).lower(): record for record in self.dirty_noise_records
        }
        candidate_records: list[dict[str, Any]] = []
        near_miss_candidates: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates):
            store = load_store(candidate.handle)
            window_text = store.get_window_text(int(candidate.window_id), chat.tokenizer)
            lower = window_text.lower()
            matched = None
            for phrase, record in records_by_phrase.items():
                if phrase in lower:
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
            for text, record in noise_by_text.items():
                if text and text in lower:
                    near_miss_candidates.append(
                        {
                            **record,
                            "rank": rank,
                            "candidate_session_id": candidate.handle.session_id,
                            "window_id": int(candidate.window_id),
                        }
                    )
                    break
        seen_fact_keys = {str(record["fact_key"]) for record in candidate_records}
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            len(seen_fact_keys) >= 10,
            "dirty ASI candidate_pool did not cover at least 10 of 12 target memories",
            seen=sorted(seen_fact_keys),
            expected=sorted(record["fact_key"] for record in self.dirty_real_world_records),
            near_miss_candidates=near_miss_candidates[:12],
        )

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
        tier_near_misses: list[dict[str, Any]] = []
        for assignment in assignments:
            key = (
                assignment.candidate.handle.session_id,
                int(assignment.candidate.window_id),
            )
            record = records_by_session_window.get(key)
            if record is not None:
                tier_records[assignment.tier.value].append(record)
            else:
                for candidate_noise in near_miss_candidates:
                    if (
                        candidate_noise["candidate_session_id"],
                        int(candidate_noise["window_id"]),
                    ) == key:
                        tier_near_misses.append(
                            {
                                **candidate_noise,
                                "tier": assignment.tier.value,
                            }
                        )
                        break
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            all(len(tier_records[tier]) == 4 for tier in ("hot", "warm", "cold")),
            "dirty top 12 tier split was not 4/4/4 target memories",
            tier_records=tier_records,
            tier_near_misses=tier_near_misses,
        )

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
        previous_max_new_tokens = int(chat.max_new_tokens)
        try:
            os.environ["LAZARUS_KV_CANDIDATE_POOL"] = "12"
            os.environ["LAZARUS_KV_ROUTE_CANDIDATE_POOL"] = "48"
            os.environ["LAZARUS_KV_DEDUP_SESSION"] = "1"
            os.environ["LAZARUS_KV_K_HOT"] = "4"
            os.environ["LAZARUS_KV_K_WARM"] = "4"
            os.environ["LAZARUS_MAX_TOTAL_INJECT_TOKENS"] = "65536"
            os.environ["LAZARUS_KV_SEMANTIC_PREFIX_TOKENS"] = "4096"
            os.environ["LAZARUS_KV_HOT_BONUS"] = "0.0"
            chat.max_new_tokens = max(previous_max_new_tokens, 240)
            chat.memory_mode = "kv_direct"
            chat.start_new_session()
            started = time.time()
            meta = chat.kv_query_turn(query)
            elapsed = time.time() - started
        finally:
            chat.max_new_tokens = previous_max_new_tokens
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assert_kv_meta("DIRTY_STORE_REAL_WORLD_RECALL", meta)
        answer = str(getattr(meta, "generated_answer", "") or "")
        hot_hits = [
            record for record in tier_records["hot"]
            if real_world_fact_mentioned(str(record["fact_key"]), answer)
        ]
        warm_hits = [
            record for record in tier_records["warm"]
            if real_world_fact_mentioned(str(record["fact_key"]), answer)
        ]
        selected_fact_keys = {
            str(record["fact_key"])
            for tier in ("hot", "warm")
            for record in tier_records[tier]
        }
        cold_only_records = [
            record for record in tier_records["cold"]
            if not real_world_fact_entailed_by_selected(
                str(record["fact_key"]),
                selected_fact_keys,
            )
        ]
        cold_hits = [
            record for record in cold_only_records
            if real_world_fact_mentioned(str(record["fact_key"]), answer)
        ]
        near_miss_hits = dirty_store_near_miss_hits(answer)
        stale_ok = dirty_store_stale_fact_marked_or_excluded(answer)
        conflict_preserved = real_world_conflict_preserved(answer)
        final_decision_present = real_world_final_decision_present(answer)

        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            len(hot_hits) == 4,
            "dirty-store answer missed one or more HOT target memories",
            hot_expected=tier_records["hot"],
            hot_hits=hot_hits,
            answer=answer,
            meta=self.meta_to_dict(meta),
        )
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            len(warm_hits) >= 3,
            "dirty-store answer recalled too few WARM target memories",
            warm_expected=tier_records["warm"],
            warm_hits=warm_hits,
            answer=answer,
            meta=self.meta_to_dict(meta),
        )
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            len(cold_hits) == 0,
            "dirty-store answer mentioned COLD-only target memories",
            cold_expected=tier_records["cold"],
            cold_hits=cold_hits,
            answer=answer,
            meta=self.meta_to_dict(meta),
        )
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            not near_miss_hits,
            "dirty-store answer leaked near-miss wrong-project facts",
            near_miss_hits=near_miss_hits,
            answer=answer,
        )
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            stale_ok,
            "dirty-store answer included stale facts as if current",
            answer=answer,
        )
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            conflict_preserved and final_decision_present,
            "dirty-store answer missed revision/final decision semantics",
            conflict_preserved=conflict_preserved,
            final_decision_present=final_decision_present,
            answer=answer,
        )
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            bool(getattr(meta, "mask_penalty_applied", False))
            and getattr(meta, "selected_tier", None) == "hot"
            and bool(getattr(meta, "semantic_prefix_active", False))
            and int(getattr(meta, "multi_session_count", 0)) >= 8,
            "dirty-store KV telemetry did not report the expected bounded multi-session path",
            meta=self.meta_to_dict(meta),
        )

        result = {
            "answer": answer,
            "target_coverage": len(seen_fact_keys),
            "hot_hits": len(hot_hits),
            "warm_hits": len(warm_hits),
            "cold_hits": len(cold_hits),
            "near_miss_leak_count": len(near_miss_hits),
            "wrong_project_detail_count": len(near_miss_hits),
            "stale_fact_marked_or_excluded": stale_ok,
            "conflict_preserved": conflict_preserved,
            "final_decision_present": final_decision_present,
            "elapsed_s": elapsed,
            "tier_records": tier_records,
            "meta": self.meta_to_dict(meta),
        }
        self.log.event("dirty_store.color_probe", **result)
        return result

    def plant_dirty_domain_probe(self, probe: dict[str, Any]) -> None:
        chat = self.load_main_chat()
        assert self.role_cls is not None
        for noise_idx, text in enumerate(probe["noise"]):
            chat.memory_mode = "off"
            chat.start_new_session()
            session_id = chat.session.session_id
            self.direct_turn(
                chat,
                self.role_cls.USER,
                f"Dirty domain noise {probe['domain']} {noise_idx}. {text}",
            )
            chat._mark_dirty()
            require(
                "DIRTY_STORE_REAL_WORLD_RECALL",
                chat.save_current_session(rebuild_retriever=False),
                "save_current_session returned False for dirty domain noise",
                domain=probe["domain"],
                noise_idx=noise_idx,
                session_id=session_id,
            )
        for fact_idx, fact in enumerate(probe["facts"]):
            chat.memory_mode = "off"
            chat.start_new_session()
            session_id = chat.session.session_id
            self.direct_turn(chat, self.role_cls.USER, str(fact["text"]))
            self.direct_turn(
                chat,
                self.role_cls.USER,
                (
                    f"Current decision continuation {secrets.token_hex(6)} for "
                    f"{probe['target_context']}."
                ),
            )
            chat._mark_dirty()
            require(
                "DIRTY_STORE_REAL_WORLD_RECALL",
                chat.save_current_session(rebuild_retriever=False),
                "save_current_session returned False for dirty domain target",
                domain=probe["domain"],
                fact_idx=fact_idx,
                session_id=session_id,
            )

    def dirty_domain_probe(self, probe: dict[str, Any]) -> dict[str, Any]:
        chat = self.load_main_chat()
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            chat.retriever is not None,
            "retriever missing",
        )
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
        previous_max_new_tokens = int(chat.max_new_tokens)
        try:
            os.environ["LAZARUS_KV_CANDIDATE_POOL"] = "12"
            os.environ["LAZARUS_KV_ROUTE_CANDIDATE_POOL"] = "64"
            os.environ["LAZARUS_KV_DEDUP_SESSION"] = "1"
            os.environ["LAZARUS_KV_K_HOT"] = "4"
            os.environ["LAZARUS_KV_K_WARM"] = "4"
            os.environ["LAZARUS_MAX_TOTAL_INJECT_TOKENS"] = "65536"
            os.environ["LAZARUS_KV_SEMANTIC_PREFIX_TOKENS"] = "4096"
            os.environ["LAZARUS_KV_HOT_BONUS"] = "0.0"
            chat.max_new_tokens = max(previous_max_new_tokens, 180)
            chat.memory_mode = "kv_direct"
            chat.start_new_session()
            started = time.time()
            meta = chat.kv_query_turn(str(probe["query"]))
            elapsed = time.time() - started
        finally:
            chat.max_new_tokens = previous_max_new_tokens
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assert_kv_meta("DIRTY_STORE_REAL_WORLD_RECALL", meta)
        answer = str(getattr(meta, "generated_answer", "") or "")
        hits = [
            str(fact["fact_key"])
            for fact in probe["facts"]
            if dirty_domain_fact_mentioned(
                str(probe["domain"]),
                str(fact["fact_key"]),
                answer,
            )
        ]
        wrong_hits = dirty_domain_wrong_project_hits(answer, probe)
        final_present = dirty_domain_final_decision_present(str(probe["domain"]), answer)
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            len(hits) >= 4,
            "dirty domain answer recalled too few target facts",
            domain=probe["domain"],
            hits=hits,
            answer=answer,
            meta=self.meta_to_dict(meta),
        )
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            not wrong_hits,
            "dirty domain answer leaked wrong-project details",
            domain=probe["domain"],
            wrong_hits=wrong_hits,
            answer=answer,
        )
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            final_present,
            "dirty domain answer missed final/current decision",
            domain=probe["domain"],
            answer=answer,
        )
        result = {
            "domain": probe["domain"],
            "hits": hits,
            "wrong_project_hits": wrong_hits,
            "final_current_decision_detected": final_present,
            "answer": answer,
            "elapsed_s": elapsed,
            "meta": self.meta_to_dict(meta),
        }
        self.log.event("dirty_store.domain_probe", **result)
        return result

    def dirty_store_real_world_recall(self) -> str:
        chat = self.load_main_chat()
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            chat.retriever is not None,
            "retriever missing",
        )
        noise_count = max(0, int(self.args.dirty_store_noise_sessions))
        pre_noise = noise_count // 2
        post_noise = noise_count - pre_noise
        self.dirty_noise_records = []
        self.dirty_noise_records.extend(
            self.plant_dirty_store_noise_sessions(count=pre_noise, start_idx=0)
        )
        self.dirty_real_world_records = self.plant_dirty_store_target_sessions()
        self.dirty_noise_records.extend(
            self.plant_dirty_store_noise_sessions(
                count=post_noise,
                start_idx=pre_noise,
            )
        )
        for probe in DIRTY_STORE_DOMAIN_PROBES:
            self.plant_dirty_domain_probe(probe)
        self.refresh_retriever_after_batch("DIRTY_STORE_REAL_WORLD_RECALL", chat)

        color_result = self.dirty_store_color_probe()
        domain_results = [self.dirty_domain_probe(probe) for probe in DIRTY_STORE_DOMAIN_PROBES]
        require(
            "DIRTY_STORE_REAL_WORLD_RECALL",
            all(len(result["hits"]) >= 4 and not result["wrong_project_hits"] for result in domain_results),
            "one or more dirty domain probes failed breadth checks",
            domain_results=domain_results,
        )
        return (
            "target_coverage="
            f"{color_result['target_coverage']}/12 "
            f"HOT={color_result['hot_hits']}/4 "
            f"WARM={color_result['warm_hits']}/4 "
            f"COLD={color_result['cold_hits']}/4 "
            f"near_miss_leak_count={color_result['near_miss_leak_count']} "
            f"wrong_project_detail_count={color_result['wrong_project_detail_count']} "
            f"conflict_preserved={color_result['conflict_preserved']} "
            f"final_decision_present={color_result['final_decision_present']} "
            f"selected_tier={color_result['meta'].get('selected_tier')} "
            f"mask_penalty_applied={color_result['meta'].get('mask_penalty_applied')} "
            f"domain_probes={len(domain_results)}/{len(DIRTY_STORE_DOMAIN_PROBES)}"
        )

    def plant_multi_fact_session(
        self,
        *,
        fact_idx: int,
        topic_key: str,
        color: str,
    ) -> dict[str, Any]:
        chat = self.load_main_chat()
        assert self.role_cls is not None
        chat.memory_mode = "off"
        chat.start_new_session()
        session_id = chat.session.session_id
        marker = f"mf{fact_idx:02d}_{secrets.token_hex(8)}"
        text = (
            f"{marker}. Unique multi-fact marker {marker}. "
            f"Website palette decision anchor {topic_key}. "
            f"For the website color scheme, favorite color answer {fact_idx} "
            f"is {color}. Design palette decisions should list {color}. "
            f"Recall key {topic_key} {marker}."
        )
        self.direct_turn(chat, self.role_cls.USER, text)
        self.direct_turn(
            chat,
            self.role_cls.USER,
            (
                f"Neutral padding note {secrets.token_hex(8)}. "
                "This line exists only to give the session more than one "
                "retrieval window; it has no design-memory content."
            ),
        )
        chat._mark_dirty()
        require(
            "MULTI_FACT_RECALL",
            chat.save_current_session(rebuild_retriever=True),
            "save_current_session returned False for multi-fact session",
            fact_idx=fact_idx,
            session_id=session_id,
            marker=marker,
            color=color,
        )
        record = {
            "fact_idx": int(fact_idx),
            "session_id": session_id,
            "marker": marker,
            "topic_key": topic_key,
            "color": color,
        }
        self.log.event("multi_fact.session_saved", **record)
        return record

    def multi_fact_recall(self) -> str:
        chat = self.load_main_chat()
        require("MULTI_FACT_RECALL", chat.retriever is not None, "retriever missing")
        colors = [
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
        topic_key = f"website_palette_decision_{self.args.run_id}_{secrets.token_hex(4)}"
        self.multi_fact_records = [
            self.plant_multi_fact_session(
                fact_idx=idx,
                topic_key=topic_key,
                color=color,
            )
            for idx, color in enumerate(colors)
        ]

        from chuk_lazarus.session_retrieval import asi_route_candidates, assign_tiers
        from chuk_lazarus.session_retrieval.enumeration import load_store

        query = (
            f"List all favorite color answers for prior website color scheme "
            f"palette decisions tagged {topic_key}. Rank by relevance to "
            "design palette decisions. Return only the color words."
        )
        candidates = asi_route_candidates(
            chat.retriever.handles,
            query,
            chat.retriever.tokenizer,
            candidate_pool=64,
        )
        candidates = dedupe_candidates_by_session(candidates)
        require(
            "MULTI_FACT_RECALL",
            len(candidates) >= 12,
            "candidate_pool did not return all 12 multi-fact candidates",
            count=len(candidates),
        )

        records_by_marker = {
            str(record["marker"]).lower(): record
            for record in self.multi_fact_records
        }
        candidate_records: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates[:12]):
            store = load_store(candidate.handle)
            window_text = store.get_window_text(int(candidate.window_id), chat.tokenizer)
            matched = None
            for marker, record in records_by_marker.items():
                if marker in window_text.lower():
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
        seen_markers = {str(record["marker"]) for record in candidate_records}
        require(
            "MULTI_FACT_RECALL",
            len(seen_markers) == 12,
            "ASI candidate_pool did not cover all planted multi-facts",
            seen=sorted(seen_markers),
            expected=sorted(record["marker"] for record in self.multi_fact_records),
        )

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
        require(
            "MULTI_FACT_RECALL",
            all(len(tier_records[tier]) == 4 for tier in ("hot", "warm", "cold")),
            "tier split was not 4/4/4 for planted multi-facts",
            tier_records=tier_records,
        )

        previous_env = {
            key: os.environ.get(key)
            for key in (
                "LAZARUS_KV_CANDIDATE_POOL",
                "LAZARUS_KV_K_HOT",
                "LAZARUS_KV_K_WARM",
                "LAZARUS_KV_ROUTE_CANDIDATE_POOL",
                "LAZARUS_KV_DEDUP_SESSION",
                "LAZARUS_MAX_TOTAL_INJECT_TOKENS",
                "LAZARUS_KV_HOT_BONUS",
            )
        }
        previous_max_new_tokens = int(chat.max_new_tokens)
        try:
            os.environ["LAZARUS_KV_CANDIDATE_POOL"] = "12"
            os.environ["LAZARUS_KV_ROUTE_CANDIDATE_POOL"] = "64"
            os.environ["LAZARUS_KV_DEDUP_SESSION"] = "1"
            os.environ["LAZARUS_KV_K_HOT"] = "4"
            os.environ["LAZARUS_KV_K_WARM"] = "4"
            os.environ["LAZARUS_MAX_TOTAL_INJECT_TOKENS"] = "65536"
            os.environ["LAZARUS_KV_HOT_BONUS"] = "0.0"
            chat.max_new_tokens = max(previous_max_new_tokens, 160)
            chat.memory_mode = "kv_direct"
            chat.start_new_session()
            meta = chat.kv_query_turn(query)
        finally:
            chat.max_new_tokens = previous_max_new_tokens
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assert_kv_meta("MULTI_FACT_RECALL", meta)
        answer = str(getattr(meta, "generated_answer", "") or "")
        answer_lower = answer.lower()
        hot_hits = [
            record for record in tier_records["hot"]
            if str(record["color"]).lower() in answer_lower
        ]
        warm_hits = [
            record for record in tier_records["warm"]
            if str(record["color"]).lower() in answer_lower
        ]
        cold_hits = [
            record for record in tier_records["cold"]
            if str(record["color"]).lower() in answer_lower
        ]
        require(
            "MULTI_FACT_RECALL",
            len(hot_hits) == 4,
            "generated answer missed one or more HOT facts",
            hot_expected=tier_records["hot"],
            hot_hits=hot_hits,
            answer=answer,
            meta=self.meta_to_dict(meta),
        )
        require(
            "MULTI_FACT_RECALL",
            len(warm_hits) >= 3,
            "generated answer recalled too few WARM facts",
            warm_expected=tier_records["warm"],
            warm_hits=warm_hits,
            answer=answer,
            meta=self.meta_to_dict(meta),
        )
        require(
            "MULTI_FACT_RECALL",
            len(cold_hits) == 0,
            "generated answer mentioned COLD facts that should be muted",
            cold_expected=tier_records["cold"],
            cold_hits=cold_hits,
            answer=answer,
            meta=self.meta_to_dict(meta),
        )
        require(
            "MULTI_FACT_RECALL",
            bool(getattr(meta, "mask_penalty_applied", False)),
            "axis-6 did not report mask_penalty_applied=True",
            meta=self.meta_to_dict(meta),
        )
        require(
            "MULTI_FACT_RECALL",
            getattr(meta, "selected_tier", None) == "hot",
            "axis-6 did not report selected_tier=hot",
            meta=self.meta_to_dict(meta),
        )
        self.log.event(
            "multi_fact.meta",
            answer=answer,
            hot_hits=len(hot_hits),
            warm_hits=len(warm_hits),
            cold_hits=len(cold_hits),
            tier_records=tier_records,
            meta=self.meta_to_dict(meta),
        )
        return (
            "multi-fact recall HOT=4/4 WARM="
            f"{len(warm_hits)}/4 COLD={len(cold_hits)}/4 "
            f"selected_tier={getattr(meta, 'selected_tier', None)}"
        )

    def token_budget(self) -> str:
        chat = self.load_main_chat()
        self.budget_observations.clear()
        budget_fixture = self.plant_budget_pressure_session(turns=8)
        original = chat._apply_token_budget
        previous_candidate_pool = os.environ.get("LAZARUS_KV_CANDIDATE_POOL")
        previous_inject_budget = os.environ.get("LAZARUS_MAX_TOTAL_INJECT_TOKENS")

        def wrapped(tier_assignments: list[Any], *args: Any, **kwargs: Any) -> list[Any]:
            before = len(tier_assignments)
            kept = original(tier_assignments, *args, **kwargs)
            cfg = getattr(chat.model, "config", None)
            head_dim = int(getattr(cfg, "head_dim", 256) or 256)
            n_kv_heads = int(getattr(cfg, "num_key_value_heads", 4) or 4)
            max_total = int(
                kwargs.get("max_total_inject_tokens")
                or os.environ.get("LAZARUS_MAX_TOTAL_INJECT_TOKENS", "4096")
            )
            observation = {
                "before": before,
                "after": len(kept),
                "head_dim": head_dim,
                "num_key_value_heads": n_kv_heads,
                "per_fact_cost": head_dim * n_kv_heads,
                "max_total_inject_tokens": max_total,
                "forced_env_budget": os.environ.get(
                    "LAZARUS_MAX_TOTAL_INJECT_TOKENS"
                ),
            }
            self.budget_observations.append(observation)
            self.log.event("token_budget.observed", **observation)
            return kept

        chat._apply_token_budget = wrapped
        try:
            chat.memory_mode = "kv_direct"
            chat.start_new_session()
            marker = str(budget_fixture["marker"])
            query = f"{marker}"
            os.environ["LAZARUS_KV_CANDIDATE_POOL"] = str(
                max(max(parse_csv_ints(self.args.candidate_pools)), 16)
            )
            os.environ["LAZARUS_MAX_TOTAL_INJECT_TOKENS"] = "1024"
            meta = chat.kv_query_turn(query)
        finally:
            chat._apply_token_budget = original
            if previous_candidate_pool is None:
                os.environ.pop("LAZARUS_KV_CANDIDATE_POOL", None)
            else:
                os.environ["LAZARUS_KV_CANDIDATE_POOL"] = previous_candidate_pool
            if previous_inject_budget is None:
                os.environ.pop("LAZARUS_MAX_TOTAL_INJECT_TOKENS", None)
            else:
                os.environ["LAZARUS_MAX_TOTAL_INJECT_TOKENS"] = previous_inject_budget
        self.assert_kv_meta("TOKEN_BUDGET", meta)
        binding = [
            item for item in self.budget_observations
            if int(item["before"]) > int(item["after"])
        ]
        require(
            "TOKEN_BUDGET",
            bool(binding),
            "token-budget governor never dropped an assignment",
            fixture=budget_fixture,
            observations=self.budget_observations,
        )
        for item in binding:
            require(
                "TOKEN_BUDGET",
                int(item["before"]) >= 2,
                "budget pressure fixture routed fewer than two assignments",
                fixture=budget_fixture,
                observation=item,
            )
            max_facts = int(item["max_total_inject_tokens"]) // int(item["per_fact_cost"])
            require(
                "TOKEN_BUDGET",
                int(item["after"]) <= max_facts,
                "token-budget kept too many assignments",
                observation=item,
                max_facts=max_facts,
            )
        return (
            f"budget bound observed {len(binding)} time(s) on dedicated "
            f"session {budget_fixture['session_id']}"
        )

    def vram_bounded(self) -> str:
        chat = self.load_main_chat()
        peaks: list[float] = []
        marker = self.scale_markers[0]["marker"] if self.scale_markers else self.primary_marker
        for pool in parse_csv_ints(self.args.candidate_pools):
            os.environ["LAZARUS_KV_CANDIDATE_POOL"] = str(pool)
            chat.memory_mode = "kv_direct"
            chat.start_new_session()
            meta = chat.kv_query_turn(f"Recall planted marker {marker}.")
            self.assert_kv_meta("VRAM_BOUNDED", meta)
            peak = float(getattr(meta, "vram_peak_mib", 0.0) or 0.0)
            peaks.append(peak)
            self.log.event(
                "vram.probe",
                candidate_pool=pool,
                vram_peak_mib=peak,
                meta=self.meta_to_dict(meta),
            )
            require(
                "VRAM_BOUNDED",
                peak <= self.args.vram_ceiling_mib,
                "VRAM peak exceeded ceiling",
                candidate_pool=pool,
                peak=peak,
                ceiling=self.args.vram_ceiling_mib,
            )
        delta = max(peaks) - min(peaks) if peaks else 0.0
        require(
            "VRAM_BOUNDED",
            delta <= self.args.vram_plateau_delta_mib,
            "VRAM peak did not plateau across candidate pools",
            peaks=peaks,
            delta=delta,
            plateau_delta_mib=self.args.vram_plateau_delta_mib,
        )
        return f"vram peaks={peaks} delta={delta:.1f} MiB"

    def fallback_truth(self) -> str:
        chat = self.load_main_chat()
        require("FALLBACK_TRUTH", chat.retriever is not None, "retriever missing")
        original = chat.retriever.answer_with_kv_direct
        original_multi = getattr(chat.retriever, "answer_with_kv_direct_multi", None)

        def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("autoverify forced kv_direct failure")

        chat.retriever.answer_with_kv_direct = boom
        if original_multi is not None:
            chat.retriever.answer_with_kv_direct_multi = boom
        capture = io.StringIO()
        try:
            chat.memory_mode = "kv_direct"
            chat.start_new_session()
            with contextlib.redirect_stdout(Tee(sys.stdout, self.log.transcript_handle, capture)):
                meta = chat.kv_query_turn(f"Recall {self.primary_marker}")
        finally:
            chat.retriever.answer_with_kv_direct = original
            if original_multi is not None:
                chat.retriever.answer_with_kv_direct_multi = original_multi
        captured = capture.getvalue()
        require(
            "FALLBACK_TRUTH",
            "[WARN] axis-6: SILENT FALLBACK DETECTED" in captured,
            "forced fallback did not print WARN",
            captured=captured[-1200:],
        )
        require(
            "FALLBACK_TRUTH",
            not bool(getattr(meta, "no_silent_fallback", True)),
            "forced fallback did not set no_silent_fallback=False",
            meta=self.meta_to_dict(meta),
        )
        return "forced KV-direct failure produced WARN and no_silent_fallback=False"

    def infinite_turn_latency(self) -> str:
        base = self.load_main_chat()
        previous_temp = os.environ.get("LAZARUS_GENERATION_TEMPERATURE")
        previous_top_p = os.environ.get("LAZARUS_GENERATION_TOP_P")
        os.environ["LAZARUS_GENERATION_TEMPERATURE"] = "0.0"
        os.environ["LAZARUS_GENERATION_TOP_P"] = "1.0"

        chat = self.make_chat(
            memory_mode="off",
            generation_engine="residual_bounded_kv_direct",
            hot_budget_mib=int(self.args.turn_latency_hot_budget_mib),
            session_cache_size=1,
            max_new_tokens=int(self.args.turn_latency_max_new_tokens),
        )
        chat.model = base.model
        chat.tokenizer = base.tokenizer
        chat._configure_generation_runtime()
        require(
            "INFINITE_TURN_LATENCY",
            chat.runtime is not None,
            "bounded generation runtime was not created",
        )
        require(
            "INFINITE_TURN_LATENCY",
            getattr(chat.runtime, "engine_mode", None) == "residual_bounded_kv_direct",
            "runtime engine is not residual_bounded_kv_direct",
            engine=getattr(chat.runtime, "engine_mode", None),
        )
        require(
            "INFINITE_TURN_LATENCY",
            getattr(chat.runtime, "session_cache", None) is not None,
            "residual session cache was not attached",
        )

        chat.start_new_session()
        total_turns = int(self.args.turn_latency_turns)
        warmup_turns = int(self.args.turn_latency_warmup_turns)
        tolerance = float(self.args.turn_latency_tolerance)
        window_size = max(1, int(self.args.turn_latency_window_size))
        latencies: list[float] = []
        meta_latencies: list[float] = []
        generated_tokens: list[int] = []
        generation_paths: list[str] = []

        try:
            for turn_idx in range(warmup_turns + total_turns):
                prompt = (
                    f"Latency invariant turn {turn_idx}. "
                    "Reply with four compact numbered lines. "
                    "Each line must include the words stable bounded memory. "
                    "Keep the wording simple and do not end early."
                )
                started = time.perf_counter()
                meta = chat.plain_chat_turn(prompt)
                elapsed = time.perf_counter() - started
                path = str(getattr(chat.runtime, "last_generation_path", "") or "")
                meta_elapsed = float(getattr(meta, "generate_time", 0.0) or 0.0)
                token_count = int(getattr(meta, "generated_tokens", 0) or 0)
                self.log.event(
                    "turn_latency.probe",
                    turn_idx=turn_idx,
                    measured=turn_idx >= warmup_turns,
                    elapsed_s=elapsed,
                    meta_elapsed_s=meta_elapsed,
                    generation_path=path,
                    generated_tokens=token_count,
                )
                if turn_idx >= warmup_turns:
                    latencies.append(elapsed)
                    meta_latencies.append(meta_elapsed)
                    generated_tokens.append(token_count)
                    generation_paths.append(path)
        finally:
            if previous_temp is None:
                os.environ.pop("LAZARUS_GENERATION_TEMPERATURE", None)
            else:
                os.environ["LAZARUS_GENERATION_TEMPERATURE"] = previous_temp
            if previous_top_p is None:
                os.environ.pop("LAZARUS_GENERATION_TOP_P", None)
            else:
                os.environ["LAZARUS_GENERATION_TOP_P"] = previous_top_p
            if getattr(chat, "session_cache", None) is not None:
                chat.session_cache.clear()

        require(
            "INFINITE_TURN_LATENCY",
            len(latencies) == total_turns,
            "latency probe did not collect the requested turn count",
            collected=len(latencies),
            expected=total_turns,
        )
        non_positive = [
            {"turn": idx, "elapsed_s": value}
            for idx, value in enumerate(latencies)
            if value <= 0.0
        ]
        require(
            "INFINITE_TURN_LATENCY",
            not non_positive,
            "monotonic latency measurement produced a non-positive value",
            non_positive=non_positive[:10],
            latencies=latencies,
        )
        window_size = min(window_size, total_turns)
        first_window = latencies[:window_size]
        last_window = latencies[-window_size:]
        first_mean = sum(first_window) / len(first_window)
        last_mean = sum(last_window) / len(last_window)
        max_latency = max(latencies)
        median_latency = sorted(latencies)[len(latencies) // 2]
        ceiling = first_mean * (1.0 + tolerance)
        require(
            "INFINITE_TURN_LATENCY",
            last_mean <= ceiling,
            "bounded-KV mean latency trend exceeded tolerance",
            first_window_mean_s=first_mean,
            last_window_mean_s=last_mean,
            ceiling_s=ceiling,
            tolerance=tolerance,
            window_size=window_size,
            latencies=latencies,
            meta_latencies=meta_latencies,
            generated_tokens=generated_tokens,
            generation_paths=generation_paths,
        )
        require(
            "INFINITE_TURN_LATENCY",
            all("session_reused" in path for path in generation_paths),
            "bounded engine did not reuse the residual session cache",
            generation_paths=generation_paths,
        )
        return (
            f"{total_turns} measured turns flat: first{window_size}_mean="
            f"{first_mean:.3f}s last{window_size}_mean={last_mean:.3f}s "
            f"median={median_latency:.3f}s max={max_latency:.3f}s "
            f"ceiling={ceiling:.3f}s"
        )

    def memory_off(self) -> str:
        chat = self.load_main_chat()
        chat.memory_mode = "off"
        chat.start_new_session()
        meta = chat.plain_chat_turn(
            f"Memory is off. What was marker {self.primary_marker}?"
        )
        require(
            "MEMORY_OFF",
            meta.mode in {"plain", "none"},
            "memory off did not use plain/no-memory path",
            meta=self.meta_to_dict(meta),
        )
        require(
            "MEMORY_OFF",
            not bool(getattr(meta, "no_silent_fallback", False)),
            "memory off should not report retrieval success",
            meta=self.meta_to_dict(meta),
        )
        return f"memory off returned mode={meta.mode}"

    def crash_gate(self) -> str:
        chat = self.load_main_chat()
        from chuk_lazarus.session_retrieval.enumeration import iter_checkpoint_handles

        before = {handle.session_id for handle in iter_checkpoint_handles(chat.checkpoints_root)}
        partial_id = f"crash-partial-{self.args.run_id}"
        partial_store = chat.checkpoints_root / partial_id / "torch_store"
        partial_store.mkdir(parents=True, exist_ok=True)
        (partial_store / "window_metadata.json").write_text("{}", encoding="utf-8")
        (partial_store / "boundaries").mkdir(exist_ok=True)
        inflight_id = f"crash-inflight-{self.args.run_id}"
        inflight_store = chat.checkpoints_root / inflight_id / "torch_store"
        inflight_store.mkdir(parents=True, exist_ok=True)
        (inflight_store / "manifest.json").write_text(
            json.dumps(
                {
                    "clause_aligned": True,
                    "num_windows": 1,
                    "num_entries": 1,
                    "num_tokens": 0,
                    "in_flight": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        after = {handle.session_id for handle in iter_checkpoint_handles(chat.checkpoints_root)}
        require(
            "CRASH_GATE",
            partial_id not in after,
            "partial store without manifest was enumerated as ready",
            before_count=len(before),
            after_count=len(after),
            partial_id=partial_id,
        )
        require(
            "CRASH_GATE",
            inflight_id not in after,
            "in-flight manifest was enumerated as ready",
            before_count=len(before),
            after_count=len(after),
            inflight_id=inflight_id,
        )
        return "manifest-absent and in-flight partial stores stayed invisible"

    def parallel_writes(self) -> str:
        base = self.load_main_chat()
        assert self.imc is not None
        assert self.role_cls is not None

        def worker(worker_idx: int) -> str:
            chat = self.make_chat(memory_mode="off")
            chat.model = base.model
            chat.tokenizer = base.tokenizer
            chat.start_new_session()
            session_id = chat.session.session_id
            marker = f"PARALLEL_MARKER_{self.args.run_id}_{worker_idx}"
            turn = chat.session.begin_turn(
                self.role_cls.USER,
                f"{marker}. Parallel writer {worker_idx} owns this session.",
            )
            chat.session.finish_turn(turn)
            chat._capture_turn_text_live(turn)
            chat._mark_dirty()
            ok = chat.save_current_session(rebuild_retriever=False)
            require(
                "PARALLEL_WRITES",
                ok,
                "parallel worker save_current_session returned False",
                worker_idx=worker_idx,
                session_id=session_id,
            )
            return session_id

        session_ids: list[str] = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker, idx) for idx in (0, 1)]
            for future in as_completed(futures):
                session_ids.append(future.result())

        from chuk_lazarus.session_retrieval.enumeration import iter_checkpoint_handles, load_store

        handles = list(iter_checkpoint_handles(base.checkpoints_root))
        handle_ids = {handle.session_id for handle in handles}
        for session_id in session_ids:
            require(
                "PARALLEL_WRITES",
                session_id in handle_ids,
                "parallel session was not enumerated",
                session_id=session_id,
            )
            handle = next(handle for handle in handles if handle.session_id == session_id)
            store = load_store(handle)
            require(
                "PARALLEL_WRITES",
                int(getattr(store, "num_windows", 0) or 0) > 0,
                "parallel store loaded but had zero windows",
                session_id=session_id,
            )
        if base.retriever is not None:
            base.retriever.refresh_handles(base.checkpoints_root)
        return f"parallel sessions loadable: {', '.join(session_ids)}"

    def find_clause_for_marker(self, session_id: str, marker: str) -> dict[str, Any] | None:
        chat = self.load_main_chat()
        inputs_dir = chat.inputs_root / session_id
        for path in sorted(inputs_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            content = str(data.get("clause_content", ""))
            if marker.lower() in content.lower():
                result = dict(data)
                result["path"] = str(path)
                return result
        return None

    def assert_kv_meta(self, check: str, meta: Any) -> None:
        require(check, meta.mode == "kv_direct", "KV-direct fell back", meta=self.meta_to_dict(meta))
        require(
            check,
            bool(getattr(meta, "kv_direct_active", False)),
            "kv_direct_active is not true",
            meta=self.meta_to_dict(meta),
        )
        require(
            check,
            bool(getattr(meta, "no_silent_fallback", False)),
            "no_silent_fallback is not true",
            meta=self.meta_to_dict(meta),
        )
        require(
            check,
            getattr(meta, "selected_tier", "not-implemented-yet") != "not-implemented-yet",
            "selected_tier is still sentinel",
            meta=self.meta_to_dict(meta),
        )
        require(
            check,
            isinstance(getattr(meta, "vram_peak_mib", None), (int, float)),
            "vram_peak_mib is missing or non-numeric",
            meta=self.meta_to_dict(meta),
        )

    def meta_to_dict(self, meta: Any) -> dict[str, Any]:
        fields = (
            "mode",
            "routing_mode",
            "source_session",
            "window_id",
            "routing_score",
            "matched_window_text",
            "strict_assertions",
            "selected_tier",
            "mask_penalty_applied",
            "kv_direct_active",
            "vram_peak_mib",
            "vram_delta_mib",
            "no_silent_fallback",
            "candidate_count",
            "tier_assignment_count",
            "budgeted_assignment_count",
            "multi_session_count",
            "semantic_prefix_active",
            "semantic_prefix_tokens",
            "generated_answer",
            "generated_tokens",
            "prompt_tokens",
        )
        data = {field_name: getattr(meta, field_name, None) for field_name in fields}
        if isinstance(data.get("matched_window_text"), str):
            data["matched_window_text"] = data["matched_window_text"][:700]
        if isinstance(data.get("generated_answer"), str):
            data["generated_answer"] = data["generated_answer"][:700]
        return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-checks", action="store_true", help="Print checks and exit.")
    parser.add_argument("--skip-env-check", action="store_true", help="Allow non-Linux/CUDA dev dry checks.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--store-root", type=Path, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--sessions", type=int, default=100)
    parser.add_argument("--turns-per-session", type=int, default=100)
    parser.add_argument("--candidate-pools", default="16,64,256")
    parser.add_argument("--routing-hit-threshold", type=float, default=0.99)
    parser.add_argument("--vram-ceiling-mib", type=float, default=30_000.0)
    parser.add_argument("--vram-plateau-delta-mib", type=float, default=2048.0)
    parser.add_argument("--min-vram-mib", type=float, default=30_000.0)
    parser.add_argument("--turn-latency-turns", type=int, default=50)
    parser.add_argument("--turn-latency-warmup-turns", type=int, default=2)
    parser.add_argument("--turn-latency-tolerance", type=float, default=0.20)
    parser.add_argument("--turn-latency-window-size", type=int, default=10)
    parser.add_argument("--turn-latency-hot-budget-mib", type=int, default=150)
    parser.add_argument("--turn-latency-max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--dirty-store-noise-sessions",
        type=int,
        default=500,
        help=(
            "Irrelevant dirty-store sessions planted around the target memories "
            "for DIRTY_STORE_REAL_WORLD_RECALL."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=utc_stamp().replace("Z", "").lower(),
        help="Stable suffix embedded in planted markers.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the selected store root before running.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 5 sessions x 3 turns for a quick wiring smoke.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_checks:
        for code, description in CHECKS:
            print(f"{code}\t{description}")
        return 0
    if args.smoke:
        args.sessions = 5
        args.turns_per_session = 3
        args.candidate_pools = "4,8,16"
        args.routing_hit_threshold = min(args.routing_hit_threshold, 0.8)
        args.turn_latency_turns = min(args.turn_latency_turns, 8)
        args.turn_latency_warmup_turns = min(args.turn_latency_warmup_turns, 1)
        args.turn_latency_window_size = min(args.turn_latency_window_size, 4)
        args.dirty_store_noise_sessions = min(args.dirty_store_noise_sessions, 50)

    run_dir = args.output_root / f"{utc_stamp()}-{args.run_id}"
    log = AuditLog(run_dir)
    log.open()
    log.event("run.args", args=vars(args))
    status = "FAIL"
    error: dict[str, Any] | None = None
    automation = ReplAutomation(args, log)
    if args.fresh and automation.store_root.exists():
        forbidden = {
            REPO_ROOT.resolve(),
            log.run_dir.resolve(),
            log.run_dir.parent.resolve(),
        }
        if automation.store_root.resolve() in forbidden:
            raise RuntimeError(f"refusing to delete {automation.store_root}")
        shutil.rmtree(automation.store_root)

    assert log.transcript_handle is not None
    with contextlib.redirect_stdout(Tee(sys.stdout, log.transcript_handle)):
        with contextlib.redirect_stderr(Tee(sys.stderr, log.transcript_handle)):
            try:
                automation.run()
                status = "PASS"
            except HarnessFailure as exc:
                error = {
                    "check": exc.check,
                    "reason": exc.reason,
                    "fields": exc.fields,
                }
                status = "FAIL"
            except Exception as exc:  # noqa: BLE001
                error = {
                    "check": "HARNESS",
                    "reason": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                status = "FAIL"
                print(traceback.format_exc(), flush=True)
            finally:
                log.write_summary(
                    status,
                    error=error,
                    store_root=str(automation.store_root),
                    run_id=args.run_id,
                    sessions=args.sessions,
                    turns_per_session=args.turns_per_session,
                )
                print(f"HARNESS_{status}", flush=True)
                print(f"run_dir={run_dir}", flush=True)
                print(f"summary={log.summary_path}", flush=True)
                print(f"events={log.events_path}", flush=True)
                print(f"transcript={log.transcript_path}", flush=True)
    log.close()
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
