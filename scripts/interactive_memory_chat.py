#!/usr/bin/env python3
"""Interactive pseudo-infinite-memory chat.

Exercises the full turn-aligned pipeline end-to-end in a REPL:

  chat_loop (streaming) -> session_close (AUS3000 emit)
                        -> session_store (clause-aligned torch store build)
                        -> session_retrieval (residual-injected recall via the
                           new canonical prefill-seeded method)

The workflow:

  1. You chat with Gemma-4-E2B-it normally. Your session accumulates in memory.
  2. Press the slash-command keybind /save (or /new) to compress the current
     session to vectors and add it to the persistent store.
  3. From that moment on, any turn you take will first query the memory store
     for a relevant prior-session window, inject the matched window's boundary
     residual via forward_pre_hook on layers[0] at position 0 (the canonical
     prefill-seeded mechanism), and generate a response anchored on it.
  4. Before every recall-style reply, a DEBUG block prints:
       - routing_mode, source_session, window_id, routing_score
       - matched_window_text (truncated)
       - window_keywords
       - strict_assertions (all six)
       - timing (tokenize / retrieve / generate)
       - token counts (prompt / generated)
     so you can eyeball the inner workings.

Commands (typed at the prompt):

  /save                 compress current session to the store; keep chatting
  /new                  /save, then start a fresh session
  /query <text>         topical recall probe (no chat-history mutation)
  /exact  <dotted-id>   exact-id recall probe (no chat-history mutation)
  /entity <text>        entity-mention recall probe (no chat-history mutation)
  /kv_query <text>      axis-5 KV-direct query — runs the real ASI-router +
                        tier-policy + KV-direct runtime path. Defaults to the
                        shipped Gemma-4 full-attention insertion path
                        (retrieval_layer=12, injection_layer=13).
                        Explicit sliding requests require
                        --insertion-family sliding plus BOTH
                        --sliding-layer-indices and
                        --sliding-head-indices. The surfaced route only
                        honors sliding when the archived checkpoint was
                        materialized from a sliding-source lineage; current
                        Gemma-4 live indexing still emits the full-attention
                        lineage by default, so selector mismatches WARN
                        truthfully instead of pretending sliding ran.
                        Surfaces the real axis-6 observability fields
                        (selected_tier, mask_penalty_applied,
                        kv_direct_active, vram_peak_mib, vram_delta_mib,
                        no_silent_fallback) on self.last_meta.
                        Env overrides: LAZARUS_KV_CANDIDATE_POOL (default 16),
                        LAZARUS_KV_K_HOT (4), LAZARUS_KV_K_WARM (8),
                        LAZARUS_KV_HOT_BUDGET_MIB (32),
                        LAZARUS_KV_HOT_BONUS (default 0.0).
  /stats                print store summary
  /last                 print last turn's routing metadata
  /history              dump the current session transcript
  /memory               toggle auto-recall-injection on / off
  /help                 this help
  /quit | /exit | EOF   graceful exit (prompts to /save first)

Environment overrides:

  LAZARUS_STORE_DIR          persistent store root (default /tmp/interactive-memory)
  LAZARUS_MODEL              model id/path (default: local Gemma snapshot -> hub id)
  LAZARUS_MAX_NEW_TOKENS     decode length (default 180)
  LAZARUS_MEMORY_MODE        one of: auto (default) | topical | entity_mention |
                              kv_direct | off
  LAZARUS_GENERATION_ENGINE  standard (default) | kv_direct | bounded_kv_direct |
                              residual_bounded_kv_direct
  LAZARUS_HOT_BUDGET_MIB     hot KV budget for bounded engines (default 150
                              when residual_bounded_kv_direct is selected)
  LAZARUS_SESSION_CACHE_SIZE residual session cache entries for bounded engine
  LAZARUS_MAX_TOTAL_INJECT_TOKENS
                              KV-direct token-equivalent budget (default 4096)

Example session:

  For manual testing, prefer the wrapper so you get a fresh repo-local store:

  $ scripts/run_interactive_memory_chat.sh

  Equivalent raw invocation:

  $ python scripts/interactive_memory_chat.py
  [store] /tmp/interactive-memory/checkpoints (empty)
  [model] loading google/gemma-4-E2B-it ... done.

  you> My dog's name is Banjo and he loves chasing dragonflies at dusk.
  gemma> That sounds lovely! ...

  you> /save
  [save] emitting AUS3000 clauses ... 6 records
  [save] building clause-aligned store ... done.
  [save] retriever refreshed: 1 session, 42 windows, 12418 tokens

  you> /new
  [new] fresh session started.

  you> What was my dog's name?
  ===== ROUTING ======================================================
  mode           : topical
  source_session : b4f8e9...
  window_id      : 3
  routing_score  : 0.742
  matched_window : Turn 1 on dog-chat: My dog's name is Banjo and he loves
                   chasing dragonflies at dusk. ...
  keywords       : ['banjo', 'dog', 'dragonflies', 'dusk', 'chasing']
  strict_asserts : cuda_available=True model_on_cuda=True residual_compat=True
                   hook_fired=True gpu_memory_grew=True store_window_nonempty=True
  timing (s)     : retrieve=0.43 generate=1.82 total=2.25
  tokens         : prompt=287 generated=52
  ====================================================================
  gemma> Your dog's name is Banjo. You mentioned he loves chasing dragonflies at dusk.

"""

from __future__ import annotations

import argparse
import atexit
import inspect
import json
import os
import shlex
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from chuk_lazarus.memory_config import (
    MemoryRecallConfig,
    get_memory_profile_names,
    memory_config_env,
    resolve_memory_config,
)
from chuk_lazarus.repl_agent_tools import (
    LocalCodingToolRunner,
    extract_tool_calls,
    format_tool_results,
)

# LiveIndexer replaces the subprocess `invoke_build` call at runtime. The
# chat loop now indexes streaming windows into the per-session torch_store
# directory live, in the same on-disk format TorchKnowledgeStore.load expects,
# so /save only needs to drain + compact + refresh. The offline/CI
# `invoke_build` entry point (src/chuk_lazarus/session_store/invoke.py) is
# intentionally retained for rebuild-from-AUS3000 workflows but is NEVER
# invoked at runtime from this REPL.
from chuk_lazarus.session_store.live_indexer import LiveIndexer

DEFAULT_STORE = "/tmp/interactive-memory"
HEADER_W = 72
KVInsertionFamily = Literal["full_attention", "sliding"]
GENERATION_ENGINES = (
    "standard",
    "kv_direct",
    "bounded_kv_direct",
    "residual_bounded_kv_direct",
)
SCRIPT_MEMORY_MODES = ("auto", "topical", "entity_mention", "kv_direct", "off")
KV_QUERY_USAGE = (
    "/kv_query <text> | "
    "/kv_query --insertion-family sliding "
    "--sliding-layer-indices 13,15 "
    "--sliding-head-indices 0,7 "
    "<text>"
)
HELP_TEXT = """Interactive memory chat commands

Regular chat uses memory_mode=auto by default. If no memories exist, Lazarus
chats normally while live-indexing the current session. If memories exist, it
attempts bounded KV-direct recall automatically and reports memory_used=false
on ordinary no-memory matches.

Commands:
  /help                         show this help
  /save-memory                  save/index the current session (alias: /save)
  /new-chat                     save, then start a fresh session (alias: /new)
  /ask-memory <question>        debug KV-direct memory query (alias: /kv_query)
  /search-memory <text>         topical memory probe (alias: /query)
  /open-memory <id>             exact memory-id probe (alias: /exact)
  /find-memory-entity <text>    entity memory probe (alias: /entity)
  /memory-status                print store summary (alias: /stats)
  /last-memory                  print last turn telemetry (alias: /last)
  /show-history                 show current session transcript (alias: /history)
  /memory auto                  enable plug-and-play memory recall
  /memory off                   disable memory recall
  /memory profile <name>        switch profile: plug_and_play, proof, smoke,
                                nightly, diagnostic
  /memory config                show resolved memory configuration
  /tools status                 show coding-agent tool status
  /tools schema                 show model tool-call instructions
  /tools reload                 reload custom tools from disk
  /tools on                     enable coding-agent tool execution for future turns
  /tools off                    disable coding-agent tool execution
  /quit                         save if needed and exit

When coding-agent tools are enabled, the model may emit tool_call JSON for
list_dir, read_file, write_file, save_custom_tool, read_custom_tool,
list_custom_tools, search, shell, apply_patch, or any loaded custom tool. Tool
outputs are appended as synthetic TOOL_RESULT user turns, bounded by the max
tool-step setting, so the transcript/tool traces are durable source-of-truth
events and memory is derived.

Power-user /ask-memory options:
  /ask-memory --insertion-family sliding --sliding-layer-indices 13,15
              --sliding-head-indices 0,7 <question>
"""


# axis-3 (Addition 2): single dirty-flag file at <store_root>/.dirty.
# Set by every assistant turn that adds tokens; cleared by emit_store
# after a successful encode. /save is a no-op when this flag is absent.
DIRTY_FLAG_FILENAME = ".dirty"

# axis-4 (Addition 1): per-session injection token budget for the
# ASI-routed warm/hot list passed into answer_with_kv_direct. Each fact
# costs head_dim * num_kv_heads tokens-equivalent; the governor sorts
# by score and truncates from the bottom. Override with
# LAZARUS_MAX_TOTAL_INJECT_TOKENS for deployment or proof harness tuning.
MAX_TOTAL_INJECT_TOKENS = 4096


class NoRelevantMemory(RuntimeError):
    """Ordinary auto-recall miss, not a diagnostics failure."""


def _asi_feedback_reward(
    strict_assertions: dict[str, Any],
    evidence_supports: list[Any],
    *,
    fallback: bool = False,
) -> float:
    """Derive bounded reward only from observable recall outcomes."""
    reward = 0.0
    if bool(strict_assertions.get("kv_direct_active", False)):
        reward += 0.55
    if evidence_supports:
        reward += 0.30
    bool_assertions = [
        bool(value) for value in strict_assertions.values() if isinstance(value, bool)
    ]
    if bool_assertions and not all(bool_assertions):
        reward -= 0.50
    if fallback:
        reward -= 0.10
    return max(-1.0, min(1.0, float(reward)))


def ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def info(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def _env_max_total_inject_tokens() -> int:
    try:
        return int(resolve_memory_config().max_total_inject_tokens)
    except Exception:
        raw_value = os.environ.get(
            "LAZARUS_MAX_TOTAL_INJECT_TOKENS",
            str(MAX_TOTAL_INJECT_TOKENS),
        )
        return int(raw_value)


def _normalize_generation_engine(raw_value: str | None) -> str:
    normalized = str(raw_value or "standard").strip().lower()
    if normalized not in GENERATION_ENGINES:
        raise ValueError("generation_engine must be one of: " + ", ".join(GENERATION_ENGINES))
    return normalized


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def rule(char: str = "─") -> str:
    return char * HEADER_W


def section(title: str) -> None:
    print()
    print(f"===== {title} ".ljust(HEADER_W, "="))


def truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _parse_csv_int_tuple(raw_value: str, *, option_name: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw_value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"{option_name} must be a comma-separated list of integers")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{option_name} must be a comma-separated list of integers") from exc


def _parse_kv_query_args(raw_arg: str) -> tuple[str, dict[str, Any]]:
    try:
        tokens = shlex.split(raw_arg)
    except ValueError as exc:
        raise ValueError(f"{KV_QUERY_USAGE} ({exc})") from exc

    if not tokens:
        raise ValueError(KV_QUERY_USAGE)

    insertion_family: KVInsertionFamily = "full_attention"
    sliding_layer_indices: tuple[int, ...] | None = None
    sliding_head_indices: tuple[int, ...] | None = None
    query_tokens: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--":
            query_tokens = tokens[idx + 1 :]
            break
        if not token.startswith("-"):
            query_tokens = tokens[idx:]
            break
        if token in ("--insertion-family", "--family"):
            idx += 1
            if idx >= len(tokens):
                raise ValueError(f"{token} requires a value ({KV_QUERY_USAGE})")
            family = tokens[idx].strip().lower()
            if family not in {"full_attention", "sliding"}:
                raise ValueError("--insertion-family must be one of {full_attention, sliding}")
            insertion_family = family
        elif token == "--sliding-layer-indices":
            idx += 1
            if idx >= len(tokens):
                raise ValueError(f"{token} requires a value ({KV_QUERY_USAGE})")
            sliding_layer_indices = _parse_csv_int_tuple(tokens[idx], option_name=token)
        elif token == "--sliding-head-indices":
            idx += 1
            if idx >= len(tokens):
                raise ValueError(f"{token} requires a value ({KV_QUERY_USAGE})")
            sliding_head_indices = _parse_csv_int_tuple(tokens[idx], option_name=token)
        else:
            raise ValueError(f"unrecognized /kv_query option {token!r}")
        idx += 1

    if not query_tokens:
        raise ValueError(KV_QUERY_USAGE)
    if insertion_family == "sliding" and (
        sliding_layer_indices is None or sliding_head_indices is None
    ):
        raise ValueError(
            "--insertion-family sliding requires both "
            "--sliding-layer-indices and --sliding-head-indices"
        )
    if insertion_family != "sliding" and (
        sliding_layer_indices is not None or sliding_head_indices is not None
    ):
        raise ValueError(
            "--sliding-layer-indices and --sliding-head-indices require --insertion-family sliding"
        )

    query_text = " ".join(query_tokens).strip()
    if not query_text:
        raise ValueError(KV_QUERY_USAGE)

    return query_text, {
        "insertion_family": insertion_family,
        "sliding_layer_indices": sliding_layer_indices,
        "sliding_head_indices": sliding_head_indices,
    }


def _select_kv_direct_kwargs(
    retriever: Any,
    *,
    insertion_family: KVInsertionFamily,
    sliding_layer_indices: tuple[int, ...] | None,
    sliding_head_indices: tuple[int, ...] | None,
) -> dict[str, Any]:
    selector_kwargs = {
        "insertion_family": insertion_family,
        "sliding_layer_indices": sliding_layer_indices,
        "sliding_head_indices": sliding_head_indices,
    }
    explicit_selector = (
        insertion_family != "full_attention"
        or sliding_layer_indices is not None
        or sliding_head_indices is not None
    )
    if not explicit_selector:
        return {}

    def _assert_selector_support(callable_obj: Any, *, label: str) -> None:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "kv_query selector requested, but "
                f"{label} cannot be introspected for selector support"
            ) from exc

        parameters = signature.parameters
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            return
        supported = {key for key in selector_kwargs if key in parameters}
        if supported == set(selector_kwargs):
            return
        missing = ", ".join(key for key in selector_kwargs if key not in supported)
        raise RuntimeError(
            f"kv_query selector requested, but {label} is missing selector kwargs: {missing}"
        )

    _assert_selector_support(
        retriever.answer_with_kv_direct,
        label="retriever.answer_with_kv_direct",
    )
    runtime = getattr(retriever, "runtime", None)
    runtime_generate = (
        None
        if runtime is None
        else getattr(runtime, "generate_with_kv_direct_materialization", None)
    )
    if runtime_generate is not None:
        _assert_selector_support(
            runtime_generate,
            label="runtime.generate_with_kv_direct_materialization",
        )
    return selector_kwargs


# ─── metadata container ─────────────────────────────────────────────────────


@dataclass
class TurnMetadata:
    mode: str  # "plain", "topical", "exact", "entity_mention", "none"
    routing_mode: str | None = None
    source_session: str | None = None
    window_id: int | None = None
    routing_score: float | None = None
    matched_window_text: str | None = None
    window_keywords: list[str] = field(default_factory=list)
    strict_assertions: dict[str, bool] = field(default_factory=dict)
    retrieve_time: float = 0.0
    generate_time: float = 0.0
    total_time: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    generated_answer: str | None = None
    # ── axis-6 (repl-observability) fields ─────────────────────────────────
    # Defaults are only sentinel values for non-KV retrieval paths.
    # The real `/kv_query` path overwrites these with truthful runtime
    # values from router/tier/KV-direct execution. `no_silent_fallback`
    # is always computed truthfully.
    selected_tier: str = "not-implemented-yet"
    mask_penalty_applied: bool = False
    kv_direct_active: bool = False
    vram_peak_mib: float | None = None
    vram_delta_mib: float | None = None
    no_silent_fallback: bool = False
    candidate_count: int = 0
    tier_assignment_count: int = 0
    budgeted_assignment_count: int = 0
    multi_session_count: int = 0
    semantic_prefix_active: bool = False
    semantic_prefix_tokens: int = 0
    memory_used: bool = False
    fallback_reason: str | None = None
    no_memory_detected: bool = False
    fallback_is_explicit_if_used: bool = False
    candidate_recall_at_4: int | None = None
    candidate_recall_at_8: int | None = None
    candidate_recall_at_12: int | None = None
    candidate_recall_at_64: int | None = None
    evidence_supports: list[dict[str, object]] = field(default_factory=list)
    evidence_support_count: int = 0

    def pretty_print(self) -> None:
        if self.mode == "plain":
            print(
                "  [plain turn] memory_used=false "
                f"generated {self.generated_tokens} tok in {self.generate_time:.2f}s",
                flush=True,
            )
            return
        if self.mode == "none":
            reason = f" reason={self.fallback_reason}" if self.fallback_reason else ""
            print(
                "  [memory] memory_used=false "
                f"generated {self.generated_tokens} tok in {self.generate_time:.2f}s"
                f"{reason}",
                flush=True,
            )
            return
        if self.mode == "plain":
            print(
                f"  [plain turn] no retrieval · generated {self.generated_tokens} tok in {self.generate_time:.2f}s",
                flush=True,
            )
            return
        if self.mode == "none":
            print(
                f"  [no memory yet] plain stream · {self.generated_tokens} tok in {self.generate_time:.2f}s",
                flush=True,
            )
            return
        section("ROUTING + STRICT ASSERTIONS")
        print(f"  mode            : {self.routing_mode}")
        print(f"  source_session  : {self.source_session}")
        print(f"  window_id       : {self.window_id}")
        if self.routing_score is not None:
            print(f"  routing_score   : {self.routing_score:.4f}")
        else:
            print("  routing_score   : (exact — no score)")
        print(f"  matched_window  : {truncate(self.matched_window_text or '', 220)}")
        if self.window_keywords:
            print(f"  keywords        : {self.window_keywords[:10]}")
        if self.strict_assertions:
            asserts = " ".join(f"{k}={v}" for k, v in self.strict_assertions.items())
            print(f"  strict_asserts  : {asserts}")
        # ── axis-6 observability ───────────────────────────────────────────
        # Sentinel tag flags non-KV retrieval paths. When the real
        # KV-direct path ran (``kv_direct_active`` True OR
        # mode == "kv_direct"), the fields below carry real values and
        # the sentinel tag is omitted.
        # ``no_silent_fallback`` is always computed truthfully so it
        # carries no sentinel tag regardless.
        real_kv = bool(self.kv_direct_active) or self.mode == "kv_direct"
        _sent = "" if real_kv else " (sentinel: non-kv path)"
        print("  axis-6 observability:")
        print(f"    memory_used          : {self.memory_used}")
        print(f"    selected_tier        : {self.selected_tier}{_sent}")
        print(f"    mask_penalty_applied : {self.mask_penalty_applied}{_sent}")
        print(f"    kv_direct_active     : {self.kv_direct_active}{_sent}")
        print(f"    vram_peak_mib        : {self.vram_peak_mib}{_sent}")
        print(f"    vram_delta_mib       : {self.vram_delta_mib}{_sent}")
        print(f"    no_silent_fallback   : {self.no_silent_fallback}")
        print(f"    candidate_count      : {self.candidate_count}{_sent}")
        print(f"    tier_assignments     : {self.tier_assignment_count}{_sent}")
        print(f"    budgeted_assignments : {self.budgeted_assignment_count}{_sent}")
        print(f"    multi_session_count  : {self.multi_session_count}{_sent}")
        print(f"    semantic_prefix      : {self.semantic_prefix_active}{_sent}")
        print(f"    semantic_prefix_tok  : {self.semantic_prefix_tokens}{_sent}")
        print(f"    evidence_supports    : {self.evidence_support_count}{_sent}")
        print(f"    no_memory_detected   : {self.no_memory_detected}{_sent}")
        print(f"    fallback_explicit    : {self.fallback_is_explicit_if_used}")
        if any(
            value is not None
            for value in (
                self.candidate_recall_at_4,
                self.candidate_recall_at_8,
                self.candidate_recall_at_12,
                self.candidate_recall_at_64,
            )
        ):
            print(
                "    candidate_recall@K : "
                f"4={self.candidate_recall_at_4} "
                f"8={self.candidate_recall_at_8} "
                f"12={self.candidate_recall_at_12} "
                f"64={self.candidate_recall_at_64}",
                flush=True,
            )
        print(
            f"  timing (s)      : retrieve={self.retrieve_time:.2f}  generate={self.generate_time:.2f}  total={self.total_time:.2f}"
        )
        print(f"  tokens          : prompt={self.prompt_tokens}  generated={self.generated_tokens}")
        print("=" * HEADER_W)


# ─── memory chat orchestrator ───────────────────────────────────────────────


class MemoryChat:
    def __init__(
        self,
        *,
        store_root: Path,
        model_path: str | None,
        max_new_tokens: int,
        memory_mode: str,
        device: str,
        generation_engine: str = "standard",
        hot_budget_mib: int | None = None,
        session_cache_size: int | None = None,
        memory_profile: str = "plug_and_play",
        memory_config: MemoryRecallConfig | None = None,
        agent_tools_enabled: bool = False,
        agent_tools_workspace: Path | None = None,
        agent_tools_timeout: int = 30,
        agent_tools_max_steps: int = 4,
        agent_tools_custom_dir: Path | None = None,
    ) -> None:
        self.store_root = Path(store_root)
        self.inputs_root = self.store_root / "inputs"
        self.checkpoints_root = self.store_root / "checkpoints"
        self.transcripts_root = self.store_root / "transcripts"
        self.tool_traces_root = self.store_root / "tool_traces"
        self.inputs_root.mkdir(parents=True, exist_ok=True)
        self.checkpoints_root.mkdir(parents=True, exist_ok=True)
        self.transcripts_root.mkdir(parents=True, exist_ok=True)
        self.tool_traces_root.mkdir(parents=True, exist_ok=True)

        if memory_config is None:
            memory_config = resolve_memory_config(
                profile=memory_profile,
                overrides={
                    "mode": memory_mode,
                    "active_generation_engine": generation_engine,
                    "hot_budget_mib": hot_budget_mib,
                },
            )
        if memory_config.mode not in SCRIPT_MEMORY_MODES:
            raise ValueError(
                f"memory mode {memory_config.mode!r} is not supported by this REPL; "
                f"supported modes: {', '.join(SCRIPT_MEMORY_MODES)}"
            )
        self.memory_profile = memory_config.profile
        self.memory_config = memory_config

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.memory_mode = memory_config.mode
        self.device = device
        self.generation_engine = _normalize_generation_engine(generation_engine)
        self.hot_budget_mib = (
            int(hot_budget_mib)
            if hot_budget_mib is not None
            else (
                int(memory_config.hot_budget_mib)
                if self.generation_engine == "residual_bounded_kv_direct"
                else None
            )
        )
        self.session_cache_size = session_cache_size
        self.agent_tools_enabled = bool(agent_tools_enabled)
        self.agent_tools_workspace = Path(agent_tools_workspace or Path.cwd()).resolve()
        self.agent_tools_custom_dir = Path(
            agent_tools_custom_dir or self.store_root / "custom_tools"
        ).resolve()
        self.agent_tools_timeout = int(agent_tools_timeout)
        self.agent_tools_max_steps = int(agent_tools_max_steps)
        self.agent_tool_runner: LocalCodingToolRunner | None = None
        self._agent_tool_prompt_added = False
        if self.agent_tools_enabled:
            self.agent_tool_runner = LocalCodingToolRunner(
                workspace_root=self.agent_tools_workspace,
                trace_root=self.tool_traces_root,
                timeout_seconds=self.agent_tools_timeout,
                custom_tools_root=self.agent_tools_custom_dir,
            )

        self.tokenizer: Any = None
        self.model: Any = None
        self.runtime: Any = None
        self.session_cache: Any = None
        self.retriever: Any = None  # SessionRetriever | None
        self.session: Any = None  # ChatLoopSession
        self.history: Any = None  # ChatHistory

        # Live clause-aligned indexer. Allocated fresh per session in
        # start_new_session(); flushed-and-closed in save_current_session().
        self.indexer: LiveIndexer | None = None
        # Per-session monotonic window id (reset when a new session starts).
        # ChunkBoundary.chunk_index resets per turn, but LiveIndexer requires
        # a session-unique id (windows across turns share the same on-disk
        # store). We assign ids ourselves instead of reusing chunk_index.
        self._window_counter: int = 0

        self.last_meta: TurnMetadata | None = None

    def _current_memory_config(self, **overrides: Any) -> MemoryRecallConfig:
        base = getattr(self, "memory_config", None)
        if base is None:
            base = resolve_memory_config()
        merged_overrides = {
            "mode": getattr(self, "memory_mode", None),
            **overrides,
        }
        return resolve_memory_config(base=base, overrides=merged_overrides)

    def _agent_tool_system_prompt(self) -> str | None:
        if not self.agent_tools_enabled:
            return None
        return self._ensure_agent_tool_runner().tool_system_prompt()

    def _ensure_agent_tool_runner(self) -> LocalCodingToolRunner:
        if self.agent_tool_runner is None:
            self.agent_tool_runner = LocalCodingToolRunner(
                workspace_root=self.agent_tools_workspace,
                trace_root=self.tool_traces_root,
                timeout_seconds=self.agent_tools_timeout,
                custom_tools_root=self.agent_tools_custom_dir,
            )
        return self.agent_tool_runner

    def _inject_agent_tool_system_prompt(self) -> None:
        prompt = self._agent_tool_system_prompt()
        if prompt is None or self.history is None or self._agent_tool_prompt_added:
            return
        self.history.add_system(prompt)
        self._agent_tool_prompt_added = True

    def _inject_agent_tool_retriever_prompt(self) -> None:
        prompt = self._agent_tool_system_prompt()
        if prompt is None or self.retriever is None:
            return
        current_prompt = getattr(self.retriever, "system_prompt", "")
        if isinstance(current_prompt, str) and prompt not in current_prompt:
            self.retriever.system_prompt = current_prompt + "\n\n" + prompt

    @contextmanager
    def _memory_runtime_env(self, config: MemoryRecallConfig):
        updates = memory_config_env(config)
        previous = {key: os.environ.get(key) for key in updates}
        try:
            for key, value in updates.items():
                os.environ[key] = value
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    # ── machinery loaders ──────────────────────────────────────────────────

    def load_model(self) -> None:
        info(f"loading model (device={self.device}) — first load is slow…")
        from chuk_lazarus.chat_loop.cli import load_gemma

        t0 = time.time()
        self.tokenizer, self.model = load_gemma(self.model_path, device=self.device)
        self._configure_generation_runtime()
        info(f"model loaded in {time.time() - t0:.1f}s")

    def _configure_generation_runtime(self) -> None:
        """Build the optional bounded generation runtime around the live model."""
        self.runtime = None
        self.session_cache = None
        if self.generation_engine == "standard":
            return
        if self.model is None or self.tokenizer is None:
            return

        from chuk_lazarus.inference.backends.torch_runtime import TorchInferenceRuntime

        self.runtime = TorchInferenceRuntime(
            self.model,
            self.tokenizer,
            device=self.device,
            engine=self.generation_engine,
            hot_budget_mib=self.hot_budget_mib,
        )
        self.session_cache = self._install_runtime_session_cache(self.runtime)
        cache_note = (
            f" session_cache={self.session_cache.max_sessions}"
            if self.session_cache is not None
            else ""
        )
        budget_note = (
            f" hot_budget_mib={self.hot_budget_mib}" if self.hot_budget_mib is not None else ""
        )
        info(f"generation runtime ready: engine={self.generation_engine}{budget_note}{cache_note}")

    def _install_runtime_session_cache(self, runtime: Any) -> Any | None:
        if getattr(runtime, "engine_mode", None) != "residual_bounded_kv_direct":
            return None
        attach = getattr(runtime, "attach_session_cache", None)
        if not callable(attach):
            return None

        from chuk_lazarus.server._residual_session_cache import (
            DEFAULT_MAX_SESSIONS,
            ResidualSessionCache,
        )

        eviction_hook = None
        try:
            from chuk_lazarus.inference.backends._torch_residual_session import (
                make_cuda_eviction_hook,
            )

            eviction_hook = make_cuda_eviction_hook()
        except Exception:  # pragma: no cover - defensive on non-torch hosts
            eviction_hook = None

        cache = ResidualSessionCache(
            max_sessions=int(self.session_cache_size or DEFAULT_MAX_SESSIONS),
            eviction_hook=eviction_hook,
        )
        attach(cache)
        return cache

    def maybe_load_retriever(self) -> None:
        """Build SessionRetriever if at least one checkpoint exists under store."""
        from chuk_lazarus.session_retrieval.enumeration import iter_checkpoint_handles

        handles = list(iter_checkpoint_handles(self.checkpoints_root))
        if not handles:
            info(f"store is empty at {self.checkpoints_root} — no retriever yet")
            self.retriever = None
            return

        info(
            f"found {len(handles)} checkpoint(s) under {self.checkpoints_root} — building retriever"
        )
        from chuk_lazarus.session_retrieval.retriever import SessionRetriever

        # System prompt tuned for chat (vs. the retriever's default recall-only prompt)
        chat_system = (
            "You are a helpful assistant with access to a growing memory of prior "
            "conversation sessions. When the user asks about earlier turns, planted "
            "phrases, or previous sessions, use the provided context excerpt to "
            "answer accurately; quote verbatim when the user asks for exact content. "
            "When no context is relevant, chat normally."
        )
        tool_prompt = self._agent_tool_system_prompt()
        if tool_prompt:
            chat_system = chat_system + "\n\n" + tool_prompt

        # SessionRetriever.from_checkpoint_root loads its own model copy — to avoid
        # doubling memory we pass the same model_id; torch/transformers caches weights.
        model_id = self.model_path or "google/gemma-4-E2B-it"
        # Local-snapshot fallback (mirrors load_gemma behaviour)
        local_snapshot = (
            "/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/"
            "snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
        )
        if os.path.isdir(local_snapshot):
            model_id = local_snapshot

        t0 = time.time()
        self.retriever = SessionRetriever.from_checkpoint_root(
            self.checkpoints_root,
            model_id=model_id,
            device=self.device,
            system_prompt=chat_system,
            engine=self.generation_engine,
            hot_budget_mib=self.hot_budget_mib,
            session_cache_size=self.session_cache_size,
        )
        info(
            f"retriever ready: {len(handles)} session(s) indexed · "
            f"crystal_layer={self.retriever.crystal_layer} · "
            f"loaded in {time.time() - t0:.1f}s"
        )

    def _mark_dirty(self) -> None:
        """axis-3 (Addition 2): mark the store as having unsaved tokens.

        Writes <store_root>/.dirty as a 0-byte sentinel. Best-effort —
        any OS error is logged and swallowed; the chat must never crash
        because of a flag-write failure (e.g. read-only mount, ENOSPC).
        """
        store_root = getattr(self, "store_root", None)
        if store_root is None:
            return
        try:
            (store_root / DIRTY_FLAG_FILENAME).touch(exist_ok=True)
        except OSError as exc:
            info(f"  .dirty mark FAILED (non-fatal): {exc!r}")

    def start_new_session(self) -> None:
        from chuk_lazarus.chat_loop.session import ChatLoopSession
        from chuk_lazarus.inference.chat import ChatHistory

        self.session = ChatLoopSession()
        self.history = ChatHistory()
        self._agent_tool_prompt_added = False
        self._inject_agent_tool_system_prompt()
        self._window_counter = 0
        self.indexer = self._make_live_indexer(self.session.session_id)
        info(f"new session started · session_id={self.session.session_id}")

    # ── live-indexer wiring ────────────────────────────────────────────────

    def _derive_arch_config(self) -> tuple[dict[str, Any], int, int]:
        """Build the LiveIndexer arch_config dict.

        Mirrors tools/build_clause_aligned_store.py (line ~760) and
        src/chuk_lazarus/cli/commands/context/prefill/_torch_sidecar.py
        (line ~148). Falls back to the empirically validated Gemma 4 E2B
        (35-layer) defaults if ArchitectureConfig cannot resolve the model.
        """
        # Gemma-4 E2B requires KV-sharing-aware injection config.
        # Only layers 0-14 have their own k_proj/v_proj. Layers 15-34 reuse K/V
        # from layer 13 (sliding) or layer 14 (full). Axis-5 KV-direct targets
        # layer 14 (full-attention master, store_full_length_kv=True); its K/V
        # is inherited by all downstream full-attention layers (19, 24, 29, 34)
        # via shared_kv_states. Therefore injection_layer=13 (→ target=14).
        #
        # Detect model family and force KV-direct-compatible values.
        model_family = type(self.model).__name__.lower()
        is_gemma4 = "gemma4" in model_family

        # Start with family-appropriate defaults.
        if is_gemma4:
            retrieval_layer = 12
            query_head = 7
            injection_layer = 13
            hidden_dim = 1536
            head_dim = 256
            crystal_layer = 13
            window_size = 512
        else:
            # Legacy Gemma-3 / other defaults.
            retrieval_layer = 28
            query_head = 7
            injection_layer = 29
            hidden_dim = 1536
            head_dim = 256
            crystal_layer = 29
            window_size = 512

        try:
            from chuk_lazarus.inference.context.knowledge.config import (
                ArchitectureConfig,
            )

            ac = ArchitectureConfig.from_model_config(self.model.config)
            # For Gemma-4, DO NOT let the registry override our KV-sharing-aware
            # layer picks — the registry was set up for non-KV-sharing Gemma-3
            # geometry. Only consume hidden_dim / head_dim / window_size.
            if not is_gemma4:
                retrieval_layer = int(getattr(ac, "retrieval_layer", retrieval_layer))
                query_head = int(getattr(ac, "query_head", query_head))
                injection_layer = int(getattr(ac, "injection_layer", injection_layer))
                ac_crystal_layer = int(getattr(ac, "crystal_layer", -1))
                if ac_crystal_layer >= 0:
                    crystal_layer = ac_crystal_layer
            ac_hidden_dim = int(getattr(ac, "hidden_dim", 0) or 0)
            if ac_hidden_dim > 0:
                hidden_dim = ac_hidden_dim
            ac_head_dim = int(getattr(ac, "head_dim", 0) or 0)
            if ac_head_dim > 0:
                head_dim = ac_head_dim
            ac_window_size = int(getattr(ac, "window_size", window_size))
            if ac_window_size > 0:
                window_size = ac_window_size
        except Exception as exc:  # noqa: BLE001
            info(
                f"  arch_config fallback engaged ({exc!r}); using hard-coded defaults "
                f"(family_is_gemma4={is_gemma4})"
            )
        info(
            f"  arch_config: family_is_gemma4={is_gemma4} "
            f"retrieval_layer={retrieval_layer} injection_layer={injection_layer} "
            f"crystal_layer={crystal_layer}"
        )

        arch_config = {
            "retrieval_layer": int(retrieval_layer),
            "query_head": int(query_head),
            "injection_layer": int(injection_layer),
            "hidden_dim": int(hidden_dim),
            "head_dim": int(head_dim),
            "crystal_layer": int(crystal_layer),
            "window_size": int(window_size),
        }
        return arch_config, int(crystal_layer), int(window_size)

    def _make_live_indexer(self, session_id: str) -> LiveIndexer | None:
        """Instantiate a LiveIndexer rooted at <checkpoints_root>/<sid>/torch_store.

        Returns None if the model/tokenizer haven't been loaded yet (e.g. when
        tests construct MemoryChat without calling run_repl). The REPL flow
        always loads them before calling start_new_session, so in practice
        this branch only trips in tests.
        """
        if self.model is None or self.tokenizer is None:
            info("  live indexer skipped: model/tokenizer not loaded yet")
            return None

        # Matches the layout iter_checkpoint_handles expects:
        #   <checkpoint_root>/<session_id>/torch_store/manifest.json
        ts_dir = self.checkpoints_root / session_id / "torch_store"
        ts_dir.mkdir(parents=True, exist_ok=True)

        arch_config, crystal_layer, window_size = self._derive_arch_config()
        try:
            return LiveIndexer(
                model=self.model,
                tokenizer=self.tokenizer,
                checkpoint_dir=ts_dir,
                crystal_layer=crystal_layer,
                window_size=window_size,
                arch_config=arch_config,
            )
        except Exception as exc:  # noqa: BLE001
            info(f"  live indexer init FAILED: {exc!r}; running without live indexing")
            return None

    def _make_on_window(self, turn: Any):
        """Return a StreamingWindower on_window adapter closure for `turn`.

        The closure bridges the StreamingWindower's (boundary, decoded_text)
        signature to LiveIndexer.enqueue's (window_id, token_ids, keywords,
        clause_metadata) signature. Every closure invocation produces a
        session-monotonic window_id via self._window_counter — we do NOT
        reuse boundary.chunk_index because that resets per turn.
        """
        # Import locally so tests that patch these names still work and so
        # that module-import cost stays off the cold path.
        from chuk_lazarus.inference.context.knowledge.route import (
            _extract_keywords_from_text,
        )

        def _on_window(boundary: Any, decoded_text: str) -> None:
            if self.indexer is None:
                # Defensive no-op: test harnesses can instantiate
                # StreamingWindower without a LiveIndexer wired up.
                return
            try:
                # Re-tokenize the decoded window text. This round-trips
                # through the tokenizer (cheap; sub-ms for 512-token
                # windows) instead of maintaining a parallel token buffer
                # inside the windower — which would require modifying
                # streaming.py (out of scope for this axis).
                token_ids = self.tokenizer(decoded_text, add_special_tokens=False).input_ids
                keywords = _extract_keywords_from_text(decoded_text, max_keywords=12)
                sid = self.session.session_id if self.session is not None else ""
                role_str = getattr(turn.role, "value", str(turn.role))
                clause_metadata = {
                    "session_id": sid,
                    "turn_index": int(turn.turn_index),
                    "role": role_str,
                    "chunk_index": int(boundary.chunk_index),
                    "start_token_offset": int(boundary.start_token_offset),
                    "end_token_offset": int(boundary.end_token_offset),
                    "emitted_at": str(boundary.emitted_at),
                    "token_count": int(boundary.end_token_offset - boundary.start_token_offset),
                }
                wid = self._window_counter
                self._window_counter += 1
                self.indexer.enqueue(wid, token_ids, keywords, clause_metadata)
            except Exception as exc:  # noqa: BLE001
                # Never let a capture hiccup kill the streaming reply.
                info(f"  live-index enqueue failed on chunk {boundary.chunk_index}: {exc!r}")

        return _on_window

    def _capture_turn_text_live(self, turn: Any) -> None:
        """Window a completed turn's text through the live indexer path."""
        if self.indexer is None:
            return
        if self.session is None or self.tokenizer is None:
            return
        turn_text = getattr(turn, "text", "")
        if not turn_text:
            return

        from chuk_lazarus.chat_loop.streaming import StreamingWindower

        windower = StreamingWindower(
            self.tokenizer,
            on_window=self._make_on_window(turn),
        )
        for boundary in windower.feed_text(turn_text):
            self.session.append_chunk(turn, boundary)
        tail = windower.flush()
        if tail is not None:
            self.session.append_chunk(turn, tail)

    # ── plain chat turn (no memory injection) ──────────────────────────────

    def _stream_runtime_assistant_reply(
        self,
        *,
        windower: Any,
        turn: Any,
        on_text_delta: Any = None,
        on_chunk: Any = None,
    ) -> str:
        """Stream via TorchInferenceRuntime so bounded engines get session reuse."""
        assert self.runtime is not None
        assert self.session is not None

        from chuk_lazarus.inference.chat import format_history
        from chuk_lazarus.inference.generation import GenerationConfig

        prompt = format_history(
            self.tokenizer,
            self.history,
            add_generation_prompt=True,
        )
        config = GenerationConfig(
            max_new_tokens=int(self.max_new_tokens),
            temperature=float(os.environ.get("LAZARUS_GENERATION_TEMPERATURE", "0.7")),
            top_p=float(os.environ.get("LAZARUS_GENERATION_TOP_P", "0.9")),
        )

        accumulated: list[str] = []
        for text_delta in self.runtime.stream_generate(
            prompt,
            config,
            conversation_id=self.session.session_id,
        ):
            if not text_delta:
                continue
            accumulated.append(text_delta)
            if on_text_delta is not None:
                on_text_delta(text_delta)
            boundaries = windower.feed_text(text_delta)
            for boundary in boundaries:
                self.session.append_chunk(turn, boundary)
                if on_chunk is not None:
                    decoded = self.tokenizer.decode(
                        windower._token_ids[
                            boundary.start_token_offset : boundary.end_token_offset
                        ],
                        skip_special_tokens=True,
                    )
                    on_chunk(boundary, decoded)

        tail = windower.flush()
        if tail is not None:
            self.session.append_chunk(turn, tail)
            if on_chunk is not None:
                decoded = self.tokenizer.decode(
                    windower._token_ids[tail.start_token_offset : tail.end_token_offset],
                    skip_special_tokens=True,
                )
                on_chunk(tail, decoded)

        self.session.finish_turn(turn)
        return "".join(accumulated)

    def plain_chat_turn(self, user_text: str) -> TurnMetadata:
        from chuk_lazarus.chat_loop.streaming import StreamingWindower
        from chuk_lazarus.inference.chat import Role

        self.history.add_user(user_text)
        user_turn = self.session.begin_turn(Role.USER, user_text)
        self.session.finish_turn(user_turn)
        self._capture_turn_text_live(user_turn)

        assistant_turn = self.session.begin_turn(Role.ASSISTANT, "")
        windower = StreamingWindower(
            self.tokenizer,
            on_window=self._make_on_window(assistant_turn),
        )

        sys.stdout.write("gemma> ")
        sys.stdout.flush()

        def _echo(delta: str) -> None:
            sys.stdout.write(delta)
            sys.stdout.flush()

        t0 = time.perf_counter()
        if self.runtime is not None and self.generation_engine != "standard":
            full_reply = self._stream_runtime_assistant_reply(
                windower=windower,
                turn=assistant_turn,
                on_text_delta=_echo,
            )
        else:
            from chuk_lazarus.chat_loop.cli import stream_assistant_reply

            full_reply = stream_assistant_reply(
                model=self.model,
                tokenizer=self.tokenizer,
                history=self.history,
                windower=windower,
                session=self.session,
                turn=assistant_turn,
                max_new_tokens=self.max_new_tokens,
                on_text_delta=_echo,
            )
        elapsed = time.perf_counter() - t0

        assistant_turn.text = full_reply
        self.history.add_assistant(full_reply)
        sys.stdout.write("\n")
        sys.stdout.flush()

        gen_ids = self.tokenizer(full_reply, add_special_tokens=False).input_ids
        self._mark_dirty()
        return TurnMetadata(
            mode="plain" if self.retriever is None else "none",
            generate_time=elapsed,
            total_time=elapsed,
            generated_tokens=len(gen_ids),
            generated_answer=full_reply,
        )

    # ── recall turn (residual-injected) ────────────────────────────────────

    def recall_chat_turn(self, user_text: str) -> TurnMetadata:
        """Route via retriever; inject matched window's residual via the new method."""
        assert self.retriever is not None
        from chuk_lazarus.inference.chat import Role

        # Record user turn in history + session
        self.history.add_user(user_text)
        user_turn = self.session.begin_turn(Role.USER, user_text)
        self.session.finish_turn(user_turn)

        # Format a chat-context question the retriever can route on. Include
        # recent turns so topical routing sees the context, not just the
        # trailing user line.
        chat_context = self._format_recent_history_for_routing(user_text)

        # kv_direct routing short-circuits to KV-direct injection (axis-5).
        # kv_query_turn handles its own session/history bookkeeping (it
        # uses plain_chat_turn on silent-fallback), so rewind here to
        # avoid double-logging.
        if self.memory_mode in {"auto", "kv_direct"}:
            self.session.turns.pop()
            if self.history.messages:
                self.history.messages.pop()
            return self.memory_recall_turn(
                user_text,
                debug_command=False,
                answer_prefix="gemma",
            )

        t_total_start = time.time()
        t_retrieve_start = time.time()
        meta = TurnMetadata(mode="topical")
        try:
            if self.memory_mode == "topical":
                result = self.retriever.query_topical(
                    chat_context,
                    conversation_id=self.session.session_id,
                )
            elif self.memory_mode == "entity_mention":
                result = self.retriever.query_entity_mention(
                    chat_context,
                    conversation_id=self.session.session_id,
                )
            else:
                raise RuntimeError(f"unknown memory_mode {self.memory_mode!r}")
        except (ValueError, RuntimeError) as exc:
            # No candidate — fall back to plain chat. Still produce a reply.
            meta.mode = "none"
            info(f"recall fallback: {exc}")
            # axis-6: SILENT FALLBACK is a first-class failure verdict.
            # Surface it as a WARN (not an info line) so operators see it
            # plainly. no_silent_fallback stays False (the default) here
            # because the intended retrieval path did NOT run end-to-end.
            print(
                f"[WARN] axis-6: SILENT FALLBACK DETECTED — mode={self.memory_mode} "
                f"reason=recall_exception",
                flush=True,
            )
            # Rewind the session/history to avoid duplicate user turn
            self.session.turns.pop()
            if self.history.messages:
                self.history.messages.pop()
            return self.plain_chat_turn(user_text)

        t_retrieve = time.time() - t_retrieve_start
        self._capture_turn_text_live(user_turn)

        # Populate metadata
        meta.routing_mode = result.routing_mode
        meta.source_session = result.source_session
        meta.window_id = result.window_id
        meta.routing_score = result.routing_score
        meta.matched_window_text = result.matched_window_text
        meta.window_keywords = list(result.window_keywords or [])
        meta.strict_assertions = dict(result.strict_assertions or {})
        meta.generated_answer = result.generated_answer
        meta.retrieve_time = t_retrieve

        # Token counts for the reply
        gen_ids = self.tokenizer(result.generated_answer, add_special_tokens=False).input_ids
        meta.generated_tokens = len(gen_ids)
        # Prompt tokens approximation: system + matched window + context
        prompt_preview = (
            self.retriever.system_prompt + "\n" + result.matched_window_text + "\n" + chat_context
        )
        meta.prompt_tokens = len(self.tokenizer(prompt_preview, add_special_tokens=False).input_ids)
        meta.generate_time = meta.total_time = time.time() - t_total_start
        # (generate_time ≈ total - retrieve is more accurate, but we don't
        # split out the non-retrieve path inside the retriever; report total.)
        meta.generate_time = max(0.0, meta.total_time - meta.retrieve_time)

        # axis-6: compute no_silent_fallback truthfully. The intended
        # retrieval-path actually ran end-to-end iff the mode is not 'none'
        # AND a window was selected AND we reached this point without the
        # except-branch having returned (we are below the except block, so
        # by construction no recall exception was caught this turn).
        # selected_tier remains the sentinel on this non-KV path.
        # Deriving a tier here would fabricate evidence; only /kv_query
        # carries real tier data.
        meta.no_silent_fallback = meta.mode != "none" and meta.window_id is not None
        meta.memory_used = True

        # Append assistant turn to session/history
        assistant_turn = self.session.begin_turn(Role.ASSISTANT, result.generated_answer)
        self.session.finish_turn(assistant_turn)
        self.history.add_assistant(result.generated_answer)

        # Print debug block BEFORE the reply so you see routing first
        meta.pretty_print()
        print(f"gemma> {result.generated_answer}\n", flush=True)
        self._mark_dirty()
        return meta

    # ── axis-4 (Addition 1) token-budget governor ──────────────────────────

    def _apply_token_budget(
        self,
        tier_assignments: list[Any],
        *,
        max_total_inject_tokens: int | None = None,
    ) -> list[Any]:
        """axis-4 (Addition 1): cap ASI-routed assignments under MAX_TOTAL_INJECT_TOKENS.

        Sorts by candidate ucb1_score (descending) — Pattern A item 1.
        Computes per-fact cost = head_dim * num_kv_heads from
        ``self.model.config`` (HuggingFace Gemma-4 native fields) — Pattern A
        item 3. Accumulates until the next fact would exceed
        ``max_total_inject_tokens``; truncates from the bottom — Pattern A
        item 2. Truncated facts are silently dropped — Pattern A item 4
        (graceful degradation under budget pressure).

        Returns the kept assignments. Empty input returns empty output
        without raising.
        """
        if not tier_assignments:
            return tier_assignments
        if max_total_inject_tokens is None:
            try:
                max_total_inject_tokens = self._current_memory_config().max_total_inject_tokens
            except Exception:
                max_total_inject_tokens = _env_max_total_inject_tokens()
        else:
            max_total_inject_tokens = int(max_total_inject_tokens)
        cfg = getattr(getattr(self, "model", None), "config", None)
        # Gemma-4-E2B-it ships head_dim=256, num_key_value_heads=4. Fall
        # back to those literals when config attributes are missing.
        head_dim = int(getattr(cfg, "head_dim", 256) or 256)
        n_kv_heads = int(getattr(cfg, "num_key_value_heads", 4) or 4)
        per_fact_cost = head_dim * n_kv_heads
        if per_fact_cost <= 0:
            return tier_assignments
        sorted_assignments = sorted(
            tier_assignments,
            key=lambda a: float(
                getattr(
                    a.candidate,
                    "utility_score",
                    getattr(a.candidate, "ucb1_score", 0.0),
                )
                or getattr(a.candidate, "ucb1_score", 0.0)
            ),
            reverse=True,
        )
        cumulative = 0
        kept: list[Any] = []
        for a in sorted_assignments:
            if cumulative + per_fact_cost > max_total_inject_tokens:
                break
            kept.append(a)
            cumulative += per_fact_cost
        if len(kept) < len(tier_assignments):
            info(
                f"  axis-4 token-budget: kept {len(kept)}/{len(tier_assignments)} "
                f"facts ({cumulative}/{max_total_inject_tokens} tokens; "
                f"per_fact={per_fact_cost})"
            )
        return kept

    # ── axis-5 KV-direct recall turn ───────────────────────────────────────

    def memory_recall_turn(
        self,
        query_text: str,
        *,
        insertion_family: KVInsertionFamily = "full_attention",
        sliding_layer_indices: tuple[int, ...] | None = None,
        sliding_head_indices: tuple[int, ...] | None = None,
        debug_command: bool = False,
        answer_prefix: str = "gemma",
    ) -> TurnMetadata:
        """Run a single bounded KV-direct memory recall turn.

        ``/kv_query`` remains a debug wrapper around this method, while normal
        chat in memory_mode=auto calls it directly when saved memories exist.

        Returns a :class:`TurnMetadata` populated with the REAL axis-6
        observability fields from ``result.strict_assertions``. On
        silent-fallback conditions (router returns nothing, or
        ``answer_with_kv_direct`` raises), prints a first-class WARN,
        sets ``no_silent_fallback=False``, and falls back to plain chat
        (same pattern as ``recall_chat_turn``).
        """
        if self.retriever is None:
            info("kv_query skipped: no retriever (store is empty — /save something first)")
            return TurnMetadata(mode="none", no_silent_fallback=False)

        from chuk_lazarus.inference.backends.torch_runtime import WarmPenaltyConfig
        from chuk_lazarus.inference.context.knowledge.gemma4_e2b_it_layers import (
            GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS,
        )
        from chuk_lazarus.inference.generation import GenerationConfig

        session_retrieval = sys.modules.get("chuk_lazarus.session_retrieval")
        if session_retrieval is None:
            import chuk_lazarus.session_retrieval as session_retrieval

        POLICY_VERSION_UTILITY_V2 = getattr(
            session_retrieval,
            "POLICY_VERSION_UTILITY_V2",
            "utility-v2",
        )
        asi_route_candidates = session_retrieval.asi_route_candidates
        assign_tiers = session_retrieval.assign_tiers
        record_asi_feedback = getattr(session_retrieval, "record_asi_feedback", None)

        # AMD 11: sliding-window-hazard precondition. The recipe-canonical
        # target_layer for KV-direct injection is 29 (Gemma-4-E2B-it global
        # attention). Assert the fixture matches; the actual runtime guard
        # lives in the KV-direct selector validation.
        assert 29 in GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS, (
            "AMD 11 invariant: target_layer=29 must be in "
            f"GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS={sorted(GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS)}"
        )

        config = self._current_memory_config()
        candidate_pool = int(config.candidate_pool)
        route_candidate_pool = int(config.route_candidate_pool)
        k_hot = int(config.k_hot)
        k_warm = int(config.k_warm)
        hot_budget_mib = int(config.hot_budget_mib)
        for env_key, env_value in memory_config_env(config).items():
            os.environ[env_key] = env_value
        # LAZARUS_KV_HOT_BONUS: pre-softmax additive bonus for HOT slots
        # (float, default 0.0 = no bonus; positive values boost HOT attention).
        hot_bonus_value = float(os.environ.get("LAZARUS_KV_HOT_BONUS", "0.0"))
        explicit_selector = (
            insertion_family != "full_attention"
            or sliding_layer_indices is not None
            or sliding_head_indices is not None
        )

        t_total_start = time.time()
        t_retrieve_start = time.time()

        # Build a meta stub; populate fields as we progress.
        meta = TurnMetadata(mode="kv_direct")
        tier_assignments: list[Any] = []
        assignments_for_generation: list[Any] = []
        feedback_archive_root = getattr(self, "store_root", None)

        try:
            cfg = getattr(getattr(self, "model", None), "config", None)
            head_dim = int(getattr(cfg, "head_dim", 256) or 256)
            n_kv_heads = int(getattr(cfg, "num_key_value_heads", 4) or 4)
            per_fact_cost = max(1, head_dim * n_kv_heads)
            if str(os.environ.get("LAZARUS_ASI_COST_MODE", "windows")).strip().lower() == "tokens":
                selector_budget = int(config.max_total_inject_tokens)
            else:
                selector_budget = int(config.max_total_inject_tokens) // per_fact_cost
            candidates = asi_route_candidates(
                self.retriever.handles,
                query_text,
                self.retriever.tokenizer,
                candidate_pool=route_candidate_pool,
                archive_root=feedback_archive_root,
                dense_scoring=config.dense_scoring,
                rrf_k=float(config.rrf_k),
                ranking_policy=config.selector_policy,
            )
            if bool(config.dedupe_sessions):
                deduped = []
                seen_sessions: set[str] = set()
                for candidate in candidates:
                    session_id = str(candidate.handle.session_id)
                    if session_id in seen_sessions:
                        continue
                    seen_sessions.add(session_id)
                    deduped.append(candidate)
                candidates = deduped
            required_substring = (
                str(os.environ.get("LAZARUS_KV_REQUIRED_SUBSTRING", "") or "").strip().lower()
            )
            if required_substring:
                from chuk_lazarus.session_retrieval.enumeration import load_store

                filtered = []
                for candidate in candidates:
                    store = load_store(candidate.handle)
                    window_text = store.get_window_text(
                        int(candidate.window_id),
                        self.retriever.tokenizer,
                    ).lower()
                    if required_substring in window_text:
                        filtered.append(candidate)
                candidates = filtered
            if not candidates:
                raise NoRelevantMemory("asi_route_candidates returned no candidates")
            top_raw_score = max(
                float(
                    getattr(
                        candidate,
                        "raw_tfidf_score_pre_normalization",
                        1.0,
                    )
                )
                for candidate in candidates
            )
            if top_raw_score <= 0.0:
                raise NoRelevantMemory("routing found no relevant memory")
            meta.candidate_count = int(len(candidates))

            tier_assignments = assign_tiers(
                candidates,
                K_HOT=k_hot,
                K_WARM=k_warm,
                candidate_pool=candidate_pool,
                policy_version=config.selector_policy,
                budget=selector_budget,
                mmr_lambda=float(config.mmr_lambda),
                rrf_k=float(config.rrf_k),
            )
            if not tier_assignments:
                raise NoRelevantMemory("assign_tiers produced zero assignments")
            meta.tier_assignment_count = int(len(tier_assignments))

            # axis-4 (Addition 1): apply the token-budget governor BEFORE
            # materialization. The real retriever can now consume assignments
            # across multiple sessions; legacy test doubles fall back to the
            # original single-handle path below.
            assignments_for_generation = self._apply_token_budget(
                list(tier_assignments),
                max_total_inject_tokens=int(config.max_total_inject_tokens),
            )
            if not assignments_for_generation:
                raise RuntimeError("axis-4 token-budget governor truncated all assignments")
            dynamic_policy_active = any(
                str(getattr(a, "policy_version", "")) == POLICY_VERSION_UTILITY_V2
                for a in tier_assignments
            )
            if dynamic_policy_active or not config.include_cold_in_kv:
                assignments_for_generation = [
                    assignment
                    for assignment in assignments_for_generation
                    if str(getattr(assignment.tier, "value", assignment.tier)) != "cold"
                ]
                if not assignments_for_generation:
                    raise NoRelevantMemory("only cold memories matched")
            meta.budgeted_assignment_count = int(len(assignments_for_generation))

            warm_config = WarmPenaltyConfig(hot_bonus_value=hot_bonus_value)
            gen_config = GenerationConfig(
                max_new_tokens=int(self.max_new_tokens),
                temperature=0.0,
                top_p=1.0,
            )

            selector_kwargs = _select_kv_direct_kwargs(
                self.retriever,
                insertion_family=insertion_family,
                sliding_layer_indices=sliding_layer_indices,
                sliding_head_indices=sliding_head_indices,
            )
            if explicit_selector:
                info(
                    "kv_query selector: "
                    f"insertion_family={selector_kwargs['insertion_family']} "
                    f"sliding_layer_indices="
                    f"{selector_kwargs['sliding_layer_indices']} "
                    f"sliding_head_indices="
                    f"{selector_kwargs['sliding_head_indices']}"
                )

            answer_multi = getattr(
                self.retriever,
                "answer_with_kv_direct_multi",
                None,
            )
            if callable(answer_multi):
                result = answer_multi(
                    query_text,
                    assignments_for_generation,
                    hot_budget_mib=hot_budget_mib,
                    warm_config=warm_config,
                    generation_config=gen_config,
                    **selector_kwargs,
                )
            else:
                # Compatibility path for older retrievers and unit-test
                # doubles. It intentionally preserves the old single-session
                # materialization behaviour when the multi-handle API is not
                # available.
                top_handle = assignments_for_generation[0].candidate.handle
                assignments_for_generation = [
                    a
                    for a in assignments_for_generation
                    if a.candidate.handle.session_id == top_handle.session_id
                ]
                if not assignments_for_generation:
                    raise RuntimeError("no tier assignments owned by top-ranked candidate's handle")
                result = self.retriever.answer_with_kv_direct(
                    query_text,
                    assignments_for_generation,
                    hot_budget_mib=hot_budget_mib,
                    warm_config=warm_config,
                    generation_config=gen_config,
                    handle=top_handle,
                    **selector_kwargs,
                )
        except (NoRelevantMemory, ValueError, RuntimeError) as exc:
            if (
                feedback_archive_root is not None
                and assignments_for_generation
                and callable(record_asi_feedback)
            ):
                try:
                    reward = _asi_feedback_reward(
                        {},
                        [],
                        fallback=True,
                    )
                    record_asi_feedback(
                        Path(feedback_archive_root),
                        assignments_for_generation,
                        reward=reward,
                        outcome=f"fallback:{type(exc).__name__}",
                    )
                except Exception as feedback_exc:  # noqa: BLE001
                    info(f"asi feedback persist FAILED (non-fatal): {feedback_exc!r}")
            ordinary_miss = isinstance(exc, NoRelevantMemory)
            if ordinary_miss and not debug_command:
                info(f"memory recall found no relevant memory: {exc}")
                fallback_meta = self.plain_chat_turn(query_text)
                fallback_meta.memory_used = False
                fallback_meta.fallback_reason = "no_relevant_memory"
                fallback_meta.fallback_is_explicit_if_used = True
                fallback_meta.no_silent_fallback = True
                return fallback_meta
            # axis-6: SILENT FALLBACK DETECTED — first-class WARN.
            print(
                f"[WARN] axis-6: SILENT FALLBACK DETECTED — mode=kv_direct "
                f"reason={type(exc).__name__}: {exc}",
                flush=True,
            )
            meta.mode = "none"
            meta.no_silent_fallback = False
            info(f"kv_direct fallback to plain chat: {exc}")
            return self.plain_chat_turn(query_text)

        t_retrieve = time.time() - t_retrieve_start

        # Populate metadata with the REAL axis-5 strict_assertions payload.
        kv_strict = dict(result.strict_assertions or {})
        meta.routing_mode = result.routing_mode
        meta.source_session = result.source_session
        meta.window_id = result.window_id
        meta.routing_score = result.routing_score
        meta.matched_window_text = result.matched_window_text or ""
        meta.window_keywords = list(result.window_keywords or [])
        # Keep legacy strict_assertions dict available for the pretty-print
        # `strict_asserts` line; these are the axis-5 KV-direct keys, NOT
        # the six strict-mode booleans (the strict booleans do not apply
        # to the KV-direct path — axis-5 defines its own invariants).
        meta.strict_assertions = {k: bool(v) for k, v in kv_strict.items()}
        meta.generated_answer = result.generated_answer
        meta.retrieve_time = t_retrieve
        answer_lower = str(result.generated_answer or "").lower()
        meta.no_memory_detected = (
            "do not have" in answer_lower
            or "don't have" in answer_lower
            or "no stored" in answer_lower
        ) and "stored decision" in answer_lower
        meta.fallback_is_explicit_if_used = bool(meta.no_memory_detected or meta.no_silent_fallback)

        # axis-6 observability fields — REAL values from result.
        meta.memory_used = True
        meta.kv_direct_active = bool(kv_strict.get("kv_direct_active", False))
        meta.mask_penalty_applied = bool(kv_strict.get("mask_penalty_applied", False))
        meta.vram_peak_mib = kv_strict.get("vram_peak_mib", None)
        meta.vram_delta_mib = kv_strict.get("vram_delta_mib", None)
        meta.multi_session_count = int(
            kv_strict.get(
                "multi_session_count",
                len(
                    {
                        str(getattr(a.candidate.handle, "session_id", ""))
                        for a in assignments_for_generation
                    }
                ),
            )
        )
        meta.semantic_prefix_active = bool(kv_strict.get("semantic_prefix_active", False))
        meta.semantic_prefix_tokens = int(kv_strict.get("semantic_prefix_tokens", 0))
        evidence_supports = list(getattr(result, "evidence_supports", []) or [])
        meta.evidence_supports = evidence_supports
        meta.evidence_support_count = int(
            kv_strict.get("evidence_support_count", len(evidence_supports))
        )
        if (
            feedback_archive_root is not None
            and assignments_for_generation
            and callable(record_asi_feedback)
        ):
            try:
                reward = _asi_feedback_reward(kv_strict, evidence_supports)
                record_asi_feedback(
                    Path(feedback_archive_root),
                    assignments_for_generation,
                    reward=reward,
                    outcome="kv_direct",
                )
            except Exception as feedback_exc:  # noqa: BLE001
                info(f"asi feedback persist FAILED (non-fatal): {feedback_exc!r}")
        # selected_tier: report the highest-priority tier that contributed to
        # this generation. Multi-fact queries can include HOT and WARM slots in
        # one answer, but the selected tier remains the top active tier.
        tiers_used = {str(getattr(a.tier, "value", a.tier)) for a in assignments_for_generation}
        if not tiers_used:
            meta.selected_tier = "kv_direct"
        elif "hot" in tiers_used:
            meta.selected_tier = "hot"
        elif "warm" in tiers_used:
            meta.selected_tier = "warm"
        elif "cold" in tiers_used:
            meta.selected_tier = "cold"
        else:
            meta.selected_tier = "mixed"

        # no_silent_fallback: True iff window_id is set (>= 0) AND KV-direct
        # path actually ran (kv_direct_active flag from the runtime).
        meta.no_silent_fallback = bool(
            meta.window_id is not None and int(meta.window_id) != -1 and meta.kv_direct_active
        )
        meta.memory_used = bool(meta.no_silent_fallback)
        meta.fallback_is_explicit_if_used = bool(meta.no_memory_detected or meta.no_silent_fallback)

        # Token counts for the reply.
        gen_ids = self.tokenizer(result.generated_answer, add_special_tokens=False).input_ids
        meta.generated_tokens = len(gen_ids)
        # Prompt tokens approximation: system + query.
        prompt_preview = (self.retriever.system_prompt or "") + "\n" + query_text
        meta.prompt_tokens = len(self.tokenizer(prompt_preview, add_special_tokens=False).input_ids)
        meta.total_time = time.time() - t_total_start
        meta.generate_time = max(0.0, meta.total_time - meta.retrieve_time)

        if (
            not debug_command
            and getattr(self, "session", None) is not None
            and getattr(self, "history", None) is not None
        ):
            from chuk_lazarus.inference.chat import Role

            self.history.add_user(query_text)
            user_turn = self.session.begin_turn(Role.USER, query_text)
            self.session.finish_turn(user_turn)
            self._capture_turn_text_live(user_turn)

            assistant_turn = self.session.begin_turn(
                Role.ASSISTANT,
                result.generated_answer,
            )
            self.session.finish_turn(assistant_turn)
            self._capture_turn_text_live(assistant_turn)
            self.history.add_assistant(result.generated_answer)

        # Print debug block, then the answer.
        meta.pretty_print()
        print(f"{answer_prefix}> {result.generated_answer}\n", flush=True)
        self._mark_dirty()
        return meta

    def kv_query_turn(
        self,
        query_text: str,
        *,
        insertion_family: KVInsertionFamily = "full_attention",
        sliding_layer_indices: tuple[int, ...] | None = None,
        sliding_head_indices: tuple[int, ...] | None = None,
    ) -> TurnMetadata:
        """Compatibility/debug wrapper for the historical /kv_query path."""
        return self.memory_recall_turn(
            query_text,
            insertion_family=insertion_family,
            sliding_layer_indices=sliding_layer_indices,
            sliding_head_indices=sliding_head_indices,
            debug_command=True,
            answer_prefix="kv_direct",
        )

    def _format_recent_history_for_routing(self, user_text: str, n_recent: int = 4) -> str:
        """Serialize the last n turns + the new user line as the routing query."""
        recent: list[str] = []
        for turn in self.session.turns[-n_recent:]:
            role = getattr(turn.role, "value", str(turn.role))
            recent.append(f"{role}: {turn.text}")
        recent.append(f"user: {user_text}")
        return "\n".join(recent)

    # ── save / close session, rebuild store ────────────────────────────────

    def save_current_session(self, *, rebuild_retriever: bool = True) -> bool:
        """Flush live index → refresh retriever. Idempotent.

        The per-turn LiveIndexer has already written boundaries and shard
        records to ``<checkpoints_root>/<sid>/torch_store/``. /save just has
        to drain + compact + write the manifest (``flush_and_close``), then
        point the retriever at the new handle.

        AUS3000 per-session inputs are still emitted for audit and for the
        offline/CI ``invoke_build`` rebuild entry point — they are not
        consumed at runtime.
        """
        from chuk_lazarus.session_close.wind_down import emit_session

        assert self.session is not None
        if not self.session.turns:
            info("save skipped: session has no turns yet.")
            return False

        sid = self.session.session_id
        info(f"saving session {sid} · {len(self.session.turns)} turns")

        # 1. Persist the transcript JSON for audit
        transcript_path = self.transcripts_root / f"{sid}.json"
        transcript_path.write_text(
            json.dumps(
                {
                    "session_id": sid,
                    "started_at": self.session.started_at,
                    "turns": [t.model_dump(mode="json") for t in self.session.turns],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        info(f"  transcript -> {transcript_path}")

        # 2. Emit AUS3000 clause JSON files under inputs_root/<sid>/.
        #    These are NOT consumed at runtime anymore; they remain as the
        #    canonical input for the offline `invoke_build` CI/rebuild entry
        #    and for post-mortem debugging.
        written = emit_session(
            list(self.session.turns),
            sid,
            self.inputs_root,
        )
        info(f"  AUS3000 clauses -> {len(written)} record(s) under {self.inputs_root / sid}")

        # 3. Flush-and-close the live indexer. This drains any still-queued
        #    windows, compacts the per-window shards into canonical
        #    window_tokens.npz/keywords.json/window_metadata.json/idf.json,
        #    and writes manifest.json LAST (the readiness gate). After this
        #    returns the per-session torch_store is load()able by
        #    TorchKnowledgeStore. No subprocess; no Gemma reload.
        t0 = time.time()
        if self.indexer is not None:
            try:
                self.indexer.flush_and_close()
            except RuntimeError as exc:
                info(f"  indexer flush FAILED: {exc}")
                return False
            # Convention: /save closes this session's indexing. If the user
            # keeps chatting after /save without /new, we skip live-indexing
            # those trailing turns (deterministic: the saved manifest
            # already reflects exactly what was committed). /new or the next
            # start_new_session() allocates a fresh indexer. The
            # _make_on_window closure's `self.indexer is None` guard
            # degrades newly-emitted windows into a no-op.
            self.indexer = None
        else:
            info("  no live indexer to flush (session had no streamed windows)")
        dt = time.time() - t0
        info(f"  live index flushed in {dt:.2f}s")

        # 3b. axis-1+axis-6 (asi-kv-direct-chat): write per-session
        # save-state.json checkpoint artifact. Best-effort: a failure here
        # MUST NOT block /save. Captures completion of the indexer flush
        # plus all the metadata available at this moment. kv_direct_ready
        # is hard-coded False; axes 2-5 (runtime fields) have not landed.
        try:
            session_root = self.checkpoints_root / sid
            torch_store_dir = session_root / "torch_store"
            manifest_abs = torch_store_dir / "manifest.json"
            boundaries_abs = torch_store_dir / "boundaries"
            residual_streams_abs = torch_store_dir / "residual_streams"
            selection_ready_abs = torch_store_dir / "selection_ready"

            # Window count: prefer the manifest the indexer just wrote, fall
            # back to counting boundary files if manifest is missing/unreadable.
            window_count = 0
            try:
                if manifest_abs.exists():
                    manifest_obj = json.loads(manifest_abs.read_text(encoding="utf-8"))
                    window_count = int(
                        manifest_obj.get("num_entries", manifest_obj.get("num_windows", 0))
                    )
            except Exception:  # noqa: BLE001 - best-effort; counting fallback
                window_count = 0

            boundary_residual_count = (
                sum(1 for _ in boundaries_abs.glob("window_*.npy"))
                if boundaries_abs.exists()
                else 0
            )
            residual_stream_count = (
                sum(1 for _ in residual_streams_abs.glob("window_*.npy"))
                if residual_streams_abs.exists()
                else 0
            )
            selection_ready_descriptor_count = (
                sum(1 for _ in selection_ready_abs.glob("window_*.json"))
                if selection_ready_abs.exists()
                else 0
            )
            if window_count == 0:
                window_count = boundary_residual_count

            save_state = {
                "schema_version": 1,
                "kind": "asi_kv_direct_chat_save_state",
                "session_id": sid,
                "saved_at_iso": datetime.now(timezone.utc).isoformat(),
                "entrypoint": (
                    "scripts/interactive_memory_chat.py::MemoryChat.save_current_session"
                ),
                "torch_store_path": "torch_store",
                "manifest_path": "torch_store/manifest.json",
                "window_count": int(window_count),
                "selection_ready_descriptor_dir": "torch_store/selection_ready",
                "selection_ready_descriptor_count": int(selection_ready_descriptor_count),
                "boundary_residual_dir": "torch_store/boundaries",
                "boundary_residual_count": int(boundary_residual_count),
                "residual_stream_dir": "torch_store/residual_streams",
                "residual_stream_count": int(residual_stream_count),
                "transcript_path": str(transcript_path),
                "aus3000_clauses_count": int(len(written)),
                "indexer_flush_seconds": float(dt),
                "kv_direct_ready": False,
                "kv_direct_pending_reason": (
                    "axes 2-5 runtime fields not yet landed (lead-runtime-router scope)"
                ),
                "axis_1_closure_artifact": True,
                "baseline_copy_artifact_id": "ve-ins-0mo9oppka000042c2c9",
                "manifest_id": "ve-ins-0mo9oke0k0000ec93cc",
            }
            save_state_path = session_root / "save-state.json"
            session_root.mkdir(parents=True, exist_ok=True)
            save_state_path.write_text(
                json.dumps(save_state, indent=2),
                encoding="utf-8",
            )
            info(f"  save-state -> {save_state_path}")
        except Exception as exc:  # noqa: BLE001 - additive; never block /save
            info(f"  save-state write FAILED (non-fatal): {exc!r}")

        # 4. Refresh retriever so subsequent turns can see the new session.
        #    First-save path: no retriever exists yet — lazy-load once
        #    (this is Gemma load #3, one-time). Subsequent saves: just
        #    re-enumerate handles on the existing retriever — no reload.
        if rebuild_retriever:
            if self.retriever is None:
                self.maybe_load_retriever()
            else:
                t0 = time.time()
                try:
                    added = self.retriever.refresh_handles(self.checkpoints_root)
                except RuntimeError as exc:
                    info(f"  retriever refresh FAILED: {exc}")
                    return False
                info(f"  retriever refreshed in {time.time() - t0:.2f}s (+{added} handle(s))")
        return True

    # ── axis-3 (Q3 + Addition 2): unified emit_store entry ─────────────────

    def emit_store(self, *, force: bool = False) -> bool:
        """axis-3 unified emit. Conversation state -> store. Idempotent.

        Q3 supervisor decision (binding):
            - /save calls emit_store (force=False)
            - session-end (atexit / signal handler / on quit-command)
              calls emit_store (force=False)
            - emit_store is the single entry; both triggers share semantics

        Addition 2 dirty-flag lifecycle:
            - Reads <store_root>/.dirty
            - When the flag is absent and force=False: prints
              "no changes since last save" and skips (graceful no-op)
            - Otherwise delegates to save_current_session() and clears
              .dirty on success
            - Idempotent across crashes (output files overwrite cleanly)

        Args:
            force: when True, bypass the .dirty check and emit
                   unconditionally (used by tests; not surfaced to the
                   user via /save).

        Returns:
            True iff save_current_session succeeded; False on no-op,
            empty session, or save_current_session failure.
        """
        dirty_path = self.store_root / DIRTY_FLAG_FILENAME
        is_dirty = dirty_path.exists() or force
        if not is_dirty:
            info("no changes since last save")
            return False
        if self.session is None or not self.session.turns:
            # Nothing to save — clear stale flag and exit cleanly.
            try:
                dirty_path.unlink(missing_ok=True)
            except OSError as exc:
                info(f"  .dirty unlink FAILED (non-fatal): {exc!r}")
            return False
        ok = self.save_current_session()
        if ok:
            try:
                dirty_path.unlink(missing_ok=True)
            except OSError as exc:
                info(f"  .dirty unlink FAILED (non-fatal): {exc!r}")
        return ok

    # ── stats ──────────────────────────────────────────────────────────────

    def print_stats(self) -> None:
        from chuk_lazarus.session_retrieval.enumeration import iter_checkpoint_handles, load_store

        section("STORE STATS")
        handles = list(iter_checkpoint_handles(self.checkpoints_root))
        print(f"  store_root       : {self.store_root}")
        print(f"  sessions indexed : {len(handles)}")
        total_windows = 0
        total_tokens = 0
        for h in handles:
            try:
                s = load_store(h)
                n_windows = len(getattr(s, "keywords", []) or [])
                tokens = int(getattr(s.config, "total_tokens", 0) or 0)
                total_windows += n_windows
                total_tokens += tokens
                print(f"   · {h.session_id[:8]}… windows={n_windows} tokens={tokens}")
            except Exception as exc:  # noqa: BLE001
                print(f"   · {h.session_id[:8]}… LOAD-FAIL: {exc}")
        print(f"  total_windows    : {total_windows}")
        print(f"  total_tokens     : {total_tokens}")
        if self.retriever is not None:
            print(f"  crystal_layer    : {self.retriever.crystal_layer}")
            print(f"  memory_mode      : {self.memory_mode}")
            print(f"  memory_profile   : {getattr(self, 'memory_profile', 'plug_and_play')}")
            print(f"  device           : {self.retriever.device}")
        print(
            f"  current_session  : session_id={self.session.session_id if self.session else '—'}  turns={len(self.session.turns) if self.session else 0}"
        )
        print("=" * HEADER_W)

    # ── probes (do NOT mutate chat history) ────────────────────────────────

    def probe(self, mode: str, query: str) -> None:
        if self.retriever is None:
            info("probe skipped: no retriever (store is empty — /save something first)")
            return
        try:
            if mode == "topical":
                result = self.retriever.query_topical(query)
            elif mode == "exact":
                result = self.retriever.query_exact_id(query)
            elif mode == "entity_mention":
                result = self.retriever.query_entity_mention(query)
            else:
                info(f"unknown probe mode {mode!r}")
                return
        except (ValueError, RuntimeError) as exc:
            info(f"probe fail ({mode}): {exc}")
            return

        meta = TurnMetadata(
            mode=mode,
            routing_mode=result.routing_mode,
            source_session=result.source_session,
            window_id=result.window_id,
            routing_score=result.routing_score,
            matched_window_text=result.matched_window_text,
            window_keywords=list(result.window_keywords or []),
            strict_assertions=dict(result.strict_assertions or {}),
            generated_answer=result.generated_answer,
        )
        gen_ids = self.tokenizer(result.generated_answer, add_special_tokens=False).input_ids
        meta.generated_tokens = len(gen_ids)
        meta.pretty_print()
        print(f"probe answer> {result.generated_answer}\n", flush=True)

    # ── coding-agent tool loop ────────────────────────────────────────────

    def _print_agent_tools_status(self) -> None:
        runner = self.agent_tool_runner
        custom_names: list[str] = []
        if runner is not None:
            custom_names = sorted(runner.reload_custom_tools())
        section("CODING-AGENT TOOLS")
        print(f"  enabled     : {self.agent_tools_enabled}")
        print(f"  workspace   : {self.agent_tools_workspace}")
        print(f"  trace_root  : {self.tool_traces_root}")
        print(f"  custom_dir  : {self.agent_tools_custom_dir}")
        print(f"  custom_tools: {', '.join(custom_names) if custom_names else 'none'}")
        print(f"  timeout_s   : {self.agent_tools_timeout}")
        print(f"  max_steps   : {self.agent_tools_max_steps}")
        print("=" * HEADER_W)

    def _run_agent_tool_loop(self, seed_meta: TurnMetadata) -> TurnMetadata:
        """Execute model-requested tools and feed results back as transcript turns.

        The durable transcript and JSONL tool traces are source-of-truth records.
        Live memory captures the synthetic TOOL_RESULT turns as derived state.
        """
        if not self.agent_tools_enabled:
            return seed_meta
        answer = seed_meta.generated_answer or ""
        calls = extract_tool_calls(answer)
        if not calls:
            return seed_meta

        runner = self._ensure_agent_tool_runner()
        meta = seed_meta
        steps = 0
        while calls and steps < max(0, self.agent_tools_max_steps):
            turn_index = None
            if self.session is not None and self.session.turns:
                turn_index = int(getattr(self.session.turns[-1], "turn_index", -1))
            session_id = self.session.session_id if self.session is not None else None
            results = [
                runner.execute(call, session_id=session_id, turn_index=turn_index)
                for call in calls
            ]
            for result in results:
                status = "ok" if result.ok else "failed"
                print(f"[tool] {result.name} {result.call_id} {status}", flush=True)

            # This synthetic user turn is intentional: it makes tool observations
            # replayable transcript events, while the memory index remains rebuildable.
            meta = self.plain_chat_turn(format_tool_results(results))
            self.last_meta = meta
            steps += 1
            calls = extract_tool_calls(meta.generated_answer or "")

        if calls:
            info(
                "tool loop stopped at max_steps="
                f"{self.agent_tools_max_steps}; remaining tool calls ignored"
            )
        return meta

    # ── main REPL ──────────────────────────────────────────────────────────

    def run_repl(self) -> None:
        self.load_model()
        self.maybe_load_retriever()
        self.start_new_session()
        # axis-3 (Q3): register session-end emit as an atexit hook.
        # Provides the non-interactive crash-safe path; the interactive
        # /quit and EOF branches also call emit_store directly.
        # emit_store is .dirty-gated and idempotent so double-firing
        # is safe.
        atexit.register(self.emit_store)
        self.print_stats()

        print()
        print(rule("═"))
        print(" interactive memory chat ready. type /help for commands.")
        print(rule("═"))
        print()

        while True:
            try:
                user_text = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                # axis-3 (Q3): session-end auto-emit. emit_store is a
                # no-op when .dirty is absent; safe to call unconditionally.
                self.emit_store()
                info("bye.")
                return

            stripped = user_text.strip()
            if not stripped:
                continue

            if stripped.startswith("/"):
                if self._handle_command(stripped):
                    return  # quit requested
                continue

            # Regular turn
            if self.retriever is None or self.memory_mode == "off":
                meta = self.plain_chat_turn(stripped)
                meta.pretty_print()
            else:
                meta = self.recall_chat_turn(stripped)
            if self.agent_tools_enabled:
                meta = self._run_agent_tool_loop(meta)
            self.last_meta = meta

    def _handle_command(self, cmd: str) -> bool:
        """Return True if the caller should quit."""
        parts = cmd.split(maxsplit=1)
        head = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if head in ("/quit", "/exit"):
            # axis-3 (Q3): session-end auto-emit; .dirty-gated.
            self.emit_store()
            info("bye.")
            return True

        if head == "/help":
            print(HELP_TEXT)
            return False

        if head == "/tools":
            tools_arg = arg.strip().lower() or "status"
            if tools_arg == "status":
                self._print_agent_tools_status()
            elif tools_arg == "schema":
                print(self._ensure_agent_tool_runner().tool_system_prompt())
            elif tools_arg == "reload":
                runner = self._ensure_agent_tool_runner()
                runner.reload_custom_tools()
                print(runner.describe_custom_tools())
            elif tools_arg == "on":
                self.agent_tools_enabled = True
                self._ensure_agent_tool_runner()
                self._inject_agent_tool_system_prompt()
                self._inject_agent_tool_retriever_prompt()
                info("coding-agent tools enabled for future turns")
            elif tools_arg == "off":
                self.agent_tools_enabled = False
                info("coding-agent tool execution disabled")
            else:
                info("usage: /tools status|schema|reload|on|off")
            return False

        if head in ("/save", "/save-memory"):
            self.emit_store()
            return False

        if head in ("/new", "/new-chat"):
            if self.session and self.session.turns:
                self.save_current_session()
            self.start_new_session()
            info("fresh session started — try asking about prior sessions.")
            return False

        if head in ("/stats", "/memory-status"):
            self.print_stats()
            return False

        if head in ("/last", "/last-memory"):
            if self.last_meta is None:
                info("no turn yet.")
            else:
                self.last_meta.pretty_print()
            return False

        if head in ("/history", "/show-history"):
            section("CURRENT SESSION TRANSCRIPT")
            print(f"  session_id: {self.session.session_id}")
            for t in self.session.turns:
                role = getattr(t.role, "value", str(t.role))
                print(f"  [{t.turn_index:03d}] {role}: {truncate(t.text, 180)}")
            print("=" * HEADER_W)
            return False

        if head == "/memory":
            memory_arg = arg.strip()
            memory_parts = memory_arg.split()
            if memory_arg == "config":
                config = self._current_memory_config()
                section("MEMORY CONFIG")
                for key, value in config.to_dict().items():
                    print(f"  {key:26}: {value}")
                print("=" * HEADER_W)
            elif memory_parts[:1] == ["profile"] and len(memory_parts) == 2:
                try:
                    config = resolve_memory_config(profile=memory_parts[1])
                except ValueError as exc:
                    info(str(exc))
                    return False
                if config.mode not in SCRIPT_MEMORY_MODES:
                    info(
                        f"memory profile {config.profile!r} resolves to unsupported "
                        f"mode {config.mode!r}; supported modes: "
                        f"{', '.join(SCRIPT_MEMORY_MODES)}"
                    )
                    return False
                self.memory_profile = config.profile
                self.memory_config = config
                self.memory_mode = config.mode
                info(f"memory_profile = {self.memory_profile}; memory_mode = {self.memory_mode}")
            elif memory_arg in SCRIPT_MEMORY_MODES:
                self.memory_mode = memory_arg
                info(f"memory_mode = {self.memory_mode}")
            elif memory_arg:
                info(
                    f"memory_mode {memory_arg!r} is not supported by this REPL; "
                    f"supported modes: {', '.join(SCRIPT_MEMORY_MODES)}"
                )
            else:
                self.memory_mode = "off" if self.memory_mode != "off" else "auto"
                info(f"memory_mode = {self.memory_mode}")
            return False

        if head in ("/query", "/search-memory"):
            if not arg:
                info("usage: /search-memory <text>")
                return False
            self.probe("topical", arg)
            return False

        if head in ("/exact", "/open-memory"):
            if not arg:
                info("usage: /open-memory <dotted-handle>  (e.g. 11a1c9ad.1.0)")
                return False
            self.probe("exact", arg)
            return False

        if head in ("/entity", "/find-memory-entity"):
            if not arg:
                info("usage: /find-memory-entity <text>")
                return False
            self.probe("entity_mention", arg)
            return False

        if head in ("/kv_query", "/ask-memory"):
            if not arg:
                info(f"usage: /ask-memory <question>  (advanced: {KV_QUERY_USAGE})")
                return False
            try:
                query_text, selector_kwargs = _parse_kv_query_args(arg)
            except ValueError as exc:
                info(f"usage: {exc}")
                return False
            meta = self.kv_query_turn(query_text, **selector_kwargs)
            self.last_meta = meta
            return False

        info(f"unknown command {head!r} — try /help")
        return False


# ─── entry point ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    default_profile = os.environ.get("LAZARUS_MEMORY_PROFILE", "plug_and_play")
    default_memory_config = resolve_memory_config(profile=default_profile)
    parser = argparse.ArgumentParser(
        prog="interactive_memory_chat.py",
        description="Interactive pseudo-infinite-memory chat with routing metadata.",
        epilog=(
            "Tip: use scripts/run_interactive_memory_chat.sh for safer manual "
            "REPL launches with a fresh repo-local store."
        ),
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=Path(os.environ.get("LAZARUS_STORE_DIR", DEFAULT_STORE)),
        help=f"Persistent store root (default: {DEFAULT_STORE}).",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("LAZARUS_MODEL"),
        help="Override model path (default: local Gemma snapshot, fallback hub id).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("LAZARUS_MAX_NEW_TOKENS", "180")),
        help="Max new tokens per assistant turn.",
    )
    parser.add_argument(
        "--memory-profile",
        choices=get_memory_profile_names(),
        default=default_memory_config.profile,
        help="Named memory preset (default: plug_and_play).",
    )
    parser.add_argument(
        "--memory-mode",
        choices=SCRIPT_MEMORY_MODES,
        default=None,
        help="Recall routing mode for post-save turns.",
    )
    parser.add_argument(
        "--generation-engine",
        choices=GENERATION_ENGINES,
        default=None,
        help=(
            "Torch generation engine for live turns. Use "
            "residual_bounded_kv_direct for video-style bounded active KV."
        ),
    )
    parser.add_argument(
        "--memory-k-hot",
        type=int,
        default=None,
        help="HOT memory tier count.",
    )
    parser.add_argument(
        "--memory-k-warm",
        type=int,
        default=None,
        help="WARM memory tier count.",
    )
    parser.add_argument(
        "--memory-candidate-pool",
        type=int,
        default=None,
        help="Tiering candidate pool size.",
    )
    parser.add_argument(
        "--memory-route-candidate-pool",
        type=int,
        default=None,
        help="Router candidate pool size.",
    )
    parser.add_argument(
        "--memory-max-inject-tokens",
        type=int,
        default=None,
        help="Token-equivalent memory injection budget.",
    )
    parser.add_argument(
        "--memory-semantic-prefix-tokens",
        type=int,
        default=None,
        help="Semantic prefix token budget.",
    )
    parser.add_argument(
        "--memory-hot-budget-mib",
        "--hot-budget-mib",
        type=int,
        dest="hot_budget_mib",
        default=None,
        help="Hot KV budget for bounded generation engines.",
    )
    parser.add_argument(
        "--session-cache-size",
        type=int,
        default=(
            int(os.environ["LAZARUS_SESSION_CACHE_SIZE"])
            if "LAZARUS_SESSION_CACHE_SIZE" in os.environ
            else None
        ),
        help="Residual session cache entries for residual_bounded_kv_direct.",
    )
    parser.add_argument(
        "--enable-agent-tools",
        action="store_true",
        default=_env_truthy("LAZARUS_AGENT_TOOLS"),
        help="Enable local coding-agent tools for model-emitted tool_call JSON.",
    )
    parser.add_argument(
        "--agent-tools-workspace",
        type=Path,
        default=Path(os.environ.get("LAZARUS_AGENT_TOOLS_WORKSPACE", Path.cwd())),
        help="Workspace root for coding-agent tools.",
    )
    parser.add_argument(
        "--agent-tools-timeout",
        type=int,
        default=int(os.environ.get("LAZARUS_AGENT_TOOLS_TIMEOUT", "30")),
        help="Maximum seconds for each coding-agent tool command.",
    )
    parser.add_argument(
        "--agent-tools-max-steps",
        type=int,
        default=int(os.environ.get("LAZARUS_AGENT_TOOLS_MAX_STEPS", "4")),
        help="Maximum tool-result feedback turns per user turn.",
    )
    parser.add_argument(
        "--agent-tools-custom-dir",
        type=Path,
        default=(
            Path(os.environ["LAZARUS_AGENT_TOOLS_CUSTOM_DIR"])
            if "LAZARUS_AGENT_TOOLS_CUSTOM_DIR" in os.environ
            else None
        ),
        help="Persistent directory for model-authored custom tools.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default cuda).",
    )
    args = parser.parse_args(argv)
    memory_config = resolve_memory_config(
        profile=args.memory_profile,
        overrides={
            "mode": args.memory_mode,
            "active_generation_engine": args.generation_engine,
            "hot_budget_mib": args.hot_budget_mib,
            "k_hot": args.memory_k_hot,
            "k_warm": args.memory_k_warm,
            "candidate_pool": args.memory_candidate_pool,
            "route_candidate_pool": args.memory_route_candidate_pool,
            "max_total_inject_tokens": args.memory_max_inject_tokens,
            "semantic_prefix_tokens": args.memory_semantic_prefix_tokens,
        },
    )
    if memory_config.mode not in SCRIPT_MEMORY_MODES:
        parser.error(
            f"memory mode {memory_config.mode!r} is not supported by this REPL; "
            f"supported modes: {', '.join(SCRIPT_MEMORY_MODES)}"
        )
    args.memory_mode = memory_config.mode
    args.generation_engine = memory_config.active_generation_engine
    args.hot_budget_mib = memory_config.hot_budget_mib

    # Also honour LAZARUS_MAX_NEW_TOKENS for the retriever's generation_kwargs.
    os.environ.setdefault("LAZARUS_MAX_NEW_TOKENS", str(args.max_new_tokens))

    chat = MemoryChat(
        store_root=args.store_root,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        memory_mode=args.memory_mode,
        device=args.device,
        generation_engine=args.generation_engine,
        hot_budget_mib=args.hot_budget_mib,
        session_cache_size=args.session_cache_size,
        memory_profile=args.memory_profile,
        memory_config=memory_config,
        agent_tools_enabled=args.enable_agent_tools,
        agent_tools_workspace=args.agent_tools_workspace,
        agent_tools_timeout=args.agent_tools_timeout,
        agent_tools_max_steps=args.agent_tools_max_steps,
        agent_tools_custom_dir=args.agent_tools_custom_dir,
    )

    try:
        chat.run_repl()
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
