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
  LAZARUS_MEMORY_MODE        one of: topical (default) | entity_mention | vec_inject | off

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

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
KV_QUERY_USAGE = (
    "/kv_query <text> | "
    "/kv_query --insertion-family sliding "
    "--sliding-layer-indices 13,15 "
    "--sliding-head-indices 0,7 "
    "<text>"
)


# axis-3 (Addition 2): single dirty-flag file at <store_root>/.dirty.
# Set by every assistant turn that adds tokens; cleared by emit_store
# after a successful encode. /save is a no-op when this flag is absent.
DIRTY_FLAG_FILENAME = ".dirty"

# axis-4 (Addition 1): per-session injection token budget for the
# ASI-routed warm/hot list passed into answer_with_kv_direct. Each fact
# costs head_dim * num_kv_heads tokens-equivalent; the governor sorts
# by score and truncates from the bottom. Override via env or config
# later; for run-4 the literal default is sufficient.
MAX_TOTAL_INJECT_TOKENS = 4096


def ts() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def info(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


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
        raise ValueError(
            f"{option_name} must be a comma-separated list of integers"
        )
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f"{option_name} must be a comma-separated list of integers"
        ) from exc


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
            query_tokens = tokens[idx + 1:]
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
                raise ValueError(
                    "--insertion-family must be one of "
                    "{full_attention, sliding}"
                )
            insertion_family = family
        elif token == "--sliding-layer-indices":
            idx += 1
            if idx >= len(tokens):
                raise ValueError(f"{token} requires a value ({KV_QUERY_USAGE})")
            sliding_layer_indices = _parse_csv_int_tuple(
                tokens[idx], option_name=token
            )
        elif token == "--sliding-head-indices":
            idx += 1
            if idx >= len(tokens):
                raise ValueError(f"{token} requires a value ({KV_QUERY_USAGE})")
            sliding_head_indices = _parse_csv_int_tuple(
                tokens[idx], option_name=token
            )
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
            "--sliding-layer-indices and --sliding-head-indices require "
            "--insertion-family sliding"
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
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return
        supported = {
            key for key in selector_kwargs if key in parameters
        }
        if supported == set(selector_kwargs):
            return
        missing = ", ".join(
            key for key in selector_kwargs if key not in supported
        )
        raise RuntimeError(
            "kv_query selector requested, but "
            f"{label} is missing selector kwargs: {missing}"
        )

    _assert_selector_support(
        retriever.answer_with_kv_direct,
        label="retriever.answer_with_kv_direct",
    )
    runtime = getattr(retriever, "runtime", None)
    runtime_generate = (
        None if runtime is None
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
    vram_peak_mib: Optional[float] = None
    vram_delta_mib: Optional[float] = None
    no_silent_fallback: bool = False

    def pretty_print(self) -> None:
        if self.mode == "plain":
            print(f"  [plain turn] no retrieval · generated {self.generated_tokens} tok in {self.generate_time:.2f}s", flush=True)
            return
        if self.mode == "none":
            print(f"  [no memory yet] plain stream · {self.generated_tokens} tok in {self.generate_time:.2f}s", flush=True)
            return
        section("ROUTING + STRICT ASSERTIONS")
        print(f"  mode            : {self.routing_mode}")
        print(f"  source_session  : {self.source_session}")
        print(f"  window_id       : {self.window_id}")
        if self.routing_score is not None:
            print(f"  routing_score   : {self.routing_score:.4f}")
        else:
            print(f"  routing_score   : (exact — no score)")
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
        print(f"  axis-6 observability:")
        print(f"    selected_tier        : {self.selected_tier}{_sent}")
        print(f"    mask_penalty_applied : {self.mask_penalty_applied}{_sent}")
        print(f"    kv_direct_active     : {self.kv_direct_active}{_sent}")
        print(f"    vram_peak_mib        : {self.vram_peak_mib}{_sent}")
        print(f"    vram_delta_mib       : {self.vram_delta_mib}{_sent}")
        print(f"    no_silent_fallback   : {self.no_silent_fallback}")
        print(f"  timing (s)      : retrieve={self.retrieve_time:.2f}  generate={self.generate_time:.2f}  total={self.total_time:.2f}")
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
    ) -> None:
        self.store_root = Path(store_root)
        self.inputs_root = self.store_root / "inputs"
        self.checkpoints_root = self.store_root / "checkpoints"
        self.transcripts_root = self.store_root / "transcripts"
        self.inputs_root.mkdir(parents=True, exist_ok=True)
        self.checkpoints_root.mkdir(parents=True, exist_ok=True)
        self.transcripts_root.mkdir(parents=True, exist_ok=True)

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.memory_mode = memory_mode  # "topical" | "entity_mention" | "off"
        self.device = device
        self.vec_inject_provider: Any = None  # lazily loaded on first vec_inject turn
        # axis-BC: set True by save_current_session() when vec_inject.npz is
        # successfully written; gates the auto-promotion of memory_mode from
        # 'topical' -> 'kv_direct' inside /save.
        self.vec_inject_available: bool = False

        self.tokenizer: Any = None
        self.model: Any = None
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

    # ── machinery loaders ──────────────────────────────────────────────────

    def load_model(self) -> None:
        info(f"loading model (device={self.device}) — first load is slow…")
        from chuk_lazarus.chat_loop.cli import load_gemma

        t0 = time.time()
        self.tokenizer, self.model = load_gemma(self.model_path, device=self.device)
        info(f"model loaded in {time.time() - t0:.1f}s")

    def maybe_load_retriever(self) -> None:
        """Build SessionRetriever if at least one checkpoint exists under store."""
        from chuk_lazarus.session_retrieval.enumeration import iter_checkpoint_handles

        handles = list(iter_checkpoint_handles(self.checkpoints_root))
        if not handles:
            info(f"store is empty at {self.checkpoints_root} — no retriever yet")
            self.retriever = None
            return

        info(f"found {len(handles)} checkpoint(s) under {self.checkpoints_root} — building retriever")
        from chuk_lazarus.session_retrieval.retriever import SessionRetriever

        # System prompt tuned for chat (vs. the retriever's default recall-only prompt)
        chat_system = (
            "You are a helpful assistant with access to a growing memory of prior "
            "conversation sessions. When the user asks about earlier turns, planted "
            "phrases, or previous sessions, use the provided context excerpt to "
            "answer accurately; quote verbatim when the user asks for exact content. "
            "When no context is relevant, chat normally."
        )

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
        )
        info(
            f"retriever ready: {len(handles)} session(s) indexed · "
            f"crystal_layer={self.retriever.crystal_layer} · "
            f"loaded in {time.time() - t0:.1f}s"
        )

    def _get_or_load_vec_inject_provider(self) -> Any:
        """Lazy-load LocalVecInjectProvider (torch/CUDA) from the first checkpoint.

        Returns the cached provider if already loaded. Returns None when the
        store is empty — callers must handle this (same contract as
        maybe_load_retriever).
        """
        if self.vec_inject_provider is not None:
            return self.vec_inject_provider
        import asyncio

        from chuk_lazarus.inference.context.research.vec_inject.providers._local_file_torch import (
            LocalVecInjectProvider,
        )
        from chuk_lazarus.session_retrieval.enumeration import iter_checkpoint_handles

        handles = list(iter_checkpoint_handles(self.checkpoints_root))
        if not handles:
            info(
                f"vec_inject skipped: store is empty at {self.checkpoints_root} — "
                f"/save something first"
            )
            return None

        ckpt_dir = Path(handles[0].checkpoint_dir)
        info(f"loading vec_inject provider from {ckpt_dir}")
        t0 = time.time()
        provider = asyncio.run(
            # torch path: pass the raw HF model so the provider uses native forward hooks
            # (bypasses make_kv_generator which builds an MLX-only GemmaBackboneAdapter)
            LocalVecInjectProvider.load(ckpt_dir, raw_model=self.model)
        )
        info(f"vec_inject provider ready: {provider.n_facts} facts · loaded in {time.time() - t0:.1f}s")
        self.vec_inject_provider = provider
        return provider

    def _emit_vec_inject_npz(
        self,
        session_root: Path,
        torch_store_dir: Path,
    ) -> bool:
        """axis-BC: emit vec_inject.npz so the next session can KV-inject.

        Reads per-window token sequences from ``torch_store_dir/window_tokens.npz``,
        derives a KV-share-aware arch config (via ``_derive_arch_config``),
        and invokes ``extract_vec_inject_index_torch`` to write
        ``session_root/vec_inject.npz``. Best-effort: returns False on any
        failure (caller must NOT raise — /save stays non-blocking).

        For Gemma-4-E2B-it the registry-default ``retrieval_layer`` (28) is
        a KV-consumer layer that lacks ``k_proj`` — projecting through it
        raises ``AttributeError``. ``_derive_arch_config`` clamps Gemma-4 to
        producer layers (12, 13); we forward those as explicit overrides so
        ``extract_vec_inject_index_torch`` projects through layers that
        actually own ``k_proj`` / ``v_proj``.

        Returns True on success.
        """
        import numpy as np
        from types import SimpleNamespace
        from chuk_lazarus.inference.context.research.vec_inject.prefill_torch import (
            extract_vec_inject_index_torch,
        )

        # 1. Pull per-window token sequences from window_tokens.npz.
        window_tokens_path = torch_store_dir / "window_tokens.npz"
        if not window_tokens_path.exists():
            info(f"  vec_inject skipped: {window_tokens_path} missing")
            return False
        with np.load(window_tokens_path) as f:
            # Keys are stringified window-ids; preserve numeric order.
            try:
                wid_keys = sorted(f.files, key=lambda k: int(k))
            except ValueError:
                wid_keys = sorted(f.files)
            windows: list[list[int]] = [
                [int(t) for t in f[k].tolist()] for k in wid_keys
            ]

        if not windows:
            info("  vec_inject skipped: no windows captured")
            return False

        # 2. Resolve KV-share-aware arch config (matches the live indexer's
        #    layer picks). _derive_arch_config returns a dict whose values
        #    extract_vec_inject_index_torch consumes via attribute access; wrap
        #    in SimpleNamespace.
        arch_dict, _crystal_layer, _window_size = self._derive_arch_config()
        arch_config = SimpleNamespace(**arch_dict)

        # 3. Extract → vec_inject.npz at session_root. Call the prefill
        #    one-window-at-a-time to avoid the cross-window
        #    boundary_residual prepend (incompatible with Gemma-4's HF
        #    rotary position embedding which is computed once on the
        #    original sequence length and would size-mismatch against the
        #    prepended hidden_states). Each independent call keeps the
        #    internal boundary_residual=None and lands a partial NPZ; we
        #    merge per-window arrays into the final session-level NPZ.
        import tempfile
        t0 = time.time()
        merged: dict[str, "np.ndarray"] = {}
        meta_set = False
        with tempfile.TemporaryDirectory(prefix="vec_inject_per_win_") as tmpdir:
            tmp_root = Path(tmpdir)
            for source_wid, w_tokens in enumerate(windows):
                if not w_tokens:
                    continue
                per_dir = tmp_root / f"w{source_wid:04d}"
                per_dir.mkdir()
                try:
                    extract_vec_inject_index_torch(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        windows=[list(w_tokens)],
                        output_path=per_dir,
                        arch_config=arch_config,
                        retrieval_layer=int(arch_dict["retrieval_layer"]),
                        query_head=int(arch_dict["query_head"]),
                        inject_layer=int(arch_dict["injection_layer"]),
                    )
                except Exception as exc:  # noqa: BLE001
                    info(
                        f"  vec_inject window {source_wid} skipped (non-fatal): {exc!r}"
                    )
                    continue
                per_npz = per_dir / "vec_inject.npz"
                if not per_npz.exists():
                    continue
                with np.load(per_npz) as f_per:
                    for k in f_per.files:
                        # Per-window keys arrive prefixed with "w0/..."; rewrite
                        # the wid to source_wid so all windows coexist in the
                        # merged store. Meta scalars (no "w" prefix) we copy
                        # once.
                        if k.startswith("w0/"):
                            new_k = f"w{source_wid}/" + k[len("w0/"):]
                            merged[new_k] = f_per[k].copy()
                        else:
                            if k not in merged:
                                merged[k] = f_per[k].copy()
                                meta_set = True

        dt = time.time() - t0
        if not merged:
            info("  vec_inject FAILED: no windows extracted (all skipped)")
            return False
        npz_path = session_root / "vec_inject.npz"
        np.savez(str(npz_path), **merged)
        n_kept = sum(1 for k in merged if k.endswith("/k_vecs"))
        info(
            f"  vec_inject.npz written ({n_kept}/{len(windows)} windows OK) "
            f"in {dt:.2f}s -> {npz_path} ({npz_path.stat().st_size} bytes)"
        )
        return True

    def _mark_dirty(self) -> None:
        """axis-3 (Addition 2): mark the store as having unsaved tokens.

        Writes <store_root>/.dirty as a 0-byte sentinel. Best-effort —
        any OS error is logged and swallowed; the chat must never crash
        because of a flag-write failure (e.g. read-only mount, ENOSPC).
        """
        try:
            (self.store_root / DIRTY_FLAG_FILENAME).touch(exist_ok=True)
        except OSError as exc:
            info(f"  .dirty mark FAILED (non-fatal): {exc!r}")

    def start_new_session(self) -> None:
        from chuk_lazarus.chat_loop.session import ChatLoopSession
        from chuk_lazarus.inference.chat import ChatHistory

        self.session = ChatLoopSession()
        self.history = ChatHistory()
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
                token_ids = self.tokenizer(
                    decoded_text, add_special_tokens=False
                ).input_ids
                keywords = _extract_keywords_from_text(
                    decoded_text, max_keywords=12
                )
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
                    "token_count": int(
                        boundary.end_token_offset - boundary.start_token_offset
                    ),
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

    def plain_chat_turn(self, user_text: str) -> TurnMetadata:
        from chuk_lazarus.chat_loop.cli import stream_assistant_reply
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

        t0 = time.time()
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
        elapsed = time.time() - t0

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
        # avoid double-logging (mirrors vec_inject and except-branch
        # rewinds below).
        if self.memory_mode == "kv_direct":
            self.session.turns.pop()
            if self.history.messages:
                self.history.messages.pop()
            return self.kv_query_turn(user_text)

        # vec_inject routing short-circuits to the dedicated torch stack.
        # _vec_inject_turn delegates to plain_chat_turn (which re-records the
        # user turn), so rewind here to avoid double-logging (mirrors the
        # except-branch rewind below).
        if self.memory_mode == "vec_inject":
            self.session.turns.pop()
            if self.history.messages:
                self.history.messages.pop()
            return self._vec_inject_turn(user_text, chat_context)

        t_total_start = time.time()
        t_retrieve_start = time.time()
        meta = TurnMetadata(mode="topical")
        try:
            if self.memory_mode == "topical":
                result = self.retriever.query_topical(chat_context)
            elif self.memory_mode == "entity_mention":
                result = self.retriever.query_entity_mention(chat_context)
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
        gen_ids = self.tokenizer(
            result.generated_answer, add_special_tokens=False
        ).input_ids
        meta.generated_tokens = len(gen_ids)
        # Prompt tokens approximation: system + matched window + context
        prompt_preview = (
            self.retriever.system_prompt
            + "\n"
            + result.matched_window_text
            + "\n"
            + chat_context
        )
        meta.prompt_tokens = len(
            self.tokenizer(prompt_preview, add_special_tokens=False).input_ids
        )
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
        meta.no_silent_fallback = (
            meta.mode != "none" and meta.window_id is not None
        )

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
        max_total_inject_tokens: int = MAX_TOTAL_INJECT_TOKENS,
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
        cfg = getattr(self.model, "config", None)
        # Gemma-4-E2B-it ships head_dim=256, num_key_value_heads=4. Fall
        # back to those literals when config attributes are missing.
        head_dim = int(getattr(cfg, "head_dim", 256) or 256)
        n_kv_heads = int(getattr(cfg, "num_key_value_heads", 4) or 4)
        per_fact_cost = head_dim * n_kv_heads
        if per_fact_cost <= 0:
            return tier_assignments
        sorted_assignments = sorted(
            tier_assignments,
            key=lambda a: float(getattr(a.candidate, "ucb1_score", 0.0)),
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

    def kv_query_turn(
        self,
        query_text: str,
        *,
        insertion_family: KVInsertionFamily = "full_attention",
        sliding_layer_indices: tuple[int, ...] | None = None,
        sliding_head_indices: tuple[int, ...] | None = None,
    ) -> TurnMetadata:
        """Run a single axis-5 KV-direct query. Used by ``/kv_query`` and by
        the REPL integration tests.

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
        from chuk_lazarus.inference.generation import GenerationConfig
        from chuk_lazarus.inference.context.knowledge.gemma4_e2b_it_layers import (
            GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS,
        )
        from chuk_lazarus.session_retrieval import (
            asi_route_candidates,
            assign_tiers,
        )

        # AMD 11: sliding-window-hazard precondition. The recipe-canonical
        # target_layer for KV-direct injection is 29 (Gemma-4-E2B-it global
        # attention). Assert the fixture matches; the actual runtime guard
        # lives in vec_inject_to_kv_direct.assert_global_attention_layer.
        assert 29 in GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS, (
            "AMD 11 invariant: target_layer=29 must be in "
            f"GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS={sorted(GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS)}"
        )

        candidate_pool = int(os.environ.get("LAZARUS_KV_CANDIDATE_POOL", "16"))
        k_hot = int(os.environ.get("LAZARUS_KV_K_HOT", "4"))
        k_warm = int(os.environ.get("LAZARUS_KV_K_WARM", "8"))
        hot_budget_mib = int(os.environ.get("LAZARUS_KV_HOT_BUDGET_MIB", "32"))
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

        try:
            candidates = asi_route_candidates(
                self.retriever.handles,
                query_text,
                self.retriever.tokenizer,
                candidate_pool=candidate_pool,
            )
            if not candidates:
                raise RuntimeError("asi_route_candidates returned no candidates")

            tier_assignments = assign_tiers(
                candidates,
                K_HOT=k_hot,
                K_WARM=k_warm,
                candidate_pool=candidate_pool,
            )
            if not tier_assignments:
                raise RuntimeError("assign_tiers produced zero assignments")

            # Group assignments by the handle that owns the candidate.
            # The top-ranked candidate's handle is used as the target
            # checkpoint; all tier assignments for that same handle are
            # passed to materialization (axis-5 requires single-handle
            # assignments per prepare_kv_direct_materialization call).
            top_handle = tier_assignments[0].candidate.handle
            assignments_for_handle = [
                a for a in tier_assignments
                if a.candidate.handle.session_id == top_handle.session_id
            ]
            if not assignments_for_handle:
                raise RuntimeError(
                    "no tier assignments owned by top-ranked candidate's handle"
                )

            # axis-4 (Addition 1): apply token-budget governor BEFORE
            # answer_with_kv_direct. Caps the per-call injection footprint
            # at MAX_TOTAL_INJECT_TOKENS; truncates lowest-scoring facts.
            assignments_for_handle = self._apply_token_budget(
                assignments_for_handle
            )
            if not assignments_for_handle:
                raise RuntimeError(
                    "axis-4 token-budget governor truncated all assignments"
                )

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

            result = self.retriever.answer_with_kv_direct(
                query_text,
                assignments_for_handle,
                hot_budget_mib=hot_budget_mib,
                warm_config=warm_config,
                generation_config=gen_config,
                handle=top_handle,
                **selector_kwargs,
            )
        except (ValueError, RuntimeError) as exc:
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

        # axis-6 observability fields — REAL values from result.
        meta.kv_direct_active = bool(kv_strict.get("kv_direct_active", False))
        meta.mask_penalty_applied = bool(kv_strict.get("mask_penalty_applied", False))
        meta.vram_peak_mib = kv_strict.get("vram_peak_mib", None)
        meta.vram_delta_mib = kv_strict.get("vram_delta_mib", None)
        # selected_tier: derive from the tier assignments used for this
        # handle. When the assignments span HOT + WARM we label "hot+warm"
        # (or "mixed" when COLD sneaks in). Single-tier sets return the
        # single label.
        tiers_used = {a.tier.value for a in assignments_for_handle}
        if not tiers_used:
            meta.selected_tier = "kv_direct"
        elif len(tiers_used) == 1:
            meta.selected_tier = next(iter(tiers_used))
        elif tiers_used == {"hot", "warm"}:
            meta.selected_tier = "hot+warm"
        else:
            meta.selected_tier = "mixed"

        # no_silent_fallback: True iff window_id is set (>= 0) AND KV-direct
        # path actually ran (kv_direct_active flag from the runtime).
        meta.no_silent_fallback = bool(
            meta.window_id is not None
            and int(meta.window_id) != -1
            and meta.kv_direct_active
        )

        # Token counts for the reply.
        gen_ids = self.tokenizer(
            result.generated_answer, add_special_tokens=False
        ).input_ids
        meta.generated_tokens = len(gen_ids)
        # Prompt tokens approximation: system + query.
        prompt_preview = (
            (self.retriever.system_prompt or "") + "\n" + query_text
        )
        meta.prompt_tokens = len(
            self.tokenizer(prompt_preview, add_special_tokens=False).input_ids
        )
        meta.total_time = time.time() - t_total_start
        meta.generate_time = max(0.0, meta.total_time - meta.retrieve_time)

        # Print debug block, then the answer.
        meta.pretty_print()
        print(f"kv_direct> {result.generated_answer}\n", flush=True)
        self._mark_dirty()
        return meta

    def _vec_inject_turn(self, user_text: str, chat_context: str) -> TurnMetadata:
        """Route a turn through the torch vec_inject stack end-to-end.

        1. Retrieve matches via LocalVecInjectProvider.retrieve_sync on CUDA.
        2. Install forward_pre_hook on model.model.layers[injection_layer] that
           applies vec_inject_all to the residual h at position -1.
        3. Generate the reply via plain_chat_turn's streaming path.
        4. Remove the hook; populate and return TurnMetadata.

        On low-confidence retrieval, zero-match, or any runtime error: emit the
        axis-6 SILENT FALLBACK WARN and delegate to plain_chat_turn (mirrors the
        recall_chat_turn / kv_query_turn fallback contract).
        """
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "vec_inject memory_mode requires CUDA (Amendment 5, GPU-only)"
            )

        from chuk_lazarus.inference.context.research.vec_inject.injection_torch import (
            vec_inject_all,
        )

        provider = self._get_or_load_vec_inject_provider()
        if provider is None or provider.n_facts == 0:
            print(
                "[WARN] axis-6: SILENT FALLBACK DETECTED — mode=vec_inject "
                "reason=no_index_or_empty",
                flush=True,
            )
            return self.plain_chat_turn(user_text)

        meta = TurnMetadata(mode="vec_inject")
        t_total_start = time.time()
        t_retrieve_start = time.time()

        try:
            query_ids = self.tokenizer(
                chat_context, add_special_tokens=False
            ).input_ids
            result = provider.retrieve_sync(
                query_ids=list(query_ids),
                query_text=chat_context,
                top_k=5,
            )
        except (ValueError, RuntimeError) as exc:
            print(
                f"[WARN] axis-6: SILENT FALLBACK DETECTED — mode=vec_inject "
                f"reason={type(exc).__name__}: {exc}",
                flush=True,
            )
            info(f"vec_inject fallback to plain chat: {exc}")
            return self.plain_chat_turn(user_text)

        meta.retrieve_time = time.time() - t_retrieve_start

        if not result.routing_confident or not result.matches:
            print(
                "[WARN] axis-6: SILENT FALLBACK DETECTED — mode=vec_inject "
                f"reason=low_confidence top_score={result.top_score}",
                flush=True,
            )
            return self.plain_chat_turn(user_text)

        # Install forward_pre_hook on the injection layer.
        # The HF Gemma model exposes the decoder stack at self.model.model.layers.
        embed_matrix = self.model.get_input_embeddings().weight
        target_layer = self.model.model.layers[result.injection_layer]

        def _pre_hook(module, inputs):
            # inputs is a tuple; first positional arg is the hidden state (1, T, D).
            # We modify only the last-position slice to match the MLX reference.
            if not inputs:
                return None
            h = inputs[0]
            if h is None or not isinstance(h, torch.Tensor):
                return None
            h_last = h[:, -1:, :]
            h_injected = vec_inject_all(h_last, result.matches, embed_matrix)
            new_h = torch.cat([h[:, :-1, :], h_injected], dim=1)
            return (new_h,) + tuple(inputs[1:])

        hook_handle = target_layer.register_forward_pre_hook(_pre_hook)
        try:
            reply_meta = self.plain_chat_turn(user_text)
        finally:
            hook_handle.remove()

        # Populate routing metadata from the vec_inject result.
        top_match = result.matches[0]
        meta.routing_mode = result.routing_stage or "vec_inject"
        meta.source_session = ""  # vec_inject index has no session provenance at retrieval time
        meta.window_id = top_match.window_id
        meta.routing_score = top_match.score
        meta.matched_window_text = ""  # provider does not carry window text
        meta.window_keywords = []
        meta.strict_assertions = {}
        meta.generated_answer = reply_meta.generated_answer
        meta.generated_tokens = reply_meta.generated_tokens
        meta.prompt_tokens = reply_meta.prompt_tokens
        meta.total_time = time.time() - t_total_start
        meta.generate_time = max(0.0, meta.total_time - meta.retrieve_time)
        meta.no_silent_fallback = True
        return meta

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
            selection_ready_abs = torch_store_dir / "selection_ready"

            # Window count: prefer the manifest the indexer just wrote, fall
            # back to counting boundary files if manifest is missing/unreadable.
            window_count = 0
            try:
                if manifest_abs.exists():
                    manifest_obj = json.loads(
                        manifest_abs.read_text(encoding="utf-8")
                    )
                    window_count = int(
                        manifest_obj.get(
                            "num_entries", manifest_obj.get("num_windows", 0)
                        )
                    )
            except Exception:  # noqa: BLE001 - best-effort; counting fallback
                window_count = 0

            boundary_residual_count = (
                sum(1 for _ in boundaries_abs.glob("window_*.npy"))
                if boundaries_abs.exists()
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
                    "scripts/interactive_memory_chat.py::"
                    "MemoryChat.save_current_session"
                ),
                "torch_store_path": "torch_store",
                "manifest_path": "torch_store/manifest.json",
                "window_count": int(window_count),
                "selection_ready_descriptor_dir": "torch_store/selection_ready",
                "selection_ready_descriptor_count": int(
                    selection_ready_descriptor_count
                ),
                "boundary_residual_dir": "torch_store/boundaries",
                "boundary_residual_count": int(boundary_residual_count),
                "transcript_path": str(transcript_path),
                "aus3000_clauses_count": int(len(written)),
                "indexer_flush_seconds": float(dt),
                "kv_direct_ready": False,
                "kv_direct_pending_reason": (
                    "axes 2-5 runtime fields not yet landed "
                    "(lead-runtime-router scope)"
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

        # 3c. axis-BC: emit vec_inject.npz so the next session's LocalVecInjectProvider
        #     can load real K/V at retrieval_layer / kv_head with per-fact coefficients.
        #     Best-effort: a failure here MUST NOT block /save.
        try:
            session_root = self.checkpoints_root / sid
            torch_store_dir = session_root / "torch_store"
            if self._emit_vec_inject_npz(session_root, torch_store_dir):
                self.vec_inject_available = True
        except Exception as exc:  # noqa: BLE001 - additive; never block /save
            info(f"  vec_inject.npz write FAILED (non-fatal): {exc!r}")

        # 3d. axis-BC auto-route: if vec_inject.npz was just written and the
        #     user is on the default 'topical' mode, auto-promote to
        #     'kv_direct' so the next chat turn injects KV memory.
        if self.vec_inject_available and self.memory_mode == "topical":
            self.memory_mode = "kv_direct"
            info("  memory_mode auto-promoted topical -> kv_direct (vec_inject.npz available)")

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
                info(
                    f"  retriever refreshed in {time.time() - t0:.2f}s "
                    f"(+{added} handle(s))"
                )
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
            print(f"  device           : {self.retriever.device}")
        print(f"  current_session  : session_id={self.session.session_id if self.session else '—'}  turns={len(self.session.turns) if self.session else 0}")
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
        gen_ids = self.tokenizer(
            result.generated_answer, add_special_tokens=False
        ).input_ids
        meta.generated_tokens = len(gen_ids)
        meta.pretty_print()
        print(f"probe answer> {result.generated_answer}\n", flush=True)

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
            print(__doc__)
            return False

        if head == "/save":
            self.emit_store()
            return False

        if head == "/new":
            if self.session and self.session.turns:
                self.save_current_session()
            self.start_new_session()
            info("fresh session started — try asking about prior sessions.")
            return False

        if head == "/stats":
            self.print_stats()
            return False

        if head == "/last":
            if self.last_meta is None:
                info("no turn yet.")
            else:
                self.last_meta.pretty_print()
            return False

        if head == "/history":
            section("CURRENT SESSION TRANSCRIPT")
            print(f"  session_id: {self.session.session_id}")
            for t in self.session.turns:
                role = getattr(t.role, "value", str(t.role))
                print(f"  [{t.turn_index:03d}] {role}: {truncate(t.text, 180)}")
            print("=" * HEADER_W)
            return False

        if head == "/memory":
            # toggle between topical <-> off (quick toggle). Pass arg for explicit.
            if arg in ("topical", "entity_mention", "vec_inject", "kv_direct", "off"):
                self.memory_mode = arg
            else:
                self.memory_mode = "off" if self.memory_mode != "off" else "topical"
            info(f"memory_mode = {self.memory_mode}")
            return False

        if head == "/query":
            if not arg:
                info("usage: /query <text>")
                return False
            self.probe("topical", arg)
            return False

        if head == "/exact":
            if not arg:
                info("usage: /exact <dotted-handle>  (e.g. 11a1c9ad.1.0)")
                return False
            self.probe("exact", arg)
            return False

        if head == "/entity":
            if not arg:
                info("usage: /entity <text>")
                return False
            self.probe("entity_mention", arg)
            return False

        if head == "/kv_query":
            if not arg:
                info(f"usage: {KV_QUERY_USAGE}")
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
        "--memory-mode",
        choices=("topical", "entity_mention", "vec_inject", "kv_direct", "off"),
        default=os.environ.get("LAZARUS_MEMORY_MODE", "topical"),
        help="Recall routing mode for post-save turns.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default cuda).",
    )
    args = parser.parse_args(argv)

    # Also honour LAZARUS_MAX_NEW_TOKENS for the retriever's generation_kwargs.
    os.environ.setdefault("LAZARUS_MAX_NEW_TOKENS", str(args.max_new_tokens))

    chat = MemoryChat(
        store_root=args.store_root,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        memory_mode=args.memory_mode,
        device=args.device,
    )

    try:
        chat.run_repl()
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
