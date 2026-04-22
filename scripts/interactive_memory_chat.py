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
  LAZARUS_MEMORY_MODE        one of: topical (default) | entity_mention | off

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
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
        # Gemma-4 E2B safety-net defaults.
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
            retrieval_layer = int(getattr(ac, "retrieval_layer", retrieval_layer))
            query_head = int(getattr(ac, "query_head", query_head))
            injection_layer = int(getattr(ac, "injection_layer", injection_layer))
            # hidden_dim / head_dim may be 0 on the registry entry; only
            # overwrite the defaults when positive.
            ac_hidden_dim = int(getattr(ac, "hidden_dim", 0) or 0)
            if ac_hidden_dim > 0:
                hidden_dim = ac_hidden_dim
            ac_head_dim = int(getattr(ac, "head_dim", 0) or 0)
            if ac_head_dim > 0:
                head_dim = ac_head_dim
            ac_crystal_layer = int(getattr(ac, "crystal_layer", -1))
            if ac_crystal_layer >= 0:
                crystal_layer = ac_crystal_layer
            ac_window_size = int(getattr(ac, "window_size", window_size))
            if ac_window_size > 0:
                window_size = ac_window_size
        except Exception as exc:  # noqa: BLE001
            info(
                f"  arch_config fallback engaged ({exc!r}); using hard-coded Gemma-4 E2B defaults"
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

        # Append assistant turn to session/history
        assistant_turn = self.session.begin_turn(Role.ASSISTANT, result.generated_answer)
        self.session.finish_turn(assistant_turn)
        self.history.add_assistant(result.generated_answer)

        # Print debug block BEFORE the reply so you see routing first
        meta.pretty_print()
        print(f"gemma> {result.generated_answer}\n", flush=True)
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
                if self.session and self.session.turns:
                    ans = input("current session has unsaved turns. /save before exit? [Y/n] ").strip().lower()
                    if ans in ("", "y", "yes"):
                        self.save_current_session()
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
            if self.session and self.session.turns:
                ans = input("save before exit? [Y/n] ").strip().lower()
                if ans in ("", "y", "yes"):
                    self.save_current_session()
            info("bye.")
            return True

        if head == "/help":
            print(__doc__)
            return False

        if head == "/save":
            self.save_current_session()
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
            if arg in ("topical", "entity_mention", "off"):
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
        choices=("topical", "entity_mention", "off"),
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
