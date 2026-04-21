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
        info(f"new session started · session_id={self.session.session_id}")

    # ── plain chat turn (no memory injection) ──────────────────────────────

    def plain_chat_turn(self, user_text: str) -> TurnMetadata:
        from chuk_lazarus.chat_loop.cli import stream_assistant_reply
        from chuk_lazarus.chat_loop.streaming import StreamingWindower
        from chuk_lazarus.inference.chat import Role

        self.history.add_user(user_text)
        user_turn = self.session.begin_turn(Role.USER, user_text)
        self.session.finish_turn(user_turn)

        windower = StreamingWindower(self.tokenizer)
        assistant_turn = self.session.begin_turn(Role.ASSISTANT, "")

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
        """Emit AUS3000 → build torch store → refresh retriever. Idempotent."""
        from chuk_lazarus.session_close.wind_down import emit_session
        from chuk_lazarus.session_store.invoke import invoke_build

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

        # 2. Emit AUS3000 clause JSON files under inputs_root/<sid>/
        written = emit_session(
            list(self.session.turns),
            sid,
            self.inputs_root,
        )
        info(f"  AUS3000 clauses -> {len(written)} record(s) under {self.inputs_root / sid}")

        # 3. Invoke the clause-aligned torch store build
        per_session_inputs = self.inputs_root / sid
        t0 = time.time()
        result = invoke_build(
            session_id=sid,
            input_dir=per_session_inputs,
            checkpoint_root=self.checkpoints_root,
            device=self.device,
            force=True,
        )
        dt = time.time() - t0
        if result.returncode != 0:
            info(f"  build FAILED (rc={result.returncode}) in {dt:.1f}s — see logs at {result.checkpoint}")
            print("---- stderr (tail) ----")
            print("\n".join((result.stderr or "").splitlines()[-20:]))
            print("-----------------------")
            return False
        info(f"  torch store built -> {result.checkpoint} in {dt:.1f}s")

        # 4. Rebuild retriever so subsequent turns can see the new session
        if rebuild_retriever:
            self.maybe_load_retriever()
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
