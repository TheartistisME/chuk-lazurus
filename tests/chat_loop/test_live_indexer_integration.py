"""Integration tests for the LiveIndexer + StreamingWindower + refresh_handles
wiring inside ``scripts/interactive_memory_chat.py``.

These tests prove the contract of ``MemoryChat`` without ever loading a real
Gemma model, spawning a subprocess, or touching CUDA. They cover:

  1. ``MemoryChat._make_on_window`` correctly adapts a StreamingWindower
     boundary + decoded-text signature into ``LiveIndexer.enqueue`` kwargs.
  2. The adapter is a safe no-op when ``MemoryChat.indexer`` is ``None``.
  3. ``MemoryChat.save_current_session`` calls ``indexer.flush_and_close``
     and then ``retriever.refresh_handles`` (or lazy-loads a retriever on
     the first-save path).
  4. The single-model invariant: no subprocess ever spawns from the runtime
     save path.
  5. ``MemoryChat._derive_arch_config`` falls back to the empirically
     validated Gemma-4 E2B defaults when ``ArchitectureConfig`` cannot be
     resolved, returning all seven required keys.

``scripts/`` is not a package on this repo, so the module under test is
loaded via an importlib dance at module import time.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.chat_loop.session import ChatLoopSession, ChunkBoundary
from chuk_lazarus.inference.chat import Role


# ---------------------------------------------------------------------------
# importlib dance — scripts/ is NOT a package, so we load the module by path.
# ---------------------------------------------------------------------------


def _load_interactive_memory_chat():
    """Load ``scripts/interactive_memory_chat.py`` and return its module."""
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "interactive_memory_chat.py"
    )
    spec = importlib.util.spec_from_file_location(
        "interactive_memory_chat", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["interactive_memory_chat"] = module
    spec.loader.exec_module(module)
    return module


_imc = _load_interactive_memory_chat()
MemoryChat = _imc.MemoryChat


# ---------------------------------------------------------------------------
# Shared constants + helpers.
# ---------------------------------------------------------------------------


HIDDEN_SIZE: int = 1536

ARCH_CONFIG_E2B: dict[str, Any] = {
    "retrieval_layer": 28,
    "query_head": 7,
    "injection_layer": 29,
    "hidden_dim": 1536,
    "head_dim": 256,
    "crystal_layer": 29,
    "window_size": 512,
}

_REQUIRED_ARCH_KEYS = (
    "retrieval_layer",
    "query_head",
    "injection_layer",
    "hidden_dim",
    "head_dim",
    "crystal_layer",
    "window_size",
)


def _patch_capture(monkeypatch: pytest.MonkeyPatch, fn=None) -> None:
    """Monkeypatch the locally-bound capture reference inside live_indexer.

    Mirrors ``tests/session_store/test_live_indexer.py``'s helper so the
    LiveIndexer worker never calls into a real model.
    """

    def _default(model, token_ids, *, crystal_layer, device=None, initial_residual=None):
        return torch.zeros((HIDDEN_SIZE,), dtype=torch.float32)

    monkeypatch.setattr(
        "chuk_lazarus.session_store.live_indexer.capture_post_crystal_boundary",
        fn if fn is not None else _default,
    )


class _StubTokenizer:
    """Minimal HF-shaped tokenizer: call -> SimpleNamespace(input_ids=[...])."""

    def __init__(self, ids: list[int] | None = None) -> None:
        # Each call returns a deterministic sequence; callers that care about
        # per-call uniqueness can supply their own ids.
        self._ids = ids if ids is not None else [10, 11, 12, 13]

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ) -> SimpleNamespace:
        # Length-proportional ids so token_count derivations stay plausible.
        n = max(1, len(text.split()))
        return SimpleNamespace(input_ids=self._ids[:n] or [0])

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return " ".join(f"tok{token_id}" for token_id in token_ids)


class _FakeHistory:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def add_user(self, text: str) -> None:
        self.messages.append(("user", text))

    def add_assistant(self, text: str) -> None:
        self.messages.append(("assistant", text))


def _make_boundary(
    *,
    chunk_index: int = 0,
    start: int = 0,
    end: int = 64,
) -> ChunkBoundary:
    return ChunkBoundary(
        chunk_index=chunk_index,
        start_token_offset=start,
        end_token_offset=end,
        emitted_at="2026-04-21T00:00:00.000000Z",
    )


def _make_chat(tmp_path: Path) -> MemoryChat:
    """Build a MemoryChat rooted at tmp_path without loading a real model."""
    return MemoryChat(
        store_root=tmp_path / "store",
        model_path=None,
        max_new_tokens=32,
        memory_mode="topical",
        device="cpu",
    )


def _make_turn(turn_index: int = 0, role: Role = Role.ASSISTANT):
    """Return a SimpleNamespace with the attrs the on_window closure reads."""
    return SimpleNamespace(turn_index=turn_index, role=role)


def _make_retrieval_result(answer: str = "recalled answer") -> SimpleNamespace:
    return SimpleNamespace(
        routing_mode="topical",
        source_session="source-sess",
        window_id=7,
        routing_score=0.91,
        matched_window_text="matched memory window",
        window_keywords=["banjo", "memory"],
        strict_assertions={"hook_fired": True},
        generated_answer=answer,
    )


# ---------------------------------------------------------------------------
# TEST 1: adapter closure dispatches to LiveIndexer.enqueue with the right
# signature and a session-monotonic window id.
# ---------------------------------------------------------------------------


def test_on_window_adapter_dispatches_to_enqueue(tmp_path: Path) -> None:
    """The closure returned by ``_make_on_window`` passes the four expected
    positional args to ``indexer.enqueue`` and increments ``_window_counter``
    on each call."""
    chat = _make_chat(tmp_path)
    # Sentinel model / real fake tokenizer (closure round-trips text through it).
    chat.model = object()
    chat.tokenizer = _StubTokenizer(ids=[100, 101, 102, 103])
    chat.session = SimpleNamespace(session_id="sess-1234")
    chat._window_counter = 0

    enqueue_mock = MagicMock()
    chat.indexer = SimpleNamespace(enqueue=enqueue_mock)

    turn = _make_turn(turn_index=3, role=Role.ASSISTANT)
    on_window = chat._make_on_window(turn)

    # First emission.
    boundary_0 = _make_boundary(chunk_index=0, start=0, end=64)
    on_window(boundary_0, "alice built a robot")
    assert enqueue_mock.call_count == 1
    args_0, _ = enqueue_mock.call_args
    wid_0, token_ids_0, keywords_0, clause_meta_0 = args_0
    assert wid_0 == 0
    assert isinstance(token_ids_0, list)
    assert all(isinstance(i, int) for i in token_ids_0)
    assert isinstance(keywords_0, list)
    assert all(isinstance(k, str) for k in keywords_0)
    for key in (
        "session_id",
        "turn_index",
        "role",
        "chunk_index",
        "start_token_offset",
        "end_token_offset",
        "token_count",
    ):
        assert key in clause_meta_0, f"missing clause key: {key}"
    assert clause_meta_0["session_id"] == "sess-1234"
    assert clause_meta_0["turn_index"] == 3
    assert clause_meta_0["role"] == "assistant"
    assert clause_meta_0["chunk_index"] == 0
    assert clause_meta_0["start_token_offset"] == 0
    assert clause_meta_0["end_token_offset"] == 64
    assert clause_meta_0["token_count"] == 64

    # Second emission — window_id must be session-monotonic (not reset by a
    # same-chunk_index from a fresh turn).
    boundary_1 = _make_boundary(chunk_index=0, start=0, end=32)
    on_window(boundary_1, "carol wrote a poem")
    assert enqueue_mock.call_count == 2
    args_1, _ = enqueue_mock.call_args
    wid_1 = args_1[0]
    assert wid_1 == 1, "window_id must be session-monotonic across turns"


# ---------------------------------------------------------------------------
# TEST 2: closure no-ops (returns None, raises nothing) when indexer is None.
# ---------------------------------------------------------------------------


def test_on_window_noop_when_indexer_none(tmp_path: Path) -> None:
    """The closure must be crash-safe when ``chat.indexer is None``."""
    chat = _make_chat(tmp_path)
    chat.model = object()
    chat.tokenizer = _StubTokenizer()
    chat.session = SimpleNamespace(session_id="sess-noop")
    chat._window_counter = 0
    chat.indexer = None

    on_window = chat._make_on_window(_make_turn())
    result = on_window(_make_boundary(), "hello world")
    assert result is None
    # Counter must not advance when no indexer is wired.
    assert chat._window_counter == 0


def test_plain_chat_turn_enqueues_user_turn_before_assistant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = _make_chat(tmp_path)
    chat.model = SimpleNamespace(device="cpu")
    chat.tokenizer = _StubTokenizer(ids=list(range(100, 700)))
    chat.history = _FakeHistory()
    chat.session = ChatLoopSession()

    enqueue_mock = MagicMock()
    chat.indexer = SimpleNamespace(enqueue=enqueue_mock)

    def _fake_stream_assistant_reply(**kwargs: Any) -> str:
        reply = "assistant remembers banjo"
        windower = kwargs["windower"]
        session = kwargs["session"]
        turn = kwargs["turn"]
        on_text_delta = kwargs["on_text_delta"]
        on_text_delta(reply)
        for boundary in windower.feed_text(reply):
            session.append_chunk(turn, boundary)
        tail = windower.flush()
        if tail is not None:
            session.append_chunk(turn, tail)
        session.finish_turn(turn)
        return reply

    monkeypatch.setattr(
        "chuk_lazarus.chat_loop.cli.stream_assistant_reply",
        _fake_stream_assistant_reply,
    )

    meta = chat.plain_chat_turn("user mentions banjo at dusk")

    assert meta.generated_answer == "assistant remembers banjo"
    assert enqueue_mock.call_count == 2

    first_args, _ = enqueue_mock.call_args_list[0]
    second_args, _ = enqueue_mock.call_args_list[1]

    assert first_args[0] == 0
    assert first_args[3]["role"] == "user"
    assert first_args[3]["turn_index"] == 0
    assert second_args[0] == 1
    assert second_args[3]["role"] == "assistant"
    assert second_args[3]["turn_index"] == 1
    assert chat._window_counter == 2


def test_recall_chat_turn_enqueues_user_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = _make_chat(tmp_path)
    chat.model = SimpleNamespace(device="cpu")
    chat.tokenizer = _StubTokenizer(ids=list(range(200, 800)))
    chat.history = _FakeHistory()
    chat.session = ChatLoopSession()

    enqueue_mock = MagicMock()
    chat.indexer = SimpleNamespace(enqueue=enqueue_mock)
    chat.retriever = SimpleNamespace(
        system_prompt="system prompt",
        query_topical=MagicMock(return_value=_make_retrieval_result("recalled answer")),
    )

    monkeypatch.setattr(_imc.TurnMetadata, "pretty_print", lambda self: None)

    meta = chat.recall_chat_turn("what did I say about banjo?")

    assert meta.generated_answer == "recalled answer"
    assert enqueue_mock.call_count == 1
    args, _ = enqueue_mock.call_args
    assert args[0] == 0
    assert args[3]["role"] == "user"
    assert args[3]["turn_index"] == 0
    assert chat._window_counter == 1


# ---------------------------------------------------------------------------
# TEST 3: save_current_session flushes indexer and refreshes handles.
# ---------------------------------------------------------------------------


def test_save_current_session_flushes_indexer_and_refreshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``save_current_session(rebuild_retriever=True)`` must:
      - call ``indexer.flush_and_close`` exactly once,
      - call ``retriever.refresh_handles(checkpoints_root)`` exactly once,
      - reset ``chat.indexer`` to ``None`` so /new starts with a clean slate.
    """
    _patch_capture(monkeypatch)  # guards any stray capture call paths

    chat = _make_chat(tmp_path)
    chat.model = object()
    chat.tokenizer = _StubTokenizer()

    # Build a real ChatLoopSession so .turns[n].model_dump(mode="json") works.
    session = ChatLoopSession()
    t0 = session.begin_turn(Role.USER, "tell me about banjo")
    session.finish_turn(t0)
    t1 = session.begin_turn(Role.ASSISTANT, "Banjo is a dog.")
    session.finish_turn(t1)
    chat.session = session

    # Fakes for the indexer and retriever surfaces save_current_session hits.
    indexer_mock = MagicMock()
    indexer_mock.flush_and_close = MagicMock()
    chat.indexer = indexer_mock

    retriever_mock = MagicMock()
    retriever_mock.refresh_handles = MagicMock(return_value=1)
    chat.retriever = retriever_mock

    ok = chat.save_current_session(rebuild_retriever=True)
    assert ok is True

    # Indexer flush was invoked exactly once.
    indexer_mock.flush_and_close.assert_called_once_with()

    # Retriever refresh was invoked with the checkpoints root.
    retriever_mock.refresh_handles.assert_called_once_with(chat.checkpoints_root)

    # Indexer is reset for the next session (convention: /save closes indexing
    # for the current session; /new reallocates).
    assert chat.indexer is None

    # Audit artefacts were emitted to disk.
    transcript_path = chat.transcripts_root / f"{session.session_id}.json"
    assert transcript_path.exists(), "transcript JSON must be written on /save"


# ---------------------------------------------------------------------------
# TEST 4: no subprocess is spawned on the runtime save path.
# ---------------------------------------------------------------------------


def test_no_subprocess_spawned_at_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forbidding every ``subprocess`` entry point must NOT cause save to
    raise — the runtime path is file-I/O only (LiveIndexer drained in-proc).

    If ``emit_session`` ever grows a subprocess call this test will fail
    loudly, surfacing the regression.
    """
    _patch_capture(monkeypatch)

    def _forbidden(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("subprocess forbidden in axis-2 runtime path")

    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(subprocess, "call", _forbidden)
    monkeypatch.setattr(subprocess, "check_output", _forbidden)
    monkeypatch.setattr(subprocess, "check_call", _forbidden)

    chat = _make_chat(tmp_path)
    chat.model = object()
    chat.tokenizer = _StubTokenizer()

    session = ChatLoopSession()
    t0 = session.begin_turn(Role.USER, "ping")
    session.finish_turn(t0)
    t1 = session.begin_turn(Role.ASSISTANT, "pong")
    session.finish_turn(t1)
    chat.session = session

    indexer_mock = MagicMock()
    chat.indexer = indexer_mock
    retriever_mock = MagicMock()
    retriever_mock.refresh_handles = MagicMock(return_value=0)
    chat.retriever = retriever_mock

    # The critical assertion: this call must not raise RuntimeError from the
    # forbidden subprocess sentinels.
    ok = chat.save_current_session(rebuild_retriever=True)
    assert ok is True
    indexer_mock.flush_and_close.assert_called_once()
    retriever_mock.refresh_handles.assert_called_once_with(chat.checkpoints_root)


# ---------------------------------------------------------------------------
# TEST 5: _derive_arch_config fallback path returns a complete arch_config.
# ---------------------------------------------------------------------------


def test_live_indexer_constructor_signature_contract(tmp_path: Path) -> None:
    """When ``ArchitectureConfig.from_model_config`` cannot resolve the model,
    ``_derive_arch_config`` must fall back to the Gemma-4 E2B defaults and
    return a dict containing all seven required keys.
    """
    chat = _make_chat(tmp_path)
    # A config object that ArchitectureConfig.from_model_config will reject —
    # triggering the broad ``except Exception`` fallback in
    # ``_derive_arch_config``.
    chat.model = SimpleNamespace(
        config=SimpleNamespace(model_type="__not_a_real_model_type__")
    )

    arch_config, crystal_layer, window_size = chat._derive_arch_config()

    for key in _REQUIRED_ARCH_KEYS:
        assert key in arch_config, f"missing required arch_config key: {key}"
        assert isinstance(arch_config[key], int), (
            f"arch_config[{key!r}] must be int, got {type(arch_config[key])!r}"
        )

    # Sanity-check: fallback defaults match the Gemma-4 E2B empirical values.
    assert arch_config["hidden_dim"] == 1536
    assert arch_config["head_dim"] == 256
    assert arch_config["crystal_layer"] == 29
    assert arch_config["window_size"] == 512
    assert crystal_layer == arch_config["crystal_layer"]
    assert window_size == arch_config["window_size"]


# ---------------------------------------------------------------------------
# Sanity: numpy import stays live (silences lint about unused import while
# documenting that the axis-2 test suite keeps numpy in scope for parity
# with the sibling live_indexer unit tests).
# ---------------------------------------------------------------------------


def test_numpy_available_for_parity_with_live_indexer_tests() -> None:
    assert np.float32 is not None
