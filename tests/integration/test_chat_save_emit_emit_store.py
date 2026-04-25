"""axis-3 (run-4): emit_store + .dirty flag lifecycle integration tests.

Verifies:
- emit_store unified entry point (Q3 supervisor decision)
- .dirty flag lifecycle (Addition 2)
- /save no-op when clean
- session-end (atexit / quit / exit) emit when dirty
- idempotent across crashes

Tests 1-8 are pure-logic flag-lifecycle tests (no model load); test 9
is a CUDA-gated smoke that exercises save_current_session via the real
prefill path.

Cross-refs:
- scope manifest: ve-ins-0moecnemd0000340b9c
- mission: chuk-lazurus-zfs (Axis-3)
- Q3 + Addition 2 supervisor bindings (verbatim in LEAD --then payload)
- baseline-of-absence: ve-ins-0moeedj8z0000ce96a3
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


# ── REPO_ROOT discovery ────────────────────────────────────────────────────
# scripts/interactive_memory_chat.py is not on the package path; inject
# the scripts/ directory so we can import MemoryChat directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ── stub session fixtures (no model load) ──────────────────────────────────


class _StubTurn:
    """Minimal turn surrogate; emit_store only checks ``len(session.turns)``."""

    pass


class _StubSession:
    """Minimal ChatLoopSession surrogate.

    emit_store interrogates only ``self.session is None`` and
    ``self.session.turns`` (truthiness via ``not self.session.turns``),
    so a list-of-stubs is sufficient to satisfy the non-empty branch.
    """

    def __init__(self, n_turns: int = 1):
        self.session_id = "test-session-id"
        self.started_at = "2026-04-25T13:39:00Z"
        self.turns = [_StubTurn() for _ in range(n_turns)]


# ── unit-test fixture (no model, no CUDA) ──────────────────────────────────


@pytest.fixture
def chat(tmp_path):
    """Construct MemoryChat with a fresh tmp_path store_root (no model load).

    MemoryChat.__init__ does NOT touch the model — it only sets attributes,
    creates the inputs/checkpoints/transcripts subdirectories, and leaves
    self.model = None. device='cpu' is therefore harmless here; the model
    is never instantiated for tests 1-8.
    """
    from interactive_memory_chat import MemoryChat  # noqa: WPS433

    return MemoryChat(
        store_root=tmp_path,
        model_path=None,
        max_new_tokens=8,
        memory_mode="off",
        device="cpu",
    )


# ── module-scoped real-model fixture for test 9 (CUDA-gated) ───────────────


@pytest.fixture(scope="module")
def real_chat(tmp_path_factory):
    """Real Gemma-4-E2B-it on CUDA. Module-scoped so the model loads once."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA-only: torch.cuda.is_available()==False")

    from interactive_memory_chat import MemoryChat  # noqa: WPS433

    chat_dir = tmp_path_factory.mktemp("chat_glue_smoke")
    chat = MemoryChat(
        store_root=chat_dir,
        model_path=None,
        max_new_tokens=8,
        memory_mode="off",
        device="cuda",
    )
    chat.load_model()
    yield chat


# ───────────────────────────────────────────────────────────────────────────
# Tests 1-8: pure logic — emit_store + _mark_dirty flag lifecycle
# ───────────────────────────────────────────────────────────────────────────


def test_emit_store_no_op_when_clean(chat, monkeypatch):
    """When .dirty is absent and force=False, emit_store is a graceful no-op.

    Q3 contract: ``/save`` calls ``emit_store(force=False)``. With no
    intervening ``_mark_dirty`` since the last successful save, the dirty
    sentinel is absent and emit_store MUST short-circuit before invoking
    save_current_session.
    """
    # Arrange — ensure flag does not exist.
    dirty_path = chat.store_root / ".dirty"
    assert not dirty_path.exists()
    save_mock = MagicMock(return_value=True)
    monkeypatch.setattr(chat, "save_current_session", save_mock)
    chat.session = _StubSession(n_turns=2)

    # Act
    result = chat.emit_store()

    # Assert
    assert result is False
    save_mock.assert_not_called()
    assert not dirty_path.exists()


def test_emit_store_force_bypasses_dirty_check(chat, monkeypatch):
    """force=True must invoke save_current_session even when .dirty is absent.

    Used by tests / programmatic callers; not exposed to the chat user.
    """
    # Arrange
    dirty_path = chat.store_root / ".dirty"
    assert not dirty_path.exists()
    save_mock = MagicMock(return_value=True)
    monkeypatch.setattr(chat, "save_current_session", save_mock)
    chat.session = _StubSession(n_turns=1)

    # Act
    result = chat.emit_store(force=True)

    # Assert
    assert result is True
    save_mock.assert_called_once()


def test_emit_store_clears_dirty_on_success(chat, monkeypatch):
    """Successful save_current_session => .dirty must be unlinked."""
    # Arrange
    dirty_path = chat.store_root / ".dirty"
    dirty_path.touch()
    assert dirty_path.exists()
    save_mock = MagicMock(return_value=True)
    monkeypatch.setattr(chat, "save_current_session", save_mock)
    chat.session = _StubSession(n_turns=3)

    # Act
    result = chat.emit_store()

    # Assert
    assert result is True
    save_mock.assert_called_once()
    assert not dirty_path.exists()


def test_emit_store_keeps_dirty_on_failure(chat, monkeypatch):
    """Failed save_current_session => .dirty MUST persist.

    Crash-safety contract: the next /save (or atexit emit) retries the
    save. Clearing .dirty on failure would silently drop unsaved tokens.
    """
    # Arrange
    dirty_path = chat.store_root / ".dirty"
    dirty_path.touch()
    save_mock = MagicMock(return_value=False)
    monkeypatch.setattr(chat, "save_current_session", save_mock)
    chat.session = _StubSession(n_turns=2)

    # Act
    result = chat.emit_store()

    # Assert
    assert result is False
    save_mock.assert_called_once()
    assert dirty_path.exists(), ".dirty must remain on save failure"


def test_emit_store_idempotent_repeat_calls(chat, monkeypatch):
    """Three back-to-back emit_store calls => save_current_session runs once.

    Idempotency follows from .dirty being unlinked after the first
    success: subsequent calls hit the no-op-when-clean branch.
    """
    # Arrange
    dirty_path = chat.store_root / ".dirty"
    dirty_path.touch()
    save_mock = MagicMock(return_value=True)
    monkeypatch.setattr(chat, "save_current_session", save_mock)
    chat.session = _StubSession(n_turns=2)

    # Act
    first = chat.emit_store()
    second = chat.emit_store()
    third = chat.emit_store()

    # Assert
    assert first is True
    assert second is False
    assert third is False
    assert save_mock.call_count == 1
    assert not dirty_path.exists()


def test_emit_store_empty_session_clears_stale_flag(chat, monkeypatch):
    """Stale .dirty + empty session => clear flag, do NOT call save.

    This covers the recovery path after a crash where _mark_dirty fired
    but the next launch has no in-memory session yet.
    """
    # Arrange
    dirty_path = chat.store_root / ".dirty"
    dirty_path.touch()
    save_mock = MagicMock(return_value=True)
    monkeypatch.setattr(chat, "save_current_session", save_mock)
    chat.session = None  # empty / not-yet-started

    # Act
    result = chat.emit_store()

    # Assert
    assert result is False
    save_mock.assert_not_called()
    assert not dirty_path.exists(), "stale .dirty must be cleaned up"


def test_mark_dirty_writes_flag(chat):
    """_mark_dirty must create .dirty as a regular file; idempotent re-call ok."""
    # Arrange
    dirty_path = chat.store_root / ".dirty"
    assert not dirty_path.exists()

    # Act — first call creates the flag.
    chat._mark_dirty()

    # Assert
    assert dirty_path.exists()
    assert dirty_path.is_file()

    # Act — second call must be a no-op (no exception, flag still present).
    chat._mark_dirty()

    # Assert
    assert dirty_path.exists()
    assert dirty_path.is_file()


def test_emit_store_force_with_empty_session_returns_false(chat, monkeypatch):
    """force=True does NOT bypass the empty-session guard.

    The is_dirty branch is satisfied by force, but the immediate
    follow-up check (``self.session is None or not self.session.turns``)
    still wins. This avoids writing an empty checkpoint.
    """
    # Arrange
    save_mock = MagicMock(return_value=True)
    monkeypatch.setattr(chat, "save_current_session", save_mock)
    chat.session = None

    # Act
    result = chat.emit_store(force=True)

    # Assert
    assert result is False
    save_mock.assert_not_called()


# ───────────────────────────────────────────────────────────────────────────
# Test 9: CUDA-gated smoke — full prefill path through save_current_session
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only: torch.cuda.is_available()==False",
)
def test_emit_store_full_prefill_path_smoke(real_chat):
    """End-to-end: emit_store -> save_current_session -> torch_store on disk.

    Loads Gemma-4-E2B-it bf16 on CUDA, forges a 2-turn session, marks
    .dirty, and calls emit_store. Asserts the on-disk artefacts the
    retriever needs:
      - <store_root>/checkpoints/<sid>/torch_store/manifest.json
      - <store_root>/checkpoints/<sid>/save-state.json

    Wall time on RTX 5090: ~30-60s. CUDA-gated per AMD 14.
    """
    from chuk_lazarus.inference.chat import Role  # noqa: WPS433

    chat = real_chat
    chat.start_new_session()
    sid = chat.session.session_id

    # Build a non-empty session: emit_store demands ``len(session.turns) > 0``.
    # We append minimally-valid Turn-like objects via the session's API where
    # possible; ChatLoopSession may expose append_turn / record_turn — but for
    # this smoke we only need ``self.session.turns`` to be truthy AND for the
    # save_current_session path to find token windows. So we route through
    # the real chat-history append surface.
    chat.history.add_user("the planted phrase is moonlit-lemur-7 — please remember it")
    chat.history.add_assistant("Got it: moonlit-lemur-7. I will remember.")
    # Mirror into ChatLoopSession.turns via the real API. ChatLoopSession
    # exposes begin_turn(role, text) -> TurnRecord followed by
    # finish_turn(turn). TurnRecord is a Pydantic BaseModel with
    # ``model_dump`` — required by save_current_session's transcript
    # serialization at scripts/interactive_memory_chat.py:1490.
    # Long enough text to produce at least one window boundary (or tail
    # on flush). Mirror plain_chat_turn's exact sequence:
    # begin_turn -> finish_turn -> _capture_turn_text_live, which routes
    # the text through StreamingWindower.feed_text + flush() to populate
    # LiveIndexer with at least one window. Without this, _compact()
    # refuses to write a manifest for an empty store and save_current_session
    # returns False.
    user_text = (
        "The planted phrase is moonlit-lemur-7. Please remember it across "
        "sessions. The phrase has three parts: an adjective (moonlit), a "
        "noun (lemur), and a digit (7). It is a unique sentinel that future "
        "queries should be able to recall verbatim. We are testing the "
        "infinite-memory loop end-to-end on Gemma-4-E2B-it bf16 on CUDA "
        "with vec_inject KV-direct injection at the global-attention layer."
    )
    assistant_text = (
        "Acknowledged. The planted phrase moonlit-lemur-7 has been recorded. "
        "I will recall it on subsequent sessions when prompted. The structure "
        "is adjective-noun-digit, three components, one sentinel string. "
        "Future queries against this stored window should retrieve the phrase "
        "verbatim through the KV-direct injection path at layer 29."
    )
    user_turn = chat.session.begin_turn(Role.USER, user_text)
    chat.session.finish_turn(user_turn)
    chat._capture_turn_text_live(user_turn)
    assistant_turn = chat.session.begin_turn(Role.ASSISTANT, assistant_text)
    chat.session.finish_turn(assistant_turn)
    chat._capture_turn_text_live(assistant_turn)

    assert len(chat.session.turns) >= 1

    # Mark the store dirty (production path: each assistant turn fires
    # _mark_dirty; we forge it directly here).
    chat._mark_dirty()
    dirty_path = chat.store_root / ".dirty"
    assert dirty_path.exists()

    # Act — full unified emit through the real prefill path.
    result = chat.emit_store()

    # Assert — emit_store delegated to save_current_session and won.
    assert result is True, "emit_store must return True on successful save"
    assert not dirty_path.exists(), ".dirty must be cleared after success"

    session_root = chat.checkpoints_root / sid
    manifest = session_root / "torch_store" / "manifest.json"
    save_state = session_root / "save-state.json"
    assert manifest.exists(), f"missing torch_store manifest at {manifest}"
    assert save_state.exists(), f"missing save-state.json at {save_state}"
