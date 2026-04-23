"""CUDA-gated integration tests: ``/kv_query`` drives the real axis-5
KV-direct runtime path via ``MemoryChat.kv_query_turn`` and surfaces truthful
axis-6 observability fields on ``chat.last_meta``.

These tests run real Gemma — expect 60-180s wall time. They are CUDA-gated
per the user mandate: never CPU-run tests in this repo.

Test matrix:

  1. ``test_kv_query_turn_surfaces_real_axis6_fields`` — the end-to-end
     happy path. This is the post-hotfix closure gate for the real
     KV-direct path on Gemma-4 E2B.
  2. ``test_kv_query_turn_surfaces_silent_fallback_truthfully`` — the
     truthful fallback path. Forces the axis-5 call to raise, asserts the
     axis-6 WARN is emitted on stdout and ``meta.no_silent_fallback is
     False``. This is the POSITIVE evidence that axis-6's own wiring is
     correct regardless of the upstream axis-5 block.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


try:
    import torch

    _HAS_CUDA = bool(torch.cuda.is_available())
except Exception:
    _HAS_CUDA = False


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for axis-5 KV-direct"),
]


def _load_interactive_memory_chat():
    """Load ``scripts/interactive_memory_chat.py`` as a module."""
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


def test_kv_query_turn_surfaces_real_axis6_fields(tmp_path: Path) -> None:
    """End-to-end: ``kv_query_turn`` must populate ``last_meta`` with REAL
    axis-5 field values (kv_direct_active True, no_silent_fallback True,
    vram_peak_mib > 0), not legacy sentinel strings.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat

    planted_phrase = "the saffron sidecar remembers hexagon gullies at sunrise."

    chat = MemoryChat(
        store_root=tmp_path / "store",
        model_path=None,
        max_new_tokens=64,
        memory_mode="topical",
        device="cuda",
    )
    chat.load_model()
    chat.maybe_load_retriever()
    chat.start_new_session()

    # Plant the phrase via scripted turns so the clause emitter captures
    # it in an assistant AUS3000 clause.
    scripted = [
        "Reply with exactly this sentence and nothing else: ready for kv probe.",
        (
            "Reply with exactly this sentence and nothing else, with no quotation "
            f"marks: {planted_phrase}"
        ),
        (
            "Repeat the exact same sentence again, unchanged and with no extra "
            f"words: {planted_phrase}"
        ),
    ]
    for prompt in scripted:
        chat.plain_chat_turn(prompt)

    assert chat.save_current_session(rebuild_retriever=True), (
        "save_current_session returned False — live indexer flush or "
        "retriever refresh failed"
    )

    # Fresh empty-context session so /kv_query's routing is not biased by
    # the user turns we just performed.
    chat.start_new_session()
    assert chat.retriever is not None, (
        "Retriever must be available after /save"
    )

    meta = chat.kv_query_turn(planted_phrase)
    chat.last_meta = meta  # the REPL branch sets this; mirror for the test

    # The run MUST NOT have silently fallen back to plain chat.
    assert meta.mode == "kv_direct", (
        f"expected mode=kv_direct, got {meta.mode!r} — silent fallback"
    )
    assert meta.kv_direct_active is True, (
        f"kv_direct_active must be True on the real axis-5 path; got {meta.kv_direct_active!r}"
    )
    assert meta.no_silent_fallback is True, (
        f"no_silent_fallback must be True when KV-direct ran; got {meta.no_silent_fallback!r}"
    )

    # selected_tier must not be the legacy sentinel string.
    allowed_tiers = {
        "hot", "warm", "cold", "mixed", "hot+warm", "kv_direct",
    }
    assert meta.selected_tier in allowed_tiers, (
        f"selected_tier {meta.selected_tier!r} not in allowed set {allowed_tiers}"
    )
    assert meta.selected_tier != "not-implemented-yet"

    # VRAM peak must be reported as a positive int on CUDA.
    assert isinstance(meta.vram_peak_mib, (int, float)), (
        f"vram_peak_mib must be numeric, got {type(meta.vram_peak_mib)}"
    )
    assert int(meta.vram_peak_mib) > 0, (
        f"vram_peak_mib must be > 0 on CUDA axis-5 path, got {meta.vram_peak_mib}"
    )


def test_kv_query_turn_surfaces_silent_fallback_truthfully(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive truthful-fallback test: axis-6 WARN fires when the axis-5
    call raises.

    Setup mirrors the happy-path test (plant phrase, /save, start a fresh
    session) but monkey-patches ``chat.retriever.answer_with_kv_direct`` to
    raise a simulated block. The call to ``chat.kv_query_turn`` must:

      * Print ``[WARN] axis-6: SILENT FALLBACK DETECTED`` to stdout
        (including the simulated reason string).
      * Return a ``TurnMetadata`` whose ``mode == "none"`` (fallback path
        in ``plain_chat_turn`` returns mode="none" when the retriever is
        still set but the KV-direct call failed).
      * Set ``no_silent_fallback = False`` (the axis-5 call did NOT run
        end-to-end).

    This test proves axis-6's wiring is correct independently of axis-5.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat

    planted_phrase = "the saffron sidecar remembers hexagon gullies at sunrise."

    chat = MemoryChat(
        store_root=tmp_path / "store",
        model_path=None,
        max_new_tokens=64,
        memory_mode="topical",
        device="cuda",
    )
    chat.load_model()
    chat.maybe_load_retriever()
    chat.start_new_session()

    scripted = [
        "Reply with exactly this sentence and nothing else: ready for kv probe.",
        (
            "Reply with exactly this sentence and nothing else, with no quotation "
            f"marks: {planted_phrase}"
        ),
        (
            "Repeat the exact same sentence again, unchanged and with no extra "
            f"words: {planted_phrase}"
        ),
    ]
    for prompt in scripted:
        chat.plain_chat_turn(prompt)

    assert chat.save_current_session(rebuild_retriever=True), (
        "save_current_session returned False — live indexer flush or "
        "retriever refresh failed"
    )

    chat.start_new_session()
    assert chat.retriever is not None, (
        "Retriever must be available after /save"
    )

    # Monkey-patch answer_with_kv_direct to simulate the axis-5 block.
    simulated_reason = "simulated axis-5 block"

    def _boom(*_args, **_kwargs):
        raise RuntimeError(simulated_reason)

    chat.retriever.answer_with_kv_direct = _boom  # type: ignore[assignment]

    meta = chat.kv_query_turn(planted_phrase)
    chat.last_meta = meta

    # The retriever is NOT None (we planted a session), so the
    # plain_chat_turn fallback returns mode="none" (see TurnMetadata
    # construction at plain_chat_turn line ~541).
    assert meta.mode == "none", (
        f"expected mode=none (retriever-present fallback), got {meta.mode!r}"
    )
    assert meta.no_silent_fallback is False, (
        "no_silent_fallback must be False after an axis-5 RuntimeError fallback"
    )

    captured = capsys.readouterr()
    assert "[WARN] axis-6: SILENT FALLBACK DETECTED" in captured.out, (
        "Expected axis-6 SILENT FALLBACK WARN on stdout; got:\n"
        f"{captured.out[:2000]}"
    )
    assert simulated_reason in captured.out, (
        f"Expected the simulated reason {simulated_reason!r} in WARN output; "
        f"got:\n{captured.out[:2000]}"
    )
