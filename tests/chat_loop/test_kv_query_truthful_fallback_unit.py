"""Pure-unit test for the ``retriever is None`` early-exit branch of
``MemoryChat.kv_query_turn``.

This test exercises ONLY the first branch of the axis-6 truthful-fallback
logic — when no retriever has been built yet. It does not load Gemma and
does not require CUDA, so it runs in any environment.

Justification: the CUDA-gated integration tests in
``test_kv_query_repl.py`` cover the axis-5-exception path, but the
``retriever is None`` path is equally important axis-6 wiring and needs a
deterministic unit-level assertion that holds on CPU-only machines too.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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


def test_kv_query_turn_returns_none_when_retriever_absent() -> None:
    """``kv_query_turn`` must short-circuit with mode="none" and
    no_silent_fallback=False when the retriever has not been built.

    This is the first branch of the axis-6 truthful-fallback logic —
    proves axis-6 does NOT fabricate a success when there is literally no
    retriever to call. Uses ``__new__`` to bypass ``__init__`` (no model,
    no CUDA, no filesystem) because the method only reads ``self.retriever``
    on this branch.
    """
    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat
    TurnMetadata = _imc.TurnMetadata

    chat = MemoryChat.__new__(MemoryChat)  # bypass __init__
    chat.retriever = None

    meta = chat.kv_query_turn("any question here")

    assert isinstance(meta, TurnMetadata), (
        f"expected TurnMetadata, got {type(meta).__name__}"
    )
    assert meta.mode == "none", (
        f"expected mode=none when retriever absent, got {meta.mode!r}"
    )
    assert meta.no_silent_fallback is False, (
        "no_silent_fallback must be False when retriever absent — "
        "axis-5 did NOT run end-to-end"
    )
