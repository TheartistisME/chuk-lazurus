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
from types import ModuleType, SimpleNamespace

import pytest


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


def _install_kv_query_dependency_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate: SimpleNamespace,
    assignment: SimpleNamespace,
) -> None:
    """Install just enough import targets for ``kv_query_turn`` to run
    without importing the real torch-backed runtime modules.
    """

    torch_runtime = ModuleType(
        "chuk_lazarus.inference.backends.torch_runtime"
    )

    class WarmPenaltyConfig:
        def __init__(
            self,
            penalty_value: float = 0.0,
            hot_bonus_value: float = 0.0,
        ) -> None:
            self.penalty_value = penalty_value
            self.hot_bonus_value = hot_bonus_value

    torch_runtime.WarmPenaltyConfig = WarmPenaltyConfig
    monkeypatch.setitem(
        sys.modules,
        "chuk_lazarus.inference.backends.torch_runtime",
        torch_runtime,
    )

    generation = ModuleType("chuk_lazarus.inference.generation")

    class GenerationConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    generation.GenerationConfig = GenerationConfig
    monkeypatch.setitem(
        sys.modules,
        "chuk_lazarus.inference.generation",
        generation,
    )

    session_retrieval = ModuleType("chuk_lazarus.session_retrieval")
    session_retrieval.asi_route_candidates = (
        lambda *_args, **_kwargs: [candidate]
    )
    session_retrieval.assign_tiers = (
        lambda *_args, **_kwargs: [assignment]
    )
    monkeypatch.setitem(
        sys.modules,
        "chuk_lazarus.session_retrieval",
        session_retrieval,
    )


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


def test_kv_query_turn_falls_back_when_runtime_lacks_selector_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Explicit selectors must reject truthfully when the runtime surface
    cannot accept the selector kwargs.
    """
    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat
    TurnMetadata = _imc.TurnMetadata

    class LegacyRuntime:
        def generate_with_kv_direct_materialization(
            self,
            prompt,
            config,
            *,
            materialization,
            per_window_token_ranges,
            tier_assignments,
            warm_config,
            warm_scores=None,
            source_layer=None,
            query_id=None,
        ):
            raise AssertionError(
                "selector support must be rejected before runtime execution"
            )

    class SelectorAwareRetriever:
        def __init__(self) -> None:
            self.handles = [SimpleNamespace(session_id="sess-1")]
            self.tokenizer = object()
            self.system_prompt = "system prompt"
            self.runtime = LegacyRuntime()

        def answer_with_kv_direct(
            self,
            query_text,
            tier_assignments,
            *,
            hot_budget_mib,
            warm_config,
            generation_config,
            warm_scores=None,
            query_id=None,
            handle=None,
            insertion_family="full_attention",
            sliding_layer_indices=None,
            sliding_head_indices=None,
        ):
            raise AssertionError(
                "selector support must be rejected before retriever execution"
            )

    retriever = SelectorAwareRetriever()
    candidate = SimpleNamespace(handle=retriever.handles[0], window_id=2)
    assignment = SimpleNamespace(
        candidate=candidate,
        tier=SimpleNamespace(value="hot"),
    )
    _install_kv_query_dependency_stubs(
        monkeypatch,
        candidate=candidate,
        assignment=assignment,
    )

    chat = MemoryChat.__new__(MemoryChat)
    chat.retriever = retriever
    chat.max_new_tokens = 32
    fallback_meta = TurnMetadata(
        mode="none",
        no_silent_fallback=False,
        generated_answer="plain fallback",
    )
    chat.plain_chat_turn = lambda _text: fallback_meta  # type: ignore[assignment]

    meta = chat.kv_query_turn(
        "remember the saffron sidecar",
        insertion_family="sliding",
        sliding_layer_indices=(13,),
        sliding_head_indices=(0, 7),
    )

    assert meta is fallback_meta
    assert meta.mode == "none"
    assert meta.no_silent_fallback is False
    captured = capsys.readouterr()
    assert "[WARN] axis-6: SILENT FALLBACK DETECTED" in captured.out
    assert (
        "runtime.generate_with_kv_direct_materialization is missing selector "
        "kwargs"
    ) in captured.out


def test_kv_query_turn_falls_back_truthfully_on_invalid_sliding_selection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid surfaced sliding selections must reject truthfully instead of
    pretending the KV-direct path succeeded.
    """
    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat
    TurnMetadata = _imc.TurnMetadata

    class SelectorAwareRuntime:
        def generate_with_kv_direct_materialization(
            self,
            prompt,
            config,
            *,
            materialization,
            per_window_token_ranges,
            tier_assignments,
            warm_config,
            warm_scores=None,
            source_layer=None,
            query_id=None,
            insertion_family="full_attention",
            sliding_layer_indices=None,
            sliding_head_indices=None,
        ):
            raise AssertionError(
                "invalid selector rejection is exercised at retriever call time"
            )

    class RejectingRetriever:
        def __init__(self) -> None:
            self.handles = [SimpleNamespace(session_id="sess-1")]
            self.tokenizer = object()
            self.system_prompt = "system prompt"
            self.runtime = SelectorAwareRuntime()

        def answer_with_kv_direct(
            self,
            query_text,
            tier_assignments,
            *,
            hot_budget_mib,
            warm_config,
            generation_config,
            warm_scores=None,
            query_id=None,
            handle=None,
            insertion_family="full_attention",
            sliding_layer_indices=None,
            sliding_head_indices=None,
        ):
            assert insertion_family == "sliding"
            assert sliding_layer_indices == (13,)
            assert sliding_head_indices == (999,)
            raise RuntimeError(
                "generate_with_kv_direct_materialization: "
                "sliding_head_indices (999,) out of range for 8 attention heads."
            )

    retriever = RejectingRetriever()
    candidate = SimpleNamespace(handle=retriever.handles[0], window_id=2)
    assignment = SimpleNamespace(
        candidate=candidate,
        tier=SimpleNamespace(value="hot"),
    )
    _install_kv_query_dependency_stubs(
        monkeypatch,
        candidate=candidate,
        assignment=assignment,
    )

    chat = MemoryChat.__new__(MemoryChat)
    chat.retriever = retriever
    chat.max_new_tokens = 32
    fallback_meta = TurnMetadata(
        mode="none",
        no_silent_fallback=False,
        generated_answer="plain fallback",
    )
    chat.plain_chat_turn = lambda _text: fallback_meta  # type: ignore[assignment]

    meta = chat.kv_query_turn(
        "remember the saffron sidecar",
        insertion_family="sliding",
        sliding_layer_indices=(13,),
        sliding_head_indices=(999,),
    )

    assert meta is fallback_meta
    assert meta.mode == "none"
    assert meta.no_silent_fallback is False
    captured = capsys.readouterr()
    assert "kv_query selector: insertion_family=sliding" in captured.out
    assert "[WARN] axis-6: SILENT FALLBACK DETECTED" in captured.out
    assert "sliding_head_indices (999,)" in captured.out
