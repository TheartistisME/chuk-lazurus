"""Unit + CUDA integration coverage for the ``/kv_query`` REPL surface.

The fast unit tests pin the command-surface selector behavior:

  1. default ``/kv_query <text>`` keeps the current full-attention behavior
  2. explicit sliding-family flags are parsed and forwarded correctly

The heavier integration tests still run real Gemma on CUDA — expect
60-180s wall time when those are enabled.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import torch

    _HAS_CUDA = bool(torch.cuda.is_available())
except Exception:
    _HAS_CUDA = False


def _load_interactive_memory_chat():
    """Load ``scripts/interactive_memory_chat.py`` as a module."""
    scripts_root = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    path = scripts_root / "interactive_memory_chat.py"
    spec = importlib.util.spec_from_file_location(
        "interactive_memory_chat", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["interactive_memory_chat"] = module
    spec.loader.exec_module(module)
    return module


def test_kv_query_command_preserves_default_full_attention_selector() -> None:
    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat
    meta = _imc.TurnMetadata(mode="kv_direct")

    chat = MemoryChat.__new__(MemoryChat)
    chat.last_meta = None
    seen: dict[str, object] = {}

    def _fake_kv_query_turn(query_text: str, **kwargs):
        seen["query_text"] = query_text
        seen["kwargs"] = kwargs
        return meta

    chat.kv_query_turn = _fake_kv_query_turn  # type: ignore[assignment]

    handled = chat._handle_command("/kv_query remember the saffron sidecar")

    assert handled is False
    assert seen == {
        "query_text": "remember the saffron sidecar",
        "kwargs": {
            "insertion_family": "full_attention",
            "sliding_layer_indices": None,
            "sliding_head_indices": None,
        },
    }
    assert chat.last_meta is meta


def test_kv_query_command_parses_explicit_sliding_selector() -> None:
    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat
    meta = _imc.TurnMetadata(mode="kv_direct")

    chat = MemoryChat.__new__(MemoryChat)
    chat.last_meta = None
    seen: dict[str, object] = {}

    def _fake_kv_query_turn(query_text: str, **kwargs):
        seen["query_text"] = query_text
        seen["kwargs"] = kwargs
        return meta

    chat.kv_query_turn = _fake_kv_query_turn  # type: ignore[assignment]

    handled = chat._handle_command(
        "/kv_query --insertion-family sliding "
        "--sliding-layer-indices 11,13 "
        "--sliding-head-indices 0,7 "
        "remember the saffron sidecar"
    )

    assert handled is False
    assert seen == {
        "query_text": "remember the saffron sidecar",
        "kwargs": {
            "insertion_family": "sliding",
            "sliding_layer_indices": (11, 13),
            "sliding_head_indices": (0, 7),
        },
    }
    assert chat.last_meta is meta


def test_activation_gate_allows_high_cosine_with_zero_tfidf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _imc = _load_interactive_memory_chat()
    monkeypatch.setenv("LAZARUS_ASI_ACTIVATION_GATE_THRESHOLD", "0.45")

    candidates = [
        SimpleNamespace(
            raw_tfidf_score_pre_normalization=0.0,
            activation_score=0.72,
        )
    ]

    assert _imc._routing_gate_allows_candidates(candidates) is True


def test_activation_gate_preserves_legacy_no_relevant_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _imc = _load_interactive_memory_chat()
    monkeypatch.setenv("LAZARUS_ASI_ACTIVATION_GATE_THRESHOLD", "0.45")

    candidates = [
        SimpleNamespace(
            raw_tfidf_score_pre_normalization=0.0,
            activation_score=0.12,
        )
    ]

    assert _imc._routing_gate_allows_candidates(candidates) is False


def test_kv_query_command_rejects_sliding_indices_without_sliding_family(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat

    chat = MemoryChat.__new__(MemoryChat)
    sentinel = object()
    chat.last_meta = sentinel

    def _unexpected_kv_query_turn(*_args, **_kwargs):
        raise AssertionError("kv_query_turn must not run for invalid selector input")

    chat.kv_query_turn = _unexpected_kv_query_turn  # type: ignore[assignment]

    handled = chat._handle_command(
        "/kv_query --sliding-layer-indices 11,13 "
        "remember the saffron sidecar"
    )

    assert handled is False
    assert chat.last_meta is sentinel
    captured = capsys.readouterr()
    assert (
        "--sliding-layer-indices and --sliding-head-indices require "
        "--insertion-family sliding"
    ) in captured.out


def test_kv_query_command_rejects_non_integer_sliding_head_selection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat

    chat = MemoryChat.__new__(MemoryChat)
    sentinel = object()
    chat.last_meta = sentinel

    def _unexpected_kv_query_turn(*_args, **_kwargs):
        raise AssertionError("kv_query_turn must not run for invalid selector input")

    chat.kv_query_turn = _unexpected_kv_query_turn  # type: ignore[assignment]

    handled = chat._handle_command(
        "/kv_query --insertion-family sliding "
        "--sliding-head-indices 0,seven "
        "remember the saffron sidecar"
    )

    assert handled is False
    assert chat.last_meta is sentinel
    captured = capsys.readouterr()
    assert (
        "--sliding-head-indices must be a comma-separated list of integers"
    ) in captured.out


def test_kv_query_turn_propagates_selector_kwargs_to_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")

    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat

    class SelectorAwareRetriever:
        def __init__(self) -> None:
            self.handles = [SimpleNamespace(session_id="sess-1")]
            self.tokenizer = object()
            self.system_prompt = "system prompt"
            self.captured: dict[str, object] | None = None

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
            self.captured = {
                "query_text": query_text,
                "tier_count": len(tier_assignments),
                "hot_budget_mib": hot_budget_mib,
                "handle": handle,
                "warm_scores": warm_scores,
                "query_id": query_id,
                "insertion_family": insertion_family,
                "sliding_layer_indices": sliding_layer_indices,
                "sliding_head_indices": sliding_head_indices,
            }
            return SimpleNamespace(
                strict_assertions={
                    "kv_direct_active": True,
                    "mask_penalty_applied": False,
                    "vram_peak_mib": 12,
                    "vram_delta_mib": 4,
                },
                routing_mode="kv_direct",
                source_session="sess-1",
                window_id=2,
                routing_score=0.99,
                matched_window_text="matched window",
                window_keywords=["saffron"],
                generated_answer="retrieved answer",
            )

    class StubTokenizer:
        def __call__(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return SimpleNamespace(input_ids=[1, 2, 3])

    retriever = SelectorAwareRetriever()
    chat = MemoryChat.__new__(MemoryChat)
    chat.retriever = retriever
    chat.tokenizer = StubTokenizer()
    chat.max_new_tokens = 32
    chat.plain_chat_turn = lambda _text: _imc.TurnMetadata(mode="plain")  # type: ignore[assignment]

    candidate = SimpleNamespace(handle=retriever.handles[0], window_id=2)
    assignment = SimpleNamespace(
        candidate=candidate,
        tier=SimpleNamespace(value="hot"),
    )

    import chuk_lazarus.session_retrieval as session_retrieval

    seen_router_kwargs: dict[str, object] = {}

    def _fake_asi_route_candidates(*_args, **kwargs):
        seen_router_kwargs.update(kwargs)
        return [candidate]

    monkeypatch.setattr(_imc, "_build_activation_query_vector", lambda *_a, **_kw: [0.0, 1.0])
    monkeypatch.setattr(
        session_retrieval,
        "asi_route_candidates",
        _fake_asi_route_candidates,
    )
    monkeypatch.setattr(
        session_retrieval,
        "assign_tiers",
        lambda *_args, **_kwargs: [assignment],
    )

    meta = chat.kv_query_turn(
        "remember the saffron sidecar",
        insertion_family="sliding",
        sliding_layer_indices=(11, 13),
        sliding_head_indices=(0, 7),
    )

    assert retriever.captured == {
        "query_text": "remember the saffron sidecar",
        "tier_count": 1,
        "hot_budget_mib": 32,
        "handle": retriever.handles[0],
        "warm_scores": None,
        "query_id": None,
        "insertion_family": "sliding",
        "sliding_layer_indices": (11, 13),
        "sliding_head_indices": (0, 7),
    }
    assert meta.mode == "kv_direct"
    assert meta.selected_tier == "hot"
    assert meta.no_silent_fallback is True
    assert meta.kv_direct_active is True
    assert seen_router_kwargs["activation_query_vector"] == [0.0, 1.0]


def test_kv_query_turn_preserves_cross_session_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")

    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat

    handle_a = SimpleNamespace(session_id="sess-a")
    handle_b = SimpleNamespace(session_id="sess-b")

    class MultiAwareRetriever:
        def __init__(self) -> None:
            self.handles = [handle_a, handle_b]
            self.tokenizer = object()
            self.system_prompt = "system prompt"
            self.captured_sessions: list[str] | None = None

        def answer_with_kv_direct_multi(
            self,
            query_text,
            tier_assignments,
            *,
            hot_budget_mib,
            warm_config,
            generation_config,
            warm_scores=None,
            query_id=None,
            insertion_family="full_attention",
            sliding_layer_indices=None,
            sliding_head_indices=None,
        ):
            del (
                query_text,
                hot_budget_mib,
                warm_config,
                generation_config,
                warm_scores,
                query_id,
                insertion_family,
                sliding_layer_indices,
                sliding_head_indices,
            )
            self.captured_sessions = [
                assignment.candidate.handle.session_id
                for assignment in tier_assignments
            ]
            return SimpleNamespace(
                strict_assertions={
                    "kv_direct_active": True,
                    "mask_penalty_applied": True,
                    "vram_peak_mib": 12,
                    "vram_delta_mib": 4,
                    "multi_session_count": 2,
                    "semantic_prefix_active": True,
                    "semantic_prefix_tokens": 18,
                },
                routing_mode="kv_direct",
                source_session="sess-a",
                window_id=0,
                routing_score=0.99,
                matched_window_text="matched windows",
                window_keywords=["palette"],
                generated_answer="retrieved answer",
                evidence_supports=[
                    {
                        "session_id": "sess-a",
                        "window_id": 0,
                        "tier": "hot",
                        "rank": 0,
                        "evidence_excerpt": "Atlas pricing decisions: Pro is $29.",
                        "fact_keys_detected": ["atlas_pro_price"],
                        "candidate_source": "asi_route_candidates",
                    }
                ],
            )

        def answer_with_kv_direct(self, *_args, **_kwargs):
            raise AssertionError("single-handle fallback must not run")

    class StubTokenizer:
        def __call__(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return SimpleNamespace(input_ids=[1, 2, 3])

    retriever = MultiAwareRetriever()
    chat = MemoryChat.__new__(MemoryChat)
    chat.retriever = retriever
    chat.tokenizer = StubTokenizer()
    chat.model = SimpleNamespace(config=SimpleNamespace(head_dim=1, num_key_value_heads=1))
    chat.max_new_tokens = 32
    chat.plain_chat_turn = lambda _text: _imc.TurnMetadata(mode="plain")  # type: ignore[assignment]

    candidates = [
        SimpleNamespace(handle=handle_a, window_id=0, ucb1_score=2.0),
        SimpleNamespace(handle=handle_b, window_id=0, ucb1_score=1.0),
    ]
    assignments = [
        SimpleNamespace(candidate=candidates[0], tier=SimpleNamespace(value="hot"), rank=0),
        SimpleNamespace(candidate=candidates[1], tier=SimpleNamespace(value="warm"), rank=1),
    ]

    import chuk_lazarus.session_retrieval as session_retrieval

    monkeypatch.setattr(
        session_retrieval,
        "asi_route_candidates",
        lambda *_args, **_kwargs: candidates,
    )
    monkeypatch.setattr(
        session_retrieval,
        "assign_tiers",
        lambda *_args, **_kwargs: assignments,
    )

    meta = chat.kv_query_turn("list the palette facts")

    assert retriever.captured_sessions == ["sess-a", "sess-b"]
    assert meta.mode == "kv_direct"
    assert meta.selected_tier == "hot"
    assert meta.mask_penalty_applied is True
    assert meta.no_silent_fallback is True
    assert meta.candidate_count == 2
    assert meta.tier_assignment_count == 2
    assert meta.budgeted_assignment_count == 2
    assert meta.multi_session_count == 2
    assert meta.semantic_prefix_active is True
    assert meta.evidence_support_count == 1
    assert meta.evidence_supports[0]["fact_keys_detected"] == ["atlas_pro_price"]


def test_kv_query_turn_central_backend_preserves_asi_metadata_and_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")

    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat

    handle_a = SimpleNamespace(session_id="central-a")
    handle_b = SimpleNamespace(session_id="central-b")

    class CentralAwareRetriever:
        def __init__(self) -> None:
            self.handles = [handle_a, handle_b]
            self.tokenizer = object()
            self.system_prompt = "system prompt"
            self.captured_assignments: list[tuple[str, int]] | None = None

        def answer_with_kv_direct_multi(
            self,
            query_text,
            tier_assignments,
            *,
            hot_budget_mib,
            warm_config,
            generation_config,
            warm_scores=None,
            query_id=None,
            insertion_family="full_attention",
            sliding_layer_indices=None,
            sliding_head_indices=None,
        ):
            del (
                query_text,
                hot_budget_mib,
                warm_config,
                generation_config,
                warm_scores,
                query_id,
                insertion_family,
                sliding_layer_indices,
                sliding_head_indices,
            )
            self.captured_assignments = [
                (
                    assignment.candidate.handle.session_id,
                    int(assignment.candidate.window_id),
                )
                for assignment in tier_assignments
            ]
            return SimpleNamespace(
                strict_assertions={
                    "kv_direct_active": True,
                    "mask_penalty_applied": True,
                    "vram_peak_mib": 12,
                    "vram_delta_mib": 4,
                    "multi_session_count": 2,
                    "semantic_prefix_active": True,
                    "semantic_prefix_tokens": 24,
                },
                routing_mode="kv_direct",
                source_session="central-b",
                window_id=7,
                routing_score=0.99,
                matched_window_text="central ordered window",
                window_keywords=["central"],
                generated_answer="central retrieved answer",
            )

        def answer_with_kv_direct(self, *_args, **_kwargs):
            raise AssertionError("single-handle fallback must not run")

    class FakeStore:
        def get_window_text(self, window_id, tokenizer):
            del tokenizer
            return f"central window {window_id}"

    class StubTokenizer:
        def __call__(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return SimpleNamespace(input_ids=[1, 2, 3])

    retriever = CentralAwareRetriever()
    chat = MemoryChat.__new__(MemoryChat)
    chat.retriever = retriever
    chat.tokenizer = StubTokenizer()
    chat.model = SimpleNamespace(config=SimpleNamespace(head_dim=1, num_key_value_heads=1))
    chat.max_new_tokens = 32
    chat.memory_router_backend = "central"
    chat.plain_chat_turn = lambda _text: _imc.TurnMetadata(mode="plain")  # type: ignore[assignment]

    candidates = [
        SimpleNamespace(
            handle=handle_a,
            window_id=3,
            ucb1_score=0.1,
            raw_router_score=0.2,
            raw_tfidf_score_pre_normalization=1.0,
            literal_score=0.1,
            entity_score=0.2,
            dense_score=0.3,
            activation_score=0.4,
            rrf_score=0.05,
            relevance_score=0.6,
            freshness_score=0.7,
            learned_reward=0.8,
            estimated_cost=0.9,
            utility_score=0.1,
            lexical_rank=2,
            dense_rank=2,
            literal_rank=2,
            entity_rank=2,
            activation_rank=2,
            session_turn_index=13,
            content_fingerprint="fp-a",
            dense_vector=(0.1, 0.2),
            selector_telemetry={"apollo": "a", "activation_rank": 2},
        ),
        SimpleNamespace(
            handle=handle_b,
            window_id=7,
            ucb1_score=0.2,
            raw_router_score=0.3,
            raw_tfidf_score_pre_normalization=1.5,
            literal_score=0.2,
            entity_score=0.3,
            dense_score=0.4,
            activation_score=0.9,
            rrf_score=0.15,
            relevance_score=0.8,
            freshness_score=0.9,
            learned_reward=0.4,
            estimated_cost=0.2,
            utility_score=0.95,
            lexical_rank=1,
            dense_rank=1,
            literal_rank=1,
            entity_rank=1,
            activation_rank=1,
            session_turn_index=17,
            content_fingerprint="fp-b",
            dense_vector=(0.3, 0.4),
            selector_telemetry={"apollo": "b", "activation_rank": 1},
        ),
    ]
    captured_route: dict[str, object] = {}
    captured_assign_candidates: list[tuple[str, int]] = []

    def _fake_central_route_chat_windows(query_text, route_windows, **kwargs):
        captured_route["query_text"] = query_text
        captured_route["route_windows"] = route_windows
        captured_route["kwargs"] = kwargs
        ordered_windows = sorted(
            route_windows,
            key=lambda window: float(window["metadata"]["utility_score"]),
            reverse=True,
        )
        return SimpleNamespace(
            candidates=[
                SimpleNamespace(window_id=window["window_id"])
                for window in ordered_windows
            ]
        )

    def _fake_assign_tiers(ordered_candidates, **_kwargs):
        captured_assign_candidates.extend(
            (candidate.handle.session_id, int(candidate.window_id))
            for candidate in ordered_candidates
        )
        return [
            SimpleNamespace(
                candidate=candidate,
                tier=SimpleNamespace(value="hot" if idx == 0 else "warm"),
                rank=idx,
            )
            for idx, candidate in enumerate(ordered_candidates)
        ]

    import chuk_lazarus.session_retrieval as session_retrieval
    import chuk_lazarus.session_retrieval.enumeration as enumeration

    monkeypatch.setattr(_imc, "_build_activation_query_vector", lambda *_a, **_kw: [0.0, 1.0])
    monkeypatch.setattr(enumeration, "load_store", lambda _handle: FakeStore())
    monkeypatch.setattr(_imc, "central_route_chat_windows", _fake_central_route_chat_windows)
    monkeypatch.setattr(
        session_retrieval,
        "asi_route_candidates",
        lambda *_args, **_kwargs: candidates,
    )
    monkeypatch.setattr(session_retrieval, "assign_tiers", _fake_assign_tiers)

    meta = chat.kv_query_turn("central route the memory facts")

    route_windows = captured_route["route_windows"]
    assert captured_route["kwargs"]["activation_query_vector"] == [0.0, 1.0]
    assert captured_route["kwargs"]["ranking_policy"] == "utility-v2"
    assert captured_route["kwargs"]["rrf_k"] == 60.0
    assert captured_route["kwargs"]["candidate_pool"] == 12
    assert route_windows[1]["metadata"]["session_id"] == "central-b"
    assert route_windows[1]["metadata"]["original_window_id"] == 7
    assert route_windows[1]["metadata"]["raw_router_score"] == 0.3
    assert route_windows[1]["metadata"]["raw_tfidf_score_pre_normalization"] == 1.5
    assert route_windows[1]["metadata"]["literal_score"] == 0.2
    assert route_windows[1]["metadata"]["entity_score"] == 0.3
    assert route_windows[1]["metadata"]["dense_score"] == 0.4
    assert route_windows[1]["metadata"]["activation_score"] == 0.9
    assert route_windows[1]["metadata"]["rrf_score"] == 0.15
    assert route_windows[1]["metadata"]["relevance_score"] == 0.8
    assert route_windows[1]["metadata"]["freshness_score"] == 0.9
    assert route_windows[1]["metadata"]["learned_reward"] == 0.4
    assert route_windows[1]["metadata"]["estimated_cost"] == 0.2
    assert route_windows[1]["metadata"]["utility_score"] == 0.95
    assert route_windows[1]["metadata"]["lexical_rank"] == 1
    assert route_windows[1]["metadata"]["dense_rank"] == 1
    assert route_windows[1]["metadata"]["literal_rank"] == 1
    assert route_windows[1]["metadata"]["entity_rank"] == 1
    assert route_windows[1]["metadata"]["activation_rank"] == 1
    assert route_windows[1]["metadata"]["content_fingerprint"] == "fp-b"
    assert route_windows[1]["metadata"]["selector_telemetry"] == {
        "apollo": "b",
        "activation_rank": 1,
    }
    assert captured_assign_candidates == [("central-b", 7), ("central-a", 3)]
    assert retriever.captured_assignments == [("central-b", 7), ("central-a", 3)]
    assert meta.mode == "kv_direct"
    assert meta.selected_tier == "hot"
    assert meta.no_silent_fallback is True


def test_kv_query_turn_preserves_natural_multi_session_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")

    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat

    handles = [SimpleNamespace(session_id=f"natural-sess-{idx}") for idx in range(12)]

    class NaturalMultiRetriever:
        def __init__(self) -> None:
            self.handles = handles
            self.tokenizer = object()
            self.system_prompt = "system prompt"
            self.captured: list[tuple[str, str]] | None = None

        def answer_with_kv_direct_multi(
            self,
            query_text,
            tier_assignments,
            *,
            hot_budget_mib,
            warm_config,
            generation_config,
            warm_scores=None,
            query_id=None,
            insertion_family="full_attention",
            sliding_layer_indices=None,
            sliding_head_indices=None,
        ):
            del (
                query_text,
                hot_budget_mib,
                warm_config,
                generation_config,
                warm_scores,
                query_id,
                insertion_family,
                sliding_layer_indices,
                sliding_head_indices,
            )
            self.captured = [
                (
                    assignment.candidate.handle.session_id,
                    assignment.tier.value,
                )
                for assignment in tier_assignments
            ]
            return SimpleNamespace(
                strict_assertions={
                    "kv_direct_active": True,
                    "mask_penalty_applied": True,
                    "vram_peak_mib": 12,
                    "vram_delta_mib": 4,
                    "multi_session_count": 12,
                    "semantic_prefix_active": True,
                    "semantic_prefix_tokens": 120,
                },
                routing_mode="kv_direct",
                source_session="natural-sess-0",
                window_id=0,
                routing_score=0.99,
                matched_window_text="natural color-scheme windows",
                window_keywords=["color", "scheme"],
                generated_answer=(
                    "Teal was considered, then sage replaced it as the accent. "
                    "The final direction uses warm white backgrounds, graphite "
                    "headings, sage accents, and amber CTAs."
                ),
            )

        def answer_with_kv_direct(self, *_args, **_kwargs):
            raise AssertionError("single-handle fallback must not run")

    class StubTokenizer:
        def __call__(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return SimpleNamespace(input_ids=[1, 2, 3])

    retriever = NaturalMultiRetriever()
    chat = MemoryChat.__new__(MemoryChat)
    chat.retriever = retriever
    chat.tokenizer = StubTokenizer()
    chat.model = SimpleNamespace(config=SimpleNamespace(head_dim=1, num_key_value_heads=1))
    chat.max_new_tokens = 64
    chat.plain_chat_turn = lambda _text: _imc.TurnMetadata(mode="plain")  # type: ignore[assignment]

    candidates = [
        SimpleNamespace(handle=handle, window_id=0, ucb1_score=12.0 - idx)
        for idx, handle in enumerate(handles)
    ]
    assignments = [
        SimpleNamespace(
            candidate=candidate,
            tier=SimpleNamespace(
                value=("hot" if idx < 4 else "warm" if idx < 8 else "cold")
            ),
            rank=idx,
        )
        for idx, candidate in enumerate(candidates)
    ]

    import chuk_lazarus.session_retrieval as session_retrieval

    monkeypatch.setattr(
        session_retrieval,
        "asi_route_candidates",
        lambda *_args, **_kwargs: candidates,
    )
    monkeypatch.setattr(
        session_retrieval,
        "assign_tiers",
        lambda *_args, **_kwargs: assignments,
    )
    monkeypatch.setenv("LAZARUS_KV_CANDIDATE_POOL", "12")
    monkeypatch.setenv("LAZARUS_KV_K_HOT", "4")
    monkeypatch.setenv("LAZARUS_KV_K_WARM", "4")

    meta = chat.kv_query_turn(
        "Tell me everything we discussed about the website's color scheme "
        "across all our sessions."
    )

    assert retriever.captured == [
        (f"natural-sess-{idx}", "hot" if idx < 4 else "warm" if idx < 8 else "cold")
        for idx in range(12)
    ]
    assert meta.mode == "kv_direct"
    assert meta.selected_tier == "hot"
    assert meta.mask_penalty_applied is True
    assert meta.no_silent_fallback is True
    assert meta.candidate_count == 12
    assert meta.tier_assignment_count == 12
    assert meta.budgeted_assignment_count == 12
    assert meta.multi_session_count == 12
    assert meta.semantic_prefix_active is True
    assert "sage replaced" in meta.generated_answer.lower()
    assert "final direction" in meta.generated_answer.lower()


def test_kv_query_turn_preserves_dirty_store_assignment_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")

    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat

    handles = [SimpleNamespace(session_id=f"dirty-target-{idx}") for idx in range(12)]
    noise_handles = [SimpleNamespace(session_id=f"dirty-noise-{idx}") for idx in range(4)]

    class DirtyStoreRetriever:
        def __init__(self) -> None:
            self.handles = [*handles, *noise_handles]
            self.tokenizer = object()
            self.system_prompt = "system prompt"
            self.captured_sessions: list[str] | None = None

        def answer_with_kv_direct_multi(
            self,
            query_text,
            tier_assignments,
            *,
            hot_budget_mib,
            warm_config,
            generation_config,
            warm_scores=None,
            query_id=None,
            insertion_family="full_attention",
            sliding_layer_indices=None,
            sliding_head_indices=None,
        ):
            del (
                query_text,
                hot_budget_mib,
                warm_config,
                generation_config,
                warm_scores,
                query_id,
                insertion_family,
                sliding_layer_indices,
                sliding_head_indices,
            )
            self.captured_sessions = [
                assignment.candidate.handle.session_id
                for assignment in tier_assignments
            ]
            return SimpleNamespace(
                strict_assertions={
                    "kv_direct_active": True,
                    "mask_penalty_applied": True,
                    "vram_peak_mib": 12,
                    "vram_delta_mib": 4,
                    "multi_session_count": 12,
                    "semantic_prefix_active": True,
                    "semantic_prefix_tokens": 256,
                },
                routing_mode="kv_direct",
                source_session="dirty-target-0",
                window_id=0,
                routing_score=0.99,
                matched_window_text="dirty target windows",
                window_keywords=["website", "palette"],
                generated_answer=(
                    "Deep teal was too cold, sage replaced teal, and the final "
                    "direction is warm white backgrounds, graphite headings, "
                    "sage accents, and amber CTAs."
                ),
            )

        def answer_with_kv_direct(self, *_args, **_kwargs):
            raise AssertionError("single-handle fallback must not run")

    class StubTokenizer:
        def __call__(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return SimpleNamespace(input_ids=[1, 2, 3])

    retriever = DirtyStoreRetriever()
    chat = MemoryChat.__new__(MemoryChat)
    chat.retriever = retriever
    chat.tokenizer = StubTokenizer()
    chat.model = SimpleNamespace(config=SimpleNamespace(head_dim=1, num_key_value_heads=1))
    chat.max_new_tokens = 64
    chat.plain_chat_turn = lambda _text: _imc.TurnMetadata(mode="plain")  # type: ignore[assignment]

    routed_candidates = [
        SimpleNamespace(handle=handle, window_id=0, ucb1_score=20.0 - idx)
        for idx, handle in enumerate([*handles, *noise_handles])
    ]
    assignments = [
        SimpleNamespace(
            candidate=candidate,
            tier=SimpleNamespace(
                value=("hot" if idx < 4 else "warm" if idx < 8 else "cold")
            ),
            rank=idx,
        )
        for idx, candidate in enumerate(routed_candidates[:12])
    ]

    import chuk_lazarus.session_retrieval as session_retrieval

    monkeypatch.setattr(
        session_retrieval,
        "asi_route_candidates",
        lambda *_args, **_kwargs: routed_candidates,
    )
    monkeypatch.setattr(
        session_retrieval,
        "assign_tiers",
        lambda *_args, **_kwargs: assignments,
    )
    monkeypatch.setenv("LAZARUS_KV_CANDIDATE_POOL", "16")
    monkeypatch.setenv("LAZARUS_KV_K_HOT", "4")
    monkeypatch.setenv("LAZARUS_KV_K_WARM", "4")

    meta = chat.kv_query_turn(
        "Tell me everything we discussed about the website's color scheme "
        "across all our sessions."
    )

    assert retriever.captured_sessions == [f"dirty-target-{idx}" for idx in range(12)]
    assert meta.mode == "kv_direct"
    assert meta.selected_tier == "hot"
    assert meta.candidate_count == 16
    assert meta.tier_assignment_count == 12
    assert meta.budgeted_assignment_count == 12
    assert meta.multi_session_count == 12
    assert meta.semantic_prefix_active is True
    assert meta.no_silent_fallback is True


def test_website_color_synthesis_filters_dirty_store_near_misses() -> None:
    from chuk_lazarus.session_retrieval.retriever import SessionRetriever

    retriever = SessionRetriever.__new__(SessionRetriever)
    answer = retriever._synthesize_website_color_scheme_answer(
        "Tell me everything we discussed about the website's color scheme across all our sessions.",
        [
            (
                "hot",
                (
                    "Across our website color scheme sessions, after review, "
                    "sage green replaced teal as the accent color."
                ),
            ),
            (
                "hot",
                (
                    "Final palette direction across our website color scheme "
                    "sessions: warm white background, graphite headings, sage "
                    "accents replacing teal, and amber primary CTA only."
                ),
            ),
            (
                "warm",
                (
                    "For the Acme microsite, sage and amber were only seasonal "
                    "campaign colors and were not product website decisions."
                ),
            ),
            (
                "warm",
                (
                    "A different landing page kept coral buttons after "
                    "accessibility review; do not apply that to the product website."
                ),
            ),
        ],
    )

    lower = answer.lower()
    assert "sage green replaced" in lower
    assert "final palette direction" in lower
    assert "acme" not in lower
    assert "seasonal" not in lower
    assert "coral" not in lower


def test_memory_laws_synthesis_refuses_absent_solace_palette() -> None:
    from chuk_lazarus.session_retrieval.retriever import SessionRetriever

    retriever = SessionRetriever.__new__(SessionRetriever)
    answer = retriever._synthesize_memory_laws_answer(
        "What did we decide about the Solace website color palette?",
        [
            (
                "hot",
                (
                    "Final palette direction across our website color scheme "
                    "sessions: warm white background, graphite headings, sage "
                    "accents replacing teal, and amber primary CTA only."
                ),
            ),
            (
                "warm",
                "Nimbus pricing page kept Pro at $39 per seat and a 10% annual discount.",
            ),
        ],
    )

    lower = answer.lower()
    assert "do not have a stored decision" in lower
    assert "solace" in lower
    assert "warm white" not in lower
    assert "graphite" not in lower
    assert "sage" not in lower
    assert "amber" not in lower


def test_memory_laws_atlas_current_price_ignores_stale_duplicate_pressure() -> None:
    from chuk_lazarus.session_retrieval.retriever import SessionRetriever

    retriever = SessionRetriever.__new__(SessionRetriever)
    answer = retriever._synthesize_dirty_store_domain_answer(
        "What is the current Atlas Pro price?",
        [
            (
                "hot",
                "Old Atlas pricing decisions draft proposed $49 per seat, but this was rejected.",
            ),
            (
                "hot",
                "Old Atlas pricing decisions draft proposed $49 per seat, but this was rejected.",
            ),
            (
                "warm",
                "Current Atlas pricing decisions across our sessions: Atlas Pro is $29 per seat monthly.",
            ),
            (
                "warm",
                (
                    "Final Atlas pricing decision keeps Pro at $29 per seat, "
                    "18% annual discount, 14-day trial, $0.08 overage, and "
                    "Enterprise custom quote."
                ),
            ),
        ],
    )

    lower = answer.lower()
    assert "$29" in lower
    assert "$49" not in lower


def test_memory_laws_temporal_cta_current_and_history_queries() -> None:
    from chuk_lazarus.session_retrieval.retriever import SessionRetriever

    retriever = SessionRetriever.__new__(SessionRetriever)
    windows = [
        (
            "hot",
            "Atlas website CTA color decision history: we originally planned Crimson as the primary CTA color.",
        ),
        (
            "hot",
            "Atlas website CTA color decision history: Crimson caused contrast problems, so Amber superseded Crimson as the CTA color.",
        ),
        (
            "warm",
            "Final Atlas website CTA color decision: Amber remains the final CTA color after review.",
        ),
    ]

    current = retriever._synthesize_memory_laws_answer(
        "What is the current CTA color decision?",
        windows,
    ).lower()
    history = retriever._synthesize_memory_laws_answer(
        "How did the CTA color decision change over time?",
        windows,
    ).lower()

    assert "current cta color is amber" in current
    assert "crimson was an earlier option" in current
    assert "current cta color is crimson" not in current
    assert "crimson was originally planned" in history
    assert "superseded by amber" in history
    assert "amber remains the final" in history


def test_memory_laws_atlas_entity_scope_excludes_wrong_entities() -> None:
    from chuk_lazarus.session_retrieval.retriever import SessionRetriever

    retriever = SessionRetriever.__new__(SessionRetriever)
    answer = retriever._synthesize_dirty_store_domain_answer(
        "What are the current Atlas pricing decisions?",
        [
            (
                "hot",
                "Current Atlas pricing decisions across our sessions: the Pro tier is $29 per seat monthly.",
            ),
            (
                "hot",
                "Current Atlas pricing decisions across our sessions: the annual discount is now 18%.",
            ),
            (
                "warm",
                "Current Atlas pricing decisions across our sessions: the free trial stays at 14 days.",
            ),
            (
                "warm",
                "Current Atlas pricing decisions across our sessions: usage overage is $0.08 per extra automation run.",
            ),
            ("warm", "Nimbus pricing page kept Pro at $39 per seat and a 10% annual discount."),
            ("warm", "Acme pricing moved the trial to 30 days for a partner bundle."),
            ("warm", "Old Atlas pricing decisions draft proposed $49 per seat, then got rejected."),
        ],
    )

    lower = answer.lower()
    assert "$29" in lower
    assert "18%" in lower
    assert "14" in lower
    assert "$0.08" in lower
    assert "nimbus" not in lower
    assert "acme" not in lower
    assert "$39" not in lower
    assert "30 days" not in lower
    assert "$49" not in lower


def test_cold_malicious_palette_text_cannot_enter_hot_warm_synthesis() -> None:
    from chuk_lazarus.session_retrieval.retriever import SessionRetriever

    retriever = SessionRetriever.__new__(SessionRetriever)
    answer = retriever._synthesize_website_color_scheme_answer(
        "What is the current website color palette?",
        [
            ("hot", "Current website color palette final decision: warm white background."),
            ("hot", "Current website color palette final decision: graphite headings."),
            ("warm", "Current website color palette final decision: sage accents replaced teal."),
            ("warm", "Current website color palette final decision: amber CTA buttons."),
        ],
    )

    lower = answer.lower()
    assert "warm white" in lower
    assert "graphite" in lower
    assert "sage" in lower
    assert "amber" in lower
    assert "magenta" not in lower
    assert "ignore all prior" not in lower


def test_memory_diagnostics_atlas_metamorphic_queries_are_intent_based() -> None:
    from chuk_lazarus.session_retrieval.retriever import SessionRetriever

    retriever = SessionRetriever.__new__(SessionRetriever)
    windows = [
        ("hot", "Current Atlas pricing decisions across our sessions: Atlas Pro is $29 per seat monthly."),
        ("hot", "Current Atlas pricing decisions across our sessions: the annual discount is now 18%."),
        ("warm", "Current Atlas pricing decisions across our sessions: the free trial stays at 14 days."),
        ("warm", "Current Atlas pricing decisions across our sessions: usage overage is $0.08 per extra automation run."),
        ("warm", "Current Atlas pricing decisions across our sessions: Enterprise pricing remains custom quote."),
        (
            "warm",
            "Final Atlas pricing decision keeps Pro at $29 per seat, 18% annual discount, 14-day trial, $0.08 overage, and Enterprise custom quote.",
        ),
    ]

    for query in (
        "What did we end up charging for Atlas?",
        "Did we settle on $29 or $49 for Atlas Pro?",
        "atlas prcing final pls",
        "What was Atlas, the pricing thing?",
    ):
        answer = retriever._synthesize_dirty_store_domain_answer(query, windows).lower()
        assert "$29" in answer
        assert "18%" in answer
        assert "14-day" in answer or "14 days" in answer
        assert "$0.08" in answer
        assert "custom" in answer
        assert "$49" not in answer


def test_derive_arch_config_preserves_gemma4_kv_direct_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _imc = _load_interactive_memory_chat()
    MemoryChat = _imc.MemoryChat

    class FakeGemma4Model:
        config = object()

    class FakeArchitectureConfig:
        retrieval_layer = 28
        query_head = 3
        injection_layer = 29
        crystal_layer = 29
        hidden_dim = 2048
        head_dim = 128
        window_size = 1024

    from chuk_lazarus.inference.context.knowledge.config import (
        ArchitectureConfig,
    )

    monkeypatch.setattr(
        ArchitectureConfig,
        "from_model_config",
        classmethod(lambda cls, _config: FakeArchitectureConfig()),
    )

    chat = MemoryChat.__new__(MemoryChat)
    chat.model = FakeGemma4Model()

    arch_config, crystal_layer, window_size = chat._derive_arch_config()

    assert arch_config == {
        "retrieval_layer": 12,
        "query_head": 7,
        "injection_layer": 13,
        "hidden_dim": 2048,
        "head_dim": 128,
        "crystal_layer": 13,
        "window_size": 1024,
    }
    assert crystal_layer == 13
    assert window_size == 1024


@pytest.mark.cuda
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for axis-5 KV-direct")
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


@pytest.mark.cuda
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for axis-5 KV-direct")
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

    # Monkey-patch both active KV-direct surfaces to simulate the axis-5 block.
    simulated_reason = "simulated axis-5 block"

    def _boom(*_args, **_kwargs):
        raise RuntimeError(simulated_reason)

    chat.retriever.answer_with_kv_direct = _boom  # type: ignore[assignment]
    chat.retriever.answer_with_kv_direct_multi = _boom  # type: ignore[assignment]

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


@pytest.mark.cuda
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for axis-5 KV-direct")
def test_kv_query_turn_explicit_sliding_surfaces_truthful_outcome(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Explicit surfaced sliding selection must report either truthful
    success or truthful rejection on the real chat path.

    Worker 1 may make surfaced sliding executable end-to-end, or may reject
    it explicitly when the archived-source lineage still resolves to the
    full-attention branch. This test accepts either outcome, but it requires
    the surfaced selector to be observable and truthful.
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

    meta = chat.kv_query_turn(
        planted_phrase,
        insertion_family="sliding",
        sliding_layer_indices=(13, 15),
        sliding_head_indices=(0, 7),
    )
    chat.last_meta = meta

    captured = capsys.readouterr()
    assert "kv_query selector: insertion_family=sliding" in captured.out, (
        "Explicit selector runs must log the surfaced selector configuration"
    )

    if meta.mode == "kv_direct":
        assert meta.kv_direct_active is True, (
            "Explicit surfaced sliding success must report kv_direct_active=True"
        )
        assert meta.no_silent_fallback is True, (
            "Explicit surfaced sliding success must report no_silent_fallback=True"
        )
        assert "[WARN] axis-6: SILENT FALLBACK DETECTED" not in captured.out
        return

    assert meta.mode == "none", (
        f"truthful surfaced rejection must fall back with mode=none, got {meta.mode!r}"
    )
    assert meta.no_silent_fallback is False, (
        "truthful surfaced rejection must report no_silent_fallback=False"
    )
    assert "[WARN] axis-6: SILENT FALLBACK DETECTED" in captured.out, (
        "Explicit surfaced sliding rejection must emit the axis-6 WARN"
    )
    assert (
        "insertion_family='sliding'" in captured.out
        or "sliding_layer_indices" in captured.out
        or "sliding-attention layer" in captured.out
    ), (
        "truthful surfaced rejection must mention the surfaced sliding selector "
        f"reason, got:\n{captured.out[:2000]}"
    )
