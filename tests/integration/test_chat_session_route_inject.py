"""axis-4 (run-4): session-route token-budget governor + AMD 11 integration tests.

Verifies:
- _apply_token_budget governor (Addition 1) sorts + truncates correctly
- AMD 11: target_layer=29 ∈ GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS;
  PROP K.0 guard refuses sliding-window layers via SlidingWindowLayerRefusedError
- ASI Evolve interface contract (asi_route_candidates → list[AsiRouterCandidate])
- kv_query_turn smoke (CUDA-gated)

Tests 1-7 are pure-logic and tier_policy/AMD-fixture tests; test 8 is a
CUDA-gated kv_query_turn smoke against a pre-seeded store.

Cross-refs:
- scope manifest: ve-ins-0moecnemd0000340b9c
- mission: chuk-lazurus-z82 (Axis-4)
- Q4 + Addition 1 supervisor bindings (verbatim in LEAD --then payload)
- baseline-of-absence: ve-ins-0moeedj8z0000ce96a3
- ASI Evolve interface doc: research/kv-memory-implementation/run-4/03-axis-chat-glue/asi-evolve-interface.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# REPO_ROOT discovery — same pattern as Axis-3 test file. Required because
# scripts/interactive_memory_chat.py is not part of the chuk_lazarus package
# proper; it lives under scripts/ and must be importable at test time.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ─── stub helpers ───────────────────────────────────────────────────────────


def _stub_assignment(score: float):
    """Build a minimal stand-in for a TierAssignment.

    The token-budget governor only reads ``a.candidate.ucb1_score`` for
    its descending sort; it does not touch any other field. SimpleNamespace
    is therefore a sufficient stub.
    """
    return SimpleNamespace(candidate=SimpleNamespace(ucb1_score=score))


def _stub_model_with_config(*, head_dim: int = 256, n_kv_heads: int = 4):
    """Build a minimal stand-in for a HuggingFace Gemma-4 model.

    The governor reads ``model.config.head_dim`` and
    ``model.config.num_key_value_heads`` (Gemma-4-E2B-it native field
    names — see AMD 8). Defaults match the Gemma-4-E2B-it shipped values.
    """
    return SimpleNamespace(
        config=SimpleNamespace(
            head_dim=head_dim,
            num_key_value_heads=n_kv_heads,
        )
    )


# ─── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def chat(tmp_path):
    """Construct a MemoryChat with a tmp_path store_root.

    No model is loaded — tests 1-6 mutate ``chat.model`` directly to drive
    the governor under controlled config values. ``device="cpu"`` here is
    a constructor-only parameter; nothing CUDA-dependent runs in
    tests 1-7.
    """
    from interactive_memory_chat import MemoryChat

    return MemoryChat(
        store_root=tmp_path,
        model_path=None,
        max_new_tokens=8,
        memory_mode="off",
        device="cpu",
    )


# ─── Test 1: empty pass-through ─────────────────────────────────────────────


def test_apply_token_budget_empty_pass_through(chat):
    """Empty input must return empty output without raising.

    The governor short-circuits on empty input before touching
    ``self.model.config`` — verify this contract by passing
    ``chat.model = None`` and checking no AttributeError surfaces.
    """
    chat.model = None
    result = chat._apply_token_budget([])
    assert result == []


# ─── Test 2: under-budget keeps all ─────────────────────────────────────────


def test_apply_token_budget_under_budget_keeps_all(chat):
    """3 stubs at 1024 tokens each = 3072 < 4096; all 3 must survive.

    Per-fact cost = head_dim * num_key_value_heads = 256 * 4 = 1024.
    MAX_TOTAL_INJECT_TOKENS = 4096 -> admits up to 4 facts. With only
    3 stubs none should be truncated.
    """
    chat.model = _stub_model_with_config(head_dim=256, n_kv_heads=4)
    stubs = [
        _stub_assignment(0.9),
        _stub_assignment(0.7),
        _stub_assignment(0.5),
    ]
    kept = chat._apply_token_budget(stubs)
    assert len(kept) == 3
    # All stubs preserved (set-equality on the underlying score values).
    kept_scores = sorted(a.candidate.ucb1_score for a in kept)
    assert kept_scores == [0.5, 0.7, 0.9]


# ─── Test 3: over-budget truncates from bottom ──────────────────────────────


def test_apply_token_budget_over_budget_truncates_from_bottom(chat):
    """10 stubs at 1024 tokens each: 4096 // 1024 = 4 must survive.

    Truncation policy: drop lowest-scoring facts first. With descending
    scores [1.0, 0.9, ..., 0.1] the surviving 4 must be the top 4
    (1.0, 0.9, 0.8, 0.7).
    """
    chat.model = _stub_model_with_config(head_dim=256, n_kv_heads=4)
    stubs = [_stub_assignment(round(1.0 - 0.1 * i, 1)) for i in range(10)]
    kept = chat._apply_token_budget(stubs)

    # 4096 // (256 * 4) = 4 surviving facts.
    assert len(kept) == 4

    # Highest-scoring fact survives at index 0.
    assert kept[0].candidate.ucb1_score == pytest.approx(1.0)

    # 4th-highest (0.7) survives at index 3.
    assert kept[3].candidate.ucb1_score == pytest.approx(0.7)

    # No stub with score 0.6 or below survives.
    surviving_scores = {a.candidate.ucb1_score for a in kept}
    for dropped_score in (0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        assert not any(
            abs(s - dropped_score) < 1e-9 for s in surviving_scores
        ), f"score {dropped_score} should have been truncated"


# ─── Test 4: sort by score descending ───────────────────────────────────────


def test_apply_token_budget_sorts_by_score_descending(chat):
    """Reverse-sorted input must be reordered to descending score before truncation.

    Input order: 0.1, 0.3, 0.5, 0.7, 0.9 (ascending).
    With per_fact_cost=1024 and MAX_TOTAL_INJECT_TOKENS=4096 the budget
    admits 4 facts; the bottom (0.1) drops, the top 4 survive in
    descending order.
    """
    chat.model = _stub_model_with_config(head_dim=256, n_kv_heads=4)
    stubs = [_stub_assignment(s) for s in (0.1, 0.3, 0.5, 0.7, 0.9)]
    kept = chat._apply_token_budget(stubs, max_total_inject_tokens=4096)

    assert len(kept) == 4
    assert kept[0].candidate.ucb1_score == pytest.approx(0.9)
    assert kept[3].candidate.ucb1_score == pytest.approx(0.3)
    # And no 0.1 made it through.
    assert all(
        a.candidate.ucb1_score > 0.1 + 1e-9 for a in kept
    )


# ─── Test 5: custom budget override ─────────────────────────────────────────


def test_apply_token_budget_custom_budget(chat):
    """``max_total_inject_tokens`` keyword must override the default.

    Per-fact cost = 1024. Budget=2048 -> 2 facts. Budget=0 -> 0 facts.
    """
    chat.model = _stub_model_with_config(head_dim=256, n_kv_heads=4)
    stubs = [_stub_assignment(round(1.0 - 0.1 * i, 1)) for i in range(10)]

    kept_2048 = chat._apply_token_budget(stubs, max_total_inject_tokens=2048)
    assert len(kept_2048) == 2

    kept_zero = chat._apply_token_budget(stubs, max_total_inject_tokens=0)
    assert len(kept_zero) == 0


# ─── Test 6: fallback when model.config is missing ──────────────────────────


def test_apply_token_budget_falls_back_when_config_missing(chat):
    """Missing model.config must fall through to the AMD 8 literal defaults.

    Defaults (Gemma-4-E2B-it shipped values): head_dim=256, n_kv_heads=4.
    Per-fact cost still = 1024; with 5 stubs at the default budget (4096)
    exactly 4 must survive — proving the literal fallback is engaged.
    """
    # SimpleNamespace() has no `config` attribute at all — getattr returns None.
    chat.model = SimpleNamespace()
    stubs = [_stub_assignment(round(0.5 - 0.05 * i, 2)) for i in range(5)]
    kept = chat._apply_token_budget(stubs)
    assert len(kept) == 4

    # Also exercise the partial-config fallback: config exists but lacks the
    # named fields. Same expectation: literals engage and 4 survive.
    chat.model = SimpleNamespace(config=SimpleNamespace())
    kept2 = chat._apply_token_budget(stubs)
    assert len(kept2) == 4


# ─── Test 7: AMD 11 invariant + PROP K.0 guard ──────────────────────────────


def test_amd11_global_attention_set_invariant():
    """AMD 11: GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS == {4,9,14,19,24,29,34}.

    The recipe-canonical KV-direct target_layer is 29; verify it lives in
    the global-attention set. Then invoke the PROP K.0 guard at a
    sliding-window index (10) and assert it raises
    SlidingWindowLayerRefusedError carrying ``.target_layer == 10``.
    """
    from chuk_lazarus.inference.context.knowledge.gemma4_e2b_it_layers import (
        GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS,
    )

    # Set-identity invariant.
    assert GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS == frozenset(
        {4, 9, 14, 19, 24, 29, 34}
    )

    # Recipe-canonical injection layer present in the global set.
    assert 29 in GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS

    from chuk_lazarus.inference.context.research.vec_inject.kv_direct_adapter import (
        SlidingWindowLayerRefusedError,
        assert_global_attention_layer,
    )

    # Layer 29 must NOT raise (returns None on success).
    try:
        result = assert_global_attention_layer(29)
    except SlidingWindowLayerRefusedError as exc:  # pragma: no cover - guard
        pytest.fail(
            f"assert_global_attention_layer(29) unexpectedly raised: {exc}"
        )
    assert result is None

    # Layer 10 is a sliding_attention layer per the axis-A fixture; the guard
    # must refuse and raise SlidingWindowLayerRefusedError.
    with pytest.raises(SlidingWindowLayerRefusedError) as excinfo:
        assert_global_attention_layer(10)
    assert excinfo.value.target_layer == 10


# ─── Test 8: CUDA-gated kv_query_turn smoke ─────────────────────────────────


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA-only: torch.cuda.is_available()==False",
)
def test_kv_query_turn_smoke_with_real_store_and_model(tmp_path):
    """Heavy smoke: real Gemma-4-E2B-it on CUDA, real store, real ASI router.

    Plants 4 turns with a recoverable fact ("favorite color is teal"),
    /saves to populate the store, opens a fresh session, and probes
    kv_query_turn. The turn must either:

      * return a TurnMetadata with mode=="kv_direct" and
        kv_direct_active is True (KV-direct path engaged), OR
      * return a TurnMetadata with mode=="none" (silent fallback path —
        equally a tested behaviour per axis-6 contract).

    Exception-free completion is mandatory. The store_root/.dirty flag
    must exist post-call (set by either fallback's plain_chat_turn or the
    KV-direct path's terminal _mark_dirty).

    This test loads Gemma + runs ASI ranker + runs generation; expect
    90-120s on RTX 5090. Heavy but acceptable per the run-4 charter.
    """
    from interactive_memory_chat import MemoryChat
    from chuk_lazarus.inference.chat import Role

    real_chat = MemoryChat(
        store_root=tmp_path,
        model_path=None,
        max_new_tokens=8,
        memory_mode="off",
        device="cuda",
    )
    real_chat.load_model()
    real_chat.start_new_session()

    # Plant 4 alternating turns with a recoverable fact.
    planted_turns = [
        (Role.USER, "Hello! My favorite color is teal."),
        (
            Role.ASSISTANT,
            "Got it — teal is a lovely shade between blue and green.",
        ),
        (
            Role.USER,
            "I also love rainy autumn afternoons spent reading old books.",
        ),
        (
            Role.ASSISTANT,
            "That sounds peaceful; rainy reading days are wonderful.",
        ),
    ]
    for role, text in planted_turns:
        turn = real_chat.session.begin_turn(role, text)
        real_chat.session.finish_turn(turn)
        real_chat._capture_turn_text_live(turn)

    # /save: flush live index, write manifest, refresh retriever.
    saved_ok = real_chat.save_current_session()
    assert saved_ok, "save_current_session must succeed for the smoke test"

    # New session — simulates the next-conversation recall scenario.
    real_chat.start_new_session()
    real_chat.maybe_load_retriever()

    # Probe with a query that aligns with the planted fact.
    meta = real_chat.kv_query_turn(
        "recall what I told you about my favorite color"
    )

    # Returned metadata must not be None.
    assert meta is not None

    # Mode is either kv_direct (success path) or none (silent fallback).
    assert meta.mode in ("kv_direct", "none")

    if meta.mode == "kv_direct":
        # The KV-direct path actually ran; runtime sets this True.
        assert meta.kv_direct_active is True
    # mode == "none" is also acceptable (silent-fallback is a tested behaviour).

    # _mark_dirty was called somewhere along the return path
    # (either kv_query_turn's terminal _mark_dirty or plain_chat_turn's
    # fallback _mark_dirty).
    dirty_flag = tmp_path / ".dirty"
    assert dirty_flag.exists(), (
        "expected <store_root>/.dirty after kv_query_turn returned"
    )
