"""Tests for :mod:`chuk_lazarus.session_retrieval.tier_policy`.

Frozen contract: ``ve-ins-0mo9p8kou0000d20e0d`` (axis-3 of
``asi-kv-direct-chat`` run 1). Satisfies the four verification obligations:

1. Property test — tier counts match the policy for arbitrary candidate lists.
2. Determinism test — identical input twice yields byte-identical output.
3. Round-trip test — :class:`TierAssignment` survives JSON serialisation.
4. Edge case — candidate list shorter than ``K_HOT`` yields all ``HOT``.

All tests are pure (no disk I/O, no CUDA, no network); they construct
:class:`AsiRouterCandidate` instances directly via a local builder so the file
stays deterministic and fast.
"""

from __future__ import annotations

import json
import random
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from chuk_lazarus.session_retrieval import tier_policy as tier_policy_module
from chuk_lazarus.session_retrieval.asi_router import AsiRouterCandidate
from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle
from chuk_lazarus.session_retrieval.tier_policy import (
    POLICY_VERSION_RANK_V1,
    POLICY_VERSION_UTILITY_V2,
    TIER_POLICY_SCHEMA_VERSION,
    TierAssignment,
    TierLabel,
    assign_tiers,
    tier_assignment_from_dict,
    tier_assignment_to_dict,
    tier_assignments_from_json,
    tier_assignments_to_json,
)


# ---------------------------------------------------------------------------
# Local builders — tier_policy is pure, so we do NOT need disk fixtures.
# ---------------------------------------------------------------------------


_FAKE_ROOT = Path("/tmp/fake")


def _make_handle(
    session_id: str = "a" * 32,
    *,
    original_input_dir: Path | None = None,
) -> CheckpointHandle:
    """Return a lightweight :class:`CheckpointHandle` — paths need not exist."""
    return CheckpointHandle(
        session_id=session_id,
        checkpoint_dir=_FAKE_ROOT / session_id,
        torch_store_dir=_FAKE_ROOT / session_id / "torch_store",
        manifest={"clause_aligned": True, "num_windows": 1, "num_entries": 1},
        original_input_dir=original_input_dir,
    )


def _make_candidate(
    *,
    session_id: str = "a" * 32,
    window_id: int = 0,
    ucb1_score: float = 1.0,
    raw_router_score: float = 0.5,
    island_id: int = 0,
    visit_count: int = 0,
    mean_reward: float = 0.0,
    original_input_dir: Path | None = None,
) -> AsiRouterCandidate:
    """Construct a single :class:`AsiRouterCandidate` with cheap defaults."""
    return AsiRouterCandidate(
        handle=_make_handle(session_id, original_input_dir=original_input_dir),
        window_id=int(window_id),
        ucb1_score=float(ucb1_score),
        raw_router_score=float(raw_router_score),
        island_id=int(island_id),
        visit_count=int(visit_count),
        mean_reward=float(mean_reward),
    )


def _make_candidate_pool(n: int) -> list[AsiRouterCandidate]:
    """Produce a deterministic descending-ranked candidate list of length ``n``.

    Session ids are ``"s{idx:032x}"``-shaped; scores monotonically decrease so
    the list is already in ucb1-descending order.
    """
    candidates: list[AsiRouterCandidate] = []
    for i in range(int(n)):
        sid = f"{i:032x}"
        candidates.append(
            _make_candidate(
                session_id=sid,
                window_id=i % 7,
                ucb1_score=float(1000 - i),
                raw_router_score=float(n - i) / max(1, n),
                island_id=i % 5,
                visit_count=i,
                mean_reward=float(i) / max(1, n),
            )
        )
    return candidates


def _count_tiers(
    assignments: list[TierAssignment],
) -> tuple[int, int, int]:
    """Return ``(hot, warm, cold)`` counts for a list of assignments."""
    hot = sum(1 for a in assignments if a.tier == TierLabel.HOT)
    warm = sum(1 for a in assignments if a.tier == TierLabel.WARM)
    cold = sum(1 for a in assignments if a.tier == TierLabel.COLD)
    return hot, warm, cold


def _assign_rank_tiers(
    candidates: list[AsiRouterCandidate],
    **kwargs: object,
) -> list[TierAssignment]:
    """Exercise the frozen rank-v1 contract after utility-v2 became default."""
    return assign_tiers(candidates, policy_version=POLICY_VERSION_RANK_V1, **kwargs)


# ---------------------------------------------------------------------------
# 1. TierLabel
# ---------------------------------------------------------------------------


class TestTierLabel:
    def test_tier_label_values(self) -> None:
        assert TierLabel.HOT.value == "hot"
        assert TierLabel.WARM.value == "warm"
        assert TierLabel.COLD.value == "cold"

    def test_tier_label_is_str_subclass(self) -> None:
        # Critical for JSON serialisation: isinstance(TierLabel.HOT, str) is True.
        assert isinstance(TierLabel.HOT, str)
        assert isinstance(TierLabel.WARM, str)
        assert isinstance(TierLabel.COLD, str)

    def test_tier_label_roundtrip_from_value(self) -> None:
        assert TierLabel("hot") is TierLabel.HOT
        assert TierLabel("warm") is TierLabel.WARM
        assert TierLabel("cold") is TierLabel.COLD


# ---------------------------------------------------------------------------
# 2. TierAssignment dataclass
# ---------------------------------------------------------------------------


class TestTierAssignmentDataclass:
    def test_tier_assignment_is_frozen(self) -> None:
        candidate = _make_candidate()
        ta = TierAssignment(
            candidate=candidate,
            tier=TierLabel.HOT,
            rank=0,
            policy_version=POLICY_VERSION_RANK_V1,
            policy_params={"K_HOT": 4, "K_WARM": 12, "candidate_pool": 64},
        )
        with pytest.raises(FrozenInstanceError):
            ta.rank = 5  # type: ignore[misc]

    def test_tier_assignment_required_fields(self) -> None:
        candidate = _make_candidate()
        ta = TierAssignment(
            candidate=candidate,
            tier=TierLabel.WARM,
            rank=7,
            policy_version=POLICY_VERSION_RANK_V1,
            policy_params={"K_HOT": 4, "K_WARM": 12, "candidate_pool": 64},
        )
        assert ta.candidate is candidate
        assert ta.tier == TierLabel.WARM
        assert ta.rank == 7
        assert ta.policy_version == POLICY_VERSION_RANK_V1
        assert ta.policy_params == {"K_HOT": 4, "K_WARM": 12, "candidate_pool": 64}

    def test_tier_assignment_equality(self) -> None:
        candidate = _make_candidate()
        params = {"K_HOT": 4, "K_WARM": 12, "candidate_pool": 64}
        ta_a = TierAssignment(
            candidate=candidate, tier=TierLabel.HOT, rank=0,
            policy_version=POLICY_VERSION_RANK_V1, policy_params=dict(params),
        )
        ta_b = TierAssignment(
            candidate=candidate, tier=TierLabel.HOT, rank=0,
            policy_version=POLICY_VERSION_RANK_V1, policy_params=dict(params),
        )
        assert ta_a == ta_b


# ---------------------------------------------------------------------------
# 3. assign_tiers — happy path
# ---------------------------------------------------------------------------


class TestAssignTiersBasics:
    def test_default_pool_yields_4_12_48_split(self) -> None:
        candidates = _make_candidate_pool(64)
        result = assign_tiers(candidates)
        assert len(result) == 64
        hot, warm, cold = _count_tiers(result)
        assert (hot, warm, cold) == (4, 12, 48)

    def test_ranks_are_contiguous_in_input_order(self) -> None:
        candidates = _make_candidate_pool(20)
        result = assign_tiers(candidates, candidate_pool=20)
        # Ranks must be 0..len-1 in emit order; assign_tiers does NOT re-sort.
        assert [a.rank for a in result] == list(range(20))

    def test_candidate_identity_preserved_at_each_rank(self) -> None:
        candidates = _make_candidate_pool(10)
        result = assign_tiers(candidates, candidate_pool=10)
        for rank, assignment in enumerate(result):
            assert assignment.candidate is candidates[rank]

    def test_default_policy_version_and_params(self) -> None:
        candidates = _make_candidate_pool(5)
        result = assign_tiers(candidates)
        assert result, "expected non-empty result for 5 candidates"
        assert result[0].policy_version == "utility-v2"
        assert result[0].policy_version == POLICY_VERSION_UTILITY_V2
        assert result[0].policy_params["K_HOT"] == 4
        assert result[0].policy_params["K_WARM"] == 12
        assert result[0].policy_params["candidate_pool"] == 64
        assert result[0].policy_params["budget"] == pytest.approx(5.0)
        assert result[0].policy_params["mmr_lambda"] == pytest.approx(0.75)
        assert result[0].policy_params["rrf_k"] == pytest.approx(60.0)
        # Every assignment in a single call shares identical policy_params.
        for a in result:
            assert a.policy_params == result[0].policy_params
            assert a.policy_version == result[0].policy_version

    def test_custom_hyperparameters_flow_to_policy_params(self) -> None:
        candidates = _make_candidate_pool(30)
        result = _assign_rank_tiers(
            candidates, K_HOT=2, K_WARM=5, candidate_pool=20,
        )
        assert len(result) == 20
        for a in result:
            assert a.policy_params == {
                "K_HOT": 2, "K_WARM": 5, "candidate_pool": 20,
            }
        hot, warm, cold = _count_tiers(result)
        assert hot == 2
        assert warm == 5
        assert cold == 20 - 2 - 5

    @pytest.mark.parametrize(
        "n_candidates,pool,expected_len",
        [
            (0, 64, 0),
            (5, 64, 5),
            (64, 64, 64),
            (100, 64, 64),
            (100, 10, 10),
            (1, 1, 1),
        ],
    )
    def test_result_len_matches_min_candidates_pool(
        self, n_candidates: int, pool: int, expected_len: int,
    ) -> None:
        candidates = _make_candidate_pool(n_candidates)
        result = assign_tiers(candidates, candidate_pool=pool)
        assert len(result) == expected_len


# ---------------------------------------------------------------------------
# 4. assign_tiers — edge cases (contract obligation #4)
# ---------------------------------------------------------------------------


class TestAssignTiersEdgeCases:
    def test_empty_candidate_list_returns_empty(self) -> None:
        result = assign_tiers([])
        assert result == []

    def test_single_candidate_gets_hot_only(self) -> None:
        candidates = _make_candidate_pool(1)
        result = assign_tiers(candidates)
        assert len(result) == 1
        hot, warm, cold = _count_tiers(result)
        assert (hot, warm, cold) == (1, 0, 0)
        assert result[0].tier == TierLabel.HOT

    def test_shorter_than_k_hot_all_hot(self) -> None:
        """Contract edge case: len < K_HOT -> all HOT, 0 WARM, 0 COLD."""
        candidates = _make_candidate_pool(3)
        result = _assign_rank_tiers(candidates, K_HOT=4, K_WARM=12)
        assert len(result) == 3
        hot, warm, cold = _count_tiers(result)
        assert (hot, warm, cold) == (3, 0, 0)
        for a in result:
            assert a.tier == TierLabel.HOT

    def test_exactly_k_hot_candidates(self) -> None:
        candidates = _make_candidate_pool(4)
        result = _assign_rank_tiers(candidates, K_HOT=4, K_WARM=12)
        hot, warm, cold = _count_tiers(result)
        assert (hot, warm, cold) == (4, 0, 0)

    def test_exactly_k_hot_plus_k_warm(self) -> None:
        candidates = _make_candidate_pool(16)
        result = _assign_rank_tiers(candidates, K_HOT=4, K_WARM=12)
        hot, warm, cold = _count_tiers(result)
        assert (hot, warm, cold) == (4, 12, 0)

    def test_excess_beyond_candidate_pool_truncated(self) -> None:
        candidates = _make_candidate_pool(100)
        result = assign_tiers(candidates, candidate_pool=10)
        assert len(result) == 10
        # The kept candidates are the first 10 from the input list.
        for rank, assignment in enumerate(result):
            assert assignment.candidate is candidates[rank]


# ---------------------------------------------------------------------------
# 5. assign_tiers — argument validation
# ---------------------------------------------------------------------------


class TestAssignTiersValidation:
    def test_negative_k_hot_raises(self) -> None:
        with pytest.raises(ValueError):
            assign_tiers(_make_candidate_pool(5), K_HOT=-1)

    def test_negative_k_warm_raises(self) -> None:
        with pytest.raises(ValueError):
            assign_tiers(_make_candidate_pool(5), K_WARM=-3)

    @pytest.mark.parametrize("bad_pool", [0, -1])
    def test_non_positive_candidate_pool_raises(self, bad_pool: int) -> None:
        with pytest.raises(ValueError):
            assign_tiers(_make_candidate_pool(5), candidate_pool=bad_pool)

    def test_k_hot_plus_k_warm_exceeds_pool_does_not_raise(self) -> None:
        """Graceful degrade invariant — overlapping budgets must not crash."""
        candidates = _make_candidate_pool(8)
        # K_HOT + K_WARM = 30 > candidate_pool = 8: no raise, no COLD.
        result = _assign_rank_tiers(
            candidates, K_HOT=10, K_WARM=20, candidate_pool=8,
        )
        assert len(result) == 8
        hot, warm, cold = _count_tiers(result)
        # First 8 ranks all fall inside K_HOT=10 -> all HOT, zero WARM/COLD.
        assert hot == 8
        assert warm == 0
        assert cold == 0


# ---------------------------------------------------------------------------
# 6. Property test (contract obligation #1)
# ---------------------------------------------------------------------------


class TestAssignTiersProperty:
    def test_tier_counts_match_policy_for_random_lengths(self) -> None:
        """For any (L, K_HOT, K_WARM, candidate_pool), tier counts satisfy the
        policy invariant: len_ret = min(L, pool);
        HOT = min(K_HOT, len_ret);
        WARM = min(K_WARM, max(0, len_ret - K_HOT));
        COLD = max(0, len_ret - K_HOT - K_WARM).
        """
        rng = random.Random(42)
        lengths = [0, 1, 5, 20, 64, 100]
        param_grid = [
            (4, 12, 64),
            (2, 5, 20),
            (0, 0, 10),
            (10, 0, 15),
            (0, 7, 7),
            (3, 3, 100),
        ]
        for length in lengths:
            # Shuffle order of the candidate list to defeat any accidental
            # dependence on input ordering (tiers are purely rank-based).
            candidates = _make_candidate_pool(length)
            rng.shuffle(candidates)
            for k_hot, k_warm, pool in param_grid:
                result = _assign_rank_tiers(
                    candidates,
                    K_HOT=k_hot,
                    K_WARM=k_warm,
                    candidate_pool=pool,
                )
                len_ret = min(length, pool)
                assert len(result) == len_ret
                hot, warm, cold = _count_tiers(result)
                expected_hot = min(k_hot, len_ret)
                expected_warm = min(k_warm, max(0, len_ret - k_hot))
                expected_cold = max(0, len_ret - k_hot - k_warm)
                assert hot == expected_hot, (
                    f"L={length} params=({k_hot},{k_warm},{pool}): "
                    f"hot={hot} expected {expected_hot}"
                )
                assert warm == expected_warm, (
                    f"L={length} params=({k_hot},{k_warm},{pool}): "
                    f"warm={warm} expected {expected_warm}"
                )
                assert cold == expected_cold, (
                    f"L={length} params=({k_hot},{k_warm},{pool}): "
                    f"cold={cold} expected {expected_cold}"
                )

    def test_ranks_form_contiguous_sequence(self) -> None:
        rng = random.Random(7)
        for length in [0, 1, 3, 10, 37, 64, 101]:
            candidates = _make_candidate_pool(length)
            rng.shuffle(candidates)
            for pool in [5, 16, 64]:
                result = assign_tiers(candidates, candidate_pool=pool)
                ranks = [a.rank for a in result]
                assert ranks == list(range(len(result)))


# ---------------------------------------------------------------------------
# 7. Determinism test (contract obligation #2)
# ---------------------------------------------------------------------------


class TestAssignTiersDeterminism:
    def test_same_input_twice_yields_byte_identical_json(self) -> None:
        candidates = _make_candidate_pool(32)
        result_a = assign_tiers(candidates, candidate_pool=32)
        result_b = assign_tiers(candidates, candidate_pool=32)
        assert result_a == result_b
        json_a = tier_assignments_to_json(result_a)
        json_b = tier_assignments_to_json(result_b)
        assert json_a == json_b
        assert json_a.encode("utf-8") == json_b.encode("utf-8")

    def test_freshly_constructed_candidates_yield_equal_assignments(self) -> None:
        # Build the same candidate list twice from scratch (different object
        # identities, identical values) and verify equality of assignments.
        pool_a = _make_candidate_pool(16)
        pool_b = _make_candidate_pool(16)
        # Sanity — different objects, equal values.
        assert pool_a is not pool_b
        assert pool_a == pool_b
        result_a = assign_tiers(pool_a, candidate_pool=16)
        result_b = assign_tiers(pool_b, candidate_pool=16)
        assert result_a == result_b


# ---------------------------------------------------------------------------
# 8. JSON round-trip (contract obligation #3)
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_roundtrip_preserves_all_fields_with_none_original_dir(self) -> None:
        candidate = _make_candidate(
            session_id="c" * 32,
            window_id=3,
            ucb1_score=2.5,
            raw_router_score=0.8,
            island_id=2,
            visit_count=7,
            mean_reward=0.42,
            original_input_dir=None,
        )
        ta = TierAssignment(
            candidate=candidate,
            tier=TierLabel.WARM,
            rank=6,
            policy_version=POLICY_VERSION_RANK_V1,
            policy_params={"K_HOT": 4, "K_WARM": 12, "candidate_pool": 64},
        )
        blob = tier_assignments_to_json([ta])
        reloaded = tier_assignments_from_json(blob)
        assert len(reloaded) == 1
        r = reloaded[0]
        assert r.tier == TierLabel.WARM
        assert r.rank == 6
        assert r.policy_version == POLICY_VERSION_RANK_V1
        assert r.policy_params == {"K_HOT": 4, "K_WARM": 12, "candidate_pool": 64}
        # Candidate + handle fields survive.
        assert r.candidate.window_id == 3
        assert r.candidate.ucb1_score == pytest.approx(2.5)
        assert r.candidate.raw_router_score == pytest.approx(0.8)
        assert r.candidate.island_id == 2
        assert r.candidate.visit_count == 7
        assert r.candidate.mean_reward == pytest.approx(0.42)
        assert r.candidate.handle.session_id == "c" * 32
        assert r.candidate.handle.manifest == {
            "clause_aligned": True, "num_windows": 1, "num_entries": 1,
        }
        assert r.candidate.handle.original_input_dir is None

    def test_roundtrip_preserves_non_none_original_input_dir(self) -> None:
        original = Path("/tmp/fake/original/abc")
        candidate = _make_candidate(
            session_id="d" * 32, original_input_dir=original,
        )
        ta = TierAssignment(
            candidate=candidate,
            tier=TierLabel.COLD,
            rank=40,
            policy_version=POLICY_VERSION_RANK_V1,
            policy_params={"K_HOT": 4, "K_WARM": 12, "candidate_pool": 64},
        )
        reloaded = tier_assignments_from_json(tier_assignments_to_json([ta]))
        r_handle = reloaded[0].candidate.handle
        assert r_handle.original_input_dir == original
        assert isinstance(r_handle.original_input_dir, Path)

    def test_roundtrip_paths_are_path_instances(self) -> None:
        candidate = _make_candidate(session_id="e" * 32)
        ta = TierAssignment(
            candidate=candidate,
            tier=TierLabel.HOT,
            rank=0,
            policy_version=POLICY_VERSION_RANK_V1,
            policy_params={"K_HOT": 4, "K_WARM": 12, "candidate_pool": 64},
        )
        reloaded = tier_assignments_from_json(tier_assignments_to_json([ta]))
        r_handle = reloaded[0].candidate.handle
        assert isinstance(r_handle.checkpoint_dir, Path)
        assert isinstance(r_handle.torch_store_dir, Path)
        # Values survive exactly.
        assert r_handle.checkpoint_dir == candidate.handle.checkpoint_dir
        assert r_handle.torch_store_dir == candidate.handle.torch_store_dir

    def test_tier_assignments_to_json_is_byte_deterministic(self) -> None:
        candidates = _make_candidate_pool(8)
        assignments = assign_tiers(candidates, candidate_pool=8)
        blob_a = tier_assignments_to_json(assignments)
        blob_b = tier_assignments_to_json(assignments)
        assert blob_a == blob_b
        # Compact separator — no whitespace in the envelope.
        assert ", " not in blob_a
        assert ": " not in blob_a
        # Round-trip through json to assert sort_keys=True at envelope level.
        loaded = json.loads(blob_a)
        assert list(loaded.keys()) == sorted(loaded.keys())

    def test_schema_version_mismatch_raises(self) -> None:
        envelope = {
            "schema_version": 999,
            "policy_version": POLICY_VERSION_RANK_V1,
            "assignments": [],
        }
        with pytest.raises(ValueError):
            tier_assignments_from_json(json.dumps(envelope))

    def test_unknown_tier_label_raises(self) -> None:
        candidate = _make_candidate()
        good = tier_assignment_to_dict(
            TierAssignment(
                candidate=candidate,
                tier=TierLabel.HOT,
                rank=0,
                policy_version=POLICY_VERSION_RANK_V1,
                policy_params={"K_HOT": 4, "K_WARM": 12, "candidate_pool": 64},
            )
        )
        good["tier"] = "mild"  # not a real TierLabel value
        envelope = {
            "schema_version": TIER_POLICY_SCHEMA_VERSION,
            "policy_version": POLICY_VERSION_RANK_V1,
            "assignments": [good],
        }
        with pytest.raises(ValueError):
            tier_assignments_from_json(json.dumps(envelope))

    def test_mismatched_outer_vs_assignment_policy_version_raises(self) -> None:
        candidate = _make_candidate()
        ta_dict = tier_assignment_to_dict(
            TierAssignment(
                candidate=candidate,
                tier=TierLabel.HOT,
                rank=0,
                policy_version=POLICY_VERSION_RANK_V1,  # "rank-v1"
                policy_params={"K_HOT": 4, "K_WARM": 12, "candidate_pool": 64},
            )
        )
        envelope = {
            "schema_version": TIER_POLICY_SCHEMA_VERSION,
            "policy_version": "rank-v2",  # does NOT match the assignment.
            "assignments": [ta_dict],
        }
        with pytest.raises(ValueError):
            tier_assignments_from_json(json.dumps(envelope))

    def test_tier_assignment_to_dict_roundtrip_single(self) -> None:
        """Single-assignment helper round-trip using the dict-level entry points."""
        candidate = _make_candidate(window_id=11, island_id=3, visit_count=2)
        ta = TierAssignment(
            candidate=candidate,
            tier=TierLabel.COLD,
            rank=25,
            policy_version=POLICY_VERSION_RANK_V1,
            policy_params={"K_HOT": 4, "K_WARM": 12, "candidate_pool": 64},
        )
        as_dict = tier_assignment_to_dict(ta)
        # JSON-ready dict must have string tier value.
        assert as_dict["tier"] == "cold"
        assert as_dict["rank"] == 25
        assert as_dict["policy_version"] == POLICY_VERSION_RANK_V1
        reloaded = tier_assignment_from_dict(as_dict)
        assert reloaded == ta


# ---------------------------------------------------------------------------
# 9. Re-exports (package-level access)
# ---------------------------------------------------------------------------


_PUBLIC_SYMBOLS = [
    "POLICY_VERSION_RANK_V1",
    "TIER_POLICY_SCHEMA_VERSION",
    "TierAssignment",
    "TierLabel",
    "assign_tiers",
    "tier_assignment_from_dict",
    "tier_assignment_to_dict",
    "tier_assignments_from_json",
    "tier_assignments_to_json",
]


class TestReExports:
    def test_all_symbols_importable_from_package(self) -> None:
        import chuk_lazarus.session_retrieval as pkg
        for name in _PUBLIC_SYMBOLS:
            assert hasattr(pkg, name), (
                f"chuk_lazarus.session_retrieval is missing {name!r}"
            )
            pkg_attr = getattr(pkg, name)
            submodule_attr = getattr(tier_policy_module, name)
            assert pkg_attr is submodule_attr, (
                f"{name!r} differs between package and submodule"
            )

    def test_all_symbols_listed_in_package_dunder_all(self) -> None:
        import chuk_lazarus.session_retrieval as pkg
        all_list = getattr(pkg, "__all__", None)
        assert all_list is not None, "session_retrieval package has no __all__"
        for name in _PUBLIC_SYMBOLS:
            assert name in all_list, (
                f"{name!r} not listed in chuk_lazarus.session_retrieval.__all__"
            )
