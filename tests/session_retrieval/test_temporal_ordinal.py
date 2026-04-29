from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from chuk_lazarus.session_retrieval.asi_router import AsiRouterCandidate
from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle
from chuk_lazarus.session_retrieval.temporal_ordinal import (
    parse_ordinal_requirement,
    select_temporal_ordinal,
    sort_temporally,
)
from chuk_lazarus.session_retrieval.tier_policy import (
    POLICY_VERSION_RANK_V1,
    TierAssignment,
    TierLabel,
    tier_assignment_from_dict,
    tier_assignment_to_dict,
)
from scripts.run_mrcr_benchmark import (
    build_activation_route_index,
    extract_windows,
    resolve_token_bin,
    write_csv_report,
)


@dataclass(frozen=True)
class Candidate:
    session_id: str
    window_id: int
    session_turn_index: int


class TinyEncoder:
    def encode(self, text: str) -> list[int]:
        return list(range(len(str(text).split())))

    def decode(self, tokens: list[int]) -> str:
        return " ".join(f"tok{token}" for token in tokens)


def test_resolve_1m_token_bin_matches_mrcr_reporting_boundary() -> None:
    assert resolve_token_bin("1m") == ("1m", 524_288, 1_048_576)


def test_activation_route_memmap_uses_25_percent_overlap(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    messages = [
        {"role": "user", "content": "alpha beta gamma delta epsilon zeta"},
        {"role": "assistant", "content": "one two three four five six"},
        {"role": "user", "content": "alpha beta gamma delta epsilon zeta"},
        {"role": "assistant", "content": "seven eight nine ten eleven twelve"},
        {"role": "user", "content": "final query"},
    ]

    index, spans = build_activation_route_index(
        messages=messages,
        encoder=TinyEncoder(),
        session_id="case-a",
        activation_route_dir=tmp_path,
        np=np,
        dim=8,
        window_tokens=4,
        overlap_tokens=1,
    )
    windows = extract_windows(messages, session_id="case-a", message_spans=spans)
    routes = np.lib.format.open_memmap(index.path, mode="r")

    assert index.path.name == "activation_routes.npy"
    assert index.stride_tokens == 3
    assert index.chunk_start_tokens[:3] == (0, 3, 6)
    assert routes.shape == (index.route_count, 8)
    assert windows[0].request_token_start == 0
    assert windows[1].session_turn_index == 2


def test_csv_report_separates_construction_and_query_time(tmp_path: Path) -> None:
    report = {
        "partition": {"token_bin": "1m"},
        "comparison_baseline": {"accuracy": 0.74, "ttft_s": 30.0},
        "execution": {"mean_activation_routes": 2048.0, "mean_message_windows": 2000.0},
        "metrics": {
            "n": 1,
            "correct": 1,
            "exact_match": 1,
            "accuracy": 1.0,
            "mean_index_build_s": 60.0,
            "mean_ttft_s": 0.007,
            "p95_ttft_s": 0.007,
            "max_ttft_s": 0.007,
        },
        "cases": [
            {
                "row_index": 500,
                "token_count": 1_000_000,
                "correct": True,
                "exact_match": True,
                "index_build_s": 60.0,
                "ttft_s": 0.007,
                "activation_route_count": 2048,
                "message_window_count": 2000,
                "route_source": "literal",
                "target_rank": 1,
            }
        ],
    }
    path = tmp_path / "summary.csv"

    write_csv_report(path, report)
    text = path.read_text(encoding="utf-8")

    assert "Construction Time (s),Query Time (s)" in text.splitlines()[0]
    assert "60.0,0.007" in text


def test_parse_mrcr_digit_ordinal_wins_over_indexing_note() -> None:
    prompt = (
        "Prepend abc123XYZ0 to the 3rd (1 indexed) song about frames. "
        "Do not include any other text in your response."
    )

    assert parse_ordinal_requirement(prompt) == 3


def test_parse_word_ordinal() -> None:
    assert parse_ordinal_requirement("Return the second poem about tapirs") == 2


def test_select_temporal_ordinal_sorts_retrieved_candidates_by_turn() -> None:
    candidates = [
        Candidate("session-a", window_id=9, session_turn_index=90),
        Candidate("session-a", window_id=3, session_turn_index=30),
        Candidate("session-a", window_id=5, session_turn_index=50),
    ]

    assert select_temporal_ordinal(candidates, ordinal=2).window_id == 5
    assert [c.window_id for c in sort_temporally(candidates)] == [3, 5, 9]


def test_select_temporal_ordinal_uses_retrieved_top_k_before_sorting() -> None:
    candidates = [
        Candidate("session-a", window_id=9, session_turn_index=90),
        Candidate("session-a", window_id=3, session_turn_index=30),
        Candidate("session-a", window_id=5, session_turn_index=50),
    ]

    assert select_temporal_ordinal(candidates, ordinal=2, top_k=2).window_id == 9


def test_select_temporal_ordinal_rejects_missing_candidate() -> None:
    with pytest.raises(IndexError):
        select_temporal_ordinal([{"window_id": 1, "session_turn_index": 10}], ordinal=2)


def test_tier_assignment_roundtrip_preserves_session_turn_index() -> None:
    handle = CheckpointHandle(
        session_id="a" * 32,
        checkpoint_dir=Path("/tmp/fake/a"),
        torch_store_dir=Path("/tmp/fake/a/torch_store"),
        manifest={"clause_aligned": True, "num_windows": 1, "num_entries": 1},
        original_input_dir=None,
    )
    candidate = AsiRouterCandidate(
        handle=handle,
        window_id=7,
        ucb1_score=1.0,
        raw_router_score=0.5,
        island_id=0,
        visit_count=0,
        mean_reward=0.0,
        session_turn_index=321,
    )
    assignment = TierAssignment(
        candidate=candidate,
        tier=TierLabel.HOT,
        rank=0,
        policy_version=POLICY_VERSION_RANK_V1,
        policy_params={"K_HOT": 1, "K_WARM": 0, "candidate_pool": 1},
    )

    reloaded = tier_assignment_from_dict(tier_assignment_to_dict(assignment))

    assert reloaded.candidate.session_turn_index == 321
