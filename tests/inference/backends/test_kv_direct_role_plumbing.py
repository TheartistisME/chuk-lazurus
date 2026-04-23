"""Role / turn-index plumbing coverage for KV-direct retrieval."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.backends._torch_residual_bounded import (
    gather_selected_residuals,
)
from chuk_lazarus.inference.backends.torch_runtime import WarmPenaltyConfig
from chuk_lazarus.inference.generation import GenerationConfig
from chuk_lazarus.session_retrieval import asi_router as asi_router_module
from chuk_lazarus.session_retrieval.asi_router import asi_route_candidates
from chuk_lazarus.session_retrieval.enumeration import (
    CheckpointHandle,
    iter_checkpoint_handles,
    load_store,
)
from chuk_lazarus.session_retrieval.tier_policy import TierLabel
from tests.inference.backends import test_kv_direct_materialized_real_gemma4 as _shared
from tests.session_retrieval.conftest import build_fake_session_checkpoint

_build_materialization = _shared._build_materialization
real_runtime = _shared.real_runtime
requires_cuda_gemma4 = _shared.requires_cuda_gemma4

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_VINDEX = REPO_ROOT / "gemma4-e2b.vindex"
PROMPT = "<start_of_turn>user\nHi\n<end_of_turn>\n<start_of_turn>model\n"

requires_real_vindex = pytest.mark.skipif(
    not (REAL_VINDEX.is_dir() and (REAL_VINDEX / "index.json").is_file()),
    reason="real gemma4-e2b.vindex not present at repo root",
)


def _make_tier_assignment(window_id: int, tier: TierLabel) -> SimpleNamespace:
    return SimpleNamespace(
        candidate=SimpleNamespace(window_id=int(window_id)),
        tier=tier,
        rank=0,
        policy_version="rank-v1",
        policy_params={},
    )


def _make_handle(session_id: str, tmp_path: Path) -> CheckpointHandle:
    return CheckpointHandle(
        session_id=session_id,
        checkpoint_dir=tmp_path / session_id,
        torch_store_dir=tmp_path / session_id / "torch_store",
        manifest={"clause_aligned": True, "num_windows": 3, "num_entries": 3},
        original_input_dir=None,
    )


class _StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = (text, add_special_tokens)
        return [10, 11, 12]


@requires_real_vindex
def test_gather_populates_role_and_turn_index(tmp_path: Path) -> None:
    vindex_meta = json.loads((REAL_VINDEX / "index.json").read_text(encoding="utf-8"))
    assert vindex_meta["family"] == "gemma4"
    assert vindex_meta["hidden_size"] == 1536

    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    session_id = "c" * 32
    build_fake_session_checkpoint(
        checkpoint_root,
        session_id,
        windows=[
            {
                "token_ids": [10, 11],
                "keywords": ["question", "seed"],
                "clause_id": f"{session_id}.0.0",
                "clause_title": "User seed",
                "speaker_role": "user",
            },
            {
                "token_ids": [12, 13],
                "keywords": ["answer", "seed"],
                "clause_id": f"{session_id}.1.0",
                "clause_title": "Assistant seed",
                "speaker_role": "assistant",
            },
        ],
    )

    metadata_path = checkpoint_root / session_id / "torch_store" / "window_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["0"].update({"role": "user", "turn_index": 0})
    metadata["1"].update({"role": "assistant", "turn_index": 1})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    handle = next(iter(iter_checkpoint_handles(checkpoint_root)))
    store = load_store(handle)
    result = gather_selected_residuals(
        [
            _make_tier_assignment(0, TierLabel.HOT),
            _make_tier_assignment(1, TierLabel.WARM),
        ],
        checkpoint_handle=handle,
        store=store,
        source_layer=int(store.config.injection_layer),
        device="cpu",
    )

    assert result.per_window_role == {0: "user", 1: "assistant"}
    assert result.per_window_turn_index == {0: 0, 1: 1}
    assert all(tensor.device.type == "cpu" for tensor in result.per_window_residuals.values())


def test_user_role_boost_reshuffles_ranking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    handle = _make_handle("a" * 32, tmp_path)
    score_by_window = {0: 0.5, 1: 0.4, 2: 0.0}

    class _FakeRouter:
        def __init__(self, window_tokens, idf):
            _ = (window_tokens, idf)

        def score_window(self, query_ids, window_id):
            _ = query_ids
            return score_by_window[int(window_id)]

    store = MagicMock()
    store.window_tokens = {0: {10}, 1: {11}, 2: {12}}
    store.idf = {10: 1.0, 11: 1.0, 12: 1.0}
    store.keywords = {0: [], 1: [], 2: []}
    store.num_windows = 3
    store.window_metadata = {
        0: {"role": "assistant", "turn_index": 1},
        1: {"role": "user", "turn_index": 0},
        2: {"role": "assistant", "turn_index": 2},
    }

    monkeypatch.setattr(asi_router_module, "TFIDFRouter", _FakeRouter)
    monkeypatch.setattr(asi_router_module, "load_store", lambda _handle: store)
    monkeypatch.delenv("LAZARUS_KV_USER_ROLE_BOOST", raising=False)

    baseline = asi_route_candidates(
        [handle],
        query_text="seed",
        tokenizer=_StubTokenizer(),
        candidate_pool=3,
        archive_root=None,
    )
    assert baseline[0].window_id == 0
    assert baseline[0].role == "assistant"

    monkeypatch.setenv("LAZARUS_KV_USER_ROLE_BOOST", "2.0")
    boosted = asi_route_candidates(
        [handle],
        query_text="seed",
        tokenizer=_StubTokenizer(),
        candidate_pool=3,
        archive_root=None,
    )

    assert boosted[0].window_id == 1
    assert boosted[0].role == "user"
    assert boosted[0].turn_index == 0
    assert boosted[0].raw_router_score > boosted[1].raw_router_score


@requires_cuda_gemma4
def test_metadata_surfaces_role_and_boost(
    real_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LAZARUS_KV_USER_ROLE_BOOST", "2.0")

    materialization = dataclasses.replace(
        _build_materialization(real_runtime, n_slots=2),
        per_window_role={0: "user", 1: "assistant"},
        per_window_turn_index={0: 0, 1: 1},
    )
    result = real_runtime.generate_with_kv_direct_materialization(
        prompt=PROMPT,
        config=GenerationConfig(max_new_tokens=1, temperature=0.0),
        materialization=materialization,
        per_window_token_ranges={0: (0, 1), 1: (1, 2)},
        tier_assignments={0: TierLabel.HOT, 1: TierLabel.HOT},
        warm_config=WarmPenaltyConfig(),
        source_layer=13,
    )

    assert result.metadata["per_window_role"] == {0: "user", 1: "assistant"}
    assert result.metadata["per_window_turn_index"] == {0: 0, 1: 1}
    assert result.metadata["user_role_boost"] == pytest.approx(2.0)
