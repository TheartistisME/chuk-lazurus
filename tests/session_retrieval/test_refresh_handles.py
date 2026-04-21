"""Tests for :meth:`SessionRetriever.refresh_handles`.

These tests build a ``SessionRetriever`` via its dataclass fields WITHOUT
calling ``from_checkpoint_root`` (which would load a real HuggingFace model
and require CUDA). ``refresh_handles`` is a pure metadata refresh that
MUST NOT touch ``runtime``/``tokenizer``/``device``/``crystal_layer``, so
opaque sentinel objects for ``runtime`` and ``tokenizer`` are safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chuk_lazarus.session_retrieval.enumeration import iter_checkpoint_handles
from chuk_lazarus.session_retrieval.retriever import SessionRetriever


def _make_retriever(handles: list) -> SessionRetriever:
    """Build a stub retriever with opaque sentinels for runtime + tokenizer."""
    return SessionRetriever(
        handles=list(handles),
        runtime=object(),
        tokenizer=object(),
        crystal_layer=29,
        system_prompt="x",
        device="cpu",
    )


def test_refresh_handles_picks_up_new_session(
    two_sessions_checkpoint_root: dict[str, Path],
    fake_session_builder,
) -> None:
    checkpoint_root = two_sessions_checkpoint_root["checkpoint_root"]
    initial = list(iter_checkpoint_handles(checkpoint_root))
    retriever = _make_retriever(initial)

    session_c = "c" * 32
    fake_session_builder(
        checkpoint_root,
        session_c,
        windows=[
            {
                "token_ids": [40, 41, 42],
                "keywords": ["charlie2", "extra"],
                "clause_id": f"{session_c}.0.0",
                "clause_title": "Charlie extra",
            }
        ],
    )

    added = retriever.refresh_handles(checkpoint_root)
    assert added == 1
    assert len(retriever.handles) == 3
    assert session_c in {h.session_id for h in retriever.handles}


def test_refresh_handles_is_idempotent(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    checkpoint_root = two_sessions_checkpoint_root["checkpoint_root"]
    initial = list(iter_checkpoint_handles(checkpoint_root))
    retriever = _make_retriever(initial)

    first = retriever.refresh_handles(checkpoint_root)
    count_after_first = len(retriever.handles)
    second = retriever.refresh_handles(checkpoint_root)
    assert second == 0
    assert len(retriever.handles) == count_after_first


def test_refresh_handles_preserves_crystal_layer_invariant(
    two_sessions_checkpoint_root: dict[str, Path],
    fake_session_builder,
) -> None:
    checkpoint_root = two_sessions_checkpoint_root["checkpoint_root"]
    initial = list(iter_checkpoint_handles(checkpoint_root))
    retriever = _make_retriever(initial)

    session_c = "c" * 32
    session_c_dir = fake_session_builder(
        checkpoint_root,
        session_c,
        windows=[
            {
                "token_ids": [50, 51],
                "keywords": ["mismatch"],
                "clause_id": f"{session_c}.0.0",
            }
        ],
    )
    mf = session_c_dir / "torch_store" / "manifest.json"
    cfg = json.loads(mf.read_text())
    cfg["arch_config"]["crystal_layer"] = 27
    cfg["arch_config"]["injection_layer"] = 27
    mf.write_text(json.dumps(cfg))

    handles_before = list(retriever.handles)
    with pytest.raises(RuntimeError) as excinfo:
        retriever.refresh_handles(checkpoint_root)
    msg = str(excinfo.value)
    assert session_c in msg
    assert "crystal_layer" in msg
    assert retriever.handles == handles_before


def test_refresh_handles_raises_on_empty_root(
    two_sessions_checkpoint_root: dict[str, Path],
    tmp_path: Path,
) -> None:
    initial = list(iter_checkpoint_handles(two_sessions_checkpoint_root["checkpoint_root"]))
    retriever = _make_retriever(initial)
    handles_before = list(retriever.handles)

    empty_root = tmp_path / "empty_root"
    empty_root.mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        retriever.refresh_handles(empty_root)
    assert "zero checkpoints" in str(excinfo.value)
    assert retriever.handles == handles_before


def test_refresh_handles_does_not_reload_model_or_tokenizer(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    checkpoint_root = two_sessions_checkpoint_root["checkpoint_root"]
    initial = list(iter_checkpoint_handles(checkpoint_root))

    runtime_sentinel = object()
    tokenizer_sentinel = object()
    retriever = SessionRetriever(
        handles=initial,
        runtime=runtime_sentinel,
        tokenizer=tokenizer_sentinel,
        crystal_layer=29,
        system_prompt="x",
        device="cpu",
    )

    retriever.refresh_handles(checkpoint_root)
    assert retriever.runtime is runtime_sentinel
    assert retriever.tokenizer is tokenizer_sentinel
    assert retriever.crystal_layer == 29
    assert retriever.device == "cpu"


def test_refresh_handles_ordering_is_deterministic(
    tmp_path: Path,
    fake_session_builder,
) -> None:
    populated_root = tmp_path / "populated"
    populated_root.mkdir()
    for sid_char in ("a", "b", "c"):
        sid = sid_char * 32
        fake_session_builder(
            populated_root,
            sid,
            windows=[
                {
                    "token_ids": [1, 2, 3],
                    "keywords": [sid_char],
                    "clause_id": f"{sid}.0.0",
                }
            ],
        )

    retriever = _make_retriever([])
    added = retriever.refresh_handles(populated_root)
    assert added == 3
    assert [h.session_id for h in retriever.handles] == [
        "a" * 32,
        "b" * 32,
        "c" * 32,
    ]


def test_refresh_handles_with_original_input_root_passes_through(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    checkpoint_root = two_sessions_checkpoint_root["checkpoint_root"]
    original_input_root = two_sessions_checkpoint_root["original_input_root"]

    # Build initial handles WITHOUT original_input_root.
    initial = list(iter_checkpoint_handles(checkpoint_root))
    retriever = _make_retriever(initial)
    assert all(h.original_input_dir is None for h in retriever.handles)

    retriever.refresh_handles(checkpoint_root, original_input_root=original_input_root)
    for h in retriever.handles:
        expected = original_input_root / h.session_id
        if expected.is_dir():
            assert h.original_input_dir == expected
        else:
            assert h.original_input_dir is None or h.original_input_dir == expected
