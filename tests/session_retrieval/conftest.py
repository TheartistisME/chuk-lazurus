"""Fixtures for axis-4 ``session_retrieval`` tests.

Builds synthetic per-session torch_store checkpoints on a tmp path without
loading CUDA or any real HuggingFace model. The fixture reaches the real
``TorchKnowledgeStore.load`` code path so enumeration and entity-mention tests
exercise the primitive contract end-to-end; tests that need a controllable
store (exact_id, topical) instead monkeypatch ``load_store`` in their own
module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Root discovery helpers.
# ---------------------------------------------------------------------------

# tests/session_retrieval/conftest.py -> two parents up = repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
PRIMITIVE_DIR: Path = REPO_ROOT / "src" / "chuk_lazarus" / "inference" / "context" / "knowledge"
SESSION_RETRIEVAL_DIR: Path = REPO_ROOT / "src" / "chuk_lazarus" / "session_retrieval"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def primitive_dir() -> Path:
    """Absolute path to the read-only primitive package."""
    return PRIMITIVE_DIR


@pytest.fixture(scope="session")
def session_retrieval_dir() -> Path:
    """Absolute path to the axis-4 package directory."""
    return SESSION_RETRIEVAL_DIR


# ---------------------------------------------------------------------------
# Synthetic checkpoint builder.
# ---------------------------------------------------------------------------

# Minimal ArchitectureConfig.from_dict input. Mirrors Gemma 4 E2B values from
# chuk_lazarus.inference.context.knowledge.config so crystal_layer resolves to 29
# and hidden size aligns with the boundary tensor width.
_ARCH_CONFIG_GEMMA4_E2B: dict[str, Any] = {
    "retrieval_layer": 28,
    "query_head": 7,
    "injection_layer": 29,
    "hidden_dim": 1536,
    "head_dim": 256,
    "crystal_layer": 29,
    "window_size": 512,
}

HIDDEN_SIZE: int = 1536


def _write_npz_int_map(path: Path, mapping: dict[int, list[int]]) -> None:
    """Persist ``mapping`` as an .npz with int64 per-window arrays."""
    arrays = {str(wid): np.asarray(tokens, dtype=np.int64) for wid, tokens in mapping.items()}
    np.savez(str(path), **arrays)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_fake_session_checkpoint(
    checkpoint_root: Path,
    session_id: str,
    *,
    windows: list[dict[str, Any]],
    clause_aligned: bool = True,
    override_num_windows: int | None = None,
    override_num_entries: int | None = None,
) -> Path:
    """Build one synthetic per-session torch_store checkpoint on disk.

    Each entry in ``windows`` has keys ``token_ids`` (list[int]),
    ``keywords`` (list[str]), and ``clause_id`` + optional ``clause_title``
    + ``speaker_role`` for ``window_metadata.json``. Boundary tensors are
    written as float32 arrays of length ``HIDDEN_SIZE`` filled with a
    deterministic constant so loading round-trips predictably.
    """
    session_dir = checkpoint_root / session_id
    torch_store = session_dir / "torch_store"
    boundaries_dir = torch_store / "boundaries"
    boundaries_dir.mkdir(parents=True, exist_ok=True)

    token_lists: dict[int, list[int]] = {}
    keywords_map: dict[str, list[str]] = {}
    window_metadata: dict[str, dict[str, Any]] = {}

    for window_id, win in enumerate(windows):
        token_lists[window_id] = [int(t) for t in win["token_ids"]]
        keywords_map[str(window_id)] = list(win["keywords"])
        meta = {
            "clause_id": str(win["clause_id"]),
            "clause_title": str(win.get("clause_title", "")),
            "speaker_role": str(win.get("speaker_role", "user")),
        }
        window_metadata[str(window_id)] = meta
        boundary_path = boundaries_dir / f"window_{window_id:03d}.npy"
        boundary = np.full(HIDDEN_SIZE, 0.001 * (window_id + 1), dtype=np.float32)
        np.save(str(boundary_path), boundary)

    _write_npz_int_map(torch_store / "window_tokens.npz", token_lists)
    _write_npz_int_map(torch_store / "window_token_lists.npz", token_lists)

    # Build a deterministic idf.json covering every token actually used.
    all_token_ids: set[int] = set()
    for tokens in token_lists.values():
        all_token_ids.update(tokens)
    idf = {str(tid): 0.5 + 0.01 * tid for tid in sorted(all_token_ids)}
    _write_json(torch_store / "idf.json", idf)

    _write_json(torch_store / "keywords.json", keywords_map)
    _write_json(torch_store / "window_metadata.json", window_metadata)

    num_windows = override_num_windows if override_num_windows is not None else len(windows)
    num_entries = override_num_entries if override_num_entries is not None else len(windows)

    manifest = {
        "clause_aligned": clause_aligned,
        "num_windows": num_windows,
        "num_entries": num_entries,
        "num_tokens": sum(len(tl) for tl in token_lists.values()),
        "window_metadata": "window_metadata.json",
        "arch_config": dict(_ARCH_CONFIG_GEMMA4_E2B),
    }
    _write_json(torch_store / "manifest.json", manifest)
    return session_dir


def build_fake_original_input(
    original_input_root: Path,
    session_id: str,
    *,
    records: list[dict[str, Any]],
) -> Path:
    """Write one ``<root>/<session_id>/NNN_ti_ci.json`` per record.

    Records mirror the axis-2 emission shape: keys ``clause_id``,
    ``clause_content``, ``iso_timestamp``, ``speaker_role``,
    ``session_uuid``, ``topic_tags``.
    """
    session_dir = original_input_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    for idx, record in enumerate(records):
        filename = f"{idx:03d}_{record.get('turn_index', 0)}_{record.get('chunk_index', 0)}.json"
        _write_json(session_dir / filename, record)
    return session_dir


# ---------------------------------------------------------------------------
# Fixtures exposed to tests.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_session_builder() -> Any:
    """Return the :func:`build_fake_session_checkpoint` helper.

    Exposed as a fixture so individual tests can compose custom layouts
    (empty roots, invalid manifests, three-session variants).
    """
    return build_fake_session_checkpoint


@pytest.fixture
def fake_original_input_builder() -> Any:
    """Return the :func:`build_fake_original_input` helper."""
    return build_fake_original_input


@pytest.fixture
def two_sessions_checkpoint_root(tmp_path: Path) -> dict[str, Path]:
    """Two-session synthetic checkpoint root, with matching original_input_root.

    ``session_a`` ("a" * 32): 2 windows. Clause-IDs are hex-prefixed
    dotted handles of the form ``f"{session_a}.{turn}.{chunk}"`` to mirror
    the axis-2 emission contract.
      - w0 keywords ``["alice","dubai","meeting"]``, clause_id ``f"{session_a}.0.0"``.
      - w1 keywords ``["bob","flight"]``,             clause_id ``f"{session_a}.1.0"``.
    ``session_b`` ("b" * 32): 1 window.
      - w0 keywords ``["charlie","lazarus"]``,        clause_id ``f"{session_b}.0.0"``.

    Original input records retain the legacy numeric clause_ids
    (``"1.0.0"``, ``"1.0.1"``, ``"2.0.0"``) to keep the existing
    enumeration test assertions stable. Entity-mention tests do not rely
    on record-crossref because their overlap score is already 1.0 from
    keywords alone.
    """
    checkpoint_root = tmp_path / "checkpoint_root"
    original_input_root = tmp_path / "original_input_root"
    checkpoint_root.mkdir()
    original_input_root.mkdir()

    session_a = "a" * 32
    session_b = "b" * 32

    build_fake_session_checkpoint(
        checkpoint_root,
        session_a,
        windows=[
            {
                "token_ids": [10, 11, 12],
                "keywords": ["alice", "dubai", "meeting"],
                "clause_id": f"{session_a}.0.0",
                "clause_title": "Alice Dubai meeting",
            },
            {
                "token_ids": [20, 21],
                "keywords": ["bob", "flight"],
                "clause_id": f"{session_a}.1.0",
                "clause_title": "Bob flight",
            },
        ],
    )
    build_fake_session_checkpoint(
        checkpoint_root,
        session_b,
        windows=[
            {
                "token_ids": [30, 31, 32],
                "keywords": ["charlie", "lazarus"],
                "clause_id": f"{session_b}.0.0",
                "clause_title": "Charlie Lazarus",
            },
        ],
    )

    build_fake_original_input(
        original_input_root,
        session_a,
        records=[
            {
                "clause_id": "1.0.0",
                "clause_content": "Alice ran the Dubai meeting as CEO.",
                "iso_timestamp": "2026-04-21T00:00:00Z",
                "speaker_role": "user",
                "session_uuid": session_a,
                "topic_tags": ["alice", "ceo"],
                "turn_index": 0,
                "chunk_index": 0,
            },
            {
                "clause_id": "1.0.1",
                "clause_content": "Bob booked the flight.",
                "iso_timestamp": "2026-04-21T00:00:01Z",
                "speaker_role": "user",
                "session_uuid": session_a,
                "topic_tags": ["bob"],
                "turn_index": 0,
                "chunk_index": 1,
            },
        ],
    )
    build_fake_original_input(
        original_input_root,
        session_b,
        records=[
            {
                "clause_id": "2.0.0",
                "clause_content": "Charlie the engineer worked on Lazarus.",
                "iso_timestamp": "2026-04-21T00:00:00Z",
                "speaker_role": "user",
                "session_uuid": session_b,
                "topic_tags": ["charlie", "engineer"],
                "turn_index": 0,
                "chunk_index": 0,
            },
        ],
    )

    return {
        "checkpoint_root": checkpoint_root,
        "original_input_root": original_input_root,
        "session_a": session_a,
        "session_b": session_b,
    }


class _StubTokenizer:
    """Minimal tokenizer used by topical tests and retriever integration.

    Returns a deterministic fixed chat template string. ``encode`` returns a
    stable list of token ids derived from ``hash(text)`` so repeated calls
    round-trip.
    """

    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        return "STUB_PROMPT::" + "::".join(m.get("content", "") for m in messages)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Deterministic small-integer encoding.
        return [10, 11, 12] if text else []

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "::".join(str(int(t)) for t in token_ids)


@pytest.fixture
def stub_tokenizer() -> _StubTokenizer:
    """Return a deterministic tokenizer stub suitable for unit tests."""
    return _StubTokenizer()
