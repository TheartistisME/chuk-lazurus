"""Round-trip conformance: emitted JSON loads cleanly via load_clause_records.

This is the axis-2 conformance definition per design §9. The test emits a
synthetic transcript via ``emit_session`` and asserts that
``tools.build_clause_aligned_store.load_clause_records`` reads every file
and returns well-formed ``ClauseRecord`` dataclasses.

If the ``tools.build_clause_aligned_store`` import fails because of heavy
ML side-effects (torch / transformers / cuda), the test is skipped with a
clear reason rather than failing the suite.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from chuk_lazarus.chat_loop.session import ChunkBoundary, TurnRecord
from chuk_lazarus.inference.chat import Role
from chuk_lazarus.session_close.wind_down import emit_session


# Make the repo root importable so ``tools.build_clause_aligned_store`` resolves.
# From tests/session_close/test_conformance.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# The loader module pulls in torch/transformers at import time via its sibling
# imports. In environments without those available (CI slim, sandboxed unit
# testing), the import raises ImportError/ModuleNotFoundError; treat that as a
# skip rather than a hard failure (design §9 + scope-binding rule H).
_LOAD_ERR: Exception | None = None
try:
    from tools.build_clause_aligned_store import load_clause_records  # type: ignore  # noqa: E402
except (ImportError, ModuleNotFoundError, OSError) as exc:  # pragma: no cover
    load_clause_records = None  # type: ignore[assignment]
    _LOAD_ERR = exc


SID = "4f9b6d0e1a2b3c4d5e6f708192a3b4c5"
HANDLE_RE = re.compile(r"^[0-9a-f]{32}\.\d+\.\d+$")


pytestmark = pytest.mark.skipif(
    load_clause_records is None,
    reason=f"tools.build_clause_aligned_store unavailable in this env: {_LOAD_ERR!r}",
)


def _synthetic_turns() -> list[TurnRecord]:
    return [
        TurnRecord(
            turn_index=0,
            role=Role.USER,
            text="What does ISO 9001 clause 4.2 require for documented information?",
            started_at="2026-04-21T10:00:00.000000Z",
            finished_at="2026-04-21T10:00:00.500000Z",
            chunk_boundaries=[],
        ),
        TurnRecord(
            turn_index=1,
            role=Role.ASSISTANT,
            text=(
                "ISO 9001 clause 4.2 requires documented information to support "
                "the operation of the quality management system, including the "
                "necessary procedures, records, and retention policies."
            ),
            started_at="2026-04-21T10:00:01.000000Z",
            finished_at="2026-04-21T10:00:02.500000Z",
            chunk_boundaries=[
                ChunkBoundary(
                    chunk_index=0,
                    start_token_offset=0,
                    end_token_offset=512,
                    emitted_at="2026-04-21T10:00:02.000000Z",
                ),
                ChunkBoundary(
                    chunk_index=1,
                    start_token_offset=512,
                    end_token_offset=900,
                    emitted_at="2026-04-21T10:00:02.300000Z",
                ),
            ],
        ),
        TurnRecord(
            turn_index=2,
            role=Role.USER,
            text="Thank you.",
            started_at="2026-04-21T10:00:03.000000Z",
            finished_at="2026-04-21T10:00:03.100000Z",
            chunk_boundaries=[],
        ),
    ]


def test_emitted_files_round_trip_through_loader(tmp_path: Path) -> None:
    turns = _synthetic_turns()
    written = emit_session(turns, SID, tmp_path)
    assert written, "emit_session wrote no files"

    session_dir = tmp_path / SID
    records = load_clause_records(session_dir)

    # (a) non-empty
    assert records, "loader returned no records from emitted session dir"
    assert len(records) == len(written)

    for record in records:
        # (b) clause_id matches canonical pattern
        assert HANDLE_RE.match(record.clause_id), record.clause_id
        # (c) clause_title non-empty
        assert record.clause_title.strip()
        # (d) clause_content non-empty
        assert record.clause_content.strip()
        # Also verify standard_id is the session_id as mapped
        assert record.standard_id == SID
        assert record.standard_title.strip()


def test_loader_skips_metadata_suffix_but_reads_ours(tmp_path: Path) -> None:
    turns = _synthetic_turns()
    emit_session(turns, SID, tmp_path)
    session_dir = tmp_path / SID

    # Drop a sibling *_metadata.json file that the loader must ignore (line 276-277).
    (session_dir / "zzz_metadata.json").write_text("{}", encoding="utf-8")

    records = load_clause_records(session_dir)
    # Our emission count should be unaffected.
    assert len(records) == 4
    for record in records:
        assert not record.source_file.endswith("_metadata.json")
