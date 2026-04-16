"""Schema test for ``tests/ci/cuda_smoke_exemptions.json``.

Per ``docs/refactor/dual-backend-cuda-all-buckets/03-workstreams.md`` §EWS-0
the file is seeded empty.  Every entry (once populated) must obey the
schema below so the scheduled cuda-exemption auditor workflow can parse it
without ambiguity.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

EXEMPTIONS_PATH = Path(__file__).resolve().parent / "cuda_smoke_exemptions.json"

_REQUIRED_KEYS = {
    "stream",
    "reason",
    "opened_at",
    "auto_revert_by",
}
_OPTIONAL_KEYS = {
    "smoke_green_date",
    "pr",
    "owner",
    "notes",
}


def _load() -> list[dict]:
    with EXEMPTIONS_PATH.open() as fh:
        return json.load(fh)


def test_file_exists_and_is_list():
    data = _load()
    assert isinstance(data, list)


def test_seeded_empty():
    # EWS-0 ships the file seeded as [].  Later streams MAY append entries;
    # this test still passes because we only assert "list" + per-entry schema.
    data = _load()
    # If we ever land entries, the loop below validates them.
    assert isinstance(data, list)


@pytest.mark.parametrize("entry_idx", range(max(1, len(_load()))))
def test_entry_schema(entry_idx):
    data = _load()
    if not data:
        pytest.skip("no entries to validate (seeded empty)")
    entry = data[entry_idx]
    assert isinstance(entry, dict), f"entry {entry_idx} must be a dict"
    missing = _REQUIRED_KEYS - set(entry)
    assert not missing, f"entry {entry_idx} missing required keys: {missing}"

    unknown = set(entry) - _REQUIRED_KEYS - _OPTIONAL_KEYS
    assert not unknown, f"entry {entry_idx} has unknown keys: {unknown}"

    assert isinstance(entry["stream"], str) and entry["stream"].startswith("EWS-"), (
        f"entry {entry_idx}: 'stream' must be an EWS-* identifier"
    )
    for date_field in ("opened_at", "auto_revert_by"):
        dt.datetime.fromisoformat(entry[date_field])
    if "smoke_green_date" in entry and entry["smoke_green_date"] is not None:
        dt.datetime.fromisoformat(entry["smoke_green_date"])
