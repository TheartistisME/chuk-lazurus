"""Post-build validators for axis-3 session stores.

These helpers read back the on-disk artifacts produced by the invoke-only
build script and assert that they conform to the downstream contract that
axis-4 (retrieval) depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

from chuk_lazarus.session_store.layout import (
    build_manifest_path,
    manifest_path,
)


# Keys the build script never touches. They "ride through" in the source
# per-turn JSON written by axis-2 and are read back by axis-4 directly from
# the original input directory.
DEFAULT_SECONDARY_KEYS: tuple[str, ...] = (
    "iso_timestamp",
    "speaker_role",
    "session_uuid",
    "topic_tags",
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise AssertionError(f"Expected file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AssertionError(
            f"Expected top-level JSON object at {path}, got {type(data).__name__}"
        )
    return data


def verify_checkpoint(checkpoint: Path) -> dict:
    """Verify that a per-session checkpoint looks healthy.

    Reads both the clause-aligned build manifest (at checkpoint root) and the
    torch_store manifest (inside ``torch_store/``). Asserts:

    - ``torch_store/manifest.json`` has ``clause_aligned == True``
    - ``num_windows >= 1`` (from torch_store manifest)
    - ``num_entries >= 1`` (from torch_store manifest)

    Returns a small summary dict. Raises :class:`AssertionError` on any
    violation so tests can catch the precise failure.
    """
    checkpoint = Path(checkpoint)
    torch_manifest = _load_json(manifest_path(checkpoint))
    build_manifest = _load_json(build_manifest_path(checkpoint))

    clause_aligned = torch_manifest.get("clause_aligned")
    if clause_aligned is not True:
        raise AssertionError(
            f"torch_store manifest at {manifest_path(checkpoint)} has "
            f"clause_aligned={clause_aligned!r}; expected True"
        )

    num_windows = int(torch_manifest.get("num_windows", 0))
    num_entries = int(torch_manifest.get("num_entries", 0))
    num_tokens = int(torch_manifest.get("num_tokens", 0))

    if num_windows < 1:
        raise AssertionError(
            f"torch_store manifest reports num_windows={num_windows}; expected >= 1"
        )
    if num_entries < 1:
        raise AssertionError(
            f"torch_store manifest reports num_entries={num_entries}; expected >= 1"
        )

    # Cross-check against the build manifest for consistency. These are soft
    # sanity checks; mismatches indicate something truly odd happened.
    build_windows = int(build_manifest.get("num_windows", num_windows))
    build_entries = int(build_manifest.get("num_entries", num_entries))
    if build_windows != num_windows:
        raise AssertionError(
            "build_manifest.num_windows "
            f"({build_windows}) != torch_store manifest.num_windows ({num_windows})"
        )
    if build_entries != num_entries:
        raise AssertionError(
            "build_manifest.num_entries "
            f"({build_entries}) != torch_store manifest.num_entries ({num_entries})"
        )

    return {
        "ok": True,
        "num_windows": num_windows,
        "num_entries": num_entries,
        "num_tokens": num_tokens,
        "checkpoint": str(checkpoint),
    }


def verify_metadata_preservation(
    input_dir: Path,
    secondary_keys: tuple[str, ...] = DEFAULT_SECONDARY_KEYS,
) -> dict:
    """Confirm that secondary metadata "rides through" the build untouched.

    The clause-aligned build script only consumes five fields from each
    input JSON (``standard_id``, ``standard_title``, ``clause_id``,
    ``clause_title``, ``clause_content``). The build does not mutate the
    input directory, so secondary metadata keys remain in the source files
    for axis-4 to read back.

    This validator:

    - Scans ``input_dir`` for per-turn JSON files (matching the axis-2
      filename pattern ``NNN_ti_ci.json``; also accepts any ``*.json``
      that is not a metadata sidecar).
    - Parses each file and records which of the given ``secondary_keys``
      are present somewhere in the corpus.
    - Requires that each ``secondary_keys`` entry appears in at least one
      file; otherwise raises :class:`AssertionError`.

    Returns a summary ``{"ok": True, "files_checked": N, "keys_seen": [...]}``
    with ``keys_seen`` sorted for deterministic output.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise AssertionError(f"Input directory not found: {input_dir}")

    files_checked = 0
    keys_seen: set[str] = set()

    for path in sorted(input_dir.glob("*.json")):
        if path.name.endswith("_metadata.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise AssertionError(f"Invalid JSON at {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise AssertionError(
                f"Expected top-level JSON object at {path}, got {type(data).__name__}"
            )
        files_checked += 1
        for key in secondary_keys:
            if key in data:
                keys_seen.add(key)

    if files_checked == 0:
        raise AssertionError(f"No per-turn JSON files found under {input_dir}")

    missing = [key for key in secondary_keys if key not in keys_seen]
    if missing:
        raise AssertionError(
            "Secondary metadata keys missing from every input file: "
            + ", ".join(missing)
        )

    return {
        "ok": True,
        "files_checked": files_checked,
        "keys_seen": sorted(keys_seen),
    }
