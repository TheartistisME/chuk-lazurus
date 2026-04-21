"""Per-session checkpoint enumeration for axis-4 retrieval.

Iterates children of a ``checkpoint_root`` directory (axis-3 convention:
``<root>/<session_id>/torch_store/manifest.json``), validates each one against
the axis-3 handoff contract, and exposes a helper to load the underlying
``TorchKnowledgeStore``.

Primitive delegated to
----------------------
``chuk_lazarus.inference.context.knowledge.torch_store.TorchKnowledgeStore``.
This module does NOT re-implement any store internals.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chuk_lazarus.inference.context.knowledge.torch_store import TorchKnowledgeStore
from chuk_lazarus.session_store.layout import manifest_path, torch_store_path


@dataclass
class CheckpointHandle:
    """Descriptor for one validated per-session clause-aligned checkpoint.

    Attributes
    ----------
    session_id:
        The 32-char lowercase hex session id (axis-1 guarantee), equal to the
        child directory name under ``checkpoint_root``.
    checkpoint_dir:
        Path to ``<checkpoint_root>/<session_id>/``.
    torch_store_dir:
        Path to ``<checkpoint_dir>/torch_store/``.
    manifest:
        Parsed ``<torch_store_dir>/manifest.json`` (dict).
    original_input_dir:
        If provided at enumeration time, the directory from which axis-2
        emitted per-turn records for this session
        (``<original_input_root>/<session_id>/``). Used by the entity-mention
        router to pull ``topic_tags`` off individual turn records.
    """

    session_id: str
    checkpoint_dir: Path
    torch_store_dir: Path
    manifest: dict[str, Any]
    original_input_dir: Path | None


def iter_checkpoint_handles(
    checkpoint_root: Path,
    original_input_root: Path | None = None,
) -> Iterator[CheckpointHandle]:
    """Yield one :class:`CheckpointHandle` per valid per-session checkpoint.

    A child ``<checkpoint_root>/<session_id>/`` is considered enumerable iff
    ``<session_id>/torch_store/manifest.json`` exists. If the file is missing
    the child is skipped silently (the session either never built a store or
    is still in flight).

    If the manifest exists but fails the axis-3 validity gate
    (``clause_aligned is True``, ``num_windows >= 1``, ``num_entries >= 1``),
    a :class:`RuntimeError` is raised naming the session id. This matches
    ``chuk_lazarus.session_store.verify.verify_checkpoint`` behaviour.

    Parameters
    ----------
    checkpoint_root:
        Directory whose immediate children are per-session checkpoints.
    original_input_root:
        Optional directory whose children are ``<session_id>/*.json`` per-turn
        records written by axis-2 ``session_close.aus3000_emit.write_records``.
        When set, each yielded handle carries the session-specific subdir so
        the entity-mention router can read ``topic_tags``.
    """
    root = Path(checkpoint_root)
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        ts_dir = torch_store_path(child)
        mf_path = manifest_path(child)
        if not mf_path.is_file():
            # Session never completed axis-3 build, or the build is in flight.
            continue
        manifest = json.loads(mf_path.read_text())
        _enforce_validity(child.name, manifest)
        original_dir: Path | None = None
        if original_input_root is not None:
            candidate = Path(original_input_root) / child.name
            original_dir = candidate
        yield CheckpointHandle(
            session_id=child.name,
            checkpoint_dir=child,
            torch_store_dir=ts_dir,
            manifest=manifest,
            original_input_dir=original_dir,
        )


def _enforce_validity(session_id: str, manifest: dict[str, Any]) -> None:
    """Raise ``RuntimeError`` if the manifest fails the axis-3 validity gate.

    The gate is: ``clause_aligned is True``, ``num_windows >= 1``,
    ``num_entries >= 1``. This mirrors
    ``chuk_lazarus.session_store.verify.verify_checkpoint``.
    """
    clause_aligned = manifest.get("clause_aligned")
    if clause_aligned is not True:
        raise RuntimeError(
            f"session {session_id!r}: torch_store/manifest.json has "
            f"clause_aligned={clause_aligned!r}; expected True"
        )
    num_windows = int(manifest.get("num_windows", 0))
    num_entries = int(manifest.get("num_entries", 0))
    if num_windows < 1:
        raise RuntimeError(
            f"session {session_id!r}: torch_store manifest reports "
            f"num_windows={num_windows}; expected >= 1"
        )
    if num_entries < 1:
        raise RuntimeError(
            f"session {session_id!r}: torch_store manifest reports "
            f"num_entries={num_entries}; expected >= 1"
        )


def load_store(handle: CheckpointHandle) -> TorchKnowledgeStore:
    """Return a loaded :class:`TorchKnowledgeStore` for this handle.

    Delegates entirely to the primitive
    ``TorchKnowledgeStore.load(handle.torch_store_dir)``.
    """
    return TorchKnowledgeStore.load(handle.torch_store_dir)


def load_original_turn_records(handle: CheckpointHandle) -> list[dict[str, Any]]:
    """Return the list of per-turn record dicts for this session, if any.

    Reads every ``*.json`` file under ``handle.original_input_dir`` (the
    emission layout of :func:`chuk_lazarus.session_close.aus3000_emit.write_records`
    is ``<out_dir>/<session_id>/<NNN>_<ti>_<ci>.json``).

    Returns an empty list when ``handle.original_input_dir`` is ``None`` or
    the directory does not exist. Each record dict has keys ``clause_id``,
    ``clause_content``, ``iso_timestamp``, ``speaker_role``, ``session_uuid``,
    ``topic_tags`` (see axis-2 design §3).
    """
    if handle.original_input_dir is None:
        return []
    directory = handle.original_input_dir
    if not directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for json_path in sorted(directory.glob("*.json")):
        records.append(json.loads(json_path.read_text(encoding="utf-8")))
    return records


__all__ = [
    "CheckpointHandle",
    "iter_checkpoint_handles",
    "load_original_turn_records",
    "load_store",
]
