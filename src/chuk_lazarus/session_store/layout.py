"""Path conventions for the axis-3 session store.

All file-name constants are mirrored from the authoritative source modules so
that a single source-of-truth exists inside the invoke-only build pipeline:

- ``TORCH_STORE_DIR`` / ``MANIFEST_FILE`` come from
  ``chuk_lazarus.inference.context.knowledge.torch_build`` and
  ``chuk_lazarus.cli.commands.context.prefill._torch_sidecar``.
- ``BUILD_MANIFEST_FILE`` is defined as a local constant inside
  ``tools/build_clause_aligned_store.py`` at module scope and equals
  ``"clause_aligned_build_manifest.json"``. We duplicate that literal here
  (with a comment) to avoid importing the invoke-only script as a module.
- ``TORCH_PREFILL_FILE`` is the sidecar at ``<checkpoint>/torch_prefill.json``.

These constants MUST stay in sync with the build script. Any drift is a bug
that should be resolved in the build script's source modules, not here.
"""

from __future__ import annotations

from pathlib import Path

# Mirrors tools/build_clause_aligned_store.py (line ~60): BUILD_MANIFEST_FILE.
# NOTE: The brief referred to this file as ``build_manifest.json`` but the
# build script actually emits ``clause_aligned_build_manifest.json``. We match
# the script exactly.
BUILD_MANIFEST_FILE = "clause_aligned_build_manifest.json"

# Mirrors chuk_lazarus.cli.commands.context.prefill._torch_sidecar.
TORCH_STORE_DIR = "torch_store"
TORCH_PREFILL_FILE = "torch_prefill.json"

# Mirrors chuk_lazarus.inference.context.knowledge.torch_build / torch_store.
MANIFEST_FILE = "manifest.json"

# Local convention for subprocess log capture (not written by the build script;
# written by invoke_build after the subprocess completes).
INVOKE_LOG_FILE = "invoke.log"


def per_session_checkpoint(root: Path, session_id: str) -> Path:
    """Return the per-session checkpoint directory under ``root``.

    Axis-3 convention: each session gets an isolated checkpoint at
    ``<root>/<session_id>/``. Axis-4 (retrieval) iterates children of ``root``
    to enumerate available sessions.
    """
    return Path(root) / session_id


def torch_store_path(checkpoint: Path) -> Path:
    """Return the torch_store directory inside a per-session checkpoint."""
    return Path(checkpoint) / TORCH_STORE_DIR


def manifest_path(checkpoint: Path) -> Path:
    """Return the torch_store manifest (``torch_store/manifest.json``).

    This manifest is written by ``build_store`` and is distinct from the
    top-level clause-aligned build manifest at the checkpoint root.
    """
    return torch_store_path(checkpoint) / MANIFEST_FILE


def build_manifest_path(checkpoint: Path) -> Path:
    """Return the clause-aligned build manifest at the checkpoint root."""
    return Path(checkpoint) / BUILD_MANIFEST_FILE


def prefill_sidecar_path(checkpoint: Path) -> Path:
    """Return the torch-prefill sidecar at the checkpoint root."""
    return Path(checkpoint) / TORCH_PREFILL_FILE


def log_path(checkpoint: Path) -> Path:
    """Return the subprocess log path inside the checkpoint."""
    return Path(checkpoint) / INVOKE_LOG_FILE
