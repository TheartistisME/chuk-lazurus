"""Axis-3 session_store module.

Zero-modification subprocess wrapper around ``tools/build_clause_aligned_store.py``.
This module NEVER edits the underlying build script; it only invokes it and
verifies the resulting on-disk artifacts.

Public API
----------
- :func:`invoke_build` -- run the build script for a single session.
- :func:`per_session_checkpoint` -- derive the per-session checkpoint path.
- :func:`verify_checkpoint` -- confirm that build manifests look healthy.
- :class:`SessionStoreResult` -- dataclass carrying subprocess outcome.
"""

from chuk_lazarus.session_store.invoke import SessionStoreResult, invoke_build
from chuk_lazarus.session_store.layout import per_session_checkpoint
from chuk_lazarus.session_store.verify import (
    verify_checkpoint,
    verify_metadata_preservation,
)

__all__ = [
    "SessionStoreResult",
    "invoke_build",
    "per_session_checkpoint",
    "verify_checkpoint",
    "verify_metadata_preservation",
]
