"""IDDIA: DDIA-backed context packages for agents."""

from .core import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_DDIA_URL,
    STAGES,
    build_context_package,
    build_context_query,
    chunk_markdown_text,
    embed_hash,
    ingest_ddia,
    next_stage,
)

__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_DDIA_URL",
    "STAGES",
    "build_context_package",
    "build_context_query",
    "chunk_markdown_text",
    "embed_hash",
    "ingest_ddia",
    "next_stage",
]
