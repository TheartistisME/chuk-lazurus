"""IDDIA: DDIA-backed context packages for agents."""

from .core import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_DDIA_URL,
    STAGES,
    adjusted_retrieval_score,
    build_context_package,
    build_context_query,
    chunk_markdown_text,
    embed_hash,
    ingest_ddia,
    is_reference_like_chunk,
    next_stage,
    open_zvec_read_only_with_retry,
)

__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_DDIA_URL",
    "STAGES",
    "adjusted_retrieval_score",
    "build_context_package",
    "build_context_query",
    "chunk_markdown_text",
    "embed_hash",
    "ingest_ddia",
    "is_reference_like_chunk",
    "next_stage",
    "open_zvec_read_only_with_retry",
]
