"""Agent context packaging backed by page Markdown and zvec indexes."""

from .ddia import (
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
