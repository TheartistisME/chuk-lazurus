"""Axis-4 session_retrieval module.

Consumes axis-3 per-session checkpoints and answers a question by routing
across all valid clause-aligned stores, then running the Apollo-11-pattern
residual-injection pipeline.

This module NEVER modifies the underlying primitive at
``src/chuk_lazarus/inference/context/knowledge/``; it imports the already-public
``TorchKnowledgeStore``, ``torch_query._residual_is_compatible``,
``TorchInferenceRuntime``, ``LazarusBackend``, ``ResidualState``, and
``GenerationConfig`` only.

Public API
----------
- :class:`SessionRetriever` - unified retrieval entry point.
- :class:`QueryResult` - structured result with 6 strict-mode assertions.
- :class:`CheckpointHandle` - per-session checkpoint descriptor.
"""

from __future__ import annotations

from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle
from chuk_lazarus.session_retrieval.retriever import QueryResult, SessionRetriever

__all__ = [
    "CheckpointHandle",
    "QueryResult",
    "SessionRetriever",
]
