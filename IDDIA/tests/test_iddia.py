from __future__ import annotations

import math

import pytest

from IDDIA import (
    STAGES,
    build_context_query,
    chunk_markdown_text,
    embed_hash,
    next_stage,
)


def test_hash_embedding_is_deterministic_and_normalized():
    first = embed_hash("durable source log materialized index")
    second = embed_hash("durable source log materialized index")

    assert first == second
    assert len(first) == 384
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)


def test_hash_embedding_handles_empty_text():
    assert embed_hash("") == [0.0] * 384


def test_chunk_markdown_text_strips_front_matter():
    text = "---\npage: 1\n---\n\nFirst paragraph.\n\nSecond paragraph."

    chunks = chunk_markdown_text(text, target_chars=40)

    assert chunks == ["First paragraph.\n\nSecond paragraph."]


def test_stage_chain_loops():
    assert STAGES == ("onboard", "plan", "build", "verify", "handoff", "exit")
    assert next_stage("onboard") == "plan"
    assert next_stage("exit") == "onboard"


def test_context_query_includes_stage_task_and_next_steps():
    query = build_context_query("Build the index", "verify", "Run source-to-index checks")

    assert "Build the index" in query
    assert "Building stage: verify" in query
    assert "Run source-to-index checks" in query
    assert "failure modes" in query


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        next_stage("ship")
