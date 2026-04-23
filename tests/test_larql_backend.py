"""End-to-end smoke tests for :mod:`chuk_lazarus.larql_backend`.

Runs the real `larql` binary against the example knowledge graph that
ships in ``larql/examples/gemma_4b_knowledge.json`` and against a
session graph emitted by ``emit_session_graph``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chuk_lazarus import larql_backend as lb


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_GRAPH = REPO_ROOT / "larql" / "examples" / "gemma_4b_knowledge.json"


requires_larql = pytest.mark.skipif(
    not lb.is_available(), reason="larql binary not on PATH"
)


@requires_larql
def test_version_returns_string():
    v = lb.version()
    assert v.startswith("larql")


@requires_larql
def test_validate_example_graph():
    assert EXAMPLE_GRAPH.is_file(), f"missing example: {EXAMPLE_GRAPH}"
    out = lb.validate(EXAMPLE_GRAPH)
    assert "no issues found" in out.lower()


@requires_larql
def test_stats_parses_example_graph():
    assert EXAMPLE_GRAPH.is_file()
    s = lb.stats(EXAMPLE_GRAPH)
    assert s.entities > 0
    assert s.edges > 0
    assert s.relations > 0
    assert s.avg_conf > 0.0
    assert isinstance(s.sources, dict)
    assert sum(s.sources.values()) == s.edges


@requires_larql
def test_describe_and_query_example_entity():
    assert EXAMPLE_GRAPH.is_file()
    desc = lb.describe(EXAMPLE_GRAPH, "France")
    assert "France" in desc
    assert "Paris" in desc
    q = lb.query(EXAMPLE_GRAPH, "France", "capital-of")
    assert "Paris" in q


REAL_VINDEX = REPO_ROOT / "gemma4-e2b.vindex"

requires_real_vindex = pytest.mark.skipif(
    not (REAL_VINDEX.is_dir() and (REAL_VINDEX / "index.json").is_file()),
    reason="real gemma4-e2b.vindex not present at repo root",
)


@requires_larql
@requires_real_vindex
def test_vindex_metadata_round_trip():
    meta = lb.vindex_metadata(REAL_VINDEX)
    assert meta["family"] == "gemma4"
    assert meta["model"] == "google/gemma-4-E2B-it"
    assert meta["num_layers"] == 35
    assert meta["hidden_size"] == 1536
    assert meta["extract_level"] == "all"
    assert "checksums" in meta and len(meta["checksums"]) == 7


@requires_larql
@requires_real_vindex
def test_vindex_verify_passes():
    out = lb.verify_vindex(REAL_VINDEX)
    assert "All 7 files verified" in out


@requires_larql
@requires_real_vindex
def test_walk_vindex_returns_layer_trace():
    out = lb.walk_vindex(REAL_VINDEX, "The capital of France is")
    # The summary line ("Walk: N layers, ...") goes to stderr, so we only
    # assert the per-layer stdout trace. 35 layers expected for gemma-4-E2B.
    assert "Layer 0:" in out
    assert "Layer 34:" in out
    assert out.count("Layer ") >= 35  # every layer prints a "Layer N:" header


@requires_larql
def test_emit_session_graph_round_trip(tmp_path: Path):
    edges = [
        {"s": "Banjo", "r": "is-breed", "o": "golden retriever", "c": 1.0},
        {"s": "Banjo", "r": "favorite-treat", "o": "peanut butter", "c": 0.95},
        {"s": "Banjo", "r": "adopted-from", "o": "Oregon Humane Society"},
    ]
    out_path = lb.emit_session_graph(
        edges,
        model="google/gemma-4-E2B-it",
        extraction_method="unit-test",
        output_path=tmp_path / "session.larql.json",
    )
    assert out_path.is_file()

    # Round-trip: the CLI must accept what we produced.
    assert "no issues found" in lb.validate(out_path).lower()
    s = lb.stats(out_path)
    assert s.entities == 4  # Banjo + 3 objects
    assert s.edges == 3
    assert s.relations == 3

    desc = lb.describe(out_path, "Banjo")
    assert "golden retriever" in desc
    assert "peanut butter" in desc
