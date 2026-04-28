from __future__ import annotations

import json
import math
import sys
import types

import pytest

from IDDIA import (
    STAGES,
    adjusted_retrieval_score,
    build_context_package,
    build_context_query,
    chunk_markdown_text,
    classify_chunk_noise,
    embed_hash,
    infer_concept_tags,
    is_reference_like_chunk,
    next_stage,
    open_zvec_read_only_with_retry,
    search_context,
)
from IDDIA.core import (
    SearchHit,
    build_chunks,
    extract_section_labels,
    render_context_package_markdown,
)
from IDDIA.install_slash_commands import install_commands


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


def test_slash_command_installer_uses_windows_safe_namespace_dir(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "commands"
    source_dir.mkdir()
    (source_dir / "build.md").write_text("build command", encoding="utf-8")

    installed = install_commands(source_dir, target_dir)

    assert installed == [target_dir / "agent-context" / "build.md"]
    assert installed[0].read_text(encoding="utf-8") == "build command"
    assert ":" not in installed[0].name


def test_reference_like_chunks_are_downranked():
    reference_text = (
        '[41] Example Author: "A Database Paper," Proceedings of ExampleConf, '
        "volume 1, pages 1-9. doi:10.1000/example.\n"
        '[42] Another Author: "A Second Paper," IEEE Symposium.'
    )
    prose_text = (
        "Atomic commit helps a system provide simple semantics when a write may fail "
        "partway through updating durable state."
    )

    assert is_reference_like_chunk(reference_text)
    assert not is_reference_like_chunk(prose_text)
    assert adjusted_retrieval_score(0.25, reference_text) < adjusted_retrieval_score(
        0.20, prose_text
    )


def test_additional_noisy_chunks_are_detected_and_downranked():
    toc_text = (
        "Contents\n\n"
        "Chapter 1 Reliable, Scalable, and Maintainable Applications ..... 1\n"
        "Chapter 2 Data Models and Query Languages ..... 29\n"
        "Chapter 3 Storage and Retrieval ..... 69\n"
        "Chapter 4 Encoding and Evolution ..... 111\n"
    )
    front_matter = "Copyright 2017 O'Reilly Media. ISBN 978-1-449-37332-0. All rights reserved."
    opener = 'Chapter 5\n\nReplication\n\n"Many hands make light work."'
    index_text = (
        "Index\n\n"
        "atomic commit, 354, 360\n"
        "batch processing, 403, 406\n"
        "checkpoints, 458, 462\n"
        "consistency, 324, 329\n"
    )

    assert "table_of_contents" in classify_chunk_noise(toc_text, page=8)
    markitdown_toc_text = (
        "Relational vs. document databases today                                                                  38\n"
        "Query Languages for Data                                                                                              42\n"
        "Declarative queries on the web                                                                                   43\n"
        "MapReduce querying                                                                                                   45\n"
        "Graph-like Data Models                                                                                                  48\n"
        "Property graphs                                                                                                             49\n"
    )

    assert "table_of_contents" in classify_chunk_noise(markitdown_toc_text, page=8)
    assert "front_matter" in classify_chunk_noise(front_matter, page=3)
    assert "chapter_opener" in classify_chunk_noise(opener, page=150)
    assert "index_like" in classify_chunk_noise(index_text, page=590)
    assert adjusted_retrieval_score(0.5, toc_text, page=8) < 0.5


def test_concept_tags_cover_iddia_retrieval_terms():
    tags = infer_concept_tags(
        "Replay an event log from a source of truth into a materialized view "
        "using deterministic checkpoints and schema manifests."
    )

    assert "event-log" in tags
    assert "materialized-view" in tags
    assert "source-of-truth" in tags
    assert "checkpoint" in tags
    assert "schema" in tags
    assert "manifest" in tags


def test_build_chunks_adds_best_effort_chapter_and_section_labels(tmp_path):
    markdown_dir = tmp_path / "pages"
    chunks_path = tmp_path / "chunks.jsonl"
    markdown_dir.mkdir()
    (markdown_dir / "page_0001.md").write_text(
        "---\npage: 1\n---\n\n"
        "Chapter 3\n\nStorage and Retrieval\n\n"
        "## Hash Indexes\n\n"
        "A log-structured storage engine can rebuild an in-memory hash index.",
        encoding="utf-8",
    )

    assert build_chunks(markdown_dir, chunks_path, force=True) == 1
    record = json.loads(chunks_path.read_text(encoding="utf-8"))

    assert record["chapter_title"] == "Chapter 3: Storage and Retrieval"
    assert record["section_title"] == "Hash Indexes"
    assert "event-log" in record["concept_tags"]


def test_section_label_heuristic_ignores_numbered_sentence_fragments():
    labels = extract_section_labels(
        "3 of the application state), then a straightforward single-threaded log consumer needs\n"
        "no concurrency control for writes."
    )

    assert labels == {}


def test_zvec_open_uses_read_only_option_and_retries_lock(tmp_path):
    calls: list[object] = []

    class FakeOption:
        def __init__(self, *, read_only: bool):
            self.read_only = read_only

    class FakeZvec:
        CollectionOption = FakeOption

        @staticmethod
        def open(path: str, option: FakeOption):
            calls.append((path, option.read_only))
            if len(calls) == 1:
                raise RuntimeError("Can't lock read-write collection")
            return "collection"

    collection = open_zvec_read_only_with_retry(
        FakeZvec,
        tmp_path / "vectors",
        attempts=2,
        initial_delay_seconds=0,
        sleep=lambda _: None,
    )

    assert collection == "collection"
    assert calls == [
        (str(tmp_path / "vectors"), True),
        (str(tmp_path / "vectors"), True),
    ]


def install_fake_zvec(monkeypatch, results):
    class FakeOption:
        def __init__(self, *, read_only: bool):
            self.read_only = read_only

    class FakeVectorQuery:
        def __init__(self, name: str, *, vector):
            self.name = name
            self.vector = vector

    class FakeCollection:
        def query(self, _query, *, topk: int):
            return results[:topk]

    fake = types.SimpleNamespace(
        CollectionOption=FakeOption,
        VectorQuery=FakeVectorQuery,
        open=lambda _path, _option: FakeCollection(),
    )
    monkeypatch.setitem(sys.modules, "zvec", fake)


def write_chunks(tmp_path, records):
    artifact_root = tmp_path / "artifacts"
    chunks_path = artifact_root / "chunks" / "chunks.jsonl"
    vector_path = artifact_root / "vectors" / "zvec"
    chunks_path.parent.mkdir(parents=True)
    vector_path.mkdir(parents=True)
    chunks_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return artifact_root


def test_search_reranks_with_lexical_and_tag_boosts(monkeypatch, tmp_path):
    artifact_root = write_chunks(
        tmp_path,
        [
            {
                "id": "generic",
                "page": 40,
                "source_path": "page_0040.md",
                "principle_tags": [],
                "stage_tags": [],
                "text": "Systems should be reliable and scalable for users.",
            },
            {
                "id": "event-log",
                "page": 41,
                "source_path": "page_0041.md",
                "principle_tags": ["source-of-truth"],
                "stage_tags": [],
                "concept_tags": ["event-log", "replay", "checkpoint", "manifest"],
                "section_title": "Event logs and replay",
                "text": (
                    "A durable event log can replay deterministic updates from a "
                    "manifest and checkpoint into a materialized projection."
                ),
            },
        ],
    )
    install_fake_zvec(
        monkeypatch,
        [{"id": "generic", "score": 0.4}, {"id": "event-log", "score": 0.33}],
    )

    hits = search_context(
        artifact_root=artifact_root,
        task="Replay an event log using checkpoints and a manifest",
        stage="build",
        next_steps="Create deterministic projection rebuilds",
        top_k=2,
    )

    assert [hit.chunk_id for hit in hits] == ["event-log", "generic"]
    assert hits[0].lexical_score > 0
    assert hits[0].query_boost > 0
    assert "event-log" in hits[0].matched_tags
    assert "replay" in hits[0].matched_terms


def test_search_uses_lexical_fallback_beyond_zvec_candidates(monkeypatch, tmp_path):
    artifact_root = write_chunks(
        tmp_path,
        [
            {
                "id": "generic",
                "page": 40,
                "source_path": "page_0040.md",
                "principle_tags": [],
                "stage_tags": [],
                "text": "A general system can be reliable.",
            },
            {
                "id": "event-sourcing",
                "page": 465,
                "source_path": "page_0465.md",
                "principle_tags": ["source-of-truth", "durability"],
                "stage_tags": [],
                "concept_tags": ["event-log", "replay", "snapshot"],
                "chapter_title": "Chapter 11: Stream Processing",
                "text": (
                    "Event sourcing records every change as an append-only log, "
                    "then uses snapshots to recover and replay derived state."
                ),
            },
        ],
    )
    install_fake_zvec(monkeypatch, [{"id": "generic", "score": 0.3}])

    hits = search_context(
        artifact_root=artifact_root,
        task="Verify experiment lineage by replaying failed rounds from an event log",
        stage="verify",
        next_steps="Use snapshots for recovery and explain parent selection",
        top_k=1,
    )

    assert hits[0].chunk_id == "event-sourcing"
    assert "affinity:lineage-log" in hits[0].matched_tags


def test_schema_migration_affinity_prefers_encoding_chapter(monkeypatch, tmp_path):
    artifact_root = write_chunks(
        tmp_path,
        [
            {
                "id": "transaction-version",
                "page": 255,
                "source_path": "page_0255.md",
                "principle_tags": ["consistency", "durability", "schema-evolution"],
                "stage_tags": [],
                "concept_tags": ["schema", "provenance-lineage"],
                "chapter_title": "Chapter 7: Transactions",
                "text": "A transaction record has an immutable version and durable state.",
            },
            {
                "id": "encoding-schema",
                "page": 142,
                "source_path": "page_0142.md",
                "principle_tags": ["schema-evolution"],
                "stage_tags": [],
                "concept_tags": ["schema"],
                "chapter_title": "Chapter 4: Encoding and Evolution",
                "text": (
                    "Schema migration needs backward compatibility, forward "
                    "compatibility, encoding versions, and old readers."
                ),
            },
        ],
    )
    install_fake_zvec(
        monkeypatch,
        [
            {"id": "transaction-version", "score": 0.34},
            {"id": "encoding-schema", "score": 0.27},
        ],
    )

    hits = search_context(
        artifact_root=artifact_root,
        task="Migrate extracted JSON triples from one schema version to another",
        stage="plan",
        next_steps="Design compatibility checks and migration manifests",
        top_k=2,
    )

    assert hits[0].chunk_id == "encoding-schema"
    assert "affinity:encoding-evolution" in hits[0].matched_tags


def test_search_handles_old_chunks_without_new_metadata(monkeypatch, tmp_path):
    artifact_root = write_chunks(
        tmp_path,
        [
            {
                "id": "old",
                "page": 70,
                "source_path": "page_0070.md",
                "text": "Schema evolution needs compatibility across old and new encodings.",
            }
        ],
    )
    install_fake_zvec(monkeypatch, [{"id": "old", "score": 0.2}])

    hits = search_context(
        artifact_root=artifact_root,
        task="Plan schema compatibility",
        stage="plan",
        next_steps="Check encoding migrations",
        top_k=1,
    )

    assert hits[0].chunk_id == "old"
    assert hits[0].concept_tags == ("schema",)
    assert hits[0].chapter_title == ""
    assert hits[0].why_this_hit


def test_why_this_hit_renders_in_markdown_and_json(monkeypatch, tmp_path):
    hit = SearchHit(
        chunk_id="ddia-p0041-c00",
        score=0.55,
        page=41,
        source_path="page_0041.md",
        principle_tags=("source-of-truth",),
        stage_tags=("build",),
        text="Replay the event log into a projection.",
        concept_tags=("event-log", "materialized-view"),
        section_title="Event logs",
        vector_score=0.3,
        lexical_score=0.18,
        query_boost=0.07,
        matched_terms=("replay", "event log"),
        matched_tags=("event-log",),
        why_this_hit={"summary": "matched terms: replay, event log"},
    )

    markdown = render_context_package_markdown(
        task="Replay event log",
        stage="build",
        next_steps="Create projection",
        hits=[hit],
        top_k=1,
        max_snippet_chars=200,
    )

    assert "Why this hit: matched terms: replay, event log" in markdown
    assert "Location: Event logs" in markdown

    monkeypatch.setattr("IDDIA.core.search_context", lambda **_kwargs: [hit])
    payload = json.loads(
        build_context_package(
            task="Replay event log",
            stage="build",
            next_steps="Create projection",
            artifact_root=tmp_path,
            top_k=1,
            output_format="json",
        )
    )

    assert payload["hits"][0]["why_this_hit"]["summary"] == "matched terms: replay, event log"
    assert payload["hits"][0]["section_title"] == "Event logs"
