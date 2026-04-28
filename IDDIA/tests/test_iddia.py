from __future__ import annotations

import math

import pytest

from IDDIA import (
    STAGES,
    adjusted_retrieval_score,
    build_context_query,
    chunk_markdown_text,
    embed_hash,
    is_reference_like_chunk,
    next_stage,
    open_zvec_read_only_with_retry,
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
