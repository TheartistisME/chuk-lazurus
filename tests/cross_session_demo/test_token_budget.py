"""Unit test for the gemma tokenizer subprocess wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from chuk_lazarus.cross_session_demo.token_budget import (
    GEMMA_SNAPSHOT_DEFAULT,
    measure_total_tokens,
)

_SNAPSHOT = Path(GEMMA_SNAPSHOT_DEFAULT)

_SNAPSHOT_MISSING = not _SNAPSHOT.is_dir()


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="local gemma snapshot not available")
def test_measure_total_tokens_returns_positive_int() -> None:
    texts = ["the quick brown fox", "jumps over the lazy dog"]
    total = measure_total_tokens(texts)
    assert isinstance(total, int)
    assert total > 0
    # Sanity floor: eight short words + specials should always be >=2 tokens.
    assert total >= 4


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="local gemma snapshot not available")
def test_measure_total_tokens_scales_with_text_size() -> None:
    short = ["a"]
    long = ["a " * 200]
    small = measure_total_tokens(short)
    big = measure_total_tokens(long)
    assert big > small


def test_measure_total_tokens_raises_when_snapshot_missing(tmp_path: Path) -> None:
    bogus = tmp_path / "nope_not_a_snapshot"
    with pytest.raises(RuntimeError, match="Gemma tokenizer snapshot"):
        measure_total_tokens(["hello"], tokenizer_path=str(bogus))
