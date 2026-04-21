"""Unit tests for session_close.condense."""

from __future__ import annotations

from chuk_lazarus.session_close.condense import condense_title


def test_empty_returns_fallback() -> None:
    assert condense_title("", role="assistant", turn_index=3) == "assistant turn 3"


def test_whitespace_only_returns_fallback() -> None:
    assert condense_title("   \n\t  ", role="user", turn_index=0) == "user turn 0"


def test_short_single_sentence_no_terminator() -> None:
    # No terminator -> full stripped content
    assert condense_title("Hello world", role="user", turn_index=1) == "Hello world"


def test_short_with_period_cuts_at_period() -> None:
    assert condense_title("Hello world. Rest ignored.", role="user", turn_index=1) == "Hello world"


def test_cuts_at_question_mark() -> None:
    assert condense_title("Is this on? It should be.", role="user", turn_index=0) == "Is this on"


def test_cuts_at_exclamation() -> None:
    assert condense_title("Great! Then we proceed.", role="assistant", turn_index=2) == "Great"


def test_cuts_at_blank_line() -> None:
    text = "First paragraph body\n\nSecond paragraph body."
    assert condense_title(text, role="assistant", turn_index=0) == "First paragraph body"


def test_whitespace_collapse() -> None:
    text = "This   has    many\t\tspaces\nand newlines inside"
    assert (
        condense_title(text, role="user", turn_index=0)
        == "This has many spaces and newlines inside"
    )


def test_first_terminator_wins_across_kinds() -> None:
    # '?' appears before '.', so we cut at '?'.
    text = "Why? Because we said so."
    assert condense_title(text, role="assistant", turn_index=5) == "Why"


def test_truncation_adds_ellipsis() -> None:
    text = "a" * 500  # no terminator, stays whole then truncated
    result = condense_title(text, role="assistant", turn_index=0, max_chars=160)
    assert len(result) == 160
    assert result.endswith("...")
    assert result[:-3] == "a" * 157


def test_truncation_exact_boundary_no_ellipsis() -> None:
    text = "b" * 160
    result = condense_title(text, role="assistant", turn_index=0, max_chars=160)
    # Length is exactly max_chars -> no truncation.
    assert result == text
    assert not result.endswith("...")


def test_title_starting_with_terminator_falls_back() -> None:
    # ".Hello" -> head is "" -> collapsed empty -> fallback
    assert condense_title(".Hello", role="user", turn_index=4) == "user turn 4"


def test_fallback_uses_role_and_turn_index() -> None:
    assert condense_title("!", role="assistant", turn_index=9) == "assistant turn 9"


def test_max_chars_override() -> None:
    text = "c" * 50
    result = condense_title(text, role="user", turn_index=0, max_chars=20)
    assert len(result) == 20
    assert result.endswith("...")
