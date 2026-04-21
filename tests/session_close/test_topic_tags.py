"""Unit tests for session_close.topic_tags."""

from __future__ import annotations

from chuk_lazarus.session_close.topic_tags import STOPWORDS, extract_topic_tags


def test_empty_text_returns_empty_list() -> None:
    assert extract_topic_tags("") == []


def test_only_stopwords_returns_empty() -> None:
    assert extract_topic_tags("the a an and or but is") == []


def test_basic_top_k_default_is_five() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    # All tokens are unique, count=1 each, first-occurrence order wins.
    tags = extract_topic_tags(text)
    assert tags == ["alpha", "beta", "gamma", "delta", "epsilon"]


def test_stopwords_are_dropped() -> None:
    text = "the apple and the banana are on the table"
    tags = extract_topic_tags(text)
    assert "the" not in tags
    assert "and" not in tags
    assert "are" not in tags
    assert "apple" in tags
    assert "banana" in tags
    assert "table" in tags


def test_short_tokens_dropped() -> None:
    # "hi" and "ok" are both 2 chars -> dropped.
    text = "hi ok apple apple banana"
    tags = extract_topic_tags(text)
    assert "hi" not in tags
    assert "ok" not in tags
    assert tags[0] == "apple"  # count=2 wins


def test_pure_numeric_tokens_dropped() -> None:
    text = "report 2024 contains 1234 numbers report data"
    tags = extract_topic_tags(text)
    assert "2024" not in tags
    assert "1234" not in tags
    assert "report" in tags  # count=2
    assert "data" in tags
    assert "numbers" in tags


def test_count_ordering_beats_first_occurrence() -> None:
    text = "alpha beta beta gamma alpha beta"
    # beta count=3, alpha count=2, gamma count=1
    tags = extract_topic_tags(text)
    assert tags[0] == "beta"
    assert tags[1] == "alpha"
    assert tags[2] == "gamma"


def test_tie_break_on_first_occurrence() -> None:
    text = "apple banana cherry date elderberry"
    # All count=1; first-occurrence order: apple, banana, cherry, date, elderberry
    tags = extract_topic_tags(text)
    assert tags == ["apple", "banana", "cherry", "date", "elderberry"]


def test_top_k_override() -> None:
    text = "alpha beta gamma delta epsilon zeta"
    tags = extract_topic_tags(text, top_k=3)
    assert tags == ["alpha", "beta", "gamma"]


def test_top_k_zero_returns_empty() -> None:
    text = "alpha beta gamma"
    assert extract_topic_tags(text, top_k=0) == []


def test_lowercasing_applied() -> None:
    text = "Apple APPLE apple Banana"
    tags = extract_topic_tags(text)
    assert "apple" in tags
    assert "banana" in tags
    assert "Apple" not in tags


def test_non_alphanumeric_splitting() -> None:
    text = "hello,world;foo-bar/baz!qux"
    tags = extract_topic_tags(text, top_k=6)
    for expected in ("hello", "world", "foo", "bar", "baz", "qux"):
        assert expected in tags


def test_repeatability() -> None:
    text = "deterministic tags deterministic computation tags tags"
    a = extract_topic_tags(text)
    b = extract_topic_tags(text)
    c = extract_topic_tags(text)
    assert a == b == c


def test_stopwords_is_frozenset() -> None:
    assert isinstance(STOPWORDS, frozenset)
    assert "the" in STOPWORDS
    assert "apple" not in STOPWORDS
