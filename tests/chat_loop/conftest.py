"""Shared fixtures for chat_loop unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class FakeTokenizer:
    """Deterministic whitespace tokenizer for unit tests.

    Behaves like a HF tokenizer for the narrow surface StreamingWindower
    uses: callable returning an object with ``.input_ids`` and a
    ``decode()`` method. Tokenization is whitespace-split; each distinct
    token is assigned a stable integer id and the reverse mapping is
    cached so ``decode`` round-trips.
    """

    def __init__(self) -> None:
        self._vocab: dict[int, str] = {}
        self._rvocab: dict[str, int] = {}
        self._next_id: int = 1

    def _id_for(self, token: str) -> int:
        existing = self._rvocab.get(token)
        if existing is not None:
            return existing
        new_id = self._next_id
        self._next_id += 1
        self._vocab[new_id] = token
        self._rvocab[token] = new_id
        return new_id

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ) -> SimpleNamespace:
        tokens = text.split()
        ids = [self._id_for(tok) for tok in tokens]
        return SimpleNamespace(input_ids=ids)

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return " ".join(self._vocab.get(i, "") for i in ids)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    """Return a fresh FakeTokenizer per test."""
    return FakeTokenizer()


def make_text(n_tokens: int, prefix: str = "t") -> str:
    """Produce whitespace-delimited text with exactly n_tokens tokens."""
    return " ".join(f"{prefix}{i}" for i in range(n_tokens))
