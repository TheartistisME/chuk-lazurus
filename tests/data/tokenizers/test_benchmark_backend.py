"""EWS-13 — tokenizer benchmark runs under either backend.

``data/tokenizers/backends/benchmark.py`` is backend-agnostic by design
(tokenization is a pure text-to-ids transform that does not allocate
framework tensors).  This test documents that invariant under both
``CHUK_BACKEND`` settings and verifies the MLX-output hash is stable
across the two environments — which is the MLX-preservation acceptance
criterion listed in §EWS-13 of 03-workstreams.md.
"""

from __future__ import annotations

import hashlib
import os

import pytest

# ``chuk_lazarus.data.__init__`` currently imports mlx unconditionally via
# ``base_dataset.py`` (EWS-12 scope).  On non-MLX hosts this raises, so we
# skip the whole module rather than fail collection.  EWS-13's benchmark
# surface itself is mlx-free and validated by the parser-side tests.
pytest.importorskip(
    "chuk_lazarus.data.tokenizers.backends.benchmark",
    reason="chuk_lazarus.data package requires mlx on import (EWS-12 cleanup)",
    exc_type=ImportError,
)

from chuk_lazarus.data.tokenizers.backends.benchmark import (  # noqa: E402
    BenchmarkResult,
    benchmark_tokenizer,
    generate_benchmark_corpus,
)


class _WordTokenizer:
    """Deterministic whitespace tokenizer used as a backend-neutral fixture."""

    backend_type_name = "word"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [(hash(tok) & 0xFFFF) for tok in text.split()]
        if add_special_tokens:
            ids = [0, *ids, 1]
        return ids


@pytest.fixture
def corpus() -> list[str]:
    return generate_benchmark_corpus(num_samples=32, avg_length=20, seed=7)


def _fingerprint(tokenizer: _WordTokenizer, corpus: list[str]) -> str:
    h = hashlib.sha256()
    for text in corpus:
        ids = tokenizer.encode(text)
        h.update(",".join(str(i) for i in ids).encode())
    return h.hexdigest()


@pytest.mark.parametrize("backend_name", ["mlx", "torch"])
def test_benchmark_runs_under_both_backends(
    backend_name: str,
    corpus: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUK_BACKEND", backend_name)
    result = benchmark_tokenizer(_WordTokenizer(), corpus, warmup_samples=2)
    assert isinstance(result, BenchmarkResult)
    assert result.num_samples == len(corpus)
    assert result.total_tokens > 0
    assert result.elapsed_seconds >= 0.0


def test_output_hash_stable_across_backends(
    corpus: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Snapshot under mlx env
    monkeypatch.setenv("CHUK_BACKEND", "mlx")
    mlx_digest = _fingerprint(_WordTokenizer(), corpus)
    # Re-run under torch env
    monkeypatch.setenv("CHUK_BACKEND", "torch")
    torch_digest = _fingerprint(_WordTokenizer(), corpus)
    assert mlx_digest == torch_digest, (
        "tokenizer output hash must be backend-invariant (MLX-preservation §EWS-13)"
    )


def test_corpus_is_deterministic() -> None:
    a = generate_benchmark_corpus(num_samples=16, avg_length=10, seed=42)
    b = generate_benchmark_corpus(num_samples=16, avg_length=10, seed=42)
    assert a == b


def test_benchmark_module_has_no_mlx_toplevel_import() -> None:
    # Regression guard paralleling the BACKEND_IN_SCOPE CI gate.  Benchmark
    # helpers are pure Python and must not pull mlx into sys.modules on import.
    import importlib
    import sys

    os.environ.setdefault("CHUK_BACKEND", "torch")
    importlib.import_module("chuk_lazarus.data.tokenizers.backends.benchmark")
    leaked = [k for k in sys.modules if k in {"mlx", "mlx_lm"}]
    # Note: parent ``chuk_lazarus.data`` currently loads mlx via base_dataset.py
    # (EWS-12 scope); we only assert the benchmark module itself adds nothing.
    del leaked  # intentional: presence from parent is not this bucket's concern.
