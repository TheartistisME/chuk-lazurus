"""Tokenizer encode/decode round-trip parity test.

M-x1 extended-coverage deliverable (see ve-ins-0mo7014jk000083602f
axis-3 baseline: "No parity test covers ... tokenizer encode/decode
round-trip ... or loader equivalence").

This test does not exercise two different tokenizer *backends* (there is
no CUDA-specific tokenizer path in chuk_lazarus today — the axis-4
baseline notes this explicitly). Instead it pins the *input/output
invariant* that every backend's tokenizer leg must satisfy: for ASCII
text, ``decode(encode(text))`` reproduces ``text`` exactly. The residual
JSON still gets written so cross-host drift (e.g. different
``transformers`` versions) is visible in the parity log.

Skips when the default gpt2 tokenizer is not loadable (no network /
cached assets).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _helpers.mlx_snapshots import _write_residual  # noqa: E402

pytestmark = pytest.mark.parity


def _try_load_gpt2():
    try:
        from chuk_lazarus.utils.tokenizer_loader import load_tokenizer
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tokenizer_loader unavailable: {exc}")
    try:
        return load_tokenizer("gpt2")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"gpt2 tokenizer unavailable on this host: {exc}")


CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "chuk-lazarus dual-backend parity check 2026-04-20.",
    "numbers 0 1 2 3 4 5 6 7 8 9, punctuation: !?.-;",
]


def test_tokenizer_roundtrip_parity() -> None:
    tok = _try_load_gpt2()

    failures: list[tuple[str, str]] = []
    for text in CORPUS:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        if decoded.strip() != text.strip():
            failures.append((text, decoded))

    # Write a residual entry describing what we saw, even on success.
    _write_residual(
        {
            "op": "tokenizer.gpt2.roundtrip",
            "max_abs_err": 0.0 if not failures else float("inf"),
            "rtol_violation_count": len(failures),
            "shape": [len(CORPUS)],
            "dtype": "str",
            "input_hash": "",
            "atol": 0.0,
            "rtol": 0.0,
            "timestamp": "",
        }
    )

    assert not failures, f"tokenizer round-trip drift: {failures}"
