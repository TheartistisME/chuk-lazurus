"""Tests for ``tools/_window_router/_encoder_sanity.py``.

These tests NEVER load a real HuggingFace model. A fake encoder class is
monkeypatched onto the module attribute ``HFSentenceEncoder`` so the
cos-sim ordering logic can be verified exactly with hand-picked vectors.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from tools._window_router import _encoder_sanity

# ── Fake encoder plumbing ────────────────────────────────────────────


class _FakeEncoder:
    """Minimal stand-in for ``HFSentenceEncoder``.

    Args:
        vectors: Mapping from prompt text to full-length vector.
        clause_feature_dim: Size of the trailing clause one-hot slice.
        dim: Total vector length (``sentence_dim + clause_feature_dim``).
    """

    def __init__(
        self,
        *,
        vectors: dict[str, list[float]],
        clause_feature_dim: int,
        dim: int,
        model_id: str = "fake/model",
        device: str = "cpu",
    ) -> None:
        self._vectors = vectors
        self._clause_feature_dim = int(clause_feature_dim)
        self._dim = int(dim)
        self.model_id = str(model_id)
        self.device = str(device)
        self.fit_called_with: list[str] | None = None

    def fit(self, texts: Iterable[str]) -> _FakeEncoder:
        self.fit_called_with = [str(t) for t in texts]
        return self

    @property
    def clause_feature_dim(self) -> int:
        return self._clause_feature_dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode_many_tensor(
        self, texts: Iterable[str], batch_size: int = 32
    ) -> Any:
        import torch  # noqa: PLC0415

        rows: list[list[float]] = []
        for text in texts:
            if text not in self._vectors:
                raise AssertionError(f"unexpected text passed to fake encoder: {text!r}")
            row = self._vectors[text]
            if len(row) != self._dim:
                raise AssertionError(
                    f"fake vector for {text!r} has length {len(row)} != dim {self._dim}"
                )
            rows.append(list(row))
        return torch.tensor(rows, dtype=torch.float32)


def _install_fake_encoder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sentence_dim: int,
    clause_feature_dim: int,
    sentence_vectors: dict[str, list[float]],
) -> list[_FakeEncoder]:
    """Register a ``HFSentenceEncoder`` replacement that records constructions."""
    constructed: list[_FakeEncoder] = []

    full_dim = sentence_dim + clause_feature_dim
    full_vectors: dict[str, list[float]] = {}
    for key, text in _encoder_sanity.REFERENCE_PROMPTS.items():
        vec = sentence_vectors[key]
        if len(vec) != sentence_dim:
            raise AssertionError(
                f"sentence vector for {key!r} has length {len(vec)} != {sentence_dim}"
            )
        full_vectors[text] = list(vec) + [0.0] * clause_feature_dim

    def _factory(
        model_id: str, device: str = "cpu", **_: object
    ) -> _FakeEncoder:
        encoder = _FakeEncoder(
            vectors=full_vectors,
            clause_feature_dim=clause_feature_dim,
            dim=full_dim,
            model_id=model_id,
            device=device,
        )
        constructed.append(encoder)
        return encoder

    monkeypatch.setattr(_encoder_sanity, "HFSentenceEncoder", _factory)
    return constructed


# ── Canonical sentence vectors ───────────────────────────────────────
# Layout (sentence_dim=3):
#   Q1 = [1, 0, 0]   T1 = [1, 0, 0]   -> cos(Q1,T1) = 1.0
#   Q2 = [0, 1, 0]   T3 = [0, 1, 0]   -> cos(Q2,T3) = 1.0
#   T2 = [0, 0, 1]                    -> orthogonal to everything else
# All required pair orderings hold (1.0 > 0.0 in each case).
_PASS_VECTORS_SENTENCE_DIM_3: dict[str, list[float]] = {
    "Q1": [1.0, 0.0, 0.0],
    "T1": [1.0, 0.0, 0.0],
    "T2": [0.0, 0.0, 1.0],
    "Q2": [0.0, 1.0, 0.0],
    "T3": [0.0, 1.0, 0.0],
}


# ── Tests ────────────────────────────────────────────────────────────


def test_reference_prompts_have_expected_keys() -> None:
    assert list(_encoder_sanity.REFERENCE_PROMPTS.keys()) == ["Q1", "T1", "T2", "Q2", "T3"]
    assert (
        _encoder_sanity.REFERENCE_PROMPTS["Q1"]
        == "What does clause 1.4.2 mean by accessible, in plain language?"
    )
    assert _encoder_sanity.REFERENCE_PROMPTS["T1"] == "Accessible"
    assert _encoder_sanity.REFERENCE_PROMPTS["T2"] == "Basic protection"
    assert (
        _encoder_sanity.REFERENCE_PROMPTS["Q2"]
        == "What mix of training, qualifications, knowledge, and experience "
        "makes someone a competent person under clause 1.4.34?"
    )
    assert _encoder_sanity.REFERENCE_PROMPTS["T3"] == "Competent person"

    assert _encoder_sanity.COS_SIM_PAIRS == (
        ("Q1", "T1", "Q1", "T2"),
        ("Q2", "T3", "Q2", "T2"),
        ("Q1", "T1", "Q2", "T1"),
        ("Q2", "T3", "Q1", "T3"),
    )


def test_compute_cosine_similarities_uses_sentence_dim_slice() -> None:
    """Cos-sim must be computed from the sentence slice only, not the clause tail."""
    pytest.importorskip("torch")

    # Sentence slice (first 3 dims) drives similarity. Clause tail (last 2
    # dims) is deliberately non-zero with patterns that would DESTROY the
    # ordering if the tail were included in the cosine calculation.
    sentence_dim = 3
    clause_feature_dim = 2
    full_dim = sentence_dim + clause_feature_dim

    # Q1 and T1 share sentence slice [1, 0, 0]. Their clause tails differ
    # wildly; the slice must still produce cos = 1.0.
    full_vectors: dict[str, list[float]] = {
        _encoder_sanity.REFERENCE_PROMPTS["Q1"]: [1.0, 0.0, 0.0, 10.0, 0.0],
        _encoder_sanity.REFERENCE_PROMPTS["T1"]: [1.0, 0.0, 0.0, 0.0, 10.0],
        _encoder_sanity.REFERENCE_PROMPTS["T2"]: [0.0, 0.0, 1.0, 5.0, 5.0],
        _encoder_sanity.REFERENCE_PROMPTS["Q2"]: [0.0, 1.0, 0.0, 7.0, 0.0],
        _encoder_sanity.REFERENCE_PROMPTS["T3"]: [0.0, 1.0, 0.0, 0.0, 7.0],
    }

    encoder = _FakeEncoder(
        vectors=full_vectors,
        clause_feature_dim=clause_feature_dim,
        dim=full_dim,
    )

    sims = _encoder_sanity.compute_cosine_similarities(encoder)

    # Sentence slice => perfect alignment on Q1/T1 and Q2/T3.
    assert sims[("Q1", "T1")] == pytest.approx(1.0)
    assert sims[("Q2", "T3")] == pytest.approx(1.0)
    # Sentence slice => orthogonal for Q1/T2 and Q2/T2.
    assert sims[("Q1", "T2")] == pytest.approx(0.0)
    assert sims[("Q2", "T2")] == pytest.approx(0.0)
    # Symmetry is preserved for both lookup orders.
    assert sims[("T1", "Q1")] == pytest.approx(1.0)


def test_run_sanity_all_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")

    constructed = _install_fake_encoder(
        monkeypatch,
        sentence_dim=3,
        clause_feature_dim=0,
        sentence_vectors=_PASS_VECTORS_SENTENCE_DIM_3,
    )

    report = _encoder_sanity.run_sanity(model_id="fake/model", device="cpu")

    assert len(constructed) == 1
    assert constructed[0].fit_called_with == list(
        _encoder_sanity.REFERENCE_PROMPTS.values()
    )
    assert report["all_passed"] is True
    assert len(report["pair_results"]) == 4
    assert all(row["pass"] for row in report["pair_results"])
    # Sanity: elapsed time is present and non-negative.
    assert report["elapsed_seconds"] >= 0.0

    # main() exits 0 when all pairs pass.
    exit_code = _encoder_sanity.main(
        ["--model-id", "fake/model", "--device", "cpu"]
    )
    assert exit_code == 0


def test_run_sanity_fail_on_bad_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    """If any ordering pair fails, ``all_passed`` is False and main() exits 1."""
    pytest.importorskip("torch")

    # Break the first ordering pair: force cos(Q1,T2) > cos(Q1,T1).
    # Q1 aligns perfectly with T2, and is orthogonal to T1.
    bad_vectors: dict[str, list[float]] = {
        "Q1": [0.0, 0.0, 1.0],
        "T1": [1.0, 0.0, 0.0],
        "T2": [0.0, 0.0, 1.0],
        "Q2": [0.0, 1.0, 0.0],
        "T3": [0.0, 1.0, 0.0],
    }

    _install_fake_encoder(
        monkeypatch,
        sentence_dim=3,
        clause_feature_dim=0,
        sentence_vectors=bad_vectors,
    )

    report = _encoder_sanity.run_sanity(model_id="fake/model", device="cpu")

    assert report["all_passed"] is False
    failed = [row for row in report["pair_results"] if not row["pass"]]
    assert failed, "at least one pair must have failed"
    # Specifically the (Q1,T1) > (Q1,T2) ordering should be the violator.
    first = report["pair_results"][0]
    assert (first["a"], first["b"], first["c"], first["d"]) == ("Q1", "T1", "Q1", "T2")
    assert first["pass"] is False

    exit_code = _encoder_sanity.main(
        ["--model-id", "fake/model", "--device", "cpu"]
    )
    assert exit_code == 1
