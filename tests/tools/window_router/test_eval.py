"""Tests for ``tools/_window_router/eval.py``."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tools._window_router.eval import (
    evaluate,
    resolve_expected_window_id,
    score_predictions,
    write_report,
)


class _StubStore:
    """Minimal ``TorchKnowledgeStore`` stand-in for evaluation tests."""

    def __init__(
        self,
        window_metadata: dict[int, dict[str, Any]],
        baseline_ranker: Any | None = None,
    ) -> None:
        self.window_metadata = window_metadata
        self._baseline = baseline_ranker

    def route_top_k(
        self,
        query_text: str,
        tokenizer: Any,
        k: int = 3,
        expansion_ids: list[int] | None = None,
    ) -> list[int]:
        if self._baseline is None:
            return []
        return list(self._baseline(query_text, k))


def _three_window_store() -> _StubStore:
    metadata = {
        0: {"clause_id": "A.1", "clause_title": "Alpha"},
        1: {"clause_id": "B.2", "clause_title": "Bravo"},
        2: {"clause_id": "C.3", "clause_title": "Charlie"},
    }

    def baseline(prompt: str, k: int) -> list[int]:
        order = [0, 1, 2]
        for wid, meta in metadata.items():
            if str(meta["clause_title"]).lower() in prompt.lower():
                order.remove(wid)
                order.insert(0, wid)
                break
        return order[:k]

    return _StubStore(metadata, baseline_ranker=baseline)


def test_resolve_expected_window_id_finds_match() -> None:
    store = _three_window_store()
    assert resolve_expected_window_id(store, "A.1") == 0
    assert resolve_expected_window_id(store, "B.2") == 1
    assert resolve_expected_window_id(store, "C.3") == 2


def test_resolve_expected_window_id_none_for_missing() -> None:
    store = _three_window_store()
    assert resolve_expected_window_id(store, "Z.9") is None
    assert resolve_expected_window_id(store, "") is None


def test_resolve_expected_window_id_prefers_lowest_part_index() -> None:
    store = _StubStore(
        {
            5: {"clause_id": "X.1", "part_index": 2},
            6: {"clause_id": "X.1", "part_index": 1},
        }
    )
    assert resolve_expected_window_id(store, "X.1") == 6


def test_score_predictions_computes_top1_top3_and_mrr() -> None:
    # Case 1: expected=0, preds=[0,1,2] -> top1 hit, rank 1
    # Case 2: expected=2, preds=[1,2,0] -> top1 miss, top3 hit, rank 2
    # Case 3: expected=0, preds=[2,1,0] -> top1 miss, top3 hit, rank 3
    preds = [[0, 1, 2], [1, 2, 0], [2, 1, 0]]
    expected = [0, 2, 0]
    metrics = score_predictions(preds, expected)
    assert metrics["n"] == 3
    assert metrics["top_1_accuracy"] == pytest.approx(1 / 3)
    assert metrics["top_3_accuracy"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx((1.0 + 0.5 + 1 / 3) / 3)


def test_score_predictions_empty() -> None:
    metrics = score_predictions([], [])
    assert metrics == {"top_1_accuracy": 0.0, "top_3_accuracy": 0.0, "mrr": 0.0, "n": 0}


class _FakeTopKModel:
    """Callable that emits one-hot logits picking a preset label per prompt hash."""

    def __init__(self, prompt_to_label: dict[str, int], num_labels: int) -> None:
        self._mapping = prompt_to_label
        self._num_labels = int(num_labels)
        self._current_prompt: str = ""

    def arm(self, prompt: str) -> None:
        self._current_prompt = prompt

    def __call__(self, features_tensor: torch.Tensor) -> torch.Tensor:
        logits = torch.full((features_tensor.shape[0], self._num_labels), -5.0)
        label = self._mapping.get(self._current_prompt, 0)
        logits[0, label] = 10.0
        return logits

    def eval(self) -> _FakeTopKModel:  # pragma: no cover - interface shim only
        return self


class _IdentityEncoder:
    def __init__(self, dim: int) -> None:
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> list[float]:
        # Pure-python deterministic feature vector; only used to feed shape info.
        vec = [0.0] * self._dim
        if text:
            vec[0] = 1.0
        return vec


def test_evaluate_synthetic_three_case_benchmark() -> None:
    store = _three_window_store()
    cases = [
        {"name": "a", "prompt": "What is Alpha?", "primary_clause_ids": ["A.1"]},
        {"name": "b", "prompt": "What is Bravo?", "primary_clause_ids": ["B.2"]},
        {"name": "c", "prompt": "What is Charlie?", "primary_clause_ids": ["C.3"]},
    ]

    mapping = {
        "What is Alpha?": 0,
        "What is Bravo?": 1,
        "What is Charlie?": 2,
    }
    model = _FakeTopKModel(mapping, num_labels=3)

    # Wrap the model call to "arm" it per case before `evaluate` invokes it.
    def model_callable(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    # Use a thin adapter so evaluate() sees a single callable but model picks
    # per-prompt by tracking the last encode() input through a closure.
    encoder = _IdentityEncoder(dim=4)
    original_encode = encoder.encode

    def encode_and_arm(text: str) -> list[float]:
        model.arm(text)
        return original_encode(text)

    encoder.encode = encode_and_arm  # type: ignore[assignment]

    # Provide a dummy tokenizer so the TF-IDF baseline path runs.
    class _DummyTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [ord(c) for c in text[:10]]

    report = evaluate(
        model=model_callable,
        encoder=encoder,
        store=store,
        cases=cases,
        tokenizer=_DummyTokenizer(),
    )

    assert report["model"]["n"] == 3
    assert report["model"]["top_1_accuracy"] == pytest.approx(1.0)
    assert report["model"]["mrr"] == pytest.approx(1.0)
    # The baseline ranker places the matching window first when the title appears
    # in the prompt.
    assert report["tfidf_baseline"]["top_1_accuracy"] == pytest.approx(1.0)
    assert len(report["cases"]) == 3


def test_evaluate_without_tokenizer_keeps_exact_match_baseline() -> None:
    store = _three_window_store()
    cases = [
        {"name": "a", "prompt": "What is Alpha?", "primary_clause_ids": ["A.1"]},
        {"name": "b", "prompt": "What is Bravo?", "primary_clause_ids": ["B.2"]},
    ]

    mapping = {
        "What is Alpha?": 0,
        "What is Bravo?": 1,
    }
    model = _FakeTopKModel(mapping, num_labels=3)
    encoder = _IdentityEncoder(dim=4)
    original_encode = encoder.encode

    def encode_and_arm(text: str) -> list[float]:
        model.arm(text)
        return original_encode(text)

    encoder.encode = encode_and_arm  # type: ignore[assignment]

    report = evaluate(
        model=lambda x: model(x),
        encoder=encoder,
        store=store,
        cases=cases,
        tokenizer=None,
    )

    assert report["tfidf_baseline"]["n"] == 2
    assert report["tfidf_baseline"]["top_1_accuracy"] == pytest.approx(1.0)
    assert report["cases"][0]["baseline_top3"] == [0, 1, 2]
    assert report["cases"][1]["baseline_top3"] == [1, 0, 2]


def test_evaluate_skips_cases_without_primary_clause() -> None:
    store = _three_window_store()
    cases = [
        {"name": "no-primary", "prompt": "question", "primary_clause_ids": []},
        {"name": "unknown", "prompt": "x", "primary_clause_ids": ["Z.Z"]},
    ]
    model = _FakeTopKModel({}, num_labels=3)
    report = evaluate(
        model=lambda x: model(x),
        encoder=_IdentityEncoder(dim=2),
        store=store,
        cases=cases,
        tokenizer=None,
    )
    assert report["model"]["n"] == 0
    assert len(report["skipped"]) == 2


def test_write_report_emits_json_and_markdown(tmp_path: Any) -> None:
    report = {
        "model": {"top_1_accuracy": 0.5, "top_3_accuracy": 0.8, "mrr": 0.65, "n": 4},
        "tfidf_baseline": {"top_1_accuracy": 0.25, "top_3_accuracy": 0.5, "mrr": 0.4, "n": 4},
        "cases": [],
        "skipped": [{"name": "x", "reason": "none"}],
        "k": 3,
    }
    out_dir = tmp_path / "rep"
    write_report(out_dir, report)
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()
    md_text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Model (MLP)" in md_text
    assert "TF-IDF Baseline" in md_text
