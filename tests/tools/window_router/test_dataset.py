"""Tests for ``tools/_window_router/dataset.py``."""

from __future__ import annotations

from typing import Any

from tools._window_router.augmentation import DEFAULT_TITLE_AUGMENTATION_TEMPLATES
from tools._window_router.dataset import (
    DEFAULT_PARAPHRASE_TEMPLATES,
    build_router_dataset,
)


class _StubStore:
    """Minimal duck-typed stand-in for a ``TorchKnowledgeStore``."""

    def __init__(
        self,
        window_metadata: dict[int, dict[str, Any]],
        window_token_lists: dict[int, list[int]] | None = None,
    ) -> None:
        self.window_metadata = window_metadata
        self.window_token_lists = window_token_lists or {}
        self.num_windows = len(window_metadata)


def _four_window_metadata() -> dict[int, dict[str, Any]]:
    return {
        0: {"clause_id": "1.1.1", "clause_title": "Scope"},
        1: {"clause_id": "1.2.3", "clause_title": "Definitions"},
        2: {"clause_id": "2.4.5", "clause_title": "Protection"},
        3: {"clause_id": "3.0.1", "clause_title": "Installation"},
    }


def test_at_least_one_sample_per_window_id() -> None:
    metadata = _four_window_metadata()
    store = _StubStore(metadata)

    samples = build_router_dataset(store)

    per_window: dict[int, list[str]] = {}
    for sample in samples:
        per_window.setdefault(int(sample["window_id"]), []).append(str(sample["text"]))

    for window_id in metadata:
        assert window_id in per_window, f"window {window_id} has no samples"
        assert per_window[window_id], f"window {window_id} has empty sample list"


def test_all_four_default_templates_fire_for_titled_windows() -> None:
    metadata = _four_window_metadata()
    store = _StubStore(metadata)

    samples = build_router_dataset(store)

    per_window: dict[int, set[str]] = {}
    for sample in samples:
        per_window.setdefault(int(sample["window_id"]), set()).add(str(sample["text"]))

    assert len(DEFAULT_PARAPHRASE_TEMPLATES) == 4
    for window_id, meta in metadata.items():
        title = str(meta["clause_title"])
        clause_id = str(meta["clause_id"])
        fired = 0
        for template in DEFAULT_PARAPHRASE_TEMPLATES:
            formatted = template.format(title=title, clause_id=clause_id)
            if formatted in per_window[window_id]:
                fired += 1
        assert fired >= 4, (
            f"window {window_id} fired only {fired}/4 default templates; "
            f"saw {per_window[window_id]!r}"
        )


def test_clause_id_and_title_pair_emitted() -> None:
    metadata = _four_window_metadata()
    store = _StubStore(metadata)

    samples = build_router_dataset(store)

    per_window_texts: dict[int, set[str]] = {}
    for sample in samples:
        per_window_texts.setdefault(int(sample["window_id"]), set()).add(str(sample["text"]))

    for window_id, meta in metadata.items():
        assert (
            f"{meta['clause_id']} {meta['clause_title']}" in per_window_texts[window_id]
        )


def test_default_augmentation_is_enabled() -> None:
    metadata: dict[int, dict[str, Any]] = {
        0: {"clause_id": "1.1.1", "clause_title": "Scope"},
    }
    store = _StubStore(metadata)

    samples = build_router_dataset(store)
    texts = {str(s["text"]) for s in samples if int(s["window_id"]) == 0}

    assert "Explain the clause titled Scope" in texts
    assert len(texts) == (
        2
        + len(DEFAULT_PARAPHRASE_TEMPLATES)
        + len(DEFAULT_TITLE_AUGMENTATION_TEMPLATES)
    )


def test_augment_false_preserves_legacy_shape() -> None:
    metadata: dict[int, dict[str, Any]] = {
        0: {"clause_id": "1.1.1", "clause_title": "Scope"},
    }
    store = _StubStore(metadata)

    samples = build_router_dataset(store, augment=False)
    texts = {str(s["text"]) for s in samples if int(s["window_id"]) == 0}

    assert "Explain the clause titled Scope" not in texts
    assert len(texts) == 2 + len(DEFAULT_PARAPHRASE_TEMPLATES)


def test_missing_clause_id_skips_clause_template_gracefully() -> None:
    metadata: dict[int, dict[str, Any]] = {
        0: {"clause_title": "Standalone Title"},  # no clause_id
    }
    store = _StubStore(metadata)

    samples = build_router_dataset(store)
    texts = {str(s["text"]) for s in samples if int(s["window_id"]) == 0}

    # The bare title still fires.
    assert "Standalone Title" in texts
    # Templates that require {title} still fire.
    assert "Define Standalone Title" in texts
    assert "What is Standalone Title?" in texts
    assert "Tell me about Standalone Title" in texts
    # The clause-id-dependent template is skipped rather than raising.
    assert not any("Clause :" in t for t in texts)


def test_missing_title_skips_title_templates_gracefully() -> None:
    metadata: dict[int, dict[str, Any]] = {
        0: {"clause_id": "4.4.4"},  # no clause_title
    }
    store = _StubStore(metadata)

    samples = build_router_dataset(store)
    # Every template needs a title, and there is no bare title — so no samples.
    assert [int(s["window_id"]) for s in samples if int(s["window_id"]) == 0] == []


def test_empty_window_metadata_yields_empty_list() -> None:
    store = _StubStore({})
    assert build_router_dataset(store) == []


def test_tokenizer_excerpt_samples_are_included_when_tokenizer_provided() -> None:
    metadata: dict[int, dict[str, Any]] = {
        0: {"clause_id": "1.1", "clause_title": "Alpha"},
    }
    token_lists = {0: list(range(1, 50))}
    store = _StubStore(metadata, window_token_lists=token_lists)

    class _StubTokenizer:
        def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
            return "excerpt:" + ",".join(str(i) for i in token_ids[:3])

    samples = build_router_dataset(store, tokenizer=_StubTokenizer())
    texts = {str(s["text"]) for s in samples if int(s["window_id"]) == 0}
    assert any(t.startswith("excerpt:") for t in texts)
