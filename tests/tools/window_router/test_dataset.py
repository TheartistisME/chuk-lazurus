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

    for window_id, meta in metadata.items():
        title = str(meta["clause_title"])
        clause_id = str(meta["clause_id"])
        fired = 0
        for template in DEFAULT_PARAPHRASE_TEMPLATES:
            formatted = template.format(title=title, clause_id=clause_id)
            if formatted in per_window[window_id]:
                fired += 1
        assert fired >= len(DEFAULT_PARAPHRASE_TEMPLATES), (
            f"window {window_id} fired only {fired}/{len(DEFAULT_PARAPHRASE_TEMPLATES)} "
            f"default templates; "
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


def test_collision_titles_do_not_produce_contradictory_labels() -> None:
    # Two windows share the same bare title but have distinct clause_ids.
    # The emitter must never produce the same ``text`` for both window_ids,
    # otherwise cross-entropy cannot converge.
    metadata: dict[int, dict[str, Any]] = {
        0: {"clause_id": "1.2.3", "clause_title": "Definitions"},
        1: {"clause_id": "4.5.6", "clause_title": "Definitions"},
        2: {"clause_id": "7.8.9", "clause_title": "Unique Title"},
    }
    store = _StubStore(metadata)

    samples = build_router_dataset(store)

    # (a) no text string is ever labeled with more than one window_id.
    text_to_window_ids: dict[str, set[int]] = {}
    for sample in samples:
        text_to_window_ids.setdefault(
            str(sample["text"]), set()
        ).add(int(sample["window_id"]))
    duplicates = {
        text: window_ids
        for text, window_ids in text_to_window_ids.items()
        if len(window_ids) > 1
    }
    assert duplicates == {}, (
        f"colliding text -> window_ids pairs leaked into dataset: {duplicates!r}"
    )

    # (b) collision-title windows still emit disambiguated samples.
    per_window_texts: dict[int, set[str]] = {}
    for sample in samples:
        per_window_texts.setdefault(
            int(sample["window_id"]), set()
        ).add(str(sample["text"]))
    assert per_window_texts[0], "collision window 0 produced no samples"
    assert per_window_texts[1], "collision window 1 produced no samples"
    # Every emitted text for a collision window must contain the clause_id.
    for text in per_window_texts[0]:
        assert "1.2.3" in text, (
            f"collision window 0 emitted non-disambiguated text: {text!r}"
        )
    for text in per_window_texts[1]:
        assert "4.5.6" in text, (
            f"collision window 1 emitted non-disambiguated text: {text!r}"
        )

    # The plain bare title is never emitted for either collision window.
    assert "Definitions" not in per_window_texts[0]
    assert "Definitions" not in per_window_texts[1]

    # (c) non-collision windows still get the bare title and plain templates.
    assert "Unique Title" in per_window_texts[2]
    assert "Define Unique Title" in per_window_texts[2]


def test_collision_title_without_clause_id_emits_no_samples() -> None:
    # Two windows share a title and neither has a clause_id to disambiguate.
    # Safe fallback: emit nothing for those windows rather than poison labels.
    metadata: dict[int, dict[str, Any]] = {
        0: {"clause_title": "Shared"},
        1: {"clause_title": "Shared"},
        2: {"clause_id": "9.9.9", "clause_title": "Fine"},
    }
    store = _StubStore(metadata)

    samples = build_router_dataset(store)

    per_window_texts: dict[int, set[str]] = {}
    for sample in samples:
        per_window_texts.setdefault(
            int(sample["window_id"]), set()
        ).add(str(sample["text"]))

    assert per_window_texts.get(0, set()) == set()
    assert per_window_texts.get(1, set()) == set()
    assert "Fine" in per_window_texts[2]


def test_multi_clause_benchmark_cases_emit_prompt_and_pairwise_rows() -> None:
    # Benchmark-derived multi-target prompts must be emitted to exactly ONE
    # window (the primary — i.e. the window resolved from
    # ``primary_clause_ids[0]``). Emitting the same prompt to every
    # participating window would collide (text, window_id) and poison
    # cross-entropy training in the single-label router schema.
    metadata: dict[int, dict[str, Any]] = {
        0: {"clause_id": "1.4.2", "clause_title": "Accessible"},
        1: {"clause_id": "1.4.3", "clause_title": "Accessible, readily"},
        2: {"clause_id": "1.4.34", "clause_title": "Competent person"},
    }
    store = _StubStore(metadata)
    cases = [
        {
            "name": "comparison",
            "prompt": "Contrast accessible with readily accessible under AS/NZS 3000.",
            "primary_clause_ids": ["1.4.2", "1.4.3"],
        },
        {
            "name": "three-way",
            "prompt": "Compare accessible, readily accessible, and competent person.",
            "primary_clause_ids": ["1.4.2", "1.4.3", "1.4.34"],
        },
    ]

    samples = build_router_dataset(store, benchmark_cases=cases)
    per_window_texts: dict[int, set[str]] = {}
    for sample in samples:
        per_window_texts.setdefault(int(sample["window_id"]), set()).add(str(sample["text"]))

    # Raw prompt is emitted to the primary window only.
    comparison_prompt = "Contrast accessible with readily accessible under AS/NZS 3000."
    assert comparison_prompt in per_window_texts[0]
    assert comparison_prompt not in per_window_texts[1]

    three_way_prompt = "Compare accessible, readily accessible, and competent person."
    assert three_way_prompt in per_window_texts[0]
    assert three_way_prompt not in per_window_texts[1]
    assert three_way_prompt not in per_window_texts[2]

    # Pairwise renderings: each rendered text lands on the left window of the
    # pair only — never both sides.
    pair_0_1 = "How do clause 1.4.2 and clause 1.4.3 differ?"
    assert pair_0_1 in per_window_texts[0]
    assert pair_0_1 not in per_window_texts[1]

    pair_0_2 = "How do clause 1.4.2 and clause 1.4.34 differ?"
    assert pair_0_2 in per_window_texts[0]
    assert pair_0_2 not in per_window_texts[2]

    # And the non-primary pair (1,2) — emit only to window 1 (the left side).
    pair_1_2 = "How do clause 1.4.3 and clause 1.4.34 differ?"
    assert pair_1_2 in per_window_texts[1]
    assert pair_1_2 not in per_window_texts[2]

    # Sanity: no benchmark-derived text ever appears in two windows.
    text_to_window_ids: dict[str, set[int]] = {}
    for sample in samples:
        text_to_window_ids.setdefault(
            str(sample["text"]), set()
        ).add(int(sample["window_id"]))
    benchmark_texts = {
        comparison_prompt,
        three_way_prompt,
        pair_0_1,
        pair_0_2,
        pair_1_2,
    }
    for text in benchmark_texts:
        assert len(text_to_window_ids.get(text, set())) <= 1, (
            f"benchmark text {text!r} landed on multiple windows: "
            f"{text_to_window_ids[text]!r}"
        )


def test_split_clause_windows_only_emit_excerpts() -> None:
    # Two windows share the SAME clause_id but with distinct ``part_index`` —
    # the split-clause case. Every title/clause_id paraphrase would render
    # identically for both windows, so the emitter must drop all of them and
    # fall back to the tokenizer excerpts (which are unique per window).
    metadata: dict[int, dict[str, Any]] = {
        0: {
            "clause_id": "1.5.4.4",
            "clause_title": "Protection by barriers or enclosures",
            "part_index": 1,
            "part_count": 2,
        },
        1: {
            "clause_id": "1.5.4.4",
            "clause_title": "Protection by barriers or enclosures",
            "part_index": 2,
            "part_count": 2,
        },
        2: {"clause_id": "9.9.9", "clause_title": "Unique Title"},
    }
    # Distinct token lists so each split-window gets a distinct excerpt.
    token_lists = {
        0: [100, 101, 102, 103, 104],
        1: [200, 201, 202, 203, 204],
        2: [300, 301, 302, 303, 304],
    }
    store = _StubStore(metadata, window_token_lists=token_lists)

    class _StubTokenizer:
        def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
            return "excerpt:" + ",".join(str(i) for i in token_ids[:3])

    samples = build_router_dataset(store, tokenizer=_StubTokenizer())

    # (a) No text is ever labeled with more than one window_id.
    text_to_window_ids: dict[str, set[int]] = {}
    for sample in samples:
        text_to_window_ids.setdefault(
            str(sample["text"]), set()
        ).add(int(sample["window_id"]))
    duplicates = {
        text: window_ids
        for text, window_ids in text_to_window_ids.items()
        if len(window_ids) > 1
    }
    assert duplicates == {}, (
        f"split-clause leaked colliding (text, window_id) pairs: {duplicates!r}"
    )

    per_window_texts: dict[int, set[str]] = {}
    for sample in samples:
        per_window_texts.setdefault(
            int(sample["window_id"]), set()
        ).add(str(sample["text"]))

    # (b) Both split-clause windows still get at least one sample (the excerpt).
    assert per_window_texts[0], "split-clause window 0 produced no samples"
    assert per_window_texts[1], "split-clause window 1 produced no samples"

    # (c) Every surviving sample for the split-clause windows is the excerpt,
    # NOT the bare title or any clause_id-based rendering.
    for text in per_window_texts[0] | per_window_texts[1]:
        assert text.startswith("excerpt:"), (
            f"split-clause window leaked non-excerpt text: {text!r}"
        )
    # In particular: bare title, "{clause_id} {title}", and
    # "Clause {clause_id}: {title}" must all be absent.
    assert "Protection by barriers or enclosures" not in per_window_texts[0]
    assert "Protection by barriers or enclosures" not in per_window_texts[1]
    assert "1.5.4.4 Protection by barriers or enclosures" not in per_window_texts[0]
    assert "1.5.4.4 Protection by barriers or enclosures" not in per_window_texts[1]
    assert (
        "Clause 1.5.4.4: Protection by barriers or enclosures"
        not in per_window_texts[0]
    )
    assert (
        "Clause 1.5.4.4: Protection by barriers or enclosures"
        not in per_window_texts[1]
    )

    # (d) Non-split-clause window is unaffected.
    assert "Unique Title" in per_window_texts[2]
    assert "Define Unique Title" in per_window_texts[2]
