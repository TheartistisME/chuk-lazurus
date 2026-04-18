"""Router training-set construction from a ``TorchKnowledgeStore``.

``build_router_dataset`` walks every window in a store's ``window_metadata``
and emits ``(text, window_id)`` pairs suitable for training a window-id
classifier. The builder is metadata-defensive: any template whose fields
are missing on a given window is skipped gracefully, so the function works
against any store that follows the ``window_metadata`` contract.
"""

from __future__ import annotations

import string
from collections.abc import Iterable
from itertools import combinations
from typing import Any, Protocol

from .augmentation import augment_clause_title


class _StoreLike(Protocol):
    window_metadata: dict[int, dict[str, Any]]
    window_token_lists: dict[int, list[int]]


DEFAULT_PARAPHRASE_TEMPLATES: tuple[str, ...] = (
    "Define {title}",
    "What is {title}?",
    "Clause {clause_id}: {title}",
    "Tell me about {title}",
    "In plain language, what does {title} mean?",
    "What does clause {clause_id} mean in practice for {title}?",
    "When does {title} apply under AS/NZS 3000?",
    "How do I interpret {title} under AS/NZS 3000?",
    "What are the key points for {title}?",
    "Summarize {title} in practical terms.",
)
"""Paraphrase templates fired against each window with a matching metadata set.

Each template is a :py:meth:`str.format` pattern. A template whose required
fields (``{title}``, ``{clause_id}``) are absent or empty on a given window
is skipped for that window — missing metadata never raises.
"""

DEFAULT_MULTI_CLAUSE_TEMPLATES: tuple[str, ...] = (
    "Compare {left_title} and {right_title} under AS/NZS 3000.",
    "What is the difference between {left_title} and {right_title}?",
    "How do clause {left_clause_id} and clause {right_clause_id} differ?",
    "Contrast clause {left_clause_id} with clause {right_clause_id}.",
)
"""Pairwise comparison prompts fired for benchmark multi-clause cases."""


def _required_fields(template: str) -> set[str]:
    formatter = string.Formatter()
    fields: set[str] = set()
    try:
        for _literal, field_name, _fmt, _conv in formatter.parse(template):
            if field_name is None:
                continue
            head = field_name.split(".", 1)[0].split("[", 1)[0]
            if head:
                fields.add(head)
    except ValueError:
        return set()
    return fields


def _meta_str(meta: dict[str, Any], key: str) -> str:
    value = meta.get(key)
    if value is None:
        return ""
    return str(value).strip()


def _decode_excerpt(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        text = tokenizer.decode(token_ids)
    return (text or "").strip()


def _resolve_case_window_ids(
    store: _StoreLike, primary_clause_ids: Iterable[str]
) -> list[int]:
    clause_to_window: dict[str, int] = {}
    for window_id in sorted(store.window_metadata):
        clause_id = _meta_str(store.window_metadata[window_id], "clause_id")
        if clause_id and clause_id not in clause_to_window:
            clause_to_window[clause_id] = int(window_id)

    resolved: list[int] = []
    seen: set[int] = set()
    for clause_id in primary_clause_ids:
        window_id = clause_to_window.get(str(clause_id).strip())
        if window_id is None or window_id in seen:
            continue
        seen.add(window_id)
        resolved.append(window_id)
    return resolved


def build_router_dataset(
    store: _StoreLike,
    paraphrase_templates: Iterable[str] | None = None,
    *,
    augment: bool = True,
    augmentation_templates: Iterable[str] | None = None,
    tokenizer: Any | None = None,
    max_excerpt_tokens: int = 120,
    benchmark_cases: Iterable[dict[str, Any]] | None = None,
    multi_clause_templates: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Produce ``(text, window_id)`` pairs from a knowledge store.

    Sources per window:
      * raw ``clause_title``
      * ``"{clause_id} {clause_title}"`` when both metadata fields exist
      * deterministic title augmentations when ``augment`` is ``True``
      * each paraphrase template whose ``{fields}`` are present on the window
      * the first ``max_excerpt_tokens`` tokens of the window text, decoded
        via ``tokenizer`` (skipped when ``tokenizer`` is ``None`` or the
        window has no token list).

    The function never raises on missing metadata fields — it simply skips
    any template that cannot be formatted for a given window.

    Args:
        store: object exposing ``window_metadata`` and ``window_token_lists``
            (a ``TorchKnowledgeStore`` in production).
        paraphrase_templates: iterable of :py:meth:`str.format` patterns.
            Defaults to :data:`DEFAULT_PARAPHRASE_TEMPLATES`.
        augment: whether to emit deterministic clause-title augmentations.
            Defaults to ``True`` so callers can add a future ``--no-augment``
            flag by passing ``augment=False``.
        augmentation_templates: optional :py:meth:`str.format` patterns used
            for deterministic title augmentation. Defaults to the templates in
            :mod:`tools._window_router.augmentation`.
        tokenizer: optional tokenizer object with ``.decode(ids)``. When
            ``None``, no excerpt samples are produced.
        max_excerpt_tokens: number of leading tokens to use for each excerpt.
        benchmark_cases: optional benchmark rows used to inject explicit
            multi-clause comparison prompts into the training set.
        multi_clause_templates: optional pairwise comparison templates for
            benchmark cases whose ``primary_clause_ids`` length is >= 2.

    Returns:
        list of ``{"text": str, "window_id": int}`` records.
    """
    templates = (
        tuple(paraphrase_templates)
        if paraphrase_templates is not None
        else DEFAULT_PARAPHRASE_TEMPLATES
    )
    template_fields: list[tuple[str, set[str]]] = [
        (tpl, _required_fields(tpl)) for tpl in templates
    ]
    pair_templates = (
        tuple(multi_clause_templates)
        if multi_clause_templates is not None
        else DEFAULT_MULTI_CLAUSE_TEMPLATES
    )
    pair_template_fields: list[tuple[str, set[str]]] = [
        (tpl, _required_fields(tpl)) for tpl in pair_templates
    ]

    # Collision detection: titles shared by 2+ windows produce contradictory
    # (text, window_id) labels and poison cross-entropy training. Compute the
    # collision set once so emission can route around it.
    #
    # Clause-id collisions (``collision_clause_ids``) are the split-clause
    # case: a single clause (e.g. ``1.5.4.4``) is split into multiple
    # windows via ``part_index``. For those windows, even the
    # ``"{clause_id} {title}"`` and "Clause {clause_id}: {title}" renderings
    # collide across sibling windows, so ALL title/clause-id paraphrases
    # must be skipped and only excerpts (unique per window) may survive.
    title_counts: dict[str, int] = {}
    clause_id_counts: dict[str, int] = {}
    for meta in store.window_metadata.values():
        normalized_title = _meta_str(meta, "clause_title")
        if normalized_title:
            title_counts[normalized_title] = title_counts.get(normalized_title, 0) + 1
        normalized_clause_id = _meta_str(meta, "clause_id")
        if normalized_clause_id:
            clause_id_counts[normalized_clause_id] = (
                clause_id_counts.get(normalized_clause_id, 0) + 1
            )
    collision_titles: set[str] = {
        title for title, count in title_counts.items() if count >= 2
    }
    collision_clause_ids: set[str] = {
        clause_id for clause_id, count in clause_id_counts.items() if count >= 2
    }

    samples: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    def _emit(window_id: int, text: str) -> None:
        stripped = text.strip()
        if not stripped:
            return
        key = (int(window_id), stripped)
        if key in seen:
            return
        seen.add(key)
        samples.append({"text": stripped, "window_id": int(window_id)})

    for window_id in sorted(store.window_metadata):
        metadata = store.window_metadata[window_id]
        title = _meta_str(metadata, "clause_title")
        clause_id = _meta_str(metadata, "clause_id")

        # Two orthogonal collision states, both of which poison cross-entropy
        # training if not routed around:
        #   - is_title_collided: bare title is shared with sibling windows.
        #       Title-only renderings ("Definitions", "Define Definitions",
        #       ...) would map a single text to multiple window_ids. Skip
        #       those and keep only clause_id-bearing renderings.
        #   - is_clause_id_collided: the clause_id itself is shared with
        #       sibling windows (split-clause via part_index). EVEN the
        #       "{clause_id} {title}" and "Clause {clause_id}: ..." forms
        #       collide across siblings. For those windows, ALL title/
        #       clause_id paraphrases must be dropped — only excerpts
        #       (unique per window) may survive.
        is_title_collided = bool(title) and title in collision_titles
        is_clause_id_collided = (
            bool(clause_id) and clause_id in collision_clause_ids
        )
        skip_title_only = is_title_collided or is_clause_id_collided
        skip_clause_id_paraphrase = is_clause_id_collided
        require_clause_id = is_title_collided and bool(clause_id)

        if title and not skip_title_only:
            _emit(window_id, title)
        if title and clause_id and not skip_clause_id_paraphrase:
            _emit(window_id, f"{clause_id} {title}")
        if augment and title and (
            not skip_clause_id_paraphrase
            and (not is_title_collided or clause_id)
        ):
            for augmented in augment_clause_title(
                title,
                clause_id=clause_id or None,
                templates=augmentation_templates,
                collision_titles=collision_titles or None,
                skip_all=skip_clause_id_paraphrase,
            ):
                _emit(window_id, augmented)

        if not skip_clause_id_paraphrase:
            fields_for_template = {"title": title, "clause_id": clause_id}
            for template, required in template_fields:
                if any(not fields_for_template.get(field) for field in required):
                    continue
                try:
                    formatted = template.format(**fields_for_template)
                except (KeyError, IndexError, ValueError):
                    continue
                if require_clause_id and clause_id not in formatted:
                    # Collision-title window: skip any rendering that would not
                    # disambiguate between sibling windows sharing the title.
                    continue
                if is_title_collided and not clause_id:
                    # No disambiguator available — skip title-only renderings.
                    continue
                _emit(window_id, formatted)

        if tokenizer is not None:
            token_list = store.window_token_lists.get(int(window_id), [])
            if token_list:
                excerpt_ids = [int(t) for t in token_list[: int(max_excerpt_tokens)]]
                excerpt = _decode_excerpt(tokenizer, excerpt_ids)
                if excerpt:
                    _emit(window_id, excerpt)

    for case in benchmark_cases or ():
        prompt = str(case.get("prompt") or "").strip()
        primary_clause_ids = [
            str(clause_id).strip()
            for clause_id in (case.get("primary_clause_ids") or ())
            if str(clause_id).strip()
        ]
        if len(primary_clause_ids) < 2:
            continue
        window_ids = _resolve_case_window_ids(store, primary_clause_ids)
        if len(window_ids) < 2:
            continue

        # Ownership rule for multi-target benchmark prompts: emit each
        # rendered text to exactly ONE window — the one resolved from the
        # FIRST entry in ``primary_clause_ids`` (which ``_resolve_case_window_ids``
        # preserves as the first element of ``window_ids``). Emitting the
        # same prompt to every window would recreate the router-single-label
        # (text, window_id) collision that poisons cross-entropy.
        primary_window_id = window_ids[0]

        if prompt:
            _emit(primary_window_id, prompt)

        for left_window_id, right_window_id in combinations(window_ids, 2):
            left_meta = store.window_metadata[int(left_window_id)]
            right_meta = store.window_metadata[int(right_window_id)]
            fields = {
                "left_title": _meta_str(left_meta, "clause_title"),
                "right_title": _meta_str(right_meta, "clause_title"),
                "left_clause_id": _meta_str(left_meta, "clause_id"),
                "right_clause_id": _meta_str(right_meta, "clause_id"),
            }
            for template, required in pair_template_fields:
                if any(not fields.get(field) for field in required):
                    continue
                try:
                    rendered = template.format(**fields)
                except (KeyError, IndexError, ValueError):
                    continue
                # Emit the rendered pair prompt to only the LEFT window of
                # the combination. Since ``combinations`` preserves input
                # order and ``window_ids[0]`` is the primary window, this
                # guarantees the primary window owns every pair it is part
                # of, and each non-primary pair is owned by the earliest
                # (leftmost) participant — unique per rendered text.
                _emit(left_window_id, rendered)

    return samples


__all__ = [
    "DEFAULT_MULTI_CLAUSE_TEMPLATES",
    "DEFAULT_PARAPHRASE_TEMPLATES",
    "build_router_dataset",
]
