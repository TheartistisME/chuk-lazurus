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
)
"""Paraphrase templates fired against each window with a matching metadata set.

Each template is a :py:meth:`str.format` pattern. A template whose required
fields (``{title}``, ``{clause_id}``) are absent or empty on a given window
is skipped for that window — missing metadata never raises.
"""


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


def build_router_dataset(
    store: _StoreLike,
    paraphrase_templates: Iterable[str] | None = None,
    *,
    augment: bool = True,
    augmentation_templates: Iterable[str] | None = None,
    tokenizer: Any | None = None,
    max_excerpt_tokens: int = 120,
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

        if title:
            _emit(window_id, title)
        if title and clause_id:
            _emit(window_id, f"{clause_id} {title}")
        if augment and title:
            for augmented in augment_clause_title(
                title,
                clause_id=clause_id or None,
                templates=augmentation_templates,
            ):
                _emit(window_id, augmented)

        fields_for_template = {"title": title, "clause_id": clause_id}
        for template, required in template_fields:
            if any(not fields_for_template.get(field) for field in required):
                continue
            try:
                formatted = template.format(**fields_for_template)
            except (KeyError, IndexError, ValueError):
                continue
            _emit(window_id, formatted)

        if tokenizer is not None:
            token_list = store.window_token_lists.get(int(window_id), [])
            if token_list:
                excerpt_ids = [int(t) for t in token_list[: int(max_excerpt_tokens)]]
                excerpt = _decode_excerpt(tokenizer, excerpt_ids)
                if excerpt:
                    _emit(window_id, excerpt)

    return samples


__all__ = ["DEFAULT_PARAPHRASE_TEMPLATES", "build_router_dataset"]
