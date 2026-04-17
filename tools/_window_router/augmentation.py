"""Deterministic clause-title augmentation for router dataset building."""

from __future__ import annotations

import string
from collections.abc import Iterable


DEFAULT_TITLE_AUGMENTATION_TEMPLATES: tuple[str, ...] = (
    "Explain the clause titled {title}",
    "Summarize the clause titled {title}",
    "What does the clause about {title} cover?",
    "What are the requirements for {title}?",
    "What obligations apply to {title}?",
    "What rules apply to {title}?",
    "What guidance is given on {title}?",
    "Find the clause about {title}",
    "Locate the section on {title}",
    "Which clause covers {title}?",
    "Where is {title} covered?",
    "Show me the clause on {title}",
    "Explain clause {clause_id} on {title}",
    "Summarize clause {clause_id} on {title}",
    "What does clause {clause_id} say about {title}?",
    "What are the requirements in clause {clause_id} for {title}?",
    "Locate clause {clause_id} about {title}",
    "Clause {clause_id} covers {title}",
)
"""Default deterministic templates used to paraphrase each clause title."""


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


def _normalize(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def augment_clause_title(
    title: str,
    *,
    clause_id: str | None = None,
    templates: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Render deterministic paraphrase variants for a clause title.

    The returned variants never include the base ``title`` or
    ``"{clause_id} {title}"`` forms because those are emitted directly by
    :func:`tools._window_router.dataset.build_router_dataset`.
    """

    normalized_title = _normalize(title)
    normalized_clause_id = _normalize(clause_id)
    if not normalized_title:
        return ()

    active_templates = (
        tuple(templates)
        if templates is not None
        else DEFAULT_TITLE_AUGMENTATION_TEMPLATES
    )
    template_fields: list[tuple[str, set[str]]] = [
        (template, _required_fields(template)) for template in active_templates
    ]
    base_forms = {normalized_title}
    if normalized_clause_id:
        base_forms.add(f"{normalized_clause_id} {normalized_title}")

    variants: list[str] = []
    seen: set[str] = set()
    fields = {"title": normalized_title, "clause_id": normalized_clause_id}

    for template, required in template_fields:
        if any(not fields.get(field) for field in required):
            continue
        try:
            rendered = template.format(**fields)
        except (KeyError, IndexError, ValueError):
            continue
        stripped = rendered.strip()
        if not stripped or stripped in base_forms or stripped in seen:
            continue
        seen.add(stripped)
        variants.append(stripped)

    return tuple(variants)


__all__ = ["DEFAULT_TITLE_AUGMENTATION_TEMPLATES", "augment_clause_title"]
