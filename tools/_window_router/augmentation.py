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
    "Give an overview of {title}",
    "Describe {title} in plain language",
    "What is the purpose of {title}?",
    "What does {title} mean in this standard?",
    "What should I know about {title}?",
    "What provisions relate to {title}?",
    "What is stated about {title}?",
    "What instructions cover {title}?",
    "What conditions apply to {title}?",
    "What limitations apply to {title}?",
    "What details are given for {title}?",
    "How is {title} addressed?",
    "How is {title} defined or described?",
    "How should {title} be handled?",
    "How should {title} be interpreted?",
    "When does {title} apply?",
    "When is {title} required?",
    "When must {title} be considered?",
    "Where can I find requirements for {title}?",
    "Where can I find guidance on {title}?",
    "Which section discusses {title}?",
    "Which requirements mention {title}?",
    "Find requirements about {title}",
    "Find guidance about {title}",
    "Show requirements for {title}",
    "Show guidance for {title}",
    "Summarize requirements for {title}",
    "Summarize guidance for {title}",
    "Explain how the standard treats {title}",
    "Explain the requirements around {title}",
    "Explain the guidance around {title}",
    "Explain the application of {title}",
    "Describe the rules for {title}",
    "Describe the requirements for {title}",
    "Describe the guidance for {title}",
    "Outline the requirements for {title}",
    "Outline the guidance for {title}",
    "Review the clause on {title}",
    "Review the requirements for {title}",
    "Review the guidance for {title}",
    "Summarize clause {clause_id} for {title}",
    "Describe clause {clause_id} for {title}",
    "Explain clause {clause_id} for {title}",
    "Review clause {clause_id} for {title}",
    "Outline clause {clause_id} for {title}",
    "What guidance does clause {clause_id} give on {title}?",
    "What requirements does clause {clause_id} set for {title}?",
    "What conditions does clause {clause_id} place on {title}?",
    "Where does clause {clause_id} discuss {title}?",
    "How does clause {clause_id} address {title}?",
    "How should clause {clause_id} be read for {title}?",
    "When does clause {clause_id} apply to {title}?",
    "Clause {clause_id} explains {title}",
    "Clause {clause_id} sets requirements for {title}",
    "Clause {clause_id} gives guidance on {title}",
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
    collision_titles: set[str] | None = None,
    skip_all: bool = False,
) -> tuple[str, ...]:
    """Render deterministic paraphrase variants for a clause title.

    The returned variants never include the base ``title`` or
    ``"{clause_id} {title}"`` forms because those are emitted directly by
    :func:`tools._window_router.dataset.build_router_dataset`.

    When ``collision_titles`` is provided and ``title`` is a member of it,
    the title alone is ambiguous (it points at multiple window ids). In that
    case only clause-id-bearing renderings are returned so the classifier
    never sees contradictory labels for identical text. A collision title
    with no ``clause_id`` yields an empty tuple because there is no safe
    disambiguator available.

    When ``skip_all`` is ``True`` (e.g. the window's clause_id is itself
    shared with sibling windows — a split-clause scenario where even
    ``"{clause_id} {title}"`` renderings would collide), an empty tuple is
    returned unconditionally. This is the caller's short-circuit for
    "this window has no safe title-based disambiguator at all".
    """

    if skip_all:
        return ()
    normalized_title = _normalize(title)
    normalized_clause_id = _normalize(clause_id)
    if not normalized_title:
        return ()

    is_collision = (
        collision_titles is not None and normalized_title in collision_titles
    )
    if is_collision and not normalized_clause_id:
        # Title alone collides and we have no disambiguator: skip entirely.
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
        if is_collision and normalized_clause_id not in stripped:
            # Collision-title augmentations must carry the clause_id token
            # so the classifier can resolve the ambiguity.
            continue
        seen.add(stripped)
        variants.append(stripped)

    return tuple(variants)


__all__ = ["DEFAULT_TITLE_AUGMENTATION_TEMPLATES", "augment_clause_title"]
