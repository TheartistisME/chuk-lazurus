"""Tests for ``tools/_window_router/augmentation.py``."""

from __future__ import annotations

from tools._window_router.augmentation import (
    DEFAULT_TITLE_AUGMENTATION_TEMPLATES,
    augment_clause_title,
)


def test_default_templates_expand_title_and_clause_id() -> None:
    variants = augment_clause_title("Scope", clause_id="1.1.1")

    assert len(variants) == len(DEFAULT_TITLE_AUGMENTATION_TEMPLATES)
    assert variants[0] == "Explain the clause titled Scope"
    assert "What does clause 1.1.1 say about Scope?" in variants
    assert "Clause 1.1.1 covers Scope" in variants


def test_missing_clause_id_skips_clause_specific_templates() -> None:
    variants = augment_clause_title(
        "Scope",
        templates=("Explain {title}", "Explain clause {clause_id} on {title}"),
    )

    assert variants == ("Explain Scope",)


def test_augmentation_dedupes_and_ignores_base_forms() -> None:
    variants = augment_clause_title(
        "Scope",
        clause_id="1.1.1",
        templates=(
            "{title}",
            " {title} ",
            "{clause_id} {title}",
            "Explain {title}",
            "Explain {title}",
        ),
    )

    assert variants == ("Explain Scope",)
