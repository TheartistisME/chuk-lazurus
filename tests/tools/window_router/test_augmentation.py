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


def test_collision_title_filters_out_title_only_renderings() -> None:
    # When the title collides with other windows, only clause-id-bearing
    # renderings may survive so the classifier can disambiguate.
    variants = augment_clause_title(
        "Definitions",
        clause_id="1.2.3",
        collision_titles={"Definitions"},
    )

    # At least one variant must survive (the clause-id-bearing ones).
    assert variants, "collision-safe variants should still be emitted"
    # Every surviving variant must contain the clause_id token.
    for variant in variants:
        assert "1.2.3" in variant, (
            f"collision variant missing clause_id disambiguator: {variant!r}"
        )
    # And none of them is the bare title or the "{clause_id} {title}" base.
    assert "Definitions" not in variants
    assert "1.2.3 Definitions" not in variants


def test_collision_title_without_clause_id_returns_empty() -> None:
    variants = augment_clause_title(
        "Definitions",
        clause_id=None,
        collision_titles={"Definitions"},
    )
    assert variants == ()


def test_collision_titles_none_matches_legacy_behaviour() -> None:
    # Passing ``collision_titles=None`` (the default) must reproduce the
    # baseline behaviour exactly — i.e. the full default template set fires.
    baseline = augment_clause_title("Scope", clause_id="1.1.1")
    explicit_none = augment_clause_title(
        "Scope", clause_id="1.1.1", collision_titles=None
    )
    empty_set = augment_clause_title(
        "Scope", clause_id="1.1.1", collision_titles=set()
    )

    assert baseline == explicit_none
    # An empty collision set also means no collision — same as None.
    assert baseline == empty_set


def test_non_colliding_title_is_unaffected_by_collision_set() -> None:
    # Title is NOT in the collision set, so all default templates fire.
    variants = augment_clause_title(
        "Scope",
        clause_id="1.1.1",
        collision_titles={"Definitions", "Other"},
    )
    baseline = augment_clause_title("Scope", clause_id="1.1.1")
    assert variants == baseline


def test_skip_all_returns_empty_unconditionally() -> None:
    # ``skip_all=True`` is the split-clause short-circuit: even
    # clause_id-bearing variants would collide with sibling windows, so
    # nothing must be emitted regardless of the other kwargs.
    assert (
        augment_clause_title(
            "Protection by barriers or enclosures",
            clause_id="1.5.4.4",
            skip_all=True,
        )
        == ()
    )

    # Still empty when ``collision_titles`` is supplied.
    assert (
        augment_clause_title(
            "Protection by barriers or enclosures",
            clause_id="1.5.4.4",
            collision_titles={"Protection by barriers or enclosures"},
            skip_all=True,
        )
        == ()
    )

    # Still empty with a custom templates iterable.
    assert (
        augment_clause_title(
            "Scope",
            clause_id="1.1.1",
            templates=("Explain {title}",),
            skip_all=True,
        )
        == ()
    )

    # And empty even when there is no clause_id at all.
    assert augment_clause_title("Scope", skip_all=True) == ()


def test_skip_all_false_preserves_baseline() -> None:
    # Explicitly passing ``skip_all=False`` must reproduce the default
    # behaviour exactly.
    assert augment_clause_title(
        "Scope", clause_id="1.1.1", skip_all=False
    ) == augment_clause_title("Scope", clause_id="1.1.1")
