"""Tests for ``tools/_window_router/pilot.py``.

Exercises determinism, per-label sampling bounds, validation, input
ordering, shallow-copy semantics, and one-shot iterable support for the
stratified pilot sampler.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

import pytest

from tools._window_router.pilot import stratified_sample


def _make_records(
    label_counts: dict[str, int],
    *,
    label_key: str = "window_id",
) -> list[dict[str, Any]]:
    """Build a flat list of records grouped by ``label_key``.

    Each record carries a monotonically increasing ``idx`` so tests can
    assert on input order preservation.
    """
    records: list[dict[str, Any]] = []
    counter = 0
    for label, count in label_counts.items():
        for local in range(count):
            records.append(
                {
                    label_key: label,
                    "idx": counter,
                    "local": local,
                    "payload": {"tag": label, "n": local},
                }
            )
            counter += 1
    return records


def test_determinism_same_seed_identical() -> None:
    records = _make_records({"a": 40, "b": 40, "c": 40})

    first = stratified_sample(records, fraction=0.25, seed=7)
    second = stratified_sample(records, fraction=0.25, seed=7)

    assert first == second


def test_determinism_different_seed_differs() -> None:
    records = _make_records({"a": 40, "b": 40, "c": 40})

    first = stratified_sample(records, fraction=0.25, seed=7)
    second = stratified_sample(records, fraction=0.25, seed=11)

    # Same per-label count, different picks.
    assert len(first) == len(second)
    first_ids = [record["idx"] for record in first]
    second_ids = [record["idx"] for record in second]
    assert first_ids != second_ids


def test_per_label_bounds_use_round_fraction() -> None:
    records = _make_records({"a": 100, "b": 40, "c": 7})

    sampled = stratified_sample(records, fraction=0.1, min_per_label=2, seed=0)

    counts: dict[str, int] = {}
    for record in sampled:
        counts[record["window_id"]] = counts.get(record["window_id"], 0) + 1

    # round(0.1 * 100) = 10 ; round(0.1 * 40) = 4 ; round(0.1 * 7) = 1 -> floor clamped to min_per_label=2
    assert counts == {"a": 10, "b": 4, "c": 2}


def test_every_label_appears_at_least_once() -> None:
    records = _make_records({str(i): 3 for i in range(25)})

    sampled = stratified_sample(records, fraction=0.01, min_per_label=1, seed=3)

    labels_in = {record["window_id"] for record in records}
    labels_out = {record["window_id"] for record in sampled}
    assert labels_out == labels_in


def test_small_group_preserved_when_min_exceeds_count() -> None:
    records = _make_records({"singleton": 1, "bigger": 10})

    sampled = stratified_sample(
        records,
        fraction=0.5,
        min_per_label=2,
        seed=99,
    )

    singleton_records = [r for r in sampled if r["window_id"] == "singleton"]
    assert len(singleton_records) == 1
    assert singleton_records[0]["idx"] == 0


@pytest.mark.parametrize("bad_fraction", [0, -0.1, 1.5, 2.0])
def test_fraction_out_of_range_raises(bad_fraction: float) -> None:
    records = _make_records({"a": 5})
    with pytest.raises(ValueError, match="fraction"):
        stratified_sample(records, fraction=bad_fraction)


def test_min_per_label_zero_raises() -> None:
    records = _make_records({"a": 5})
    with pytest.raises(ValueError, match="min_per_label"):
        stratified_sample(records, fraction=0.5, min_per_label=0)


def test_input_order_preserved_within_label() -> None:
    records = _make_records({"only": 10})

    sampled = stratified_sample(
        records,
        fraction=0.7,
        min_per_label=1,
        seed=1,
    )

    idxs = [record["idx"] for record in sampled]
    assert idxs == sorted(idxs)
    assert len(idxs) == round(0.7 * 10)


def test_shallow_copy_does_not_alias_input() -> None:
    records = _make_records({"a": 3})

    sampled = stratified_sample(records, fraction=1.0, min_per_label=1, seed=0)

    assert len(sampled) == 3
    sampled[0]["idx"] = -999
    sampled[0]["new_field"] = "mutated"

    # Top-level mutation must not leak into source record.
    assert records[0]["idx"] == 0
    assert "new_field" not in records[0]


def test_custom_label_key_switches_grouping() -> None:
    records = [
        {"clause_id": "x", "window_id": "shared", "idx": 0},
        {"clause_id": "x", "window_id": "shared", "idx": 1},
        {"clause_id": "y", "window_id": "shared", "idx": 2},
        {"clause_id": "y", "window_id": "shared", "idx": 3},
        {"clause_id": "z", "window_id": "shared", "idx": 4},
    ]

    sampled = stratified_sample(
        records,
        fraction=0.5,
        min_per_label=1,
        seed=0,
        label_key="clause_id",
    )

    grouped: dict[str, int] = {}
    for record in sampled:
        grouped[record["clause_id"]] = grouped.get(record["clause_id"], 0) + 1

    # round(0.5 * 2) = 1 for x and y ; round(0.5 * 1) = 1 (clamped) for z.
    assert grouped == {"x": 1, "y": 1, "z": 1}


def test_accepts_one_shot_iterable() -> None:
    records = _make_records({"a": 5, "b": 5})

    def generator() -> Iterator[dict[str, Any]]:
        yield from records

    sampled = stratified_sample(
        generator(),
        fraction=0.4,
        min_per_label=1,
        seed=0,
    )

    counts: dict[str, int] = {}
    for record in sampled:
        counts[record["window_id"]] = counts.get(record["window_id"], 0) + 1

    assert counts == {"a": 2, "b": 2}


def test_missing_label_key_raises_keyerror() -> None:
    records = [{"idx": 0}, {"idx": 1}]
    with pytest.raises(KeyError, match="window_id"):
        stratified_sample(records)


def test_output_ordering_sorted_by_label() -> None:
    records = _make_records({"zulu": 2, "alpha": 2, "mike": 2})

    sampled = stratified_sample(records, fraction=1.0, min_per_label=1, seed=0)

    labels_in_output = [record["window_id"] for record in sampled]
    assert labels_in_output == ["alpha", "alpha", "mike", "mike", "zulu", "zulu"]


def test_sampling_uses_random_sample_semantics() -> None:
    """Spot-check that sampling for a single group matches ``random.Random.sample``."""
    records = _make_records({"solo": 20})

    sampled = stratified_sample(
        records,
        fraction=0.5,
        min_per_label=1,
        seed=123,
    )

    rng = random.Random(123)
    expected_indices = sorted(rng.sample(range(20), round(0.5 * 20)))
    assert [record["idx"] for record in sampled] == expected_indices
