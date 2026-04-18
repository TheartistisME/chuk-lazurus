"""Stratified pilot sampling for the window router training dataset.

The learned router pilot artefact must cover every cleaned label so that
encoder diagnostics (pre-training sanity checks) can distinguish between
"works on a handful of anchor classes" and "works on the full label
space". This module implements a deterministic stratified subsample
keyed by an arbitrary label field (``window_id`` by default).

The sampler is pure: given the same records, ``fraction``, ``min_per_label``,
``seed``, and ``label_key`` it always returns the same output list.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["stratified_sample"]


def stratified_sample(
    records: Iterable[Mapping[str, Any]],
    *,
    fraction: float = 0.05,
    min_per_label: int = 2,
    seed: int = 42,
    label_key: str = "window_id",
) -> list[dict[str, Any]]:
    """Return a deterministic stratified subsample grouped by ``label_key``.

    For each distinct label value present in ``records`` the sampler retains
    ``max(min_per_label, round(fraction * count))`` records, clamped to the
    group size. Labels with fewer records than the target keep all of their
    records. Within-group input ordering is preserved; across-group ordering
    is stabilised by sorting labels lexicographically on their string form.

    Sampling uses a single ``random.Random(seed)`` instance so results are
    fully reproducible. Records are shallow-copied into fresh ``dict``
    objects so callers can freely mutate the output without aliasing the
    input.

    Args:
        records: Iterable of record dicts. Each record must contain
            ``label_key``. A one-shot iterable (e.g. a generator) is
            accepted; it is materialised internally.
        fraction: Target fraction per label. Must lie in the half-open
            interval ``(0, 1]``.
        min_per_label: Minimum retained records per label. Must be
            ``>= 1``.
        seed: Seed for deterministic sampling.
        label_key: Record key to group by. Defaults to ``"window_id"``.

    Returns:
        List of record dicts (shallow copies) ordered first by sorted
        label (string form) and then by input order within each label.

    Raises:
        ValueError: If ``fraction`` is not in ``(0, 1]`` or
            ``min_per_label < 1``.
        KeyError: If any record is missing ``label_key``.
    """
    if not (0 < fraction <= 1):
        raise ValueError(
            f"fraction must be in (0, 1], got {fraction!r}",
        )
    if min_per_label < 1:
        raise ValueError(
            f"min_per_label must be >= 1, got {min_per_label!r}",
        )

    groups: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        if label_key not in record:
            raise KeyError(
                f"record missing required label_key {label_key!r}: {record!r}",
            )
        label = record[label_key]
        groups.setdefault(label, []).append(dict(record))

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for label in sorted(groups, key=lambda value: str(value)):
        group = groups[label]
        group_size = len(group)
        target = max(min_per_label, round(fraction * group_size))
        target = min(target, group_size)

        if target >= group_size:
            sampled.extend(group)
            continue

        indices = sorted(rng.sample(range(group_size), target))
        sampled.extend(group[index] for index in indices)

    return sampled
