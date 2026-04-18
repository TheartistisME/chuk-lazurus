"""Tests for ``tools/_window_router/cache.py``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools._window_router.cache import (
    CachedFeatureDataset,
    default_feature_cache_path,
    encoder_version,
    feature_cache_key,
    load_feature_cache,
    resolve_device,
    write_feature_cache,
)


def test_feature_cache_key_changes_with_dataset_hash() -> None:
    version = encoder_version(
        encoder_type="gemma-embed",
        model_id="google/gemma-test",
    )
    key_a = feature_cache_key(
        encoder_type="gemma-embed",
        encoder_version=version,
        dataset_hash="a" * 64,
    )
    key_b = feature_cache_key(
        encoder_type="gemma-embed",
        encoder_version=version,
        dataset_hash="b" * 64,
    )
    assert key_a != key_b


def test_write_and_load_feature_cache_roundtrip(tmp_path: Path) -> None:
    out_path = default_feature_cache_path(
        tmp_path / "dataset.jsonl",
        encoder_type="gemma-embed",
        encoder_version="embedding-layer-v1::google/gemma-test",
        dataset_hash="a" * 64,
    )
    write_feature_cache(
        out_path,
        features=[[1.0, 2.0], [3.0, 4.0]],
        labels=[0, 1],
        meta={
            "encoder_type": "gemma-embed",
            "encoder_version": "embedding-layer-v1::google/gemma-test",
            "dataset_hash": "a" * 64,
            "sample_count": 2,
            "num_labels": 2,
            "input_size": 2,
        },
    )
    payload = load_feature_cache(out_path)
    dataset = CachedFeatureDataset(payload["features"], payload["labels"])
    assert len(dataset) == 2
    assert tuple(payload["features"].shape) == (2, 2)
    assert tuple(payload["labels"].shape) == (2,)
    assert int(dataset[1]["label"]) == 1
    assert [float(v) for v in dataset[1]["features"].tolist()] == [3.0, 4.0]


def test_resolve_device_auto_prefers_available_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    assert resolve_device("auto", preferred_device="cpu") == "cpu"


def test_encoder_version_sentence_transformer_prefix() -> None:
    """Sentence-transformer encoder must use a distinct version prefix so the
    SHA256 cache key invalidates when switching encoders.
    """
    version = encoder_version(
        encoder_type="sentence-transformer",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    )
    assert version == "sentence-transformer-v1::sentence-transformers/all-MiniLM-L6-v2"

    gemma_version = encoder_version(
        encoder_type="gemma-embed",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    )
    key_st = feature_cache_key(
        encoder_type="sentence-transformer",
        encoder_version=version,
        dataset_hash="a" * 64,
    )
    key_gemma = feature_cache_key(
        encoder_type="gemma-embed",
        encoder_version=gemma_version,
        dataset_hash="a" * 64,
    )
    assert key_st != key_gemma


def test_encoder_version_sentence_transformer_missing_model_id() -> None:
    """Missing model_id falls through to ``::none`` so the prefix still changes."""
    version = encoder_version(encoder_type="sentence-transformer", model_id=None)
    assert version == "sentence-transformer-v1::none"
