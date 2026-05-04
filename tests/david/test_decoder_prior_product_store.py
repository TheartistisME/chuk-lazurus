from __future__ import annotations

from pathlib import Path

import pytest

from chuk_lazarus.david.decoder_prior_store import (
    DecoderPriorProductStore,
    DecoderPriorRecord,
    DecoderPriorScope,
    DecoderPriorSelection,
    IncompatibleDecoderPriorScope,
)


def _scope(**overrides: object) -> DecoderPriorScope:
    data = {
        "model_id": "gemma-e2b",
        "tokenizer_id": "gemma-tokenizer",
        "adapter_family": "gemma",
        "layer": 12,
        "task_type": "patch_target",
        "steering_version": "v1",
        "model_revision": "abc123",
        "adapter_config_id": "gemma-e2b-default",
        "insertion_family": "kv_direct",
    }
    data.update(overrides)
    return DecoderPriorScope(**data)  # type: ignore[arg-type]


def test_decoder_prior_store_updates_stats_and_seed_alpha(tmp_path: Path) -> None:
    store = DecoderPriorProductStore(tmp_path / "priors.json")
    record = store.get_or_create(_scope(), seed_alpha=0.2)

    assert record.seed_alpha == 0.2

    updated = store.update(
        _scope(),
        accepted=True,
        temptation_event=True,
        steering_applied=True,
        alpha_delta=0.1,
        decay=0.5,
    )

    assert updated.seed_alpha == pytest.approx(0.2)
    assert updated.accepted_case_count == 1
    assert updated.successful_streak == 1
    assert updated.temptation_events == 1
    assert updated.steering_applications == 1

    rejected = store.update(_scope(), accepted=False, alpha_delta=0.05, decay=1.0)

    assert rejected.seed_alpha == pytest.approx(0.15)
    assert rejected.rejected_case_count == 1
    assert rejected.successful_streak == 0


def test_decoder_prior_store_rejects_incompatible_scopes() -> None:
    store = DecoderPriorProductStore("unused.json")

    with pytest.raises(IncompatibleDecoderPriorScope, match="tokenizer_id"):
        store.require_compatible(_scope(), _scope(tokenizer_id="other-tokenizer"))

    with pytest.raises(IncompatibleDecoderPriorScope, match="layer"):
        _scope().assert_compatible(_scope(layer=13))


def test_decoder_prior_store_lookup_returns_runtime_metadata_on_hit(tmp_path: Path) -> None:
    store = DecoderPriorProductStore(tmp_path / "priors.json")
    store.put(DecoderPriorRecord(scope=_scope(), seed_alpha=0.37, metadata={"dialect": "python"}))

    selection = store.lookup(_scope())
    payload = selection.to_json()

    assert isinstance(selection, DecoderPriorSelection)
    assert selection.hit is True
    assert selection.miss is False
    assert selection.scope_json == _scope().to_json()
    assert selection.record_json is not None
    assert selection.seed_alpha == pytest.approx(0.37)
    assert payload["hit"] is True
    assert payload["miss"] is False
    assert payload["scope"] == _scope().to_json()
    assert payload["seed_alpha"] == pytest.approx(0.37)
    assert payload["record"]["metadata"] == {"dialect": "python"}


def test_decoder_prior_store_lookup_reports_closed_miss() -> None:
    store = DecoderPriorProductStore("unused.json")

    selection = store.lookup(_scope())
    payload = selection.to_json()

    assert selection.hit is False
    assert selection.miss is True
    assert selection.seed_alpha is None
    assert payload == {
        "hit": False,
        "miss": True,
        "reason": "miss",
        "scope": _scope().to_json(),
    }


def test_decoder_prior_store_lookup_refuses_incompatible_scope() -> None:
    store = DecoderPriorProductStore("unused.json")
    store.put(DecoderPriorRecord(scope=_scope(tokenizer_id="other-tokenizer"), seed_alpha=0.9))

    selection = store.lookup(_scope())
    payload = selection.to_json()

    assert selection.hit is False
    assert selection.miss is True
    assert selection.seed_alpha is None
    assert selection.reason == "incompatible_scope: tokenizer_id"
    assert payload["reason"] == "incompatible_scope: tokenizer_id"
    assert payload["scope"] == _scope().to_json()
    assert "record" not in payload
    assert "seed_alpha" not in payload


def test_decoder_prior_store_loads_and_saves_json(tmp_path: Path) -> None:
    path = tmp_path / "state" / "decoder_priors.json"
    store = DecoderPriorProductStore(path)
    record = DecoderPriorRecord(scope=_scope(), seed_alpha=0.4, metadata={"dialect": "python"})
    record.update_stats(accepted=True)
    store.put(record)
    store.save()

    loaded = DecoderPriorProductStore.load(path)
    loaded_record = loaded.get(_scope())

    assert loaded_record is not None
    assert loaded_record.seed_alpha == pytest.approx(0.43)
    assert loaded_record.accepted_case_count == 1
    assert loaded_record.metadata == {"dialect": "python"}
    assert loaded.to_json()["version"] == 1

    selection = loaded.select(_scope())

    assert selection.hit is True
    assert selection.seed_alpha == pytest.approx(0.43)


def test_decoder_prior_scope_accepts_runtime_style_mapping() -> None:
    scope = DecoderPriorScope.from_mapping(
        {
            "model_id": "gemma-e2b",
            "tokenizer_id": "gemma-tokenizer",
            "adapter_family": "gemma",
            "kv_target_layer": 12,
            "method": "patch_target",
            "steering": "v1",
        }
    )

    assert scope.layer == 12
    assert scope.task_type == "patch_target"
    assert scope.steering_version == "v1"
