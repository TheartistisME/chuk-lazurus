from __future__ import annotations

from chuk_lazarus.david.config import AdapterSessionMetadata
from chuk_lazarus.david.residual_replay_bridge import residual_sidecar_metadata
from chuk_lazarus.david.residual_store import ResidualReference, ResidualSidecarManifest


def test_residual_sidecar_bridge_accepts_matching_metadata() -> None:
    metadata = residual_sidecar_metadata(
        adapter=_adapter(),
        sidecar=_sidecar(),
        consumer=_consumer(),
        memory_family="task",
    )

    assert metadata["accepted"] is True
    assert metadata["refused"] is False
    assert metadata["eligible_for_tensor_replay"] is True
    assert metadata["tensor_replay_applied"] is False
    assert metadata["tensors_loaded"] is False
    assert metadata["adapter_scope"]["hidden_size"] == 4
    assert metadata["sidecar_scope"]["layer"] == 7
    assert metadata["consumer"]["consumer_id"] == "unit-residual-replay"
    assert metadata["residual_refs"] == [
        {
            "kind": "boundary_residual",
            "layer": 7,
            "dtype": "float32",
            "shape": [1, 4],
            "uri": "sidecar://a1/boundary",
            "inline_value_count": 0,
            "metadata": {"span_id": "hot-1"},
        }
    ]


def test_residual_sidecar_bridge_refuses_model_mismatch() -> None:
    metadata = residual_sidecar_metadata(
        adapter=_adapter(model_id="model-b"),
        sidecar=_sidecar(),
        consumer=_consumer(),
        memory_family="task",
    )

    assert metadata["refused"] is True
    assert "model_id mismatch" in metadata["reason"]


def test_residual_sidecar_bridge_refuses_tokenizer_mismatch() -> None:
    metadata = residual_sidecar_metadata(
        adapter=_adapter(tokenizer_id="tokenizer-b"),
        sidecar=_sidecar(),
        consumer=_consumer(),
        memory_family="task",
    )

    assert metadata["refused"] is True
    assert "tokenizer_id mismatch" in metadata["reason"]


def test_residual_sidecar_bridge_refuses_layer_mismatch() -> None:
    metadata = residual_sidecar_metadata(
        adapter=_adapter(boundary_layer=8),
        sidecar=_sidecar(),
        consumer=_consumer(),
        memory_family="task",
    )

    assert metadata["refused"] is True
    assert "layer mismatch" in metadata["reason"]
    assert "ref layer mismatch" in metadata["reason"]


def test_residual_sidecar_bridge_refuses_memory_family_mismatch() -> None:
    metadata = residual_sidecar_metadata(
        adapter=_adapter(),
        sidecar=_sidecar(memory_family="user"),
        consumer=_consumer(),
        memory_family="task",
    )

    assert metadata["refused"] is True
    assert "memory_family mismatch" in metadata["reason"]


def test_residual_sidecar_bridge_refuses_missing_evidence() -> None:
    metadata = residual_sidecar_metadata(
        adapter=_adapter(hidden_size=None, boundary_layer=None),
        sidecar=_sidecar(hidden_size=None, boundary_layer=None, refs=()),
        consumer={"consumer_id": "empty-consumer"},
        memory_family="task",
    )

    assert metadata["refused"] is True
    assert metadata["eligible_for_tensor_replay"] is False
    assert "adapter missing layer evidence" in metadata["reason"]
    assert "adapter missing hidden_size evidence" in metadata["reason"]
    assert "sidecar missing layer evidence" in metadata["reason"]
    assert "sidecar missing hidden_size evidence" in metadata["reason"]
    assert "missing residual replay refs" in metadata["reason"]
    assert "lacks materialization.replay.residual_stream.v1" in metadata["reason"]


def _adapter(**overrides: object) -> AdapterSessionMetadata:
    values = {
        "model_id": "model-a",
        "tokenizer_id": "tokenizer-a",
        "model_revision": "rev-a",
        "adapter_family": "family-a",
        "hidden_size": 4,
        "boundary_layer": 7,
        "insertion_family": "kv_direct",
    }
    values.update(overrides)
    return AdapterSessionMetadata(**values)


def _sidecar(**overrides: object) -> ResidualSidecarManifest:
    values = {
        "schema_version": 1,
        "artifact_id": "hot-window-1",
        "memory_family": "task",
        "model_id": "model-a",
        "tokenizer_id": "tokenizer-a",
        "model_revision": "rev-a",
        "adapter_family": "family-a",
        "insertion_family": "kv_direct",
        "boundary_layer": 7,
        "residual_layer": None,
        "hidden_size": 4,
        "refs": (
            ResidualReference(
                kind="boundary_residual",
                layer=7,
                dtype="float32",
                shape=(1, 4),
                uri="sidecar://a1/boundary",
                metadata={"span_id": "hot-1"},
            ),
        ),
    }
    values.update(overrides)
    return ResidualSidecarManifest(**values)


def _consumer() -> dict[str, object]:
    return {
        "consumer_id": "unit-residual-replay",
        "capabilities": ["materialization.replay.residual_stream.v1"],
    }
