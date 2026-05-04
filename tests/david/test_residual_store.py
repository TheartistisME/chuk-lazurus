from __future__ import annotations

import json
from pathlib import Path

import pytest

from chuk_lazarus.david.residual_store import ResidualStore, ResidualStoreError


def test_residual_store_loads_manifest_refs_from_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_id": "a1",
                "memory_family": "task",
                "model_id": "m",
                "tokenizer_id": "t",
                "model_revision": "r",
                "adapter_family": "gemma",
                "insertion_family": "full_attention",
                "boundary_layer": 3,
                "refs": [
                    {
                        "kind": "boundary_residual",
                        "layer": 3,
                        "dtype": "float32",
                        "shape": [1, 2],
                        "inline_values": [[1.0, 2.0]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = ResidualStore().load_manifest(path)

    assert manifest.artifact_id == "a1"
    assert manifest.scope()["boundary_layer"] == 3
    assert manifest.refs[0].inline_values == ([1.0, 2.0],)
    assert manifest.to_plan_ref()["refs"][0]["shape"] == [1, 2]


def test_residual_store_rejects_large_inline_arrays() -> None:
    payload = {
        "schema_version": 1,
        "artifact_id": "too-big",
        "memory_family": "task",
        "model_id": "m",
        "tokenizer_id": "t",
        "insertion_family": "full_attention",
        "refs": [{"kind": "boundary_residual", "layer": 1, "inline_values": list(range(4097))}],
    }

    with pytest.raises(ResidualStoreError, match="test-safe limit"):
        ResidualStore().load_manifest(payload)
