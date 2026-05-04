from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chuk_lazarus.david.residual_store import LoadedResidualTensor, ResidualStore, ResidualStoreError


def _manifest_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_id": "a1",
        "memory_family": "task",
        "model_id": "m",
        "tokenizer_id": "t",
        "model_revision": "r",
        "adapter_family": "gemma",
        "insertion_family": "full_attention",
        "boundary_layer": 3,
        "hidden_size": 2,
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
    payload.update(overrides)
    return payload


def test_residual_store_loads_manifest_refs_from_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            _manifest_payload(hidden_size=None)
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


def test_residual_store_loads_manifest_relative_npy(tmp_path: Path) -> None:
    tensor_dir = tmp_path / "tensors"
    tensor_dir.mkdir()
    np.save(tensor_dir / "boundary.npy", np.array([[1.0, 2.0]], dtype=np.float32))
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            _manifest_payload(
                refs=[
                    {
                        "kind": "boundary_residual",
                        "layer": 3,
                        "dtype": "float32",
                        "shape": [1, 2],
                        "uri": "tensors/boundary.npy",
                    }
                ]
            )
        ),
        encoding="utf-8",
    )

    loaded = ResidualStore().load_tensor(path)

    assert isinstance(loaded, LoadedResidualTensor)
    assert loaded.kind == "boundary_residual"
    assert loaded.layer == 3
    assert loaded.dtype == "float32"
    assert loaded.shape == (1, 2)
    assert loaded.matches_manifest_hidden_size()
    assert loaded.has_hidden_size(2)
    np.testing.assert_array_equal(loaded.array, np.array([[1.0, 2.0]], dtype=np.float32))


def test_residual_store_loads_inline_values() -> None:
    payload = _manifest_payload(
        refs=[
            {
                "kind": "boundary_residual",
                "layer": 3,
                "dtype": "float16",
                "shape": [2, 2],
                "inline_values": [[1.0, 2.0], [3.0, 4.0]],
            }
        ]
    )

    loaded = ResidualStore().load_tensor(payload)

    assert loaded.dtype == "float16"
    assert loaded.shape == (2, 2)
    assert loaded.array.dtype == np.dtype("float16")


@pytest.mark.parametrize("uri", ["../escape.npy", r"..\escape.npy", r"C:\escape.npy", None])
def test_residual_store_rejects_unsafe_npy_paths(tmp_path: Path, uri: str | None) -> None:
    unsafe_uri = str(tmp_path / "escape.npy") if uri is None else uri
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            _manifest_payload(
                refs=[
                    {
                        "kind": "boundary_residual",
                        "layer": 3,
                        "dtype": "float32",
                        "shape": [1, 2],
                        "uri": unsafe_uri,
                    }
                ]
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ResidualStoreError, match="manifest-relative|escape"):
        ResidualStore().load_tensor(path)


def test_residual_store_rejects_shape_mismatch(tmp_path: Path) -> None:
    np.save(tmp_path / "boundary.npy", np.array([[1.0], [2.0]], dtype=np.float32))
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            _manifest_payload(
                refs=[
                    {
                        "kind": "boundary_residual",
                        "layer": 3,
                        "dtype": "float32",
                        "shape": [1, 2],
                        "uri": "boundary.npy",
                    }
                ]
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ResidualStoreError, match="shape mismatch"):
        ResidualStore().load_tensor(path)


def test_residual_store_rejects_unsupported_dtype() -> None:
    payload = _manifest_payload(
        refs=[
            {
                "kind": "boundary_residual",
                "layer": 3,
                "dtype": "int64",
                "shape": [1, 2],
                "inline_values": [[1, 2]],
            }
        ]
    )

    with pytest.raises(ResidualStoreError, match="unsupported.*dtype"):
        ResidualStore().load_tensor(payload)


def test_residual_store_rejects_size_cap(tmp_path: Path) -> None:
    np.save(tmp_path / "boundary.npy", np.array([[1.0, 2.0]], dtype=np.float32))
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            _manifest_payload(
                refs=[
                    {
                        "kind": "boundary_residual",
                        "layer": 3,
                        "dtype": "float32",
                        "shape": [1, 2],
                        "uri": "boundary.npy",
                    }
                ]
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ResidualStoreError, match="size cap"):
        ResidualStore(max_tensor_bytes=4).load_tensor(path)


def test_residual_store_rejects_pickle_object_npy(tmp_path: Path) -> None:
    np.save(tmp_path / "object.npy", np.array([{"unsafe": "object"}], dtype=object))
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            _manifest_payload(
                refs=[
                    {
                        "kind": "boundary_residual",
                        "layer": 3,
                        "dtype": "float32",
                        "shape": [1],
                        "uri": "object.npy",
                    }
                ]
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ResidualStoreError, match="cannot safely load"):
        ResidualStore().load_tensor(path)
