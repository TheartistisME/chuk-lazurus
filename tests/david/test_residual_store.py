from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chuk_lazarus.david.residual_store import (
    LoadedResidualTensor,
    ResidualSidecarCatalog,
    ResidualSidecarCatalogEntry,
    ResidualSidecarCatalogKey,
    ResidualStore,
    ResidualStoreError,
)


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


def test_sidecar_catalog_adds_saves_loads_and_resolves_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "boundary-sidecar.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog = ResidualSidecarCatalog(base_path=tmp_path)
    catalog.add_entry(
        ResidualSidecarCatalogEntry(
            key=ResidualSidecarCatalogKey(
                workspace_id="workspace-a",
                session_id="session-a",
                window_id="window-a",
                artifact_family="boundary_residual",
            ),
            manifest_ref="boundary-sidecar.json",
            scope={"model_id": "m", "tokenizer_id": "t", "insertion_family": "full_attention"},
        )
    )
    catalog.save(catalog_path)

    loaded = ResidualSidecarCatalog.load(catalog_path)
    resolved = loaded.resolve(
        workspace_id="workspace-a",
        session_id="session-a",
        window_id="window-a",
        artifact_family="boundary_residual",
    )

    assert resolved.ready is True
    assert resolved.action == "use_sidecar_metadata"
    assert resolved.manifest is not None
    assert resolved.manifest.artifact_id == "a1"
    assert [ref.kind for ref in resolved.refs] == ["boundary_residual"]
    assert resolved.metadata_only is True
    assert resolved.kv_replay_ready is False


def test_sidecar_catalog_resolves_window_metadata_by_family(tmp_path: Path) -> None:
    manifest_path = tmp_path / "boundary-sidecar.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    catalog = ResidualSidecarCatalog(base_path=tmp_path)
    entries = catalog.add_from_window_metadata(
        {
            "window_id": "hot-window-1",
            "sidecar_refs": {"boundary_residual": "boundary-sidecar.json"},
        },
        workspace_id="workspace-a",
        session_id="session-a",
        scope={"model_id": "m", "tokenizer_id": "t"},
    )

    resolved = catalog.resolve(
        workspace_id="workspace-a",
        session_id="session-a",
        window_id="hot-window-1",
        artifact_family="boundary_residual",
    )

    assert len(entries) == 1
    assert resolved.ready is True
    assert resolved.refs[0].layer == 3


def test_sidecar_catalog_marks_scope_mismatch_stale(tmp_path: Path) -> None:
    manifest_path = tmp_path / "boundary-sidecar.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    catalog = ResidualSidecarCatalog(base_path=tmp_path)
    catalog.add_manifest(
        manifest_path,
        workspace_id="workspace-a",
        session_id="session-a",
        window_id="hot-window-1",
        artifact_family="boundary_residual",
    )

    resolved = catalog.resolve(
        workspace_id="workspace-a",
        session_id="session-a",
        window_id="hot-window-1",
        artifact_family="boundary_residual",
        expected_scope={"model_id": "different-model"},
    )

    assert resolved.ready is False
    assert resolved.stale is True
    assert resolved.action == "recapture_sidecar"
    assert "scope mismatch" in resolved.reason


def test_sidecar_catalog_exposes_missing_sidecar_action_from_readiness() -> None:
    catalog = ResidualSidecarCatalog()
    catalog.add_from_readiness_metadata(
        {
            "workspace_root": "workspace-a",
            "adapter_scope": {"model_id": "m", "tokenizer_id": "t"},
            "artifact_families": {
                "boundary_residual": {
                    "status": "missing",
                    "capture_action": "recapture_hot_windows",
                    "missing_window_ids": ["hot-window-1"],
                }
            },
        },
        session_id="session-a",
    )

    resolved = catalog.resolve(
        workspace_id="workspace-a",
        session_id="session-a",
        window_id="hot-window-1",
        artifact_family="boundary_residual",
    )

    assert resolved.ready is False
    assert resolved.missing is True
    assert resolved.action == "recapture_hot_windows"
    assert "missing boundary_residual sidecar" in resolved.reason


def test_sidecar_catalog_keeps_kv_sidecars_metadata_only(tmp_path: Path) -> None:
    kv_manifest = _manifest_payload(
        artifact_id="kv-a",
        kv_source_layer=6,
        kv_target_layer=7,
        refs=[
            {
                "kind": "kv_cache",
                "layer": 7,
                "dtype": "float16",
                "shape": [2, 4, 8],
                "metadata": {"layout": "metadata-only"},
            }
        ],
    )
    manifest_path = tmp_path / "kv-sidecar.json"
    manifest_path.write_text(json.dumps(kv_manifest), encoding="utf-8")
    catalog = ResidualSidecarCatalog(base_path=tmp_path)
    catalog.add_from_window_metadata(
        {
            "window_id": "hot-window-1",
            "sidecar_refs": {"kv_cache": "kv-sidecar.json"},
        },
        workspace_id="workspace-a",
        session_id="session-a",
        scope={"model_id": "m", "tokenizer_id": "t", "kv_target_layer": 7},
    )

    resolved = catalog.resolve(
        workspace_id="workspace-a",
        session_id="session-a",
        window_id="hot-window-1",
        artifact_family="kv_cache",
    )

    assert resolved.ready is True
    assert resolved.refs[0].kind == "kv_cache"
    assert resolved.refs[0].uri is None
    assert resolved.metadata_only is True
    assert resolved.kv_replay_ready is False
