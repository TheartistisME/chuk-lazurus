"""Manifest-backed residual/KV sidecar references for David materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


SIDECAR_SCHEMA_VERSION = 1
MAX_INLINE_ARRAY_VALUES = 4096
REFERENCE_KINDS = {"boundary_residual", "residual_stream", "kv_cache"}


class ResidualStoreError(ValueError):
    """Raised when a residual/KV sidecar manifest is not safe to load."""


@dataclass(frozen=True)
class ResidualReference:
    kind: str
    layer: int | None
    dtype: str | None = None
    shape: tuple[int, ...] = ()
    uri: str | None = None
    inline_values: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "ResidualReference":
        kind = str(payload.get("kind") or "")
        if kind not in REFERENCE_KINDS:
            raise ResidualStoreError(f"unsupported sidecar ref kind: {kind or '<missing>'}")
        layer = _optional_int(payload.get("layer"), "layer")
        shape = _shape_tuple(payload.get("shape"))
        inline_values = _inline_values(payload.get("inline_values"))
        return cls(
            kind=kind,
            layer=layer,
            dtype=None if payload.get("dtype") is None else str(payload["dtype"]),
            shape=shape,
            uri=None if payload.get("uri") is None else str(payload["uri"]),
            inline_values=inline_values,
            metadata=_dict(payload.get("metadata")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "layer": self.layer,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "uri": self.uri,
            "inline_values": list(self.inline_values),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResidualSidecarManifest:
    schema_version: int
    artifact_id: str
    memory_family: str
    model_id: str
    tokenizer_id: str
    model_revision: str | None
    adapter_family: str | None
    insertion_family: str
    boundary_layer: int | None = None
    residual_layer: int | None = None
    kv_source_layer: int | None = None
    kv_target_layer: int | None = None
    hidden_size: int | None = None
    refs: tuple[ResidualReference, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    manifest_path: str | None = None

    @classmethod
    def from_json(cls, payload: Mapping[str, Any], *, manifest_path: Path | None = None) -> "ResidualSidecarManifest":
        schema_version = _required_int(payload, "schema_version")
        if schema_version != SIDECAR_SCHEMA_VERSION:
            raise ResidualStoreError(f"unsupported sidecar schema_version: {schema_version}")
        refs_payload = payload.get("refs")
        if not isinstance(refs_payload, list) or not refs_payload:
            raise ResidualStoreError("sidecar manifest requires non-empty refs")
        refs = tuple(ResidualReference.from_json(_mapping(item, "ref")) for item in refs_payload)
        artifact_id = _required_str(payload, "artifact_id")
        return cls(
            schema_version=schema_version,
            artifact_id=artifact_id,
            memory_family=_required_str(payload, "memory_family"),
            model_id=_required_str(payload, "model_id"),
            tokenizer_id=_required_str(payload, "tokenizer_id"),
            model_revision=_optional_str(payload.get("model_revision")),
            adapter_family=_optional_str(payload.get("adapter_family")),
            insertion_family=_required_str(payload, "insertion_family"),
            boundary_layer=_optional_int(payload.get("boundary_layer"), "boundary_layer"),
            residual_layer=_optional_int(payload.get("residual_layer"), "residual_layer"),
            kv_source_layer=_optional_int(payload.get("kv_source_layer"), "kv_source_layer"),
            kv_target_layer=_optional_int(payload.get("kv_target_layer"), "kv_target_layer"),
            hidden_size=_optional_int(payload.get("hidden_size"), "hidden_size"),
            refs=refs,
            provenance=_dict(payload.get("provenance")),
            manifest_path=None if manifest_path is None else str(manifest_path),
        )

    def scope(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "model_revision": self.model_revision,
            "adapter_family": self.adapter_family,
            "insertion_family": self.insertion_family,
            "memory_family": self.memory_family,
            "boundary_layer": self.boundary_layer,
            "residual_layer": self.residual_layer,
            "kv_source_layer": self.kv_source_layer,
            "kv_target_layer": self.kv_target_layer,
            "hidden_size": self.hidden_size,
        }

    def to_plan_ref(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "manifest_path": self.manifest_path,
            "memory_family": self.memory_family,
            "scope": self.scope(),
            "refs": [ref.to_json() for ref in self.refs],
            "provenance": dict(self.provenance),
        }


class ResidualStore:
    """Loads small manifest metadata for captured residual/KV sidecars."""

    def load_manifest(self, manifest: str | Path | Mapping[str, Any]) -> ResidualSidecarManifest:
        if isinstance(manifest, Mapping):
            return ResidualSidecarManifest.from_json(manifest)
        path = Path(manifest).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ResidualStoreError(f"cannot read sidecar manifest {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ResidualStoreError(f"invalid sidecar manifest JSON {path}: {exc}") from exc
        return ResidualSidecarManifest.from_json(_mapping(payload, "manifest"), manifest_path=path)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ResidualStoreError(f"sidecar manifest requires string {key}")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ResidualStoreError("optional sidecar string fields must be non-empty strings")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ResidualStoreError(f"sidecar manifest requires integer {key}")
    return value


def _optional_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ResidualStoreError(f"sidecar manifest field {key} must be an integer")
    return value


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ResidualStoreError("sidecar ref shape must be a list")
    shape: list[int] = []
    for dim in value:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
            raise ResidualStoreError("sidecar ref shape dimensions must be non-negative integers")
        shape.append(dim)
    return tuple(shape)


def _inline_values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ResidualStoreError("inline_values must be a list")
    count = _nested_value_count(value)
    if count > MAX_INLINE_ARRAY_VALUES:
        raise ResidualStoreError(f"inline_values exceed test-safe limit: {count} > {MAX_INLINE_ARRAY_VALUES}")
    return tuple(value)


def _nested_value_count(value: Any) -> int:
    if isinstance(value, list):
        return sum(_nested_value_count(item) for item in value)
    return 1


def _dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ResidualStoreError("sidecar metadata fields must be objects")
    return {str(key): item for key, item in value.items()}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResidualStoreError(f"sidecar {name} must be an object")
    return value
