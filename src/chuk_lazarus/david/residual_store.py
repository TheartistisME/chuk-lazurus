"""Manifest-backed residual/KV sidecar references for David materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

import numpy as np


SIDECAR_SCHEMA_VERSION = 1
MAX_INLINE_ARRAY_VALUES = 4096
DEFAULT_MAX_TENSOR_BYTES = 16 * 1024 * 1024
REFERENCE_KINDS = {"boundary_residual", "residual_stream", "kv_cache"}
SUPPORTED_TENSOR_DTYPES = {"float16", "bfloat16", "float32"}


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
class LoadedResidualTensor:
    kind: str
    layer: int | None
    dtype: str
    shape: tuple[int, ...]
    array: Any
    ref: ResidualReference
    manifest: "ResidualSidecarManifest"

    def has_hidden_size(self, hidden_size: int) -> bool:
        return bool(self.shape) and self.shape[-1] == hidden_size

    def matches_manifest_hidden_size(self) -> bool:
        if self.manifest.hidden_size is None:
            return True
        return self.has_hidden_size(self.manifest.hidden_size)


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

    def __init__(self, *, max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES) -> None:
        if (
            not isinstance(max_tensor_bytes, int)
            or isinstance(max_tensor_bytes, bool)
            or max_tensor_bytes <= 0
        ):
            raise ResidualStoreError("max_tensor_bytes must be a positive integer")
        self.max_tensor_bytes = max_tensor_bytes

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

    def load_tensor(
        self,
        manifest: ResidualSidecarManifest | str | Path | Mapping[str, Any],
        ref: ResidualReference | int = 0,
        *,
        max_tensor_bytes: int | None = None,
    ) -> LoadedResidualTensor:
        loaded_manifest = (
            manifest if isinstance(manifest, ResidualSidecarManifest) else self.load_manifest(manifest)
        )
        loaded_ref = _select_ref(loaded_manifest, ref)
        dtype_name = _validated_dtype_name(loaded_ref)
        np_dtype = _numpy_dtype(dtype_name)
        has_inline = bool(loaded_ref.inline_values)
        has_uri = loaded_ref.uri is not None
        if has_inline and has_uri:
            raise ResidualStoreError("sidecar ref must use only one tensor source")
        if has_inline:
            array = np.asarray(loaded_ref.inline_values, dtype=np_dtype)
        elif has_uri:
            array = self._load_npy_ref(loaded_manifest, loaded_ref)
        else:
            raise ResidualStoreError("sidecar ref requires inline_values or manifest-relative uri")
        _validate_tensor_array(
            array,
            loaded_ref,
            dtype_name=dtype_name,
            max_tensor_bytes=self.max_tensor_bytes if max_tensor_bytes is None else max_tensor_bytes,
        )
        return LoadedResidualTensor(
            kind=loaded_ref.kind,
            layer=loaded_ref.layer,
            dtype=dtype_name,
            shape=tuple(int(dim) for dim in array.shape),
            array=array,
            ref=loaded_ref,
            manifest=loaded_manifest,
        )

    def _load_npy_ref(
        self, manifest: ResidualSidecarManifest, ref: ResidualReference
    ) -> Any:
        path = _resolve_manifest_relative_ref(manifest, ref)
        if path.suffix != ".npy":
            raise ResidualStoreError("sidecar tensor uri must point to a .npy file")
        try:
            return np.load(path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise ResidualStoreError(f"cannot safely load sidecar tensor {ref.uri}: {exc}") from exc


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


def _select_ref(
    manifest: ResidualSidecarManifest, ref: ResidualReference | int
) -> ResidualReference:
    if isinstance(ref, ResidualReference):
        if ref not in manifest.refs:
            raise ResidualStoreError("sidecar ref does not belong to manifest")
        return ref
    if not isinstance(ref, int) or isinstance(ref, bool):
        raise ResidualStoreError("sidecar ref selector must be a ref or integer index")
    try:
        return manifest.refs[ref]
    except IndexError as exc:
        raise ResidualStoreError(f"sidecar ref index out of range: {ref}") from exc


def _validated_dtype_name(ref: ResidualReference) -> str:
    if ref.dtype is None:
        raise ResidualStoreError("sidecar tensor ref requires dtype")
    dtype_name = ref.dtype.strip().lower()
    if dtype_name not in SUPPORTED_TENSOR_DTYPES:
        raise ResidualStoreError(f"unsupported sidecar tensor dtype: {ref.dtype}")
    return dtype_name


def _numpy_dtype(dtype_name: str) -> Any:
    if dtype_name == "bfloat16":
        try:
            return np.dtype("bfloat16")
        except TypeError as exc:
            raise ResidualStoreError("bfloat16 sidecar tensors are not supported by this numpy build") from exc
    return np.dtype(dtype_name)


def _resolve_manifest_relative_ref(
    manifest: ResidualSidecarManifest, ref: ResidualReference
) -> Path:
    if not manifest.manifest_path:
        raise ResidualStoreError("sidecar uri refs require a manifest file path")
    if not ref.uri:
        raise ResidualStoreError("sidecar ref requires uri")
    candidate = Path(ref.uri)
    windows_candidate = PureWindowsPath(ref.uri)
    if candidate.is_absolute() or windows_candidate.is_absolute():
        raise ResidualStoreError("sidecar tensor uri must be manifest-relative")
    if any(part == ".." for part in candidate.parts) or any(
        part == ".." for part in windows_candidate.parts
    ):
        raise ResidualStoreError("sidecar tensor uri cannot escape the manifest directory")
    base = Path(manifest.manifest_path).parent.resolve()
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ResidualStoreError("sidecar tensor uri cannot escape the manifest directory") from exc
    return resolved


def _validate_tensor_array(
    array: Any,
    ref: ResidualReference,
    *,
    dtype_name: str,
    max_tensor_bytes: int,
) -> None:
    if (
        not isinstance(max_tensor_bytes, int)
        or isinstance(max_tensor_bytes, bool)
        or max_tensor_bytes <= 0
    ):
        raise ResidualStoreError("max_tensor_bytes must be a positive integer")
    expected_dtype = _numpy_dtype(dtype_name)
    if array.dtype != expected_dtype:
        raise ResidualStoreError(
            f"sidecar tensor dtype mismatch: manifest {dtype_name}, tensor {array.dtype}"
        )
    if not ref.shape:
        raise ResidualStoreError("sidecar tensor ref requires shape")
    if tuple(int(dim) for dim in array.shape) != ref.shape:
        raise ResidualStoreError(
            f"sidecar tensor shape mismatch: manifest {ref.shape}, tensor {tuple(array.shape)}"
        )
    if int(array.nbytes) > max_tensor_bytes:
        raise ResidualStoreError(
            f"sidecar tensor exceeds size cap: {array.nbytes} > {max_tensor_bytes}"
        )


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
