"""Manifest-backed residual/KV sidecar references for David materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

import numpy as np


SIDECAR_SCHEMA_VERSION = 1
SIDECAR_CATALOG_SCHEMA = "david.residual_sidecar_catalog.v1"
SIDECAR_CATALOG_VERSION = 1
MAX_INLINE_ARRAY_VALUES = 4096
DEFAULT_MAX_TENSOR_BYTES = 16 * 1024 * 1024
REFERENCE_KINDS = {"boundary_residual", "residual_stream", "kv_cache"}
CATALOG_ARTIFACT_FAMILIES = REFERENCE_KINDS
SUPPORTED_TENSOR_DTYPES = {"float16", "bfloat16", "float32"}
CATALOG_SCOPE_KEYS = (
    "model_id",
    "tokenizer_id",
    "model_revision",
    "adapter_family",
    "insertion_family",
    "memory_family",
    "boundary_layer",
    "residual_layer",
    "kv_source_layer",
    "kv_target_layer",
    "hidden_size",
)


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
class ResidualSidecarCatalogKey:
    workspace_id: str
    session_id: str
    window_id: str
    artifact_family: str

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "ResidualSidecarCatalogKey":
        return cls(
            workspace_id=_required_str(payload, "workspace_id"),
            session_id=_required_str(payload, "session_id"),
            window_id=_required_str(payload, "window_id"),
            artifact_family=_catalog_artifact_family(payload.get("artifact_family")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "window_id": self.window_id,
            "artifact_family": self.artifact_family,
        }


@dataclass(frozen=True)
class ResidualSidecarCatalogEntry:
    key: ResidualSidecarCatalogKey
    manifest_ref: str | None
    artifact_id: str | None = None
    memory_family: str | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    ready: bool = True
    stale: bool = False
    status: str = "ready"
    capture_action: str = "capture_sidecar"
    reason: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "ResidualSidecarCatalogEntry":
        key = ResidualSidecarCatalogKey.from_json(_mapping(payload.get("key"), "catalog key"))
        return cls(
            key=key,
            manifest_ref=_optional_str(payload.get("manifest_ref")),
            artifact_id=_optional_str(payload.get("artifact_id")),
            memory_family=_optional_str(payload.get("memory_family")),
            scope=_dict(payload.get("scope")),
            ready=bool(payload.get("ready", True)),
            stale=bool(payload.get("stale", False)),
            status=str(payload.get("status") or "ready"),
            capture_action=str(payload.get("capture_action") or "capture_sidecar"),
            reason=_optional_str(payload.get("reason")),
            provenance=_dict(payload.get("provenance")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key.to_json(),
            "manifest_ref": self.manifest_ref,
            "artifact_id": self.artifact_id,
            "memory_family": self.memory_family,
            "scope": dict(self.scope),
            "ready": self.ready,
            "stale": self.stale,
            "status": self.status,
            "capture_action": self.capture_action,
            "reason": self.reason,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ResidualSidecarCatalogResolution:
    key: ResidualSidecarCatalogKey
    ready: bool
    action: str
    reason: str
    entry: ResidualSidecarCatalogEntry | None = None
    manifest: "ResidualSidecarManifest" | None = None
    refs: tuple[ResidualReference, ...] = ()
    missing: bool = False
    stale: bool = False
    metadata_only: bool = True
    kv_replay_ready: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key.to_json(),
            "ready": self.ready,
            "action": self.action,
            "reason": self.reason,
            "entry": None if self.entry is None else self.entry.to_json(),
            "manifest": None if self.manifest is None else self.manifest.to_plan_ref(),
            "refs": [ref.to_json() for ref in self.refs],
            "missing": self.missing,
            "stale": self.stale,
            "metadata_only": self.metadata_only,
            "kv_replay_ready": self.kv_replay_ready,
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


class ResidualSidecarCatalog:
    """Metadata catalog for residual/KV sidecar manifests.

    The catalog resolves manifest metadata only. Tensor loading and KV replay stay
    behind explicit materialization paths.
    """

    def __init__(
        self,
        *,
        entries: list[ResidualSidecarCatalogEntry] | None = None,
        base_path: str | Path | None = None,
        residual_store: ResidualStore | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.base_path = None if base_path is None else Path(base_path).expanduser().resolve()
        self.residual_store = residual_store or ResidualStore()
        self.provenance = _dict(provenance)
        self._entries: dict[tuple[str, str, str, str], ResidualSidecarCatalogEntry] = {}
        for entry in entries or []:
            self.add_entry(entry)

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, Any],
        *,
        catalog_path: Path | None = None,
        residual_store: ResidualStore | None = None,
    ) -> "ResidualSidecarCatalog":
        if payload.get("schema") != SIDECAR_CATALOG_SCHEMA:
            raise ResidualStoreError(f"unsupported sidecar catalog schema: {payload.get('schema')!r}")
        version = _required_int(payload, "version")
        if version != SIDECAR_CATALOG_VERSION:
            raise ResidualStoreError(f"unsupported sidecar catalog version: {version}")
        entries_payload = payload.get("entries")
        if not isinstance(entries_payload, list):
            raise ResidualStoreError("sidecar catalog entries must be a list")
        base_path = payload.get("base_path")
        if base_path is None and catalog_path is not None:
            base_path = str(catalog_path.parent)
        return cls(
            entries=[
                ResidualSidecarCatalogEntry.from_json(_mapping(item, "catalog entry"))
                for item in entries_payload
            ],
            base_path=None if base_path is None else str(base_path),
            residual_store=residual_store,
            provenance=_dict(payload.get("provenance")),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        residual_store: ResidualStore | None = None,
    ) -> "ResidualSidecarCatalog":
        catalog_path = Path(path).expanduser().resolve()
        try:
            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ResidualStoreError(f"cannot read sidecar catalog {catalog_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ResidualStoreError(f"invalid sidecar catalog JSON {catalog_path}: {exc}") from exc
        return cls.from_json(
            _mapping(payload, "catalog"),
            catalog_path=catalog_path,
            residual_store=residual_store,
        )

    def save(self, path: str | Path) -> None:
        catalog_path = Path(path).expanduser().resolve()
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SIDECAR_CATALOG_SCHEMA,
            "version": SIDECAR_CATALOG_VERSION,
            "base_path": None if self.base_path is None else str(self.base_path),
            "entries": [entry.to_json() for entry in self.entries()],
            "provenance": dict(self.provenance),
        }

    def entries(self) -> list[ResidualSidecarCatalogEntry]:
        return [
            self._entries[key]
            for key in sorted(self._entries)
        ]

    def add_entry(self, entry: ResidualSidecarCatalogEntry) -> ResidualSidecarCatalogEntry:
        self._entries[_catalog_key_tuple(entry.key)] = entry
        return entry

    def add_manifest(
        self,
        manifest: str | Path | Mapping[str, Any],
        *,
        workspace_id: str,
        session_id: str,
        window_id: str,
        artifact_family: str,
        scope: Mapping[str, Any] | None = None,
        status: str = "ready",
        capture_action: str = "capture_sidecar",
        provenance: Mapping[str, Any] | None = None,
    ) -> ResidualSidecarCatalogEntry:
        if isinstance(manifest, Mapping):
            raise ResidualStoreError("sidecar catalog entries require a file-backed manifest ref")
        manifest_ref = str(manifest)
        loaded = self.residual_store.load_manifest(manifest)
        family = _catalog_artifact_family(artifact_family)
        entry = ResidualSidecarCatalogEntry(
            key=ResidualSidecarCatalogKey(
                workspace_id=_non_empty_catalog_str(workspace_id, "workspace_id"),
                session_id=_non_empty_catalog_str(session_id, "session_id"),
                window_id=_non_empty_catalog_str(window_id, "window_id"),
                artifact_family=family,
            ),
            manifest_ref=manifest_ref,
            artifact_id=loaded.artifact_id,
            memory_family=loaded.memory_family,
            scope=dict(scope or loaded.scope()),
            ready=status == "ready",
            stale=False,
            status=status,
            capture_action=capture_action,
            provenance=_dict(provenance),
        )
        return self.add_entry(entry)

    def add_from_readiness_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        session_id: str = "workspace",
        base_path: str | Path | None = None,
    ) -> list[ResidualSidecarCatalogEntry]:
        workspace_id = str(metadata.get("workspace_root") or metadata.get("workspace_id") or "workspace")
        adapter_scope = dict(metadata.get("adapter_scope") or {})
        entries: list[ResidualSidecarCatalogEntry] = []
        families = dict(metadata.get("artifact_families") or {})
        for family_text, readiness_payload in families.items():
            family = _catalog_artifact_family_or_none(family_text)
            if family is None:
                continue
            readiness = _mapping(readiness_payload, "artifact readiness")
            status = str(readiness.get("status") or "missing")
            capture_action = str(readiness.get("capture_action") or "capture_sidecar")
            stale_window_ids = {str(item) for item in readiness.get("stale_window_ids", [])}
            missing_window_ids = {str(item) for item in readiness.get("missing_window_ids", [])}
            sidecar_refs = dict(readiness.get("sidecar_refs") or {})
            for window_id, manifest_ref in sidecar_refs.items():
                entry = self._entry_from_ref(
                    workspace_id=workspace_id,
                    session_id=session_id,
                    window_id=str(window_id),
                    artifact_family=family,
                    manifest_ref=str(manifest_ref),
                    base_path=base_path,
                    scope=adapter_scope,
                    ready=status == "ready" and str(window_id) not in stale_window_ids,
                    stale=str(window_id) in stale_window_ids,
                    status="stale" if str(window_id) in stale_window_ids else status,
                    capture_action=capture_action,
                    reason=None,
                    provenance={"source": "readiness_manifest"},
                )
                entries.append(entry)
            for window_id in sorted(missing_window_ids - set(str(key) for key in sidecar_refs)):
                entry = self._entry_from_ref(
                    workspace_id=workspace_id,
                    session_id=session_id,
                    window_id=window_id,
                    artifact_family=family,
                    manifest_ref=None,
                    base_path=base_path,
                    scope=adapter_scope,
                    ready=False,
                    stale=False,
                    status="missing",
                    capture_action=capture_action,
                    reason=f"missing {family} sidecar for window {window_id}",
                    provenance={"source": "readiness_manifest"},
                )
                entries.append(entry)
        return entries

    def add_from_window_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        workspace_id: str,
        session_id: str,
        base_path: str | Path | None = None,
        scope: Mapping[str, Any] | None = None,
    ) -> list[ResidualSidecarCatalogEntry]:
        window_id = _non_empty_catalog_str(metadata.get("window_id"), "window_id")
        entries: list[ResidualSidecarCatalogEntry] = []
        sidecar_refs = dict(metadata.get("sidecar_refs") or {})
        for family_text, manifest_ref in sidecar_refs.items():
            family = _catalog_artifact_family_or_none(family_text)
            if family is None:
                continue
            entry = self._entry_from_ref(
                workspace_id=workspace_id,
                session_id=session_id,
                window_id=window_id,
                artifact_family=family,
                manifest_ref=str(manifest_ref),
                base_path=base_path,
                scope=scope,
                ready=True,
                stale=False,
                status="ready",
                capture_action="capture_sidecar",
                reason=None,
                provenance={"source": "window_metadata"},
            )
            entries.append(entry)
        return entries

    def resolve(
        self,
        *,
        workspace_id: str,
        session_id: str,
        window_id: str,
        artifact_family: str,
        expected_scope: Mapping[str, Any] | None = None,
    ) -> ResidualSidecarCatalogResolution:
        key = ResidualSidecarCatalogKey(
            workspace_id=_non_empty_catalog_str(workspace_id, "workspace_id"),
            session_id=_non_empty_catalog_str(session_id, "session_id"),
            window_id=_non_empty_catalog_str(window_id, "window_id"),
            artifact_family=_catalog_artifact_family(artifact_family),
        )
        entry = self._entries.get(_catalog_key_tuple(key))
        if entry is None:
            return _catalog_resolution(
                key,
                ready=False,
                action="capture_sidecar",
                reason=f"missing {key.artifact_family} sidecar catalog entry for window {key.window_id}",
                missing=True,
            )
        if entry.stale or entry.status == "stale":
            return _catalog_resolution(
                key,
                ready=False,
                action="recapture_sidecar",
                reason=entry.reason or f"stale {key.artifact_family} sidecar for window {key.window_id}",
                entry=entry,
                stale=True,
            )
        if not entry.ready or entry.status == "missing" or not entry.manifest_ref:
            return _catalog_resolution(
                key,
                ready=False,
                action=entry.capture_action or "capture_sidecar",
                reason=entry.reason or f"missing {key.artifact_family} sidecar manifest for window {key.window_id}",
                entry=entry,
                missing=True,
            )
        manifest_path = self._manifest_path(entry.manifest_ref)
        if not manifest_path.exists():
            return _catalog_resolution(
                key,
                ready=False,
                action=entry.capture_action or "capture_sidecar",
                reason=f"missing sidecar manifest file: {manifest_path}",
                entry=entry,
                missing=True,
            )
        try:
            manifest = self.residual_store.load_manifest(manifest_path)
        except ResidualStoreError as exc:
            return _catalog_resolution(
                key,
                ready=False,
                action="recapture_sidecar",
                reason=f"invalid sidecar manifest: {exc}",
                entry=entry,
                stale=True,
            )
        scope_reason = _catalog_scope_mismatch_reason(
            manifest,
            _merged_scope(entry.scope, expected_scope),
        )
        if scope_reason:
            return _catalog_resolution(
                key,
                ready=False,
                action="recapture_sidecar",
                reason=scope_reason,
                entry=entry,
                manifest=manifest,
                stale=True,
            )
        refs = tuple(ref for ref in manifest.refs if ref.kind == key.artifact_family)
        if not refs:
            return _catalog_resolution(
                key,
                ready=False,
                action="recapture_sidecar",
                reason=f"sidecar manifest has no {key.artifact_family} refs",
                entry=entry,
                manifest=manifest,
                stale=True,
            )
        return _catalog_resolution(
            key,
            ready=True,
            action="use_sidecar_metadata",
            reason="sidecar metadata ready",
            entry=entry,
            manifest=manifest,
            refs=refs,
        )

    def _entry_from_ref(
        self,
        *,
        workspace_id: str,
        session_id: str,
        window_id: str,
        artifact_family: str,
        manifest_ref: str | None,
        base_path: str | Path | None,
        scope: Mapping[str, Any] | None,
        ready: bool,
        stale: bool,
        status: str,
        capture_action: str,
        reason: str | None,
        provenance: Mapping[str, Any] | None,
    ) -> ResidualSidecarCatalogEntry:
        scoped_ref = _base_relative_ref(manifest_ref, base_path)
        entry = ResidualSidecarCatalogEntry(
            key=ResidualSidecarCatalogKey(
                workspace_id=_non_empty_catalog_str(workspace_id, "workspace_id"),
                session_id=_non_empty_catalog_str(session_id, "session_id"),
                window_id=_non_empty_catalog_str(window_id, "window_id"),
                artifact_family=_catalog_artifact_family(artifact_family),
            ),
            manifest_ref=scoped_ref,
            scope=dict(scope or {}),
            ready=ready,
            stale=stale,
            status=status,
            capture_action=capture_action,
            reason=reason,
            provenance=_dict(provenance),
        )
        self.add_entry(entry)
        return entry

    def _manifest_path(self, manifest_ref: str) -> Path:
        path = Path(manifest_ref).expanduser()
        windows_path = PureWindowsPath(manifest_ref)
        if path.is_absolute() or windows_path.is_absolute():
            return path.resolve()
        base = self.base_path or Path.cwd()
        return (base / path).resolve()


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ResidualStoreError(f"sidecar manifest requires string {key}")
    return value


def _non_empty_catalog_str(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResidualStoreError(f"sidecar catalog requires string {key}")
    return value


def _catalog_artifact_family(value: Any) -> str:
    family = _non_empty_catalog_str(value, "artifact_family")
    if family not in CATALOG_ARTIFACT_FAMILIES:
        raise ResidualStoreError(f"unsupported sidecar catalog artifact_family: {family}")
    return family


def _catalog_artifact_family_or_none(value: Any) -> str | None:
    if not isinstance(value, str) or value not in CATALOG_ARTIFACT_FAMILIES:
        return None
    return value


def _catalog_key_tuple(
    key: ResidualSidecarCatalogKey,
) -> tuple[str, str, str, str]:
    return (key.workspace_id, key.session_id, key.window_id, key.artifact_family)


def _catalog_resolution(
    key: ResidualSidecarCatalogKey,
    *,
    ready: bool,
    action: str,
    reason: str,
    entry: ResidualSidecarCatalogEntry | None = None,
    manifest: ResidualSidecarManifest | None = None,
    refs: tuple[ResidualReference, ...] = (),
    missing: bool = False,
    stale: bool = False,
) -> ResidualSidecarCatalogResolution:
    return ResidualSidecarCatalogResolution(
        key=key,
        ready=ready,
        action=action,
        reason=reason,
        entry=entry,
        manifest=manifest,
        refs=refs,
        missing=missing,
        stale=stale,
        metadata_only=True,
        kv_replay_ready=False,
    )


def _base_relative_ref(manifest_ref: str | None, base_path: str | Path | None) -> str | None:
    if manifest_ref is None:
        return None
    if base_path is None:
        return manifest_ref
    path = Path(manifest_ref).expanduser()
    windows_path = PureWindowsPath(manifest_ref)
    if path.is_absolute() or windows_path.is_absolute():
        return str(path.resolve())
    return str((Path(base_path).expanduser().resolve() / path).resolve())


def _merged_scope(
    entry_scope: Mapping[str, Any],
    expected_scope: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output = dict(entry_scope)
    if expected_scope is not None:
        output.update(dict(expected_scope))
    return output


def _catalog_scope_mismatch_reason(
    manifest: ResidualSidecarManifest,
    expected_scope: Mapping[str, Any],
) -> str | None:
    actual = manifest.scope()
    for key in CATALOG_SCOPE_KEYS:
        if key not in expected_scope:
            continue
        expected = expected_scope.get(key)
        if expected is None or expected == "":
            continue
        if actual.get(key) != expected:
            return f"sidecar scope mismatch for {key}: expected {expected!r}, got {actual.get(key)!r}"
    return None


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
