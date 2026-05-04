"""Local model artifact inspection for David readiness checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


REQUIRED_VINDEX_FILES = ("index.json", "tokenizer.json", "weight_manifest.json")


@dataclass(frozen=True)
class VindexArtifactMetadata:
    path: str
    available: bool
    family: str | None = None
    source_hf_path: str | None = None
    hidden_size: int | None = None
    layers: int | None = None
    vocab: int | None = None
    dtype: str | None = None
    file_presence: dict[str, bool] = field(default_factory=dict)
    total_bytes: int = 0
    missing_files: list[str] = field(default_factory=list)
    binary_weight_files: list[str] = field(default_factory=list)
    manifest_declared_files: list[str] = field(default_factory=list)
    missing_binary_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "available": self.available,
            "family": self.family,
            "source_hf_path": self.source_hf_path,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "vocab": self.vocab,
            "dtype": self.dtype,
            "file_presence": dict(self.file_presence),
            "total_bytes": self.total_bytes,
            "missing_files": list(self.missing_files),
            "binary_weight_files": list(self.binary_weight_files),
            "manifest_declared_files": list(self.manifest_declared_files),
            "missing_binary_files": list(self.missing_binary_files),
        }


def is_vindex_artifact_path(path: str | Path) -> bool:
    artifact_path = Path(path)
    return artifact_path.is_dir() and artifact_path.name.endswith(".vindex.ple")


def inspect_vindex_artifact(path: str | Path) -> VindexArtifactMetadata:
    artifact_path = Path(path)
    file_presence = {name: (artifact_path / name).is_file() for name in REQUIRED_VINDEX_FILES}
    missing = [name for name, present in file_presence.items() if not present]
    index_data = _read_json_object(artifact_path / "index.json") if file_presence["index.json"] else {}
    manifest_data = _read_json_value(artifact_path / "weight_manifest.json") if file_presence["weight_manifest.json"] else []
    manifest_declared_files = _manifest_declared_files(manifest_data)
    binary_files = _binary_weight_files(artifact_path, manifest_declared_files)
    missing_binary_files = [name for name in manifest_declared_files if not (artifact_path / name).is_file()]

    for name in binary_files:
        file_presence.setdefault(name, (artifact_path / name).is_file())
    missing.extend(missing_binary_files)

    total_bytes = 0
    for child in artifact_path.iterdir() if artifact_path.is_dir() else ():
        if child.is_file():
            total_bytes += child.stat().st_size

    return VindexArtifactMetadata(
        path=str(artifact_path),
        available=is_vindex_artifact_path(artifact_path) and not missing,
        family=_string_or_none(index_data.get("family")),
        source_hf_path=_source_hf_path(index_data),
        hidden_size=_int_or_none(index_data.get("hidden_size")),
        layers=_int_or_none(index_data.get("num_layers")),
        vocab=_int_or_none(index_data.get("vocab_size")),
        dtype=_string_or_none(index_data.get("dtype")),
        file_presence=file_presence,
        total_bytes=total_bytes,
        missing_files=missing,
        binary_weight_files=binary_files,
        manifest_declared_files=manifest_declared_files,
        missing_binary_files=missing_binary_files,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    data = _read_json_value(path)
    return data if isinstance(data, dict) else {}


def _read_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _source_hf_path(index_data: dict[str, Any]) -> str | None:
    source = index_data.get("source")
    if isinstance(source, dict):
        hf_path = _string_or_none(source.get("huggingface_repo"))
        if hf_path:
            return hf_path
    return _string_or_none(index_data.get("model"))


def _manifest_declared_files(manifest_data: Any) -> list[str]:
    files: set[str] = set()
    pending = [manifest_data]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            file_name = _string_or_none(item.get("file"))
            if file_name:
                files.add(file_name)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return sorted(files)


def _binary_weight_files(artifact_path: Path, manifest_declared_files: list[str]) -> list[str]:
    files: set[str] = set(manifest_declared_files)
    if artifact_path.is_dir():
        files.update(child.name for child in artifact_path.glob("*.bin") if child.is_file())
    return sorted(files)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
