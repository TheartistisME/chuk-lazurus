"""Lightweight source indexing for David workspace readiness."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

from .indexing import (
    ARTIFACT_FAMILY_LEXICAL_SOURCE,
    CAPTURE_REQUIRED_FAMILIES,
)
from .patch_routing import is_protected_path, normalize_path


SCHEMA = "david.source_index.v1"
DEFAULT_MAX_FILES = 200
DEFAULT_MAX_FILE_BYTES = 256_000
DEFAULT_PRUNED_DIRS = frozenset(
    {
        ".david",
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
        "venv",
    }
)
DEFAULT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".jsx",
        ".md",
        ".py",
        ".rs",
        ".sh",
        ".toml",
        ".ts",
        ".tsx",
        ".yaml",
        ".yml",
    }
)


@dataclass(frozen=True)
class SourceWindowRecord:
    window_id: str
    start_line: int
    end_line: int
    sha256: str
    vector_ref: str | None = None
    sidecar_refs: dict[str, str] = field(default_factory=dict)
    capture_status: dict[str, str] = field(default_factory=dict)

    @property
    def line_span(self) -> dict[str, int]:
        return {"start_line": self.start_line, "end_line": self.end_line}

    def to_json(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "line_span": self.line_span,
            "sha256": self.sha256,
            "vector_ref": self.vector_ref,
            "sidecar_refs": dict(self.sidecar_refs),
            "capture_status": _capture_status(self.capture_status),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any], *, path: str, file_sha256: str) -> "SourceWindowRecord":
        span = dict(data.get("line_span") or {})
        start_line = int(span.get("start_line") or data.get("start_line") or 1)
        end_line = int(span.get("end_line") or data.get("end_line") or start_line)
        return cls(
            window_id=str(data.get("window_id") or stable_source_window_id(path, start_line, end_line)),
            start_line=start_line,
            end_line=end_line,
            sha256=str(data.get("sha256") or file_sha256),
            vector_ref=_optional_str(data.get("vector_ref")),
            sidecar_refs={str(key): str(value) for key, value in dict(data.get("sidecar_refs") or {}).items()},
            capture_status=_capture_status(dict(data.get("capture_status") or {})),
        )


@dataclass(frozen=True)
class SourceFileRecord:
    path: str
    size_bytes: int
    sha256: str
    language: str
    symbols: list[str] = field(default_factory=list)
    import_tokens: list[str] = field(default_factory=list)
    windows: list[SourceWindowRecord] = field(default_factory=list)

    @property
    def window_ids(self) -> list[str]:
        return [window.window_id for window in self._windows()]

    @property
    def line_span(self) -> dict[str, int]:
        return self._windows()[0].line_span

    def to_json(self) -> dict[str, Any]:
        windows = self._windows()
        primary = windows[0]
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "language": self.language,
            "symbols": self.symbols,
            "import_tokens": self.import_tokens,
            "window_id": primary.window_id,
            "window_ids": [window.window_id for window in windows],
            "line_span": primary.line_span,
            "vector_ref": primary.vector_ref,
            "sidecar_refs": dict(primary.sidecar_refs),
            "capture_status": _capture_status(primary.capture_status),
            "windows": [window.to_json() for window in windows],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SourceFileRecord":
        path = str(data["path"])
        sha256 = str(data["sha256"])
        windows = [
            SourceWindowRecord.from_json(dict(item), path=path, file_sha256=sha256)
            for item in data.get("windows", [])
        ]
        if not windows:
            span = dict(data.get("line_span") or {})
            start_line = int(span.get("start_line") or 1)
            end_line = int(span.get("end_line") or start_line)
            windows = [
                SourceWindowRecord(
                    window_id=str(data.get("window_id") or stable_source_window_id(path, start_line, end_line)),
                    start_line=start_line,
                    end_line=end_line,
                    sha256=sha256,
                    vector_ref=_optional_str(data.get("vector_ref")),
                    sidecar_refs={str(key): str(value) for key, value in dict(data.get("sidecar_refs") or {}).items()},
                    capture_status=_capture_status(dict(data.get("capture_status") or {})),
                )
            ]
        return cls(
            path=path,
            size_bytes=int(data["size_bytes"]),
            sha256=sha256,
            language=str(data.get("language") or "text"),
            symbols=[str(item) for item in data.get("symbols", [])],
            import_tokens=[str(item) for item in data.get("import_tokens", [])],
            windows=windows,
        )

    def _windows(self) -> list[SourceWindowRecord]:
        if self.windows:
            return self.windows
        return [
            SourceWindowRecord(
                window_id=stable_source_window_id(self.path, 1, 1),
                start_line=1,
                end_line=1,
                sha256=self.sha256,
                capture_status=_capture_status({}),
            )
        ]


@dataclass(frozen=True)
class SourceIndexManifest:
    workspace_root: str
    adapter_scope: dict[str, Any]
    files: list[SourceFileRecord]
    pruned_dirs: list[str]
    max_files: int = DEFAULT_MAX_FILES
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    truncated: bool = False
    schema: str = SCHEMA
    indexed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat())

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "indexed_at": self.indexed_at,
            "workspace_root": self.workspace_root,
            "adapter_scope": self.adapter_scope,
            "max_files": self.max_files,
            "max_file_bytes": self.max_file_bytes,
            "pruned_dirs": self.pruned_dirs,
            "truncated": self.truncated,
            "file_count": len(self.files),
            "files": [record.to_json() for record in self.files],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SourceIndexManifest":
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported source index schema: {data.get('schema')!r}")
        return cls(
            workspace_root=str(data["workspace_root"]),
            adapter_scope=dict(data.get("adapter_scope") or {}),
            files=[SourceFileRecord.from_json(item) for item in data.get("files", [])],
            pruned_dirs=[str(item) for item in data.get("pruned_dirs", [])],
            max_files=int(data.get("max_files") or DEFAULT_MAX_FILES),
            max_file_bytes=int(data.get("max_file_bytes") or DEFAULT_MAX_FILE_BYTES),
            truncated=bool(data.get("truncated")),
            indexed_at=str(data.get("indexed_at") or ""),
        )


def build_source_index(
    workspace_root: Path,
    adapter_scope: dict[str, Any],
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    pruned_dirs: Iterable[str] = DEFAULT_PRUNED_DIRS,
    suffixes: Iterable[str] = DEFAULT_SUFFIXES,
) -> SourceIndexManifest:
    root = Path(workspace_root).resolve()
    pruned = set(pruned_dirs)
    allowed_suffixes = {suffix.lower() for suffix in suffixes}
    records: list[SourceFileRecord] = []
    truncated = False

    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(dirname for dirname in dirs if dirname not in pruned)
        for filename in sorted(files):
            if len(records) >= max_files:
                truncated = True
                dirs[:] = []
                break
            path = Path(current) / filename
            if path.suffix.lower() not in allowed_suffixes:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > max_file_bytes:
                continue
            record = _index_file(root, path, stat.st_size)
            if record is not None:
                records.append(record)

    return SourceIndexManifest(
        workspace_root=str(root),
        adapter_scope=dict(adapter_scope),
        files=records,
        pruned_dirs=sorted(pruned),
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        truncated=truncated,
    )


def save_source_index(path: Path, manifest: SourceIndexManifest) -> SourceIndexManifest:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_source_index(path: Path) -> SourceIndexManifest:
    return SourceIndexManifest.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def index_source_file(root: Path, path: Path) -> SourceFileRecord | None:
    """Index one workspace file using the same rules as the bounded source index."""
    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    try:
        stat = resolved_path.stat()
    except OSError:
        return None
    return _index_file(resolved_root, resolved_path, stat.st_size)


def _index_file(root: Path, path: Path, size_bytes: int) -> SourceFileRecord | None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError:
        return None
    relative = normalize_path(path.relative_to(root).as_posix())
    if is_protected_path(relative):
        return None
    line_count = _line_count(text)
    window = SourceWindowRecord(
        window_id=stable_source_window_id(relative, 1, line_count),
        start_line=1,
        end_line=line_count,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        capture_status=_capture_status({}),
    )
    return SourceFileRecord(
        path=relative,
        size_bytes=size_bytes,
        sha256=window.sha256,
        language=_language_for_suffix(path.suffix),
        symbols=_unique(_symbol_tokens(text, path.suffix)),
        import_tokens=_unique(_import_tokens(text)),
        windows=[window],
    )


def stable_source_window_id(path: str, start_line: int, end_line: int) -> str:
    normalized = normalize_path(path)
    del end_line
    digest = hashlib.sha256(f"{normalized}:{start_line}".encode("utf-8")).hexdigest()[:16]
    return f"source-window:{digest}"


def _symbol_tokens(text: str, suffix: str) -> list[str]:
    tokens: list[str] = []
    if suffix == ".py":
        tokens.extend(re.findall(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE))
    elif suffix in {".js", ".jsx", ".ts", ".tsx"}:
        tokens.extend(re.findall(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)", text, re.MULTILINE))
        tokens.extend(re.findall(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)", text, re.MULTILINE))
        tokens.extend(re.findall(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=", text, re.MULTILINE))
    elif suffix == ".rs":
        tokens.extend(re.findall(r"^\s*(?:pub\s+)?(?:fn|struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE))
    elif suffix == ".go":
        tokens.extend(re.findall(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE))
        tokens.extend(re.findall(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE))
    return tokens[:100]


def _import_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if stripped.startswith(("import ", "from ", "#include", "require(", "use ", "mod ")):
            tokens.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_.:-]*", stripped))
    return tokens[:200]


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _language_for_suffix(suffix: str) -> str:
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".cs": "csharp",
        ".sh": "shell",
        ".md": "markdown",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(suffix.lower(), "text")


def _line_count(text: str) -> int:
    return max(1, len(text.splitlines()))


def _capture_status(overrides: dict[str, str]) -> dict[str, str]:
    status = {ARTIFACT_FAMILY_LEXICAL_SOURCE: "captured"}
    status.update({family: "required" for family in CAPTURE_REQUIRED_FAMILIES})
    status.update({str(key): str(value) for key, value in overrides.items()})
    return status


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
