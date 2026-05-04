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
class SourceFileRecord:
    path: str
    size_bytes: int
    sha256: str
    language: str
    symbols: list[str] = field(default_factory=list)
    import_tokens: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "language": self.language,
            "symbols": self.symbols,
            "import_tokens": self.import_tokens,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SourceFileRecord":
        return cls(
            path=str(data["path"]),
            size_bytes=int(data["size_bytes"]),
            sha256=str(data["sha256"]),
            language=str(data.get("language") or "text"),
            symbols=[str(item) for item in data.get("symbols", [])],
            import_tokens=[str(item) for item in data.get("import_tokens", [])],
        )


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


def _index_file(root: Path, path: Path, size_bytes: int) -> SourceFileRecord | None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError:
        return None
    relative = path.relative_to(root).as_posix()
    return SourceFileRecord(
        path=relative,
        size_bytes=size_bytes,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        language=_language_for_suffix(path.suffix),
        symbols=_unique(_symbol_tokens(text, path.suffix)),
        import_tokens=_unique(_import_tokens(text)),
    )


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
