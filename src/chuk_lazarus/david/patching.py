"""Patch validation and application helpers for David local edits."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .patch_routing import is_protected_path, is_workspace_local_path, normalize_path


SEARCH_REPLACE_RE = re.compile(
    r"<<<<\s*SEARCH\s*(?P<old>.*?)\s*====\s*REPLACE\s*(?P<new>.*?)\s*>>>>",
    flags=re.IGNORECASE | re.DOTALL,
)
PROTOCOL_MARKER_RE = re.compile(r"<<<<|>>>>|====|\bSEARCH\b|\bREPLACE\b", flags=re.IGNORECASE)
DIFF_HEADER_RE = re.compile(r"^diff --git (?P<old>\S+) (?P<new>\S+)$")
HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


@dataclass(frozen=True)
class StrictSearchReplaceBlock:
    path: str
    old: str
    new: str
    block_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatchApplyDiagnostic:
    mode: str
    applied: bool
    changed_paths: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    block_count: int = 0
    protected_paths: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.applied and not self.failures

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["changed_paths"] = list(self.changed_paths)
        data["failures"] = list(self.failures)
        data["protected_paths"] = list(self.protected_paths)
        data["details"] = dict(self.details)
        return data


@dataclass(frozen=True)
class _DiffHunk:
    old_start: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _DiffFile:
    old_path: str
    new_path: str
    hunks: tuple[_DiffHunk, ...]

    @property
    def target_path(self) -> str:
        return self.new_path if self.new_path != "/dev/null" else self.old_path


def canonicalize_search_replace_output(generated: str) -> str:
    text = generated or ""
    if "<<<<" not in text or "SEARCH" not in text.upper():
        return text
    text = re.sub(r"(?m)^={4,}\s*$", "==== REPLACE", text)
    return re.sub(r"(?m)^>{4,}\s*REPLACE\s*$", ">>>>", text, flags=re.IGNORECASE)


def parse_strict_search_replace_blocks(
    patch_text: str,
    *,
    known_paths: Mapping[str, Any] | set[str] | tuple[str, ...] | list[str] = (),
) -> tuple[StrictSearchReplaceBlock, ...]:
    text = canonicalize_search_replace_output(patch_text)
    path_set = set(known_paths.keys() if isinstance(known_paths, Mapping) else known_paths)
    blocks: list[StrictSearchReplaceBlock] = []
    for block_index, match in enumerate(SEARCH_REPLACE_RE.finditer(text or ""), start=1):
        prefix = text[: match.start()]
        blocks.append(
            StrictSearchReplaceBlock(
                path=_path_before_block(prefix, path_set),
                old=match.group("old").strip("\n"),
                new=match.group("new").strip("\n"),
                block_index=block_index,
            )
        )
    return tuple(blocks)


def validate_patch_candidate(
    patch_text: str,
    *,
    known_paths: Mapping[str, Any] | set[str] | tuple[str, ...] | list[str] = (),
) -> PatchApplyDiagnostic:
    blocks = parse_strict_search_replace_blocks(patch_text, known_paths=known_paths)
    if blocks:
        failures, protected = _validate_strict_blocks(blocks)
        return PatchApplyDiagnostic(
            mode="strict_search_replace",
            applied=False,
            failures=tuple(failures),
            block_count=len(blocks),
            protected_paths=tuple(protected),
        )
    if looks_like_unified_diff(patch_text):
        try:
            diff_files = parse_unified_diff(patch_text)
        except ValueError as exc:
            return PatchApplyDiagnostic(mode="unified_diff", applied=False, failures=(str(exc),))
        failures, protected = _validate_diff_files(diff_files, patch_text)
        return PatchApplyDiagnostic(
            mode="unified_diff",
            applied=False,
            failures=tuple(failures),
            block_count=sum(len(item.hunks) for item in diff_files),
            protected_paths=tuple(protected),
        )
    return PatchApplyDiagnostic(mode="none", applied=False, failures=("no strict blocks or unified diff",))


def apply_patch_candidate(workspace_root: str | Path, patch_text: str) -> PatchApplyDiagnostic:
    root = Path(workspace_root).resolve()
    if not root.exists():
        return PatchApplyDiagnostic(
            mode="none", applied=False, failures=(f"workspace root does not exist: {root}",)
        )
    blocks = parse_strict_search_replace_blocks(patch_text)
    if blocks:
        return _apply_strict_blocks(root, blocks)
    if looks_like_unified_diff(patch_text):
        try:
            diff_files = parse_unified_diff(patch_text)
        except ValueError as exc:
            return PatchApplyDiagnostic(mode="unified_diff", applied=False, failures=(str(exc),))
        return _apply_unified_diff(root, diff_files, patch_text)
    return PatchApplyDiagnostic(mode="none", applied=False, failures=("no strict blocks or unified diff",))


def looks_like_unified_diff(text: str) -> bool:
    return bool(re.search(r"(?m)^(diff --git|---\s+a/|\+\+\+\s+b/|@@\s)", text or ""))


def parse_unified_diff(patch_text: str) -> tuple[_DiffFile, ...]:
    lines = (patch_text or "").splitlines()
    diff_files: list[_DiffFile] = []
    index = 0
    current_old = ""
    current_new = ""
    while index < len(lines):
        line = lines[index]
        header = DIFF_HEADER_RE.match(line)
        if header:
            current_old = _strip_diff_prefix(header.group("old"))
            current_new = _strip_diff_prefix(header.group("new"))
            index += 1
            continue
        if line.startswith("--- "):
            old_path = _strip_diff_prefix(line[4:].split("\t", 1)[0].strip())
            index += 1
            if index >= len(lines) or not lines[index].startswith("+++ "):
                raise ValueError("unified diff missing +++ header")
            new_path = _strip_diff_prefix(lines[index][4:].split("\t", 1)[0].strip())
            current_old = old_path if old_path else current_old
            current_new = new_path if new_path else current_new
            index += 1
            hunks: list[_DiffHunk] = []
            while index < len(lines):
                if DIFF_HEADER_RE.match(lines[index]) or lines[index].startswith("--- "):
                    break
                hunk_header = HUNK_HEADER_RE.match(lines[index])
                if not hunk_header:
                    index += 1
                    continue
                old_start = int(hunk_header.group("old_start"))
                index += 1
                hunk_lines: list[str] = []
                while index < len(lines):
                    body = lines[index]
                    if body.startswith("\\ No newline at end of file"):
                        index += 1
                        continue
                    if (
                        HUNK_HEADER_RE.match(body)
                        or DIFF_HEADER_RE.match(body)
                        or body.startswith("--- ")
                    ):
                        break
                    if not body or body[0] not in {" ", "+", "-"}:
                        raise ValueError(f"invalid unified diff hunk line: {body[:80]}")
                    hunk_lines.append(body)
                    index += 1
                hunks.append(_DiffHunk(old_start=old_start, lines=tuple(hunk_lines)))
            if not current_old and not current_new:
                raise ValueError("unified diff missing file path")
            diff_files.append(_DiffFile(current_old, current_new, tuple(hunks)))
            continue
        index += 1
    if not diff_files:
        raise ValueError("unified diff contains no file hunks")
    return tuple(diff_files)


def _apply_strict_blocks(root: Path, blocks: tuple[StrictSearchReplaceBlock, ...]) -> PatchApplyDiagnostic:
    failures, protected = _validate_strict_blocks(blocks)
    if failures:
        return PatchApplyDiagnostic(
            mode="strict_search_replace",
            applied=False,
            failures=tuple(failures),
            block_count=len(blocks),
            protected_paths=tuple(protected),
        )
    changed: list[str] = []
    for block in blocks:
        resolved, failure = _resolve_workspace_path(root, block.path)
        if failure:
            failures.append(f"block {block.block_index}: {failure}")
            continue
        assert resolved is not None
        try:
            content = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            failures.append(f"block {block.block_index}: cannot read {block.path}: {exc}")
            continue
        if block.old not in content:
            failures.append(f"block {block.block_index}: SEARCH text not found in {block.path}")
            continue
        resolved.write_text(content.replace(block.old, block.new, 1), encoding="utf-8")
        changed.append(block.path)
    return PatchApplyDiagnostic(
        mode="strict_search_replace",
        applied=not failures and bool(changed),
        changed_paths=tuple(dict.fromkeys(changed)),
        failures=tuple(failures),
        block_count=len(blocks),
        protected_paths=tuple(protected),
    )


def _apply_unified_diff(
    root: Path, diff_files: tuple[_DiffFile, ...], patch_text: str
) -> PatchApplyDiagnostic:
    failures, protected = _validate_diff_files(diff_files, patch_text)
    if failures:
        return PatchApplyDiagnostic(
            mode="unified_diff",
            applied=False,
            failures=tuple(failures),
            block_count=sum(len(item.hunks) for item in diff_files),
            protected_paths=tuple(protected),
        )
    changed: list[str] = []
    for diff_file in diff_files:
        path = diff_file.target_path
        resolved, failure = _resolve_workspace_path(root, path)
        if failure:
            failures.append(f"{path}: {failure}")
            continue
        assert resolved is not None
        original = ""
        if diff_file.old_path != "/dev/null":
            try:
                original = resolved.read_text(encoding="utf-8")
            except OSError as exc:
                failures.append(f"{path}: cannot read: {exc}")
                continue
        try:
            updated = _apply_hunks_to_text(original, diff_file.hunks)
        except ValueError as exc:
            failures.append(f"{path}: {exc}")
            continue
        if diff_file.new_path == "/dev/null":
            try:
                resolved.unlink()
            except OSError as exc:
                failures.append(f"{path}: cannot delete: {exc}")
                continue
        else:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(updated, encoding="utf-8")
        changed.append(path)
    return PatchApplyDiagnostic(
        mode="unified_diff",
        applied=not failures and bool(changed),
        changed_paths=tuple(dict.fromkeys(changed)),
        failures=tuple(failures),
        block_count=sum(len(item.hunks) for item in diff_files),
        protected_paths=tuple(protected),
    )


def _validate_strict_blocks(
    blocks: tuple[StrictSearchReplaceBlock, ...],
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    protected: list[str] = []
    for block in blocks:
        if not block.path:
            failures.append(f"block {block.block_index}: path unavailable")
            continue
        if not is_workspace_local_path(block.path):
            failures.append(f"block {block.block_index}: path is not workspace-local: {block.path}")
        if is_protected_path(block.path):
            protected.append(block.path)
            failures.append(f"block {block.block_index}: protected proof-rig path: {block.path}")
        if not block.old:
            failures.append(f"block {block.block_index}: empty SEARCH section")
        marker = PROTOCOL_MARKER_RE.search(block.new)
        if marker is not None:
            failures.append(
                f"block {block.block_index}: replacement contains nested protocol marker {marker.group(0)!r}"
            )
    return failures, protected


def _validate_diff_files(
    diff_files: tuple[_DiffFile, ...], patch_text: str
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    protected: list[str] = []
    for diff_file in diff_files:
        path = diff_file.target_path
        if not path:
            failures.append("unified diff file missing target path")
            continue
        if not is_workspace_local_path(path):
            failures.append(f"diff target path is not workspace-local: {path}")
        if is_protected_path(path):
            protected.append(path)
            failures.append(f"protected proof-rig path: {path}")
        if not diff_file.hunks:
            failures.append(f"{path}: no hunks")
    for line_number, line in enumerate((patch_text or "").splitlines(), start=1):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        marker = PROTOCOL_MARKER_RE.search(line[1:])
        if marker is not None:
            failures.append(
                f"line {line_number}: added line contains nested protocol marker {marker.group(0)!r}"
            )
    return failures, protected


def _apply_hunks_to_text(original: str, hunks: tuple[_DiffHunk, ...]) -> str:
    original_had_newline = original.endswith("\n")
    source_lines = original.splitlines()
    output: list[str] = []
    source_index = 0
    for hunk in hunks:
        target_index = max(0, hunk.old_start - 1)
        if target_index < source_index:
            raise ValueError("overlapping or out-of-order hunks")
        output.extend(source_lines[source_index:target_index])
        source_index = target_index
        for line in hunk.lines:
            op = line[0]
            body = line[1:]
            if op == " ":
                if source_index >= len(source_lines) or source_lines[source_index] != body:
                    raise ValueError(f"context mismatch near line {source_index + 1}")
                output.append(body)
                source_index += 1
            elif op == "-":
                if source_index >= len(source_lines) or source_lines[source_index] != body:
                    raise ValueError(f"delete mismatch near line {source_index + 1}")
                source_index += 1
            elif op == "+":
                output.append(body)
    output.extend(source_lines[source_index:])
    text = "\n".join(output)
    if original_had_newline or text:
        text += "\n"
    return text


def _path_before_block(prefix: str, known_paths: set[str]) -> str:
    basenames = {Path(path).name: path for path in known_paths}
    for raw_line in reversed((prefix or "").splitlines()):
        line = raw_line.strip()
        if not line or line.startswith(("<<<<", "====", ">>>>")):
            continue
        candidate = normalize_path(line.strip("`'\""))
        if candidate in known_paths:
            return candidate
        if Path(candidate).name in basenames:
            return basenames[Path(candidate).name]
        if candidate:
            return candidate
    return ""


def _strip_diff_prefix(path: str) -> str:
    cleaned = normalize_path(path)
    if cleaned in {"/dev/null", "dev/null"}:
        return "/dev/null"
    if cleaned.startswith("a/") or cleaned.startswith("b/"):
        return cleaned[2:]
    return cleaned


def _resolve_workspace_path(root: Path, rel_path: str) -> tuple[Path | None, str]:
    normalized = normalize_path(rel_path)
    if not is_workspace_local_path(normalized):
        return None, f"path is not workspace-local: {rel_path}"
    resolved = (root / normalized).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None, f"path escapes workspace: {rel_path}"
    return resolved, ""
