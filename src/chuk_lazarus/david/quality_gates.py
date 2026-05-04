"""Workspace-local quality gate discovery for David agent tasks."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from .patch_routing import classify_path, is_benchmark_padding_path, normalize_path


DEFAULT_MAX_CANDIDATES = 8
MAX_SELECTED_PATH_ARGS = 16
MAX_SIGNAL_FILE_BYTES = 128 * 1024
PACKAGE_SCRIPT_PRIORITY = ("test", "lint", "typecheck", "check", "build")


@dataclass(frozen=True)
class QualityGateCandidate:
    """A verification command inferred from tiny workspace signals.

    The command is an argv list suitable for a non-shell subprocess call. Discovery
    never executes it.
    """

    name: str
    command: list[str]
    reason: str
    confidence: float
    provenance: dict[str, Any] = field(default_factory=dict)
    kind: str = "verification"
    cwd: str = "."

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["provenance"] = dict(self.provenance)
        data["command"] = list(self.command)
        return data


def discover_quality_gates(
    workspace_root: Path,
    selected_tests: Sequence[str] = (),
    selected_paths: Sequence[str] = (),
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[QualityGateCandidate]:
    """Infer likely verification commands without running them.

    Discovery is intentionally shallow: it checks known root-level metadata files,
    direct ``tests`` directory presence, and explicitly selected route paths.
    """

    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ValueError(f"workspace_root must be an existing directory: {workspace_root}")

    limit = _candidate_limit(max_candidates)
    rejected: dict[str, list[str]] = {}
    safe_tests = _safe_existing_paths(root, selected_tests, expected="test", rejected=rejected)
    safe_sources = _safe_existing_paths(root, selected_paths, expected="python_source", rejected=rejected)
    candidates: list[QualityGateCandidate] = []
    seen_commands: set[tuple[str, ...]] = set()

    if safe_tests:
        _append_candidate(
            candidates,
            seen_commands,
            QualityGateCandidate(
                name="pytest_selected",
                command=["python", "-m", "pytest", *safe_tests],
                reason="explicit selected tests from product routing",
                confidence=0.95,
                provenance={
                    "source": "selected_tests",
                    "selected_tests": safe_tests,
                    "rejected_selected_paths": rejected,
                },
            ),
            limit=limit,
        )

    if safe_sources:
        _append_candidate(
            candidates,
            seen_commands,
            QualityGateCandidate(
                name="py_compile_selected",
                command=["python", "-m", "py_compile", *safe_sources],
                reason="selected Python source files can be syntax-checked without test discovery",
                confidence=0.78,
                provenance={
                    "source": "selected_paths",
                    "selected_paths": safe_sources,
                    "rejected_selected_paths": rejected,
                },
            ),
            limit=limit,
        )

    pytest_signal = _pytest_signal(root)
    if pytest_signal is not None and _has_direct_tests_dir(root):
        _append_candidate(
            candidates,
            seen_commands,
            QualityGateCandidate(
                name="pytest_workspace",
                command=["python", "-m", "pytest", "tests"],
                reason=pytest_signal["reason"],
                confidence=pytest_signal["confidence"],
                provenance=pytest_signal["provenance"],
            ),
            limit=limit,
        )

    for candidate in _package_json_candidates(root):
        _append_candidate(candidates, seen_commands, candidate, limit=limit)
        if len(candidates) >= limit:
            break

    return candidates[:limit]


def _candidate_limit(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_CANDIDATES
    return max(0, min(parsed, DEFAULT_MAX_CANDIDATES))


def _append_candidate(
    candidates: list[QualityGateCandidate],
    seen_commands: set[tuple[str, ...]],
    candidate: QualityGateCandidate,
    *,
    limit: int,
) -> None:
    if len(candidates) >= limit:
        return
    key = tuple(candidate.command)
    if key in seen_commands:
        return
    seen_commands.add(key)
    candidates.append(candidate)


def _safe_existing_paths(
    root: Path,
    paths: Sequence[str],
    *,
    expected: str,
    rejected: dict[str, list[str]],
) -> list[str]:
    safe: list[str] = []
    for raw in paths:
        path_part, selector = _split_pytest_selector(raw) if expected == "test" else (str(raw), "")
        normalized, reasons = _safe_workspace_path(root, path_part)
        if reasons:
            rejected[str(raw)] = reasons
            continue
        if normalized is None:
            rejected[str(raw)] = ["empty path"]
            continue
        if selector and not _safe_pytest_selector(selector):
            rejected[str(raw)] = ["unsafe pytest selector"]
            continue
        target = (root / normalized).resolve()
        if not target.is_file():
            rejected[str(raw)] = ["path does not exist as a file"]
            continue
        classification = classify_path(normalized)
        suffix = PurePosixPath(normalized).suffix.lower()
        if expected == "test" and not classification.is_test:
            rejected[str(raw)] = ["path is not test-like"]
            continue
        if expected == "python_source" and (suffix != ".py" or classification.is_test):
            rejected[str(raw)] = ["path is not a selected Python source file"]
            continue
        command_arg = f"{normalized}{selector}"
        if command_arg not in safe:
            safe.append(command_arg)
        if len(safe) >= MAX_SELECTED_PATH_ARGS:
            break
    return safe


def _split_pytest_selector(raw: str | Path) -> tuple[str, str]:
    raw_text = str(raw or "")
    path_part, separator, selector = raw_text.partition("::")
    if not separator:
        return raw_text, ""
    return path_part, f"{separator}{selector}"


def _safe_pytest_selector(selector: str) -> bool:
    return bool(selector.startswith("::") and "\x00" not in selector and "\n" not in selector)


def _safe_workspace_path(root: Path, raw: str | Path) -> tuple[str | None, list[str]]:
    raw_text = str(raw or "").strip()
    if not raw_text:
        return None, ["empty path"]

    normalized = normalize_path(raw_text)
    candidate_path = Path(raw_text)
    if candidate_path.is_absolute():
        try:
            resolved = candidate_path.resolve()
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            return None, ["path escapes workspace"]
        normalized = relative.as_posix()

    classification = classify_path(normalized)
    reasons: list[str] = []
    if not classification.is_workspace_local:
        reasons.append("path is not workspace-local")
    if classification.is_protected:
        reasons.append("protected proof-rig path")
    if is_benchmark_padding_path(normalized):
        reasons.append("benchmark padding path")
    if _looks_like_windows_absolute(normalized):
        reasons.append("path is not workspace-local")
    if reasons:
        return None, list(dict.fromkeys(reasons))

    try:
        resolved_target = (root / normalized).resolve()
        resolved_target.relative_to(root)
    except (OSError, ValueError):
        return None, ["path escapes workspace"]
    return normalized, []


def _looks_like_windows_absolute(path: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:", normalize_path(path)))


def _has_direct_tests_dir(root: Path) -> bool:
    try:
        return (root / "tests").is_dir()
    except OSError:
        return False


def _pytest_signal(root: Path) -> dict[str, Any] | None:
    pyproject = root / "pyproject.toml"
    text = _read_small_text(pyproject)
    if text is None:
        return None
    lower_text = text.lower()
    if "[tool.pytest" in lower_text:
        return {
            "reason": "pyproject.toml declares pytest configuration and tests directory exists",
            "confidence": 0.86,
            "provenance": {
                "source": "pyproject.toml",
                "signal": "tool.pytest",
                "test_root": "tests",
            },
        }
    if "pytest" in lower_text:
        return {
            "reason": "pyproject.toml references pytest and tests directory exists",
            "confidence": 0.72,
            "provenance": {
                "source": "pyproject.toml",
                "signal": "pytest_text",
                "test_root": "tests",
            },
        }
    return {
        "reason": "pyproject.toml and tests directory exist",
        "confidence": 0.62,
        "provenance": {
            "source": "pyproject.toml",
            "signal": "project_metadata",
            "test_root": "tests",
        },
    }


def _package_json_candidates(root: Path) -> list[QualityGateCandidate]:
    package_json = root / "package.json"
    text = _read_small_text(package_json)
    if text is None:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    scripts = data.get("scripts")
    if not isinstance(scripts, dict):
        return []

    script_names = [
        name
        for name in PACKAGE_SCRIPT_PRIORITY
        if isinstance(scripts.get(name), str) and scripts.get(name)
    ]
    script_names.extend(
        sorted(
            name
            for name, body in scripts.items()
            if name not in script_names and isinstance(name, str) and isinstance(body, str)
        )
    )

    candidates: list[QualityGateCandidate] = []
    for name in script_names[:DEFAULT_MAX_CANDIDATES]:
        candidates.append(
            QualityGateCandidate(
                name=f"package_{name}",
                command=["npm", "run", name],
                reason=f"package.json defines a {name!r} script",
                confidence=0.82 if name == "test" else 0.68,
                provenance={
                    "source": "package.json",
                    "script": name,
                    "package_manager": "npm",
                },
            )
        )
    return candidates


def _read_small_text(path: Path) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_SIGNAL_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
