"""Patch-target routing helpers for the David product runtime.

These helpers turn LoCo/SWE-style proof-rig lessons into a product API: rank
workspace-local implementation files ahead of tests/docs/assets, keep tests as
verification hints, and refuse protected proof-rig paths.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".py",
    ".pyi",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".cs",
    ".rb",
    ".php",
}
DOC_SUFFIXES = {".md", ".mdx", ".rst", ".txt", ".adoc"}
ASSET_SUFFIXES = {
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".lock",
    ".map",
}
TEST_DIR_PARTS = {"__tests__", "test", "tests", "spec", "specs"}
FIXTURE_DIR_PARTS = {
    "__fixtures__",
    "__mocks__",
    "__snapshots__",
    "fixture",
    "fixtures",
    "mock",
    "mocks",
    "snapshot",
    "snapshots",
}
DOC_DIR_PARTS = {"doc", "docs", "documentation", "examples", "example"}
GENERATED_PATH_PARTS = {"generated", "gen", "coverage"}
MIGRATION_PATH_PARTS = {"migration", "migrations", "upgrade", "upgrades"}
BENCHMARK_PADDING_DIR = "__lazarus_benchmark_padding__"
BENCHMARK_PADDING_PATH = f"zzzz_{BENCHMARK_PADDING_DIR}/context_padding.txt"
PROTECTED_PROOF_RIG_PATHS = frozenset(
    {
        "scripts/run_swebench_pro_parity.py",
        "scripts/run_locobench_benchmark.py",
        "scripts/run_ruler_benchmark.py",
        "scripts/run_mrcr_benchmark.py",
        "scripts/interactive_memory_chat.py",
        "scripts/benchmark_jit_indexing.py",
        "David/central router.py",
        "David/decoding_prior_store.py",
    }
)
_PROTECTED_PROOF_RIG_PATHS_LOWER = frozenset(path.lower() for path in PROTECTED_PROOF_RIG_PATHS)
TRIAD_ADAPTER_TOKENS = {
    "adapter",
    "adapters",
    "database",
    "db",
    "storage",
    "repository",
    "repositories",
    "repo",
    "dao",
}
TRIAD_ENTRYPOINT_TOKENS = {
    "api",
    "controller",
    "controllers",
    "grpc",
    "handler",
    "handlers",
    "route",
    "routes",
    "rpc",
    "socket",
    "sockets",
    "service",
    "services",
}
TRIAD_LIFECYCLE_TOKENS = {
    "cleanup",
    "delete",
    "destroy",
    "destructor",
    "expire",
    "lifecycle",
    "purge",
    "remove",
}
TRIAD_ROLE_TOKENS = TRIAD_ADAPTER_TOKENS | TRIAD_ENTRYPOINT_TOKENS | TRIAD_LIFECYCLE_TOKENS
TERM_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class PathClassification:
    path: str
    kind: str
    rank: int
    is_workspace_local: bool
    is_source: bool = False
    is_test: bool = False
    is_docs: bool = False
    is_asset: bool = False
    is_generated: bool = False
    is_fixture: bool = False
    is_migration: bool = False
    is_protected: bool = False
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatchRouteEvidence:
    path: str
    reason: str
    score: float
    kind: str
    window_ids: tuple[str, ...] = ()
    route_index: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["details"] = dict(self.details)
        return data


@dataclass(frozen=True)
class PatchRoutePlan:
    selected_paths: tuple[str, ...]
    selected_tests: tuple[str, ...]
    source_hint_paths: tuple[str, ...]
    triad_augmented_paths: tuple[str, ...]
    windows_by_path: Mapping[str, tuple[Mapping[str, Any], ...]]
    evidence: tuple[PatchRouteEvidence, ...]
    reasons: Mapping[str, tuple[str, ...]]
    scores: Mapping[str, float]
    classifications: Mapping[str, PathClassification]
    rejected_paths: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_paths": list(self.selected_paths),
            "selected_tests": list(self.selected_tests),
            "source_hint_paths": list(self.source_hint_paths),
            "triad_augmented_paths": list(self.triad_augmented_paths),
            "windows_by_path": {
                path: [dict(window) for window in windows]
                for path, windows in self.windows_by_path.items()
            },
            "evidence": [item.to_dict() for item in self.evidence],
            "reasons": {path: list(items) for path, items in self.reasons.items()},
            "scores": dict(self.scores),
            "classifications": {
                path: classification.to_dict()
                for path, classification in self.classifications.items()
            },
            "rejected_paths": {
                path: list(reasons) for path, reasons in self.rejected_paths.items()
            },
        }


def normalize_path(path: str) -> str:
    normalized = str(path or "").replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def is_workspace_local_path(path: str) -> bool:
    normalized = normalize_path(path)
    if not normalized or normalized.startswith("/"):
        return False
    if re.match(r"^[A-Za-z]:", normalized):
        return False
    parts = PurePosixPath(normalized).parts
    return ".." not in parts


def normalized_path_parts(path: str) -> set[str]:
    return {part.lower() for part in PurePosixPath(normalize_path(path)).parts}


def path_name_tokens(path: str) -> set[str]:
    normalized = normalize_path(path)
    posix = PurePosixPath(normalized)
    tokens = set(re.split(r"[^a-z0-9]+", posix.name.lower()))
    tokens.update(re.split(r"[^a-z0-9]+", posix.stem.lower()))
    return {token for token in tokens if token}


def is_protected_path(path: str) -> bool:
    return normalize_path(path).lower() in _PROTECTED_PROOF_RIG_PATHS_LOWER


def is_benchmark_padding_path(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized == BENCHMARK_PADDING_PATH or BENCHMARK_PADDING_DIR in normalized_path_parts(
        normalized
    )


def is_test_like_path(path: str) -> bool:
    parts = normalized_path_parts(path)
    name = PurePosixPath(normalize_path(path)).name.lower()
    tokens = path_name_tokens(path)
    return bool(
        parts & TEST_DIR_PARTS
        or {"test", "tests", "spec", "specs"} & tokens
        or re.search(r"(^test_|_test\.|[._-](test|spec)\.)", name)
        or re.search(r"(\.test|\.spec)\.[a-z0-9]+$", name)
    )


def is_fixture_like_path(path: str) -> bool:
    name = PurePosixPath(normalize_path(path)).name.lower()
    tokens = path_name_tokens(path)
    return bool(
        normalized_path_parts(path) & FIXTURE_DIR_PARTS
        or tokens & {"fixture", "fixtures", "mock", "mocks", "snapshot", "snapshots"}
        or name.endswith((".snap", ".snapshot"))
    )


def is_doc_like_path(path: str) -> bool:
    normalized = normalize_path(path)
    return bool(
        normalized_path_parts(path) & DOC_DIR_PARTS
        or PurePosixPath(normalized).suffix.lower() in DOC_SUFFIXES
    )


def is_generated_like_path(path: str) -> bool:
    normalized = normalize_path(path)
    posix = PurePosixPath(normalized)
    name = posix.name.lower()
    tokens = path_name_tokens(path)
    return bool(
        normalized_path_parts(path) & GENERATED_PATH_PARTS
        or tokens & {"generated", "autogen", "bundle", "bundled", "min"}
        or name.endswith((".generated.js", ".generated.ts", ".min.js", ".bundle.js"))
    )


def is_migration_like_path(path: str) -> bool:
    return bool(normalized_path_parts(path) & MIGRATION_PATH_PARTS)


def is_asset_like_path(path: str) -> bool:
    return PurePosixPath(normalize_path(path)).suffix.lower() in ASSET_SUFFIXES


def is_implementation_source_path(path: str) -> bool:
    suffix = PurePosixPath(normalize_path(path)).suffix.lower()
    return bool(
        suffix in SOURCE_SUFFIXES
        and not is_test_like_path(path)
        and not is_fixture_like_path(path)
        and not is_doc_like_path(path)
        and not is_generated_like_path(path)
        and not is_migration_like_path(path)
        and not is_benchmark_padding_path(path)
        and not is_protected_path(path)
    )


def classify_path(path: str) -> PathClassification:
    normalized = normalize_path(path)
    suffix = PurePosixPath(normalized).suffix.lower()
    local = is_workspace_local_path(normalized)
    protected = is_protected_path(normalized)
    test = is_test_like_path(normalized)
    fixture = is_fixture_like_path(normalized)
    docs = is_doc_like_path(normalized)
    generated = is_generated_like_path(normalized)
    migration = is_migration_like_path(normalized)
    asset = is_asset_like_path(normalized)
    source = is_implementation_source_path(normalized)
    reasons: list[str] = []
    if not local:
        reasons.append("path is not workspace-local")
    if protected:
        reasons.append("protected proof-rig path")
    if source:
        kind = "source"
        rank = 0
        reasons.append("implementation source")
    elif protected:
        kind = "protected"
        rank = 100
    elif suffix in SOURCE_SUFFIXES and test:
        kind = "test"
        rank = 4
        reasons.append("verification test")
    elif suffix in SOURCE_SUFFIXES and fixture:
        kind = "fixture"
        rank = 5
        reasons.append("test fixture")
    elif suffix in SOURCE_SUFFIXES and generated:
        kind = "generated"
        rank = 6
        reasons.append("generated source")
    elif docs:
        kind = "docs"
        rank = 7
        reasons.append("documentation")
    elif asset:
        kind = "asset"
        rank = 8
        reasons.append("asset/static file")
    elif generated:
        kind = "generated"
        rank = 9
        reasons.append("generated artifact")
    elif migration:
        kind = "migration"
        rank = 9
        reasons.append("migration-like path")
    elif suffix in SOURCE_SUFFIXES:
        kind = "source_candidate"
        rank = 2
        reasons.append("source-like path")
    else:
        kind = "unknown"
        rank = 9
    if is_benchmark_padding_path(normalized):
        kind = "generated"
        rank = 99
        reasons.append("benchmark padding excluded")
    return PathClassification(
        path=normalized,
        kind=kind,
        rank=rank,
        is_workspace_local=local,
        is_source=source,
        is_test=test,
        is_docs=docs,
        is_asset=asset,
        is_generated=generated or is_benchmark_padding_path(normalized),
        is_fixture=fixture,
        is_migration=migration,
        is_protected=protected,
        reasons=tuple(reasons),
    )


def route_patch_targets(
    task: str,
    *,
    files: Mapping[str, str] | None = None,
    windows: Sequence[Mapping[str, Any]] | None = None,
    path_hints: Sequence[str] = (),
    limit: int = 8,
    source_hint_limit: int = 4,
    triad_limit: int = 4,
) -> PatchRoutePlan:
    route_windows = _coerce_windows(files or {}, windows or ())
    paths = list(dict.fromkeys([*_paths_from_windows(route_windows), *map(normalize_path, path_hints)]))
    classifications = {path: classify_path(path) for path in paths if path}
    rejected = {
        path: classification.reasons
        for path, classification in classifications.items()
        if classification.is_protected or not classification.is_workspace_local
    }
    task_terms = _terms(task)
    scored: list[tuple[int, float, int, str]] = []
    route_index = {path: index for index, path in enumerate(paths)}
    for path, classification in classifications.items():
        if path in rejected or is_benchmark_padding_path(path):
            continue
        score = _path_score(path, task_terms, route_windows.get(path, ()), classification)
        scored.append((classification.rank, score, route_index.get(path, 999_999), path))
    source_available = any(classifications[item[3]].is_source for item in scored)
    scored.sort(
        key=lambda item: (
            item[0] if source_available else min(item[0], 4),
            -item[1],
            item[2],
            item[3],
        )
    )
    selected_paths = tuple(path for _rank, _score, _index, path in scored[: max(1, int(limit))])
    selected_tests = tuple(
        path
        for _rank, _score, _index, path in sorted(
            (
                item
                for item in scored
                if classifications[item[3]].is_test or classifications[item[3]].kind == "test"
            ),
            key=lambda item: (-item[1], item[2], item[3]),
        )
    )
    source_hint_paths = _discover_source_hints(
        task_terms,
        scored,
        classifications,
        selected_paths=selected_paths,
        limit=source_hint_limit,
    )
    triad_augmented_paths = _discover_triad_paths(
        scored,
        classifications,
        selected_paths=(*selected_paths, *source_hint_paths),
        limit=triad_limit,
    )
    routed_paths = tuple(dict.fromkeys([*selected_paths, *source_hint_paths, *triad_augmented_paths]))
    windows_by_path = {
        path: route_windows.get(path, _synthetic_window(path, files or {})) for path in routed_paths
    }
    score_by_path = {path: score for _rank, score, _index, path in scored}
    reasons = {
        path: _route_reasons(path, classifications[path], score_by_path.get(path, 0.0), selected_tests)
        for path in routed_paths
    }
    evidence = tuple(
        PatchRouteEvidence(
            path=path,
            reason=reasons[path][0] if reasons.get(path) else "candidate retained",
            score=float(score_by_path.get(path, 0.0)),
            kind=classifications[path].kind,
            window_ids=tuple(
                str(window.get("window_id") or "")
                for window in windows_by_path.get(path, ())
                if window.get("window_id")
            ),
            route_index=route_index.get(path),
            details={"classification_rank": classifications[path].rank},
        )
        for path in routed_paths
    )
    return PatchRoutePlan(
        selected_paths=selected_paths,
        selected_tests=selected_tests,
        source_hint_paths=source_hint_paths,
        triad_augmented_paths=triad_augmented_paths,
        windows_by_path=windows_by_path,
        evidence=evidence,
        reasons=reasons,
        scores={path: float(score_by_path.get(path, 0.0)) for path in routed_paths},
        classifications=classifications,
        rejected_paths=rejected,
    )


def triad_role_for_path(path: str) -> str:
    tokens = normalized_path_parts(path) | path_name_tokens(path)
    if tokens & TRIAD_ADAPTER_TOKENS:
        return "adapter"
    if tokens & TRIAD_ENTRYPOINT_TOKENS:
        return "entrypoint"
    if tokens & TRIAD_LIFECYCLE_TOKENS:
        return "lifecycle"
    return ""


def _coerce_windows(
    files: Mapping[str, str],
    windows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    by_path: dict[str, list[Mapping[str, Any]]] = {}
    for path, text in files.items():
        normalized = normalize_path(path)
        by_path.setdefault(normalized, []).append(
            {"path": normalized, "text": text, "route_source": "file_map"}
        )
    for index, window in enumerate(windows):
        raw_path = (
            window.get("path")
            or window.get("source_path")
            or window.get("file_path")
            or window.get("rel_path")
            or ""
        )
        normalized = normalize_path(str(raw_path))
        if not normalized:
            continue
        enriched = dict(window)
        enriched.setdefault("path", normalized)
        enriched.setdefault("window_id", f"window-{index}")
        by_path.setdefault(normalized, []).append(enriched)
    return {path: tuple(items) for path, items in by_path.items()}


def _paths_from_windows(windows: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    return list(windows)


def _synthetic_window(path: str, files: Mapping[str, str]) -> tuple[Mapping[str, Any], ...]:
    if path not in files:
        return ()
    return ({"path": path, "text": files[path], "route_source": "file_map"},)


def _terms(*texts: str) -> set[str]:
    return {
        term.lower()
        for text in texts
        for term in TERM_RE.findall(str(text or ""))
        if len(term) >= 3
    } - {"and", "are", "for", "from", "into", "not", "the", "this", "that", "with"}


def _path_score(
    path: str,
    task_terms: set[str],
    windows: Sequence[Mapping[str, Any]],
    classification: PathClassification,
) -> float:
    path_terms = _terms(path, *path_name_tokens(path), *normalized_path_parts(path))
    text_terms = _terms(*(str(window.get("text", ""))[:8_000] for window in windows))
    overlap = task_terms & (path_terms | text_terms)
    score = float(len(overlap) * 25.0)
    if classification.is_source:
        score += 100.0
    elif classification.is_test:
        score += 35.0
    elif classification.is_docs:
        score += 8.0
    elif classification.is_asset:
        score -= 15.0
    elif classification.is_generated:
        score -= 25.0
    for window in windows:
        for field_name, weight in (("activation_score", 8.0), ("lexical_score", 1.0), ("score", 1.0)):
            try:
                score += max(0.0, float(window.get(field_name, 0.0))) * weight
            except (TypeError, ValueError):
                continue
    return score


def _discover_source_hints(
    task_terms: set[str],
    scored: Sequence[tuple[int, float, int, str]],
    classifications: Mapping[str, PathClassification],
    *,
    selected_paths: Sequence[str],
    limit: int,
) -> tuple[str, ...]:
    selected = set(selected_paths)
    hints: list[tuple[float, int, str]] = []
    for _rank, score, index, path in scored:
        if path in selected or not classifications[path].is_source:
            continue
        overlap = task_terms & (_terms(path) | normalized_path_parts(path) | path_name_tokens(path))
        if not overlap:
            continue
        hints.append((score + len(overlap) * 10.0, index, path))
    hints.sort(key=lambda item: (-item[0], item[1], item[2]))
    return tuple(path for _score, _index, path in hints[: max(0, int(limit))])


def _discover_triad_paths(
    scored: Sequence[tuple[int, float, int, str]],
    classifications: Mapping[str, PathClassification],
    *,
    selected_paths: Sequence[str],
    limit: int,
) -> tuple[str, ...]:
    selected = set(selected_paths)
    triad: list[tuple[str, float, int, str]] = []
    for _rank, score, index, path in scored:
        if path in selected or not classifications[path].is_source:
            continue
        role = triad_role_for_path(path)
        if role:
            triad.append((role, score + 18.0, index, path))
    triad.sort(key=lambda item: (item[0], -item[1], item[2], item[3]))
    return tuple(path for _role, _score, _index, path in triad[: max(0, int(limit))])


def _route_reasons(
    path: str,
    classification: PathClassification,
    score: float,
    selected_tests: Sequence[str],
) -> tuple[str, ...]:
    reasons = list(classification.reasons)
    if classification.is_source:
        reasons.append("ranked ahead of tests/docs/assets for patch target routing")
    if classification.is_test:
        reasons.append("retained as verification hint")
    if selected_tests and classification.is_source:
        reasons.append("tests preserved separately in selected_tests")
    reasons.append(f"route_score={score:.3f}")
    return tuple(dict.fromkeys(reasons))
