#!/usr/bin/env python3
"""Run a Lazarus SWE-bench Pro parity benchmark slice.

This runner keeps the LoCoBench snapshot report contract while adapting the
coding loop to SWE-bench Pro's repo-level, multi-file tasks. It deliberately
imports LoCo primitives instead of copying the LoCo runner wholesale.
"""

# ruff: noqa: E402 - local script imports require sys.path bootstrapping first.

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT, Path(__file__).resolve().parent):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from benchmark_jit_indexing import jit_index_dataset_windows, release_cuda_memory

from run_locobench_benchmark import (
    COMMON_LOCAL_IDENTIFIERS,
    DEFAULT_DIFF_RESERVED_TOKENS,
    DEFAULT_KV_DECODE_HEARTBEAT_SECONDS,
    DEFAULT_KV_DECODE_PAYLOAD_TOKEN_CAP,
    DEFAULT_KV_DECODE_STALL_SECONDS,
    DEFAULT_KV_DECODE_TIMEOUT_SECONDS,
    DEFAULT_KV_HOT_BUDGET_MIB,
    DEFAULT_KV_MAX_NEW_TOKENS,
    DEFAULT_KV_REPLACEMENT_SPILLOVER_TOKENS,
    DIFF_PROTOCOL_CONSTRAINT,
    SEARCH_REPLACE_RE,
    STRUCTURAL_ANCHOR,
    TASK_TYPE,
    TOKEN_BIN,
    DependencyWindow,
    LoCoBenchCase,
    apply_search_replace_blocks,
    c11_case_style_violations,
    canonicalize_search_replace_output,
    command_prediction,
    dependency_generation_bundle,
    dependency_name_terms,
    dialect_literal_violations,
    dialect_lock_telemetry,
    extract_identifiers,
    identifiers_from_windows,
    infer_dialect_lock_for_source,
    kv_direct_prediction,
    local_prediction,
    phase3_anchored_query,
    phase3_metadata,
    readout_terms,
    route_domain_score,
    route_domain_weight,
    route_hot_windows,
    route_path_depth_penalty,
    stringify,
    unsafe_identifier_violations,
    visible_prompt,
)

from chuk_lazarus.benchmarks.baselines import COMPARISON_BASELINES
from chuk_lazarus.memory_config import DynamicAllocatorConfig

DATASET_ID = "ScaleAI/SWE-bench_Pro"
DATASET_SPLIT = "test"
BENCHMARK_NAME = "ScaleAI/SWE-bench_Pro"
SNAPSHOT_DATE = datetime.now(timezone.utc).date().isoformat()
DEFAULT_OUTPUT_PATH = Path(".benchmarks") / f"swebench_pro_parity_{SNAPSHOT_DATE}.json"
DEFAULT_CSV_PATH = Path(".benchmarks") / f"swebench_pro_parity_{SNAPSHOT_DATE}.csv"
DEFAULT_JIT_DIR = Path(".benchmarks") / "jit"
DEFAULT_REPO_CACHE = Path(".benchmarks") / "swebench_pro_repos"
RULER_JIT_INDEXING_COMPAT_SIGNAL = "RULER_JIT_INDEXING_COMPAT_ACTIVE"
RULER_JIT_INDEXING_IMPLEMENTATION = "benchmark_jit_indexing.jit_index_dataset_windows"
RULER_JIT_INDEXING_ENTRYPOINT = "scripts/run_ruler_benchmark.py"
DEFAULT_TARGET_FILE_BUDGET = 5
DEFAULT_WINDOWS_PER_TARGET = 3
MAX_PHASE2_HOT_WINDOWS = 15
DEFAULT_MAX_CONTEXT_FILES = 80
DEFAULT_MAX_CONTEXT_BYTES = 2_500_000
DEFAULT_TARGET_CONTEXT_TOKENS = 1_000_000
REQUIRED_KV_RUNTIME_PACKAGES = ("torch", "transformers", "accelerate", "einops")
MIN_REQUIRED_VRAM_GIB = 24.0
REQUIRED_GPU_NAME_FRAGMENT = "5090"
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
PROTOCOL_REPLACEMENT_MARKER_RE = re.compile(r"<<<<|====|\bSEARCH\b|\bREPLACE\b")
SKIP_DIR_PARTS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "target",
    "vendor",
}
TEST_DIR_PARTS = {
    "__tests__",
    "test",
    "tests",
    "spec",
    "specs",
}
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
TRIAD_STOP_TERMS = {
    "index",
    "main",
    "src",
    "lib",
    "app",
    "test",
    "tests",
    "js",
    "ts",
    "py",
    "go",
}


@dataclass(frozen=True)
class SWEBenchProCase:
    row_index: int
    instruction: str
    files: dict[str, str]
    expected_patch: str
    raw: dict[str, Any]
    repo: str
    instance_id: str
    base_commit: str
    repo_language: str
    requirements: str
    interface: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    before_repo_set_cmd: str
    selected_test_files_to_run: list[str]
    dockerhub_tag: str
    checkout_path: str


@dataclass(frozen=True)
class StrictBlock:
    path: str
    old: str
    new: str


@dataclass(frozen=True)
class TestGateResult:
    passed: bool
    skipped: bool
    required: bool
    reason: str
    command: str
    returncode: int | None
    stdout_tail: str
    stderr_tail: str
    elapsed_ms: float
    selected_tests: list[str]


class SegmentedKVDirectUnavailable(RuntimeError):
    """Raised when segmented KV-direct cannot fail safely into generation."""

    def __init__(self, reason: str, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.metadata = metadata or {}


class RuntimePreflightError(RuntimeError):
    """Raised when mandatory KV-direct runtime checks fail before generation."""

    def __init__(self, reason: str, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.metadata = metadata or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--split", default=DATASET_SPLIT)
    parser.add_argument(
        "--local-dataset",
        type=Path,
        default=None,
        help="Optional JSON, JSONL, or CSV dataset fallback.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        default=None,
        help="JSONL predictions artifact for official SWE-bench Pro evaluation.",
    )
    parser.add_argument("--jit-output-dir", type=Path, default=DEFAULT_JIT_DIR)
    parser.add_argument("--jit-model-path", default="")
    parser.add_argument("--jit-device", default="")
    parser.add_argument(
        "--jit-reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--jit-batch-size", type=int, default=0)
    parser.add_argument("--window-tokens", type=int, default=512)
    parser.add_argument("--activation-overlap-tokens", type=int, default=128)
    parser.add_argument("--prediction-command", default="")
    parser.add_argument(
        "--prediction-mode",
        choices=("kv_direct", "row_prediction", "empty", "gold_patch"),
        default="kv_direct",
    )
    parser.add_argument("--diff-reserved-tokens", type=int, default=DEFAULT_DIFF_RESERVED_TOKENS)
    parser.add_argument("--kv-hot-budget-mib", type=int, default=DEFAULT_KV_HOT_BUDGET_MIB)
    parser.add_argument("--kv-max-new-tokens", type=int, default=DEFAULT_KV_MAX_NEW_TOKENS)
    parser.add_argument(
        "--kv-replacement-spillover-tokens",
        type=int,
        default=DEFAULT_KV_REPLACEMENT_SPILLOVER_TOKENS,
    )
    parser.add_argument(
        "--kv-decode-payload-token-cap",
        type=int,
        default=DEFAULT_KV_DECODE_PAYLOAD_TOKEN_CAP,
        help=(
            "Safety cap for replacement+spillover tokens in one KV-direct "
            "decode segment. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--kv-decode-timeout-seconds",
        type=int,
        default=DEFAULT_KV_DECODE_TIMEOUT_SECONDS,
        help="Wall-clock timeout for one KV-direct decode segment. Use 0 to disable.",
    )
    parser.add_argument(
        "--kv-decode-stall-seconds",
        type=int,
        default=DEFAULT_KV_DECODE_STALL_SECONDS,
        help="Force a close marker after this many seconds without decoded text progress.",
    )
    parser.add_argument(
        "--kv-decode-heartbeat-seconds",
        type=int,
        default=DEFAULT_KV_DECODE_HEARTBEAT_SECONDS,
        help="Emit KV-direct decode progress telemetry at this interval. Use 0 to disable.",
    )
    parser.add_argument("--repo-cache", type=Path, default=DEFAULT_REPO_CACHE)
    parser.add_argument(
        "--clone-behavior",
        choices=("auto", "never", "refresh"),
        default="auto",
        help="How to populate repo cache before checking out base_commit.",
    )
    parser.add_argument("--max-context-files", type=int, default=DEFAULT_MAX_CONTEXT_FILES)
    parser.add_argument("--max-context-bytes", type=int, default=DEFAULT_MAX_CONTEXT_BYTES)
    parser.add_argument("--target-context-tokens", type=int, default=DEFAULT_TARGET_CONTEXT_TOKENS)
    parser.add_argument(
        "--exact-context-packing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Tokenizer-pack and pad context to target-context-tokens. "
            "Use --no-exact-context-packing for official 1M benchmark runs "
            "that should not spend time calculating an exact 1,000,000-token context."
        ),
    )
    parser.add_argument("--target-file-budget", type=int, default=DEFAULT_TARGET_FILE_BUDGET)
    parser.add_argument("--windows-per-target", type=int, default=DEFAULT_WINDOWS_PER_TARGET)
    parser.add_argument("--max-system2-retries", type=int, default=0)
    parser.add_argument("--test-command", default="")
    parser.add_argument("--test-template", default="")
    parser.add_argument("--official-eval-script", type=Path, default=None)
    parser.add_argument("--official-eval-scripts-dir", type=Path, default=None)
    parser.add_argument("--official-docker-image-repo", default="jefzda/sweap-images")
    parser.add_argument("--official-eval-timeout-seconds", type=int, default=3600)
    parser.add_argument("--dockerhub-username", default="")
    parser.add_argument("--use-local-docker", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ensure-runtime-deps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-rtx5090", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-test-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-gold-path-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow gold patch paths to seed routing. Off by default to avoid leakage.",
    )
    return parser.parse_args()


def missing_runtime_packages() -> list[str]:
    return [
        package
        for package in REQUIRED_KV_RUNTIME_PACKAGES
        if importlib.util.find_spec(package) is None
    ]


def ensure_runtime_dependencies(args: argparse.Namespace) -> None:
    if args.prediction_mode != "kv_direct":
        args.runtime_dependency_setup = {
            "active": False,
            "reason": "prediction_mode_is_not_kv_direct",
        }
        return
    missing = missing_runtime_packages()
    metadata: dict[str, Any] = {
        "active": bool(args.ensure_runtime_deps),
        "required_packages": list(REQUIRED_KV_RUNTIME_PACKAGES),
        "missing_before_install": list(missing),
        "python": sys.executable,
        "installer": "",
        "command": [],
        "returncode": 0,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if missing and args.ensure_runtime_deps:
        uv = shutil.which("uv")
        command = (
            [uv, "pip", "install", "--python", sys.executable, *missing]
            if uv
            else [sys.executable, "-m", "pip", "install", *missing]
        )
        metadata["installer"] = "uv" if uv else "pip"
        metadata["command"] = command
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=1800,
        )
        metadata["returncode"] = int(completed.returncode)
        metadata["stdout_tail"] = tail(completed.stdout)
        metadata["stderr_tail"] = tail(completed.stderr)
        importlib.invalidate_caches()
        missing = missing_runtime_packages()
        metadata["missing_after_install"] = list(missing)
        args.runtime_dependency_setup = metadata
        if completed.returncode != 0 or missing:
            raise RuntimePreflightError("runtime_dependency_install_failed", metadata)
        return
    metadata["missing_after_install"] = list(missing)
    args.runtime_dependency_setup = metadata
    if missing:
        raise RuntimePreflightError("runtime_dependency_missing", metadata)


def kv_runtime_preflight(args: argparse.Namespace) -> None:
    if args.prediction_mode != "kv_direct":
        args.runtime_preflight = {
            "active": False,
            "reason": "prediction_mode_is_not_kv_direct",
        }
        return
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - surfaced as structured preflight.
        raise RuntimePreflightError(
            "torch_import_failed",
            {"error": f"{type(exc).__name__}: {exc}"},
        ) from exc
    metadata: dict[str, Any] = {
        "active": True,
        "torch_version": getattr(torch, "__version__", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "required_gpu_name_fragment": REQUIRED_GPU_NAME_FRAGMENT,
        "required_vram_gib": MIN_REQUIRED_VRAM_GIB,
        "require_rtx5090": bool(args.require_rtx5090),
    }
    if not torch.cuda.is_available():
        args.runtime_preflight = metadata
        raise RuntimePreflightError("cuda_unavailable", metadata)
    try:
        props = torch.cuda.get_device_properties(0)
        device_name = str(getattr(props, "name", ""))
        total_vram_gib = float(getattr(props, "total_memory", 0)) / float(1024**3)
        metadata.update(
            {
                "device_index": 0,
                "device_name": device_name,
                "total_vram_gib": total_vram_gib,
            }
        )
    except Exception as exc:  # noqa: BLE001
        args.runtime_preflight = metadata
        raise RuntimePreflightError(
            "cuda_device_properties_unavailable",
            {**metadata, "error": f"{type(exc).__name__}: {exc}"},
        ) from exc
    if total_vram_gib < MIN_REQUIRED_VRAM_GIB:
        args.runtime_preflight = metadata
        raise RuntimePreflightError("insufficient_vram", metadata)
    if args.require_rtx5090 and REQUIRED_GPU_NAME_FRAGMENT not in device_name.lower():
        args.runtime_preflight = metadata
        raise RuntimePreflightError("unsupported_gpu", metadata)
    try:
        from chuk_lazarus.inference.backends._torch_residual_bounded import (
            materialize_kv_direct,
        )

        _ = materialize_kv_direct
        probe = torch.empty((1,), device="cuda")
        del probe
        torch.cuda.synchronize()
        metadata["kv_direct_kernel_import"] = True
        metadata["cuda_allocation_probe"] = True
    except Exception as exc:  # noqa: BLE001
        args.runtime_preflight = metadata
        raise RuntimePreflightError(
            "kv_direct_kernel_initialization_failed",
            {**metadata, "error": f"{type(exc).__name__}: {exc}"},
        ) from exc
    args.runtime_preflight = metadata


def load_context_tokenizer(args: argparse.Namespace) -> None:
    args._context_tokenizer = None
    args.context_token_packing = {
        "active": False,
        "target_context_tokens": int(args.target_context_tokens),
        "tokenizer": "",
        "reason": "not_loaded",
    }
    if args.prediction_mode != "kv_direct":
        args.context_token_packing["reason"] = "prediction_mode_is_not_kv_direct"
        return
    if not bool(getattr(args, "exact_context_packing", True)):
        args.context_token_packing = {
            "active": False,
            "target_context_tokens": int(args.target_context_tokens),
            "tokenizer": "",
            "reason": "exact_context_packing_disabled_for_official_1m_benchmark",
            "exact_context_packing_enabled": False,
            "official_1m_benchmark_mode": True,
        }
        return
    try:
        from transformers import AutoTokenizer

        tokenizer_id = (
            args.jit_model_path
            or os.environ.get("LAZARUS_MODEL")
            or "google/gemma-2-2b-it"
        )
        args._context_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            use_fast=True,
            local_files_only=bool(args.jit_model_path or os.environ.get("LAZARUS_MODEL")),
        )
        args.context_token_packing = {
            "active": True,
            "target_context_tokens": int(args.target_context_tokens),
            "tokenizer": str(tokenizer_id),
            "reason": "gemma_tokenizer_loaded",
        }
    except Exception as exc:  # noqa: BLE001 - packing falls back to byte cap.
        args.context_token_packing = {
            "active": False,
            "target_context_tokens": int(args.target_context_tokens),
            "tokenizer": "",
            "reason": "tokenizer_unavailable_byte_cap_fallback",
            "error": f"{type(exc).__name__}: {exc}",
        }


def token_count_for_context(args: argparse.Namespace, text: str) -> int | None:
    tokenizer = getattr(args, "_context_tokenizer", None)
    if tokenizer is None:
        return None
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:  # noqa: BLE001 - caller falls back to byte cap for this file.
        return None


def load_benchmark_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.local_dataset:
        args.loader = "local_dataset"
        args.resolved_split = str(args.local_dataset)
        return load_local_dataset(args.local_dataset)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it or pass --local-dataset."
        ) from exc
    try:
        dataset = load_dataset(args.dataset_id, split=args.split)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Failed to load Hugging Face dataset {args.dataset_id!r} "
            f"split={args.split!r}: {exc}"
        ) from exc
    args.loader = "datasets.load_dataset"
    args.resolved_split = args.split
    return [dict(row) for row in dataset]


def load_local_dataset(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("cases", data.get("rows", data.get("data", [])))
        if not isinstance(data, list):
            raise SystemExit(f"JSON dataset must contain a list of rows: {path}")
        return [dict(row) for row in data]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise SystemExit(f"Unsupported local dataset format: {path}")


def as_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                value = [value]
    if value is None:
        return []
    if isinstance(value, list):
        return [stringify(item) for item in value if stringify(item)]
    text = stringify(value).strip()
    return [text] if text else []


def repo_cache_path(args: argparse.Namespace, repo: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", repo).strip("_") or "repo"
    return args.repo_cache / safe


def run_git(args: list[str], cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def checkout_absolute_path(case: SWEBenchProCase) -> Path:
    checkout = Path(case.checkout_path)
    return checkout if checkout.is_absolute() else REPO_ROOT / checkout


def ensure_repo_checkout(row: dict[str, Any], args: argparse.Namespace) -> tuple[Path | None, dict[str, Any]]:
    repo = stringify(row.get("repo"))
    base_commit = stringify(row.get("base_commit"))
    if not repo:
        return None, {"available": False, "reason": "row_missing_repo"}
    path = repo_cache_path(args, repo)
    meta: dict[str, Any] = {
        "available": False,
        "repo": repo,
        "path": str(path),
        "base_commit": base_commit,
        "clone_behavior": args.clone_behavior,
    }
    if args.clone_behavior == "refresh" and path.exists():
        shutil.rmtree(path)
    if not path.exists():
        if args.clone_behavior == "never":
            meta["reason"] = "clone_behavior_never_and_cache_missing"
            return None, meta
        path.parent.mkdir(parents=True, exist_ok=True)
        clone_url = repo if "://" in repo or repo.endswith(".git") else f"https://github.com/{repo}.git"
        completed = run_git(["clone", "--no-tags", clone_url, str(path)], timeout=900)
        if completed.returncode != 0:
            meta.update(
                {
                    "reason": "git_clone_failed",
                    "returncode": completed.returncode,
                    "stderr_tail": tail(completed.stderr),
                }
            )
            return None, meta
    if base_commit:
        fetch = run_git(["fetch", "--no-tags", "origin", base_commit], cwd=path, timeout=600)
        checkout = run_git(["checkout", "--force", base_commit], cwd=path, timeout=300)
        if checkout.returncode != 0:
            meta.update(
                {
                    "reason": "git_checkout_failed",
                    "fetch_returncode": fetch.returncode,
                    "returncode": checkout.returncode,
                    "stderr_tail": tail(checkout.stderr),
                }
            )
            return None, meta
    clean = run_git(["clean", "-fdx"], cwd=path, timeout=300)
    reset = run_git(["reset", "--hard"], cwd=path, timeout=300)
    meta.update(
        {
            "available": reset.returncode == 0,
            "reason": "" if reset.returncode == 0 else "git_reset_failed",
            "clean_returncode": clean.returncode,
            "reset_returncode": reset.returncode,
            "stderr_tail": tail(reset.stderr),
        }
    )
    return (path if reset.returncode == 0 else None), meta


def normalized_path_parts(rel_path: str) -> set[str]:
    return {part.lower() for part in Path(rel_path.replace("\\", "/")).parts}


def path_name_tokens(rel_path: str) -> set[str]:
    path = Path(rel_path.replace("\\", "/"))
    name = path.name.lower()
    stem = path.stem.lower()
    tokens = set(re.split(r"[^a-z0-9]+", name))
    tokens.update(re.split(r"[^a-z0-9]+", stem))
    return {token for token in tokens if token}


def is_benchmark_padding_path(rel_path: str) -> bool:
    normalized = rel_path.replace("\\", "/")
    return normalized == BENCHMARK_PADDING_PATH or BENCHMARK_PADDING_DIR in normalized_path_parts(normalized)


def is_test_like_path(rel_path: str) -> bool:
    parts = normalized_path_parts(rel_path)
    name = Path(rel_path.replace("\\", "/")).name.lower()
    tokens = path_name_tokens(rel_path)
    if parts & TEST_DIR_PARTS:
        return True
    if {"test", "tests", "spec", "specs"} & tokens:
        return True
    return bool(
        re.search(r"(^test_|_test\.|[._-](test|spec)\.)", name)
        or re.search(r"(\.test|\.spec)\.[a-z0-9]+$", name)
    )


def is_fixture_like_path(rel_path: str) -> bool:
    parts = normalized_path_parts(rel_path)
    name = Path(rel_path.replace("\\", "/")).name.lower()
    tokens = path_name_tokens(rel_path)
    return bool(
        parts & FIXTURE_DIR_PARTS
        or tokens & {"fixture", "fixtures", "mock", "mocks", "snapshot", "snapshots"}
        or name.endswith((".snap", ".snapshot"))
    )


def is_doc_like_path(rel_path: str) -> bool:
    path = Path(rel_path.replace("\\", "/"))
    return bool(normalized_path_parts(rel_path) & DOC_DIR_PARTS or path.suffix.lower() in DOC_SUFFIXES)


def is_generated_like_path(rel_path: str) -> bool:
    path = Path(rel_path.replace("\\", "/"))
    name = path.name.lower()
    tokens = path_name_tokens(rel_path)
    return bool(
        normalized_path_parts(rel_path) & GENERATED_PATH_PARTS
        or tokens & {"generated", "autogen", "bundle", "bundled", "min"}
        or name.endswith((".generated.js", ".generated.ts", ".min.js", ".bundle.js"))
    )


def is_migration_like_path(rel_path: str) -> bool:
    return bool(normalized_path_parts(rel_path) & MIGRATION_PATH_PARTS)


def is_asset_like_path(rel_path: str) -> bool:
    return Path(rel_path.replace("\\", "/")).suffix.lower() in ASSET_SUFFIXES


def is_implementation_source_path(rel_path: str) -> bool:
    suffix = Path(rel_path.replace("\\", "/")).suffix.lower()
    return bool(
        suffix in SOURCE_SUFFIXES
        and not is_test_like_path(rel_path)
        and not is_fixture_like_path(rel_path)
        and not is_doc_like_path(rel_path)
        and not is_generated_like_path(rel_path)
        and not is_migration_like_path(rel_path)
        and not is_benchmark_padding_path(rel_path)
    )


def patch_target_rank(rel_path: str) -> int:
    if is_benchmark_padding_path(rel_path):
        return 99
    if is_implementation_source_path(rel_path):
        return 0
    if Path(rel_path.replace("\\", "/")).suffix.lower() in SOURCE_SUFFIXES:
        if is_test_like_path(rel_path):
            return 4
        if is_fixture_like_path(rel_path):
            return 5
        if is_generated_like_path(rel_path):
            return 6
        return 2
    if is_doc_like_path(rel_path):
        return 7
    if is_asset_like_path(rel_path):
        return 8
    return 9


def prioritize_patch_target_paths(paths: list[str], files: dict[str, str], limit: int) -> tuple[list[str], dict[str, Any]]:
    seen: list[str] = []
    for path in paths:
        if path and path in files and not is_benchmark_padding_path(path) and path not in seen:
            seen.append(path)
    route_index = {path: index for index, path in enumerate(seen)}
    source_available = any(is_implementation_source_path(path) for path in seen)
    prioritized = sorted(
        seen,
        key=lambda path: (
            patch_target_rank(path) if source_available else min(patch_target_rank(path), 4),
            route_index.get(path, 999_999),
            path,
        ),
    )
    limited = prioritized[: max(1, int(limit))]
    return limited, {
        "active": True,
        "source_available": bool(source_available),
        "original_order": seen,
        "prioritized_order": prioritized,
        "implementation_source_paths": [
            path for path in prioritized if is_implementation_source_path(path)
        ],
        "deprioritized_non_source_paths": [
            path for path in prioritized if not is_implementation_source_path(path)
        ],
        "rank_by_path": {path: patch_target_rank(path) for path in prioritized},
    }


def exact_token_text(args: argparse.Namespace, header: str, body_unit: str, target_tokens: int) -> tuple[str, int]:
    tokenizer = getattr(args, "_context_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("context_tokenizer_unavailable")
    target_tokens = max(0, int(target_tokens))
    if target_tokens == 0:
        return "", 0

    header_ids = tokenizer.encode(header, add_special_tokens=False)
    if len(header_ids) >= target_tokens:
        text = tokenizer.decode(header_ids[:target_tokens], skip_special_tokens=False)
        count = token_count_for_context(args, text)
        if count == target_tokens:
            return text, int(count)
        raise RuntimeError("exact_context_padding_header_roundtrip_failed")

    needed = target_tokens - len(header_ids)
    body_ids = tokenizer.encode(body_unit, add_special_tokens=False)
    if not body_ids:
        raise RuntimeError("exact_context_padding_unit_tokenized_empty")
    repeats = math.ceil(needed / len(body_ids)) + 2
    repeated_ids = tokenizer.encode(body_unit * repeats, add_special_tokens=False)
    while len(repeated_ids) < needed:
        repeats *= 2
        repeated_ids = tokenizer.encode(body_unit * repeats, add_special_tokens=False)
    text = tokenizer.decode(
        [*header_ids, *repeated_ids[:needed]],
        skip_special_tokens=False,
    )
    count = token_count_for_context(args, text)
    if count == target_tokens:
        return text, int(count)
    if count is not None and count > target_tokens:
        text = tokenizer.decode(
            tokenizer.encode(text, add_special_tokens=False)[:target_tokens],
            skip_special_tokens=False,
        )
        count = token_count_for_context(args, text)
        if count == target_tokens:
            return text, int(count)
    raise RuntimeError(
        f"exact_context_padding_roundtrip_failed: requested={target_tokens} observed={count}"
    )


def deterministic_context_padding(args: argparse.Namespace, missing_tokens: int) -> tuple[str, int]:
    header = (
        "LAZARUS BENCHMARK CONTEXT PADDING\n"
        "This deterministic section is not repository source, not a test, not a patch, "
        "and not task evidence. It exists only to verify the requested token budget. "
        "Ignore this padding when selecting files or generating edits.\n"
    )
    body_unit = (
        "neutral benchmark padding line with no project identifier and no solution signal.\n"
    )
    return exact_token_text(args, header, body_unit, int(missing_tokens))


def term_variants(terms: set[str]) -> set[str]:
    expanded = set(terms)
    for term in list(terms):
        if len(term) > 3 and term.endswith("ies"):
            expanded.add(term[:-3] + "y")
        if len(term) > 3 and term.endswith("es"):
            expanded.add(term[:-2])
        if len(term) > 3 and term.endswith("s"):
            expanded.add(term[:-1])
    return {term for term in expanded if len(term) >= 2}


def list_repo_files(checkout: Path, args: argparse.Namespace, row: dict[str, Any]) -> dict[str, str]:
    selected = set(as_list(row.get("selected_test_files_to_run")))
    routing_text = "\n".join(
        [
            stringify(row.get("problem_statement")),
            stringify(row.get("requirements")),
            stringify(row.get("interface")),
            "\n".join(selected),
        ]
    )
    mentions = path_like_mentions(routing_text)
    routing_terms = term_variants(dependency_name_terms(routing_text))
    tracked = run_git(["ls-files"], cwd=checkout, timeout=120)
    if tracked.returncode == 0:
        candidates = [line.strip() for line in tracked.stdout.splitlines() if line.strip()]
    else:
        candidates = [
            str(path.relative_to(checkout)).replace("\\", "/")
            for path in checkout.rglob("*")
            if path.is_file()
        ]
    scored: list[tuple[float, str]] = []
    for rel_path in candidates:
        path = Path(rel_path)
        parts = normalized_path_parts(rel_path)
        suffix = path.suffix.lower()
        if parts & SKIP_DIR_PARTS:
            continue
        if suffix not in SOURCE_SUFFIXES and rel_path not in mentions:
            continue
        score = 0.0
        if rel_path in mentions or path.name in mentions:
            score += 1000.0
        if rel_path in selected or path.name in selected:
            score += 100.0
        if suffix in SOURCE_SUFFIXES:
            score += 20.0
        if is_implementation_source_path(rel_path):
            score += 60.0
            path_overlap = term_variants(dependency_name_terms(rel_path)) & routing_terms
            if path_overlap:
                score += 55.0 * len(path_overlap)
                if (normalized_path_parts(rel_path) | path_name_tokens(rel_path)) & TRIAD_ROLE_TOKENS:
                    score += 35.0
        if any(part in {"src", "lib", "app", "packages"} for part in parts):
            score += 8.0
        if is_test_like_path(rel_path):
            score -= 40.0
        if is_fixture_like_path(rel_path) or is_generated_like_path(rel_path):
            score -= 75.0
        if is_migration_like_path(rel_path):
            score -= 65.0
        if is_doc_like_path(rel_path) or is_asset_like_path(rel_path):
            score -= 100.0
        scored.append((route_domain_score(score, rel_path), rel_path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    files: dict[str, str] = {}
    total_bytes = 0
    total_tokens = 0
    token_aware = bool(getattr(args, "_context_tokenizer", None))
    target_context_tokens = max(1, int(getattr(args, "target_context_tokens", 1_000_000)))
    skipped_for_token_cap = 0
    padding_added = False
    padding_tokens = 0
    padding_error = ""
    for _score, rel_path in scored:
        if len(files) >= int(args.max_context_files):
            break
        abs_path = checkout / rel_path
        try:
            data = abs_path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:4096]:
            continue
        if total_bytes + len(data) > int(args.max_context_bytes):
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        file_tokens = token_count_for_context(args, text) if token_aware else None
        if (
            file_tokens is not None
            and total_tokens + int(file_tokens) > target_context_tokens
        ):
            remaining = target_context_tokens - total_tokens
            tokenizer = getattr(args, "_context_tokenizer", None)
            if remaining <= 0 or tokenizer is None:
                skipped_for_token_cap += 1
                continue
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
                text = tokenizer.decode(ids[:remaining], skip_special_tokens=False)
                data = text.encode("utf-8")
                file_tokens = remaining
            except Exception:  # noqa: BLE001 - fall back to skipping oversize text.
                skipped_for_token_cap += 1
                continue
        files[rel_path] = text
        total_bytes += len(data)
        if file_tokens is not None:
            total_tokens += int(file_tokens)
        if token_aware and total_tokens >= target_context_tokens:
            break
    repo_file_count = len(files)
    if token_aware and total_tokens < target_context_tokens:
        missing_tokens = target_context_tokens - total_tokens
        try:
            padding_text, padding_tokens = deterministic_context_padding(args, missing_tokens)
            files[BENCHMARK_PADDING_PATH] = padding_text
            total_tokens += int(padding_tokens)
            total_bytes += len(padding_text.encode("utf-8"))
            padding_added = True
        except Exception as exc:  # noqa: BLE001 - captured so the run can fail closed with telemetry.
            padding_error = f"{type(exc).__name__}: {exc}"
    args.context_token_packing = {
        **getattr(args, "context_token_packing", {}),
        "token_aware_file_packing": token_aware,
        "selected_file_count": len(files),
        "selected_repo_file_count": int(repo_file_count),
        "selected_context_tokens": int(total_tokens) if token_aware else None,
        "target_context_tokens": int(target_context_tokens),
        "exact_1m_context": bool(token_aware and total_tokens == target_context_tokens),
        "skipped_for_token_cap": int(skipped_for_token_cap),
        "padding_added": bool(padding_added),
        "padding_path": BENCHMARK_PADDING_PATH if padding_added else "",
        "padding_tokens": int(padding_tokens),
        "padding_error": padding_error,
        "padding_policy": (
            "deterministic_non_leaking_benchmark_padding_excluded_from_routing"
            if padding_added
            else "not_needed"
        ),
        "legacy_max_context_bytes": int(args.max_context_bytes),
    }
    return files


def path_like_mentions(text: str) -> set[str]:
    mentions = set()
    for match in re.finditer(r"[\w./-]+\.[A-Za-z0-9]+", text or ""):
        mentions.add(match.group(0).strip("`'\".,:;()[]{}"))
    return {mention for mention in mentions if "/" in mention or "." in mention}


def gold_patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in (patch or "").splitlines():
        if line.startswith(("--- a/", "+++ b/")):
            path = line[6:].strip()
        elif line.startswith("diff --git "):
            parts = line.split()
            path = parts[2][2:] if len(parts) >= 3 and parts[2].startswith("a/") else ""
        else:
            path = ""
        if path and path != "/dev/null" and path not in paths:
            paths.append(path)
    return paths


def extract_case(row: dict[str, Any], row_index: int, args: argparse.Namespace) -> SWEBenchProCase:
    checkout_path, checkout_meta = ensure_repo_checkout(row, args)
    files = list_repo_files(checkout_path, args, row) if checkout_path else {}
    context_meta = getattr(args, "context_token_packing", {})
    if args.prediction_mode == "kv_direct" and not context_meta.get("exact_1m_context"):
        args.context_token_packing = {
            **context_meta,
            "exact_1m_context_required": False,
            "exact_1m_context_warning": (
                "official_swebench_pro_1m_lane_does_not_abort_on_nonexact_pack"
            ),
        }
        print(
            "SWE-bench Pro 1M context packing note: continuing without exact "
            "1,000,000-token abort: "
            + json.dumps(args.context_token_packing, sort_keys=True),
            flush=True,
        )
    problem = stringify(row.get("problem_statement"))
    requirements = stringify(row.get("requirements"))
    interface = stringify(row.get("interface"))
    instruction = "\n\n".join(part for part in (problem, requirements, interface) if part)
    if args.allow_gold_path_context:
        paths = gold_patch_paths(stringify(row.get("patch")))
        if paths:
            instruction += "\n\nGold target path hints enabled: " + ", ".join(paths)
    raw = dict(row)
    raw["_checkout"] = checkout_meta
    return SWEBenchProCase(
        row_index=int(row_index),
        instruction=instruction,
        files=files,
        expected_patch=stringify(row.get("patch")),
        raw=raw,
        repo=stringify(row.get("repo")),
        instance_id=stringify(row.get("instance_id")),
        base_commit=stringify(row.get("base_commit")),
        repo_language=stringify(row.get("repo_language")),
        requirements=requirements,
        interface=interface,
        fail_to_pass=as_list(row.get("fail_to_pass")),
        pass_to_pass=as_list(row.get("pass_to_pass")),
        before_repo_set_cmd=stringify(row.get("before_repo_set_cmd")),
        selected_test_files_to_run=as_list(row.get("selected_test_files_to_run")),
        dockerhub_tag=stringify(row.get("dockerhub_tag")),
        checkout_path=str(checkout_path or ""),
    )


def loco_case(case: SWEBenchProCase, *, instruction: str | None = None) -> LoCoBenchCase:
    return LoCoBenchCase(
        row_index=case.row_index,
        instruction=case.instruction if instruction is None else instruction,
        files=case.files,
        expected_patch=case.expected_patch,
        raw=case.raw,
    )


def swe_jit_case_id(case: SWEBenchProCase) -> str:
    return case.instance_id or f"row_{case.row_index}"


def swe_ruler_jit_cache_digest(
    cases: list[SWEBenchProCase],
    args: argparse.Namespace,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(args.dataset_id).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(args.split).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(args.start_index).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(args.limit).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(args.window_tokens).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(args.activation_overlap_tokens).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(bool(args.exact_context_packing)).encode("utf-8"))
    for case in cases:
        digest.update(swe_jit_case_id(case).encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(str(case.repo).encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(str(case.base_commit).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def swe_ruler_jit_benchmark_slug(
    cases: list[SWEBenchProCase],
    args: argparse.Namespace,
) -> str:
    start = max(0, int(args.start_index))
    stop = start + max(0, len(cases)) - 1
    digest = swe_ruler_jit_cache_digest(cases, args)
    if stop < start:
        return f"swebench_pro_{TOKEN_BIN.lower()}_empty_{digest}"
    return f"swebench_pro_{TOKEN_BIN.lower()}_rows_{start}_{stop}_{digest}"


def ruler_compatible_jit_indexing_inactive(reason: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "active": False,
        "signal": RULER_JIT_INDEXING_COMPAT_SIGNAL,
        "reason": reason,
        "implementation": RULER_JIT_INDEXING_IMPLEMENTATION,
        "ruler_entrypoint": RULER_JIT_INDEXING_ENTRYPOINT,
        "output_dir": str(args.jit_output_dir),
        "window_tokens": int(args.window_tokens),
        "overlap_tokens": int(args.activation_overlap_tokens),
    }


def run_ruler_compatible_jit_indexing(
    cases: list[SWEBenchProCase],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.prediction_mode != "kv_direct":
        return ruler_compatible_jit_indexing_inactive(
            "prediction_mode_is_not_kv_direct",
            args,
        )

    benchmark_slug = swe_ruler_jit_benchmark_slug(cases, args)
    case_ids = [swe_jit_case_id(case) for case in cases]
    result = jit_index_dataset_windows(
        benchmark_slug=benchmark_slug,
        raw_texts=(visible_prompt(loco_case(case)) for case in cases),
        output_dir=args.jit_output_dir,
        tokens_per_window=int(args.window_tokens),
        overlap_tokens=int(args.activation_overlap_tokens),
        model_path=args.jit_model_path or None,
        device=args.jit_device or None,
        reuse_existing=bool(args.jit_reuse_existing),
        batch_size=int(args.jit_batch_size) if int(args.jit_batch_size) > 0 else None,
    ).as_dict()
    compat = {
        "active": True,
        "signal": RULER_JIT_INDEXING_COMPAT_SIGNAL,
        "implementation": RULER_JIT_INDEXING_IMPLEMENTATION,
        "ruler_entrypoint": RULER_JIT_INDEXING_ENTRYPOINT,
        "same_shared_helper": True,
        "fast_path_mode": result.get("mode", ""),
        "expected_fast_path_mode": "real_gemma4_fast_batch_layer12",
        "layer": int(result.get("layer", 12)),
        "raw_text_source": "visible_prompt(loco_case(case))",
        "case_count": len(cases),
        "case_ids": case_ids,
        "cache_scope": "selected_swebench_rows",
        "cache_key": benchmark_slug,
        "cache_key_digest": swe_ruler_jit_cache_digest(cases, args),
        "cache_key_material": {
            "dataset_id": str(args.dataset_id),
            "split": str(args.split),
            "start_index": int(args.start_index),
            "limit": int(args.limit),
            "window_tokens": int(args.window_tokens),
            "activation_overlap_tokens": int(args.activation_overlap_tokens),
            "exact_context_packing": bool(args.exact_context_packing),
        },
        "reuse_existing": bool(args.jit_reuse_existing),
        "exact_context_packing_required": False,
        "context_token_packing": getattr(args, "context_token_packing", {}),
        "phase2_router": {
            "implementation": "run_locobench_benchmark.route_hot_windows",
            "recursive_passes": ["pass1_map", "pass2_hop"],
            "padding_excluded_from_routing": True,
        },
        "phase3_re_capture": {
            "implementation": "run_locobench_benchmark.kv_direct_prediction",
            "source_layer": 13,
            "target_layer": 14,
            "materializer": "chuk_lazarus.inference.backends._torch_residual_bounded.materialize_kv_direct",
        },
    }
    result["ruler_compat"] = compat
    result["compat_signal"] = RULER_JIT_INDEXING_COMPAT_SIGNAL
    result["benchmark_slug"] = benchmark_slug
    print(
        f"{RULER_JIT_INDEXING_COMPAT_SIGNAL}: "
        f"implementation={RULER_JIT_INDEXING_IMPLEMENTATION} "
        f"mode={result.get('mode', '')} layer={result.get('layer', '')} "
        f"cases={len(cases)} cache_key={benchmark_slug}",
        flush=True,
    )
    return result


def ruler_jit_case_metadata(
    args: argparse.Namespace,
    case: SWEBenchProCase,
) -> dict[str, Any]:
    jit = getattr(args, "jit_indexing", {}) or {}
    compat = dict(jit.get("ruler_compat", {}))
    case_id = swe_jit_case_id(case)
    case_ids = list(compat.get("case_ids", []))
    try:
        case_index = case_ids.index(case_id)
    except ValueError:
        case_index = None
    return {
        "active": bool(compat.get("active", False)),
        "signal": RULER_JIT_INDEXING_COMPAT_SIGNAL,
        "implementation": compat.get("implementation", RULER_JIT_INDEXING_IMPLEMENTATION),
        "ruler_entrypoint": compat.get("ruler_entrypoint", RULER_JIT_INDEXING_ENTRYPOINT),
        "mode": jit.get("mode", ""),
        "layer": jit.get("layer"),
        "benchmark_slug": jit.get("benchmark_slug", compat.get("cache_key", "")),
        "metadata_path": jit.get("metadata_path", ""),
        "activation_route_dir": jit.get("activation_route_dir", ""),
        "case_id": case_id,
        "case_index_in_jit_corpus": case_index,
        "cache_scope": compat.get("cache_scope", ""),
    }


def triad_role_for_path(path: str) -> str:
    tokens = path_name_tokens(path) | normalized_path_parts(path)
    if tokens & TRIAD_ADAPTER_TOKENS:
        return "adapter"
    if tokens & TRIAD_ENTRYPOINT_TOKENS:
        return "entrypoint"
    if tokens & TRIAD_LIFECYCLE_TOKENS:
        return "lifecycle"
    return ""


def triad_seed_terms(path: str, text: str) -> set[str]:
    terms = dependency_name_terms(path, text)
    identifiers = [identifier.lower() for identifier in extract_identifiers(text)]
    terms.update(identifiers[:80])
    return {
        term
        for term in terms
        if len(term) >= 3 and term not in TRIAD_STOP_TERMS and not term.isdigit()
    }


def triad_reference_score(
    candidate_path: str,
    candidate_text: str,
    *,
    seed_path: str,
    seed_text: str,
    seed_terms: set[str],
) -> float:
    candidate_terms = dependency_name_terms(candidate_path)
    candidate_path_overlap = seed_terms & candidate_terms
    score = float(len(candidate_path_overlap) * 7.0)
    candidate_body_terms = dependency_name_terms(candidate_text[:40_000])
    score += float(len(seed_terms & candidate_body_terms) * 1.5)
    seed_stem = Path(seed_path.replace("\\", "/")).stem.lower()
    candidate_stem = Path(candidate_path.replace("\\", "/")).stem.lower()
    if seed_stem and re.search(rf"\b{re.escape(seed_stem)}\b", candidate_text, re.IGNORECASE):
        score += 12.0
    if candidate_stem and re.search(rf"\b{re.escape(candidate_stem)}\b", seed_text, re.IGNORECASE):
        score += 8.0
    role = triad_role_for_path(candidate_path)
    if role:
        score += 18.0
    if is_implementation_source_path(candidate_path):
        score += 10.0
    if is_test_like_path(candidate_path) or is_fixture_like_path(candidate_path):
        score -= 30.0
    if is_doc_like_path(candidate_path) or is_asset_like_path(candidate_path):
        score -= 40.0
    return route_domain_score(score, candidate_path)


def discover_triad_paths(
    case: SWEBenchProCase,
    *,
    seed_paths: list[str],
    selected_paths: list[str],
    limit: int,
) -> tuple[list[str], dict[str, Any]]:
    if limit <= 0:
        return [], {
            "active": True,
            "seed_paths": seed_paths,
            "discovered_paths": [],
            "discovered_by_role": {"adapter": [], "entrypoint": [], "lifecycle": []},
            "seed_by_path": {},
            "score_by_path": {},
            "candidate_count": 0,
            "limit": int(limit),
            "generic_tokens": sorted(TRIAD_ROLE_TOKENS),
        }
    seen = set(selected_paths)
    candidates: list[tuple[float, str, str, str]] = []
    route_files = {
        path: text
        for path, text in case.files.items()
        if is_implementation_source_path(path) and not is_benchmark_padding_path(path)
    }
    for seed_path in seed_paths:
        seed_text = case.files.get(seed_path, "")
        if not seed_text or not is_implementation_source_path(seed_path):
            continue
        seed_terms = triad_seed_terms(seed_path, seed_text)
        if not seed_terms:
            continue
        for candidate_path, candidate_text in route_files.items():
            if candidate_path == seed_path or candidate_path in seen:
                continue
            role = triad_role_for_path(candidate_path)
            if not role:
                continue
            score = triad_reference_score(
                candidate_path,
                candidate_text,
                seed_path=seed_path,
                seed_text=seed_text,
                seed_terms=seed_terms,
            )
            if score <= 0.0:
                continue
            candidates.append((score, candidate_path, role, seed_path))
    candidates.sort(
        key=lambda item: (
            -item[0],
            -route_domain_weight(item[1]),
            route_path_depth_penalty(item[1]),
            item[1],
        )
    )
    discovered: list[str] = []
    by_role: dict[str, list[str]] = {"adapter": [], "entrypoint": [], "lifecycle": []}
    source_by_path: dict[str, str] = {}
    score_by_path: dict[str, float] = {}
    for score, path, role, seed_path in candidates:
        if path in seen:
            continue
        seen.add(path)
        discovered.append(path)
        by_role.setdefault(role, []).append(path)
        source_by_path[path] = seed_path
        score_by_path[path] = float(score)
        if len(discovered) >= max(0, int(limit)):
            break
    return discovered, {
        "active": True,
        "seed_paths": seed_paths,
        "discovered_paths": discovered,
        "discovered_by_role": by_role,
        "seed_by_path": source_by_path,
        "score_by_path": score_by_path,
        "candidate_count": len(candidates),
        "limit": int(limit),
        "generic_tokens": sorted(TRIAD_ROLE_TOKENS),
    }


def discover_source_hint_paths(
    case: SWEBenchProCase,
    instruction: str,
    *,
    selected_paths: list[str],
    limit: int,
) -> tuple[list[str], dict[str, Any]]:
    routing_terms = term_variants(dependency_name_terms(instruction))
    seen = set(selected_paths)
    scored: list[tuple[float, str, list[str]]] = []
    for path in case.files:
        if path in seen or not is_implementation_source_path(path):
            continue
        path_terms = term_variants(dependency_name_terms(path))
        overlap = sorted(path_terms & routing_terms)
        if not overlap:
            continue
        role_tokens = sorted((path_terms | normalized_path_parts(path)) & TRIAD_ROLE_TOKENS)
        score = float(len(overlap) * 45.0 + len(role_tokens) * 18.0)
        stem = Path(path.replace("\\", "/")).stem.lower()
        parts_and_terms = path_terms | normalized_path_parts(path)
        if stem in {"api", "helper", "helpers", "util", "utils"}:
            score -= 60.0
        if stem in {"main", "index"} and parts_and_terms & TRIAD_ADAPTER_TOKENS:
            score += 80.0
        if parts_and_terms & {"controller", "controllers", "handler", "handlers", "socket", "sockets"}:
            score += 45.0
        if parts_and_terms & TRIAD_LIFECYCLE_TOKENS:
            score += 35.0
        if "database" in overlap and parts_and_terms & TRIAD_ADAPTER_TOKENS:
            score += 40.0
        if Path(path.replace("\\", "/")).stem.lower() in routing_terms:
            score += 18.0
        score = route_domain_score(score, path)
        if score <= 0.0:
            continue
        scored.append((score, path, overlap))
    scored.sort(
        key=lambda item: (
            -item[0],
            -route_domain_weight(item[1]),
            route_path_depth_penalty(item[1]),
            item[1],
        )
    )
    selected_scored: list[tuple[float, str, list[str]]] = []
    selected_seen: set[str] = set()

    def pick_bucket(token_filter: set[str], count: int) -> None:
        for item in scored:
            if len(selected_scored) >= int(limit):
                return
            _score, path, _overlap = item
            if path in selected_seen:
                continue
            path_terms = term_variants(dependency_name_terms(path)) | normalized_path_parts(path)
            if not path_terms & token_filter:
                continue
            selected_scored.append(item)
            selected_seen.add(path)
            count -= 1
            if count <= 0:
                return

    def pick_overlap_rich(count: int) -> None:
        for item in scored:
            if len(selected_scored) >= int(limit):
                return
            _score, path, overlap = item
            if path in selected_seen or len(overlap) < 2:
                continue
            path_terms = term_variants(dependency_name_terms(path)) | normalized_path_parts(path)
            if path_terms & TRIAD_ROLE_TOKENS:
                continue
            selected_scored.append(item)
            selected_seen.add(path)
            count -= 1
            if count <= 0:
                return

    pick_bucket(TRIAD_ADAPTER_TOKENS, 4)
    pick_bucket({"socket", "sockets", "rpc", "grpc"}, 2)
    pick_overlap_rich(2)
    pick_bucket({"controller", "controllers", "handler", "handlers", "route", "routes"}, 3)
    pick_bucket(TRIAD_LIFECYCLE_TOKENS, 2)
    for item in scored:
        if len(selected_scored) >= int(limit):
            break
        _score, path, _overlap = item
        if path in selected_seen:
            continue
        selected_scored.append(item)
        selected_seen.add(path)

    selected_scored = selected_scored[: max(0, int(limit))]
    discovered = [path for _score, path, _overlap in selected_scored]
    return discovered, {
        "active": True,
        "routing_term_count": len(routing_terms),
        "candidate_count": len(scored),
        "discovered_paths": discovered,
        "score_by_path": {path: float(score) for score, path, _overlap in selected_scored},
        "overlap_by_path": {path: overlap for _score, path, overlap in selected_scored},
        "limit": int(limit),
    }


def multi_target_route_hot_windows(
    case: SWEBenchProCase,
    args: argparse.Namespace,
    *,
    instruction: str,
    max_k: int,
) -> tuple[list[DependencyWindow], dict[str, Any]]:
    windows_per_target = max(1, int(getattr(args, "windows_per_target", DEFAULT_WINDOWS_PER_TARGET)))
    phase2_file_budget = max(1, int(args.target_file_budget))
    uncapped_budget = max(int(max_k), 8)
    budget = max(1, min(MAX_PHASE2_HOT_WINDOWS, uncapped_budget))
    route_files = {
        path: content
        for path, content in case.files.items()
        if not is_benchmark_padding_path(path)
    }
    routed = route_hot_windows(
        instruction,
        route_files,
        max_k=budget,
        tokens_per_window=int(args.window_tokens),
        overlap_tokens=int(args.activation_overlap_tokens),
        activation_route_dir=args.jit_output_dir / "swebench_pro_source_activation_routes",
        case_id=case.instance_id or case.row_index,
    )
    by_path: dict[str, list[DependencyWindow]] = {}
    for window in routed:
        by_path.setdefault(window.path, []).append(window)
    selected_paths, target_priority_meta = prioritize_patch_target_paths(
        list(by_path),
        case.files,
        phase2_file_budget,
    )
    source_hint_room = max(0, budget - len(selected_paths))
    source_hint_paths, source_hint_meta = discover_source_hint_paths(
        case,
        instruction,
        selected_paths=selected_paths,
        limit=source_hint_room,
    )
    selected_paths = list(dict.fromkeys([*source_hint_paths, *selected_paths]))[:budget]
    seed_paths = list(
        dict.fromkeys(
            [
                *selected_paths,
                *[
                    window.path
                    for window in routed
                    if is_implementation_source_path(window.path)
                ][:phase2_file_budget],
            ]
        )
    )
    triad_room = max(0, budget - len(selected_paths))
    triad_paths, triad_meta = discover_triad_paths(
        case,
        seed_paths=seed_paths,
        selected_paths=selected_paths,
        limit=triad_room,
    )
    target_paths = list(dict.fromkeys([*selected_paths, *triad_paths]))
    windows_per_selected_path = max(
        1,
        min(windows_per_target, budget // max(1, len(target_paths))),
    )
    selected: list[DependencyWindow] = []
    for path in target_paths:
        routed_for_path = by_path.get(path, [])
        selected.extend(routed_for_path[:windows_per_selected_path])
        if len(routed_for_path) < windows_per_selected_path:
            selected.extend(
                supplemental_source_windows_for_path(
                    case,
                    args,
                    path,
                    existing_windows=routed_for_path,
                    needed=windows_per_selected_path - len(routed_for_path),
                    window_id_seed=-200_000 - (len(selected) * 10),
                )
            )
    for window in routed:
        if len(selected) >= budget:
            break
        if window not in selected:
            selected.append(window)
    selected = selected[:budget]
    metadata = {
        "active": True,
        "target_file_budget": int(args.target_file_budget),
        "windows_per_target": int(windows_per_target),
        "windows_per_selected_path": int(windows_per_selected_path),
        "dynamic_budget": int(budget),
        "uncapped_dynamic_budget": int(uncapped_budget),
        "max_phase2_hot_windows": int(MAX_PHASE2_HOT_WINDOWS),
        "window_cap_enforced": bool(budget <= MAX_PHASE2_HOT_WINDOWS),
        "allocator_max_k": int(max_k),
        "selected_paths": selected_paths,
        "source_hint_paths": source_hint_paths,
        "triad_augmented_paths": triad_paths,
        "routed_target_paths": target_paths,
        "selected_path_count": len(selected_paths),
        "routed_target_path_count": len(target_paths),
        "window_count": len(selected),
        "windows_by_selected_path": {
            path: len([window for window in selected if window.path == path])
            for path in target_paths
        },
        "fallback_window_count": len(
            [
                window
                for window in selected
                if window.route_source.startswith("multi_target_source_fallback")
            ]
        ),
        "route_sources_by_path": {
            path: [window.route_source for window in windows]
            for path, windows in by_path.items()
            if path in selected_paths
        },
        "patch_target_priority": target_priority_meta,
        "source_hint_discovery": source_hint_meta,
        "triad_discovery": triad_meta,
        "routing_file_count": len(route_files),
        "padding_excluded_from_routing": any(
            is_benchmark_padding_path(path) for path in case.files
        ),
        "gold_path_context_allowed": bool(args.allow_gold_path_context),
        "known_path_mentions": sorted(path for path in case.files if path in instruction),
    }
    return selected, metadata


def supplemental_source_windows_for_path(
    case: SWEBenchProCase,
    args: argparse.Namespace,
    path: str,
    *,
    existing_windows: list[DependencyWindow],
    needed: int,
    window_id_seed: int,
) -> list[DependencyWindow]:
    content = case.files.get(path, "")
    if needed <= 0 or not content.strip():
        return []
    existing_spans = {
        (int(window.start_line), int(window.end_line), window.text)
        for window in existing_windows
    }
    max_chars = max(1_000, min(16_000, int(getattr(args, "window_tokens", 512)) * 14))
    lines = content.splitlines(keepends=True)
    windows: list[DependencyWindow] = []
    cursor = 0
    line_no = 1
    while cursor < len(lines) and len(windows) < needed:
        chunk_lines: list[str] = []
        chunk_chars = 0
        start_line = line_no
        while cursor < len(lines) and (chunk_chars < max_chars or not chunk_lines):
            line = lines[cursor]
            chunk_lines.append(line)
            chunk_chars += len(line)
            cursor += 1
            line_no += 1
            if chunk_chars >= max_chars:
                break
        text = "".join(chunk_lines).rstrip()
        if not text:
            continue
        end_line = max(start_line, line_no - 1)
        span_key = (start_line, end_line, text)
        if span_key in existing_spans:
            continue
        window_index = len(windows)
        windows.append(
            DependencyWindow(
                window_id=window_id_seed - window_index,
                path=path,
                text=text,
                score=0.0,
                start_line=start_line,
                end_line=end_line,
                start_token=0,
                end_token=max(1, len(text.split())),
                activation_route_id=None,
                lexical_score=0.0,
                activation_score=0.0,
                exact_scope_match=False,
                route_source="multi_target_source_fallback_window_floor",
                recursive_map_window_id=None,
                recursive_pass="multi_target_window_floor",
            )
        )
    return windows


def parse_strict_blocks(generated: str, files: dict[str, str]) -> list[StrictBlock]:
    blocks: list[StrictBlock] = []
    for match in SEARCH_REPLACE_RE.finditer(generated or ""):
        prefix = (generated or "")[: match.start()]
        path = path_before_block(prefix, files)
        blocks.append(
            StrictBlock(
                path=path,
                old=match.group("old").strip("\n"),
                new=match.group("new").strip("\n"),
            )
        )
    return blocks


def replacement_protocol_marker_findings(patch_text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    text = patch_text or ""
    for block_index, match in enumerate(SEARCH_REPLACE_RE.finditer(text), start=1):
        replacement_body = match.group("new")
        for marker in PROTOCOL_REPLACEMENT_MARKER_RE.finditer(replacement_body):
            findings.append(
                {
                    "format": "strict_search_replace",
                    "block_index": block_index,
                    "marker": marker.group(0),
                    "replacement_offset": marker.start(),
                }
            )
    if findings:
        return findings
    if looks_like_unified_diff(text):
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.startswith("+++") or not line.startswith("+"):
                continue
            body_line = line[1:]
            marker = PROTOCOL_REPLACEMENT_MARKER_RE.search(body_line)
            if marker is not None:
                findings.append(
                    {
                        "format": "unified_diff",
                        "line_number": line_number,
                        "marker": marker.group(0),
                        "line_excerpt": body_line[:200],
                    }
                )
    return findings


def patch_integrity_filter(patch_text: str) -> dict[str, Any]:
    findings = replacement_protocol_marker_findings(patch_text)
    return {
        "passed": not findings,
        "reason": (
            "clean"
            if not findings
            else "patch_contains_nested_protocol_markers"
        ),
        "marker_count": len(findings),
        "findings": findings[:20],
    }


def path_before_block(prefix: str, files: dict[str, str]) -> str:
    basenames = {Path(path).name: path for path in files}
    for raw_line in reversed((prefix or "").splitlines()):
        line = raw_line.strip()
        if not line or line.startswith(("<<<<", "====", ">>>>")):
            continue
        candidate = line.replace("\\", "/").strip("`'\"")
        if candidate in files:
            return candidate
        if Path(candidate).name in basenames:
            return basenames[Path(candidate).name]
        if candidate:
            return candidate
    return ""


def multi_file_verbatim_readout_bridge(
    case: SWEBenchProCase,
    hot_windows: list[DependencyWindow],
    generated: str,
) -> tuple[str, dict[str, Any]]:
    if not re.search(r"<<<<\s*SEARCH", generated or "", flags=re.IGNORECASE):
        return generated, {"active": False, "reason": "search_marker_absent", "file_path_sequence": []}
    blocks = parse_strict_blocks(generated, case.files)
    if not blocks:
        return generated, {"active": False, "reason": "no_strict_blocks", "file_path_sequence": []}
    query_terms = readout_terms(case.instruction, generated)
    rebuilt: list[str] = []
    file_path_sequence: list[str] = []
    bridged_count = 0
    block_meta: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks, start=1):
        path, span, span_meta = best_verbatim_span_for_block(
            case,
            hot_windows,
            block,
            query_terms,
        )
        chosen_old = span or block.old
        chosen_path = path or block.path
        if span:
            bridged_count += 1
        file_path_sequence.append(chosen_path)
        rebuilt.append(f"{chosen_path}\n<<<< SEARCH\n{chosen_old}\n==== REPLACE\n{block.new}\n>>>>")
        block_meta.append({"block_index": block_index, "path": chosen_path, **span_meta})
    bridged = "\n".join(rebuilt)
    authority_identifiers = set(identifiers_from_windows(hot_windows))
    for block in blocks:
        authority_identifiers.update(extract_identifiers(block.old))
    return bridged, {
        "active": bridged_count > 0,
        "reason": "multi_file_search_sections_bridged_to_verbatim_hot_windows",
        "block_count": len(blocks),
        "bridged_block_count": bridged_count,
        "file_path_sequence": file_path_sequence,
        "blocks": block_meta,
        "identifier_authority": {
            "paths": file_path_sequence,
            "identifiers": sorted(authority_identifiers),
            "identifier_count": len(authority_identifiers),
        },
    }


def best_verbatim_span_for_block(
    case: SWEBenchProCase,
    hot_windows: list[DependencyWindow],
    block: StrictBlock,
    query_terms: set[str],
) -> tuple[str, str, dict[str, Any]]:
    candidates = [window for window in hot_windows if not block.path or window.path == block.path]
    if not candidates and block.path:
        block_base = Path(block.path).name
        candidates = [window for window in hot_windows if Path(window.path).name == block_base]
    if not candidates:
        candidates = list(hot_windows)
    best: tuple[float, DependencyWindow | None, str] = (float("-inf"), None, "")
    old_terms = readout_terms(block.old)
    for window in candidates:
        content = case.files.get(window.path, "")
        if not content or window.text not in content:
            continue
        span_terms = readout_terms(window.text)
        score = float(window.score) + len(span_terms & query_terms) * 8.0
        score += len(span_terms & old_terms) * 12.0
        if block.old and block.old in content and block.old in window.text:
            score += 1000.0
        if block.path and window.path == block.path:
            score += 500.0
        if score > best[0]:
            best = (score, window, window.text)
    window = best[1]
    if window is None:
        return block.path, "", {"bridged": False, "reason": "no_matching_hot_window_span"}
    return window.path, best[2], {
        "bridged": True,
        "window_id": int(window.window_id),
        "start_line": int(window.start_line),
        "end_line": int(window.end_line),
        "score": float(best[0]),
        "route_source": window.route_source,
    }


def infer_dynamic_dialect_plan(
    case: SWEBenchProCase,
    hot_windows: list[DependencyWindow],
    paths: list[str],
    *,
    prediction_mode: str,
) -> dict[str, Any]:
    unique_paths = []
    for path in [*paths, *(window.path for window in hot_windows)]:
        if path and path in case.files and path not in unique_paths:
            unique_paths.append(path)
    specs: dict[str, dict[str, Any]] = {}
    for path in unique_paths:
        nearby = [window.text for window in hot_windows if window.path == path]
        spec = infer_dialect_lock_for_source(path, case.files.get(path, ""), nearby_texts=nearby)
        telemetry = dialect_lock_telemetry(spec)
        specs[path] = {
            **telemetry,
            "forbidden_literal_count": len(telemetry.get("forbidden_literals", [])),
        }
    hard_switch_available, hard_switch_reason = kv_direct_sampler_hard_switch_available()
    needs_hard_switch = prediction_mode == "kv_direct" and len(unique_paths) > 1
    segmented_restart = bool(needs_hard_switch and not hard_switch_available and hot_windows)
    failed_closed = bool(needs_hard_switch and not hard_switch_available and not segmented_restart)
    sampler_state = (
        "active"
        if hard_switch_available
        else ("segmented_restart" if segmented_restart else ("failed_closed" if failed_closed else "unavailable"))
    )
    return {
        "active": bool(specs),
        "mode": (
            "path_aware_hard_sampler_switch"
            if hard_switch_available
            else (
                "segmented_single_path_restarts"
                if segmented_restart
                else "path_indexed_posthoc_validation"
            )
        ),
        "sampler_hard_switch": bool(hard_switch_available),
        "sampler_hard_switch_state": sampler_state,
        "sampler_hard_switch_reason": hard_switch_reason,
        "requires_hard_switch": bool(needs_hard_switch),
        "segmented_restart": segmented_restart,
        "failed_closed": failed_closed,
        "failure_reason": (
            "kv_direct_segmented_restart_requires_hot_windows"
            if failed_closed
            else ""
        ),
        "note": (
            "KV-direct generation restarts once per selected target path so each "
            "LoCo call installs a fresh single-file dialect mask; this is not a "
            "mid-stream hard sampler switch."
            if segmented_restart
            else (
                "KV-direct generation cannot safely use one LoCo dialect mask across "
                "multiple selected target paths unless true path-aware sampler switching "
                "or segmented restarts are available."
                if failed_closed
                else (
                    "KV-direct generation installs a path-aware decoder plan; generated "
                    "file-path lines refresh forbidden dialect tokens before each sample."
                )
            )
        ),
        "selected_path_count": len(unique_paths),
        "selected_paths": unique_paths,
        "paths": specs,
        "forbidden_token_counts": {
            path: int(spec.get("forbidden_literal_count", 0))
            for path, spec in specs.items()
        },
        "segmentation": {
            "active": segmented_restart,
            "restart_unit": "selected_target_path",
            "fresh_mask_per_segment": segmented_restart,
            "per_segment_budget_policy": (
                "effective_kv_max_new_tokens_passed_to_each_restart"
                if segmented_restart
                else ""
            ),
            "path_count": len(unique_paths) if segmented_restart else 0,
            "paths": unique_paths if segmented_restart else [],
        },
    }


def kv_direct_sampler_hard_switch_available() -> tuple[bool, str]:
    """Return whether kv_direct accepts a true path-aware dialect switch plan."""
    try:
        signature = inspect.signature(kv_direct_prediction)
    except (TypeError, ValueError):
        return False, "kv_direct_prediction signature is not introspectable"
    supported_params = {"dialect_plan", "dynamic_dialect_plan", "sampler_hard_switch"}
    if supported_params & set(signature.parameters):
        return True, "kv_direct_prediction exposes a path-aware dialect switch parameter"
    return (
        False,
        "kv_direct_prediction currently installs one LoCo sampler dialect lock for one target path",
    )


def dynamic_token_budget_metadata(
    args: argparse.Namespace,
    *,
    selected_paths: list[str],
    strict_expected_paths: list[str] | None = None,
) -> dict[str, Any]:
    expected_paths = list(dict.fromkeys(path for path in (strict_expected_paths or []) if path))
    phase2_paths = list(dict.fromkeys(path for path in selected_paths if path))
    budget_paths = expected_paths or phase2_paths
    n_files = max(1, len(budget_paths))
    spillover = max(
        0,
        int(
            getattr(
                args,
                "kv_replacement_spillover_tokens",
                DEFAULT_KV_REPLACEMENT_SPILLOVER_TOKENS,
            )
        ),
    )
    dynamic_payload_budget = n_files * 1024
    dynamic_total_budget = dynamic_payload_budget + spillover
    current_diff_reserved = int(getattr(args, "diff_reserved_tokens", DEFAULT_DIFF_RESERVED_TOKENS))
    current_kv_max_new = int(getattr(args, "kv_max_new_tokens", DEFAULT_KV_MAX_NEW_TOKENS))
    return {
        "active": True,
        "formula": "payload=(N_files * 1024); total=payload+spillover",
        "n_files": int(n_files),
        "phase2_selected_paths": phase2_paths,
        "strict_expected_paths": expected_paths,
        "budget_paths": budget_paths,
        "spillover_tokens": int(spillover),
        "dynamic_payload_budget_tokens": int(dynamic_payload_budget),
        "dynamic_total_budget_tokens": int(dynamic_total_budget),
        "dynamic_budget_tokens": int(dynamic_total_budget),
        "requested_diff_reserved_tokens": current_diff_reserved,
        "effective_diff_reserved_tokens": max(current_diff_reserved, dynamic_total_budget),
        "diff_reserved_tokens_updated": dynamic_total_budget > current_diff_reserved,
        "requested_kv_max_new_tokens": current_kv_max_new,
        "effective_kv_max_new_tokens": max(current_kv_max_new, dynamic_payload_budget),
        "kv_max_new_tokens_updated": dynamic_payload_budget > current_kv_max_new,
        "decode_payload_token_cap": int(
            max(
                0,
                int(
                    getattr(
                        args,
                        "kv_decode_payload_token_cap",
                        DEFAULT_KV_DECODE_PAYLOAD_TOKEN_CAP,
                    )
                ),
            )
        ),
        "per_segment_payload_cap_applies_in_kv_direct": True,
    }


def scoped_generation_args(
    args: argparse.Namespace,
    token_budget: dict[str, Any],
) -> argparse.Namespace:
    scoped = argparse.Namespace(**vars(args))
    scoped.diff_reserved_tokens = int(token_budget["effective_diff_reserved_tokens"])
    scoped.kv_max_new_tokens = int(token_budget["effective_kv_max_new_tokens"])
    scoped.dynamic_token_budget = token_budget
    return scoped


def apply_patch_to_workspace(
    case: SWEBenchProCase,
    generated: str,
) -> tuple[bool, dict[str, Any]]:
    checkout = Path(case.checkout_path) if case.checkout_path else None
    if checkout is None or not checkout.exists():
        return False, {"available": False, "reason": "checkout_unavailable"}
    reset_workspace(checkout)
    strict = parse_strict_blocks(generated, case.files)
    if strict:
        failures: list[str] = []
        changed: list[str] = []
        for index, block in enumerate(strict, start=1):
            path = resolve_generated_path(block.path, case.files)
            if not path:
                failures.append(f"block {index}: path unavailable")
                continue
            abs_path = checkout / path
            try:
                content = abs_path.read_text(encoding="utf-8")
            except OSError as exc:
                failures.append(f"block {index}: cannot read {path}: {exc}")
                continue
            if block.old not in content:
                failures.append(f"block {index}: SEARCH text not found in {path}")
                continue
            abs_path.write_text(content.replace(block.old, block.new, 1), encoding="utf-8")
            changed.append(path)
        return not failures and bool(changed), {
            "mode": "strict_search_replace",
            "changed_files": changed,
            "failures": failures,
        }
    if looks_like_unified_diff(generated):
        completed = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=str(checkout),
            input=generated,
            text=True,
            capture_output=True,
            check=False,
        )
        changed = run_git(["diff", "--name-only"], cwd=checkout)
        return completed.returncode == 0, {
            "mode": "git_apply",
            "returncode": completed.returncode,
            "stderr_tail": tail(completed.stderr),
            "changed_files": [
                line.strip() for line in changed.stdout.splitlines() if line.strip()
            ],
        }
    return False, {"mode": "none", "failures": ["no strict blocks or unified diff"]}


def reset_workspace(checkout: Path) -> None:
    run_git(["reset", "--hard"], cwd=checkout, timeout=300)
    run_git(["clean", "-fdx"], cwd=checkout, timeout=300)


def resolve_generated_path(path: str, files: dict[str, str]) -> str:
    normalized = (path or "").replace("\\", "/").strip()
    if normalized in files:
        return normalized
    base = Path(normalized).name
    for candidate in files:
        if Path(candidate).name == base:
            return candidate
    return normalized if normalized else ""


def looks_like_unified_diff(text: str) -> bool:
    return bool(re.search(r"(?m)^(diff --git|---\s+a/|\+\+\+\s+b/|@@\s)", text or ""))


def run_post_patch_test_gate(
    case: SWEBenchProCase,
    args: argparse.Namespace,
    *,
    patch_applied: bool,
) -> TestGateResult:
    if not patch_applied:
        return TestGateResult(False, False, bool(args.require_test_gate), "patch_not_applied", "", None, "", "", 0.0, [])
    checkout = Path(case.checkout_path) if case.checkout_path else None
    if checkout is None or not checkout.exists():
        return TestGateResult(
            passed=not args.require_test_gate,
            skipped=True,
            required=bool(args.require_test_gate),
            reason="checkout_unavailable",
            command="",
            returncode=None,
            stdout_tail="",
            stderr_tail="",
            elapsed_ms=0.0,
            selected_tests=[],
        )
    selected = case.selected_test_files_to_run or case.fail_to_pass or case.pass_to_pass
    command, reason = infer_test_command(case, args, checkout, selected)
    if not command:
        return TestGateResult(
            passed=not args.require_test_gate,
            skipped=True,
            required=bool(args.require_test_gate),
            reason=reason,
            command="",
            returncode=None,
            stdout_tail="",
            stderr_tail="",
            elapsed_ms=0.0,
            selected_tests=selected,
        )
    if case.before_repo_set_cmd:
        setup = subprocess.run(
            case.before_repo_set_cmd,
            cwd=str(checkout),
            shell=True,
            text=True,
            capture_output=True,
            check=False,
            timeout=600,
        )
        if setup.returncode != 0:
            return TestGateResult(
                passed=False,
                skipped=False,
                required=bool(args.require_test_gate),
                reason="before_repo_set_cmd_failed",
                command=case.before_repo_set_cmd,
                returncode=setup.returncode,
                stdout_tail=tail(setup.stdout),
                stderr_tail=tail(setup.stderr),
                elapsed_ms=0.0,
                selected_tests=selected,
            )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(checkout),
        shell=True,
        text=True,
        capture_output=True,
        check=False,
        timeout=1800,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return TestGateResult(
        passed=completed.returncode == 0,
        skipped=False,
        required=bool(args.require_test_gate),
        reason="passed" if completed.returncode == 0 else "failed",
        command=command,
        returncode=completed.returncode,
        stdout_tail=tail(completed.stdout),
        stderr_tail=tail(completed.stderr),
        elapsed_ms=elapsed_ms,
        selected_tests=selected,
    )


def infer_test_command(
    case: SWEBenchProCase,
    args: argparse.Namespace,
    checkout: Path,
    selected: list[str],
) -> tuple[str, str]:
    context = {
        "selected_tests": " ".join(shlex.quote(test) for test in selected),
        "instance_id": shlex.quote(case.instance_id),
        "repo_language": shlex.quote(case.repo_language),
    }
    template = args.test_command or args.test_template
    if template:
        return template.format(**context), ""
    language = case.repo_language.lower()
    if language == "python" or any((checkout / name).exists() for name in ("pyproject.toml", "setup.py", "pytest.ini")):
        return "python -m pytest -q " + context["selected_tests"], ""
    if language == "go" or (checkout / "go.mod").exists():
        return "go test ./...", ""
    if language in {"javascript", "typescript"} or (checkout / "package.json").exists():
        return ("npm test", "") if (checkout / "package.json").exists() else ("", "package_json_absent")
    return "", f"no_inferred_test_command_for_language:{case.repo_language or 'unknown'}"


def eval_results_path(output_dir: Path) -> Path | None:
    direct = output_dir / "eval_results.json"
    if direct.exists():
        return direct
    matches = sorted(output_dir.rglob("eval_results.json")) if output_dir.exists() else []
    return matches[0] if matches else None


def coerce_official_passed(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized == "RESOLVED":
            return True
        if normalized in {"UNRESOLVED", "FAILED", "ERROR", "TIMEOUT"}:
            return False
    if isinstance(value, dict):
        for key in ("resolved_status", "status", "result"):
            if isinstance(value.get(key), str):
                coerced = coerce_official_passed(value[key])
                if coerced is not None:
                    return coerced
        for key in ("passed", "resolved", "success"):
            if isinstance(value.get(key), bool):
                return bool(value[key])
    return None


def parse_official_eval_result(output_dir: Path, instance_id: str) -> dict[str, Any]:
    result_path = eval_results_path(output_dir)
    if result_path is None:
        return {
            "passed": False,
            "reason": "eval_results_json_missing",
            "eval_results_path": "",
        }
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "reason": "eval_results_json_invalid",
            "eval_results_path": str(result_path),
            "error": str(exc),
        }
    value: Any = None
    if isinstance(payload, dict):
        if instance_id in payload:
            value = payload[instance_id]
        elif isinstance(payload.get("eval_results"), dict):
            value = payload["eval_results"].get(instance_id)
        elif isinstance(payload.get("results"), dict):
            value = payload["results"].get(instance_id)
        else:
            value = payload
    passed = coerce_official_passed(value)
    if passed is None:
        return {
            "passed": False,
            "reason": "instance_result_missing",
            "eval_results_path": str(result_path),
            "instance_id": instance_id,
        }
    return {
        "passed": bool(passed),
        "reason": "official_eval_passed" if passed else "official_eval_failed",
        "eval_results_path": str(result_path),
        "instance_id": instance_id,
    }


def official_eval_unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "attempted": False,
        "passed": False,
        "official_eval_required": True,
        "reason": reason,
        **extra,
    }


def official_eval_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    if bool(args.use_local_docker) and int(args.official_eval_timeout_seconds) <= 0:
        env["SWEAP_LOCAL_DOCKER_CLIENT_TIMEOUT_SECONDS"] = "0"
    return env


def clean_official_instance_id(instance_id: str) -> str:
    cleaned = str(instance_id or "").strip()
    for suffix in ("-vnan",):
        if cleaned.endswith(suffix):
            return cleaned[: -len(suffix)]
    return cleaned


def official_docker_tag(case: SWEBenchProCase) -> str:
    if case.dockerhub_tag:
        tag = str(case.dockerhub_tag).strip()
        image_tag = tag.rsplit(":", 1)[-1] if "/" in tag and ":" in tag else tag
        return clean_official_instance_id(image_tag)
    instance_id = clean_official_instance_id(case.instance_id)
    if not instance_id:
        return ""
    if case.repo and "/" in case.repo:
        repo_base, repo_name = case.repo.lower().split("/", 1)
        return f"{repo_base}.{repo_name}-{instance_id.replace('instance_', '', 1)}"
    return instance_id


def official_docker_image(case: SWEBenchProCase, args: argparse.Namespace) -> str:
    tag = official_docker_tag(case)
    if not tag:
        return ""
    if "/" in tag and ":" in tag:
        return tag
    return f"{args.official_docker_image_repo}:{tag}"


def write_official_eval_inputs(
    case: SWEBenchProCase,
    generated: str,
    eval_root: Path,
) -> dict[str, Path]:
    raw_sample_path = eval_root / "raw_sample.csv"
    predictions_path = eval_root / "predictions.jsonl"
    patch_json_path = eval_root / "patches.json"
    with raw_sample_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "repo",
            "instance_id",
            "base_commit",
            "before_repo_set_cmd",
            "selected_test_files_to_run",
            "fail_to_pass",
            "pass_to_pass",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "repo": case.repo,
                "instance_id": case.instance_id,
                "base_commit": case.base_commit,
                "before_repo_set_cmd": case.before_repo_set_cmd,
                "selected_test_files_to_run": repr(case.selected_test_files_to_run),
                "fail_to_pass": repr(case.fail_to_pass),
                "pass_to_pass": repr(case.pass_to_pass),
            }
        )
    prediction = {"instance_id": case.instance_id, "patch": generated, "prefix": "lazarus"}
    predictions_path.write_text(
        json.dumps(prediction, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    patch_json_path.write_text(json.dumps([prediction], indent=2) + "\n", encoding="utf-8")
    return {
        "raw_sample_path": raw_sample_path,
        "predictions_path": predictions_path,
        "patch_json_path": patch_json_path,
    }


def official_docker_shell_command(
    case: SWEBenchProCase,
    args: argparse.Namespace,
    *,
    script_mount: str,
) -> str:
    script_candidates = [
        "/official_eval/swe_bench_pro_eval.py" if script_mount else "",
        "/swe_bench_pro_eval.py",
        "/workspace/swe_bench_pro_eval.py",
        "/app/swe_bench_pro_eval.py",
    ]
    candidate_checks = " ".join(shlex.quote(path) for path in script_candidates if path)
    return (
        "set -euo pipefail; "
        f"for p in {candidate_checks}; do "
        "if [ -f \"$p\" ]; then export SWE_BENCH_PRO_EVAL_SCRIPT=\"$p\"; break; fi; "
        "done; "
        "if [ -z \"${SWE_BENCH_PRO_EVAL_SCRIPT:-}\" ]; then "
        "command -v swe_bench_pro_eval.py >/dev/null 2>&1 "
        "&& SWE_BENCH_PRO_EVAL_SCRIPT=$(command -v swe_bench_pro_eval.py) || true; "
        "fi; "
        "if [ -z \"${SWE_BENCH_PRO_EVAL_SCRIPT:-}\" ]; then "
        "echo 'swe_bench_pro_eval.py not found in official Docker image or mounted evaluator dir' >&2; "
        "exit 127; "
        "fi; "
        "python \"$SWE_BENCH_PRO_EVAL_SCRIPT\" "
        "--raw_sample_path /eval/raw_sample.csv "
        "--patch_path /eval/predictions.jsonl "
        "--output_dir /eval/outputs "
        f"--instance_id {shlex.quote(case.instance_id)}"
    )


def run_host_official_eval(
    case: SWEBenchProCase,
    args: argparse.Namespace,
    generated: str,
    *,
    script: Path,
    image: str,
) -> dict[str, Any]:
    scripts_dir = args.official_eval_scripts_dir
    if scripts_dir is None or not scripts_dir.exists():
        return official_eval_unavailable(
            "official_eval_scripts_dir_missing",
            image=image,
            script=str(script),
            scripts_dir=str(scripts_dir or ""),
        )
    eval_root = Path(tempfile.mkdtemp(prefix="swebench_pro_eval_"))
    input_paths = write_official_eval_inputs(case, generated, eval_root)
    output_dir = eval_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    dockerhub_username = (
        args.dockerhub_username
        or str(args.official_docker_image_repo).split("/", 1)[0]
    )
    command = [
        sys.executable,
        str(script.resolve(strict=False)),
        "--raw_sample_path",
        str(input_paths["raw_sample_path"]),
        "--patch_path",
        str(input_paths["patch_json_path"]),
        "--output_dir",
        str(output_dir),
        "--scripts_dir",
        str(scripts_dir.resolve(strict=False)),
        "--num_workers",
        "1",
        "--dockerhub_username",
        dockerhub_username,
    ]
    if args.use_local_docker:
        command.append("--use_local_docker")
    legacy_context = {
        "script": str(script),
        "scripts_dir": str(scripts_dir),
        "dockerhub_username": dockerhub_username,
        "use_local_docker": bool(args.use_local_docker),
        "patch_json_path": str(input_paths["patch_json_path"]),
        "host_official_eval": True,
        "timeout_seconds": int(args.official_eval_timeout_seconds),
    }
    official_eval_timeout = (
        None
        if int(args.official_eval_timeout_seconds) <= 0
        else int(args.official_eval_timeout_seconds)
    )
    eval_env = official_eval_subprocess_env(args)
    legacy_context["subprocess_timeout_seconds"] = official_eval_timeout
    legacy_context["docker_sdk_timeout_env"] = eval_env.get(
        "SWEAP_LOCAL_DOCKER_CLIENT_TIMEOUT_SECONDS",
        "",
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(script.resolve(strict=False).parent),
            text=True,
            capture_output=True,
            check=False,
            timeout=official_eval_timeout,
            env=eval_env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "attempted": True,
            "passed": False,
            "official_eval_required": True,
            "reason": "official_docker_eval_timeout",
            "image": image,
            "cwd": str(script.resolve(strict=False).parent),
            "command": command,
            "workspace": str(eval_root),
            "predictions_path": str(input_paths["predictions_path"]),
            "raw_sample_path": str(input_paths["raw_sample_path"]),
            "output_dir": str(output_dir),
            "eval_results_path": "",
            "returncode": None,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "stdout_tail": tail(exc.stdout if isinstance(exc.stdout, str) else ""),
            "stderr_tail": tail(exc.stderr if isinstance(exc.stderr, str) else ""),
            "legacy_context": legacy_context,
        }
    parsed = parse_official_eval_result(output_dir, case.instance_id)
    if completed.returncode != 0 and parsed["reason"] == "eval_results_json_missing":
        parsed["reason"] = "official_docker_eval_unavailable_no_results"
    return {
        "attempted": True,
        "passed": bool(parsed["passed"]),
        "official_eval_required": True,
        "reason": parsed["reason"],
        "image": image,
        "cwd": str(script.resolve(strict=False).parent),
        "command": command,
        "workspace": str(eval_root),
        "predictions_path": str(input_paths["predictions_path"]),
        "raw_sample_path": str(input_paths["raw_sample_path"]),
        "output_dir": str(output_dir),
        "eval_results_path": parsed.get("eval_results_path", ""),
        "returncode": completed.returncode,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
        "legacy_context": legacy_context,
    }


def run_official_eval(case: SWEBenchProCase, args: argparse.Namespace, generated: str) -> dict[str, Any]:
    integrity = patch_integrity_filter(generated)
    if not integrity["passed"]:
        return official_eval_unavailable(
            "patch_contains_nested_protocol_markers",
            integrity_filter=integrity,
        )
    image = official_docker_image(case, args)
    if not image:
        return official_eval_unavailable("official_docker_image_unavailable")
    checkout = checkout_absolute_path(case) if case.checkout_path else None
    if checkout is None or not checkout.exists():
        return official_eval_unavailable("official_eval_checkout_unavailable", image=image)
    checkout_cwd = str(checkout.resolve(strict=False))
    script = args.official_eval_script
    if script is None and args.official_eval_scripts_dir:
        candidate = args.official_eval_scripts_dir / "swe_bench_pro_eval.py"
        if candidate.exists():
            script = candidate
    script_mount = ""
    if script is not None:
        if not script.exists():
            return official_eval_unavailable(
                "official_eval_script_missing",
                image=image,
                script=str(script),
            )
        script_mount = str(script.resolve().parent)
    if script is not None and args.official_eval_scripts_dir is not None:
        return run_host_official_eval(
            case,
            args,
            generated,
            script=script,
            image=image,
        )
    eval_root = Path(tempfile.mkdtemp(prefix="swebench_pro_eval_"))
    input_paths = write_official_eval_inputs(case, generated, eval_root)
    output_dir = eval_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    shell_command = official_docker_shell_command(case, args, script_mount=script_mount)
    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "",
        "-v",
        f"{checkout_cwd}:/workspace",
        "-v",
        f"{eval_root.resolve()}:/eval",
        "-w",
        "/workspace",
    ]
    if script_mount:
        command.extend(["-v", f"{script_mount}:/official_eval:ro"])
    command.extend(
        [
            image,
            "bash",
            "-lc",
            shell_command,
        ]
    )
    legacy_context = {
        "script": str(script or ""),
        "scripts_dir": str(args.official_eval_scripts_dir or ""),
        "dockerhub_username": args.dockerhub_username,
        "use_local_docker": bool(args.use_local_docker),
        "patch_json_path": str(input_paths["patch_json_path"]),
    }
    if not shutil.which("docker"):
        return official_eval_unavailable(
            "docker_cli_unavailable",
            image=image,
            workspace=str(eval_root),
            output_dir=str(output_dir),
            predictions_path=str(input_paths["predictions_path"]),
            raw_sample_path=str(input_paths["raw_sample_path"]),
            legacy_context=legacy_context,
        )
    started = time.perf_counter()
    official_eval_timeout = (
        None
        if int(args.official_eval_timeout_seconds) <= 0
        else int(args.official_eval_timeout_seconds)
    )
    eval_env = official_eval_subprocess_env(args)
    legacy_context["timeout_seconds"] = int(args.official_eval_timeout_seconds)
    legacy_context["subprocess_timeout_seconds"] = official_eval_timeout
    legacy_context["docker_sdk_timeout_env"] = eval_env.get(
        "SWEAP_LOCAL_DOCKER_CLIENT_TIMEOUT_SECONDS",
        "",
    )
    try:
        completed = subprocess.run(
            command,
            cwd=checkout_cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=official_eval_timeout,
            env=eval_env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "attempted": True,
            "passed": False,
            "official_eval_required": True,
            "reason": "official_docker_eval_timeout",
            "image": image,
            "cwd": checkout_cwd,
            "command": command,
            "workspace": str(eval_root),
            "predictions_path": str(input_paths["predictions_path"]),
            "raw_sample_path": str(input_paths["raw_sample_path"]),
            "output_dir": str(output_dir),
            "eval_results_path": "",
            "returncode": None,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "stdout_tail": tail(exc.stdout if isinstance(exc.stdout, str) else ""),
            "stderr_tail": tail(exc.stderr if isinstance(exc.stderr, str) else ""),
            "legacy_context": legacy_context,
        }
    parsed = parse_official_eval_result(output_dir, case.instance_id)
    if completed.returncode != 0 and parsed["reason"] == "eval_results_json_missing":
        parsed["reason"] = "official_docker_eval_unavailable_no_results"
    return {
        "attempted": True,
        "passed": bool(parsed["passed"]),
        "official_eval_required": True,
        "reason": parsed["reason"],
        "image": image,
        "cwd": checkout_cwd,
        "command": command,
        "workspace": str(eval_root),
        "predictions_path": str(input_paths["predictions_path"]),
        "raw_sample_path": str(input_paths["raw_sample_path"]),
        "output_dir": str(output_dir),
        "eval_results_path": parsed.get("eval_results_path", ""),
        "returncode": completed.returncode,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
        "legacy_context": legacy_context,
    }


def scale_prediction_patch(
    case: SWEBenchProCase,
    generated: str,
    workspace_meta: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if looks_like_unified_diff(generated):
        return generated, {
            "mode": "generated_unified_diff",
            "available": True,
            "chars": len(generated or ""),
        }
    checkout = Path(case.checkout_path) if case.checkout_path else None
    if checkout is None or not checkout.exists():
        return generated, {
            "mode": "strict_search_replace_unconverted",
            "available": False,
            "reason": "checkout_unavailable_for_git_diff",
            "chars": len(generated or ""),
        }
    if workspace_meta.get("mode") != "strict_search_replace" or workspace_meta.get("failures"):
        return generated, {
            "mode": "strict_search_replace_unconverted",
            "available": False,
            "reason": "workspace_patch_not_cleanly_applied",
            "chars": len(generated or ""),
        }
    completed = run_git(["diff", "--binary"], cwd=checkout, timeout=120)
    patch_text = completed.stdout if completed.returncode == 0 else ""
    if patch_text.strip():
        return patch_text, {
            "mode": "workspace_git_diff",
            "available": True,
            "returncode": completed.returncode,
            "chars": len(patch_text),
            "changed_files": workspace_meta.get("changed_files", []),
        }
    return generated, {
        "mode": "strict_search_replace_unconverted",
        "available": False,
        "reason": "workspace_git_diff_empty",
        "returncode": completed.returncode,
        "stderr_tail": tail(completed.stderr),
        "chars": len(generated or ""),
    }


def selected_segment_paths(
    case: SWEBenchProCase,
    hot_windows: list[DependencyWindow],
    dialect_plan: dict[str, Any],
) -> list[str]:
    paths = list(dialect_plan.get("selected_paths", []))
    if not paths:
        paths = [window.path for window in hot_windows if window.path]
    ordered: list[str] = []
    for path in paths:
        resolved = resolve_generated_path(path, case.files)
        if resolved and resolved not in ordered:
            ordered.append(resolved)
    return ordered


def fallback_segment_window(
    case: SWEBenchProCase,
    path: str,
    args: argparse.Namespace,
    *,
    window_id: int,
) -> DependencyWindow | None:
    content = case.files.get(path, "")
    if not content.strip():
        return None
    max_chars = max(1_000, min(12_000, int(getattr(args, "window_tokens", 512)) * 12))
    excerpt = content[:max_chars].rstrip()
    if not excerpt:
        return None
    end_line = max(1, excerpt.count("\n") + 1)
    return DependencyWindow(
        window_id=window_id,
        path=path,
        text=excerpt,
        score=0.0,
        start_line=1,
        end_line=end_line,
        start_token=0,
        end_token=max(1, len(excerpt.split())),
        activation_route_id=None,
        lexical_score=0.0,
        activation_score=0.0,
        exact_scope_match=False,
        route_source="segmented_restart_fallback_source_excerpt",
        recursive_map_window_id=None,
        recursive_pass="segmented_restart_fallback",
    )


def segment_hot_windows_for_path(
    case: SWEBenchProCase,
    args: argparse.Namespace,
    hot_windows: list[DependencyWindow],
    path: str,
    *,
    segment_index: int,
) -> tuple[list[DependencyWindow], dict[str, Any]]:
    exact = [window for window in hot_windows if window.path == path]
    if exact:
        return exact, {
            "path": path,
            "window_count": len(exact),
            "window_ids": [int(window.window_id) for window in exact],
            "fallback_window": False,
        }
    fallback = fallback_segment_window(
        case,
        path,
        args,
        window_id=-100_000 - int(segment_index),
    )
    if fallback is None:
        return [], {
            "path": path,
            "window_count": 0,
            "window_ids": [],
            "fallback_window": False,
            "reason": "target_path_has_no_hot_window_or_source_excerpt",
        }
    return [fallback], {
        "path": path,
        "window_count": 1,
        "window_ids": [int(fallback.window_id)],
        "fallback_window": True,
        "reason": "target_path_had_no_routed_hot_window",
    }


def segmented_instruction(instruction: str, path: str, index: int, total: int) -> str:
    return (
        f"{instruction}\n\n"
        f"Segment {index} of {total}: generate strict SEARCH/REPLACE block(s) "
        f"only for {path}. Do not edit or mention any other target file. Stop "
        "after the closing >>>> marker for this file segment."
    )


def segmented_loco_case(
    case: SWEBenchProCase,
    *,
    instruction: str,
    path: str,
) -> LoCoBenchCase:
    return LoCoBenchCase(
        row_index=case.row_index,
        instruction=instruction,
        files={path: case.files[path]},
        expected_patch=case.expected_patch,
        raw=case.raw,
    )


def generate_segmented_kv_direct(
    case: SWEBenchProCase,
    args: argparse.Namespace,
    *,
    instruction: str,
    hot_windows: list[DependencyWindow],
    prompt: str,
    phase3: dict[str, Any],
    dialect_plan: dict[str, Any],
) -> tuple[str, float, dict[str, Any]]:
    paths = selected_segment_paths(case, hot_windows, dialect_plan)
    if len(paths) <= 1:
        raise SegmentedKVDirectUnavailable(
            "kv_direct_segmented_restart_requires_multiple_target_paths",
            {"paths": paths},
        )
    if not hot_windows:
        raise SegmentedKVDirectUnavailable(
            "kv_direct_segmented_restart_requires_hot_windows",
            {"paths": paths},
        )

    generated_parts: list[str] = []
    segments: list[dict[str, Any]] = []
    total_ttft_ms = 0.0
    for index, path in enumerate(paths, start=1):
        segment_windows, window_meta = segment_hot_windows_for_path(
            case,
            args,
            hot_windows,
            path,
            segment_index=index,
        )
        if not segment_windows:
            raise SegmentedKVDirectUnavailable(
                "kv_direct_segmented_restart_path_has_no_safe_windows",
                {"path": path, "segment": window_meta, "paths": paths},
            )
        segment_instruction = segmented_instruction(instruction, path, index, len(paths))
        segment_case = segmented_loco_case(case, instruction=segment_instruction, path=path)
        segment_anchored_query = phase3_anchored_query(segment_case, segment_windows)
        segment_prompt = visible_prompt(segment_case, anchored_query=segment_anchored_query)
        args._current_row_index = int(case.row_index)
        try:
            phase3_result = kv_direct_prediction(
                segment_case,
                args,
                hot_windows=segment_windows,
                anchored_query=segment_anchored_query,
                prompt=segment_prompt,
                dialect_plan=dialect_plan,
            )
        except Exception as exc:  # noqa: BLE001 - convert segmented fallback failures to fail-closed metadata.
            raise SegmentedKVDirectUnavailable(
                f"kv_direct_segmented_restart_runtime_unavailable: {exc}",
                {"path": path, "segment": window_meta, "paths": paths},
            ) from exc
        generated_parts.append(phase3_result.text.strip())
        segment_ttft_ms = float(phase3_result.ttft_ms or 0.0)
        total_ttft_ms += segment_ttft_ms
        segments.append(
            {
                **window_meta,
                "segment_index": index,
                "segment_count": len(paths),
                "anchored_query": segment_anchored_query,
                "generated_chars": len(phase3_result.text or ""),
                "ttft_ms": segment_ttft_ms,
                "kv_max_new_tokens": int(args.kv_max_new_tokens),
                "kv_replacement_spillover_tokens": int(
                    getattr(
                        args,
                        "kv_replacement_spillover_tokens",
                        DEFAULT_KV_REPLACEMENT_SPILLOVER_TOKENS,
                    )
                ),
                "kv_decode_payload_token_cap": int(
                    getattr(
                        args,
                        "kv_decode_payload_token_cap",
                        DEFAULT_KV_DECODE_PAYLOAD_TOKEN_CAP,
                    )
                ),
                "runtime": phase3_result.metadata,
            }
        )

    phase3["runtime"] = {
        "engine": "segmented_kv_direct_restart",
        "generation_skipped": False,
        "segment_count": len(segments),
        "segments": segments,
        "dynamic_token_budget": getattr(args, "dynamic_token_budget", {}),
        "dynamic_dialect_switching": dialect_plan,
        "multi_file_context_stitching": {
            "active": True,
            "source_layer": 13,
            "target_layer": 14,
            "hot_window_count": len(hot_windows),
            "restart_unit": "selected_target_path",
            "fresh_mask_per_segment": True,
            "note": (
                "KV-direct restarted once per selected SWE target path; each "
                "restart passed only single-path LoCo hot windows so the "
                "dialect mask was freshly installed for that file."
            ),
        },
        "original_prompt_chars": len(prompt or ""),
    }
    return "\n".join(part for part in generated_parts if part), total_ttft_ms, phase3


def generation_failed_closed_row(
    case: SWEBenchProCase,
    args: argparse.Namespace,
    *,
    attempt: int,
    attempts: list[dict[str, Any]],
    dialect_plan: dict[str, Any],
    token_budget: dict[str, Any],
    hot_windows: list[DependencyWindow],
    max_k: int,
    anchored_query: str,
    phase3: dict[str, Any],
    processor_meta: dict[str, Any],
    route_meta: dict[str, Any],
    self_correction_payload: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    failure_reason = reason or str(dialect_plan.get("failure_reason", "generation_failed_closed"))
    dialect_plan["failed_closed"] = True
    dialect_plan["failure_reason"] = failure_reason
    phase3["runtime"] = {
        "generation_skipped": True,
        "reason": failure_reason,
        "dynamic_token_budget": token_budget,
        "dynamic_dialect_switching": dialect_plan,
    }
    phase2_map_windows = [window for window in hot_windows if window.recursive_pass == "pass1_map"]
    phase2_hop_windows = [window for window in hot_windows if window.recursive_pass == "pass2_hop"]
    return {
        "row_index": int(case.row_index),
        "instruction": case.instruction,
        "generated": "",
        "raw_generated": "",
        "format_canonicalized": False,
        "verbatim_readout_bridge": {
            "active": False,
            "reason": "generation_failed_closed",
            "file_path_sequence": [],
        },
        "expected_patch_present": bool(case.expected_patch),
        "pass_at_1": False,
        "patch_applied": False,
        "patch_block_count": 0,
        "changed_files": [],
        "patch_failures": [failure_reason],
        "diff_format": {
            "valid_strict_search_replace": False,
            "reason": "generation_failed_closed",
        },
        "dependency_constraint_pass": False,
        "benchmark_pass_policy": {
            "official_eval_required": True,
            "benchmark_pass_source": "official_eval",
            "local_test_gate_role": "preflight_only",
            "local_pass_can_set_benchmark_pass": False,
            "official_eval_available": False,
            "official_eval_reason": "generation_failed_closed_before_official_eval",
        },
        "unsafe_identifiers": [],
        "dialect_lock": {
            "active": bool(dialect_plan.get("active")),
            "target_path": "",
            "path_indexed": dialect_plan.get("paths", {}),
            "literal_violations": [],
            "style_violations": [],
            "dynamic_dialect_switching": True,
            "sampler_hard_switch_state": dialect_plan.get("sampler_hard_switch_state", ""),
            "failed_closed": True,
            "failure_reason": failure_reason,
        },
        "ttft_ms": 0.0,
        "dynamic_max_k": int(max_k),
        "dynamic_token_budget": token_budget,
        "hot_window_ids": [int(window.window_id) for window in hot_windows],
        "hot_window_paths": [window.path for window in hot_windows],
        "hot_window_route_sources": [window.route_source for window in hot_windows],
        "phase2_multi_hop": phase2_metadata(hot_windows, phase2_map_windows, phase2_hop_windows),
        "hot_window_spans": [window_span_payload(window) for window in hot_windows],
        "anchored_query": anchored_query,
        "phase3": phase3,
        "dependency_logits_processor": {
            **processor_meta,
            "context_whitelist": {
                "active": True,
                "authority_identifier_count": len(identifiers_from_windows(hot_windows)),
                "intervention_count": int(processor_meta.get("unsafe_token_id_count", 0)),
                "retry_attempt": attempt,
            },
            "runtime_context_whitelist": runtime_decoder_telemetry(phase3),
            "logit_mask": dialect_plan,
            "dynamic_dialect_switching": dialect_plan,
            "dynamic_token_budget": token_budget,
        },
        "runner_mode": "kv_direct_failed_closed",
        "swebench_pro": swe_case_metadata(case),
        "ruler_jit_indexing_compat": ruler_jit_case_metadata(args, case),
        "context_token_packing": getattr(args, "context_token_packing", {}),
        "multi_target_route": route_meta,
        "post_patch_test_gate": {
            "passed": False,
            "skipped": True,
            "required": bool(args.require_test_gate),
            "reason": "generation_failed_closed",
            "command": "",
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "elapsed_ms": 0.0,
            "selected_tests": [],
        },
        "official_eval": {
            "attempted": False,
            "passed": False,
            "official_eval_required": True,
            "reason": "generation_failed_closed_before_generation",
        },
        "self_correction": {
            **self_correction_payload,
            "attempts": attempts,
            "final_attempt": attempt,
            "retry_triggered": bool(attempts),
        },
    }


def generate_once(
    case: SWEBenchProCase,
    args: argparse.Namespace,
    *,
    instruction: str,
    hot_windows: list[DependencyWindow],
    anchored_query: str,
    prompt: str,
    phase3: dict[str, Any],
    processor_meta: dict[str, Any],
    dialect_plan: dict[str, Any],
    self_correction_payload: dict[str, Any],
) -> tuple[str, float, str, dict[str, Any]]:
    started = time.perf_counter()
    if args.prediction_mode == "kv_direct":
        if dialect_plan.get("segmented_restart"):
            generated, ttft_ms, phase3 = generate_segmented_kv_direct(
                case,
                args,
                instruction=instruction,
                hot_windows=hot_windows,
                prompt=prompt,
                phase3=phase3,
                dialect_plan=dialect_plan,
            )
            if not ttft_ms:
                ttft_ms = (time.perf_counter() - started) * 1000.0
            return generated, float(ttft_ms), "kv_direct_segmented_restart", phase3
        args._current_row_index = int(case.row_index)
        phase3_result = kv_direct_prediction(
            loco_case(case, instruction=instruction),
            args,
            hot_windows=hot_windows,
            anchored_query=anchored_query,
            prompt=prompt,
            dialect_plan=dialect_plan,
        )
        phase3["runtime"] = {
            **phase3_result.metadata,
            "multi_file_context_stitching": {
                "active": True,
                "source_layer": 13,
                "target_layer": 14,
                "hot_window_count": len(hot_windows),
                "note": (
                    "Layer-13 residuals are recaptured per selected SWE hot "
                    "window and stitched into one KV-direct materialization."
                ),
            },
        }
        ttft_ms = phase3_result.ttft_ms or ((time.perf_counter() - started) * 1000.0)
        return phase3_result.text, float(ttft_ms), "kv_direct", phase3
    if args.prediction_command:
        generated, command_ttft_ms = command_prediction(
            args.prediction_command,
            {
                "benchmark": BENCHMARK_NAME,
                "task_type": TASK_TYPE,
                "system_prompt": DIFF_PROTOCOL_CONSTRAINT,
                "visible_prompt": prompt,
                "instruction": anchored_query,
                "raw_instruction": case.instruction,
                "swebench_pro": swe_case_metadata(case),
                "files": case.files,
                "hot_windows": [window_payload(window) for window in hot_windows],
                "phase3": phase3,
                "generation_constraints": processor_meta,
                "logit_mask": dialect_plan,
                "dynamic_dialect_switching": dialect_plan,
                "dynamic_token_budget": getattr(args, "dynamic_token_budget", {}),
                "self_correction": self_correction_payload,
                "diff_reserved_tokens": int(args.diff_reserved_tokens),
            },
        )
        ttft_ms = command_ttft_ms or ((time.perf_counter() - started) * 1000.0)
        return generated, float(ttft_ms), "prediction_command", phase3
    generated, runner_mode = local_prediction(loco_case(case, instruction=instruction), args)
    return generated, (time.perf_counter() - started) * 1000.0, runner_mode, phase3


def run_case(case: SWEBenchProCase, args: argparse.Namespace) -> dict[str, Any]:
    base_instruction = case.instruction
    attempts: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    previous_digest = ""
    for attempt in range(max(0, int(args.max_system2_retries)) + 1):
        instruction = augment_instruction(base_instruction, previous_digest, attempt)
        prompt_length = len(visible_prompt(loco_case(case, instruction=instruction)).split())
        max_k = DynamicAllocatorConfig.calculate_max_k(TASK_TYPE, prompt_length)
        hot_windows, route_meta = multi_target_route_hot_windows(
            case,
            args,
            instruction=instruction,
            max_k=max_k,
        )
        anchored_query = phase3_anchored_query(loco_case(case, instruction=instruction), hot_windows)
        prompt = visible_prompt(loco_case(case, instruction=instruction), anchored_query=anchored_query)
        _generation_kwargs, processor_meta, safe_identifiers = dependency_generation_bundle(
            hot_windows,
            instruction,
        )
        generated_paths_hint = route_meta.get("selected_paths", [])
        token_budget = dynamic_token_budget_metadata(
            args,
            selected_paths=list(generated_paths_hint),
            strict_expected_paths=list(generated_paths_hint),
        )
        generation_args = scoped_generation_args(args, token_budget)
        dialect_plan = infer_dynamic_dialect_plan(
            case,
            hot_windows,
            list(generated_paths_hint),
            prediction_mode=generation_args.prediction_mode,
        )
        phase3 = phase3_metadata(
            generation_args,
            hot_windows=hot_windows,
            active=generation_args.prediction_mode == "kv_direct",
            fallback_reason=(
                "" if generation_args.prediction_mode == "kv_direct" else "prediction_mode is not kv_direct"
            ),
        )
        phase3["dynamic_token_budget"] = token_budget
        phase3["dynamic_dialect_switching"] = dialect_plan
        self_correction_payload = {
            "attempt": attempt,
            "max_retries": int(args.max_system2_retries),
            "previous_failure_digest": previous_digest,
            "authority_context_adjustment": {
                "failure_digest_identifiers": sorted(extract_identifiers(previous_digest)),
                "failure_digest_paths": sorted(path_like_mentions(previous_digest)),
                "mask_mode": "tighten_with_failure_context" if previous_digest else "base",
            },
        }
        if dialect_plan.get("failed_closed"):
            row = generation_failed_closed_row(
                case,
                args,
                attempt=attempt,
                attempts=attempts,
                dialect_plan=dialect_plan,
                token_budget=token_budget,
                hot_windows=hot_windows,
                max_k=max_k,
                anchored_query=anchored_query,
                phase3=phase3,
                processor_meta=processor_meta,
                route_meta=route_meta,
                self_correction_payload=self_correction_payload,
                reason=str(dialect_plan.get("failure_reason", "generation_failed_closed")),
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "patch_applied": False,
                    "dependency_constraint_pass": False,
                    "test_gate_passed": False,
                    "test_gate_reason": "generation_failed_closed",
                    "official_eval_passed": False,
                    "official_eval_reason": "generation_failed_closed_before_generation",
                    "changed_files": [],
                }
            )
            final = row
            break
        try:
            raw_generated, ttft_ms, runner_mode, phase3 = generate_once(
                case,
                generation_args,
                instruction=instruction,
                hot_windows=hot_windows,
                anchored_query=anchored_query,
                prompt=prompt,
                phase3=phase3,
                processor_meta=processor_meta,
                dialect_plan=dialect_plan,
                self_correction_payload=self_correction_payload,
            )
        except SegmentedKVDirectUnavailable as exc:
            if exc.metadata:
                dialect_plan["segmentation_failure"] = exc.metadata
            row = generation_failed_closed_row(
                case,
                args,
                attempt=attempt,
                attempts=attempts,
                dialect_plan=dialect_plan,
                token_budget=token_budget,
                hot_windows=hot_windows,
                max_k=max_k,
                anchored_query=anchored_query,
                phase3=phase3,
                processor_meta=processor_meta,
                route_meta=route_meta,
                self_correction_payload=self_correction_payload,
                reason=exc.reason,
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "patch_applied": False,
                    "dependency_constraint_pass": False,
                    "test_gate_passed": False,
                    "test_gate_reason": "generation_failed_closed",
                    "official_eval_passed": False,
                    "official_eval_reason": "generation_failed_closed_before_generation",
                    "changed_files": [],
                }
            )
            final = row
            break
        canonicalized = canonicalize_search_replace_output(raw_generated)
        bridged, bridge_meta = multi_file_verbatim_readout_bridge(case, hot_windows, canonicalized)
        memory_patch = apply_search_replace_blocks(case.files, bridged)
        workspace_applied, workspace_meta = apply_patch_to_workspace(case, bridged)
        patch_applied = bool(memory_patch.applied or workspace_applied)
        prediction_patch, prediction_patch_meta = scale_prediction_patch(
            case,
            bridged,
            workspace_meta,
        )
        changed_files = list(dict.fromkeys([*memory_patch.changed_files, *workspace_meta.get("changed_files", [])]))
        patch_failures = list(memory_patch.failures) + list(workspace_meta.get("failures", []))
        diff_diagnostic = diff_format_diagnostic(bridged, memory_patch, workspace_meta)
        block_paths = [block.path for block in parse_strict_blocks(bridged, case.files)]
        dialect_eval = evaluate_dialect(case, hot_windows, bridged, block_paths, safe_identifiers)
        dependency_constraint_pass = not (
            dialect_eval["unsafe_identifiers"]
            or dialect_eval["literal_violations"]
            or dialect_eval["style_violations"]
        )
        test_gate = run_post_patch_test_gate(case, args, patch_applied=patch_applied)
        has_replace_separator = bool(
            re.search(r"====\s*REPLACE", bridged or "", flags=re.IGNORECASE)
        )
        official_eval_patch_applicable = bool(
            patch_applied
            and (
                has_replace_separator
                or workspace_meta.get("mode") == "git_apply"
            )
        )
        integrity_filter = patch_integrity_filter(prediction_patch)
        if official_eval_patch_applicable and not integrity_filter["passed"]:
            official_eval = official_eval_unavailable(
                "patch_contains_nested_protocol_markers",
                integrity_filter=integrity_filter,
            )
        elif official_eval_patch_applicable:
            official_eval = run_official_eval(case, args, prediction_patch)
        else:
            official_eval = official_eval_unavailable(
                "patch_not_applicable_before_official_eval",
                patch_applied=patch_applied,
                has_replace_separator=has_replace_separator,
                patch_block_count=int(memory_patch.block_count),
                patch_failures=patch_failures,
                workspace_apply=workspace_meta,
                integrity_filter=integrity_filter,
            )
        pass_at_1 = bool(official_eval.get("passed"))
        benchmark_pass_policy = {
            "official_eval_required": True,
            "benchmark_pass_source": "official_eval",
            "local_test_gate_role": "preflight_only",
            "local_pass_can_set_benchmark_pass": False,
            "official_eval_available": bool(official_eval.get("attempted")),
            "official_eval_reason": official_eval.get("reason", ""),
        }
        phase2_map_windows = [window for window in hot_windows if window.recursive_pass == "pass1_map"]
        phase2_hop_windows = [window for window in hot_windows if window.recursive_pass == "pass2_hop"]
        row = {
            "row_index": int(case.row_index),
            "instruction": case.instruction,
            "generated": bridged,
            "scale_prediction_patch": prediction_patch,
            "scale_prediction_patch_meta": prediction_patch_meta,
            "raw_generated": raw_generated,
            "format_canonicalized": canonicalized != raw_generated,
            "verbatim_readout_bridge": bridge_meta,
            "expected_patch_present": bool(case.expected_patch),
            "pass_at_1": pass_at_1,
            "patch_applied": patch_applied,
            "patch_block_count": int(memory_patch.block_count),
            "changed_files": changed_files,
            "patch_failures": patch_failures,
            "diff_format": diff_diagnostic,
            "dependency_constraint_pass": bool(dependency_constraint_pass),
            "benchmark_pass_policy": benchmark_pass_policy,
            "unsafe_identifiers": dialect_eval["unsafe_identifiers"],
            "dialect_lock": dialect_eval["dialect_lock"],
            "ttft_ms": float(ttft_ms),
            "dynamic_max_k": int(max_k),
            "dynamic_token_budget": token_budget,
            "hot_window_ids": [int(window.window_id) for window in hot_windows],
            "hot_window_paths": [window.path for window in hot_windows],
            "hot_window_route_sources": [window.route_source for window in hot_windows],
            "phase2_multi_hop": phase2_metadata(hot_windows, phase2_map_windows, phase2_hop_windows),
            "hot_window_spans": [window_span_payload(window) for window in hot_windows],
            "anchored_query": anchored_query,
            "phase3": phase3,
            "dependency_logits_processor": {
                **processor_meta,
                "context_whitelist": {
                    "active": True,
                    "authority_identifier_count": len(identifiers_from_windows(hot_windows)),
                    "intervention_count": int(
                        processor_meta.get("unsafe_token_id_count", 0)
                    )
                    + len(extract_identifiers(previous_digest)),
                    "retry_attempt": attempt,
                },
                "runtime_context_whitelist": runtime_decoder_telemetry(phase3),
                "logit_mask": dialect_plan,
                "dynamic_dialect_switching": dialect_plan,
                "dynamic_token_budget": token_budget,
            },
            "runner_mode": runner_mode,
            "swebench_pro": swe_case_metadata(case),
            "ruler_jit_indexing_compat": ruler_jit_case_metadata(args, case),
            "context_token_packing": getattr(args, "context_token_packing", {}),
            "multi_target_route": route_meta,
            "post_patch_test_gate": test_gate.__dict__,
            "official_eval": official_eval,
            "self_correction": {
                **self_correction_payload,
                "attempts": attempts,
                "final_attempt": attempt,
                "retry_triggered": bool(attempts),
            },
        }
        attempts.append(
            {
                "attempt": attempt,
                "patch_applied": patch_applied,
                "dependency_constraint_pass": dependency_constraint_pass,
                "test_gate_passed": test_gate.passed,
                "test_gate_reason": test_gate.reason,
                "official_eval_passed": official_eval.get("passed", False),
                "official_eval_reason": official_eval.get("reason", ""),
                "changed_files": changed_files,
            }
        )
        final = row
        if pass_at_1:
            break
        previous_digest = failure_digest(row)
    return final or {}


def augment_instruction(instruction: str, failure: str, attempt: int) -> str:
    if attempt <= 0 or not failure:
        return instruction
    return (
        f"{instruction}\n\n"
        "Previous generated patch failed validation. Correct only the strict "
        "SEARCH/REPLACE diff. Failure digest:\n"
        f"{failure}"
    )


def failure_digest(row: dict[str, Any]) -> str:
    gate = row.get("post_patch_test_gate", {})
    parts = [
        "patch_failures=" + "; ".join(row.get("patch_failures", [])[:3]),
        "unsafe_identifiers=" + ", ".join(row.get("unsafe_identifiers", [])[:12]),
        "test_reason=" + stringify(gate.get("reason")),
        "stderr=" + stringify(gate.get("stderr_tail"))[-1000:],
    ]
    return "\n".join(part for part in parts if part.split("=", 1)[-1])


def evaluate_dialect(
    case: SWEBenchProCase,
    hot_windows: list[DependencyWindow],
    generated: str,
    paths: list[str],
    safe_identifiers: set[str],
) -> dict[str, Any]:
    authority = set(safe_identifiers) | identifiers_from_windows(hot_windows) | set(COMMON_LOCAL_IDENTIFIERS)
    unsafe = unsafe_identifier_violations(generated, authority)
    literal_violations: list[str] = []
    style_violations: list[str] = []
    per_path: dict[str, Any] = {}
    for path in dict.fromkeys(path for path in paths if path):
        resolved = resolve_generated_path(path, case.files)
        nearby = [window.text for window in hot_windows if window.path == resolved]
        spec = infer_dialect_lock_for_source(
            resolved,
            case.files.get(resolved, ""),
            nearby_texts=nearby,
        )
        literals = dialect_literal_violations(generated, spec)
        styles = c11_case_style_violations(generated, authority, spec)
        literal_violations.extend(literals)
        style_violations.extend(styles)
        per_path[resolved] = {
            **dialect_lock_telemetry(spec),
            "literal_violations": literals,
            "style_violations": styles,
        }
    return {
        "unsafe_identifiers": unsafe,
        "literal_violations": sorted(set(literal_violations)),
        "style_violations": sorted(set(style_violations)),
        "dialect_lock": {
            "active": bool(per_path),
            "target_path": paths[0] if paths else "",
            "path_indexed": per_path,
            "literal_violations": sorted(set(literal_violations)),
            "style_violations": sorted(set(style_violations)),
            "dynamic_dialect_switching": True,
        },
    }


def diff_format_diagnostic(
    generated: str,
    memory_patch: Any,
    workspace_meta: dict[str, Any],
) -> dict[str, Any]:
    text = generated or ""
    has_search = bool(re.search(r"<<<<\s*SEARCH", text, flags=re.IGNORECASE))
    strict_ok = bool(memory_patch.block_count > 0 and not memory_patch.failures)
    git_ok = workspace_meta.get("mode") == "git_apply" and workspace_meta.get("returncode") == 0
    return {
        "valid_strict_search_replace": strict_ok,
        "reason": (
            "strict_search_replace_block_applied"
            if strict_ok
            else "unified_diff_applied"
            if git_ok
            else "malformed_or_unapplied_diff"
        ),
        "generated_chars": len(text),
        "has_search_marker": has_search,
        "has_separator": "====" in text,
        "has_replace_marker": bool(re.search(r"\bREPLACE\b", text, flags=re.IGNORECASE)),
        "has_close_marker": ">>>>" in text,
        "workspace_apply": workspace_meta,
    }


def phase2_metadata(
    hot_windows: list[DependencyWindow],
    phase2_map_windows: list[DependencyWindow],
    phase2_hop_windows: list[DependencyWindow],
) -> dict[str, Any]:
    return {
        "active": bool(phase2_map_windows and phase2_hop_windows),
        "blend": {"query_weight": 0.4, "map_weight": 0.6, "normalized": True},
        "map_window_ids": [int(window.window_id) for window in phase2_map_windows],
        "map_paths": [window.path for window in phase2_map_windows],
        "hop_window_ids": [int(window.window_id) for window in phase2_hop_windows],
        "hop_paths": [window.path for window in phase2_hop_windows],
        "injected_identifier_count": len(identifiers_from_windows(hot_windows)),
    }


def runtime_decoder_telemetry(phase3: dict[str, Any]) -> dict[str, Any]:
    runtime = phase3.get("runtime", {}) if isinstance(phase3, dict) else {}
    if not isinstance(runtime, dict):
        return {}
    phase4 = runtime.get("phase4", {})
    if isinstance(phase4, dict) and isinstance(phase4.get("context_whitelist"), dict):
        return dict(phase4["context_whitelist"])
    segments = runtime.get("segments", [])
    if isinstance(segments, list):
        segment_telemetry = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_runtime = segment.get("runtime", {})
            segment_phase4 = (
                segment_runtime.get("phase4", {})
                if isinstance(segment_runtime, dict)
                else {}
            )
            context = (
                segment_phase4.get("context_whitelist", {})
                if isinstance(segment_phase4, dict)
                else {}
            )
            if isinstance(context, dict):
                segment_telemetry.append(
                    {
                        "path": segment.get("path", ""),
                        "segment_index": segment.get("segment_index"),
                        **context,
                    }
                )
        return {
            "active": bool(segment_telemetry),
            "mode": "segmented_runtime_context_whitelist",
            "segments": segment_telemetry,
            "boundary_token_count": sum(
                int(segment.get("boundary_token_count", 0))
                for segment in segment_telemetry
            ),
            "intervention_count": sum(
                int(segment.get("intervention_count", 0))
                for segment in segment_telemetry
            ),
        }
    return {}


def window_payload(window: DependencyWindow) -> dict[str, Any]:
    return {**window_span_payload(window), "text": window.text}


def window_span_payload(window: DependencyWindow) -> dict[str, Any]:
    return {
        "window_id": int(window.window_id),
        "path": window.path,
        "start_line": int(window.start_line),
        "end_line": int(window.end_line),
        "start_token": int(window.start_token),
        "end_token": int(window.end_token),
        "activation_route_id": window.activation_route_id,
        "score": float(window.score),
        "lexical_score": float(window.lexical_score),
        "activation_score": float(window.activation_score),
        "route_source": window.route_source,
        "exact_scope_match": bool(window.exact_scope_match),
        "recursive_map_window_id": window.recursive_map_window_id,
        "recursive_pass": window.recursive_pass,
    }


def swe_case_metadata(case: SWEBenchProCase) -> dict[str, Any]:
    return {
        "repo": case.repo,
        "instance_id": case.instance_id,
        "base_commit": case.base_commit,
        "repo_language": case.repo_language,
        "requirements": case.requirements,
        "interface": case.interface,
        "fail_to_pass": case.fail_to_pass,
        "pass_to_pass": case.pass_to_pass,
        "before_repo_set_cmd": case.before_repo_set_cmd,
        "selected_test_files_to_run": case.selected_test_files_to_run,
        "dockerhub_tag": case.dockerhub_tag,
        "checkout_path": case.checkout_path,
        "official_eval_rule": "(fail_to_pass | pass_to_pass) <= passed_tests",
    }


def tail(text: str, limit: int = 4000) -> str:
    text = text or ""
    return text[-limit:]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(pct)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def metric_key(system: str, metric_name: str) -> str:
    system_key = re.sub(r"[^a-z0-9]+", "_", system.lower()).strip("_")
    metric_key_part = re.sub(r"[^a-z0-9]+", "_", metric_name.lower()).strip("_")
    return f"{system_key}_{metric_key_part}_delta"


def matching_baselines() -> list[dict[str, Any]]:
    return [
        baseline
        for baseline in COMPARISON_BASELINES
        if str(baseline.get("benchmark")) in {BENCHMARK_NAME, "SWE-bench Pro"}
    ]


def build_report(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    n = len(results)
    passed = sum(1 for result in results if result["pass_at_1"])
    ttft_values = [float(result["ttft_ms"]) for result in results]
    pass_at_1 = 0.0 if n == 0 else float(passed / n)
    metrics: dict[str, Any] = {
        "n": int(n),
        "passed": int(passed),
        "pass_at_1": pass_at_1,
        "mean_ttft_ms": 0.0 if n == 0 else float(sum(ttft_values) / n),
        "p95_ttft_ms": percentile(ttft_values, 0.95),
        "max_ttft_ms": max(ttft_values) if ttft_values else 0.0,
        "test_gate_required": bool(args.require_test_gate),
        "test_gate_passed": sum(
            1 for result in results if result.get("post_patch_test_gate", {}).get("passed")
        ),
        "test_gate_skipped": sum(
            1 for result in results if result.get("post_patch_test_gate", {}).get("skipped")
        ),
        "official_eval_required": True,
        "official_eval_attempted": sum(
            1 for result in results if result.get("official_eval", {}).get("attempted")
        ),
        "official_eval_passed": sum(
            1 for result in results if result.get("official_eval", {}).get("passed")
        ),
        "official_eval_unavailable": sum(
            1 for result in results if not result.get("official_eval", {}).get("attempted")
        ),
        "benchmark_pass_requires_official_eval": True,
        "local_pass_can_set_benchmark_pass": False,
    }
    comparison_rows = []
    for baseline in matching_baselines():
        score = float(baseline["score"])
        delta = (pass_at_1 - score) * 100.0
        metrics[metric_key(str(baseline["system"]), "pass_at_1")] = delta
        comparison_rows.append(
            {
                **baseline,
                "lazarus_score": pass_at_1,
                "delta_percentage_points": delta,
            }
        )
    summary_grid = [
        {
            "system": "Lazarus Layer 14",
            "benchmark": BENCHMARK_NAME,
            "token_bin": TOKEN_BIN,
            "metric_name": "pass_at_1",
            "score": pass_at_1,
            "mean_ttft_ms": metrics["mean_ttft_ms"],
        },
        *comparison_rows,
    ]
    runner_mode = args.prediction_mode if not args.prediction_command else "prediction_command"
    hard_switch_available, hard_switch_reason = kv_direct_sampler_hard_switch_available()
    failed_closed_count = sum(
        1
        for result in results
        if result.get("dependency_logits_processor", {})
        .get("dynamic_dialect_switching", {})
        .get("failed_closed")
    )
    segmented_restart_count = sum(
        1
        for result in results
        if result.get("dependency_logits_processor", {})
        .get("dynamic_dialect_switching", {})
        .get("sampler_hard_switch_state")
        == "segmented_restart"
    )
    sampler_hard_switch_state = (
        "active"
        if hard_switch_available
        else (
            "segmented_restart"
            if segmented_restart_count
            else ("failed_closed" if failed_closed_count else "unavailable")
        )
    )
    return {
        "benchmark": BENCHMARK_NAME,
        "token_bin": TOKEN_BIN,
        "dataset": {"id": args.dataset_id, "split": args.split},
        "snapshot_date": SNAPSHOT_DATE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference": {
            "task_type": TASK_TYPE,
            "dynamic_allocator": "DynamicAllocatorConfig.calculate_max_k",
            "structural_anchor": STRUCTURAL_ANCHOR,
            "diff_protocol_constraint": DIFF_PROTOCOL_CONSTRAINT,
            "dependency_logits_processor_active": True,
            "runner_mode": runner_mode,
            "start_index": int(args.start_index),
            "limit": int(args.limit),
            "artifacts": {
                "report_json": str(args.output),
                "csv_output": str(args.csv_output),
                "predictions_jsonl": str(args.predictions_jsonl),
            },
            "window_tokens": int(args.window_tokens),
            "activation_overlap_tokens": int(args.activation_overlap_tokens),
            "diff_reserved_tokens": int(args.diff_reserved_tokens),
            "dynamic_token_budget": {
                "active": True,
                "formula": "payload=(N_files * 1024); total=payload+spillover",
                "spillover_tokens": int(
                    getattr(
                        args,
                        "kv_replacement_spillover_tokens",
                        DEFAULT_KV_REPLACEMENT_SPILLOVER_TOKENS,
                    )
                ),
                "decode_payload_token_cap": int(
                    getattr(
                        args,
                        "kv_decode_payload_token_cap",
                        DEFAULT_KV_DECODE_PAYLOAD_TOKEN_CAP,
                    )
                ),
                "effective_values_recorded_per_case": True,
            },
            "dynamic_dialect_switching": {
                "active": True,
                "sampler_hard_switch": bool(hard_switch_available),
                "sampler_hard_switch_state": sampler_hard_switch_state,
                "sampler_hard_switch_reason": hard_switch_reason,
                "kv_direct_multi_file_policy": (
                    "segmented_restart_when_sampler_hard_switch_unavailable"
                ),
                "segmented_restart_count": int(segmented_restart_count),
                "failed_closed_count": int(failed_closed_count),
            },
            "loader": getattr(args, "loader", ""),
            "resolved_split": getattr(args, "resolved_split", args.split),
            "repo_cache": str(args.repo_cache),
            "clone_behavior": args.clone_behavior,
            "target_file_budget": int(args.target_file_budget),
            "max_context_files": int(args.max_context_files),
            "max_context_bytes": int(args.max_context_bytes),
            "target_context_tokens": int(args.target_context_tokens),
            "exact_context_packing": bool(args.exact_context_packing),
            "context_token_packing": getattr(args, "context_token_packing", {}),
            "runtime_dependency_setup": getattr(args, "runtime_dependency_setup", {}),
            "runtime_preflight": getattr(args, "runtime_preflight", {}),
            "max_system2_retries": int(args.max_system2_retries),
            "require_test_gate": bool(args.require_test_gate),
            "phase3": phase3_metadata(
                args,
                active=args.prediction_mode == "kv_direct",
                fallback_reason="" if args.prediction_mode == "kv_direct" else "prediction_mode is not kv_direct",
            ),
            "jit_indexing": getattr(args, "jit_indexing", {}),
            "ruler_jit_indexing_compat": (
                getattr(args, "jit_indexing", {}) or {}
            ).get("ruler_compat", {}),
            "preflight": {
                "phase_2_indexer": {
                    "fast_unfold_batched_path": bool(
                        (
                            getattr(args, "jit_indexing", {}) or {}
                        ).get("ruler_compat", {}).get("active", False)
                    ),
                    "implementation": RULER_JIT_INDEXING_IMPLEMENTATION,
                    "ruler_entrypoint": RULER_JIT_INDEXING_ENTRYPOINT,
                    "compat_signal": RULER_JIT_INDEXING_COMPAT_SIGNAL,
                    "swe_router": "run_locobench_benchmark.route_hot_windows",
                    "recursive_passes": ["pass1_map", "pass2_hop"],
                },
                "phase_3_injector": {
                    "kv_direct_layer14": args.prediction_mode == "kv_direct",
                    "source_layer": 13,
                    "target_layer": 14,
                    "multi_file_hot_windows": True,
                },
                "phase_4_diff": {
                    "strict_search_replace_required": True,
                    "regex": SEARCH_REPLACE_RE.pattern,
                    "multi_file_verbatim_bridge": True,
                    "dynamic_dialect_switching": sampler_hard_switch_state,
                    "sampler_hard_switch": bool(hard_switch_available),
                },
                "post_patch_test_gate": {
                    "required": bool(args.require_test_gate),
                    "role": "preflight_only",
                    "can_set_benchmark_pass": False,
                },
                "official_eval": {
                    "required": True,
                    "benchmark_pass_source": "official_docker_eval_results_json",
                    "docker_image_repo": str(args.official_docker_image_repo),
                    "script": str(args.official_eval_script or ""),
                    "scripts_dir": str(args.official_eval_scripts_dir or ""),
                    "predictions_jsonl": str(args.predictions_jsonl),
                },
            },
        },
        "metrics": metrics,
        "comparison_baselines": comparison_rows,
        "summary_grid": summary_grid,
        "cases": results,
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_type",
        "system",
        "benchmark",
        "sample",
        "token_bin",
        "metric_name",
        "score",
        "pass_at_1",
        "passed",
        "patch_applied",
        "patch_block_count",
        "ttft_ms",
        "test_gate_passed",
        "test_gate_skipped",
        "delta_percentage_points",
        "source",
    ]
    rows: list[dict[str, Any]] = [
        {
            "row_type": "summary",
            "system": "Lazarus Layer 14",
            "benchmark": BENCHMARK_NAME,
            "sample": report["metrics"]["n"],
            "token_bin": TOKEN_BIN,
            "metric_name": "pass_at_1",
            "score": report["metrics"]["pass_at_1"],
            "pass_at_1": report["metrics"]["pass_at_1"],
            "ttft_ms": report["metrics"]["mean_ttft_ms"],
            "source": "official_eval",
        }
    ]
    for baseline in report["comparison_baselines"]:
        rows.append(
            {
                "row_type": "baseline",
                "system": baseline["system"],
                "benchmark": baseline["benchmark"],
                "token_bin": baseline["token_bin"],
                "metric_name": baseline["metric_name"],
                "score": baseline["score"],
                "delta_percentage_points": baseline["delta_percentage_points"],
                "source": baseline["source"],
            }
        )
    for result in report["cases"]:
        gate = result.get("post_patch_test_gate", {})
        rows.append(
            {
                "row_type": "sample",
                "system": "Lazarus Layer 14",
                "benchmark": BENCHMARK_NAME,
                "sample": result["row_index"],
                "token_bin": TOKEN_BIN,
                "metric_name": "pass_at_1",
                "score": 1.0 if result["pass_at_1"] else 0.0,
                "pass_at_1": 1.0 if result["pass_at_1"] else 0.0,
                "passed": result["pass_at_1"],
                "patch_applied": result["patch_applied"],
                "patch_block_count": result["patch_block_count"],
                "ttft_ms": result["ttft_ms"],
                "test_gate_passed": gate.get("passed", False),
                "test_gate_skipped": gate.get("skipped", False),
                "source": "official_eval",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_predictions_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            swe = result.get("swebench_pro", {})
            handle.write(
                json.dumps(
                    {
                        "instance_id": stringify(swe.get("instance_id")),
                        "patch": stringify(
                            result.get("scale_prediction_patch")
                            or result.get("generated")
                        ),
                        "prefix": "lazarus",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def predictions_jsonl_path(args: argparse.Namespace) -> Path:
    if args.predictions_jsonl is not None:
        return args.predictions_jsonl
    return args.output.with_name(f"{args.output.stem}_predictions.jsonl")


def main() -> int:
    args = parse_args()
    args.predictions_jsonl = predictions_jsonl_path(args)
    ensure_runtime_dependencies(args)
    kv_runtime_preflight(args)
    load_context_tokenizer(args)
    if (
        args.prediction_mode == "kv_direct"
        and bool(args.exact_context_packing)
        and getattr(args, "_context_tokenizer", None) is None
    ):
        raise SystemExit(
            "Exact token context packing requires a tokenizer before KV-direct generation: "
            + json.dumps(getattr(args, "context_token_packing", {}), sort_keys=True)
        )
    if args.prediction_mode == "kv_direct" and not bool(args.exact_context_packing):
        print(
            "SWE-bench Pro official 1M lane: exact context packing disabled; "
            "tokenizer and deterministic padding are not required.",
            flush=True,
        )
    rows = load_benchmark_dataset(args)
    start = max(0, int(args.start_index))
    selected = rows[start:]
    if args.limit and args.limit > 0:
        selected = selected[: int(args.limit)]
    if not selected:
        raise SystemExit(
            f"SWE-bench Pro selection produced zero rows: start={start} limit={args.limit}."
        )
    cases = [extract_case(dict(row), start + index, args) for index, row in enumerate(selected)]
    args.jit_indexing = run_ruler_compatible_jit_indexing(cases, args)
    release_cuda_memory()
    results = [run_case(case, args) for case in cases]
    write_predictions_jsonl(args.predictions_jsonl, results)
    report = build_report(results, args)
    write_json(args.output, report)
    write_csv(args.csv_output, report)
    print(
        f"SWE-bench Pro parity complete: pass_at_1={report['metrics']['pass_at_1']:.6f} "
        f"n={report['metrics']['n']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
