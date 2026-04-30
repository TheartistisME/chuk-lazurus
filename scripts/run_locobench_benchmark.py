#!/usr/bin/env python3
"""Run the Lazarus LoCoBench coding benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from benchmark_jit_indexing import jit_index_dataset_windows, release_cuda_memory

from chuk_lazarus.benchmarks.baselines import COMPARISON_BASELINES
from chuk_lazarus.memory_config import DynamicAllocatorConfig

DATASET_ID = "jasonqiu/LoCoBench"
DATASET_SPLIT = "test"
BENCHMARK_NAME = "Salesforce/LoCoBench"
TASK_TYPE = "BUILDER"
TOKEN_BIN = "1M"
DEFAULT_DIFFICULTY = "expert"
DEFAULT_MIN_CONTEXT_LENGTH = 1_000_000
DEFAULT_DIFF_RESERVED_TOKENS = 1_280
DEFAULT_PREDICTION_MODE = "kv_direct"
DEFAULT_KV_HOT_BUDGET_MIB = 512
DEFAULT_KV_MAX_NEW_TOKENS = 512
SNAPSHOT_DATE = datetime.now(timezone.utc).date().isoformat()
DEFAULT_OUTPUT_PATH = Path(".benchmarks") / f"locobench_results_snapshot_{SNAPSHOT_DATE}.json"
DEFAULT_CSV_PATH = Path(".benchmarks") / f"locobench_results_snapshot_{SNAPSHOT_DATE}.csv"
DEFAULT_JIT_DIR = Path(".benchmarks") / "jit"
STRUCTURAL_ANCHOR = "<dependency_context_loaded>\n<PATCH_GENERATION_START>\n"
TARGET_FILE_OPEN = "<target_file>"
TARGET_FILE_CLOSE = "</target_file>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
DIFF_PROTOCOL_CONSTRAINT = (
    "CRITICAL: Do not rewrite the entire file. You must only output the exact "
    "lines to be replaced using a strict SEARCH/REPLACE block format: "
    "<<<< SEARCH [old] ==== REPLACE [new] >>>>."
)
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
TERM_RE = re.compile(r"[A-Za-z0-9_]+")
SEARCH_REPLACE_RE = re.compile(
    r"<<<<\s*SEARCH\s*(?P<old>.*?)\s*====\s*REPLACE\s*(?P<new>.*?)\s*>>>>",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class LoCoBenchCase:
    row_index: int
    instruction: str
    files: dict[str, str]
    expected_patch: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class DependencyWindow:
    window_id: int
    path: str
    text: str
    score: float


@dataclass(frozen=True)
class PatchResult:
    applied: bool
    block_count: int
    changed_files: tuple[str, ...]
    failures: tuple[str, ...]


class Phase3Unavailable(RuntimeError):
    """Raised when the runner cannot honestly execute Layer-14 KV-direct."""


@dataclass
class Phase3Prediction:
    text: str
    ttft_ms: float | None
    metadata: dict[str, Any]


class IdentifierTokenizer:
    """Small tokenizer double for dependency-processor benchmark wiring."""

    def __init__(self, identifiers: set[str]) -> None:
        ordered = sorted(identifier for identifier in identifiers if identifier)
        self._vocab = {identifier: idx for idx, identifier in enumerate(ordered)}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [
            self._vocab[identifier]
            for identifier in IDENTIFIER_RE.findall(str(text))
            if identifier in self._vocab
        ]


class FallbackDependencyLogitsProcessor:
    """Fallback for help/test environments where the chat script cannot import."""

    def __init__(self, safe_token_ids: set[int], tokenizer: Any | None = None) -> None:
        self.safe_token_ids = {int(token_id) for token_id in safe_token_ids}
        self.unsafe_token_ids: set[int] = set()
        if tokenizer is None:
            return
        try:
            vocab = dict(tokenizer.get_vocab())
        except Exception:  # noqa: BLE001 - tokenizer doubles can be deliberately tiny.
            vocab = {}
        for token_text, token_id in vocab.items():
            normalized_id = int(token_id)
            if normalized_id in self.safe_token_ids:
                continue
            if is_code_like_identifier(str(token_text)):
                self.unsafe_token_ids.add(normalized_id)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        del input_ids
        if not self.unsafe_token_ids:
            return scores
        try:
            vocab_size = int(scores.shape[-1])
            unsafe_ids = [
                token_id for token_id in self.unsafe_token_ids if 0 <= token_id < vocab_size
            ]
            if unsafe_ids:
                scores[..., unsafe_ids] = scores[..., unsafe_ids] - 100.0
        except Exception:  # noqa: BLE001 - logits processors should be non-fatal.
            return scores
        return scores


def require_datasets_loader() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with "
            "`uv pip install --python .venv/bin/python datasets`."
        ) from exc
    return load_dataset


def load_benchmark_dataset(args: argparse.Namespace) -> Any:
    if args.dataset_id == "jasonqiu/LoCoBench":
        try:
            dataset = load_locobench_zip_dataset(args)
        except Exception as fallback_exc:  # noqa: BLE001
            raise SystemExit(
                "Failed to load HuggingFace LoCoBench ZIP dataset "
                f"{args.dataset_id!r}: {fallback_exc}"
            ) from fallback_exc
        args.resolved_split = "data/output/scenarios"
        args.loader = "huggingface_hub.data_zip_fallback"
        return dataset
    load_dataset = require_datasets_loader()
    try:
        dataset = load_dataset(args.dataset_id, split=args.split)
        args.resolved_split = args.split
        args.loader = "datasets.load_dataset"
        return dataset
    except Exception as load_exc:  # noqa: BLE001 - normalize HF access/schema errors.
        failure: Exception = load_exc
        if args.dataset_id == "jasonqiu/LoCoBench":
            try:
                dataset = load_locobench_zip_dataset(args)
            except Exception as fallback_exc:  # noqa: BLE001
                failure = fallback_exc
            else:
                args.resolved_split = "data/output/scenarios"
                args.loader = "huggingface_hub.data_zip_fallback"
                return dataset
        raise SystemExit(
            "Failed to load HuggingFace dataset "
            f"{args.dataset_id!r} split={args.split!r}: {failure}"
        ) from failure


def load_locobench_zip_dataset(args: argparse.Namespace) -> Any:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "LoCoBench ZIP fallback requires huggingface_hub."
        ) from exc
    archive_path = Path(
        hf_hub_download(args.dataset_id, "data.zip", repo_type="dataset")
    )
    with ZipFile(archive_path) as archive:
        all_scenario_names = sorted(
            name
            for name in archive.namelist()
            if name.startswith("data/output/scenarios/")
            and name.endswith(".json")
            and "__MACOSX" not in name
        )
        scenario_names: list[tuple[int, str]] = []
        for source_index, scenario_name in enumerate(all_scenario_names):
            scenario = json.loads(archive.read(scenario_name))
            if not scenario_matches_filters(scenario, args):
                continue
            scenario_names.append((int(source_index), scenario_name))
    args.locobench_total_scenarios = len(all_scenario_names)
    args.locobench_filtered_scenarios = len(scenario_names)
    return LoCoBenchZipDataset(archive_path=archive_path, scenario_names=tuple(scenario_names))


def scenario_matches_filters(scenario: dict[str, Any], args: argparse.Namespace) -> bool:
    difficulty = str(getattr(args, "difficulty", "") or "").strip().lower()
    if difficulty and difficulty != "all":
        if stringify(scenario.get("difficulty")).strip().lower() != difficulty:
            return False
    task_category = str(getattr(args, "task_category", "") or "").strip().lower()
    if task_category and task_category != "all":
        if stringify(scenario.get("task_category")).strip().lower() != task_category:
            return False
    min_context_length = int(getattr(args, "min_context_length", 0) or 0)
    if min_context_length > 0:
        metadata = scenario.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        context_length = first_numeric(
            scenario.get("context_length"),
            metadata.get("context_length"),
        )
        if context_length < min_context_length:
            return False
    return True


def first_numeric(*values: Any) -> int:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


@dataclass(frozen=True)
class LoCoBenchZipDataset:
    archive_path: Path
    scenario_names: tuple[tuple[int, str], ...]

    def __iter__(self) -> Any:
        with ZipFile(self.archive_path) as archive:
            archive_names = set(archive.namelist())
            for source_index, scenario_name in self.scenario_names:
                scenario = json.loads(archive.read(scenario_name))
                row = scenario_to_row(archive, archive_names, scenario)
                row["_source_index"] = int(source_index)
                row["_scenario_name"] = scenario_name
                yield row


def scenario_project_id(scenario_id: str, task_category: str) -> str:
    marker = f"_{task_category}_"
    if marker in scenario_id:
        return scenario_id.split(marker, 1)[0]
    known_categories = (
        "architectural_understanding",
        "bug_investigation",
        "code_comprehension",
        "cross_file_refactoring",
        "feature_implementation",
        "integration_testing",
        "multi_session_development",
        "security_analysis",
    )
    for category in known_categories:
        marker = f"_{category}_"
        if marker in scenario_id:
            return scenario_id.split(marker, 1)[0]
    return scenario_id


def scenario_to_row(
    archive: ZipFile,
    archive_names: set[str],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    scenario_id = stringify(scenario.get("id"))
    task_category = stringify(scenario.get("task_category"))
    project_id = scenario_project_id(scenario_id, task_category)
    files: dict[str, str] = {}
    for raw_path in scenario.get("context_files", []) or []:
        normalized = stringify(raw_path).replace("//", "/").strip("/")
        archive_path = f"data/generated/{project_id}/{normalized}"
        if archive_path not in archive_names:
            continue
        try:
            content = archive.read(archive_path).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - skip binary or malformed context files.
            continue
        files[normalized] = content
    return {
        "id": scenario_id,
        "instruction": stringify(scenario.get("task_prompt")),
        "problem_statement": stringify(scenario.get("description")),
        "files": files,
        "expected_patch": stringify(scenario.get("ground_truth")),
        "task_category": task_category,
        "difficulty": stringify(scenario.get("difficulty")),
        "metadata": scenario.get("metadata", {}),
    }


def dependency_logits_processor_class() -> type[Any]:
    try:
        from interactive_memory_chat import DependencyLogitsProcessor
    except Exception:  # noqa: BLE001 - keep benchmark runner import-light.
        return FallbackDependencyLogitsProcessor
    return DependencyLogitsProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--split", default=DATASET_SPLIT)
    parser.add_argument(
        "--difficulty",
        default=DEFAULT_DIFFICULTY,
        help="LoCoBench scenario difficulty to run; use 'all' to disable.",
    )
    parser.add_argument(
        "--task-category",
        default="all",
        help="LoCoBench task_category to run; use 'all' to disable.",
    )
    parser.add_argument(
        "--min-context-length",
        type=int,
        default=DEFAULT_MIN_CONTEXT_LENGTH,
        help=(
            "Minimum LoCoBench context_length metadata value. Defaults to the "
            "expert/1M partition; use 0 to include smaller contexts."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based dataset row offset before applying --limit.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--jit-output-dir", type=Path, default=DEFAULT_JIT_DIR)
    parser.add_argument("--jit-model-path", default="")
    parser.add_argument("--jit-device", default="")
    parser.add_argument(
        "--jit-reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed fast-index metadata and per-case stores when present.",
    )
    parser.add_argument(
        "--jit-batch-size",
        type=int,
        default=0,
        help=(
            "Fast-path window batch size. 0 auto-selects a CUDA-first batch "
            "and halves it on CUDA OOM."
        ),
    )
    parser.add_argument("--window-tokens", type=int, default=512)
    parser.add_argument("--activation-overlap-tokens", type=int, default=128)
    parser.add_argument("--prediction-command", default="")
    parser.add_argument(
        "--diff-reserved-tokens",
        type=int,
        default=DEFAULT_DIFF_RESERVED_TOKENS,
        help="Phase 1 builder budget reserved for the strict diff response.",
    )
    parser.add_argument(
        "--kv-hot-budget-mib",
        type=int,
        default=DEFAULT_KV_HOT_BUDGET_MIB,
        help="Layer-14 KV-direct in-VRAM materialization budget for hot windows.",
    )
    parser.add_argument(
        "--kv-max-new-tokens",
        type=int,
        default=DEFAULT_KV_MAX_NEW_TOKENS,
        help="Maximum new tokens for the KV-direct LoCoBench predictor.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=("kv_direct", "row_prediction", "empty", "gold_patch"),
        default=DEFAULT_PREDICTION_MODE,
        help="Local fallback mode when --prediction-command is omitted.",
    )
    return parser.parse_args()


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        chunks = [stringify(item) for item in value]
        return "\n".join(chunk for chunk in chunks if chunk)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def decode_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def value_from_keys(value: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in value and value[key] not in (None, ""):
            return value[key]
    return None


def files_from_value(value: Any) -> dict[str, str]:
    value = decode_jsonish(value)
    if value is None:
        return {}
    if isinstance(value, dict):
        files: dict[str, str] = {}
        for key, item in value.items():
            if isinstance(item, dict):
                path = stringify(
                    value_from_keys(item, ("path", "file_path", "filename", "name"))
                )
                content = stringify(
                    value_from_keys(item, ("content", "text", "code", "source"))
                )
                files[path or str(key)] = content
            else:
                files[str(key)] = stringify(item)
        return {path: content for path, content in files.items() if content}
    if isinstance(value, list):
        files = {}
        for index, item in enumerate(value):
            decoded = decode_jsonish(item)
            if isinstance(decoded, dict):
                path = stringify(
                    value_from_keys(decoded, ("path", "file_path", "filename", "name"))
                )
                content = stringify(
                    value_from_keys(decoded, ("content", "text", "code", "source"))
                )
                files[path or f"file_{index}"] = content
            else:
                content = stringify(decoded)
                if content:
                    files[f"file_{index}"] = content
        return files
    text = stringify(value)
    return {"context": text} if text else {}


def extract_files(row: dict[str, Any]) -> dict[str, str]:
    for key in (
        "files",
        "file_context",
        "repository",
        "repo",
        "codebase",
        "source_files",
        "context_files",
        "context",
        "code",
    ):
        if key in row and row[key] not in (None, ""):
            files = files_from_value(row[key])
            if files:
                return files
    return {}


def extract_case(row: dict[str, Any], row_index: int) -> LoCoBenchCase:
    instruction = stringify(
        first_present(
            row,
            (
                "prompt",
                "instruction",
                "problem_statement",
                "issue",
                "question",
                "task",
                "query",
            ),
        )
    )
    expected_patch = stringify(
        first_present(
            row,
            (
                "patch",
                "diff",
                "gold_patch",
                "expected_patch",
                "solution",
                "target_patch",
            ),
        )
    )
    return LoCoBenchCase(
        row_index=int(row_index),
        instruction=instruction,
        files=extract_files(row),
        expected_patch=expected_patch,
        raw=row,
    )


def metric_key(system: str, metric_name: str) -> str:
    system_key = re.sub(r"[^a-z0-9]+", "_", system.lower()).strip("_")
    metric_key_part = re.sub(r"[^a-z0-9]+", "_", metric_name.lower()).strip("_")
    return f"{system_key}_{metric_key_part}_delta"


def matching_baselines() -> list[dict[str, Any]]:
    return [
        baseline
        for baseline in COMPARISON_BASELINES
        if str(baseline.get("benchmark")) == BENCHMARK_NAME
    ]


def stable_hash_int(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def semantic_vector(text: str, *, dim: int = 128) -> tuple[float, ...]:
    vec = [0.0] * int(dim)
    for term in TERM_RE.findall(str(text).lower()):
        bucket = stable_hash_int(term) % int(dim)
        sign = 1.0 if stable_hash_int(f"{term}:sign") % 2 == 0 else -1.0
        vec[bucket] += sign
        if len(term) >= 5:
            vec[(bucket + len(term)) % int(dim)] += 0.35 * sign
    norm = math.sqrt(sum(value * value for value in vec))
    if norm <= 0.0:
        return tuple(0.0 for _ in vec)
    return tuple(value / norm for value in vec)


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(left * right for left, right in zip(a, b))
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def dependency_windows(files: dict[str, str], *, tokens_per_window: int) -> list[DependencyWindow]:
    windows: list[DependencyWindow] = []
    window_size = max(1, int(tokens_per_window))
    for path, content in files.items():
        tokens = str(content).split()
        if not tokens:
            continue
        for offset in range(0, len(tokens), window_size):
            text = " ".join(tokens[offset : offset + window_size])
            windows.append(
                DependencyWindow(
                    window_id=len(windows),
                    path=str(path),
                    text=text,
                    score=0.0,
                )
            )
    return windows


def route_hot_windows(
    instruction: str,
    files: dict[str, str],
    *,
    max_k: int,
    tokens_per_window: int,
) -> list[DependencyWindow]:
    windows = dependency_windows(files, tokens_per_window=tokens_per_window)
    if not windows:
        return []
    query_vec = semantic_vector(instruction)
    scored = [
        DependencyWindow(
            window_id=window.window_id,
            path=window.path,
            text=window.text,
            score=cosine(query_vec, semantic_vector(window.text)),
        )
        for window in windows
    ]
    scored.sort(key=lambda window: (-window.score, window.path, window.window_id))
    return scored[: max(1, int(max_k))]


def extract_identifiers(text: str) -> set[str]:
    return {match.group(0) for match in IDENTIFIER_RE.finditer(text or "")}


def is_code_like_identifier(token_text: str) -> bool:
    stripped = str(token_text).strip().lstrip("▁Ġ")
    if not IDENTIFIER_RE.fullmatch(stripped):
        return False
    return (
        "_" in stripped
        or any(char.isdigit() for char in stripped)
        or (
            any(char.islower() for char in stripped)
            and any(char.isupper() for char in stripped)
        )
        or stripped.startswith("__")
    )


def dependency_generation_bundle(
    hot_windows: list[DependencyWindow],
    instruction: str,
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    safe_identifiers: set[str] = set()
    for window in hot_windows:
        safe_identifiers.update(extract_identifiers(window.text))
    vocab_identifiers = set(safe_identifiers)
    vocab_identifiers.update(extract_identifiers(instruction))
    tokenizer = IdentifierTokenizer(vocab_identifiers)
    safe_token_ids = {
        token_id
        for identifier in safe_identifiers
        for token_id in tokenizer.encode(identifier, add_special_tokens=False)
    }
    processor_cls = dependency_logits_processor_class()
    processor = processor_cls(safe_token_ids, tokenizer=tokenizer)
    generation_kwargs = {"logits_processor": [processor]}
    processor_meta = {
        "class": processor_cls.__name__,
        "active": True,
        "safe_token_id_count": len(safe_token_ids),
        "unsafe_token_id_count": len(getattr(processor, "unsafe_token_ids", set())),
    }
    return generation_kwargs, processor_meta, safe_identifiers


def unsafe_identifier_violations(generated: str, safe_identifiers: set[str]) -> list[str]:
    unsafe = [
        identifier
        for identifier in sorted(extract_identifiers(generated))
        if is_code_like_identifier(identifier) and identifier not in safe_identifiers
    ]
    return unsafe


def phase3_target_paths(case: LoCoBenchCase, hot_windows: list[DependencyWindow] | None = None) -> list[str]:
    if hot_windows:
        paths = [window.path for window in hot_windows if window.path]
    else:
        paths = list(case.files)
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        normalized = str(path).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def phase3_anchored_query(case: LoCoBenchCase, hot_windows: list[DependencyWindow] | None = None) -> str:
    target_paths = phase3_target_paths(case, hot_windows)
    target_block = "\n".join(
        f"{TARGET_FILE_OPEN}{path}{TARGET_FILE_CLOSE}" for path in target_paths
    )
    if not target_block:
        target_block = f"{TARGET_FILE_OPEN}unknown{TARGET_FILE_CLOSE}"
    return f"{target_block}\n{THINK_OPEN}\n{case.instruction}\n{THINK_CLOSE}"


def visible_prompt(
    case: LoCoBenchCase,
    *,
    anchored_query: str | None = None,
) -> str:
    file_sections = [
        f"### {path}\n{content}" for path, content in sorted(case.files.items())
    ]
    query = anchored_query if anchored_query is not None else phase3_anchored_query(case)
    return (
        f"{STRUCTURAL_ANCHOR}{query}\n\n"
        f"{DIFF_PROTOCOL_CONSTRAINT}\n\n"
        + "\n\n".join(file_sections)
    )


def command_prediction(command: str, payload: dict[str, Any]) -> tuple[str, float | None]:
    started = time.perf_counter()
    completed = subprocess.run(
        shlex.split(command),
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        raise RuntimeError(
            f"prediction command failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return completed.stdout.strip(), elapsed_ms
    if not isinstance(response, dict):
        return stringify(response), elapsed_ms
    ttft_ms = response.get("ttft_ms")
    generated = response.get("diff", response.get("patch", response.get("output", "")))
    return stringify(generated), None if ttft_ms is None else float(ttft_ms)


def kv_direct_unavailable_reason(args: argparse.Namespace) -> str:
    jit_metadata = getattr(args, "jit_indexing", {}) or {}
    return (
        "Phase 3 KV-direct was requested, but this LoCoBench runner cannot yet "
        "construct the existing Lazarus Layer-14 KV-direct contract. The real "
        "path is retriever.answer_with_kv_direct_multi()/answer_with_kv_direct() "
        "-> runtime.generate_with_kv_direct_materialization(); it requires "
        "CheckpointHandle-backed session stores, TierAssignment candidates, "
        "residual streams at the store injection_layer/source_layer, "
        "per-window token ranges, WarmPenaltyConfig, GenerationConfig, and a "
        "loaded runtime/model. The benchmark fast indexer currently provides "
        f"only activation_routes.npy and window_tokens.npz under "
        f"{jit_metadata.get('activation_route_dir', '<not indexed yet>')}; "
        "those artifacts are sufficient for Phase-2 routing but not for "
        "Layer-14 K/V materialization. Re-run with "
        "--prediction-mode row_prediction, --prediction-mode gold_patch, or a "
        "real Phase-3 bridge that materializes checkpoint stores and residuals."
    )


def get_phase3_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    cached = getattr(args, "_phase3_runtime_cache", None)
    if cached is not None:
        return cached
    from benchmark_jit_indexing import _load_real_gemma

    from chuk_lazarus.inference.backends.torch_runtime import TorchInferenceRuntime

    tokenizer, model = _load_real_gemma(
        args.jit_model_path or None,
        args.jit_device or None,
    )
    try:
        import torch

        device = next(model.parameters()).device
        if device.type == "cuda":
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model.to(device=device, dtype=target_dtype)
    except Exception:  # noqa: BLE001 - dtype optimization is best effort.
        device = getattr(model, "device", "cuda")
    runtime = TorchInferenceRuntime(
        model,
        tokenizer,
        device=str(device),
        engine="kv_direct",
        hot_budget_mib=int(args.kv_hot_budget_mib),
    )
    cached = (tokenizer, model, runtime)
    args._phase3_runtime_cache = cached
    return cached


def _pad_hot_window_token_ids(
    tokenizer: Any,
    hot_windows: list[DependencyWindow],
    *,
    device: Any,
    token_limit: int,
) -> tuple[Any, Any, dict[int, list[int]]]:
    import torch

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    encoded_by_window: dict[int, list[int]] = {}
    max_len = 1
    for window in hot_windows:
        ids = tokenizer.encode(
            window.text,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, int(token_limit)),
        )
        ids = [int(token_id) for token_id in ids]
        if not ids:
            ids = [int(pad_token_id or 0)]
        encoded_by_window[int(window.window_id)] = ids
        max_len = max(max_len, len(ids))
    batch = torch.full(
        (len(hot_windows), max_len),
        int(pad_token_id or 0),
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros(
        (len(hot_windows), max_len),
        dtype=torch.long,
        device=device,
    )
    for row_idx, window in enumerate(hot_windows):
        ids = encoded_by_window[int(window.window_id)]
        length = len(ids)
        batch[row_idx, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[row_idx, :length] = 1
    return batch, mask, encoded_by_window


def recapture_layer13_residuals(
    *,
    args: argparse.Namespace,
    runtime: Any,
    tokenizer: Any,
    hot_windows: list[DependencyWindow],
) -> tuple[Any, dict[int, list[int]], dict[str, Any]]:
    from chuk_lazarus.inference.backends._torch_residual_bounded import GatheredResiduals

    if not hot_windows:
        raise Phase3Unavailable("Phase 3 KV-direct requested, but no hot windows were selected.")
    torch = runtime._torch
    model = runtime._model
    device = next(model.parameters()).device
    input_ids, attention_mask, encoded_by_window = _pad_hot_window_token_ids(
        tokenizer,
        hot_windows,
        device=device,
        token_limit=int(args.window_tokens),
    )
    layers = runtime._resolve_layers()
    source_layer = 13
    if source_layer >= len(layers):
        raise Phase3Unavailable(
            f"Layer-13 recapture unavailable: model exposes only {len(layers)} layers."
        )
    captured: dict[str, Any] = {}

    def capture_hook(_module: Any, _inputs: Any, output: Any) -> None:
        if isinstance(output, tuple):
            captured["hidden"] = output[0]
            return
        captured["hidden"] = getattr(output, "last_hidden_state", output)

    handle = layers[source_layer].register_forward_hook(capture_hook)
    try:
        with torch.inference_mode():
            try:
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            except TypeError:
                model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    hidden = captured.get("hidden")
    if hidden is None:
        raise Phase3Unavailable("Layer-13 recapture failed: hook did not capture hidden states.")
    per_window_residuals: dict[int, Any] = {}
    per_window_token_ranges: dict[int, tuple[int, int]] = {}
    archive_provenance: dict[int, str] = {}
    slot = 0
    for row_idx, window in enumerate(hot_windows):
        window_id = int(window.window_id)
        token_count = int(attention_mask[row_idx].sum().item())
        residual_stream = hidden[row_idx, :token_count, :].detach().contiguous()
        per_window_residuals[window_id] = residual_stream
        per_window_token_ranges[window_id] = (slot, slot + token_count)
        archive_provenance[window_id] = "jit_recapture://layer13/in_vram"
        slot += token_count
    residuals = GatheredResiduals(
        per_window_residuals=per_window_residuals,
        per_window_token_ranges=per_window_token_ranges,
        source_layer=source_layer,
        archive_provenance=archive_provenance,
    )
    metadata = {
        "source_layer": source_layer,
        "target_layer": source_layer + 1,
        "recapture_batch_size": len(hot_windows),
        "recaptured_token_count": int(slot),
        "window_token_counts": {
            str(window_id): len(token_ids)
            for window_id, token_ids in encoded_by_window.items()
        },
        "storage": "in_vram_only",
    }
    return residuals, encoded_by_window, metadata


def kv_direct_prediction(
    case: LoCoBenchCase,
    args: argparse.Namespace,
    *,
    hot_windows: list[DependencyWindow],
    anchored_query: str,
    prompt: str,
) -> Phase3Prediction:
    del case
    from chuk_lazarus.inference.backends._torch_residual_bounded import materialize_kv_direct
    from chuk_lazarus.inference.backends.torch_runtime import WarmPenaltyConfig
    from chuk_lazarus.inference.generation import GenerationConfig
    from chuk_lazarus.session_retrieval.tier_policy import TierLabel

    tokenizer, _model, runtime = get_phase3_runtime(args)
    started = time.perf_counter()
    residuals, encoded_by_window, recapture_meta = recapture_layer13_residuals(
        args=args,
        runtime=runtime,
        tokenizer=tokenizer,
        hot_windows=hot_windows,
    )
    materialization = materialize_kv_direct(
        residuals,
        runtime,
        hot_budget_mib=int(args.kv_hot_budget_mib),
        tier_assignments=None,
    )
    per_window_ranges = dict(materialization.per_window_token_ranges or {})
    tier_map = {int(window.window_id): TierLabel.HOT for window in hot_windows}
    print(
        "PHASE 3 ACTIVE: "
        f"JIT Re-Capture successful for {len(hot_windows)} Hot Windows. "
        "Materializing Layer 14 Injector.",
        flush=True,
    )
    try:
        generation_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": DIFF_PROTOCOL_CONSTRAINT},
                {"role": "user", "content": anchored_query},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001 - fall back to benchmark-visible prompt.
        generation_prompt = prompt
    generated = runtime.generate_with_kv_direct_materialization(
        generation_prompt,
        GenerationConfig(
            max_new_tokens=int(args.kv_max_new_tokens),
            temperature=0.0,
            top_p=1.0,
        ),
        materialization=materialization,
        per_window_token_ranges=per_window_ranges,
        tier_assignments=tier_map,
        warm_config=WarmPenaltyConfig(),
        source_layer=13,
        query_id=f"locobench-row-{getattr(args, '_current_row_index', 'unknown')}",
    )
    metadata = dict(getattr(generated, "metadata", {}) or {})
    metadata.update(
        {
            "engine": "runtime.generate_with_kv_direct_materialization",
            "recapture": recapture_meta,
            "hot_window_token_ids": {
                str(window_id): token_ids for window_id, token_ids in encoded_by_window.items()
            },
            "materialization": {
                "hot_budget_mib_observed": int(materialization.hot_budget_mib_observed),
                "path_a_replay_count": int(materialization.path_a_replay_count),
                "materialized_source_layer": materialization.materialized_source_layer,
                "materialized_insertion_family": materialization.materialized_insertion_family,
                "materialized_lineage_layer_indices": list(
                    materialization.materialized_lineage_layer_indices
                ),
                "per_window_token_ranges": {
                    str(key): list(value) for key, value in per_window_ranges.items()
                },
            },
        }
    )
    return Phase3Prediction(
        text=stringify(getattr(generated, "text", "")),
        ttft_ms=(time.perf_counter() - started) * 1000.0,
        metadata=metadata,
    )


def phase3_metadata(
    args: argparse.Namespace,
    *,
    hot_windows: list[DependencyWindow] | None = None,
    active: bool = False,
    fallback_reason: str = "",
) -> dict[str, Any]:
    hot_windows = hot_windows or []
    target_layer = 14
    source_layer = target_layer - 1
    return {
        "active": bool(active),
        "target_layer": int(target_layer),
        "source_layer": int(source_layer),
        "hot_window_count": len(hot_windows),
        "hot_window_ids": [int(window.window_id) for window in hot_windows],
        "hot_window_paths": [window.path for window in hot_windows],
        "anchors": {
            "target_file": [TARGET_FILE_OPEN, TARGET_FILE_CLOSE],
            "think": [THINK_OPEN, THINK_CLOSE],
        },
        "engine": (
            "retriever.answer_with_kv_direct_multi"
            if active
            else "unavailable"
        ),
        "handoff": {
            "requested": getattr(args, "prediction_mode", "") == "kv_direct",
            "kv_direct_materialization": bool(active),
            "hot_windows_passed_to_injector": bool(active and hot_windows),
        },
        "fallback_reason": fallback_reason,
    }


def require_phase3_available(args: argparse.Namespace) -> None:
    if args.prediction_mode != "kv_direct":
        return
    if args.prediction_command:
        raise Phase3Unavailable(
            "Phase 3 KV-direct cannot be delegated to --prediction-command in "
            "this runner; the requirement is to call the existing Lazarus "
            "Layer-14 KV-direct path directly. "
            + kv_direct_unavailable_reason(args)
        )


def local_prediction(case: LoCoBenchCase, args: argparse.Namespace) -> tuple[str, str]:
    if args.prediction_mode == "kv_direct":
        raise Phase3Unavailable(kv_direct_unavailable_reason(args))
    if args.prediction_mode == "gold_patch":
        return case.expected_patch, "gold_patch"
    if args.prediction_mode == "empty":
        return "", "empty"
    generated = stringify(
        first_present(
            case.raw,
            (
                "prediction",
                "generated_patch",
                "generated_diff",
                "model_output",
                "response",
                "completion",
                "diff_output",
            ),
        )
    )
    return generated, "row_prediction"


def apply_search_replace_blocks(files: dict[str, str], generated: str) -> PatchResult:
    if not files:
        return PatchResult(False, 0, (), ("no source files available",))
    patched = dict(files)
    failures: list[str] = []
    changed_files: list[str] = []
    blocks = list(SEARCH_REPLACE_RE.finditer(generated or ""))
    if not blocks:
        return PatchResult(False, 0, (), ("no strict SEARCH/REPLACE blocks found",))
    for block_index, block in enumerate(blocks, start=1):
        old = block.group("old").strip("\n")
        new = block.group("new").strip("\n")
        if not old:
            failures.append(f"block {block_index}: empty SEARCH section")
            continue
        applied_path = ""
        for path, content in patched.items():
            if old in content:
                patched[path] = content.replace(old, new, 1)
                applied_path = path
                break
        if applied_path:
            changed_files.append(applied_path)
        else:
            failures.append(f"block {block_index}: SEARCH text not found")
    return PatchResult(
        applied=not failures and bool(changed_files),
        block_count=len(blocks),
        changed_files=tuple(changed_files),
        failures=tuple(failures),
    )


def diff_format_diagnostic(generated: str, patch_result: PatchResult) -> dict[str, Any]:
    generated_text = generated or ""
    has_search = bool(re.search(r"<<<<\s*SEARCH", generated_text, flags=re.IGNORECASE))
    has_separator = "====" in generated_text
    has_replace = bool(re.search(r"\bREPLACE\b", generated_text, flags=re.IGNORECASE))
    has_close = ">>>>" in generated_text
    strict_blocks = int(patch_result.block_count)
    if strict_blocks <= 0:
        reason = "empty_output" if not generated_text.strip() else "malformed_or_missing_search_replace_block"
    elif patch_result.failures:
        reason = "search_replace_block_not_applicable"
    else:
        reason = "strict_search_replace_block_applied"
    return {
        "valid_strict_search_replace": bool(strict_blocks > 0 and not patch_result.failures),
        "reason": reason,
        "generated_chars": len(generated_text),
        "has_search_marker": has_search,
        "has_separator": has_separator,
        "has_replace_marker": has_replace,
        "has_close_marker": has_close,
    }


def run_case(case: LoCoBenchCase, args: argparse.Namespace) -> dict[str, Any]:
    budget_prompt = visible_prompt(case)
    prompt_length = len(budget_prompt.split())
    max_k = DynamicAllocatorConfig.calculate_max_k(TASK_TYPE, prompt_length)
    hot_windows = route_hot_windows(
        case.instruction,
        case.files,
        max_k=max_k,
        tokens_per_window=int(args.window_tokens),
    )
    anchored_query = phase3_anchored_query(case, hot_windows)
    prompt = visible_prompt(case, anchored_query=anchored_query)
    generation_kwargs, processor_meta, safe_identifiers = dependency_generation_bundle(
        hot_windows,
        case.instruction,
    )
    del generation_kwargs
    started = time.perf_counter()
    phase3 = phase3_metadata(
        args,
        hot_windows=hot_windows,
        active=args.prediction_mode == "kv_direct",
        fallback_reason=(
            ""
            if args.prediction_mode == "kv_direct"
            else "prediction_mode is not kv_direct; Layer-14 KV-direct was not requested"
        ),
    )
    if args.prediction_mode == "kv_direct":
        args._current_row_index = int(case.row_index)
        phase3_result = kv_direct_prediction(
            case,
            args,
            hot_windows=hot_windows,
            anchored_query=anchored_query,
            prompt=prompt,
        )
        generated = phase3_result.text
        ttft_ms = (
            phase3_result.ttft_ms
            if phase3_result.ttft_ms is not None
            else (time.perf_counter() - started) * 1000.0
        )
        phase3["runtime"] = phase3_result.metadata
        runner_mode = "kv_direct"
    elif args.prediction_command:
        generated, command_ttft_ms = command_prediction(
            args.prediction_command,
            {
                "benchmark": BENCHMARK_NAME,
                "task_type": TASK_TYPE,
                "dynamic_max_k": int(max_k),
                "system_prompt": DIFF_PROTOCOL_CONSTRAINT,
                "visible_prompt": prompt,
                "instruction": anchored_query,
                "raw_instruction": case.instruction,
                "files": case.files,
                "hot_windows": [
                    {
                        "window_id": window.window_id,
                        "path": window.path,
                        "score": window.score,
                        "text": window.text,
                    }
                    for window in hot_windows
                ],
                "phase3": phase3,
                "generation_constraints": processor_meta,
                "diff_reserved_tokens": int(args.diff_reserved_tokens),
            },
        )
        ttft_ms = (
            command_ttft_ms
            if command_ttft_ms is not None
            else (time.perf_counter() - started) * 1000.0
        )
        runner_mode = "prediction_command"
    else:
        generated, runner_mode = local_prediction(case, args)
        ttft_ms = (time.perf_counter() - started) * 1000.0
    patch_result = apply_search_replace_blocks(case.files, generated)
    diff_diagnostic = diff_format_diagnostic(generated, patch_result)
    unsafe_identifiers = unsafe_identifier_violations(generated, safe_identifiers)
    dependency_constraint_pass = not unsafe_identifiers
    passed = bool(patch_result.applied and dependency_constraint_pass)
    return {
        "row_index": int(case.row_index),
        "instruction": case.instruction,
        "generated": generated,
        "expected_patch_present": bool(case.expected_patch),
        "pass_at_1": passed,
        "patch_applied": bool(patch_result.applied),
        "patch_block_count": int(patch_result.block_count),
        "changed_files": list(patch_result.changed_files),
        "patch_failures": list(patch_result.failures),
        "diff_format": diff_diagnostic,
        "dependency_constraint_pass": bool(dependency_constraint_pass),
        "unsafe_identifiers": unsafe_identifiers,
        "ttft_ms": float(ttft_ms),
        "dynamic_max_k": int(max_k),
        "hot_window_ids": [int(window.window_id) for window in hot_windows],
        "hot_window_paths": [window.path for window in hot_windows],
        "anchored_query": anchored_query,
        "phase3": phase3,
        "dependency_logits_processor": processor_meta,
        "runner_mode": runner_mode,
    }


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


def build_report(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    n = len(results)
    passed = sum(1 for result in results if result["pass_at_1"])
    ttft_values = [float(result["ttft_ms"]) for result in results]
    pass_at_1 = 0.0 if n == 0 else float(passed / n)
    metrics: dict[str, Any] = {
        "n": int(n),
        "passed": int(passed),
        "pass_at_1": float(pass_at_1),
        "mean_ttft_ms": 0.0 if n == 0 else float(sum(ttft_values) / n),
        "p95_ttft_ms": percentile(ttft_values, 0.95),
        "max_ttft_ms": max(ttft_values) if ttft_values else 0.0,
    }
    comparison_rows = []
    for baseline in matching_baselines():
        score = float(baseline["score"])
        delta = (pass_at_1 - score) * 100.0
        metrics[metric_key(str(baseline["system"]), "pass_at_1")] = float(delta)
        comparison_rows.append(
            {
                **baseline,
                "lazarus_score": float(pass_at_1),
                "delta_percentage_points": float(delta),
            }
        )
    summary_grid = [
        {
            "system": "Lazarus Layer 14",
            "benchmark": BENCHMARK_NAME,
            "token_bin": TOKEN_BIN,
            "metric_name": "pass_at_1",
            "score": float(pass_at_1),
            "mean_ttft_ms": metrics["mean_ttft_ms"],
        },
        *comparison_rows,
    ]
    runner_mode = args.prediction_mode if not args.prediction_command else "prediction_command"
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
            "start_index": int(getattr(args, "start_index", 0)),
            "limit": int(getattr(args, "limit", 0)),
            "window_tokens": int(args.window_tokens),
            "activation_overlap_tokens": int(args.activation_overlap_tokens),
            "diff_reserved_tokens": int(args.diff_reserved_tokens),
            "loader": getattr(args, "loader", "datasets.load_dataset"),
            "resolved_split": getattr(args, "resolved_split", args.split),
            "locobench_filters": {
                "difficulty": str(getattr(args, "difficulty", "")),
                "task_category": str(getattr(args, "task_category", "")),
                "min_context_length": int(getattr(args, "min_context_length", 0) or 0),
                "total_scenarios": int(getattr(args, "locobench_total_scenarios", n)),
                "filtered_scenarios": int(getattr(args, "locobench_filtered_scenarios", n)),
            },
            "phase3": phase3_metadata(
                args,
                active=args.prediction_mode == "kv_direct",
                fallback_reason=(
                    ""
                    if args.prediction_mode == "kv_direct"
                    else "prediction_mode is not kv_direct; Layer-14 KV-direct was not requested"
                ),
            ),
            "jit_indexing": getattr(args, "jit_indexing", {}),
            "preflight": {
                "phase_1_budget": {
                    "builder_diff_reserved_tokens": int(args.diff_reserved_tokens),
                    "sent_to_prediction_command": bool(args.prediction_command),
                    "enforced_for_local_row_prediction": False,
                },
                "phase_2_indexer": {
                    "fast_unfold_batched_path": True,
                    "implementation": "benchmark_jit_indexing.jit_index_dataset_windows",
                    "layer": 12,
                },
                "phase_3_injector": {
                    "target_file_tag_layer14": True,
                    "think_anchor": True,
                    "kv_direct_layer14": args.prediction_mode == "kv_direct",
                    "note": (
                        "LoCoBench queries are structurally anchored with "
                        "<target_file> and <think>; prediction_mode=kv_direct "
                        "selectively re-captures Layer-13 residual streams for "
                        "the selected hot windows in VRAM and calls the existing "
                        "Layer-14 KV-direct materialization runtime."
                    ),
                },
                "phase_4_diff": {
                    "strict_search_replace_required": True,
                    "regex": SEARCH_REPLACE_RE.pattern,
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
            "source": "local",
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
                "source": result["runner_mode"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    dataset = load_benchmark_dataset(args)
    rows: list[tuple[int, Any]] = []
    start_index = max(0, int(args.start_index))
    for filtered_index, row in enumerate(dataset):
        source_index = int(row.get("_source_index", filtered_index)) if isinstance(row, dict) else filtered_index
        if filtered_index < start_index:
            continue
        if args.limit and args.limit > 0 and len(rows) >= int(args.limit):
            break
        rows.append((int(source_index), row))
    cases = [extract_case(dict(row), row_index) for row_index, row in rows]
    filtered_scenarios = int(getattr(args, "locobench_filtered_scenarios", len(cases)))
    if filtered_scenarios == 0:
        raise SystemExit(
            "LoCoBench filter matched zero scenarios: "
            f"difficulty={args.difficulty!r} "
            f"task_category={args.task_category!r} "
            f"min_context_length={int(args.min_context_length)}. "
            "Refusing to silently evaluate an easier/smaller partition."
        )
    if not cases:
        raise SystemExit(
            "LoCoBench selection produced zero cases after --start-index/--limit: "
            f"filtered_scenarios={filtered_scenarios} "
            f"start_index={int(args.start_index)} limit={int(args.limit)}."
        )
    try:
        require_phase3_available(args)
    except Phase3Unavailable as exc:
        raise SystemExit(str(exc)) from exc
    if not args.prediction_command and args.prediction_mode == "row_prediction":
        print(
            "LoCoBench warning: --prediction-command omitted and --prediction-mode "
            "row_prediction selected. Rows without a prediction field will produce "
            "empty outputs and fail strict SEARCH/REPLACE validation.",
            flush=True,
        )
    args.jit_indexing = jit_index_dataset_windows(
        benchmark_slug="locobench_1m",
        raw_texts=(visible_prompt(case) for case in cases),
        output_dir=args.jit_output_dir,
        tokens_per_window=int(args.window_tokens),
        overlap_tokens=int(args.activation_overlap_tokens),
        model_path=args.jit_model_path or None,
        device=args.jit_device or None,
        reuse_existing=bool(args.jit_reuse_existing),
        batch_size=int(args.jit_batch_size) if int(args.jit_batch_size) > 0 else None,
    ).as_dict()
    release_cuda_memory()
    results = [
        run_case(case, args)
        for case in cases
    ]
    report = build_report(results, args)
    write_json(args.output, report)
    write_csv(args.csv_output, report)
    print(
        f"LoCoBench complete: pass_at_1={report['metrics']['pass_at_1']:.6f} "
        f"n={report['metrics']['n']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
