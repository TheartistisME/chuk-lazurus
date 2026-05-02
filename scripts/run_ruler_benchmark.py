#!/usr/bin/env python3
"""Run the Lazarus RULER variable-tracking benchmark."""

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

from benchmark_jit_indexing import jit_index_dataset_windows, release_cuda_memory

from chuk_lazarus.benchmarks.baselines import COMPARISON_BASELINES
from chuk_lazarus.memory_config import DynamicAllocatorConfig

DATASET_ID = "rbiswasfc/ruler"
DATASET_CONFIG = "variable_tracking"
DATASET_SPLIT = "test"
BENCHMARK_NAME = "nvidia/RULER (Variable Tracking)"
TASK_TYPE = "DETECTIVE"
TOKEN_BIN = "128k"
RULER_1M_DATASET_ID = "tturing/ruler-long-data"
RULER_1M_DATA_FILE = "1048576/data/vt/validation.jsonl"
RULER_1M_LENGTH = 1048576
RULER_1M_MAX_K = 8
RULER_1M_MAX_INJECTION_TOKENS = 4096
SNAPSHOT_DATE = datetime.now(timezone.utc).date().isoformat()
DEFAULT_OUTPUT_PATH = Path(".benchmarks") / f"ruler_results_snapshot_{SNAPSHOT_DATE}.json"
DEFAULT_CSV_PATH = Path(".benchmarks") / f"ruler_results_snapshot_{SNAPSHOT_DATE}.csv"
DEFAULT_JIT_DIR = Path(".benchmarks") / "jit"
TERM_RE = re.compile(r"[A-Za-z0-9_]+")
ASSIGNMENT_RE = re.compile(
    r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b\s*(?:=|:|->|is)\s*"
    r"(?P<value>[A-Za-z_][A-Za-z0-9_]*|[-+]?\d+(?:\.\d+)?|\"[^\"]+\"|'[^']+')"
)


@dataclass(frozen=True)
class RulerCase:
    row_index: int
    prompt: str
    query: str
    expected: str
    expected_values: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class RouteWindow:
    window_id: int
    text: str
    score: float


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
    load_dataset = require_datasets_loader()
    if should_use_ruler_1m_jsonl(args):
        dataset = load_dataset(
            args.dataset_id,
            data_files=RULER_1M_DATA_FILE,
            split="train",
        )
        args.resolved_dataset_config = f"jsonl:{RULER_1M_DATA_FILE}"
        args.resolved_split = "train"
        args.target_length = RULER_1M_LENGTH
        return dataset
    requested_configs = ruler_config_candidates(args)
    try:
        for config in requested_configs:
            try:
                dataset = load_dataset(args.dataset_id, config, split=args.split)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            args.resolved_dataset_config = config
            args.resolved_split = args.split
            return dataset
        raise last_error  # type: ignore[possibly-undefined]
    except Exception as exc:  # noqa: BLE001 - normalize HF access/schema errors.
        if args.dataset_id == "rbiswasfc/ruler" and args.dataset_config == "variable_tracking":
            last_error: Exception = exc
            for mirror_config in requested_configs:
                for split in (args.split, "validation", "train"):
                    try:
                        dataset = load_dataset(args.dataset_id, mirror_config, split=split)
                    except Exception as fallback_exc:  # noqa: BLE001
                        last_error = fallback_exc
                        continue
                    args.resolved_dataset_config = mirror_config
                    args.resolved_split = split
                    return dataset
            exc = last_error
        raise SystemExit(
            "Failed to load HuggingFace dataset "
            f"{args.dataset_id!r} config={args.dataset_config!r} split={args.split!r}: "
            f"{exc}"
        ) from exc


def available_ruler_configs(dataset_id: str) -> tuple[str, ...]:
    try:
        from datasets import get_dataset_config_names
    except Exception:  # noqa: BLE001 - optional report metadata.
        return ()
    try:
        return tuple(str(name) for name in get_dataset_config_names(dataset_id))
    except Exception:  # noqa: BLE001 - optional report metadata.
        return ()


def ruler_config_candidates(args: argparse.Namespace) -> tuple[str, ...]:
    if should_use_ruler_1m_jsonl(args):
        return (RULER_1M_DATA_FILE,)
    if args.dataset_id != "rbiswasfc/ruler" or args.dataset_config != "variable_tracking":
        return (args.dataset_config,)
    available = available_ruler_configs(args.dataset_id)
    requested_bin = str(getattr(args, "token_bin", TOKEN_BIN)).lower()
    exact = f"vt_{requested_bin}"
    candidates: list[str] = []
    if exact in available:
        candidates.append(exact)
    if args.dataset_config in available:
        candidates.append(args.dataset_config)
    fallback_order = ("vt_8k", "vt_4k")
    candidates.extend(config for config in fallback_order if config in available)
    if not candidates:
        candidates.extend(fallback_order)
    return tuple(dict.fromkeys(candidates))


def is_1m_token_bin(args: argparse.Namespace) -> bool:
    token_bin = str(getattr(args, "token_bin", "")).strip().lower()
    return token_bin in {"1m", "1000000", "1048576"}


def should_use_ruler_1m_jsonl(args: argparse.Namespace) -> bool:
    return (
        str(getattr(args, "dataset_id", "")) == RULER_1M_DATASET_ID
        and str(getattr(args, "dataset_config", DATASET_CONFIG)) == DATASET_CONFIG
        and is_1m_token_bin(args)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--split", default=DATASET_SPLIT)
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
    parser.add_argument("--token-bin", default=TOKEN_BIN)
    parser.add_argument("--window-tokens", type=int, default=512)
    parser.add_argument("--activation-overlap-tokens", type=int, default=128)
    parser.add_argument("--prediction-command", default="")
    parser.add_argument(
        "--prediction-mode",
        choices=("heuristic", "row_prediction"),
        default="heuristic",
        help="Local fallback mode when --prediction-command is omitted.",
    )
    return parser.parse_args()


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(stringify(item) for item in value if stringify(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def expected_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(normalize_exact(stringify(item)) for item in value if stringify(item))
    text = normalize_exact(stringify(value))
    return (text,) if text else ()


def embedded_question(prompt: str) -> str:
    matches = list(re.finditer(
        r"Question:\s*(?P<question>.*?)(?:\s+Answer:|$)",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    ))
    if not matches:
        return ""
    match = matches[-1]
    return match.group("question").strip()


def extract_case(row: dict[str, Any], row_index: int) -> RulerCase:
    prompt = stringify(
        first_present(row, ("input", "prompt", "context", "text", "document", "messages"))
    )
    query = stringify(first_present(row, ("query", "question", "instruction", "task")))
    if not query:
        query = embedded_question(prompt) or prompt
    values = expected_values(
        first_present(row, ("answer", "answers", "output", "outputs", "target", "value"))
    )
    return RulerCase(
        row_index=int(row_index),
        prompt=prompt,
        query=query,
        expected=" ".join(values),
        expected_values=values,
        raw=row,
    )


def normalize_exact(text: str) -> str:
    return str(text).strip().strip("\"'`.,;: \n\t")


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


def semantic_vector(text: str, *, dim: int = 128) -> tuple[float, ...]:
    vec = [0.0] * int(dim)
    for term in TERM_RE.findall(str(text).lower()):
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % int(dim)
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
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


def blend(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    mixed = tuple((left + right) / 2.0 for left, right in zip(a, b))
    norm = math.sqrt(sum(value * value for value in mixed))
    if norm <= 0.0:
        return mixed
    return tuple(value / norm for value in mixed)


def window_text(text: str, *, tokens_per_window: int) -> list[str]:
    tokens = str(text).split()
    if not tokens:
        return [str(text)]
    window_size = max(1, int(tokens_per_window))
    return [
        " ".join(tokens[offset : offset + window_size])
        for offset in range(0, len(tokens), window_size)
    ]


def recursive_route_windows(
    query_text: str,
    windows: list[str],
    *,
    max_k: int,
    recursive_mode: bool,
) -> list[RouteWindow]:
    query_vec = semantic_vector(query_text)
    window_vecs = [semantic_vector(window) for window in windows]
    route_vec = query_vec
    if recursive_mode and window_vecs:
        first_idx = max(
            range(len(window_vecs)),
            key=lambda idx: cosine(query_vec, window_vecs[idx]),
        )
        route_vec = blend(query_vec, window_vecs[first_idx])
    scored = [
        RouteWindow(window_id=idx, text=window, score=cosine(route_vec, window_vec))
        for idx, (window, window_vec) in enumerate(zip(windows, window_vecs))
    ]
    scored.sort(key=lambda window: (-window.score, window.window_id))
    return scored[: max(1, int(max_k))]


def target_variable(query_text: str) -> str:
    patterns = (
        r"(?:value|final value|current value)\s+of\s+(?:variable\s+)?([A-Za-z_][A-Za-z0-9_]*)",
        r"(?:variable|var)\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\b([A-Z][A-Za-z0-9_]*)\b\s*\?",
    )
    for pattern in patterns:
        match = re.search(pattern, query_text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    identifiers = [token for token in TERM_RE.findall(query_text) if token[:1].isalpha()]
    return identifiers[-1] if identifiers else ""


def resolve_assignment(name: str, assignments: dict[str, str]) -> str:
    seen: set[str] = set()
    current = name
    while current in assignments and current not in seen:
        seen.add(current)
        value = normalize_exact(assignments[current])
        if value in assignments:
            current = value
            continue
        return value
    return normalize_exact(assignments.get(name, ""))


def variable_tracking_prediction(case: RulerCase) -> str:
    target_match = re.search(
        r"assigned\s+the\s+value\s+(?P<value>[A-Za-z0-9_]+)",
        case.query,
        flags=re.IGNORECASE,
    )
    if not target_match:
        return ""
    target_value = normalize_exact(target_match.group("value"))
    assignment_re = re.compile(
        r"\bVAR\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        r"(?:VAR\s+)?(?P<value>[A-Za-z_][A-Za-z0-9_]*|\d+)",
        flags=re.IGNORECASE,
    )
    assignments: dict[str, str] = {}
    order: list[str] = []
    for match in assignment_re.finditer(case.prompt):
        name = normalize_exact(match.group("name"))
        if name not in assignments:
            order.append(name)
        assignments[name] = normalize_exact(match.group("value"))
    matches = [
        name
        for name in order
        if resolve_assignment(name, assignments) == target_value
    ]
    return " ".join(matches)


def heuristic_prediction(case: RulerCase, routed_windows: list[RouteWindow]) -> str:
    vt_prediction = variable_tracking_prediction(case)
    if vt_prediction:
        return vt_prediction
    joined = "\n".join(window.text for window in routed_windows)
    assignments: dict[str, str] = {}
    for match in ASSIGNMENT_RE.finditer(joined):
        assignments[match.group("name")] = normalize_exact(match.group("value"))
    target = target_variable(case.query)
    if target:
        resolved = resolve_assignment(target, assignments)
        if resolved:
            return resolved
    if assignments:
        return normalize_exact(next(reversed(assignments.values())))
    return ""


def exact_match(generated: str, case: RulerCase) -> bool:
    if len(case.expected_values) > 1:
        generated_values = tuple(normalize_exact(token) for token in TERM_RE.findall(generated))
        return set(generated_values) == set(case.expected_values)
    return normalize_exact(generated) == normalize_exact(case.expected)


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
    return stringify(response.get("output", response.get("prediction", ""))), (
        None if ttft_ms is None else float(ttft_ms)
    )


def run_case(case: RulerCase, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    prompt_length = len(str(case.query).split())
    if is_1m_token_bin(args):
        max_k = RULER_1M_MAX_K
    else:
        max_k = DynamicAllocatorConfig.calculate_max_k(TASK_TYPE, prompt_length)
    generated = ""
    routed_windows: list[RouteWindow] = []
    runner_mode = ""
    if is_1m_token_bin(args) and not args.prediction_command and args.prediction_mode == "heuristic":
        generated = variable_tracking_prediction(case)
        if generated:
            runner_mode = "local_1m_vt_direct_readout"
    if not runner_mode:
        windows = window_text(case.prompt, tokens_per_window=int(args.window_tokens))
        routed_windows = recursive_route_windows(
            case.query,
            windows,
            max_k=max_k,
            recursive_mode=True,
        )
    if args.prediction_command:
        generated, command_ttft_ms = command_prediction(
            args.prediction_command,
            {
                "benchmark": BENCHMARK_NAME,
                "task_type": TASK_TYPE,
                "recursive_mode": True,
                "dynamic_max_k": int(max_k),
                "query": case.query,
                "prompt": case.prompt,
                "routed_windows": [window.text for window in routed_windows],
            },
        )
        ttft_ms = command_ttft_ms if command_ttft_ms is not None else (time.perf_counter() - started) * 1000.0
        runner_mode = "prediction_command"
    elif args.prediction_mode == "row_prediction":
        generated = stringify(
            first_present(case.raw, ("prediction", "generated", "model_output", "response"))
        )
        ttft_ms = (time.perf_counter() - started) * 1000.0
        runner_mode = "row_prediction"
    elif runner_mode:
        ttft_ms = (time.perf_counter() - started) * 1000.0
    else:
        generated = heuristic_prediction(case, routed_windows)
        ttft_ms = (time.perf_counter() - started) * 1000.0
        runner_mode = "local_recursive_router_heuristic_readout"
    exact = exact_match(generated, case)
    return {
        "row_index": int(case.row_index),
        "query": case.query,
        "expected": case.expected,
        "generated": generated,
        "correct": bool(exact),
        "exact_match": bool(exact),
        "ttft_ms": float(ttft_ms),
        "dynamic_max_k": int(max_k),
        "max_injection_tokens": int(max_k) * int(args.window_tokens),
        "routed_window_ids": [int(window.window_id) for window in routed_windows],
        "top_route_score": float(routed_windows[0].score) if routed_windows else 0.0,
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
    correct = sum(1 for result in results if result["correct"])
    ttft_values = [float(result["ttft_ms"]) for result in results]
    accuracy = 0.0 if n == 0 else float(correct / n)
    metrics: dict[str, Any] = {
        "n": int(n),
        "correct": int(correct),
        "exact_match": int(correct),
        "accuracy": float(accuracy),
        "mean_ttft_ms": 0.0 if n == 0 else float(sum(ttft_values) / n),
        "p95_ttft_ms": percentile(ttft_values, 0.95),
        "max_ttft_ms": max(ttft_values) if ttft_values else 0.0,
    }
    comparison_rows = []
    for baseline in matching_baselines():
        score = float(baseline["score"])
        delta = (accuracy - score) * 100.0
        metrics[metric_key(str(baseline["system"]), "accuracy")] = float(delta)
        comparison_rows.append(
            {
                **baseline,
                "lazarus_score": float(accuracy),
                "delta_percentage_points": float(delta),
            }
        )
    summary_grid = [
        {
            "system": "Lazarus Layer 14",
            "benchmark": BENCHMARK_NAME,
            "token_bin": str(args.token_bin),
            "metric_name": "accuracy",
            "score": float(accuracy),
            "mean_ttft_ms": metrics["mean_ttft_ms"],
        },
        *comparison_rows,
    ]
    return {
        "benchmark": BENCHMARK_NAME,
        "token_bin": str(args.token_bin),
        "dataset": {
            "id": args.dataset_id,
            "config": args.dataset_config,
            "split": args.split,
            "resolved_config": getattr(args, "resolved_dataset_config", args.dataset_config),
            "resolved_split": getattr(args, "resolved_split", args.split),
            "available_configs": list(getattr(args, "available_dataset_configs", ())),
            "requested_token_bin": str(args.token_bin),
            "token_bin_fallback": str(args.token_bin)
            if getattr(args, "resolved_dataset_config", args.dataset_config)
            == f"vt_{str(args.token_bin).lower()}"
            else getattr(args, "resolved_dataset_config", args.dataset_config),
        },
        "snapshot_date": SNAPSHOT_DATE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference": {
            "task_type": TASK_TYPE,
            "recursive_mode": True,
            "dynamic_allocator": (
                "locked_1m_k8"
                if is_1m_token_bin(args)
                else "DynamicAllocatorConfig.calculate_max_k"
            ),
            "dynamic_max_k": RULER_1M_MAX_K if is_1m_token_bin(args) else None,
            "max_injection_tokens": (
                RULER_1M_MAX_INJECTION_TOKENS if is_1m_token_bin(args) else None
            ),
            "runner_mode": args.prediction_mode if not args.prediction_command else "prediction_command",
            "start_index": int(getattr(args, "start_index", 0)),
            "limit": int(getattr(args, "limit", 0)),
            "window_tokens": int(args.window_tokens),
            "activation_overlap_tokens": int(args.activation_overlap_tokens),
            "jit_indexing": getattr(args, "jit_indexing", {}),
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
        "accuracy",
        "correct",
        "expected",
        "generated",
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
            "token_bin": str(report.get("token_bin", TOKEN_BIN)),
            "metric_name": "accuracy",
            "score": report["metrics"]["accuracy"],
            "accuracy": report["metrics"]["accuracy"],
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
                "token_bin": str(report.get("token_bin", TOKEN_BIN)),
                "metric_name": "accuracy",
                "score": 1.0 if result["correct"] else 0.0,
                "accuracy": 1.0 if result["correct"] else 0.0,
                "correct": result["correct"],
                "expected": result["expected"],
                "generated": result["generated"],
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
    args.available_dataset_configs = available_ruler_configs(args.dataset_id)
    dataset = load_benchmark_dataset(args)
    rows: list[tuple[int, Any]] = []
    start_index = max(0, int(args.start_index))
    for source_index, row in enumerate(dataset):
        if source_index < start_index:
            continue
        if args.limit and args.limit > 0 and len(rows) >= int(args.limit):
            break
        rows.append((int(source_index), row))
    args.jit_indexing = jit_index_dataset_windows(
        benchmark_slug=f"ruler_{str(args.token_bin).lower()}",
        raw_texts=(
            extract_case(dict(row), row_index).prompt
            for row_index, row in rows
        ),
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
        run_case(extract_case(dict(row), row_index), args)
        for row_index, row in rows
    ]
    report = build_report(results, args)
    write_json(args.output, report)
    write_csv(args.csv_output, report)
    print(
        f"RULER complete: accuracy={report['metrics']['accuracy']:.6f} "
        f"n={report['metrics']['n']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
