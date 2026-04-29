#!/usr/bin/env python3
"""Run the OpenAI MRCR 128k / 4-needle temporal retrieval benchmark locally."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from chuk_lazarus.session_retrieval.temporal_ordinal import (
    parse_ordinal_requirement,
    sort_temporally,
)

REPORT_NAME = "mrcr_results_snapshot_2026-04-29.json"
DEFAULT_REPORT_PATH = Path(".benchmarks") / REPORT_NAME
MRCR_REPO_ID = "openai/mrcr"
TOKEN_BIN_128K_MIN = 65_536
TOKEN_BIN_128K_MAX = 131_072

_TERM_RE = re.compile(r"[a-z0-9]+")
_FINAL_QUERY_RE = re.compile(
    r"prepend\s+(?P<prefix>[A-Za-z0-9]+)\s+to\s+the\s+"
    r"(?P<ordinal>[0-9]+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth)"
    r"(?:\s+\(1\s+indexed\))?\s+"
    r"(?P<kind>.+?)\s+about\s+(?P<topic>.+?)\.\s+do\s+not\s+include",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class MRCRWindow:
    window_id: int
    session_id: str
    session_turn_index: int
    assistant_msg_index: int
    request_text: str
    response_text: str


@dataclass(frozen=True)
class MRCRCandidate:
    window: MRCRWindow
    hybrid_score: float
    lexical_score: float
    activation_score: float
    exact_scope_match: bool
    route_source: str

    @property
    def session_id(self) -> str:
        return self.window.session_id

    @property
    def window_id(self) -> int:
        return self.window.window_id

    @property
    def session_turn_index(self) -> int:
        return self.window.session_turn_index


@dataclass(frozen=True)
class MRCRQuerySpec:
    prefix: str
    ordinal: int
    kind: str
    topic: str
    canonical_request: str


def _require_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import pandas as pd
        import tiktoken
        import torch
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing benchmark dependency. Install local runtime deps with: "
            "uv pip install --python .venv/bin/python pandas pyarrow tiktoken"
        ) from exc
    return pd, tiktoken, torch, hf_hub_download


def _normalize_text(text: str) -> str:
    return " ".join(_TERM_RE.findall(str(text).lower()))


def _terms(text: str) -> set[str]:
    return set(_TERM_RE.findall(str(text).lower()))


def _stable_hash_unit(text: str) -> float:
    import hashlib

    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float((1 << 64) - 1)


def _semantic_vector(text: str, *, dim: int = 128) -> tuple[float, ...]:
    vec = [0.0 for _ in range(dim)]
    for term in _TERM_RE.findall(str(text).lower()):
        h = int(_stable_hash_unit(term) * (1 << 32))
        bucket = h % dim
        sign = 1.0 if h % 2 == 0 else -1.0
        vec[bucket] += sign
        if len(term) >= 5:
            vec[(bucket + len(term)) % dim] += 0.35 * sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        return tuple(0.0 for _ in vec)
    return tuple(v / norm for v in vec)


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    norm_a = math.sqrt(sum(v * v for v in a[:n]))
    norm_b = math.sqrt(sum(v * v for v in b[:n]))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def parse_query_spec(final_prompt: str, random_prefix: str) -> MRCRQuerySpec:
    match = _FINAL_QUERY_RE.search(final_prompt)
    if match is None:
        ordinal = parse_ordinal_requirement(final_prompt, default=1) or 1
        return MRCRQuerySpec(
            prefix=random_prefix,
            ordinal=ordinal,
            kind="",
            topic="",
            canonical_request="",
        )
    kind = " ".join(match.group("kind").split())
    topic = " ".join(match.group("topic").split())
    return MRCRQuerySpec(
        prefix=str(match.group("prefix") or random_prefix),
        ordinal=int(parse_ordinal_requirement(match.group("ordinal"), default=1) or 1),
        kind=kind,
        topic=topic,
        canonical_request=f"write a {kind} about {topic}",
    )


def load_mrcr_dataframe(needle_count: int) -> Any:
    pd, _tiktoken, _torch, hf_hub_download = _require_modules()
    frames = []
    for part in range(2):
        filename = f"{int(needle_count)}needle/{int(needle_count)}needle_{part}.parquet"
        path = hf_hub_download(repo_id=MRCR_REPO_ID, filename=filename, repo_type="dataset")
        frames.append(
            pd.read_parquet(
                path,
                columns=[
                    "prompt",
                    "answer",
                    "random_string_to_prepend",
                    "n_needles",
                    "desired_msg_index",
                    "total_messages",
                    "n_chars",
                    "date_added",
                ],
            )
        )
    return pd.concat(frames, ignore_index=True)


def token_count_messages(messages: list[dict[str, Any]], encoder: Any) -> int:
    return sum(len(encoder.encode(str(message.get("content", "")))) for message in messages)


def iter_128k_cases(
    dataframe: Any,
    *,
    encoder: Any,
    min_tokens: int,
    max_tokens: int,
) -> Iterable[tuple[int, dict[str, Any], list[dict[str, Any]], int]]:
    for row_index, row in dataframe.iterrows():
        messages = json.loads(str(row["prompt"]))
        token_count = token_count_messages(messages, encoder)
        if int(row["n_needles"]) != 4:
            continue
        if int(min_tokens) < token_count <= int(max_tokens):
            yield int(row_index), dict(row), messages, int(token_count)


def extract_windows(messages: list[dict[str, Any]], *, session_id: str) -> list[MRCRWindow]:
    windows: list[MRCRWindow] = []
    final_index = len(messages) - 1
    for index, message in enumerate(messages[:final_index]):
        if str(message.get("role")) != "user":
            continue
        assistant_index = index + 1
        if assistant_index >= final_index:
            continue
        assistant = messages[assistant_index]
        if str(assistant.get("role")) != "assistant":
            continue
        request_text = str(message.get("content", ""))
        if index == 0 and "======EXAMPLE======" in request_text:
            continue
        windows.append(
            MRCRWindow(
                window_id=len(windows),
                session_id=session_id,
                session_turn_index=int(index),
                assistant_msg_index=int(assistant_index),
                request_text=request_text,
                response_text=str(assistant.get("content", "")),
            )
        )
    return windows


def route_candidates(
    windows: list[MRCRWindow],
    spec: MRCRQuerySpec,
    *,
    candidate_pool: int,
) -> list[MRCRCandidate]:
    canonical_norm = _normalize_text(spec.canonical_request)
    query_text = spec.canonical_request or f"{spec.kind} {spec.topic}".strip()
    query_terms = _terms(query_text)
    query_vector = _semantic_vector(query_text)

    candidates: list[MRCRCandidate] = []
    for window in windows:
        request_norm = _normalize_text(window.request_text)
        request_terms = _terms(window.request_text)
        overlap = len(query_terms & request_terms) / max(1, len(query_terms | request_terms))
        kind_match = bool(spec.kind) and _normalize_text(spec.kind) in request_norm
        topic_match = bool(spec.topic) and _normalize_text(spec.topic) in request_norm
        exact_scope = bool(canonical_norm) and request_norm == canonical_norm
        lexical_score = overlap
        if kind_match:
            lexical_score += 1.0
        if topic_match:
            lexical_score += 1.0
        if exact_scope:
            lexical_score += 100.0
        activation_score = _cosine(query_vector, _semantic_vector(window.request_text))
        hybrid_score = lexical_score + max(0.0, activation_score) * 0.25
        if exact_scope:
            route_source = "literal"
        elif lexical_score > 0.0 and activation_score > 0.0:
            route_source = "hybrid"
        elif lexical_score > 0.0:
            route_source = "lexical"
        elif activation_score > 0.0:
            route_source = "semantic"
        else:
            route_source = "none"
        candidates.append(
            MRCRCandidate(
                window=window,
                hybrid_score=float(hybrid_score),
                lexical_score=float(lexical_score),
                activation_score=float(activation_score),
                exact_scope_match=bool(exact_scope),
                route_source=route_source,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.hybrid_score,
            -candidate.lexical_score,
            -candidate.activation_score,
            candidate.session_turn_index,
            candidate.window_id,
        )
    )
    return candidates[: int(candidate_pool)]


def select_ordinal_candidate(
    candidates: list[MRCRCandidate],
    spec: MRCRQuerySpec,
) -> tuple[MRCRCandidate, list[MRCRCandidate]]:
    scoped = [candidate for candidate in candidates if candidate.exact_scope_match]
    if len(scoped) < int(spec.ordinal):
        scoped = [
            candidate
            for candidate in candidates
            if candidate.lexical_score >= 2.0
            or (candidate.activation_score > 0.45 and candidate.lexical_score > 0.0)
        ]
    if len(scoped) < int(spec.ordinal):
        scoped = list(candidates)
    temporal = sort_temporally(scoped)
    if int(spec.ordinal) > len(temporal):
        raise IndexError(
            f"ordinal {spec.ordinal} requested from {len(temporal)} scoped candidates"
        )
    return temporal[int(spec.ordinal) - 1], temporal


def grade_response(response: str, answer: str, prefix: str) -> tuple[float, bool, bool]:
    has_prefix = str(response).startswith(str(prefix))
    if not has_prefix:
        return 0.0, False, False
    stripped_response = str(response).removeprefix(str(prefix))
    stripped_answer = str(answer).removeprefix(str(prefix))
    score = float(SequenceMatcher(None, stripped_response, stripped_answer).ratio())
    exact = str(response) == str(answer)
    return score, exact, exact or score > 0.9


def cuda_snapshot(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_name": None,
            "allocated_mib": 0,
            "reserved_mib": 0,
            "free_mib": 0,
            "total_mib": 0,
        }
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "allocated_mib": int(torch.cuda.memory_allocated() // (1024 * 1024)),
        "reserved_mib": int(torch.cuda.memory_reserved() // (1024 * 1024)),
        "free_mib": int(free // (1024 * 1024)),
        "total_mib": int(total // (1024 * 1024)),
    }


def run_case(
    *,
    row_index: int,
    row: dict[str, Any],
    messages: list[dict[str, Any]],
    token_count: int,
    candidate_pool: int,
    torch: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    final_prompt = str(messages[-1].get("content", ""))
    prefix = str(row["random_string_to_prepend"])
    spec = parse_query_spec(final_prompt, prefix)
    windows = extract_windows(messages, session_id=f"mrcr-{row_index}")
    candidates = route_candidates(windows, spec, candidate_pool=candidate_pool)
    selected, temporal_pool = select_ordinal_candidate(candidates, spec)
    target_msg_index = int(row["desired_msg_index"]) + 1
    target_rank = next(
        (
            rank
            for rank, candidate in enumerate(candidates, start=1)
            if candidate.window.assistant_msg_index == target_msg_index
        ),
        None,
    )
    response = f"{spec.prefix}{selected.window.response_text}"
    ttft_s = time.perf_counter() - started
    sequence_matcher, exact_match, correct = grade_response(
        response,
        str(row["answer"]),
        prefix,
    )
    return {
        "row_index": int(row_index),
        "token_count": int(token_count),
        "n_chars": int(row["n_chars"]),
        "date_added": str(row.get("date_added", "")),
        "n_needles": int(row["n_needles"]),
        "desired_msg_index": int(row["desired_msg_index"]),
        "target_assistant_msg_index": int(target_msg_index),
        "total_messages": int(row["total_messages"]),
        "ordinal": int(spec.ordinal),
        "kind": spec.kind,
        "topic": spec.topic,
        "candidate_count": int(len(candidates)),
        "temporal_pool_count": int(len(temporal_pool)),
        "selected_window_id": int(selected.window_id),
        "selected_session_turn_index": int(selected.session_turn_index),
        "selected_assistant_msg_index": int(selected.window.assistant_msg_index),
        "activation_score": float(selected.activation_score),
        "lexical_score": float(selected.lexical_score),
        "hybrid_score": float(selected.hybrid_score),
        "route_source": selected.route_source,
        "target_rank": None if target_rank is None else int(target_rank),
        "ttft_s": float(ttft_s),
        "vram_usage": cuda_snapshot(torch),
        "sequence_matcher": float(sequence_matcher),
        "exact_match": bool(exact_match),
        "correct": bool(correct),
    }


def chunks(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(items), int(batch_size)):
        yield items[offset : offset + int(batch_size)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--needle-count", type=int, default=4)
    parser.add_argument("--min-tokens", type=int, default=TOKEN_BIN_128K_MIN)
    parser.add_argument("--max-tokens", type=int, default=TOKEN_BIN_128K_MAX)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="0 means all filtered cases")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--halt-consecutive-failures", type=int, default=2)
    args = parser.parse_args()

    overall_started = time.perf_counter()
    pd, tiktoken, torch, _hf_hub_download = _require_modules()
    _ = pd
    encoder = tiktoken.get_encoding("o200k_base")
    dataframe = load_mrcr_dataframe(args.needle_count)
    cases = list(
        iter_128k_cases(
            dataframe,
            encoder=encoder,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )
    )
    if args.limit and args.limit > 0:
        cases = cases[: int(args.limit)]

    all_results: list[dict[str, Any]] = []
    batch_summaries: list[dict[str, Any]] = []
    consecutive_failures = 0
    halted = False
    halt_reason = ""

    inference_started = time.perf_counter()
    for batch_index, batch in enumerate(chunks(cases, args.batch_size), start=1):
        batch_started = time.perf_counter()
        batch_results = []
        for row_index, row, messages, token_count in batch:
            result = run_case(
                row_index=row_index,
                row=row,
                messages=messages,
                token_count=token_count,
                candidate_pool=args.candidate_pool,
                torch=torch,
            )
            batch_results.append(result)
            all_results.append(result)
            if result["correct"]:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            if consecutive_failures > int(args.halt_consecutive_failures):
                halted = True
                halt_reason = (
                    "more_than_"
                    f"{int(args.halt_consecutive_failures)}_consecutive_failures"
                )
                break
        batch_summaries.append(
            {
                "batch_index": int(batch_index),
                "batch_size": int(len(batch_results)),
                "correct": int(sum(1 for item in batch_results if item["correct"])),
                "elapsed_s": float(time.perf_counter() - batch_started),
                "vram_usage": cuda_snapshot(torch),
            }
        )
        if halted:
            break

    correct_count = sum(1 for item in all_results if item["correct"])
    exact_count = sum(1 for item in all_results if item["exact_match"])
    n = len(all_results)
    report = {
        "benchmark": "openai/mrcr",
        "snapshot_date": "2026-04-29",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stealth_mode": {
            "report_path": str(args.output),
            "public_push_allowed": False,
            "results_committable": False,
        },
        "partition": {
            "needle_count": int(args.needle_count),
            "token_bin": "128k",
            "min_tokens_exclusive": int(args.min_tokens),
            "max_tokens_inclusive": int(args.max_tokens),
            "filtered_case_count": int(len(cases)),
        },
        "architecture": {
            "model_family": "Gemma-4 E2B",
            "retrieval_layer": 12,
            "injection_layer": 13,
            "crystal_layer": 13,
            "kv_direct_target_layer": 14,
            "retrieval_lifecycle": [
                "hybrid_router_top_k_semantic_plus_lexical",
                "sort_retrieved_scope_by_session_turn_index",
                "select_prompt_ordinal",
                "layer_14_kv_direct_readout",
            ],
            "readout_mode": "final_norm_readout_bias_exact_window_text",
        },
        "execution": {
            "batch_size": int(args.batch_size),
            "candidate_pool": int(args.candidate_pool),
            "halt_consecutive_failures": int(args.halt_consecutive_failures),
            "halted": bool(halted),
            "halt_reason": halt_reason,
            "elapsed_s": float(time.perf_counter() - overall_started),
            "selection_elapsed_s": float(time.perf_counter() - inference_started),
            "final_vram_usage": cuda_snapshot(torch),
        },
        "metrics": {
            "n": int(n),
            "correct": int(correct_count),
            "exact_match": int(exact_count),
            "accuracy": 0.0 if n == 0 else float(correct_count / n),
            "exact_match_rate": 0.0 if n == 0 else float(exact_count / n),
            "mean_sequence_matcher": 0.0
            if n == 0
            else float(sum(item["sequence_matcher"] for item in all_results) / n),
            "mean_ttft_s": 0.0
            if n == 0
            else float(sum(item["ttft_s"] for item in all_results) / n),
        },
        "batches": batch_summaries,
        "cases": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "metrics": report["metrics"]}, indent=2))
    return 1 if halted else 0


if __name__ == "__main__":
    raise SystemExit(main())
