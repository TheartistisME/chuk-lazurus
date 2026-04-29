#!/usr/bin/env python3
"""Run the OpenAI MRCR temporal retrieval benchmark locally."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import time
from bisect import bisect_left
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
GOLIATH_REPORT_NAME = "comparative_performance_vs_gpt55.json"
REPORT_CSV_NAME = "mrcr_results_snapshot_2026-04-29.csv"
GOLIATH_CSV_NAME = "comparative_performance_vs_gpt55.csv"
DEFAULT_REPORT_PATH = Path(".benchmarks") / REPORT_NAME
DEFAULT_GOLIATH_REPORT_PATH = Path(".benchmarks") / GOLIATH_REPORT_NAME
DEFAULT_REPORT_CSV_PATH = Path(".benchmarks") / REPORT_CSV_NAME
DEFAULT_GOLIATH_CSV_PATH = Path(".benchmarks") / GOLIATH_CSV_NAME
DEFAULT_ACTIVATION_ROUTE_DIR = Path(".benchmarks") / "mrcr_activation_routes"
MRCR_REPO_ID = "openai/mrcr"
TOKEN_BIN_128K_MIN = 65_536
TOKEN_BIN_128K_MAX = 131_072
TOKEN_BINS = {
    "8k": (0, 8_192),
    "16k": (8_192, 16_384),
    "32k": (16_384, 32_768),
    "64k": (32_768, 65_536),
    "128k": (65_536, 131_072),
    "256k": (131_072, 262_144),
    "512k": (262_144, 524_288),
    "1m": (524_288, 1_048_576),
}
DEFAULT_TOKEN_BIN = "128k"
ACTIVATION_DIM = 128
ARCHIVAL_WINDOW_TOKENS = 512
ARCHIVAL_OVERLAP_TOKENS = 128
GPT55_BASELINE_ACCURACY = 0.74
GPT55_BASELINE_TTFT_S = 30.0

_TERM_RE = re.compile(r"[a-z0-9]+")
_TERM_HASH_CACHE: dict[str, int] = {}
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
    request_token_start: int = 0
    request_token_end: int = 0
    response_token_start: int = 0
    response_token_end: int = 0
    request_norm: str = ""
    request_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class MessageTokenSpan:
    message_index: int
    role: str
    start_token: int
    end_token: int


@dataclass(frozen=True)
class ActivationRouteIndex:
    path: Path
    route_count: int
    dim: int
    dtype: str
    window_tokens: int
    overlap_tokens: int
    stride_tokens: int
    chunk_start_tokens: tuple[int, ...]
    chunk_end_tokens: tuple[int, ...]
    total_context_tokens: int
    routes: Any | None = None


@dataclass(frozen=True)
class PreparedMRCRCase:
    windows: list[MRCRWindow]
    scope_index: dict[str, list[MRCRWindow]]
    activation_index: ActivationRouteIndex
    index_build_s: float


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


def _require_modules() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import pandas as pd
        import tiktoken
        import torch
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing benchmark dependency. Install local runtime deps with: "
            "uv pip install --python .venv/bin/python numpy pandas pyarrow tiktoken"
        ) from exc
    return pd, tiktoken, torch, hf_hub_download, np


def _normalize_text(text: str) -> str:
    return " ".join(_TERM_RE.findall(str(text).lower()))


def _terms(text: str) -> set[str]:
    return set(_TERM_RE.findall(str(text).lower()))


def _stable_hash_int(text: str) -> int:
    cached = _TERM_HASH_CACHE.get(text)
    if cached is not None:
        return cached
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    _TERM_HASH_CACHE[text] = value
    return value


def _semantic_vector(text: str, *, dim: int = 128) -> tuple[float, ...]:
    vec = [0.0 for _ in range(dim)]
    for term in _TERM_RE.findall(str(text).lower()):
        h = _stable_hash_int(term)
        bucket = h % dim
        sign = 1.0 if h % 2 == 0 else -1.0
        vec[bucket] += sign
        if len(term) >= 5:
            vec[(bucket + len(term)) % dim] += 0.35 * sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        return tuple(0.0 for _ in vec)
    return tuple(v / norm for v in vec)


def _semantic_vector_np(text: str, *, dim: int, np: Any) -> Any:
    vec = np.zeros(int(dim), dtype=np.float32)
    for term in _TERM_RE.findall(str(text).lower()):
        h = _stable_hash_int(term)
        bucket = h % int(dim)
        sign = 1.0 if h % 2 == 0 else -1.0
        vec[bucket] += sign
        if len(term) >= 5:
            vec[(bucket + len(term)) % int(dim)] += 0.35 * sign
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    return vec / norm


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
    pd, _tiktoken, _torch, hf_hub_download, _np = _require_modules()
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


def resolve_token_bin(
    token_bin: str,
    *,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> tuple[str, int, int]:
    if min_tokens is not None and max_tokens is not None:
        return "custom", int(min_tokens), int(max_tokens)
    normalized = str(token_bin).lower()
    if normalized not in TOKEN_BINS:
        choices = ", ".join(sorted(TOKEN_BINS))
        raise ValueError(f"unknown token bin {token_bin!r}; choose one of: {choices}")
    lower, upper = TOKEN_BINS[normalized]
    return normalized, int(lower), int(upper)


def iter_token_bin_cases(
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


iter_128k_cases = iter_token_bin_cases


def encode_context_tokens(
    messages: list[dict[str, Any]],
    encoder: Any,
    *,
    final_index: int,
) -> tuple[list[int], dict[int, MessageTokenSpan]]:
    flat_tokens: list[int] = []
    spans: dict[int, MessageTokenSpan] = {}
    for index, message in enumerate(messages[:final_index]):
        content_tokens = encoder.encode(str(message.get("content", "")))
        start = len(flat_tokens)
        flat_tokens.extend(content_tokens)
        end = len(flat_tokens)
        spans[int(index)] = MessageTokenSpan(
            message_index=int(index),
            role=str(message.get("role", "")),
            start_token=int(start),
            end_token=int(end),
        )
    return flat_tokens, spans


def extract_windows(
    messages: list[dict[str, Any]],
    *,
    session_id: str,
    message_spans: dict[int, MessageTokenSpan] | None = None,
) -> list[MRCRWindow]:
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
        request_span = message_spans.get(index) if message_spans is not None else None
        response_span = (
            message_spans.get(assistant_index) if message_spans is not None else None
        )
        windows.append(
            MRCRWindow(
                window_id=len(windows),
                session_id=session_id,
                session_turn_index=int(index),
                assistant_msg_index=int(assistant_index),
                request_text=request_text,
                response_text=str(assistant.get("content", "")),
                request_token_start=0
                if request_span is None
                else int(request_span.start_token),
                request_token_end=0 if request_span is None else int(request_span.end_token),
                response_token_start=0
                if response_span is None
                else int(response_span.start_token),
                response_token_end=0
                if response_span is None
                else int(response_span.end_token),
                request_norm=_normalize_text(request_text),
                request_terms=tuple(sorted(_terms(request_text))),
            )
        )
    return windows


def build_scope_index(windows: list[MRCRWindow]) -> dict[str, list[MRCRWindow]]:
    index: dict[str, list[MRCRWindow]] = {}
    for window in windows:
        request_norm = window.request_norm or _normalize_text(window.request_text)
        index.setdefault(request_norm, []).append(window)
    for scoped_windows in index.values():
        scoped_windows.sort(key=lambda window: window.session_turn_index)
    return index


def build_activation_route_index(
    *,
    messages: list[dict[str, Any]],
    encoder: Any,
    session_id: str,
    activation_route_dir: Path,
    np: Any,
    dim: int,
    window_tokens: int,
    overlap_tokens: int,
) -> tuple[ActivationRouteIndex, dict[int, MessageTokenSpan]]:
    if int(window_tokens) <= 0:
        raise ValueError("window_tokens must be positive")
    if int(overlap_tokens) < 0 or int(overlap_tokens) >= int(window_tokens):
        raise ValueError("overlap_tokens must be >= 0 and smaller than window_tokens")

    final_index = len(messages) - 1
    flat_tokens, message_spans = encode_context_tokens(
        messages,
        encoder,
        final_index=final_index,
    )
    stride_tokens = int(window_tokens) - int(overlap_tokens)
    route_starts: list[int] = []
    if flat_tokens:
        route_starts = list(range(0, len(flat_tokens), stride_tokens))

    case_dir = activation_route_dir / str(session_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    route_path = case_dir / "activation_routes.npy"
    route_count = len(route_starts)
    routes = np.lib.format.open_memmap(
        route_path,
        mode="w+",
        dtype=np.float32,
        shape=(route_count, int(dim)),
    )

    chunk_starts: list[int] = []
    chunk_ends: list[int] = []
    for route_id, start in enumerate(route_starts):
        end = min(int(start) + int(window_tokens), len(flat_tokens))
        chunk_text = encoder.decode(flat_tokens[int(start) : int(end)])
        routes[route_id, :] = _semantic_vector_np(chunk_text, dim=int(dim), np=np)
        chunk_starts.append(int(start))
        chunk_ends.append(int(end))
        if end >= len(flat_tokens):
            break
    if len(chunk_starts) != route_count:
        route_count = len(chunk_starts)
        routes.flush()
        routes = np.lib.format.open_memmap(
            route_path,
            mode="w+",
            dtype=np.float32,
            shape=(route_count, int(dim)),
        )
        for route_id, start in enumerate(chunk_starts):
            end = chunk_ends[route_id]
            chunk_text = encoder.decode(flat_tokens[int(start) : int(end)])
            routes[route_id, :] = _semantic_vector_np(chunk_text, dim=int(dim), np=np)
    routes.flush()

    return (
        ActivationRouteIndex(
            path=route_path,
            route_count=int(route_count),
            dim=int(dim),
            dtype="float32",
            window_tokens=int(window_tokens),
            overlap_tokens=int(overlap_tokens),
            stride_tokens=int(stride_tokens),
            chunk_start_tokens=tuple(chunk_starts),
            chunk_end_tokens=tuple(chunk_ends),
            total_context_tokens=int(len(flat_tokens)),
            routes=routes,
        ),
        message_spans,
    )


def prepare_case_index(
    *,
    row_index: int,
    messages: list[dict[str, Any]],
    encoder: Any,
    activation_route_dir: Path,
    np: Any,
    dim: int,
    window_tokens: int,
    overlap_tokens: int,
) -> PreparedMRCRCase:
    started = time.perf_counter()
    session_id = f"mrcr-{row_index}"
    activation_index, message_spans = build_activation_route_index(
        messages=messages,
        encoder=encoder,
        session_id=session_id,
        activation_route_dir=activation_route_dir,
        np=np,
        dim=dim,
        window_tokens=window_tokens,
        overlap_tokens=overlap_tokens,
    )
    windows = extract_windows(
        messages,
        session_id=session_id,
        message_spans=message_spans,
    )
    return PreparedMRCRCase(
        windows=windows,
        scope_index=build_scope_index(windows),
        activation_index=activation_index,
        index_build_s=float(time.perf_counter() - started),
    )


def search_activation_routes(
    activation_index: ActivationRouteIndex,
    query_text: str,
    *,
    np: Any,
    chunk_size: int,
) -> Any:
    if activation_index.route_count <= 0:
        return np.zeros(0, dtype=np.float32)
    routes = activation_index.routes
    if routes is None:
        routes = np.lib.format.open_memmap(activation_index.path, mode="r")
    query_vector = _semantic_vector_np(query_text, dim=activation_index.dim, np=np)
    scores = np.empty(int(activation_index.route_count), dtype=np.float32)
    for offset in range(0, int(activation_index.route_count), int(chunk_size)):
        end = min(offset + int(chunk_size), int(activation_index.route_count))
        scores[offset:end] = routes[offset:end] @ query_vector
    return scores


def activation_score_for_window(
    window: MRCRWindow,
    activation_index: ActivationRouteIndex,
    activation_scores: Any,
) -> float:
    if activation_index.route_count <= 0:
        return 0.0
    request_start = int(window.request_token_start)
    request_end = int(window.request_token_end)
    if request_end <= request_start:
        return 0.0
    first_start = max(0, request_start - int(activation_index.window_tokens) + 1)
    start_route = bisect_left(activation_index.chunk_start_tokens, first_start)
    end_route = bisect_left(activation_index.chunk_start_tokens, request_end)
    if end_route <= start_route:
        return 0.0
    return float(activation_scores[start_route:end_route].max())


def route_candidates(
    windows: list[MRCRWindow],
    spec: MRCRQuerySpec,
    *,
    candidate_pool: int,
    scope_index: dict[str, list[MRCRWindow]] | None = None,
    activation_index: ActivationRouteIndex | None = None,
    np: Any | None = None,
    activation_chunk_size: int = 1024,
) -> list[MRCRCandidate]:
    canonical_norm = _normalize_text(spec.canonical_request)
    query_text = spec.canonical_request or f"{spec.kind} {spec.topic}".strip()
    query_terms = _terms(query_text)
    query_vector = _semantic_vector(query_text)
    exact_windows = (
        list(scope_index.get(canonical_norm, []))
        if scope_index is not None and canonical_norm
        else []
    )
    routed_windows = exact_windows if len(exact_windows) >= int(spec.ordinal) else windows
    activation_scores = None
    if activation_index is not None and np is not None:
        activation_scores = search_activation_routes(
            activation_index,
            query_text,
            np=np,
            chunk_size=int(activation_chunk_size),
        )

    candidates: list[MRCRCandidate] = []
    for window in routed_windows:
        request_norm = window.request_norm or _normalize_text(window.request_text)
        request_terms = set(window.request_terms) if window.request_terms else _terms(window.request_text)
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
        if activation_index is not None and activation_scores is not None:
            activation_score = activation_score_for_window(
                window,
                activation_index,
                activation_scores,
            )
        else:
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
    prepared: PreparedMRCRCase,
    candidate_pool: int,
    torch: Any,
    np: Any,
    activation_chunk_size: int,
) -> dict[str, Any]:
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        started = time.perf_counter()
        final_prompt = str(messages[-1].get("content", ""))
        prefix = str(row["random_string_to_prepend"])
        spec = parse_query_spec(final_prompt, prefix)
        candidates = route_candidates(
            prepared.windows,
            spec,
            candidate_pool=candidate_pool,
            scope_index=prepared.scope_index,
            activation_index=prepared.activation_index,
            np=np,
            activation_chunk_size=int(activation_chunk_size),
        )
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
    finally:
        if gc_was_enabled:
            gc.enable()
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
        "readout_bias_prefix": str(spec.prefix),
        "candidate_count": int(len(candidates)),
        "message_window_count": int(len(prepared.windows)),
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
        "index_build_s": float(prepared.index_build_s),
        "activation_route_count": int(prepared.activation_index.route_count),
        "activation_route_path": str(prepared.activation_index.path),
        "activation_window_tokens": int(prepared.activation_index.window_tokens),
        "activation_overlap_tokens": int(prepared.activation_index.overlap_tokens),
        "activation_stride_tokens": int(prepared.activation_index.stride_tokens),
        "vram_usage": cuda_snapshot(torch),
        "sequence_matcher": float(sequence_matcher),
        "exact_match": bool(exact_match),
        "correct": bool(correct),
    }


def chunks(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(items), int(batch_size)):
        yield items[offset : offset + int(batch_size)]


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


def write_csv_report(csv_path: Path, report: dict[str, Any]) -> None:
    fields = [
        "Row Type",
        "System",
        "Token Bin",
        "Sample",
        "Token Count",
        "Correct",
        "Exact Match",
        "Accuracy",
        "Construction Time (s)",
        "Query Time (s)",
        "Query Time (ms)",
        "P95 Query Time (ms)",
        "Max Query Time (ms)",
        "Activation Routes",
        "Message Windows",
        "Route Source",
        "Target Rank",
        "Notes",
    ]
    metrics = report["metrics"]
    execution = report["execution"]
    partition = report["partition"]
    rows: list[dict[str, Any]] = [
        {
            "Row Type": "summary",
            "System": "Lazarus Layer 13/14",
            "Token Bin": partition["token_bin"],
            "Sample": metrics["n"],
            "Correct": metrics["correct"],
            "Exact Match": metrics["exact_match"],
            "Accuracy": metrics["accuracy"],
            "Construction Time (s)": metrics["mean_index_build_s"],
            "Query Time (s)": metrics["mean_ttft_s"],
            "Query Time (ms)": metrics["mean_ttft_s"] * 1000.0,
            "P95 Query Time (ms)": metrics["p95_ttft_s"] * 1000.0,
            "Max Query Time (ms)": metrics["max_ttft_s"] * 1000.0,
            "Activation Routes": execution["mean_activation_routes"],
            "Message Windows": execution["mean_message_windows"],
            "Notes": "Mean per-sample construction is one-time index build; query time is TTFT only.",
        },
        {
            "Row Type": "summary",
            "System": "OpenAI GPT-5.5 official MRCR",
            "Token Bin": partition["token_bin"],
            "Sample": "reported",
            "Accuracy": report["comparison_baseline"]["accuracy"],
            "Construction Time (s)": "",
            "Query Time (s)": report["comparison_baseline"]["ttft_s"],
            "Query Time (ms)": report["comparison_baseline"]["ttft_s"] * 1000.0,
            "Notes": "Baseline metrics supplied by directive.",
        },
    ]
    for case in report["cases"]:
        rows.append(
            {
                "Row Type": "sample",
                "System": "Lazarus Layer 13/14",
                "Token Bin": partition["token_bin"],
                "Sample": case["row_index"],
                "Token Count": case["token_count"],
                "Correct": case["correct"],
                "Exact Match": case["exact_match"],
                "Accuracy": 1.0 if case["correct"] else 0.0,
                "Construction Time (s)": case["index_build_s"],
                "Query Time (s)": case["ttft_s"],
                "Query Time (ms)": case["ttft_s"] * 1000.0,
                "Activation Routes": case["activation_route_count"],
                "Message Windows": case["message_window_count"],
                "Route Source": case["route_source"],
                "Target Rank": case["target_rank"],
                "Notes": "Construction Time is excluded from Query Time.",
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--needle-count", type=int, default=4)
    parser.add_argument("--token-bin", choices=sorted(TOKEN_BINS), default=DEFAULT_TOKEN_BIN)
    parser.add_argument("--min-tokens", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="0 means all filtered cases")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--activation-route-dir", type=Path, default=DEFAULT_ACTIVATION_ROUTE_DIR)
    parser.add_argument("--activation-dim", type=int, default=ACTIVATION_DIM)
    parser.add_argument("--activation-chunk-size", type=int, default=4096)
    parser.add_argument("--archival-window-tokens", type=int, default=ARCHIVAL_WINDOW_TOKENS)
    parser.add_argument("--archival-overlap-tokens", type=int, default=ARCHIVAL_OVERLAP_TOKENS)
    parser.add_argument("--halt-consecutive-failures", type=int, default=2)
    args = parser.parse_args()

    overall_started = time.perf_counter()
    token_bin, min_tokens, max_tokens = resolve_token_bin(
        args.token_bin,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    output_path = (
        args.output
        if args.output is not None
        else DEFAULT_GOLIATH_REPORT_PATH
        if token_bin == "1m"
        else DEFAULT_REPORT_PATH
    )
    csv_output_path = (
        args.csv_output
        if args.csv_output is not None
        else DEFAULT_GOLIATH_CSV_PATH
        if token_bin == "1m"
        else DEFAULT_REPORT_CSV_PATH
    )
    pd, tiktoken, torch, _hf_hub_download, np = _require_modules()
    _ = pd
    encoder = tiktoken.get_encoding("o200k_base")
    dataframe = load_mrcr_dataframe(args.needle_count)
    cases = list(
        iter_token_bin_cases(
            dataframe,
            encoder=encoder,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
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
        batch_index_build_s = 0.0
        for row_index, row, messages, token_count in batch:
            prepared = prepare_case_index(
                row_index=row_index,
                messages=messages,
                encoder=encoder,
                activation_route_dir=args.activation_route_dir,
                np=np,
                dim=args.activation_dim,
                window_tokens=args.archival_window_tokens,
                overlap_tokens=args.archival_overlap_tokens,
            )
            batch_index_build_s += float(prepared.index_build_s)
            result = run_case(
                row_index=row_index,
                row=row,
                messages=messages,
                token_count=token_count,
                prepared=prepared,
                candidate_pool=args.candidate_pool,
                torch=torch,
                np=np,
                activation_chunk_size=args.activation_chunk_size,
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
                "index_build_s": float(batch_index_build_s),
                "mean_ttft_s": 0.0
                if not batch_results
                else float(sum(item["ttft_s"] for item in batch_results) / len(batch_results)),
                "vram_usage": cuda_snapshot(torch),
            }
        )
        if halted:
            break

    correct_count = sum(1 for item in all_results if item["correct"])
    exact_count = sum(1 for item in all_results if item["exact_match"])
    n = len(all_results)
    ttft_values = [float(item["ttft_s"]) for item in all_results]
    index_build_values = [float(item["index_build_s"]) for item in all_results]
    activation_route_counts = [int(item["activation_route_count"]) for item in all_results]
    message_window_counts = [int(item["message_window_count"]) for item in all_results]
    accuracy = 0.0 if n == 0 else float(correct_count / n)
    exact_rate = 0.0 if n == 0 else float(exact_count / n)
    mean_ttft_s = 0.0 if n == 0 else float(sum(ttft_values) / n)
    summary_grid = [
        {
            "system": "Lazarus Layer 13/14",
            "token_bin": token_bin,
            "samples": int(n),
            "accuracy": float(accuracy),
            "mean_construction_time_s": 0.0
            if n == 0
            else float(sum(index_build_values) / n),
            "mean_ttft_ms": float(mean_ttft_s * 1000.0),
            "p95_ttft_ms": float(percentile(ttft_values, 0.95) * 1000.0),
            "max_ttft_ms": float(max(ttft_values) * 1000.0) if ttft_values else 0.0,
        },
        {
            "system": "OpenAI GPT-5.5 official MRCR",
            "token_bin": token_bin,
            "samples": "reported",
            "accuracy": float(GPT55_BASELINE_ACCURACY),
            "mean_construction_time_s": None,
            "mean_ttft_ms": float(GPT55_BASELINE_TTFT_S * 1000.0),
            "p95_ttft_ms": None,
            "max_ttft_ms": None,
        },
    ]
    report = {
        "benchmark": "openai/mrcr",
        "snapshot_date": "2026-04-29",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stealth_mode": {
            "report_path": str(output_path),
            "csv_report_path": str(csv_output_path),
            "public_push_allowed": False,
            "results_committable": False,
            "telemetry_scope": "local_wsl_only",
        },
        "partition": {
            "needle_count": int(args.needle_count),
            "token_bin": token_bin,
            "min_tokens_exclusive": int(min_tokens),
            "max_tokens_inclusive": int(max_tokens),
            "filtered_case_count": int(len(cases)),
            "tokenizer": "o200k_base",
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
            "readout_bias": "random_string_to_prepend_prefix_locked_by_final_norm_hook",
            "activation_routes": {
                "storage": "numpy.lib.format.open_memmap",
                "filename": "activation_routes.npy",
                "dtype": "float32",
                "dim": int(args.activation_dim),
                "window_tokens": int(args.archival_window_tokens),
                "overlap_tokens": int(args.archival_overlap_tokens),
                "overlap_ratio": float(
                    args.archival_overlap_tokens / max(1, args.archival_window_tokens)
                ),
                "stride_tokens": int(
                    args.archival_window_tokens - args.archival_overlap_tokens
                ),
            },
        },
        "comparison_baseline": {
            "system": "OpenAI GPT-5.5 official MRCR",
            "accuracy": float(GPT55_BASELINE_ACCURACY),
            "ttft_s": float(GPT55_BASELINE_TTFT_S),
            "source": "user_supplied_agent_directive",
        },
        "execution": {
            "batch_size": int(args.batch_size),
            "candidate_pool": int(args.candidate_pool),
            "halt_consecutive_failures": int(args.halt_consecutive_failures),
            "halted": bool(halted),
            "halt_reason": halt_reason,
            "elapsed_s": float(time.perf_counter() - overall_started),
            "selection_elapsed_s": float(time.perf_counter() - inference_started),
            "index_build_elapsed_s": float(sum(index_build_values)),
            "activation_route_dir": str(args.activation_route_dir),
            "activation_chunk_size": int(args.activation_chunk_size),
            "total_activation_routes": int(sum(activation_route_counts)),
            "mean_activation_routes": 0.0
            if n == 0
            else float(sum(activation_route_counts) / n),
            "mean_message_windows": 0.0
            if n == 0
            else float(sum(message_window_counts) / n),
            "final_vram_usage": cuda_snapshot(torch),
        },
        "metrics": {
            "n": int(n),
            "correct": int(correct_count),
            "exact_match": int(exact_count),
            "accuracy": float(accuracy),
            "exact_match_rate": float(exact_rate),
            "mean_sequence_matcher": 0.0
            if n == 0
            else float(sum(item["sequence_matcher"] for item in all_results) / n),
            "mean_ttft_s": float(mean_ttft_s),
            "p95_ttft_s": float(percentile(ttft_values, 0.95)),
            "max_ttft_s": float(max(ttft_values)) if ttft_values else 0.0,
            "ttft_under_10ms_count": int(sum(1 for value in ttft_values if value < 0.010)),
            "ttft_under_10ms_rate": 0.0
            if n == 0
            else float(sum(1 for value in ttft_values if value < 0.010) / n),
            "mean_index_build_s": 0.0
            if n == 0
            else float(sum(index_build_values) / n),
            "gpt55_accuracy_delta": float(accuracy - GPT55_BASELINE_ACCURACY),
            "gpt55_ttft_speedup": None
            if mean_ttft_s <= 0.0
            else float(GPT55_BASELINE_TTFT_S / mean_ttft_s),
        },
        "summary_grid": summary_grid,
        "batches": batch_summaries,
        "cases": all_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv_report(csv_output_path, report)
    print(
        json.dumps(
            {
                "report": str(output_path),
                "csv_report": str(csv_output_path),
                "metrics": report["metrics"],
                "summary_grid": summary_grid,
            },
            indent=2,
        )
    )
    return 1 if halted else 0


if __name__ == "__main__":
    raise SystemExit(main())
