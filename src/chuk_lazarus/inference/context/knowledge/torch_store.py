"""Torch-clean Apollo knowledge store loader.

This module mirrors the v12 knowledge-store layout without importing MLX.
It is intentionally narrow: enough to load the Apollo v12 store, route
queries, decode window text, and load per-window boundary tensors for
downstream CUDA use.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:  # Torch is required for downstream Apollo execution, but keep import local-safe.
    import torch
except Exception:  # pragma: no cover - exercised only on hosts without torch
    torch = None

from .config import ArchitectureConfig
from .route import ClauseMetadataRouter, KeywordRouter, TFIDFRouter

MANIFEST_FILE = "manifest.json"
WINDOW_TOKENS_FILE = "window_tokens.npz"
WINDOW_TOKEN_LISTS_FILE = "window_token_lists.npz"
IDF_FILE = "idf.json"
KEYWORDS_FILE = "keywords.json"
WINDOW_METADATA_FILE = "window_metadata.json"
BOUNDARIES_DIR = "boundaries"
STORE_VERSION = 12


def _sorted_npz_keys(files: list[str]) -> list[str]:
    return sorted(files, key=lambda key: int(key))


def _load_npz_int_set_map(path: Path) -> dict[int, set[int]]:
    if not path.exists():
        return {}
    npz = np.load(str(path), allow_pickle=False)
    result: dict[int, set[int]] = {}
    for key in _sorted_npz_keys(list(npz.files)):
        result[int(key)] = {int(v) for v in np.asarray(npz[key]).tolist()}
    return result


def _load_npz_int_list_map(path: Path) -> dict[int, list[int]]:
    if not path.exists():
        return {}
    npz = np.load(str(path), allow_pickle=False)
    result: dict[int, list[int]] = {}
    for key in _sorted_npz_keys(list(npz.files)):
        result[int(key)] = [int(v) for v in np.asarray(npz[key]).tolist()]
    return result


def _load_json_int_float_map(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {int(k): float(v) for k, v in raw.items()}


def _load_json_int_list_map(path: Path) -> dict[int, list[str]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {int(k): [str(v) for v in values] for k, values in raw.items()}


def _load_window_metadata_map(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    result: dict[int, dict[str, Any]] = {}

    if isinstance(raw, dict):
        for key, value in raw.items():
            result[int(key)] = dict(value)
        return result

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict) or "window_id" not in item:
                continue
            result[int(item["window_id"])] = dict(item)
        return result

    raise ValueError(f"Unsupported window metadata format in {path}")


def _encode_token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(t) for t in token_ids]


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


@dataclass
class TorchKnowledgeStore:
    """Torch-native view of the Apollo v12 knowledge store."""

    window_tokens: dict[int, set[int]]
    window_token_lists: dict[int, list[int]]
    idf: dict[int, float]
    keywords: dict[int, list[str]]
    config: ArchitectureConfig
    window_metadata: dict[int, dict[str, Any]] = field(default_factory=dict)
    num_windows: int = 0
    num_tokens: int = 0
    _store_path: Path | None = field(default=None, repr=False)
    _tfidf_router: TFIDFRouter | None = field(default=None, repr=False)
    _keyword_router: KeywordRouter | None = field(default=None, repr=False)
    _clause_router: ClauseMetadataRouter | None = field(default=None, repr=False)

    @classmethod
    def load(cls, path: Path | str) -> TorchKnowledgeStore:
        path = Path(path)
        manifest = json.loads((path / MANIFEST_FILE).read_text())
        config = ArchitectureConfig.from_dict(manifest["arch_config"])

        window_tokens = _load_npz_int_set_map(path / WINDOW_TOKENS_FILE)
        window_token_lists = _load_npz_int_list_map(path / WINDOW_TOKEN_LISTS_FILE)
        if not window_token_lists and window_tokens:
            window_token_lists = {wid: sorted(tokens) for wid, tokens in window_tokens.items()}
        if not window_tokens and window_token_lists:
            window_tokens = {wid: set(tokens) for wid, tokens in window_token_lists.items()}

        idf = _load_json_int_float_map(path / IDF_FILE)
        if not idf and window_tokens:
            idf = TFIDFRouter.compute_idf(window_tokens)

        keywords = _load_json_int_list_map(path / KEYWORDS_FILE)
        window_metadata_name = manifest.get("window_metadata", WINDOW_METADATA_FILE)
        window_metadata_path = Path(window_metadata_name)
        if not window_metadata_path.is_absolute():
            window_metadata_path = path / window_metadata_path
        window_metadata = _load_window_metadata_map(window_metadata_path)

        num_windows = int(manifest.get("num_windows", max(window_tokens.keys(), default=-1) + 1))
        if num_windows == 0 and window_token_lists:
            num_windows = max(window_token_lists.keys(), default=-1) + 1
        if window_metadata:
            metadata_num_windows = max(window_metadata.keys(), default=-1) + 1
            num_windows = max(num_windows, metadata_num_windows)

        num_tokens = int(
            manifest.get(
                "num_tokens",
                sum(len(token_list) for token_list in window_token_lists.values()),
            )
        )

        store = cls(
            window_tokens=window_tokens,
            window_token_lists=window_token_lists,
            idf=idf,
            keywords=keywords,
            config=config,
            window_metadata=window_metadata,
            num_windows=num_windows,
            num_tokens=num_tokens,
        )
        store._store_path = path
        return store

    def route(self, query_text: str, tokenizer=None, method: str = "auto") -> int | None:
        if method == "auto":
            exact_windows = self._route_exact(query_text, tokenizer=tokenizer)
            if exact_windows:
                return exact_windows[0]
            if tokenizer is not None and self.window_tokens:
                wid = self._route_tfidf(query_text, tokenizer)
                if wid is not None:
                    return wid
            return self._route_keyword(query_text)
        if method == "tfidf":
            if tokenizer is None:
                raise ValueError("TF-IDF routing requires a tokenizer")
            return self._route_tfidf(query_text, tokenizer)
        if method == "keyword":
            return self._route_keyword(query_text)
        raise ValueError(f"Unknown routing method: {method!r}")

    def route_top_k(
        self,
        query_text: str,
        tokenizer,
        k: int = 3,
        expansion_ids: list[int] | None = None,
    ) -> list[int]:
        if k <= 0:
            return []

        exact_windows = self._route_exact(query_text, tokenizer=tokenizer)
        if exact_windows:
            if len(exact_windows) >= k or tokenizer is None:
                return exact_windows

            router = self._get_tfidf_router()
            query_ids = _encode_token_ids(tokenizer, query_text, add_special_tokens=False)
            if expansion_ids:
                useful = [int(t) for t in expansion_ids if self.idf.get(int(t), 0.0) > 0]
                if useful:
                    query_ids = query_ids + useful
            tfidf_result = router.route(query_ids, top_k=k)
            if not isinstance(tfidf_result, list):
                tfidf_result = [tfidf_result] if tfidf_result is not None else []

            seen = set(exact_windows)
            merged = list(exact_windows)
            for wid in tfidf_result:
                if wid not in seen:
                    merged.append(wid)
                    seen.add(wid)
                if len(merged) >= k:
                    break
            return merged

        router = self._get_tfidf_router()
        query_ids = _encode_token_ids(tokenizer, query_text, add_special_tokens=False)

        base_result = router.route(query_ids, top_k=k)
        if not isinstance(base_result, list):
            base_result = [base_result] if base_result is not None else []

        if not expansion_ids or len(base_result) >= k:
            return base_result[:k]

        useful = [int(t) for t in expansion_ids if self.idf.get(int(t), 0.0) > 0]
        if not useful:
            return base_result[:k]

        expanded_ids = query_ids + useful
        exp_result = router.route(expanded_ids, top_k=k)
        if not isinstance(exp_result, list):
            exp_result = [exp_result] if exp_result is not None else []

        seen = set(base_result)
        merged = list(base_result)
        for wid in exp_result:
            if wid not in seen and len(merged) < k:
                merged.append(wid)
                seen.add(wid)
        return merged[:k]

    def get_window_text(self, window_id: int, tokenizer) -> str:
        token_list = self.window_token_lists.get(window_id, [])
        return _decode_token_ids(tokenizer, token_list)

    def load_boundary(self, window_id: int, device: str | Any = "cpu"):
        if torch is None:  # pragma: no cover - local safety net
            raise RuntimeError("TorchKnowledgeStore requires torch to load boundaries")
        if self._store_path is None:
            raise ValueError("No store path available for boundary loading")
        boundary_path = self._store_path / BOUNDARIES_DIR / f"window_{window_id:03d}.npy"
        if not boundary_path.exists():
            raise FileNotFoundError(f"Boundary not found: {boundary_path}")

        boundary_np = np.array(np.load(str(boundary_path), allow_pickle=False), dtype=np.float32)
        boundary_1d = boundary_np.reshape(-1)
        return torch.from_numpy(boundary_1d).to(device=device)

    def log_stats(self, file=sys.stderr) -> None:
        window_token_bytes = sum(len(tokens) * 2 for tokens in self.window_tokens.values())
        window_list_bytes = sum(len(tokens) * 4 for tokens in self.window_token_lists.values())
        idf_bytes = len(self.idf) * 12
        keyword_bytes = sum(sum(len(keyword) for keyword in kws) for kws in self.keywords.values())
        total_bytes = window_token_bytes + window_list_bytes + idf_bytes + keyword_bytes

        print(
            f"  TorchKnowledgeStore v{STORE_VERSION}: "
            f"{self.num_windows} windows  "
            f"tokens={window_token_bytes / 1024:.1f}KB  "
            f"token_lists={window_list_bytes / 1024:.1f}KB  "
            f"idf={idf_bytes / 1024:.1f}KB  "
            f"keywords={keyword_bytes / 1024:.1f}KB  "
            f"total~{total_bytes / 1024:.1f}KB  "
            f"doc={self.num_tokens} tokens  "
            f"crystal=L{self.config.crystal_layer}  "
            f"window={self.config.window_size}",
            file=file,
        )

    def _route_tfidf(self, query_text: str, tokenizer) -> int | None:
        router = self._get_tfidf_router()
        query_ids = _encode_token_ids(tokenizer, query_text, add_special_tokens=False)
        return router.route(query_ids)

    def _route_keyword(self, query_text: str) -> int | None:
        router = self._get_keyword_router()
        return router.route(query_text)

    def _route_exact(self, query_text: str, tokenizer=None) -> list[int]:
        router = self._get_clause_router()
        if router is None:
            return []
        strong_matches, weak_matches = router.collect_matches(query_text)
        matches = strong_matches or weak_matches
        if not matches:
            return []

        if tokenizer is None or not self.window_tokens:
            window_ids: list[int] = []
            seen_window_ids: set[int] = set()
            for match in matches:
                for window_id in router.clause_id_to_primary_windows.get(match.clause_id, []):
                    if window_id not in seen_window_ids:
                        seen_window_ids.add(window_id)
                        window_ids.append(window_id)
            return window_ids

        query_ids = _encode_token_ids(tokenizer, query_text, add_special_tokens=False)
        tfidf_router = self._get_tfidf_router()

        if strong_matches:
            scored: list[tuple[int, float, int]] = []
            for match in strong_matches:
                for window_id in router.clause_id_to_primary_windows.get(match.clause_id, []):
                    scored.append((match.start, tfidf_router.score_window(query_ids, window_id), window_id))
            scored.sort(key=lambda item: (item[0], -item[1], item[2]))
            window_ids: list[int] = []
            seen_window_ids: set[int] = set()
            for _, _, window_id in scored:
                if window_id not in seen_window_ids:
                    seen_window_ids.add(window_id)
                    window_ids.append(window_id)
            return window_ids

        tfidf_candidates = tfidf_router.route(query_ids, top_k=max(5, len(weak_matches)))
        if not isinstance(tfidf_candidates, list):
            tfidf_candidates = [tfidf_candidates] if tfidf_candidates is not None else []

        candidate_ids: list[int] = []
        weak_ids = set()
        for match in weak_matches:
            for window_id in router.clause_id_to_primary_windows.get(match.clause_id, []):
                if window_id not in weak_ids:
                    weak_ids.add(window_id)
                    candidate_ids.append(window_id)
        for window_id in tfidf_candidates:
            if window_id not in weak_ids and window_id not in candidate_ids:
                candidate_ids.append(window_id)

        if not candidate_ids:
            return []

        scored_candidates: list[tuple[float, int, int]] = []
        for window_id in candidate_ids:
            score = tfidf_router.score_window(query_ids, window_id)
            if window_id in weak_ids:
                score -= 0.25
            scored_candidates.append((score, window_id, window_id))

        scored_candidates.sort(key=lambda item: (-item[0], item[1]))
        window_ids: list[int] = []
        seen_window_ids: set[int] = set()
        for _, _, window_id in scored_candidates:
            if window_id not in seen_window_ids:
                seen_window_ids.add(window_id)
                window_ids.append(window_id)
        return window_ids

    def _get_tfidf_router(self) -> TFIDFRouter:
        if self._tfidf_router is None:
            self._tfidf_router = TFIDFRouter(self.window_tokens, self.idf)
        return self._tfidf_router

    def _get_keyword_router(self) -> KeywordRouter:
        if self._keyword_router is None:
            self._keyword_router = KeywordRouter(self.keywords)
        return self._keyword_router

    def _get_clause_router(self) -> ClauseMetadataRouter | None:
        if self._clause_router is None and self.window_metadata:
            self._clause_router = ClauseMetadataRouter.from_window_metadata(self.window_metadata)
        return self._clause_router


__all__ = ["TorchKnowledgeStore", "STORE_VERSION"]
