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
RESIDUAL_STREAMS_DIR = "residual_streams"
ACTIVATION_ROUTES_DIR = "activation_routes"
ACTIVATION_ROUTES_FILE = "activation_routes.npy"
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
    activation_routes_metadata: dict[str, Any] | None = None
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
        activation_routes_metadata = manifest.get("activation_routes")
        if activation_routes_metadata is not None and not isinstance(activation_routes_metadata, dict):
            activation_routes_metadata = None

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
            activation_routes_metadata=activation_routes_metadata,
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

        exact_mode, exact_matches = self._collect_exact_matches(query_text)
        exact_windows = self._primary_windows_for_matches(exact_matches)
        if exact_windows:
            if tokenizer is None:
                return exact_windows
            if exact_mode == "clause_id" and len(exact_windows) >= k:
                return exact_windows
            tfidf_result = self._route_tfidf_ranked(
                query_text,
                tokenizer,
                top_k=max(k, len(exact_windows) + 1),
            )
            if expansion_ids and len(tfidf_result) < max(k, len(exact_windows) + 1):
                expanded_result = self._route_tfidf_ranked(
                    query_text,
                    tokenizer,
                    top_k=max(k, len(exact_windows) + 1),
                    expansion_ids=expansion_ids,
                )
                seen_ranked = set(tfidf_result)
                for wid in expanded_result:
                    if wid not in seen_ranked:
                        tfidf_result.append(wid)
                        seen_ranked.add(wid)

            seen = set(exact_windows)
            merged = list(exact_windows)
            for wid in tfidf_result:
                if wid not in seen:
                    merged.append(wid)
                    seen.add(wid)
                if exact_mode == "clause_id" and len(merged) >= k:
                    break
            return merged

        base_result = self._route_tfidf_ranked(
            query_text,
            tokenizer,
            top_k=k,
        )
        if not expansion_ids or len(base_result) >= k:
            return base_result[:k]

        expanded_result = self._route_tfidf_ranked(
            query_text,
            tokenizer,
            top_k=k,
            expansion_ids=expansion_ids,
        )
        seen = set(base_result)
        merged = list(base_result)
        for wid in expanded_result:
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

    def load_residual_stream(self, window_id: int, device: str | Any = "cpu"):
        """Load a persisted post-crystal residual stream for ``window_id``.

        New live-indexed stores persist the full hidden stream that was already
        computed while capturing the boundary residual. Older stores only carry
        one boundary vector; callers that need a semantic token stream should
        treat ``FileNotFoundError`` as a strict unsupported-store condition.
        """
        if torch is None:  # pragma: no cover - local safety net
            raise RuntimeError("TorchKnowledgeStore requires torch to load residual streams")
        if self._store_path is None:
            raise ValueError("No store path available for residual stream loading")
        stream_path = self._store_path / RESIDUAL_STREAMS_DIR / f"window_{window_id:03d}.npy"
        if not stream_path.exists():
            raise FileNotFoundError(f"Residual stream not found: {stream_path}")

        stream_np = np.array(np.load(str(stream_path), allow_pickle=False), dtype=np.float32)
        stream_2d = stream_np.reshape(-1, stream_np.shape[-1])
        return torch.from_numpy(stream_2d).to(device=device)

    def load_activation_routes(self, device: str | Any = "cpu"):
        """Load the mean-pooled activation route matrix, if present."""
        if torch is None:  # pragma: no cover - local safety net
            raise RuntimeError("TorchKnowledgeStore requires torch to load activation routes")
        if self._store_path is None:
            raise ValueError("No store path available for activation route loading")

        route_path = self._store_path / ACTIVATION_ROUTES_FILE
        if self.activation_routes_metadata:
            manifest_path = self.activation_routes_metadata.get("path")
            if isinstance(manifest_path, str) and manifest_path:
                route_path = Path(manifest_path)
                if not route_path.is_absolute():
                    route_path = self._store_path / route_path

        if not route_path.exists():
            raise FileNotFoundError(f"Activation routes not found: {route_path}")

        routes_np = np.array(np.load(str(route_path), allow_pickle=False), dtype=np.float16)
        routes_2d = routes_np.reshape(-1, routes_np.shape[-1])
        return torch.from_numpy(routes_2d).to(device=device)

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
        ranked = self._route_tfidf_ranked(query_text, tokenizer, top_k=1)
        return ranked[0] if ranked else None

    def _route_keyword(self, query_text: str) -> int | None:
        router = self._get_keyword_router()
        return router.route(query_text)

    def _route_exact(self, query_text: str, tokenizer=None) -> list[int]:
        _, matches = self._collect_exact_matches(query_text)
        return self._primary_windows_for_matches(matches)

    def _route_tfidf_ranked(
        self,
        query_text: str,
        tokenizer,
        *,
        top_k: int,
        expansion_ids: list[int] | None = None,
    ) -> list[int]:
        if top_k <= 0:
            return []

        router = self._get_tfidf_router()
        query_ids = _encode_token_ids(tokenizer, query_text, add_special_tokens=False)
        if expansion_ids:
            useful = [int(t) for t in expansion_ids if self.idf.get(int(t), 0.0) > 0]
            if useful:
                query_ids = query_ids + useful

        hint_windows = set(self._route_hint_windows(query_text))
        scored: list[tuple[float, float, int]] = []
        for window_id in self.window_tokens:
            base_score = router.score_window(query_ids, window_id)
            if base_score <= 0.0 and window_id not in hint_windows:
                continue
            boosted_score = base_score + (0.1 if window_id in hint_windows else 0.0)
            scored.append((boosted_score, base_score, window_id))

        if not scored:
            return []

        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        if top_k == 1:
            return [scored[0][2]]

        top_score = scored[0][0]
        threshold = top_score * 0.5
        adaptive = [window_id for score, _, window_id in scored if score >= threshold]
        return adaptive[:top_k]

    def _collect_exact_matches(self, query_text: str) -> tuple[str | None, list[Any]]:
        router = self._get_clause_router()
        if router is None:
            return None, []
        clause_id_matches = router.collect_clause_id_matches(query_text)
        if clause_id_matches:
            return "clause_id", clause_id_matches
        title_matches = router.collect_exact_title_matches(query_text)
        if title_matches:
            return "title", title_matches
        return None, []

    def _route_hint_windows(self, query_text: str) -> list[int]:
        router = self._get_clause_router()
        if router is None:
            return []
        if router.collect_clause_id_matches(query_text):
            return []
        return self._primary_windows_for_matches(router.collect_hint_matches(query_text))

    def _primary_windows_for_matches(self, matches: list[Any]) -> list[int]:
        router = self._get_clause_router()
        if router is None:
            return []

        window_ids: list[int] = []
        seen_window_ids: set[int] = set()
        for match in matches:
            for window_id in router.clause_id_to_primary_windows.get(match.clause_id, []):
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


__all__ = [
    "ACTIVATION_ROUTES_DIR",
    "ACTIVATION_ROUTES_FILE",
    "TorchKnowledgeStore",
    "STORE_VERSION",
]
