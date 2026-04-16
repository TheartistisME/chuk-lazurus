"""Query routing — TF-IDF token overlap and keyword matching.

Two routing strategies:
    TFIDFRouter   — model-native: query token IDs vs window token sets, weighted by IDF
    KeywordRouter — text-based: keyword phrase matching (wraps SparseKeywordIndex)

Both return a window_id (int) for the best-matching window.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..research._stopwords import FUNCTION_WORDS as _FUNCTION_WORDS
from ..research._stopwords import WORD_RE as _WORD_RE

_CLAUSE_ID_RE = re.compile(r"\b\d+(?:\.\d+)+\b")
_CLAUSE_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PAREN_CONTENT_RE = re.compile(r"\(([^)]*)\)")
_GENERIC_SINGLE_TOKEN_TITLES = {"rcds", "protection", "live", "conductors", "general"}

# ── Keyword extraction helpers ───────────────────────────────────────


def _extract_keywords_from_text(text: str, max_keywords: int = 10) -> list[str]:
    """Extract content words from text, excluding function words."""
    words = _WORD_RE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        wl = w.lower()
        if wl not in _FUNCTION_WORDS and wl not in seen and len(wl) >= 2:
            seen.add(wl)
            result.append(wl)
            if len(result) >= max_keywords:
                break
    return result


def _normalize_clause_tokens(text: str) -> list[str]:
    return _CLAUSE_TOKEN_RE.findall(text.lower())


def _title_alias_token_sequences(title: str) -> list[tuple[str, ...]]:
    """Generate normalized token sequences that can identify a clause title.

    The full title is always included. If the title contains parenthetical
    acronyms, also include the stripped title and the acronym itself.
    """

    candidates: list[tuple[str, ...]] = []

    full_title = tuple(_normalize_clause_tokens(title))
    if full_title:
        candidates.append(full_title)

    stripped_title = tuple(_normalize_clause_tokens(_PAREN_CONTENT_RE.sub(" ", title)))
    if stripped_title and stripped_title not in candidates:
        candidates.append(stripped_title)

    for content in _PAREN_CONTENT_RE.findall(title):
        content_tokens = tuple(_normalize_clause_tokens(content))
        if content_tokens and content_tokens not in candidates:
            candidates.append(content_tokens)

    return candidates


def extract_window_keywords(
    token_ids: list[int],
    tokenizer,
    max_keywords: int = 5,
) -> list[str]:
    """Extract keywords from a window's tokens for the sparse index."""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return _extract_keywords_from_text(text, max_keywords=max_keywords)


@dataclass(frozen=True)
class _ClauseAliasPattern:
    tokens: tuple[str, ...]
    clause_id: str


@dataclass(frozen=True)
class _ClauseMatch:
    clause_id: str
    start: int
    token_count: int
    is_weak: bool


@dataclass
class ClauseMetadataRouter:
    """Exact clause router built from window_metadata.json.

    The router is intentionally conservative: it only matches exact clause
    ids or normalized clause-title / acronym phrases, then returns the
    primary window for each matched clause in query order.
    """

    window_metadata: dict[int, dict[str, Any]]
    clause_id_to_primary_windows: dict[str, list[int]]
    alias_patterns: list[_ClauseAliasPattern]

    @classmethod
    def from_window_metadata(
        cls, window_metadata: dict[int, dict[str, Any]]
    ) -> ClauseMetadataRouter:
        clause_groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for window_id, metadata in window_metadata.items():
            clause_id = str(metadata.get("clause_id", "")).strip()
            if not clause_id:
                continue
            clause_groups.setdefault(clause_id, []).append((int(window_id), metadata))

        clause_id_to_primary_windows: dict[str, list[int]] = {}
        alias_patterns: list[_ClauseAliasPattern] = []

        for clause_id, rows in clause_groups.items():
            rows.sort(
                key=lambda item: (
                    int(item[1].get("part_index", 1) or 1),
                    int(item[0]),
                )
            )
            primary_windows = [
                window_id
                for window_id, metadata in rows
                if int(metadata.get("part_index", 1) or 1) == 1
            ]
            if not primary_windows:
                primary_windows = [rows[0][0]]
            clause_id_to_primary_windows[clause_id] = primary_windows

            representative = rows[0][1]
            title = str(representative.get("clause_title", "")).strip()
            alias_tokens = _title_alias_token_sequences(title)
            for tokens in alias_tokens:
                alias_patterns.append(
                    _ClauseAliasPattern(
                        tokens=tokens,
                        clause_id=clause_id,
                    )
                )

        alias_patterns.sort(key=lambda pattern: (-len(pattern.tokens), pattern.clause_id))
        return cls(
            window_metadata=window_metadata,
            clause_id_to_primary_windows=clause_id_to_primary_windows,
            alias_patterns=alias_patterns,
        )

    def collect_matches(self, query_text: str) -> tuple[list[_ClauseMatch], list[_ClauseMatch]]:
        """Return strong and weak clause matches in query order."""

        if not self.window_metadata:
            return [], []

        strong_by_clause: dict[str, _ClauseMatch] = {}
        weak_by_clause: dict[str, _ClauseMatch] = {}

        for clause_match in _CLAUSE_ID_RE.finditer(query_text):
            clause_id = clause_match.group(0)
            if clause_id in self.clause_id_to_primary_windows:
                strong_by_clause[clause_id] = _ClauseMatch(
                    clause_id=clause_id,
                    start=clause_match.start(),
                    token_count=0,
                    is_weak=False,
                )

        query_tokens = _normalize_clause_tokens(query_text)
        for pattern in self.alias_patterns:
            tokens = pattern.tokens
            token_count = len(tokens)
            if token_count == 0 or token_count > len(query_tokens):
                continue
            max_start = len(query_tokens) - token_count
            for start in range(max_start + 1):
                if tuple(query_tokens[start : start + token_count]) != tokens:
                    continue

                match = _ClauseMatch(
                    clause_id=pattern.clause_id,
                    start=start,
                    token_count=token_count,
                    is_weak=token_count == 1 and tokens[0] in _GENERIC_SINGLE_TOKEN_TITLES,
                )
                if match.is_weak:
                    if pattern.clause_id not in strong_by_clause and pattern.clause_id not in weak_by_clause:
                        weak_by_clause[pattern.clause_id] = match
                else:
                    existing = strong_by_clause.get(pattern.clause_id)
                    if existing is None or (match.start, -match.token_count) < (existing.start, -existing.token_count):
                        strong_by_clause[pattern.clause_id] = match
                break

        strong_matches = sorted(
            strong_by_clause.values(),
            key=lambda item: (item.start, -item.token_count, item.clause_id),
        )
        weak_matches = sorted(
            weak_by_clause.values(),
            key=lambda item: (item.start, -item.token_count, item.clause_id),
        )
        return strong_matches, weak_matches

    def route_primary_windows(self, query_text: str) -> list[int]:
        strong_matches, weak_matches = self.collect_matches(query_text)
        matches = strong_matches or weak_matches
        window_ids: list[int] = []
        seen_window_ids: set[int] = set()
        for match in matches:
            for window_id in self.clause_id_to_primary_windows.get(match.clause_id, []):
                if window_id not in seen_window_ids:
                    seen_window_ids.add(window_id)
                    window_ids.append(window_id)
        return window_ids


# ── Sparse keyword index ─────────────────────────────────────────────


@dataclass
class SparseKeywordIndex:
    """Keyword -> passage mapping for query-time routing.

    ~800 bytes total for a 370K-token document.
    """

    entries: dict[str, list[int]] = field(default_factory=dict)
    """keyword (lowercase) -> list of passage indices."""

    def add(self, passage_idx: int, keywords: list[str]) -> None:
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in self.entries:
                self.entries[kw_lower] = []
            if passage_idx not in self.entries[kw_lower]:
                self.entries[kw_lower].append(passage_idx)

    def query(self, query_text: str) -> list[int]:
        """Return passage indices matching any query keyword, ranked by hit count."""
        query_words = _extract_keywords_from_text(query_text)
        hits: dict[int, int] = {}
        for word in query_words:
            for idx in self.entries.get(word, []):
                hits[idx] = hits.get(idx, 0) + 1
        return sorted(hits, key=lambda i: (hits[i], i), reverse=True)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.entries, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> SparseKeywordIndex:
        data = json.loads(path.read_text())
        return cls(entries=data)

    @property
    def num_keywords(self) -> int:
        return len(self.entries)


# ── TF-IDF Router ────────────────────────────────────────────────────


class TFIDFRouter:
    """Route queries by TF-IDF weighted token overlap.

    During build, each window's unique token IDs and a global IDF table
    are computed.  At query time, the query's token IDs are intersected
    with each window's token set, scored by IDF weight.
    """

    def __init__(
        self,
        window_tokens: dict[int, set[int]],
        idf: dict[int, float],
    ) -> None:
        self.window_tokens = window_tokens
        self.idf = idf

    def route(self, query_token_ids: list[int], top_k: int = 1) -> int | list[int] | None:
        """Return the top window_id(s) by TF-IDF overlap score.

        top_k=1: returns single int (backward compat). top_k>1: returns list.
        When top_k > 1, uses adaptive selection: includes windows within
        50% of the top score. If the top window is 10× stronger than #2,
        only that window is returned.
        """
        if not self.window_tokens:
            return None if top_k == 1 else []
        query_set = set(query_token_ids)
        scored: list[tuple[float, int, int]] = []
        for wid, tokens in self.window_tokens.items():
            overlap = query_set & tokens
            if not overlap:
                continue
            score = sum(self.idf.get(t, 0.0) for t in overlap)
            scored.append((score, len(overlap), wid))
        if not scored:
            return None if top_k == 1 else []
        scored.sort(reverse=True)
        if top_k == 1:
            return scored[0][2]

        # Adaptive: include windows within 50% of the top score
        top_score = scored[0][0]
        threshold = top_score * 0.5
        adaptive = [wid for s, _, wid in scored if s >= threshold]
        # Cap at top_k
        return adaptive[:top_k]

    def route_with_score(self, query_token_ids: list[int]) -> tuple[int | None, float]:
        """Return (window_id, score) for best match."""
        if not self.window_tokens:
            return None, 0.0
        query_set = set(query_token_ids)
        best_wid: int | None = None
        best_score: float = 0.0
        best_rank: tuple[float, int, int] | None = None
        for wid, tokens in self.window_tokens.items():
            overlap = query_set & tokens
            if not overlap:
                continue
            score = sum(self.idf.get(t, 0.0) for t in overlap)
            rank = (score, len(overlap), wid)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_score = score
                best_wid = wid
        return best_wid, best_score

    def score_window(self, query_token_ids: list[int], window_id: int) -> float:
        tokens = self.window_tokens.get(window_id)
        if not tokens:
            return 0.0
        query_set = set(query_token_ids)
        overlap = query_set & tokens
        return sum(self.idf.get(t, 0.0) for t in overlap)

    # ── Build helpers ─────────────────────────────────────────────────

    @staticmethod
    def compute_idf(window_tokens: dict[int, set[int]]) -> dict[int, float]:
        """Compute IDF from window token sets.

        IDF(t) = log(N / df(t))  where df(t) = number of windows containing token t.
        """
        n_windows = len(window_tokens)
        if n_windows == 0:
            return {}
        token_df: dict[int, int] = {}
        for tokens in window_tokens.values():
            for t in tokens:
                token_df[t] = token_df.get(t, 0) + 1
        return {t: math.log(n_windows / df) for t, df in token_df.items()}


# ── Keyword Router ────────────────────────────────────────────────────


class KeywordRouter:
    """Route queries by keyword phrase matching.

    Wraps SparseKeywordIndex with the same interface as TFIDFRouter.
    """

    def __init__(self, keywords: dict[int, list[str]]) -> None:
        self._index = SparseKeywordIndex()
        for wid, kws in keywords.items():
            self._index.add(wid, kws)

    def route(self, query_text: str) -> int | None:
        """Return the window_id with most keyword hits, or None."""
        candidates = self._index.query(query_text)
        return candidates[0] if candidates else None

    @property
    def sparse_index(self) -> SparseKeywordIndex:
        return self._index
