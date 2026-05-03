#!/usr/bin/env python3
"""Run the Lazarus LoCoBench coding benchmark."""

from __future__ import annotations

import _thread
import argparse
import csv
import difflib
import hashlib
import json
import math
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

from benchmark_jit_indexing import jit_index_dataset_windows, release_cuda_memory

from chuk_lazarus.benchmarks.baselines import COMPARISON_BASELINES
from chuk_lazarus.memory_config import DynamicAllocatorConfig

from central_router_bridge import (
    DEPENDENCY_SOURCE,
    add_router_backend_argument,
    candidate_scores_by_id,
    central_route_plan,
    is_central_backend,
    tier_by_window_id,
)

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
DEFAULT_KV_MAX_NEW_TOKENS = 1536
DEFAULT_KV_REPLACEMENT_SPILLOVER_TOKENS = 768
DEFAULT_KV_DECODE_PAYLOAD_TOKEN_CAP = (
    DEFAULT_KV_MAX_NEW_TOKENS + DEFAULT_KV_REPLACEMENT_SPILLOVER_TOKENS
)
DEFAULT_KV_DECODE_TIMEOUT_SECONDS = 300
DEFAULT_KV_DECODE_STALL_SECONDS = 90
DEFAULT_KV_DECODE_HEARTBEAT_SECONDS = 30
DEFAULT_DECODER_DIALECT_STEERING_TOP_K = 8
DEFAULT_DECODER_DIALECT_STEERING_MARGIN = -0.75
DEFAULT_DECODER_DIALECT_STEERING_DECAY = 0.82
DEFAULT_DECODER_DIALECT_STEERING_FLOOR = 0.03
DEFAULT_DECODER_DIALECT_STEERING_FRACTION = 0.12
DEFAULT_DECODER_DIALECT_STEERING_MAX_DELTA_RMS = 0.20
DEFAULT_DECODER_DIALECT_STEERING_MAX_EVENTS = 64
SNAPSHOT_DATE = datetime.now(timezone.utc).date().isoformat()
DEFAULT_OUTPUT_PATH = Path(".benchmarks") / f"locobench_results_snapshot_{SNAPSHOT_DATE}.json"
DEFAULT_CSV_PATH = Path(".benchmarks") / f"locobench_results_snapshot_{SNAPSHOT_DATE}.csv"
DEFAULT_JIT_DIR = Path(".benchmarks") / "jit"
STRUCTURAL_ANCHOR = "<dependency_context_loaded>\n<PATCH_GENERATION_START>\n"
TARGET_FILE_OPEN = "<target_file>"
TARGET_FILE_CLOSE = "</target_file>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
APOLLO_PHASE3_KIND = "benchmark_jit_apollo_sequential_residual"
DIFF_PROTOCOL_CONSTRAINT = """# ROLE:

Language Specific Coding Agent

A precise language specific coding assistant

# TASK:

Take user query
Determine working coding language: e.g c++ vs c11, java vs javascript  ect.
 Output diff

OUTPUT_DIFF_ONLY

[MANDATORY_FORMAT]
<file_path>
<<<< SEARCH
[verbatim lines from source, no placeholders]
==== REPLACE
[new lines]
>>>>

# MANDATORY_BEHAVIOR
- Follow EXACT language and code conventions: 1. Paradigm Lock
- DO NOT introduce new programming paradigms or language features (e.g., no C++ Classes/Namespaces in C files).

2. Signature Fidelity
- Mirror existing function signatures and naming patterns (e.g., use snake_case if the file uses it, PascalCase if it does not).

3. Vocabulary Ceiling
- Use ONLY identifiers, types, and methods already present in the code; do not invent "plausible" API calls.

[RULES]
1. Start response immediately with file path.
2. ZERO prose, ZERO explanation, ZERO ellipses (...).
3. SEARCH block must be a character-for-character clone of source."""
PATTERN_WINDOW_ID = -10_001
PATTERN_WINDOW_PATH = "__phase4_search_replace_pattern__"
SEARCH_REPLACE_PATTERN_WINDOW = """examples/alpha.py
<<<< SEARCH
def old_name(value):
    return value + 1
==== REPLACE
def new_name(value):
    return value + 1
>>>>
examples/beta.py
<<<< SEARCH
if enabled:
    run_task()
==== REPLACE
if enabled:
    run_task()
else:
    skip_task()
>>>>
examples/gamma.py
<<<< SEARCH
return handler.process(request)
==== REPLACE
return handler.process(request, context=context)
>>>>"""
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
TERM_RE = re.compile(r"[A-Za-z0-9_]+")
SEARCH_REPLACE_RE = re.compile(
    r"<<<<\s*SEARCH\s*(?P<old>.*?)\s*====\s*REPLACE\s*(?P<new>.*?)\s*>>>>",
    flags=re.IGNORECASE | re.DOTALL,
)
DIFF_STATE_OUTSIDE = "OUTSIDE"
DIFF_STATE_SEARCH_BLOCK = "SEARCH_BLOCK"
DIFF_STATE_REPLACE_BLOCK = "REPLACE_BLOCK"
DIFF_STATE_CLOSED = "CLOSED"
REPLACE_BLOCK_PROTOCOL_MARKERS = (
    "<<<< SEARCH",
    "<<<<",
    "==== REPLACE",
    "====",
    ">>>>",
    "SEARCH",
    "REPLACE",
    "<file_path>",
)
IDENTIFIER_TAIL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
IDENTIFIER_CHAR_RE = re.compile(r"[A-Za-z0-9_]")
IDENTIFIER_SEGMENT_GROWTH_CHAIN_RE = re.compile(
    r"(?P<chain>[A-Za-z_][A-Za-z0-9_]*"
    r"(?:(?:\\|::|->|\.)[A-Za-z_][A-Za-z0-9_]*)+)"
    r"(?P<trailer>[;,\)\]\}]*)$"
)
IDENTIFIER_SEGMENT_SEPARATOR_RE = re.compile(r"(\\|::|->|\.)")
C_KEYWORD_IDENTIFIERS = {
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Bool",
    "_Complex",
    "_Imaginary",
}
COMMON_LOCAL_IDENTIFIERS = {
    "NULL",
    "SEARCH",
    "REPLACE",
    "a",
    "b",
    "c",
    "ch",
    "cnt",
    "count",
    "ctx",
    "data",
    "dst",
    "end",
    "err",
    "fp",
    "i",
    "idx",
    "j",
    "k",
    "len",
    "n",
    "ok",
    "p",
    "ptr",
    "rc",
    "ret",
    "src",
    "tmp",
    "value",
}
DOMAIN_CORE_PATH_PARTS = {
    "api",
    "app",
    "cmd",
    "core",
    "database",
    "db",
    "handler",
    "handlers",
    "internal",
    "lib",
    "model",
    "models",
    "pkg",
    "repository",
    "repositories",
    "repo",
    "route",
    "routes",
    "server",
    "service",
    "services",
    "src",
    "storage",
}
DOMAIN_PERIPHERAL_PATH_PARTS = {
    "asset",
    "assets",
    "build",
    "css",
    "dist",
    "docs",
    "examples",
    "frontend",
    "images",
    "public",
    "static",
    "style",
    "styles",
    "template",
    "templates",
    "theme",
    "themes",
    "view",
    "views",
}
DOMAIN_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".go",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".java",
    ".js",
    ".jsx",
    ".php",
    ".py",
    ".pyi",
    ".rb",
    ".rs",
    ".ts",
    ".tsx",
}
DOMAIN_PERIPHERAL_SUFFIXES = {
    ".css",
    ".gif",
    ".html",
    ".ico",
    ".jpeg",
    ".jpg",
    ".less",
    ".map",
    ".md",
    ".mdx",
    ".png",
    ".sass",
    ".scss",
    ".svg",
    ".template",
    ".tmpl",
    ".vue",
}


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
    start_line: int = 1
    end_line: int = 1
    start_token: int = 0
    end_token: int = 0
    activation_route_id: int | None = None
    lexical_score: float = 0.0
    activation_score: float = 0.0
    exact_scope_match: bool = False
    route_source: str = "hash_cosine"
    recursive_map_window_id: int | None = None
    recursive_pass: str = ""


@dataclass(frozen=True)
class LoCoActivationRouteIndex:
    path: Path
    metadata_path: Path
    route_count: int
    dim: int
    dtype: str
    window_tokens: int
    overlap_tokens: int
    stride_tokens: int
    routes: Any | None = None


@dataclass(frozen=True)
class PatchResult:
    applied: bool
    block_count: int
    changed_files: tuple[str, ...]
    failures: tuple[str, ...]


@dataclass(frozen=True)
class DialectLockSpec:
    dialect: str
    extension: str
    forbidden_literals: tuple[str, ...]
    enforce_c11_case: bool = False
    replace_region_only: bool = True
    inference_mode: str = "extension"
    confidence: float = 1.0
    signals: tuple[str, ...] = ()
    contrast_families_banned: tuple[str, ...] = ()


C11_CPLUSPLUS_FORBIDDEN_LITERALS = (
    "class",
    "public",
    "private",
    "protected",
    "public:",
    "private:",
    "protected:",
    "new",
    "delete",
    "template",
    "namespace",
    "std",
    "std::",
    "::",
    "nullptr",
)

DIALECT_CONTRAST_LOCKS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "c11": (("cpp", C11_CPLUSPLUS_FORBIDDEN_LITERALS),),
    "cpp": (
        ("python", ("def", "self")),
        ("go", ("package", "func", "defer", ":=")),
        ("rust", ("fn", "impl", "crate", "let", "mut")),
        ("java", ("extends", "implements", "throws", "import")),
    ),
    "python": (
        ("c_cpp", ("{", "}", ";", "void", "unsigned", "struct", "uint32_t")),
        ("cpp", ("std::", "::", "template", "namespace", "nullptr")),
        ("go", ("package", "func", "defer", ":=")),
        ("rust", ("fn", "impl", "crate", "let", "mut")),
        ("java_csharp", ("public", "private", "protected")),
    ),
    "java": (
        ("c_cpp", ("struct", "free", "malloc", "uint32_t", "std::", "::")),
        ("python", ("def", "self")),
        ("go", ("func", "defer", ":=")),
        ("rust", ("fn", "impl", "crate", "let", "mut")),
    ),
    "csharp": (
        ("c_cpp", ("template", "std::", "::", "free", "malloc", "uint32_t")),
        ("python", ("def", "self")),
        ("go", ("package", "func", "defer", ":=")),
        ("rust", ("fn", "impl", "crate", "let", "mut")),
        ("java", ("extends", "implements", "throws", "import")),
    ),
    "go": (
        ("python", ("def", "self", "class")),
        ("cpp", ("template", "namespace", "std", "std::", "::", "nullptr")),
        ("java_csharp", ("public", "private", "protected", "extends", "throws")),
        ("rust", ("fn", "impl", "crate", "let", "mut")),
    ),
    "rust": (
        ("python", ("def", "class")),
        ("cpp", ("public:", "private:", "protected:", "template", "namespace")),
        ("go", ("package", "func", "defer", ":=")),
        ("java_csharp", ("public", "private", "protected", "extends", "throws")),
    ),
    "typescript": (
        ("python", ("def", "self")),
        ("c_cpp", ("std::", "::", "malloc", "free", "uint32_t")),
        ("go", ("func", "defer", ":=")),
        ("rust", ("fn", "impl", "crate", "mut")),
        ("java", ("throws", "extends", "implements")),
    ),
    "javascript": (
        ("python", ("def", "self")),
        ("c_cpp", ("std::", "::", "malloc", "free", "uint32_t")),
        ("go", ("func", "defer", ":=")),
        ("rust", ("fn", "impl", "crate", "mut")),
        ("java", ("throws", "extends", "implements")),
    ),
}

DIALECT_STEERING_TARGET_ANCHORS: dict[str, tuple[str, ...]] = {
    "c11": ("struct", "static", "void", "return"),
    "cpp": ("class", "template", "namespace", "std::", "return"),
    "python": ("def", "self", "import", "None", "return"),
    "java": ("class", "public", "private", "return"),
    "csharp": ("class", "public", "private", "return"),
    "go": ("func", "package", "defer", "return"),
    "rust": ("fn", "impl", "let", "mut", "return"),
    "typescript": ("function", "const", "let", "=>", "interface", "type", "return"),
    "javascript": ("function", "const", "let", "=>", "require", "return"),
}

DIALECT_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "c11": (".c", ".h"),
    "cpp": (".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx"),
    "python": (".py", ".pyi"),
    "java": (".java",),
    "csharp": (".cs",),
    "go": (".go",),
    "rust": (".rs",),
    "typescript": (".ts", ".tsx"),
    "javascript": (".js", ".jsx"),
}

DIALECT_BY_EXTENSION = {
    extension: dialect
    for dialect, extensions in DIALECT_EXTENSIONS.items()
    for extension in extensions
}

AMBIGUOUS_DIALECT_EXTENSIONS = {".h", ".hh", ".hxx"}

DIALECT_SYNTAX_FINGERPRINTS: dict[str, tuple[tuple[str, str, float], ...]] = {
    "c11": (
        ("c_include", r"(?m)^\s*#\s*include\s+[<\"][^>\"]+\.h[>\"]", 2.0),
        ("c_typedef_struct", r"\btypedef\s+struct\b", 2.5),
        ("c_struct", r"\bstruct\s+[A-Za-z_][A-Za-z0-9_]*\s*\{", 1.75),
        ("c_pointer_signature", r"\b(?:int|void|char|float|double)\s+\*?\w+\s*\(", 1.0),
    ),
    "cpp": (
        ("cpp_namespace", r"\bnamespace\s+[A-Za-z_][A-Za-z0-9_]*\s*\{", 3.0),
        ("cpp_std_scope", r"\bstd::[A-Za-z_][A-Za-z0-9_]*", 3.0),
        ("cpp_template", r"\btemplate\s*<", 3.0),
        ("cpp_class", r"(?m)^\s*class\s+[A-Za-z_][A-Za-z0-9_]*", 2.5),
        ("cpp_access_label", r"(?m)^\s*(?:public|private|protected):\s*$", 2.0),
    ),
    "python": (
        ("python_def", r"(?m)^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", 3.0),
        ("python_import", r"(?m)^\s*(?:from\s+\S+\s+)?import\s+\S+", 1.75),
        ("python_indented_block", r"(?m)^\s{4,}[A-Za-z_][^{}\n;]*$", 1.25),
        ("python_self", r"\bself\.", 1.25),
    ),
    "java": (
        ("java_package", r"(?m)^\s*package\s+[a-zA-Z_][\w.]*\s*;", 3.0),
        ("java_import", r"(?m)^\s*import\s+(?:java|javax|org)\.", 2.5),
        ("java_public_class", r"\bpublic\s+(?:final\s+)?class\s+\w+", 2.0),
        ("java_throws", r"\bthrows\s+[A-Z][A-Za-z0-9_]*", 1.5),
    ),
    "csharp": (
        ("csharp_using", r"(?m)^\s*using\s+[A-Z][A-Za-z0-9_.]*\s*;", 3.0),
        ("csharp_namespace", r"(?m)^\s*namespace\s+[A-Z][A-Za-z0-9_.]*", 3.0),
        ("csharp_public_class", r"\bpublic\s+(?:sealed\s+)?class\s+\w+", 2.0),
        ("csharp_property", r"\{\s*get;\s*(?:private\s+)?set;\s*\}", 1.75),
    ),
    "go": (
        ("go_package", r"(?m)^\s*package\s+[A-Za-z_][A-Za-z0-9_]*\s*$", 3.0),
        ("go_func", r"(?m)^\s*func\s+(?:\([^)]+\)\s*)?[A-Za-z_]\w*\s*\(", 3.0),
        ("go_short_assign", r":=", 2.0),
        ("go_defer", r"\bdefer\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", 1.5),
    ),
    "rust": (
        ("rust_fn", r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+\w+", 3.0),
        ("rust_impl", r"(?m)^\s*impl(?:\s*<[^>]+>)?\s+", 3.0),
        ("rust_use_crate", r"\buse\s+crate::", 3.0),
        ("rust_result", r"\bResult\s*<[^>]+>", 2.0),
        ("rust_let_mut", r"\blet\s+mut\s+[A-Za-z_][A-Za-z0-9_]*", 1.5),
    ),
    "typescript": (
        ("typescript_interface", r"(?m)^\s*export\s+interface\s+\w+", 3.0),
        ("typescript_type_alias", r"(?m)^\s*(?:export\s+)?type\s+\w+\s*=", 2.5),
        ("typescript_annotation", r"\b(?:const|let|var)\s+\w+\s*:\s*[A-Za-z_]", 2.0),
        ("typescript_access_modifier", r"\b(?:public|private|protected)\s+\w+", 1.5),
    ),
    "javascript": (
        ("javascript_import", r"(?m)^\s*import\s+.+\s+from\s+['\"][^'\"]+['\"]", 2.0),
        ("javascript_export", r"(?m)^\s*export\s+(?:default\s+)?(?:class|function|const)\b", 2.0),
        ("javascript_const", r"\bconst\s+[A-Za-z_$][\w$]*\s*=", 1.25),
        ("javascript_arrow", r"=>", 1.25),
    ),
}


def _literal_present_in_context(literal: str, context: str) -> bool:
    if not literal or not context:
        return False
    if IDENTIFIER_RE.fullmatch(literal):
        return bool(re.search(rf"\b{re.escape(literal)}\b", context))
    return literal in context


def _dialect_contrast_literals(
    dialect: str,
    *,
    context_text: str = "",
    contextual_filter: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    literals: list[str] = []
    families: list[str] = []
    for family, family_literals in DIALECT_CONTRAST_LOCKS.get(dialect, ()):
        kept = [
            literal
            for literal in family_literals
            if not (
                contextual_filter
                and dialect != "c11"
                and _literal_present_in_context(literal, context_text)
            )
        ]
        if kept:
            families.append(family)
            literals.extend(kept)
    return tuple(dict.fromkeys(literals)), tuple(dict.fromkeys(families))


def _dialect_lock_spec(
    dialect: str,
    extension: str,
    *,
    inference_mode: str = "extension",
    confidence: float = 1.0,
    signals: tuple[str, ...] = (),
    context_text: str = "",
    contextual_filter: bool = False,
) -> DialectLockSpec:
    forbidden_literals, contrast_families = _dialect_contrast_literals(
        dialect,
        context_text=context_text,
        contextual_filter=contextual_filter,
    )
    return DialectLockSpec(
        dialect=dialect,
        extension=extension,
        forbidden_literals=forbidden_literals,
        enforce_c11_case=dialect == "c11",
        replace_region_only=dialect != "c11",
        inference_mode=inference_mode,
        confidence=round(float(confidence), 3),
        signals=signals,
        contrast_families_banned=contrast_families,
    )


DIALECT_LOCKS_BY_EXTENSION: dict[str, DialectLockSpec] = {
    extension: _dialect_lock_spec(
        dialect,
        extension,
        confidence=0.8 if extension in AMBIGUOUS_DIALECT_EXTENSIONS else 0.9,
        signals=(f"extension:{extension}",),
    )
    for extension, dialect in DIALECT_BY_EXTENSION.items()
}


class Phase3Unavailable(RuntimeError):
    """Raised when the runner cannot honestly execute Layer-14 KV-direct."""


@dataclass
class Phase3Prediction:
    text: str
    ttft_ms: float | None
    metadata: dict[str, Any]


class KVDecodeSafetyTriggered(RuntimeError):
    """Raised when the guarded KV-direct decode must fail closed."""


@dataclass(frozen=True)
class IdentifierContinuationToken:
    token_id: int
    token_text: str
    leading_identifier: str
    leading_open: bool


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


class LogitLockedDiffProcessor:
    """Force a strict diff prefix and ban identifiers absent from authority."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        safe_token_ids: set[int],
        forced_prefix_ids: list[int],
        authority_identifiers: set[str] | None = None,
        banned_token_ids: set[int] | None = None,
        pre_replace_token_ids: set[int] | None = None,
        dialect_token_ids: set[int] | None = None,
        dialect_token_ids_by_path: dict[str, set[int]] | None = None,
        style_token_ids: set[int] | None = None,
        protocol_file_paths: set[str] | None = None,
        dialect_replace_region_only: bool = True,
        forced_token_count: int = 5,
        lsp_bias: float = 5.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.safe_token_ids = {int(token_id) for token_id in safe_token_ids}
        self.forced_prefix_ids = [int(token_id) for token_id in forced_prefix_ids]
        self.lsp_bias = float(lsp_bias)
        self.lsp_bias_token_ids = set(self.safe_token_ids)
        self.lsp_bias_tensor_cache: dict[tuple[str, str, int], Any] = {}
        self.lsp_bias_apply_count = 0
        self.lsp_bias_token_count = len(self.lsp_bias_token_ids)
        self.banned_token_ids = {int(token_id) for token_id in (banned_token_ids or set())}
        self.pre_replace_token_ids = {
            int(token_id) for token_id in (pre_replace_token_ids or set())
        }
        self.dialect_token_ids = {
            int(token_id) for token_id in (dialect_token_ids or set())
        }
        self.dialect_token_ids_by_path = {
            str(path): {int(token_id) for token_id in token_ids}
            for path, token_ids in (dialect_token_ids_by_path or {}).items()
        }
        self.active_dialect_path = ""
        self.dialect_path_switch_count = 0
        self.dialect_path_step_count = 0
        self.dialect_path_observed: dict[str, int] = {}
        self.style_token_ids = {
            int(token_id) for token_id in (style_token_ids or set())
        }
        self.protocol_file_paths = self._normalize_protocol_file_paths(
            protocol_file_paths or set(self.dialect_token_ids_by_path)
        )
        self.replace_protocol_literals = self._build_replace_protocol_literals()
        self.replace_protocol_max_literal_chars = max(
            (len(literal) for literal in self.replace_protocol_literals),
            default=0,
        )
        self.replace_protocol_candidate_token_ids: set[int] = set()
        self.replace_protocol_candidate_text_by_id: dict[int, str] = {}
        self.replace_protocol_cache: dict[str, set[int]] = {}
        self.replace_protocol_apply_count = 0
        self.replace_protocol_blocked_token_total = 0
        self.replace_protocol_last_blocked_token_count = 0
        self.replace_protocol_last_state = DIFF_STATE_OUTSIDE
        self.diff_state_observed: dict[str, int] = {}
        self.dialect_replace_region_only = bool(dialect_replace_region_only)
        self.forced_token_count = max(0, int(forced_token_count))
        self.unsafe_token_ids: set[int] = set()
        self.authority_identifiers = {
            str(identifier)
            for identifier in (authority_identifiers or set())
            if IDENTIFIER_RE.fullmatch(str(identifier))
        }
        self.common_identifier_allowlist = set(C_KEYWORD_IDENTIFIERS) | set(
            COMMON_LOCAL_IDENTIFIERS
        )
        self.identifier_whitelist = (
            set(self.authority_identifiers) | set(self.common_identifier_allowlist)
        )
        self.identifier_prefixes = identifier_prefixes(self.identifier_whitelist)
        self.identifier_lock_token_text: dict[int, str] = {}
        self.static_context_blocked_token_ids: set[int] = set()
        self.context_extension_tokens_by_first_char: dict[
            str,
            list[IdentifierContinuationToken],
        ] = {}
        self.context_extension_token_ids_by_first_char: dict[str, set[int]] = {}
        self.context_extension_token_ids: set[int] = set()
        self.context_boundary_token_ids: set[int] = set()
        self.context_prefix_cache: dict[str, set[int]] = {}
        self.context_whitelist_candidate_count = 0
        self.context_whitelist_step_count = 0
        self.context_whitelist_active_step_count = 0
        self.context_whitelist_skipped_not_replace_count = 0
        self.context_whitelist_cache_hit_count = 0
        self.context_whitelist_cache_miss_count = 0
        self.context_whitelist_inspected_candidate_count = 0
        self.context_whitelist_examples: dict[str, int] = {}
        self.hard_mask_apply_count = 0
        self.hard_mask_token_total = 0
        self.hard_mask_last_token_count = 0
        self.hard_mask_lsp_overlap_token_total = 0
        self.hard_mask_lsp_overlap_last_token_count = 0
        try:
            vocab = dict(tokenizer.get_vocab())
        except Exception:  # noqa: BLE001 - tokenizer implementations vary.
            vocab = {}
        for token_text, token_id in vocab.items():
            normalized_id = int(token_id)
            self.identifier_lock_token_text[normalized_id] = decode_single_token(
                tokenizer,
                normalized_id,
                str(token_text),
            )
            decoded_text = self.identifier_lock_token_text[normalized_id]
            normalized_text = normalize_token_text(decoded_text)
            if (
                normalized_id not in self.safe_token_ids
                and (
                    is_code_like_identifier(str(token_text))
                    or is_code_like_identifier(normalized_text)
                )
            ):
                self.unsafe_token_ids.add(normalized_id)
            if self._candidate_identifier_violation("", decoded_text):
                self.static_context_blocked_token_ids.add(normalized_id)
            continuation = self._identifier_continuation_token(
                normalized_id,
                decoded_text,
            )
            if continuation is not None:
                first_char = continuation.leading_identifier[0]
                self.context_extension_tokens_by_first_char.setdefault(
                    first_char,
                    [],
                ).append(continuation)
                self.context_extension_token_ids_by_first_char.setdefault(
                    first_char,
                    set(),
                ).add(normalized_id)
                self.context_extension_token_ids.add(normalized_id)
            elif self._identifier_boundary_token(decoded_text):
                self.context_boundary_token_ids.add(normalized_id)
            if self._token_can_complete_replace_protocol(decoded_text):
                self.replace_protocol_candidate_token_ids.add(normalized_id)
                self.replace_protocol_candidate_text_by_id[normalized_id] = decoded_text
        self._add_encoded_replace_protocol_candidates()
        self.always_blocked_token_ids = (
            set(self.unsafe_token_ids)
            | set(self.banned_token_ids)
        )
        self.dialect_style_token_ids = (
            set(self.dialect_token_ids)
            | set(self.style_token_ids)
        )

    def _replace_region_active(self, generated_text: str) -> bool:
        return self._diff_generation_state(generated_text) == DIFF_STATE_REPLACE_BLOCK

    def _diff_generation_state(self, generated_text: str) -> str:
        text = generated_text or ""
        search = None
        for match in re.finditer(r"<<<<\s*SEARCH", text, flags=re.IGNORECASE):
            search = match
        separator = None
        for match in re.finditer(r"====\s*REPLACE", text, flags=re.IGNORECASE):
            separator = match
        if separator is not None:
            if re.search(r">>>>", text[separator.end() :]) is None:
                return DIFF_STATE_REPLACE_BLOCK
            return DIFF_STATE_CLOSED
        if search is not None and re.search(r">>>>", text[search.end() :]) is None:
            return DIFF_STATE_SEARCH_BLOCK
        return DIFF_STATE_OUTSIDE

    def _normalize_protocol_file_paths(self, paths: set[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        for raw_path in paths:
            path = str(raw_path or "").strip()
            if not path:
                continue
            variants = {
                path,
                path.replace("\\", "/"),
                path.replace("/", "\\"),
            }
            basename = path.replace("\\", "/").rsplit("/", 1)[-1]
            if basename:
                variants.add(basename)
            for variant in variants:
                if variant and variant not in normalized:
                    normalized.append(variant)
        return tuple(normalized)

    def _build_replace_protocol_literals(self) -> tuple[str, ...]:
        literals = [*REPLACE_BLOCK_PROTOCOL_MARKERS, *self.protocol_file_paths]
        return tuple(dict.fromkeys(literal for literal in literals if literal))

    def _add_encoded_replace_protocol_candidates(self) -> None:
        for literal in self.replace_protocol_literals:
            for variant in (literal, f"\n{literal}", f" {literal}"):
                try:
                    token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                except Exception:  # noqa: BLE001 - tokenizer edge cases are non-fatal.
                    continue
                for token_id in token_ids:
                    normalized_id = int(token_id)
                    self.replace_protocol_candidate_token_ids.add(normalized_id)
                    decoded_text = self.identifier_lock_token_text.get(normalized_id)
                    if decoded_text is None:
                        decoded_text = decode_single_token(
                            self.tokenizer,
                            normalized_id,
                            "",
                        )
                    self.replace_protocol_candidate_text_by_id.setdefault(
                        normalized_id,
                        decoded_text,
                    )

    def _token_can_complete_replace_protocol(self, token_text: str) -> bool:
        if not token_text:
            return False
        for literal in self.replace_protocol_literals:
            if literal in token_text:
                return True
            max_prefix = min(len(literal), len(token_text))
            for prefix_len in range(1, max_prefix + 1):
                if literal.startswith(token_text[-prefix_len:]):
                    return True
        return False

    def _replace_protocol_blocked_ids(self, generated_text: str) -> set[int]:
        if not self.replace_protocol_candidate_token_ids:
            return set()
        replacement_tail = self._current_replacement_body(generated_text)[
            -self.replace_protocol_max_literal_chars :
        ]
        cached = self.replace_protocol_cache.get(replacement_tail)
        if cached is not None:
            return set(cached)
        blocked: set[int] = set()
        for token_id in self.replace_protocol_candidate_token_ids:
            token_text = self.replace_protocol_candidate_text_by_id.get(int(token_id), "")
            if not token_text:
                continue
            combined = f"{replacement_tail}{token_text}"
            completed_literals = [
                literal for literal in self.replace_protocol_literals if literal in combined
            ]
            if (
                completed_literals == [">>>>"]
                and self._replace_close_transition_allowed(combined)
            ):
                continue
            if completed_literals:
                blocked.add(int(token_id))
        self.replace_protocol_cache[replacement_tail] = set(blocked)
        return blocked

    def _replace_close_transition_allowed(self, replacement_body_with_token: str) -> bool:
        close_index = replacement_body_with_token.find(">>>>")
        if close_index < 0:
            return False
        body_before_close = replacement_body_with_token[:close_index]
        if not body_before_close.strip():
            return False
        close_line_prefix = body_before_close.rsplit("\n", 1)[-1]
        return not close_line_prefix.strip()

    def _current_replacement_body(self, generated_text: str) -> str:
        text = generated_text or ""
        separator = None
        for match in re.finditer(r"====\s*REPLACE", text, flags=re.IGNORECASE):
            separator = match
        if separator is None:
            return ""
        body = text[separator.end() :]
        close = re.search(r">>>>", body)
        return body[: close.start()] if close is not None else body

    def _identifier_continuation_token(
        self,
        token_id: int,
        token_text: str,
    ) -> IdentifierContinuationToken | None:
        if not token_text:
            return None
        match = re.match(r"[A-Za-z0-9_]+", token_text)
        if match is None:
            return None
        leading_identifier = match.group(0)
        leading_open = (
            match.end() == len(token_text)
            and bool(IDENTIFIER_CHAR_RE.fullmatch(token_text[-1]))
        )
        return IdentifierContinuationToken(
            token_id=int(token_id),
            token_text=token_text,
            leading_identifier=leading_identifier,
            leading_open=bool(leading_open),
        )

    def _identifier_boundary_token(self, token_text: str) -> bool:
        if not token_text:
            return False
        return not bool(IDENTIFIER_CHAR_RE.fullmatch(token_text[0]))

    def __call__(self, scores: Any, *, step: int, generated_text: str = "") -> Any:
        torch = getattr(scores, "new_full", None)
        if torch is None:
            return scores
        if step < self.forced_token_count and step < len(self.forced_prefix_ids):
            forced_id = int(self.forced_prefix_ids[step])
            locked = scores.new_full(scores.shape, float("-inf"))
            if 0 <= forced_id < int(scores.shape[-1]):
                locked[..., forced_id] = 0.0
                return locked
        scores = self._apply_lsp_bias(scores)
        diff_state = self._diff_generation_state(generated_text)
        self.replace_protocol_last_state = diff_state
        self.diff_state_observed[diff_state] = self.diff_state_observed.get(diff_state, 0) + 1
        replace_region_active = diff_state == DIFF_STATE_REPLACE_BLOCK
        blocked_ids = set(self.always_blocked_token_ids)
        if not replace_region_active:
            blocked_ids |= self.pre_replace_token_ids
        else:
            protocol_blocked_ids = self._replace_protocol_blocked_ids(generated_text)
            if protocol_blocked_ids:
                self.replace_protocol_apply_count += 1
                self.replace_protocol_blocked_token_total += len(protocol_blocked_ids)
                self.replace_protocol_last_blocked_token_count = len(protocol_blocked_ids)
                blocked_ids |= protocol_blocked_ids
        if not self.dialect_replace_region_only or replace_region_active:
            blocked_ids |= self._active_dialect_style_token_ids(generated_text)
        if (
            self.identifier_whitelist
            and step >= self.forced_token_count
            and replace_region_active
        ):
            context_blocked_ids = self._context_whitelist_blocked_ids(generated_text)
            if context_blocked_ids:
                self.context_whitelist_step_count += 1
                self.context_whitelist_candidate_count += len(context_blocked_ids)
                blocked_ids |= context_blocked_ids
        elif self.identifier_whitelist and step >= self.forced_token_count:
            self.context_whitelist_skipped_not_replace_count += 1
        if blocked_ids:
            vocab_size = int(scores.shape[-1])
            unsafe_ids = [
                token_id for token_id in blocked_ids if 0 <= token_id < vocab_size
            ]
            if unsafe_ids:
                scores = scores.clone()
                scores[..., unsafe_ids] = float("-inf")
                overlap_count = len(set(unsafe_ids) & self.lsp_bias_token_ids)
                self.hard_mask_apply_count += 1
                self.hard_mask_token_total += len(unsafe_ids)
                self.hard_mask_last_token_count = len(unsafe_ids)
                self.hard_mask_lsp_overlap_token_total += overlap_count
                self.hard_mask_lsp_overlap_last_token_count = overlap_count
        return scores

    def _active_dialect_style_token_ids(self, generated_text: str) -> set[int]:
        active_ids = set(self.dialect_style_token_ids)
        if not self.dialect_token_ids_by_path:
            return active_ids
        active_path = self._active_generated_path(generated_text)
        if active_path:
            self.dialect_path_step_count += 1
            self.dialect_path_observed[active_path] = (
                self.dialect_path_observed.get(active_path, 0) + 1
            )
            if active_path != self.active_dialect_path:
                self.active_dialect_path = active_path
                self.dialect_path_switch_count += 1
            path_ids = self.dialect_token_ids_by_path.get(active_path)
            if path_ids is not None:
                active_ids = set(path_ids) | set(self.style_token_ids)
        return active_ids

    def _active_generated_path(self, generated_text: str) -> str:
        known_paths = set(self.dialect_token_ids_by_path)
        if not known_paths:
            return ""
        for raw_line in reversed((generated_text or "").splitlines()):
            line = raw_line.strip().strip("`'\"")
            if not line or line.startswith(("<<<<", "====", ">>>>")):
                continue
            normalized = line.replace("\\", "/")
            if normalized in known_paths:
                return normalized
            basename_matches = [
                path for path in known_paths if path.rsplit("/", 1)[-1] == normalized
            ]
            if len(basename_matches) == 1:
                return basename_matches[0]
            if normalized:
                break
        return self.active_dialect_path

    def _apply_lsp_bias(self, scores: Any) -> Any:
        if self.lsp_bias <= 0.0 or not self.lsp_bias_token_ids:
            return scores
        try:
            vocab_size = int(scores.shape[-1])
            device_key = str(getattr(scores, "device", "cpu"))
            dtype_key = str(getattr(scores, "dtype", ""))
            cache_key = (device_key, dtype_key, vocab_size)
            bias = self.lsp_bias_tensor_cache.get(cache_key)
            if bias is None:
                bias = scores.new_zeros((vocab_size,))
                valid_ids = [
                    token_id
                    for token_id in self.lsp_bias_token_ids
                    if 0 <= token_id < vocab_size
                ]
                if valid_ids:
                    bias[valid_ids] = self.lsp_bias
                self.lsp_bias_tensor_cache[cache_key] = bias
            if len(getattr(scores, "shape", ())) <= 1:
                biased = scores + bias
            else:
                view_shape = [1] * (len(scores.shape) - 1) + [vocab_size]
                biased = scores + bias.view(*view_shape)
            self.lsp_bias_apply_count += 1
            return biased
        except Exception:  # noqa: BLE001 - guidance must never break sampling.
            return scores

    def _context_whitelist_blocked_ids(self, generated_text: str) -> set[int]:
        self.context_whitelist_active_step_count += 1
        prefix_match = IDENTIFIER_TAIL_RE.search(generated_text or "")
        prefix = prefix_match.group(0) if prefix_match else ""
        if not prefix:
            blocked_ids = set(self.static_context_blocked_token_ids)
            self._record_context_examples(blocked_ids)
            return blocked_ids
        cached = self.context_prefix_cache.get(prefix)
        if cached is not None:
            self.context_whitelist_cache_hit_count += 1
            blocked_ids = set(cached)
            self._record_context_examples(blocked_ids)
            return blocked_ids
        self.context_whitelist_cache_miss_count += 1
        blocked_ids = set(self.static_context_blocked_token_ids) - set(
            self.context_extension_token_ids
        )
        if not self._identifier_allowed(prefix, prefix_ok=False):
            blocked_ids.update(self.context_boundary_token_ids)
        allowed_next_chars = self._allowed_next_identifier_chars(prefix)
        for first_char, token_ids in self.context_extension_token_ids_by_first_char.items():
            if first_char not in allowed_next_chars:
                blocked_ids.update(token_ids)
        for first_char in allowed_next_chars:
            for token in self.context_extension_tokens_by_first_char.get(first_char, []):
                self.context_whitelist_inspected_candidate_count += 1
                if not self._leading_continuation_allowed(prefix, token):
                    blocked_ids.add(int(token.token_id))
        self.context_prefix_cache[prefix] = set(blocked_ids)
        self._record_context_examples(blocked_ids)
        return blocked_ids

    def _record_context_examples(self, blocked_ids: set[int]) -> None:
        if len(self.context_whitelist_examples) >= 12:
            return
        for token_id in sorted(blocked_ids):
            token_text = self.identifier_lock_token_text.get(int(token_id), "")
            example = normalize_token_text(token_text) or token_text
            if not example:
                continue
            self.context_whitelist_examples[example] = (
                self.context_whitelist_examples.get(example, 0) + 1
            )
            if len(self.context_whitelist_examples) >= 12:
                break

    def _allowed_next_identifier_chars(self, prefix: str) -> set[str]:
        short_plain_prefix = len(prefix) < 2 and "_" not in prefix
        if prefix not in self.identifier_prefixes:
            if short_plain_prefix:
                return set(self.context_extension_tokens_by_first_char)
            return set()
        prefix_len = len(prefix)
        next_chars = {
            identifier[prefix_len]
            for identifier in self.identifier_whitelist
            if len(identifier) > prefix_len and identifier.startswith(prefix)
        }
        if short_plain_prefix:
            next_chars.update(self.context_extension_tokens_by_first_char)
        return next_chars

    def _leading_continuation_allowed(
        self,
        prefix: str,
        token: IdentifierContinuationToken,
    ) -> bool:
        identifier = f"{prefix}{token.leading_identifier}"
        return self._identifier_allowed(identifier, prefix_ok=token.leading_open)

    def _candidate_identifier_violation(
        self,
        generated_text: str,
        token_text: str,
    ) -> bool:
        prefix_match = IDENTIFIER_TAIL_RE.search(generated_text or "")
        prefix = prefix_match.group(0) if prefix_match else ""
        composite = f"{prefix}{token_text}"
        added_start = len(prefix)
        for match in IDENTIFIER_RE.finditer(composite):
            if match.end() <= added_start:
                continue
            identifier = match.group(0)
            open_fragment = (
                match.end() == len(composite)
                and bool(composite)
                and bool(IDENTIFIER_CHAR_RE.fullmatch(composite[-1]))
            )
            if self._identifier_allowed(identifier, prefix_ok=open_fragment):
                continue
            return True
        return False

    def _identifier_allowed(self, identifier: str, *, prefix_ok: bool) -> bool:
        if identifier in self.identifier_whitelist:
            return True
        if len(identifier) <= 2 and "_" not in identifier:
            return True
        if prefix_ok and identifier in self.identifier_prefixes:
            return True
        return False

    def context_whitelist_metadata(self) -> dict[str, Any]:
        return {
            "active": bool(self.identifier_whitelist),
            "mode": "hard_identifier_whitelist",
            "lsp_bias": {
                "active": bool(self.lsp_bias > 0.0 and self.lsp_bias_token_ids),
                "bias": float(self.lsp_bias),
                "token_id_count": int(self.lsp_bias_token_count),
                "apply_count": int(self.lsp_bias_apply_count),
                "order": "raw_logits_plus_lsp_bias_then_hard_mask",
            },
            "hard_mask": {
                "active": bool(
                    self.always_blocked_token_ids
                    or self.pre_replace_token_ids
                    or self.dialect_style_token_ids
                    or self.dialect_token_ids_by_path
                    or self.identifier_whitelist
                ),
                "order": "after_lsp_bias",
                "value": "-inf",
                "apply_count": int(self.hard_mask_apply_count),
                "masked_token_total": int(self.hard_mask_token_total),
                "last_masked_token_count": int(self.hard_mask_last_token_count),
                "lsp_overlap_masked_token_total": int(
                    self.hard_mask_lsp_overlap_token_total
                ),
                "lsp_overlap_last_masked_token_count": int(
                    self.hard_mask_lsp_overlap_last_token_count
                ),
                "always_blocked_token_count": len(self.always_blocked_token_ids),
                "dialect_style_token_count": len(self.dialect_style_token_ids),
            },
            "dynamic_dialect_switch": {
                "active": bool(self.dialect_token_ids_by_path),
                "active_path": self.active_dialect_path,
                "path_count": len(self.dialect_token_ids_by_path),
                "switch_count": int(self.dialect_path_switch_count),
                "path_step_count": int(self.dialect_path_step_count),
                "observed_paths": dict(self.dialect_path_observed),
                "mode": "generated_file_path_line_recomputes_forbidden_tokens",
            },
            "authority_identifier_count": len(self.authority_identifiers),
            "allowed_identifier_count": len(self.identifier_whitelist),
            "prefix_count": len(self.identifier_prefixes),
            "replace_region_only": True,
                "pre_replace_token_id_count": len(self.pre_replace_token_ids),
                "static_context_blocked_token_count": len(
                    self.static_context_blocked_token_ids
                ),
                "state_aware_protocol_mask": {
                    "active": bool(self.replace_protocol_literals),
                    "state": DIFF_STATE_REPLACE_BLOCK,
                    "last_state": self.replace_protocol_last_state,
                    "observed_states": dict(self.diff_state_observed),
                    "marker_count": len(REPLACE_BLOCK_PROTOCOL_MARKERS),
                    "file_path_count": len(self.protocol_file_paths),
                    "literal_count": len(self.replace_protocol_literals),
                    "candidate_token_count": len(
                        self.replace_protocol_candidate_token_ids
                    ),
                    "apply_count": int(self.replace_protocol_apply_count),
                    "blocked_token_total": int(
                        self.replace_protocol_blocked_token_total
                    ),
                    "last_blocked_token_count": int(
                        self.replace_protocol_last_blocked_token_count
                    ),
                    "close_transition": "line_start_after_nonempty_replacement_allowed",
                    "value": "-inf",
                },
                "extension_token_count": len(self.context_extension_token_ids),
                "extension_bucket_count": len(self.context_extension_tokens_by_first_char),
            "boundary_token_count": len(self.context_boundary_token_ids),
            "prefix_cache_entry_count": len(self.context_prefix_cache),
            "cache_hit_count": int(self.context_whitelist_cache_hit_count),
            "cache_miss_count": int(self.context_whitelist_cache_miss_count),
            "active_step_count": int(self.context_whitelist_active_step_count),
            "skipped_not_replace_count": int(
                self.context_whitelist_skipped_not_replace_count
            ),
            "inspected_candidate_count": int(
                self.context_whitelist_inspected_candidate_count
            ),
            "intervention_count": int(self.context_whitelist_candidate_count),
            "intervention_step_count": int(self.context_whitelist_step_count),
            "examples": [
                {"token": token, "count": count}
                for token, count in sorted(
                    self.context_whitelist_examples.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
        }


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
        help=(
            "Nominal replacement-payload token budget for the KV-direct "
            "LoCoBench predictor. The runtime cap also includes "
            "--kv-replacement-spillover-tokens."
        ),
    )
    parser.add_argument(
        "--kv-replacement-spillover-tokens",
        type=int,
        default=DEFAULT_KV_REPLACEMENT_SPILLOVER_TOKENS,
        help=(
            "Extra new-token cushion beyond --kv-max-new-tokens so decoding "
            "does not stop on an open identifier or near-complete diff."
        ),
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
        help="Wall-clock safety timeout for one KV-direct decode segment. Use 0 to disable.",
    )
    parser.add_argument(
        "--kv-decode-stall-seconds",
        type=int,
        default=DEFAULT_KV_DECODE_STALL_SECONDS,
        help=(
            "Force a close marker if decoded text makes no progress for this "
            "many seconds within one KV-direct segment. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--kv-decode-heartbeat-seconds",
        type=int,
        default=DEFAULT_KV_DECODE_HEARTBEAT_SECONDS,
        help="Emit KV-direct decode progress telemetry at this interval. Use 0 to disable.",
    )
    parser.add_argument(
        "--decoder-dialect-steering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt into decoder-informed Layer-14 dialect steering when raw "
            "logits show high-confidence temptation for a banned dialect family."
        ),
    )
    parser.add_argument(
        "--decoder-dialect-steering-top-k",
        type=int,
        default=DEFAULT_DECODER_DIALECT_STEERING_TOP_K,
        help="Raw-logit rank cutoff for triggering dialect steering.",
    )
    parser.add_argument(
        "--decoder-dialect-steering-margin",
        type=float,
        default=DEFAULT_DECODER_DIALECT_STEERING_MARGIN,
        help=(
            "Minimum raw-logit margin versus the chosen token for top-k banned "
            "dialect tokens. Non-negative margins always trigger."
        ),
    )
    parser.add_argument(
        "--decoder-dialect-steering-decay",
        type=float,
        default=DEFAULT_DECODER_DIALECT_STEERING_DECAY,
        help="Per-token alpha decay applied before any new temptation spike.",
    )
    parser.add_argument(
        "--decoder-dialect-steering-floor",
        type=float,
        default=DEFAULT_DECODER_DIALECT_STEERING_FLOOR,
        help="Alpha values below this floor are snapped back to zero.",
    )
    parser.add_argument(
        "--decoder-dialect-steering-fraction",
        type=float,
        default=DEFAULT_DECODER_DIALECT_STEERING_FRACTION,
        help="Fraction of current hidden-state RMS used for the steering delta.",
    )
    parser.add_argument(
        "--decoder-dialect-steering-max-delta-rms",
        type=float,
        default=DEFAULT_DECODER_DIALECT_STEERING_MAX_DELTA_RMS,
        help="Maximum steering delta RMS as a fraction of current hidden-state RMS.",
    )
    parser.add_argument(
        "--decoder-dialect-steering-max-events",
        type=int,
        default=DEFAULT_DECODER_DIALECT_STEERING_MAX_EVENTS,
        help="Maximum detailed steering events retained in benchmark telemetry.",
    )
    parser.add_argument(
        "--require-phase3-apollo-store",
        action="store_true",
        help=(
            "Fail KV-direct instead of recapturing if the selected case lacks a "
            "ready Apollo residual store."
        ),
    )
    parser.add_argument(
        "--prediction-mode",
        choices=("kv_direct", "row_prediction", "empty", "gold_patch"),
        default=DEFAULT_PREDICTION_MODE,
        help="Local fallback mode when --prediction-command is omitted.",
    )
    add_router_backend_argument(parser)
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


def semantic_vector_np(text: str, *, dim: int, np: Any) -> Any:
    vec = np.zeros(int(dim), dtype=np.float32)
    for term in TERM_RE.findall(str(text).lower()):
        bucket = stable_hash_int(term) % int(dim)
        sign = 1.0 if stable_hash_int(f"{term}:sign") % 2 == 0 else -1.0
        vec[bucket] += sign
        if len(term) >= 5:
            vec[(bucket + len(term)) % int(dim)] += 0.35 * sign
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    return vec / norm


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(left * right for left, right in zip(a, b))
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _line_spans_for_tokens(
    content: str,
    *,
    tokens_per_window: int,
    overlap_tokens: int = 0,
) -> list[tuple[int, int, int, int, str]]:
    lines = str(content).splitlines()
    if not lines:
        return []
    window_size = max(1, int(tokens_per_window))
    overlap = max(0, min(int(overlap_tokens), window_size - 1))
    spans: list[tuple[int, int, int, int, str]] = []
    current_lines: list[str] = []
    start_line = 1
    start_token = 0
    token_count = 0
    for line_number, line in enumerate(lines, start=1):
        line_tokens = max(1, len(str(line).split()))
        if current_lines and token_count + line_tokens > window_size:
            spans.append(
                (
                    int(start_line),
                    int(line_number - 1),
                    int(start_token),
                    int(start_token + token_count),
                    "\n".join(current_lines),
                )
            )
            if overlap <= 0:
                current_lines = []
                start_line = line_number
                start_token += token_count
                token_count = 0
            else:
                kept_lines: list[str] = []
                kept_tokens = 0
                kept_start_line = line_number
                for kept_line_number, kept_line in reversed(
                    list(enumerate(current_lines, start=start_line))
                ):
                    kept_line_tokens = max(1, len(str(kept_line).split()))
                    if kept_tokens + kept_line_tokens > overlap:
                        break
                    kept_lines.insert(0, kept_line)
                    kept_tokens += kept_line_tokens
                    kept_start_line = kept_line_number
                current_lines = kept_lines
                start_line = int(kept_start_line)
                start_token = max(0, start_token + token_count - kept_tokens)
                token_count = kept_tokens
        current_lines.append(line)
        token_count += line_tokens
    if current_lines:
        spans.append(
            (
                int(start_line),
                int(len(lines)),
                int(start_token),
                int(start_token + token_count),
                "\n".join(current_lines),
            )
        )
    return spans


def dependency_windows(
    files: dict[str, str],
    *,
    tokens_per_window: int,
    overlap_tokens: int = 0,
) -> list[DependencyWindow]:
    windows: list[DependencyWindow] = []
    for path, content in files.items():
        for start_line, end_line, start_token, end_token, text in _line_spans_for_tokens(
            content,
            tokens_per_window=tokens_per_window,
            overlap_tokens=overlap_tokens,
        ):
            windows.append(
                DependencyWindow(
                    window_id=len(windows),
                    path=str(path),
                    text=text,
                    score=0.0,
                    start_line=int(start_line),
                    end_line=int(end_line),
                    start_token=int(start_token),
                    end_token=int(end_token),
                )
            )
    return windows


def known_path_mentions(text: str, files: dict[str, str]) -> set[str]:
    lowered = str(text or "").lower()
    mentions: set[str] = set()
    for path in files:
        normalized_path = str(path).replace("\\", "/")
        basename = normalized_path.rsplit("/", 1)[-1]
        stem = basename.rsplit(".", 1)[0]
        candidates = {normalized_path, basename, stem}
        if any(candidate and candidate.lower() in lowered for candidate in candidates):
            mentions.add(str(path))
    return mentions


def generated_target_paths(generated: str, files: dict[str, str]) -> list[str]:
    lines = [line.strip() for line in str(generated or "").splitlines()]
    ordered: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if not line or line.startswith(("<<<<", "====", ">>>>")):
            continue
        candidate = line.replace("\\", "/")
        if candidate in files and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
        elif candidate.rsplit("/", 1)[-1] in {
            path.rsplit("/", 1)[-1] for path in files
        }:
            for path in files:
                if path.rsplit("/", 1)[-1] == candidate.rsplit("/", 1)[-1] and path not in seen:
                    seen.add(path)
                    ordered.append(path)
                    break
        if ordered:
            break
    return ordered


def window_identifier_score(window: DependencyWindow, query_terms: set[str]) -> float:
    identifiers = {identifier.lower() for identifier in extract_identifiers(window.text)}
    return float(len(identifiers & query_terms))


def split_identifier_terms(identifier: str) -> list[str]:
    parts = re.findall(
        r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[0-9]+",
        str(identifier),
    )
    lowered = [part.lower() for part in parts if part]
    if lowered and len(lowered[0]) == 1 and len(lowered) > 1:
        lowered.append("".join(lowered[1:]))
    if len(lowered) > 1:
        lowered.append("".join(lowered))
        lowered.append("_".join(lowered))
        lowered.append("-".join(lowered))
    return lowered


def dependency_name_terms(*texts: str) -> set[str]:
    terms: set[str] = set()
    ordered_words: list[str] = []
    for text in texts:
        raw_terms = [term.lower() for term in TERM_RE.findall(str(text or ""))]
        ordered_words.extend(raw_terms)
        terms.update(raw_terms)
        for identifier in extract_identifiers(str(text or "")):
            split_terms = split_identifier_terms(identifier)
            terms.update(split_terms)
            for term in split_terms:
                for suffix in ("service", "manager", "gateway"):
                    if term.endswith(suffix) and len(term) > len(suffix):
                        terms.add(term[: -len(suffix)])
        for left, right in zip(raw_terms, raw_terms[1:]):
            terms.update(
                {
                    f"{left}{right}",
                    f"{left}_{right}",
                    f"{left}-{right}",
                }
            )
            if right in {"service", "manager", "gateway"}:
                terms.add(left)
                terms.add(f"i{left}")
                terms.add(f"{left}{right}")
                terms.add(f"{left}_{right}")
                terms.add(f"{left}-{right}")
    for term in list(terms):
        if term.startswith("i") and len(term) > 2:
            terms.add(term[1:])
    return {term for term in terms if len(term) >= 2}


def code_source_path_score(path: str) -> float:
    normalized = str(path).replace("\\", "/").lower()
    suffixes = (
        ".h",
        ".hpp",
        ".hh",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".py",
        ".pyi",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".java",
        ".cs",
        ".go",
        ".rs",
    )
    score = 0.0
    if normalized.endswith(suffixes):
        score += 12.0
    if any(
        marker in normalized
        for marker in (
            "services/",
            "/services/",
            "api-gateway/",
            "/api-gateway/",
            "libs/",
            "/libs/",
            "include/",
            "/include/",
            "src/",
            "/src/",
        )
    ):
        score += 8.0
    if any(
        marker in normalized
        for marker in ("docs/", "/docs/", ".puml", ".md", ".rst")
    ):
        score -= 6.0
    return score


def normalized_route_path_parts(path: str) -> tuple[str, ...]:
    return tuple(
        part.lower()
        for part in str(path or "").replace("\\", "/").split("/")
        if part and part not in {".", ".."}
    )


def route_path_depth_penalty(path: str) -> float:
    parts = normalized_route_path_parts(path)
    if not parts:
        return 0.0
    directory_depth = max(0, len(parts) - 1)
    return float(max(0, directory_depth - 3) * 0.45)


def route_domain_weight(path: str) -> float:
    normalized = str(path or "").replace("\\", "/").lower()
    parts = set(normalized_route_path_parts(normalized))
    suffix = Path(normalized).suffix.lower()
    weight = 1.0
    if suffix in DOMAIN_SOURCE_SUFFIXES:
        weight += 0.18
    if parts & DOMAIN_CORE_PATH_PARTS:
        weight += 0.32
    if any(part.endswith(("service", "handler", "controller", "repository")) for part in parts):
        weight += 0.18
    if parts & DOMAIN_PERIPHERAL_PATH_PARTS:
        weight -= 0.45
    if suffix in DOMAIN_PERIPHERAL_SUFFIXES:
        weight -= 0.35
    if any(part in {"test", "tests", "spec", "specs"} for part in parts):
        weight -= 0.12
    return max(0.25, min(1.8, float(weight)))


def route_domain_score(relevance: float, path: str) -> float:
    return (float(relevance) * route_domain_weight(path)) - route_path_depth_penalty(path)


def dependency_name_match_score(
    window: DependencyWindow,
    expanded_terms: set[str],
) -> float:
    if not expanded_terms:
        return 0.0
    path_terms = dependency_name_terms(window.path)
    text_terms = dependency_name_terms(window.text)
    path_overlap = expanded_terms & path_terms
    text_overlap = expanded_terms & text_terms
    return float(len(path_overlap) * 8.0 + len(text_overlap) * 3.0)


def build_loco_activation_route_index(
    windows: list[DependencyWindow],
    *,
    route_dir: Path | None,
    case_id: int | str,
    tokens_per_window: int,
    overlap_tokens: int,
    dim: int = 128,
) -> LoCoActivationRouteIndex | None:
    if not windows:
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    root = route_dir or (DEFAULT_JIT_DIR / "locobench_source_activation_routes")
    case_dir = root / f"case_{case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    route_path = case_dir / "activation_routes.npy"
    metadata_path = case_dir / "window_spans.json"
    routes = np.lib.format.open_memmap(
        route_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(windows), int(dim)),
    )
    for route_id, window in enumerate(windows):
        routes[route_id, :] = semantic_vector_np(window.text, dim=int(dim), np=np)
    routes.flush()
    metadata = {
        "mode": "locobench_source_window_hash_activation_routes",
        "activation_matrix_path": str(route_path),
        "window_count": len(windows),
        "dim": int(dim),
        "dtype": "float32",
        "window_tokens": int(tokens_per_window),
        "overlap_tokens": int(overlap_tokens),
        "stride_tokens": max(1, int(tokens_per_window) - int(overlap_tokens)),
        "windows": [
            {
                "window_id": int(window.window_id),
                "route_id": int(route_id),
                "path": window.path,
                "start_line": int(window.start_line),
                "end_line": int(window.end_line),
                "start_token": int(window.start_token),
                "end_token": int(window.end_token),
                "chars": len(window.text),
            }
            for route_id, window in enumerate(windows)
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return LoCoActivationRouteIndex(
        path=route_path,
        metadata_path=metadata_path,
        route_count=len(windows),
        dim=int(dim),
        dtype="float32",
        window_tokens=int(tokens_per_window),
        overlap_tokens=int(overlap_tokens),
        stride_tokens=max(1, int(tokens_per_window) - int(overlap_tokens)),
        routes=routes,
    )


def _activation_route_modules() -> tuple[Any, Any] | None:
    try:
        import numpy as np
    except ImportError:
        return None
    return np, np


def _normalized_np_vector(vector: Any, np: Any) -> Any:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector
    return vector / norm


def search_loco_activation_vector(
    activation_index: LoCoActivationRouteIndex,
    query_vector: Any,
    *,
    chunk_size: int = 1024,
) -> Any | None:
    modules = _activation_route_modules()
    if modules is None:
        return None
    np, _ = modules
    if activation_index.route_count <= 0:
        return np.zeros(0, dtype=np.float32)
    routes = activation_index.routes
    if routes is None:
        routes = np.lib.format.open_memmap(activation_index.path, mode="r")
    scores = np.empty(int(activation_index.route_count), dtype=np.float32)
    for offset in range(0, int(activation_index.route_count), int(chunk_size)):
        end = min(offset + int(chunk_size), int(activation_index.route_count))
        scores[offset:end] = routes[offset:end] @ query_vector
    return scores


def search_loco_activation_routes(
    activation_index: LoCoActivationRouteIndex,
    query_text: str,
    *,
    chunk_size: int = 1024,
) -> Any | None:
    modules = _activation_route_modules()
    if modules is None:
        return None
    np, _ = modules
    query_vector = semantic_vector_np(query_text, dim=activation_index.dim, np=np)
    return search_loco_activation_vector(
        activation_index,
        query_vector,
        chunk_size=chunk_size,
    )


def identifiers_from_windows(windows: list[DependencyWindow]) -> set[str]:
    identifiers: set[str] = set()
    for window in windows:
        identifiers.update(extract_identifiers(window.text))
        identifiers.update(extract_identifiers(window.path))
    return identifiers


def copy_dependency_window(
    window: DependencyWindow,
    *,
    score: float | None = None,
    lexical_score: float | None = None,
    activation_score: float | None = None,
    exact_scope_match: bool | None = None,
    route_source: str | None = None,
    recursive_map_window_id: int | None = None,
    recursive_pass: str | None = None,
) -> DependencyWindow:
    return DependencyWindow(
        window_id=window.window_id,
        path=window.path,
        text=window.text,
        score=window.score if score is None else float(score),
        start_line=window.start_line,
        end_line=window.end_line,
        start_token=window.start_token,
        end_token=window.end_token,
        activation_route_id=window.activation_route_id,
        lexical_score=window.lexical_score
        if lexical_score is None
        else float(lexical_score),
        activation_score=window.activation_score
        if activation_score is None
        else float(activation_score),
        exact_scope_match=window.exact_scope_match
        if exact_scope_match is None
        else bool(exact_scope_match),
        route_source=window.route_source if route_source is None else route_source,
        recursive_map_window_id=recursive_map_window_id,
        recursive_pass=window.recursive_pass if recursive_pass is None else recursive_pass,
    )


def central_route_dependency_windows(
    instruction: str,
    files: dict[str, str],
    *,
    max_k: int,
    tokens_per_window: int,
    overlap_tokens: int = 0,
    capability_mode: str = DEPENDENCY_SOURCE,
) -> list[DependencyWindow]:
    windows = dependency_windows(
        files,
        tokens_per_window=tokens_per_window,
        overlap_tokens=overlap_tokens,
    )
    if not windows:
        return []
    target_paths = known_path_mentions(instruction, files)
    identifiers = sorted(dependency_name_terms(instruction))
    route_windows = [
        {
            "window_id": str(window.window_id),
            "text": window.text,
            "source_path": window.path,
            "metadata": {
                "path": window.path,
                "start_line": int(window.start_line),
                "end_line": int(window.end_line),
                "start_token": int(window.start_token),
                "end_token": int(window.end_token),
            },
        }
        for window in windows
    ]
    plan = central_route_plan(
        query=instruction,
        capability_mode=capability_mode,
        windows=route_windows,
        path_hints=target_paths,
        identifiers=identifiers,
    )
    tiers = tier_by_window_id(plan)
    scores = candidate_scores_by_id(plan)
    by_window_id = {str(window.window_id): window for window in windows}
    route_limit = max(3, int(max_k)) if len(windows) >= 3 else max(1, int(max_k))
    selected: list[DependencyWindow] = []
    for window_id in scores:
        window = by_window_id.get(window_id)
        if window is None:
            continue
        tier = tiers.get(window_id, "UNTIERED").lower()
        capability_label = capability_mode.lower()
        selected.append(
            copy_dependency_window(
                window,
                score=float(scores[window_id]),
                lexical_score=float(scores[window_id]),
                activation_score=0.0,
                exact_scope_match=window.path in target_paths,
                route_source=f"central_router:{tier}:{capability_label}",
                recursive_map_window_id=None,
                recursive_pass=f"central_router_{tier}",
            )
        )
        if len(selected) >= route_limit:
            break
    return selected


def route_hot_windows(
    instruction: str,
    files: dict[str, str],
    *,
    max_k: int,
    tokens_per_window: int,
    overlap_tokens: int = 0,
    activation_route_dir: Path | None = None,
    case_id: int | str = "adhoc",
    routing_backend: str = "native",
    central_capability_mode: str = DEPENDENCY_SOURCE,
) -> list[DependencyWindow]:
    if is_central_backend(routing_backend):
        return central_route_dependency_windows(
            instruction,
            files,
            max_k=max_k,
            tokens_per_window=tokens_per_window,
            overlap_tokens=overlap_tokens,
            capability_mode=central_capability_mode,
        )
    windows = dependency_windows(
        files,
        tokens_per_window=tokens_per_window,
        overlap_tokens=overlap_tokens,
    )
    if not windows:
        return []
    query_terms = readout_terms(instruction)
    expanded_query_terms = dependency_name_terms(instruction)
    target_paths = known_path_mentions(instruction, files)
    activation_index = build_loco_activation_route_index(
        windows,
        route_dir=activation_route_dir,
        case_id=case_id,
        tokens_per_window=tokens_per_window,
        overlap_tokens=overlap_tokens,
    )
    query_vec = semantic_vector(instruction)
    modules = _activation_route_modules() if activation_index is not None else None
    np = modules[0] if modules is not None else None
    query_vec_np = (
        semantic_vector_np(instruction, dim=activation_index.dim, np=np)
        if activation_index is not None and np is not None
        else None
    )
    activation_scores = (
        search_loco_activation_vector(activation_index, query_vec_np)
        if activation_index is not None and query_vec_np is not None
        else None
    )
    scored: list[DependencyWindow] = []
    for route_id, window in enumerate(windows):
        lexical_overlap = len(query_terms & readout_terms(window.path, window.text))
        identifier_bonus = window_identifier_score(window, query_terms)
        dependency_bonus = dependency_name_match_score(window, expanded_query_terms)
        path_exact = window.path in target_paths
        lexical_score = float(lexical_overlap) + (identifier_bonus * 2.0) + dependency_bonus
        if path_exact:
            lexical_score += 100.0
        if activation_scores is not None:
            activation_score = float(activation_scores[int(route_id)])
            route_source = "exact_scope_activation" if path_exact else "activation_route"
        else:
            activation_score = cosine(query_vec, semantic_vector(window.text))
            route_source = "exact_scope_hash_cosine" if path_exact else "hash_cosine"
        relevance_score = lexical_score + max(0.0, activation_score) * 0.25
        hybrid_score = route_domain_score(relevance_score, window.path)
        scored.append(
            DependencyWindow(
                window_id=window.window_id,
                path=window.path,
                text=window.text,
                score=float(hybrid_score),
                start_line=window.start_line,
                end_line=window.end_line,
                start_token=window.start_token,
                end_token=window.end_token,
                activation_route_id=int(route_id) if activation_scores is not None else None,
                lexical_score=float(lexical_score),
                activation_score=float(activation_score),
                exact_scope_match=bool(path_exact),
                route_source=route_source,
                recursive_pass="single_pass",
            )
        )
    file_order = {path: index for index, path in enumerate(files)}
    def score_key(window: DependencyWindow) -> tuple[float, ...]:
        return (
            0 if window.path in target_paths else 1,
            -window.score,
            -window.lexical_score,
            -window.activation_score,
            file_order.get(window.path, 999_999),
            window.start_line,
            window.window_id,
        )
    if activation_index is None or activation_scores is None or np is None:
        scored.sort(key=score_key)
        return scored[: max(1, int(max_k))]

    pass1_best = max(
        scored,
        key=lambda window: (
            window.score,
            route_domain_weight(window.path),
            window.activation_score,
            1 if window.path in target_paths else 0,
            -file_order.get(window.path, 999_999),
            -window.start_line,
            -window.window_id,
        ),
    )
    routes = activation_index.routes
    if routes is None:
        routes = np.lib.format.open_memmap(activation_index.path, mode="r")
    best_route_id = int(pass1_best.activation_route_id or 0)
    best_vector = routes[best_route_id]
    blended_vector = _normalized_np_vector((query_vec_np * 0.4) + (best_vector * 0.6), np)
    expanded_hop_terms = dependency_name_terms(instruction, pass1_best.path, pass1_best.text)
    hop_scores = search_loco_activation_vector(
        activation_index,
        blended_vector,
    )
    if hop_scores is None:
        scored.sort(key=score_key)
        return scored[: max(1, int(max_k))]

    route_budget = max(int(max_k), 5)
    hop_k = max(4, route_budget - 1)
    by_window_id = {window.window_id: window for window in scored}
    pass1_map = copy_dependency_window(
        pass1_best,
        score=float(pass1_best.score),
        route_source=(
            "exact_scope_recursive_map"
            if pass1_best.exact_scope_match
            else "recursive_map"
        ),
        recursive_map_window_id=int(pass1_best.window_id),
        recursive_pass="pass1_map",
    )
    hop_candidates: list[DependencyWindow] = []
    for route_id in range(len(scored)):
        window = by_window_id.get(route_id)
        if window is None:
            continue
        hop_activation_score = float(hop_scores[int(route_id)])
        hop_dependency_bonus = dependency_name_match_score(window, expanded_hop_terms)
        hop_lexical_score = float(window.lexical_score + hop_dependency_bonus)
        hop_relevance_score = float(
            hop_lexical_score + max(0.0, hop_activation_score) * 0.25
        )
        hop_score = float(route_domain_score(hop_relevance_score, window.path))
        hop_candidates.append(
            copy_dependency_window(
                window,
                score=hop_score,
                lexical_score=hop_lexical_score,
                activation_score=hop_activation_score,
                route_source=(
                    "exact_scope_recursive_hop"
                    if window.exact_scope_match
                    else "recursive_hop"
                ),
                recursive_map_window_id=int(pass1_map.window_id),
                recursive_pass="pass2_hop",
            )
        )
    hop_candidates.sort(
        key=lambda window: (
            1 if window.path == pass1_map.path else 0,
            -dependency_name_match_score(window, expanded_hop_terms),
            -route_domain_weight(window.path),
            -window.activation_score,
            -window.lexical_score,
            0 if window.path in target_paths else 1,
            file_order.get(window.path, 999_999),
            window.start_line,
            window.window_id,
        )
    )
    selected: list[DependencyWindow] = [pass1_map]
    seen_ids = {int(pass1_map.window_id)}
    seen_paths = {pass1_map.path}
    for window in hop_candidates:
        if len(selected) >= hop_k + 1:
            break
        if int(window.window_id) in seen_ids:
            continue
        if window.path in seen_paths:
            continue
        selected.append(window)
        seen_ids.add(int(window.window_id))
        seen_paths.add(window.path)
    for window in hop_candidates:
        if len(selected) >= hop_k + 1:
            break
        if int(window.window_id) in seen_ids:
            continue
        selected.append(window)
        seen_ids.add(int(window.window_id))
    whitelist_new = len(identifiers_from_windows(selected) - identifiers_from_windows([pass1_map]))
    hopped_paths = [
        window.path
        for window in selected
        if window.path != pass1_map.path
    ]
    hopped_path = hopped_paths[0] if hopped_paths else pass1_map.path
    print(
        "PHASE 2 MULTI-HOP: "
        f"Found Map [{pass1_map.path}], Hopped to Code [{hopped_path}]. "
        f"Whitelisting [{whitelist_new}] new identifiers.",
        flush=True,
    )
    return selected


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


def identifier_prefixes(identifiers: set[str]) -> set[str]:
    prefixes: set[str] = set()
    for identifier in identifiers:
        if not IDENTIFIER_RE.fullmatch(str(identifier)):
            continue
        for length in range(1, len(identifier) + 1):
            prefixes.add(identifier[:length])
    return prefixes


def decode_single_token(tokenizer: Any, token_id: int, fallback: str) -> str:
    try:
        decoded = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    except Exception:  # noqa: BLE001 - tokenizer decode support varies.
        decoded = ""
    if decoded:
        return str(decoded)
    return str(fallback)


def _combined_dialect_context(
    source_text: str,
    nearby_texts: list[str] | tuple[str, ...] | None = None,
) -> str:
    texts = [str(source_text or "")]
    texts.extend(str(text or "") for text in (nearby_texts or ()) if text)
    return "\n".join(texts)


def _dialect_syntax_scores(
    context_text: str,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    scores: dict[str, float] = {}
    signals: dict[str, list[str]] = {}
    if not context_text.strip():
        return scores, signals
    for dialect, fingerprints in DIALECT_SYNTAX_FINGERPRINTS.items():
        for signal, pattern, weight in fingerprints:
            if re.search(pattern, context_text):
                scores[dialect] = scores.get(dialect, 0.0) + float(weight)
                signals.setdefault(dialect, []).append(signal)
    return scores, signals


def _dialect_inference_confidence(
    *,
    best_score: float,
    runner_up_score: float,
    has_extension: bool,
    has_syntax: bool,
    extension_matched: bool,
) -> float:
    margin = max(0.0, float(best_score) - float(runner_up_score))
    confidence = 0.45 + min(float(best_score), 10.0) * 0.035
    confidence += min(margin, 6.0) * 0.045
    if has_extension:
        confidence += 0.15 if extension_matched else 0.05
    if has_syntax:
        confidence += 0.08
    if margin < 1.0 and runner_up_score > 0:
        confidence -= 0.18
    return max(0.0, min(0.99, confidence))


def infer_dialect_lock_for_source(
    path: str,
    source_text: str,
    nearby_texts: list[str] | None = None,
) -> DialectLockSpec | None:
    """Infer a deterministic language lock from extension plus source syntax."""

    suffix = Path(str(path or "")).suffix.lower()
    extension_dialect = DIALECT_BY_EXTENSION.get(suffix)
    context_text = _combined_dialect_context(source_text, nearby_texts)
    scores, signals = _dialect_syntax_scores(context_text)
    if extension_dialect is not None:
        extension_weight = 2.5 if suffix in AMBIGUOUS_DIALECT_EXTENSIONS else 5.0
        scores[extension_dialect] = scores.get(extension_dialect, 0.0) + extension_weight
        signals.setdefault(extension_dialect, []).insert(0, f"extension:{suffix}")
    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_dialect, _best_score = ranked[0]
    if (
        extension_dialect is not None
        and best_dialect != extension_dialect
        and suffix not in AMBIGUOUS_DIALECT_EXTENSIONS
    ):
        best_dialect = extension_dialect
    elif (
        extension_dialect is not None
        and best_dialect != extension_dialect
        and suffix in AMBIGUOUS_DIALECT_EXTENSIONS
        and scores[best_dialect] - scores.get(extension_dialect, 0.0) < 1.0
    ):
        best_dialect = extension_dialect

    best_score = scores.get(best_dialect, 0.0)
    runner_up_score = max(
        (score for dialect, score in scores.items() if dialect != best_dialect),
        default=0.0,
    )
    chosen_signals = tuple(signals.get(best_dialect, ()))
    syntax_signals = tuple(
        signal for signal in chosen_signals if not signal.startswith("extension:")
    )
    if extension_dialect is None and (
        best_score < 3.0 or best_score - runner_up_score < 0.75
    ):
        return None

    confidence = _dialect_inference_confidence(
        best_score=best_score,
        runner_up_score=runner_up_score,
        has_extension=extension_dialect is not None,
        has_syntax=bool(syntax_signals),
        extension_matched=best_dialect == extension_dialect,
    )
    if extension_dialect is None and confidence < 0.58:
        return None

    if extension_dialect == best_dialect and syntax_signals:
        inference_mode = "extension_plus_syntax"
    elif extension_dialect == best_dialect:
        inference_mode = "extension"
    elif extension_dialect is not None:
        inference_mode = "syntax_overrode_ambiguous_extension"
    else:
        inference_mode = "syntax"

    return _dialect_lock_spec(
        best_dialect,
        suffix,
        inference_mode=inference_mode,
        confidence=confidence,
        signals=chosen_signals,
        context_text=context_text,
        contextual_filter=True,
    )


def dialect_lock_for_path(path: str) -> DialectLockSpec | None:
    suffix = Path(str(path or "")).suffix.lower()
    return DIALECT_LOCKS_BY_EXTENSION.get(suffix)


def dialect_lock_telemetry(spec: DialectLockSpec | None) -> dict[str, Any]:
    return {
        "active": spec is not None,
        "target_dialect": spec.dialect if spec else "",
        "extension": spec.extension if spec else "",
        "inference_mode": spec.inference_mode if spec else "none",
        "confidence": spec.confidence if spec else 0.0,
        "signals": list(spec.signals) if spec else [],
        "contrast_families_banned": (
            list(spec.contrast_families_banned) if spec else []
        ),
        "forbidden_literals": list(spec.forbidden_literals) if spec else [],
        "replace_region_only": spec.replace_region_only if spec is not None else True,
        "c11_case_enforced": bool(spec is not None and spec.enforce_c11_case),
    }


def normalize_token_text(token_text: str) -> str:
    normalized = str(token_text or "").strip()
    for marker in ("\u0120", "\u2581", "â–", "Ä "):
        while normalized.startswith(marker):
            normalized = normalized[len(marker):].lstrip()
    return normalized.strip()


def token_ids_for_dialect_lock(tokenizer: Any, spec: DialectLockSpec | None) -> set[int]:
    if spec is None:
        return set()
    forbidden = set(spec.forbidden_literals)
    token_ids: set[int] = set()
    try:
        vocab = dict(tokenizer.get_vocab())
    except Exception:  # noqa: BLE001 - tokenizer implementations vary.
        vocab = {}
    for token_text, token_id in vocab.items():
        if normalize_token_text(str(token_text)) in forbidden:
            token_ids.add(int(token_id))
    for literal in forbidden:
        variants = (literal, f" {literal}", f"\n{literal}", f"\t{literal}")
        for variant in variants:
            try:
                encoded = [
                    int(token_id)
                    for token_id in tokenizer.encode(variant, add_special_tokens=False)
                ]
            except Exception:  # noqa: BLE001 - tokenizer edge cases are non-fatal.
                continue
            if len(encoded) == 1:
                token_ids.update(encoded)
    return token_ids


def is_camel_case_identifier(identifier: str) -> bool:
    if not IDENTIFIER_RE.fullmatch(str(identifier or "")):
        return False
    if "_" in identifier or identifier.upper() == identifier:
        return False
    has_lower = any(char.islower() for char in identifier)
    has_upper = any(char.isupper() for char in identifier)
    if not (has_lower and has_upper):
        return False
    return bool(
        re.fullmatch(r"[A-Z][A-Za-z0-9]*", identifier)
        or re.fullmatch(r"[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*", identifier)
    )


def token_ids_for_c11_case_lock(tokenizer: Any, safe_identifiers: set[str]) -> set[int]:
    safe = set(safe_identifiers)
    token_ids: set[int] = set()
    try:
        vocab = dict(tokenizer.get_vocab())
    except Exception:  # noqa: BLE001 - tokenizer implementations vary.
        return token_ids
    for token_text, token_id in vocab.items():
        normalized = normalize_token_text(str(token_text))
        if normalized in safe:
            continue
        if is_camel_case_identifier(normalized):
            token_ids.add(int(token_id))
    return token_ids


def replace_payload_text(generated: str) -> str:
    blocks = list(SEARCH_REPLACE_RE.finditer(generated or ""))
    if not blocks:
        return generated or ""
    return "\n".join(match.group("new") for match in blocks)


def c11_case_style_violations(
    generated: str,
    safe_identifiers: set[str],
    spec: DialectLockSpec | None,
) -> list[str]:
    if spec is None or not spec.enforce_c11_case:
        return []
    safe = set(safe_identifiers)
    return [
        identifier
        for identifier in sorted(extract_identifiers(replace_payload_text(generated)))
        if identifier not in safe and is_camel_case_identifier(identifier)
    ]


def dialect_literal_violations(
    generated: str,
    spec: DialectLockSpec | None,
) -> list[str]:
    if spec is None:
        return []
    payload = replace_payload_text(generated)
    violations: list[str] = []
    for literal in spec.forbidden_literals:
        if IDENTIFIER_RE.fullmatch(literal):
            if re.search(rf"\b{re.escape(literal)}\b", payload):
                violations.append(literal)
        elif literal in payload:
            violations.append(literal)
    return sorted(set(violations))


def dependency_generation_bundle(
    hot_windows: list[DependencyWindow],
    instruction: str,
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    safe_identifiers: set[str] = set()
    for window in hot_windows:
        safe_identifiers.update(extract_identifiers(window.text))
        safe_identifiers.update(extract_identifiers(window.path))
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


def identifier_authority_from_span(path: str, span: str) -> dict[str, Any]:
    identifiers = sorted(extract_identifiers(span))
    path_identifiers = sorted(extract_identifiers(path))
    return {
        "path": path,
        "identifiers": identifiers,
        "path_identifiers": path_identifiers,
        "identifier_count": len(identifiers),
        "path_identifier_count": len(path_identifiers),
    }


def selected_source_identifier_authority(
    case: LoCoBenchCase,
    hot_windows: list[DependencyWindow],
    generated: str = "",
) -> tuple[str, str, dict[str, Any]]:
    path, span, span_meta = select_verbatim_readout_span(case, hot_windows, generated)
    authority = identifier_authority_from_span(path, span)
    return path, span, {**span_meta, "identifier_authority": authority}


def safe_token_ids_from_windows(
    tokenizer: Any,
    windows: list[DependencyWindow],
    *,
    extra_texts: tuple[str, ...] = (),
    extra_identifiers: set[str] | None = None,
) -> set[int]:
    identifiers: set[str] = set()
    safe_texts: list[str] = []
    for window in windows:
        identifiers.update(extract_identifiers(window.text))
        identifiers.update(extract_identifiers(window.path))
        safe_texts.extend([window.path, window.text])
    identifiers.update(extra_identifiers or set())
    safe_texts.extend(str(text) for text in extra_texts if text)
    safe_ids: set[int] = set()
    for text in safe_texts:
        try:
            safe_ids.update(
                int(token_id)
                for token_id in tokenizer.encode(text, add_special_tokens=False)
            )
        except Exception:  # noqa: BLE001 - tokenizer edge cases are non-fatal.
            continue
    for identifier in identifiers:
        for variant in (identifier, f" {identifier}"):
            try:
                safe_ids.update(
                    int(token_id)
                    for token_id in tokenizer.encode(variant, add_special_tokens=False)
                )
            except Exception:  # noqa: BLE001 - tokenizer edge cases are non-fatal.
                continue
    for literal in ("<<<<", " SEARCH", "\n====", "\n>>>>", "/", ".", "_", "-"):
        try:
            safe_ids.update(
                int(token_id)
                for token_id in tokenizer.encode(literal, add_special_tokens=False)
            )
        except Exception:  # noqa: BLE001
            continue
    return safe_ids


def token_ids_for_literals(tokenizer: Any, literals: tuple[str, ...]) -> set[int]:
    token_ids: set[int] = set()
    for literal in literals:
        try:
            token_ids.update(
                int(token_id)
                for token_id in tokenizer.encode(literal, add_special_tokens=False)
            )
        except Exception:  # noqa: BLE001 - tokenizer edge cases are non-fatal.
            continue
    return token_ids


def token_ids_for_dialect_literals(tokenizer: Any, literals: tuple[str, ...]) -> set[int]:
    literal_set = set(literals)
    token_ids: set[int] = set()
    try:
        vocab = dict(tokenizer.get_vocab())
    except Exception:  # noqa: BLE001 - tokenizer implementations vary.
        vocab = {}
    for token_text, token_id in vocab.items():
        if normalize_token_text(str(token_text)) in literal_set:
            token_ids.add(int(token_id))
    for literal in literal_set:
        for variant in (literal, f" {literal}", f"\n{literal}", f"\t{literal}"):
            try:
                encoded = [
                    int(token_id)
                    for token_id in tokenizer.encode(variant, add_special_tokens=False)
                ]
            except Exception:  # noqa: BLE001 - tokenizer edge cases are non-fatal.
                continue
            if len(encoded) == 1:
                token_ids.update(encoded)
    return token_ids


def dialect_family_token_ids_for_spec(
    tokenizer: Any,
    spec: DialectLockSpec | None,
) -> dict[str, set[int]]:
    if spec is None:
        return {}
    allowed_families = set(spec.contrast_families_banned)
    forbidden = set(spec.forbidden_literals)
    family_token_ids: dict[str, set[int]] = {}
    for family, literals in DIALECT_CONTRAST_LOCKS.get(spec.dialect, ()):
        if allowed_families and family not in allowed_families:
            continue
        kept = tuple(literal for literal in literals if literal in forbidden)
        token_ids = token_ids_for_dialect_literals(tokenizer, kept)
        if token_ids:
            family_token_ids[family] = token_ids
    return family_token_ids


def dialect_target_anchor_token_ids(tokenizer: Any, dialect: str) -> set[int]:
    anchors = DIALECT_STEERING_TARGET_ANCHORS.get(str(dialect or ""), ())
    return token_ids_for_dialect_literals(tokenizer, anchors)


def _clamp_unit(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def decoder_dialect_steering_should_trigger(
    rank: int,
    margin: float,
    *,
    top_k: int = DEFAULT_DECODER_DIALECT_STEERING_TOP_K,
    margin_floor: float = DEFAULT_DECODER_DIALECT_STEERING_MARGIN,
) -> bool:
    if not math.isfinite(float(margin)):
        return False
    if float(margin) >= 0.0:
        return True
    return int(rank) <= max(1, int(top_k)) and float(margin) >= float(margin_floor)


def decoder_dialect_steering_severity(
    rank: int,
    margin: float,
    *,
    top_k: int = DEFAULT_DECODER_DIALECT_STEERING_TOP_K,
    margin_floor: float = DEFAULT_DECODER_DIALECT_STEERING_MARGIN,
) -> float:
    if not decoder_dialect_steering_should_trigger(
        rank,
        margin,
        top_k=top_k,
        margin_floor=margin_floor,
    ):
        return 0.0
    rank_window = max(1, int(top_k))
    bounded_rank = max(1, min(int(rank), rank_window))
    rank_score = (rank_window + 1 - bounded_rank) / rank_window
    margin_ceiling = 1.5
    margin_score = _clamp_unit(
        (float(margin) - float(margin_floor))
        / max(1.0e-6, margin_ceiling - float(margin_floor))
    )
    severity = _clamp_unit(0.65 * margin_score + 0.35 * rank_score)
    if float(margin) >= 0.0:
        severity = max(severity, 0.75)
    return severity


def decoder_dialect_steering_alpha_update(
    alpha: float,
    severity: float,
    *,
    decay: float = DEFAULT_DECODER_DIALECT_STEERING_DECAY,
    floor: float = DEFAULT_DECODER_DIALECT_STEERING_FLOOR,
) -> float:
    decayed = _clamp_unit(alpha) * _clamp_unit(decay)
    updated = max(decayed, _clamp_unit(severity))
    return 0.0 if updated < max(0.0, float(floor)) else min(1.0, updated)


def decoder_dialect_steering_prior_from_args(
    args: argparse.Namespace,
    dialects_by_path: dict[str, str],
) -> tuple[dict[str, float], dict[str, Any]]:
    prior = getattr(args, "_decoder_dialect_prior", None)
    if not isinstance(prior, dict):
        return {}, {"active": False, "reason": "no_cross_case_prior"}
    current_dialects = sorted(
        {
            str(dialect)
            for dialect in dialects_by_path.values()
            if str(dialect or "")
        }
    )
    if not current_dialects:
        return {}, {"active": False, "reason": "no_current_dialect"}
    dialect_entries = prior.get("dialects", {})
    if not isinstance(dialect_entries, dict):
        return {}, {"active": False, "reason": "malformed_prior"}
    max_seed_alpha = max(0.0, min(1.0, float(prior.get("max_seed_alpha", 0.0) or 0.0)))
    seed_by_family: dict[str, float] = {}
    matched: dict[str, dict[str, Any]] = {}
    for dialect in current_dialects:
        entry = dialect_entries.get(dialect, {})
        if not isinstance(entry, dict):
            continue
        families = entry.get("families", {})
        if not isinstance(families, dict):
            continue
        accepted_cases = int(entry.get("accepted_case_count", 0) or 0)
        family_seeds: dict[str, float] = {}
        for family, family_entry in families.items():
            if not isinstance(family_entry, dict):
                continue
            seed_alpha = max(
                0.0,
                min(
                    max_seed_alpha if max_seed_alpha > 0.0 else 1.0,
                    float(family_entry.get("seed_alpha", 0.0) or 0.0),
                ),
            )
            if seed_alpha <= 0.0:
                continue
            family_key = str(family)
            seed_by_family[family_key] = max(seed_by_family.get(family_key, 0.0), seed_alpha)
            family_seeds[family_key] = seed_alpha
        if accepted_cases or family_seeds:
            matched[dialect] = {
                "seen_cases": int(entry.get("seen_cases", 0) or 0),
                "accepted_case_count": accepted_cases,
                "successful_case_streak": int(entry.get("successful_case_streak", 0) or 0),
                "seed_alpha_by_family": family_seeds,
                "last_row_index": entry.get("last_row_index"),
                "last_instance_id": entry.get("last_instance_id", ""),
            }
    return seed_by_family, {
        "active": bool(matched),
        "mode": str(prior.get("mode", "same_dialect_bounded_seed")),
        "current_dialects": current_dialects,
        "matched_dialects": sorted(matched),
        "matched": matched,
        "seen_cases": int(prior.get("seen_cases", 0) or 0),
        "accepted_cases": int(prior.get("accepted_cases", 0) or 0),
        "max_seed_alpha": max_seed_alpha,
        "seed_alpha_by_family": dict(sorted(seed_by_family.items())),
    }


class DecoderDialectSteeringState:
    """Connect post-mask temptation telemetry to a one-token-later layer hook."""

    def __init__(
        self,
        *,
        enabled: bool,
        tokenizer: Any,
        model: Any,
        layer_index: int,
        default_path: str,
        dialects_by_path: dict[str, str],
        family_token_ids_by_path: dict[str, dict[str, set[int]]],
        anchor_token_ids_by_dialect: dict[str, set[int]],
        top_k: int = DEFAULT_DECODER_DIALECT_STEERING_TOP_K,
        margin_floor: float = DEFAULT_DECODER_DIALECT_STEERING_MARGIN,
        decay: float = DEFAULT_DECODER_DIALECT_STEERING_DECAY,
        floor: float = DEFAULT_DECODER_DIALECT_STEERING_FLOOR,
        steer_fraction: float = DEFAULT_DECODER_DIALECT_STEERING_FRACTION,
        max_delta_rms: float = DEFAULT_DECODER_DIALECT_STEERING_MAX_DELTA_RMS,
        max_events: int = DEFAULT_DECODER_DIALECT_STEERING_MAX_EVENTS,
        initial_alphas_by_family: dict[str, float] | None = None,
        prior_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.requested_enabled = bool(enabled)
        self.active = False
        self.status = "disabled"
        self.tokenizer = tokenizer
        self.layer_index = int(layer_index)
        self.default_path = str(default_path or "")
        self.dialects_by_path = {
            str(path): str(dialect)
            for path, dialect in dialects_by_path.items()
            if str(path or "") and str(dialect or "")
        }
        self.family_token_ids_by_path = {
            str(path): {
                str(family): {int(token_id) for token_id in token_ids}
                for family, token_ids in families.items()
                if token_ids
            }
            for path, families in family_token_ids_by_path.items()
            if families
        }
        self.anchor_token_ids_by_dialect = {
            str(dialect): {int(token_id) for token_id in token_ids}
            for dialect, token_ids in anchor_token_ids_by_dialect.items()
        }
        self.top_k = max(1, int(top_k))
        self.margin_floor = float(margin_floor)
        self.decay = _clamp_unit(decay)
        self.floor = max(0.0, float(floor))
        self.steer_fraction = max(0.0, float(steer_fraction))
        self.max_delta_rms = max(0.0, float(max_delta_rms))
        self.max_events = max(0, int(max_events))
        self.vectors: dict[tuple[str, str], Any] = {}
        self.vector_ids: dict[tuple[str, str], str] = {}
        self.vector_sources: dict[tuple[str, str], str] = {}
        self.alphas: dict[tuple[str, str], float] = {}
        self.alpha_peaks: dict[tuple[str, str], float] = {}
        self.prior_initial_alphas_by_family = {
            str(family): _clamp_unit(alpha)
            for family, alpha in (initial_alphas_by_family or {}).items()
            if _clamp_unit(alpha) > 0.0
        }
        self.prior_initial_alphas_by_path_family: dict[str, float] = {}
        self.prior_metadata = dict(prior_metadata or {})
        self.events: list[dict[str, Any]] = []
        self._family_tensor_cache: dict[tuple[str, str, str, int], Any] = {}
        self._hook_handle: Any | None = None
        self._hook_uses_kwargs = True
        self.last_error = ""
        self.trigger_count = 0
        self.trigger_step_count = 0
        self.applied_count = 0
        self.clean_decay_count = 0
        self.clean_step_count = 0
        self.skipped_step_count = 0
        self.candidate_step_count = 0
        self.masked_but_not_tempting_count = 0
        self.last_applied_alpha_total = 0.0
        if not self.requested_enabled:
            return
        if not self.family_token_ids_by_path:
            self.status = "no_dialect_family_token_ids"
            return
        weight = self._resolve_lm_head_weight(model)
        if weight is None:
            self.status = "missing_lm_head_weight"
            return
        self._build_vectors(weight)
        if not self.vectors:
            self.status = self.status if self.last_error else "no_usable_vectors"
            return
        self._apply_prior_initial_alphas()
        self.active = True
        self.status = "active"

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        tokenizer: Any,
        model: Any,
        layer_index: int,
        default_path: str,
        dialects_by_path: dict[str, str],
        family_token_ids_by_path: dict[str, dict[str, set[int]]],
        anchor_token_ids_by_dialect: dict[str, set[int]],
    ) -> "DecoderDialectSteeringState":
        initial_alphas_by_family, prior_metadata = decoder_dialect_steering_prior_from_args(
            args,
            dialects_by_path,
        )
        return cls(
            enabled=bool(getattr(args, "decoder_dialect_steering", False)),
            tokenizer=tokenizer,
            model=model,
            layer_index=layer_index,
            default_path=default_path,
            dialects_by_path=dialects_by_path,
            family_token_ids_by_path=family_token_ids_by_path,
            anchor_token_ids_by_dialect=anchor_token_ids_by_dialect,
            top_k=int(
                getattr(
                    args,
                    "decoder_dialect_steering_top_k",
                    DEFAULT_DECODER_DIALECT_STEERING_TOP_K,
                )
            ),
            margin_floor=float(
                getattr(
                    args,
                    "decoder_dialect_steering_margin",
                    DEFAULT_DECODER_DIALECT_STEERING_MARGIN,
                )
            ),
            decay=float(
                getattr(
                    args,
                    "decoder_dialect_steering_decay",
                    DEFAULT_DECODER_DIALECT_STEERING_DECAY,
                )
            ),
            floor=float(
                getattr(
                    args,
                    "decoder_dialect_steering_floor",
                    DEFAULT_DECODER_DIALECT_STEERING_FLOOR,
                )
            ),
            steer_fraction=float(
                getattr(
                    args,
                    "decoder_dialect_steering_fraction",
                    DEFAULT_DECODER_DIALECT_STEERING_FRACTION,
                )
            ),
            max_delta_rms=float(
                getattr(
                    args,
                    "decoder_dialect_steering_max_delta_rms",
                    DEFAULT_DECODER_DIALECT_STEERING_MAX_DELTA_RMS,
                )
            ),
            max_events=int(
                getattr(
                    args,
                    "decoder_dialect_steering_max_events",
                    DEFAULT_DECODER_DIALECT_STEERING_MAX_EVENTS,
                )
            ),
            initial_alphas_by_family=initial_alphas_by_family,
            prior_metadata=prior_metadata,
        )

    def _resolve_lm_head_weight(self, model: Any) -> Any | None:
        getter = getattr(model, "get_output_embeddings", None)
        if callable(getter):
            try:
                head = getter()
                weight = getattr(head, "weight", None)
                if weight is not None:
                    return weight
            except Exception as exc:  # noqa: BLE001 - optional steering only.
                self.last_error = f"get_output_embeddings_failed:{type(exc).__name__}"
        for attr_path in (
            ("lm_head",),
            ("model", "lm_head"),
            ("language_model", "lm_head"),
            ("model", "language_model", "lm_head"),
        ):
            target = model
            for attr in attr_path:
                target = getattr(target, attr, None)
                if target is None:
                    break
            weight = getattr(target, "weight", None)
            if weight is not None:
                return weight
        return None

    def _rms_normalize(self, tensor: Any, *, dim: int = -1, eps: float = 1.0e-6) -> Any:
        import torch

        squared = tensor.float().pow(2).mean(dim=dim, keepdim=True)
        return tensor.float() * torch.rsqrt(squared.clamp_min(eps * eps))

    def _build_vectors(self, weight: Any) -> None:
        import torch

        vocab_size = int(weight.shape[0])
        hidden_size = int(weight.shape[-1])
        for path, family_map in self.family_token_ids_by_path.items():
            dialect = self.dialects_by_path.get(path, "")
            target_ids = self.anchor_token_ids_by_dialect.get(dialect, set())
            valid_target_ids = [
                token_id for token_id in sorted(target_ids) if 0 <= token_id < vocab_size
            ]
            for family, token_ids in family_map.items():
                valid_foreign_ids = [
                    token_id
                    for token_id in sorted(token_ids)
                    if 0 <= token_id < vocab_size
                ]
                if not valid_foreign_ids:
                    continue
                try:
                    with torch.no_grad():
                        foreign_index = torch.tensor(
                            valid_foreign_ids,
                            device=weight.device,
                            dtype=torch.long,
                        )
                        foreign_rows = self._rms_normalize(
                            weight.index_select(0, foreign_index),
                        )
                        foreign_mean = foreign_rows.mean(dim=0)
                        source = "lm_head_foreign_only"
                        if valid_target_ids:
                            target_index = torch.tensor(
                                valid_target_ids,
                                device=weight.device,
                                dtype=torch.long,
                            )
                            target_rows = self._rms_normalize(
                                weight.index_select(0, target_index),
                            )
                            vector = foreign_mean - target_rows.mean(dim=0)
                            source = "lm_head_foreign_minus_target"
                        else:
                            vector = foreign_mean
                        vector = self._rms_normalize(vector, dim=-1).detach()
                except Exception as exc:  # noqa: BLE001 - optional steering only.
                    self.last_error = f"vector_build_failed:{type(exc).__name__}"
                    continue
                key = (path, family)
                self.vectors[key] = vector
                self.alphas[key] = 0.0
                self.alpha_peaks[key] = 0.0
                digest = hashlib.sha1(
                    (
                        f"{dialect}|{family}|{hidden_size}|"
                        f"{','.join(str(token_id) for token_id in valid_foreign_ids)}|"
                        f"{','.join(str(token_id) for token_id in valid_target_ids)}"
                    ).encode("utf-8")
                ).hexdigest()
                self.vector_ids[key] = digest[:16]
                self.vector_sources[key] = source

    def _apply_prior_initial_alphas(self) -> None:
        if not self.prior_initial_alphas_by_family:
            return
        for key in sorted(self.alphas):
            _path, family = key
            seed_alpha = float(self.prior_initial_alphas_by_family.get(family, 0.0))
            if seed_alpha < self.floor:
                continue
            self.alphas[key] = max(float(self.alphas.get(key, 0.0)), seed_alpha)
            self.prior_initial_alphas_by_path_family[f"{key[0]}::{key[1]}"] = seed_alpha

    def install(self, runtime: Any) -> None:
        if not self.active:
            return
        try:
            layers = runtime._resolve_layers()
            if self.layer_index < 0 or self.layer_index >= len(layers):
                self.active = False
                self.status = "layer_index_out_of_range"
                return
            target_layer = layers[self.layer_index]
            try:
                self._hook_handle = target_layer.register_forward_pre_hook(
                    self._forward_pre_hook,
                    with_kwargs=True,
                )
                self._hook_uses_kwargs = True
            except TypeError:
                self._hook_handle = target_layer.register_forward_pre_hook(
                    self._forward_pre_hook_no_kwargs,
                )
                self._hook_uses_kwargs = False
            self.status = "active_hook_installed"
        except Exception as exc:  # noqa: BLE001 - optional steering only.
            self.active = False
            self.status = "hook_install_failed"
            self.last_error = f"{type(exc).__name__}:{exc}"

    def remove(self) -> None:
        handle = self._hook_handle
        self._hook_handle = None
        if handle is None:
            return
        try:
            handle.remove()
        except Exception as exc:  # noqa: BLE001 - metadata only.
            self.last_error = f"hook_remove_failed:{type(exc).__name__}"

    def _forward_pre_hook_no_kwargs(self, module: Any, args: tuple[Any, ...]) -> Any:
        result = self._forward_pre_hook(module, args, {})
        if result is None:
            return args
        new_args, _kwargs = result
        return new_args

    def _hidden_from_hook_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any | None, str]:
        if args:
            return args[0], "args"
        hidden = kwargs.get("hidden_states")
        if hidden is not None:
            return hidden, "kwargs"
        return None, ""

    def _forward_pre_hook(
        self,
        _module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if not self.active:
            return None
        active_items = [
            (key, alpha)
            for key, alpha in self.alphas.items()
            if alpha >= self.floor and key in self.vectors
        ]
        if not active_items:
            return None
        hidden, location = self._hidden_from_hook_inputs(args, kwargs)
        if hidden is None or len(getattr(hidden, "shape", ())) < 3:
            return None
        if int(hidden.shape[1]) != 1:
            return None
        vector_dim = int(self.vectors[active_items[0][0]].shape[-1])
        if int(hidden.shape[-1]) != vector_dim:
            self.active = False
            self.status = "hidden_size_mismatch"
            self.last_error = (
                f"hidden_size_mismatch:hidden={int(hidden.shape[-1])}:"
                f"vector={vector_dim}"
            )
            return None
        try:
            import torch

            with torch.no_grad():
                mix = None
                alpha_total = 0.0
                for key, alpha in active_items:
                    vector = self.vectors[key].to(device=hidden.device, dtype=torch.float32)
                    weighted = float(alpha) * vector
                    mix = weighted if mix is None else mix + weighted
                    alpha_total += float(alpha)
                if mix is None:
                    return None
                mix = self._rms_normalize(mix, dim=-1).view(1, 1, -1)
                h_float = hidden[:, -1:, :].float()
                h_rms = torch.sqrt(
                    torch.mean(h_float.pow(2), dim=-1, keepdim=True).clamp_min(1.0e-12)
                )
                delta = self.steer_fraction * h_rms * mix
                delta_rms = torch.sqrt(
                    torch.mean(delta.pow(2), dim=-1, keepdim=True).clamp_min(1.0e-12)
                )
                max_delta = self.max_delta_rms * h_rms
                scale = torch.clamp(max_delta / (delta_rms + 1.0e-6), max=1.0)
                delta = delta * scale
                hidden_new = hidden.clone()
                hidden_new[:, -1:, :] = (h_float - delta).to(dtype=hidden.dtype)
        except Exception as exc:  # noqa: BLE001 - optional steering only.
            self.last_error = f"hook_apply_failed:{type(exc).__name__}"
            return None
        self.applied_count += 1
        self.last_applied_alpha_total = float(alpha_total)
        if location == "kwargs":
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = hidden_new
            return args, new_kwargs
        return (hidden_new,) + tuple(args[1:]), kwargs

    def _token_id_from_sampler(self, token: Any) -> int:
        try:
            return int(token.detach().to("cpu").reshape(-1)[0].item())
        except Exception:  # noqa: BLE001 - sampler return types vary.
            try:
                return int(token)
            except Exception:  # noqa: BLE001
                return -1

    def _decode_token_text(self, token_id: int) -> str:
        return decode_single_token(self.tokenizer, int(token_id), "")

    def _family_tensor(self, path: str, family: str, raw: Any) -> Any | None:
        import torch

        vocab_size = int(raw.shape[-1])
        key = (path, family, str(raw.device), vocab_size)
        cached = self._family_tensor_cache.get(key)
        if cached is not None:
            return cached
        token_ids = self.family_token_ids_by_path.get(path, {}).get(family, set())
        valid_ids = sorted(token_id for token_id in token_ids if 0 <= token_id < vocab_size)
        if not valid_ids:
            return None
        tensor = torch.tensor(valid_ids, device=raw.device, dtype=torch.long)
        self._family_tensor_cache[key] = tensor
        return tensor

    def _decay_all(self, *, count_clean: bool) -> None:
        had_alpha = any(alpha >= self.floor for alpha in self.alphas.values())
        for key, alpha in list(self.alphas.items()):
            self.alphas[key] = decoder_dialect_steering_alpha_update(
                alpha,
                0.0,
                decay=self.decay,
                floor=self.floor,
            )
        if count_clean:
            self.clean_step_count += 1
            if had_alpha:
                self.clean_decay_count += 1

    def observe_clean(self, reason: str = "clean_decode_step") -> None:
        if not self.active:
            return
        self.skipped_step_count += 1 if reason != "clean_decode_step" else 0
        self._decay_all(count_clean=True)

    def observe(
        self,
        raw_logits: Any,
        scores: Any,
        chosen_token: Any,
        *,
        generated_text: str,
        step: int,
        replace_region_active: bool,
        active_path: str,
    ) -> None:
        del generated_text
        if not self.active:
            return
        if not replace_region_active:
            self.observe_clean("not_replace_region")
            return
        path = str(active_path or self.default_path or "")
        if path not in self.family_token_ids_by_path:
            path = self.default_path
        family_map = self.family_token_ids_by_path.get(path, {})
        if not family_map:
            self.observe_clean("no_family_for_path")
            return
        try:
            raw = raw_logits.detach()
            if len(raw.shape) == 2:
                raw = raw[0]
            elif len(raw.shape) > 2:
                raw = raw.reshape(-1, raw.shape[-1])[0]
            raw = raw.float()
        except Exception as exc:  # noqa: BLE001 - optional steering only.
            self.last_error = f"observe_logits_failed:{type(exc).__name__}"
            self.observe_clean("invalid_logits")
            return
        chosen_id = self._token_id_from_sampler(chosen_token)
        if chosen_id < 0 or chosen_id >= int(raw.shape[-1]):
            self.observe_clean("invalid_chosen_token")
            return
        self.candidate_step_count += 1
        chosen_logit = float(raw[chosen_id].item())
        post_mask_chosen_logit = None
        try:
            score_view = scores.detach()
            if len(score_view.shape) == 2:
                score_view = score_view[0]
            elif len(score_view.shape) > 2:
                score_view = score_view.reshape(-1, score_view.shape[-1])[0]
            post_mask_chosen_logit = float(score_view.float()[chosen_id].item())
        except Exception:  # noqa: BLE001 - telemetry only.
            post_mask_chosen_logit = None
        top_count = min(self.top_k, int(raw.shape[-1]))
        top_ids: list[int] = []
        if top_count > 0:
            try:
                top_ids = [
                    int(token_id)
                    for token_id in raw.topk(top_count, dim=-1).indices.detach().tolist()
                ]
            except Exception:  # noqa: BLE001 - rank fallback below.
                top_ids = []
        tempting: list[dict[str, Any]] = []
        not_tempting_count = 0
        for family in sorted(family_map):
            family_tensor = self._family_tensor(path, family, raw)
            if family_tensor is None:
                continue
            family_logits = raw.index_select(0, family_tensor)
            best_offset = int(family_logits.argmax(dim=-1).item())
            best_id = int(family_tensor[best_offset].item())
            best_logit = float(family_logits[best_offset].item())
            margin = best_logit - chosen_logit
            if best_id in top_ids:
                rank = top_ids.index(best_id) + 1
            elif margin >= 0.0:
                rank = int((raw > best_logit).sum().item()) + 1
            else:
                rank = self.top_k + 1
            severity = decoder_dialect_steering_severity(
                rank,
                margin,
                top_k=self.top_k,
                margin_floor=self.margin_floor,
            )
            if severity > 0.0:
                tempting.append(
                    {
                        "family": family,
                        "offense_token_id": best_id,
                        "offense_token_text": self._decode_token_text(best_id),
                        "offense_rank": int(rank),
                        "offense_raw_logit": best_logit,
                        "chosen_token_id": int(chosen_id),
                        "chosen_token_text": self._decode_token_text(chosen_id),
                        "chosen_raw_logit": chosen_logit,
                        "chosen_post_mask_logit": post_mask_chosen_logit,
                        "margin": float(margin),
                        "severity": float(severity),
                        "candidate_token_count": int(family_tensor.numel()),
                    }
                )
            else:
                not_tempting_count += int(family_tensor.numel())
        self.masked_but_not_tempting_count += int(not_tempting_count)
        if not tempting:
            self._decay_all(count_clean=True)
            return
        before_alpha = dict(self.alphas)
        self._decay_all(count_clean=False)
        self.trigger_step_count += 1
        for event in tempting:
            family = str(event["family"])
            key = (path, family)
            alpha_before = float(before_alpha.get(key, 0.0))
            alpha_after_decay = float(self.alphas.get(key, 0.0))
            alpha_after = max(alpha_after_decay, float(event["severity"]))
            alpha_after = 0.0 if alpha_after < self.floor else min(1.0, alpha_after)
            self.alphas[key] = alpha_after
            self.alpha_peaks[key] = max(float(self.alpha_peaks.get(key, 0.0)), alpha_after)
            self.trigger_count += 1
            if len(self.events) < self.max_events:
                self.events.append(
                    {
                        "step": int(step),
                        "target_path": path,
                        "target_dialect": self.dialects_by_path.get(path, ""),
                        "offense_family": family,
                        "trigger_condition": {
                            "top_k": int(self.top_k),
                            "margin_floor": float(self.margin_floor),
                            "rule": (
                                "margin>=0 or "
                                "(offense_rank<=top_k and margin>=margin_floor)"
                            ),
                        },
                        **{
                            name: event[name]
                            for name in (
                                "offense_token_id",
                                "offense_token_text",
                                "offense_rank",
                                "offense_raw_logit",
                                "chosen_token_id",
                                "chosen_token_text",
                                "chosen_raw_logit",
                                "chosen_post_mask_logit",
                                "margin",
                                "severity",
                                "candidate_token_count",
                            )
                        },
                        "alpha_before": alpha_before,
                        "alpha_after_decay": alpha_after_decay,
                        "alpha_after": alpha_after,
                        "apply_layer": int(self.layer_index),
                        "apply_next_step": int(step) + 1,
                        "vector_id": self.vector_ids.get(key, ""),
                        "vector_source": self.vector_sources.get(key, ""),
                    }
                )

    def telemetry(self) -> dict[str, Any]:
        family_peaks: dict[str, float] = {}
        for (_path, family), alpha in self.alpha_peaks.items():
            family_peaks[family] = max(family_peaks.get(family, 0.0), float(alpha))
        candidate_steps = max(1, int(self.candidate_step_count))
        return {
            "active": bool(self.active),
            "requested": bool(self.requested_enabled),
            "status": self.status,
            "mode": "decoder_informed_layer_pre_hook",
            "layer_index": int(self.layer_index),
            "hook": {
                "installed": self._hook_handle is not None,
                "with_kwargs": bool(self._hook_uses_kwargs),
                "prefill_guard": "skip_when_seq_len_not_1",
            },
            "trigger_condition": {
                "top_k": int(self.top_k),
                "margin_floor": float(self.margin_floor),
                "red_alert_margin": 0.0,
                "rule": "margin>=0 or (rank<=top_k and margin>=margin_floor)",
            },
            "alpha": {
                "decay": float(self.decay),
                "floor": float(self.floor),
                "steer_fraction": float(self.steer_fraction),
                "max_delta_rms": float(self.max_delta_rms),
                "current_by_path_family": {
                    f"{path}::{family}": float(alpha)
                    for (path, family), alpha in sorted(self.alphas.items())
                    if alpha > 0.0
                },
                "peak_by_path_family": {
                    f"{path}::{family}": float(alpha)
                    for (path, family), alpha in sorted(self.alpha_peaks.items())
                    if alpha > 0.0
                },
                "peak_by_family": family_peaks,
            },
            "prior": {
                **self.prior_metadata,
                "initial_alpha_by_family": dict(
                    sorted(self.prior_initial_alphas_by_family.items())
                ),
                "initial_alpha_by_path_family": dict(
                    sorted(self.prior_initial_alphas_by_path_family.items())
                ),
                "seed_applied": bool(self.prior_initial_alphas_by_path_family),
            },
            "vectors": {
                "source": "lm_head_logit_lens",
                "count": len(self.vectors),
                "ids": {
                    f"{path}::{family}": vector_id
                    for (path, family), vector_id in sorted(self.vector_ids.items())
                },
                "family_token_counts_by_path": {
                    path: {
                        family: len(token_ids)
                        for family, token_ids in sorted(families.items())
                    }
                    for path, families in sorted(self.family_token_ids_by_path.items())
                },
                "target_dialects_by_path": dict(sorted(self.dialects_by_path.items())),
            },
            "trigger_count": int(self.trigger_count),
            "trigger_step_count": int(self.trigger_step_count),
            "applied_count": int(self.applied_count),
            "clean_step_count": int(self.clean_step_count),
            "clean_decay_count": int(self.clean_decay_count),
            "skipped_step_count": int(self.skipped_step_count),
            "candidate_step_count": int(self.candidate_step_count),
            "temptation_rate_per_candidate_step": (
                float(self.trigger_count) / float(candidate_steps)
            ),
            "steering_application_rate_per_candidate_step": (
                float(self.applied_count) / float(candidate_steps)
            ),
            "masked_but_not_tempting_count": int(self.masked_but_not_tempting_count),
            "last_applied_alpha_total": float(self.last_applied_alpha_total),
            "events": list(self.events),
            "last_error": self.last_error,
        }


def pattern_window() -> DependencyWindow:
    return DependencyWindow(
        window_id=PATTERN_WINDOW_ID,
        path=PATTERN_WINDOW_PATH,
        text=SEARCH_REPLACE_PATTERN_WINDOW,
        score=1.0,
    )


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
    file_context = "\n\n".join(file_sections)
    return f"{STRUCTURAL_ANCHOR}{query}\n\n{file_context}" if file_context else f"{STRUCTURAL_ANCHOR}{query}"


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


def canonicalize_search_replace_output(generated: str) -> str:
    """Normalize common conflict-marker drift back to the strict local dialect."""

    text = generated or ""
    if "<<<<" not in text or "SEARCH" not in text.upper():
        return text
    text = re.sub(
        r"(?m)^={4,}\s*$",
        "==== REPLACE",
        text,
    )
    text = re.sub(
        r"(?m)^>{4,}\s*REPLACE\s*$",
        ">>>>",
        text,
        flags=re.IGNORECASE,
    )
    return text


def normalized_window_text(text: str) -> str:
    return " ".join(str(text or "").split())


def readout_terms(*texts: str) -> set[str]:
    stopwords = {
        "and",
        "are",
        "for",
        "from",
        "into",
        "not",
        "the",
        "this",
        "that",
        "to",
        "with",
        "you",
    }
    terms = {
        term.lower()
        for text in texts
        for term in TERM_RE.findall(str(text or ""))
        if len(term) >= 3
    }
    return terms - stopwords


def span_relevance_score(span: str, query_terms: set[str], window: DependencyWindow) -> float:
    span_terms = readout_terms(span)
    overlap = len(query_terms & span_terms)
    identifier_bonus = len(extract_identifiers(span) & query_terms)
    activation_bonus = max(0.0, float(window.activation_score)) * 3.0
    lexical_bonus = max(0.0, float(window.lexical_score))
    exact_bonus = 100.0 if window.exact_scope_match else 0.0
    return (
        (overlap * 10.0)
        + (identifier_bonus * 2.0)
        + lexical_bonus
        + activation_bonus
        + exact_bonus
        + float(window.score)
    )


def select_verbatim_readout_span(
    case: LoCoBenchCase,
    hot_windows: list[DependencyWindow],
    generated: str,
) -> tuple[str, str, dict[str, Any]]:
    generated_paths = generated_target_paths(generated, case.files)
    instruction_mentions = known_path_mentions(case.instruction, case.files)
    instruction_paths = [path for path in case.files if path in instruction_mentions]
    hot_paths = phase3_target_paths(case, hot_windows)
    target_paths = []
    for path in [*generated_paths, *instruction_paths, *hot_paths]:
        if path and path not in target_paths:
            target_paths.append(path)
    target_path = target_paths[0] if target_paths else ""
    candidate_windows = [
        window for window in hot_windows if not target_paths or window.path in target_paths
    ] or list(hot_windows)
    query_terms = readout_terms(case.instruction, generated)
    best_path = target_path
    best_span = ""
    best_score = float("-inf")
    best_window_id: int | None = None
    best_window: DependencyWindow | None = None
    for window in candidate_windows:
        content = case.files.get(window.path, "")
        if not content:
            continue
        span = window.text
        if not span.strip() or span not in content:
            continue
        score = span_relevance_score(span, query_terms, window) + 1000.0
        if window.path == target_path:
            score += 500.0
        if window.route_source.startswith("exact_scope"):
            score += 250.0
        if score > best_score or (
            score == best_score
            and best_window is not None
            and (
                list(case.files).index(window.path) if window.path in case.files else 999_999,
                window.start_line,
                window.window_id,
            )
            < (
                list(case.files).index(best_window.path)
                if best_window.path in case.files
                else 999_999,
                best_window.start_line,
                best_window.window_id,
            )
        ):
            best_path = window.path
            best_span = span
            best_score = score
            best_window_id = int(window.window_id)
            best_window = window
    if not best_span and target_path in case.files:
        lines = case.files[target_path].splitlines()
        best_span = "\n".join(lines[:5]) if lines else case.files[target_path]
        best_path = target_path
    metadata = {
        "target_path": best_path,
        "window_id": best_window_id,
        "activation_route_id": None if best_window is None else best_window.activation_route_id,
        "route_source": "fallback_target_prefix" if best_window is None else best_window.route_source,
        "source_start_line": None if best_window is None else int(best_window.start_line),
        "source_end_line": None if best_window is None else int(best_window.end_line),
        "source_start_token": None if best_window is None else int(best_window.start_token),
        "source_end_token": None if best_window is None else int(best_window.end_token),
        "lexical_score": None if best_window is None else float(best_window.lexical_score),
        "activation_score": None if best_window is None else float(best_window.activation_score),
        "exact_scope_match": False if best_window is None else bool(best_window.exact_scope_match),
        "span_line_count": len(best_span.splitlines()),
        "span_chars": len(best_span),
        "score": best_score if best_score != float("-inf") else None,
    }
    return best_path, best_span, metadata


def replace_payload_after_separator(generated: str) -> str:
    text = generated or ""
    separator = re.search(r"====\s*REPLACE", text, flags=re.IGNORECASE)
    if separator:
        payload = text[separator.end() :]
    else:
        marker = re.search(r"<<<<\s*SEARCH", text, flags=re.IGNORECASE)
        payload = text[marker.end() :] if marker else text
    payload = re.split(r">>>>", payload, maxsplit=1)[0]
    return payload.strip("\n")


def identifier_segment_growth_signature(
    line: str,
) -> tuple[str, str, str, str, str, int] | None:
    text = (line or "").strip()
    match = IDENTIFIER_SEGMENT_GROWTH_CHAIN_RE.search(text)
    if match is None:
        return None
    chain = match.group("chain")
    parts = IDENTIFIER_SEGMENT_SEPARATOR_RE.split(chain)
    if len(parts) < 3 or len(parts) % 2 == 0:
        return None
    segment = parts[-1]
    separator = parts[-2]
    repeat_count = 1
    index = len(parts) - 3
    while index >= 1:
        if parts[index] != segment or parts[index - 1] != separator:
            break
        repeat_count += 1
        index -= 2
    base = "".join(parts[: index + 1])
    if not base or not IDENTIFIER_RE.search(base):
        return None
    prefix = text[: match.start("chain")]
    trailer = match.group("trailer") or ""
    return prefix, base, separator, segment, trailer, repeat_count


def repeated_identifier_segment_growth_metadata(
    payload: str,
    lines: list[str],
    *,
    min_repeats: int,
) -> dict[str, Any] | None:
    if len(lines) < int(min_repeats):
        return None
    signatures = [identifier_segment_growth_signature(line) for line in lines]
    candidate_ends = [len(lines)]
    if IDENTIFIER_TAIL_RE.search(payload) and len(lines) > 1:
        candidate_ends.append(len(lines) - 1)
    for end_index in candidate_ends:
        if end_index < int(min_repeats):
            continue
        tail_signature = signatures[end_index - 1]
        if tail_signature is None:
            continue
        tail_key = tail_signature[:5]
        run_indices = [end_index - 1]
        previous_count = int(tail_signature[5])
        for line_index in range(end_index - 2, -1, -1):
            signature = signatures[line_index]
            if signature is None or signature[:5] != tail_key:
                break
            current_count = int(signature[5])
            if current_count >= previous_count:
                break
            run_indices.append(line_index)
            previous_count = current_count
        if len(run_indices) < int(min_repeats):
            continue
        run_indices.reverse()
        first_signature = signatures[run_indices[0]]
        last_signature = signatures[run_indices[-1]]
        if first_signature is None or last_signature is None:
            continue
        preview_lines = [lines[index] for index in run_indices[-3:]]
        return {
            "active": True,
            "reason": "repeated_identifier_segment_growth",
            "growth_line_count": len(run_indices),
            "first_segment_repeat_count": int(first_signature[5]),
            "last_segment_repeat_count": int(last_signature[5]),
            "segment": str(last_signature[3]),
            "separator": str(last_signature[2]),
            "payload_chars": len(payload),
            "ignored_trailing_open_identifier": end_index != len(lines),
            "line_preview": "\n".join(preview_lines),
        }
    return None


def repeated_replacement_suffix_metadata(
    generated: str,
    *,
    min_repeats: int = 4,
    max_block_lines: int = 8,
    min_payload_chars: int = 320,
) -> dict[str, Any] | None:
    text = generated or ""
    separator = re.search(r"====\s*REPLACE", text, flags=re.IGNORECASE)
    if separator is None or re.search(r">>>>", text[separator.end() :]):
        return None
    payload = text[separator.end() :]
    if len(payload) < int(min_payload_chars):
        return None
    lines = [line.strip() for line in payload.rstrip().splitlines() if line.strip()]
    growth_meta = repeated_identifier_segment_growth_metadata(
        payload,
        lines,
        min_repeats=min_repeats,
    )
    if growth_meta is not None:
        return growth_meta
    if IDENTIFIER_TAIL_RE.search(payload):
        return None
    max_lines = min(int(max_block_lines), len(lines) // max(1, int(min_repeats)))
    for block_len in range(1, max_lines + 1):
        block = lines[-block_len:]
        if not any(IDENTIFIER_RE.search(line) for line in block):
            continue
        repeats = 1
        offset = len(lines) - block_len
        while offset - block_len >= 0 and lines[offset - block_len : offset] == block:
            repeats += 1
            offset -= block_len
        if repeats >= int(min_repeats):
            return {
                "active": True,
                "reason": "repeated_replacement_suffix",
                "block_line_count": block_len,
                "repeat_count": repeats,
                "payload_chars": len(payload),
                "block_preview": "\n".join(block[-min(3, len(block)) :]),
            }
    return None


def extract_open_search_body(generated: str) -> str:
    text = generated or ""
    marker = None
    for match in re.finditer(r"<<<<\s*SEARCH\s*", text, flags=re.IGNORECASE):
        marker = match
    if marker is None:
        return ""
    body = text[marker.end() :]
    separator = re.search(r"====\s*REPLACE", body, flags=re.IGNORECASE)
    if separator is not None:
        body = body[: separator.start()]
    close = re.search(r">>>>", body)
    if close is not None:
        body = body[: close.start()]
    return body


def normalized_diff_body(text: str) -> str:
    lines = str(text or "").strip().splitlines()
    return "\n".join(line.rstrip() for line in lines).strip()


def magnetized_search_bridge_decision(
    generated_text: str,
    *,
    target_path: str,
    target_span: str,
    tokenizer: Any | None = None,
    threshold: float = 0.90,
) -> dict[str, Any]:
    if not target_span or not re.search(r"<<<<\s*SEARCH", generated_text or "", re.IGNORECASE):
        return {"active": False, "reason": "search_or_target_absent"}
    if re.search(r"====\s*REPLACE", generated_text or "", flags=re.IGNORECASE):
        return {"active": False, "reason": "already_in_replace_or_closed"}
    search_body = normalized_diff_body(extract_open_search_body(generated_text))
    target_body = normalized_diff_body(target_span)
    if not search_body or not target_body:
        return {"active": False, "reason": "empty_search_or_target"}
    if len(search_body) < min(24, len(target_body)):
        return {"active": False, "reason": "search_body_too_short"}
    similarity = difflib.SequenceMatcher(None, search_body, target_body).ratio()
    active = similarity >= float(threshold)
    emitted_prefix = "\n==== REPLACE\n" if active else ""
    token_ids: list[int] = []
    if active and tokenizer is not None:
        try:
            token_ids = [
                int(token_id)
                for token_id in tokenizer.encode(emitted_prefix, add_special_tokens=False)
            ]
        except Exception:  # noqa: BLE001 - metadata still records the text prefix.
            token_ids = []
    return {
        "active": bool(active),
        "reason": "search_matches_target_span" if active else "similarity_below_threshold",
        "similarity": float(similarity),
        "threshold": float(threshold),
        "target_path": str(target_path or ""),
        "target_span_chars": len(target_body),
        "search_body_chars": len(search_body),
        "emitted_prefix": emitted_prefix,
        "force_transition_token_ids": token_ids,
        "state_transition": (
            f"{DIFF_STATE_SEARCH_BLOCK}->{DIFF_STATE_REPLACE_BLOCK}" if active else ""
        ),
    }


def verbatim_readout_bridge(
    case: LoCoBenchCase,
    hot_windows: list[DependencyWindow],
    generated: str,
) -> tuple[str, dict[str, Any]]:
    if not re.search(r"<<<<\s*SEARCH", generated or "", flags=re.IGNORECASE):
        return generated, {"active": False, "reason": "search_marker_absent"}
    path, span, span_meta = select_verbatim_readout_span(case, hot_windows, generated)
    if not path or not span:
        return generated, {"active": False, "reason": "no_hot_window_span"}
    replacement = replace_payload_after_separator(generated)
    bridged = f"{path}\n<<<< SEARCH\n{span}\n==== REPLACE\n{replacement}\n>>>>"
    return bridged, {
        "active": True,
        "reason": "search_marker_bridged_to_verbatim_hot_window_span",
        "identifier_authority": identifier_authority_from_span(path, span),
        **span_meta,
    }


def kv_direct_unavailable_reason(args: argparse.Namespace) -> str:
    jit_metadata = getattr(args, "jit_indexing", {}) or {}
    return (
        "Phase 3 KV-direct was requested, but the runner could not initialize "
        "the existing Lazarus Layer-14 KV-direct runtime path. The active path "
        "is retriever.answer_with_kv_direct_multi()/answer_with_kv_direct() "
        "-> runtime.generate_with_kv_direct_materialization(); it requires a "
        "loaded runtime/model plus either ready Apollo residual sidecars "
        "(manifest.json, boundaries/window_*.npy, residual_streams/window_*.npy) "
        "or the fallback Layer-13 recapture path. JIT metadata path: "
        f"{jit_metadata.get('metadata_path', '<not indexed yet>')}; "
        "activation route dir: "
        f"{jit_metadata.get('activation_route_dir', '<not indexed yet>')}. "
        "Re-run with --prediction-mode row_prediction or "
        "--prediction-mode gold_patch if KV-direct is not required."
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
    source_layer: int | None = None,
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
    source_layer = 13 if source_layer is None else int(source_layer)
    if source_layer >= len(layers):
        raise Phase3Unavailable(
            f"Layer-{source_layer} recapture unavailable: model exposes only {len(layers)} layers."
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
        archive_provenance[window_id] = f"jit_recapture://layer{source_layer}/in_vram"
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


def _phase3_bool_arg(args: argparse.Namespace, name: str) -> bool:
    return bool(getattr(args, name, False))


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise Phase3Unavailable(f"Expected JSON object at {path}")
    return data


def _phase3_jit_metadata(args: argparse.Namespace) -> dict[str, Any]:
    jit = dict(getattr(args, "jit_indexing", {}) or {})
    metadata_path = str(jit.get("metadata_path", "") or "")
    if metadata_path and not jit.get("case_matrices"):
        path = Path(metadata_path)
        if path.exists():
            file_metadata = _read_json_object(path)
            jit = {**file_metadata, **jit}
            if file_metadata.get("case_matrices") and not jit.get("case_matrices"):
                jit["case_matrices"] = file_metadata.get("case_matrices")
    return jit


def _explicit_phase3_apollo_case_metadata(args: argparse.Namespace) -> dict[str, Any] | None:
    for name in (
        "_current_jit_case_metadata",
        "_phase3_jit_case_metadata",
        "_phase3_apollo_case_metadata",
        "phase3_apollo_case_metadata",
    ):
        value = getattr(args, name, None)
        if isinstance(value, dict):
            return dict(value)

    for name in (
        "_phase3_apollo_manifest_path",
        "phase3_apollo_manifest_path",
        "apollo_manifest_path",
    ):
        raw = str(getattr(args, name, "") or "")
        if raw:
            manifest_path = Path(raw)
            return {
                "apollo_manifest_path": str(manifest_path),
                "store_dir": str(manifest_path.parent),
                "case_index": getattr(args, "_current_row_index", None),
            }

    for name in ("_phase3_apollo_store_dir", "phase3_apollo_store_dir", "apollo_store_dir"):
        raw = str(getattr(args, name, "") or "")
        if raw:
            store_dir = Path(raw)
            return {
                "store_dir": str(store_dir),
                "apollo_manifest_path": str(store_dir / "manifest.json"),
                "case_index": getattr(args, "_current_row_index", None),
            }
    return None


def _phase3_apollo_case_metadata(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    explicit = _explicit_phase3_apollo_case_metadata(args)
    if explicit is not None:
        return explicit, "explicit_override"

    jit = _phase3_jit_metadata(args)
    case_matrices = jit.get("case_matrices", [])
    if not isinstance(case_matrices, list) or not case_matrices:
        return None, "jit_metadata_missing_case_matrices"

    current_row = getattr(args, "_current_row_index", None)
    selected_index: int | None = None
    row_map = getattr(args, "_jit_case_index_by_row", None)
    if isinstance(row_map, dict) and current_row is not None:
        for key in (current_row, str(current_row)):
            if key in row_map:
                selected_index = int(row_map[key])
                break

    if selected_index is None and current_row is not None:
        try:
            row_index = int(current_row)
        except (TypeError, ValueError):
            row_index = -1
        if 0 <= row_index < len(case_matrices):
            selected_index = row_index
        else:
            for index, item in enumerate(case_matrices):
                if isinstance(item, dict) and int(item.get("case_index", -1)) == row_index:
                    selected_index = index
                    break

    if selected_index is None and len(case_matrices) == 1:
        selected_index = 0

    if selected_index is None or not (0 <= selected_index < len(case_matrices)):
        return None, f"cannot_map_row_to_jit_case:{current_row!r}"
    selected = case_matrices[selected_index]
    if not isinstance(selected, dict):
        return None, f"jit_case_metadata_not_object:{selected_index}"
    return dict(selected), "jit_case_matrices"


def _phase3_runtime_device(runtime: Any) -> str:
    model = runtime._model
    try:
        return str(next(model.parameters()).device)
    except Exception:  # noqa: BLE001 - tests and tiny stubs may be parameterless.
        return "cpu"


def _phase3_runtime_layer_count(runtime: Any) -> int:
    try:
        return len(runtime._resolve_layers())
    except Exception as exc:  # noqa: BLE001
        raise Phase3Unavailable(
            f"Apollo residual validation failed: cannot resolve runtime layers: {exc}"
        ) from exc


def _apollo_manifest_source_layer(manifest: dict[str, Any]) -> int:
    for key in ("source_layer", "injection_layer", "layer", "crystal_layer"):
        if key in manifest:
            return int(manifest[key])
    arch_config = manifest.get("arch_config", {})
    if isinstance(arch_config, dict) and "injection_layer" in arch_config:
        return int(arch_config["injection_layer"])
    raise Phase3Unavailable("Apollo manifest is missing source_layer/injection_layer.")


def _load_selected_apollo_residuals(
    *,
    args: argparse.Namespace,
    runtime: Any,
    tokenizer: Any,
    hot_windows: list[DependencyWindow],
    case_metadata: dict[str, Any],
    case_metadata_source: str,
) -> tuple[Any, dict[int, list[int]], dict[str, Any]]:
    from chuk_lazarus.inference.backends._torch_residual_bounded import (
        GatheredResiduals,
        gather_selected_residuals,
    )
    from chuk_lazarus.inference.context.knowledge.torch_store import TorchKnowledgeStore
    from chuk_lazarus.session_retrieval.tier_policy import TierLabel

    manifest_raw = str(case_metadata.get("apollo_manifest_path", "") or "")
    store_raw = str(case_metadata.get("store_dir", "") or "")
    if not manifest_raw and store_raw:
        manifest_raw = str(Path(store_raw) / "manifest.json")
    if not manifest_raw:
        raise Phase3Unavailable("Apollo case metadata has no apollo_manifest_path.")
    manifest_path = Path(manifest_raw)
    store_dir = Path(store_raw) if store_raw else manifest_path.parent
    if not manifest_path.exists():
        raise Phase3Unavailable(f"Apollo manifest is missing: {manifest_path}")
    manifest = _read_json_object(manifest_path)
    if manifest.get("kind") != APOLLO_PHASE3_KIND:
        raise Phase3Unavailable(
            "Apollo manifest kind mismatch: "
            f"{manifest.get('kind')!r} != {APOLLO_PHASE3_KIND!r}"
        )
    if manifest.get("status") != "ready" or manifest.get("apollo_ready") is not True:
        raise Phase3Unavailable(
            "Apollo manifest is not ready: "
            f"status={manifest.get('status')!r} apollo_ready={manifest.get('apollo_ready')!r}"
        )

    num_windows = int(manifest.get("num_windows", -1))
    selected_ids = [int(window.window_id) for window in hot_windows]
    strict_apollo = _phase3_bool_arg(args, "require_phase3_apollo_store")
    apollo_windows = [
        window
        for window in hot_windows
        if 0 <= int(window.window_id) < int(num_windows)
    ]
    recapture_windows = [
        window
        for window in hot_windows
        if int(window.window_id) < 0 or int(window.window_id) >= int(num_windows)
    ]
    if strict_apollo and recapture_windows:
        raise Phase3Unavailable(
            "Apollo residual store cannot map selected synthetic/out-of-range window ids: "
            + ",".join(str(int(window.window_id)) for window in recapture_windows)
        )
    if not apollo_windows:
        raise Phase3Unavailable(
            "Apollo residual store cannot map any selected window ids: "
            + ",".join(str(window_id) for window_id in selected_ids)
        )

    source_layer = _apollo_manifest_source_layer(manifest)
    arch_config = manifest.get("arch_config", {})
    if not isinstance(arch_config, dict):
        raise Phase3Unavailable("Apollo manifest arch_config is missing or invalid.")
    arch_injection_layer = int(arch_config.get("injection_layer", source_layer))
    if source_layer != arch_injection_layer:
        raise Phase3Unavailable(
            "Apollo source_layer does not match arch_config injection_layer: "
            f"{source_layer} != {arch_injection_layer}"
        )

    layer_count = _phase3_runtime_layer_count(runtime)
    if source_layer < 0 or source_layer + 1 >= layer_count:
        raise Phase3Unavailable(
            "Apollo source_layer is incompatible with runtime layer stack: "
            f"source_layer={source_layer} layer_count={layer_count}"
        )

    store = TorchKnowledgeStore.load(store_dir)
    store_injection_layer = int(store.config.injection_layer)
    if source_layer != store_injection_layer:
        raise Phase3Unavailable(
            "Apollo source_layer does not match store.config.injection_layer: "
            f"{source_layer} != {store_injection_layer}"
        )
    if num_windows >= 0 and int(store.num_windows) < num_windows:
        raise Phase3Unavailable(
            f"Apollo store num_windows={store.num_windows} is smaller than manifest "
            f"num_windows={num_windows}"
        )

    streams_dir = store_dir / str(manifest.get("residual_streams_dir", "residual_streams"))
    apollo_window_ids = [int(window.window_id) for window in apollo_windows]
    missing_streams = [
        window_id
        for window_id in apollo_window_ids
        if not (streams_dir / f"window_{window_id:03d}.npy").exists()
    ]
    residual_mode = "boundary" if missing_streams else "stream"
    assignments = [
        SimpleNamespace(
            candidate=SimpleNamespace(window_id=int(window_id)),
            tier=TierLabel.HOT,
            rank=index,
            policy_version="apollo-phase3",
            policy_params={},
        )
        for index, window_id in enumerate(apollo_window_ids)
    ]
    checkpoint_handle = SimpleNamespace(
        checkpoint_dir=store_dir,
        torch_store_dir=store_dir,
        session_id=f"locobench-row-{getattr(args, '_current_row_index', 'unknown')}",
        manifest=manifest,
        original_input_dir=None,
    )
    residuals = gather_selected_residuals(
        assignments,
        checkpoint_handle=checkpoint_handle,
        store=store,
        source_layer=source_layer,
        include_tiers=(TierLabel.HOT,),
        device=_phase3_runtime_device(runtime),
        residual_mode=residual_mode,
    )
    if not residuals.per_window_residuals:
        raise Phase3Unavailable("Apollo residual store produced no selected residuals.")

    encoded_by_window: dict[int, list[int]] = {}
    for window in apollo_windows:
        window_id = int(window.window_id)
        token_ids = list(getattr(store, "window_token_lists", {}).get(window_id, []) or [])
        if not token_ids:
            token_ids = tokenizer.encode(
                window.text,
                add_special_tokens=False,
                truncation=True,
                max_length=max(1, int(args.window_tokens)),
            )
        encoded_by_window[window_id] = [int(token_id) for token_id in token_ids]

    recapture_meta: dict[str, Any] = {}
    if recapture_windows:
        recaptured, recaptured_encoded, recapture_meta = recapture_layer13_residuals(
            args=args,
            runtime=runtime,
            tokenizer=tokenizer,
            hot_windows=recapture_windows,
            source_layer=source_layer,
        )
        encoded_by_window.update(recaptured_encoded)
        per_window_residuals: dict[int, Any] = {}
        per_window_token_ranges: dict[int, tuple[int, int]] = {}
        archive_provenance: dict[int, str] = {}
        slot = 0
        for window in hot_windows:
            window_id = int(window.window_id)
            if window_id in residuals.per_window_residuals:
                tensor = residuals.per_window_residuals[window_id]
                provenance = residuals.archive_provenance.get(window_id, "")
            else:
                tensor = recaptured.per_window_residuals[window_id]
                provenance = recaptured.archive_provenance.get(window_id, "")
            per_window_residuals[window_id] = tensor
            slot_count = 1
            dim_fn = getattr(tensor, "dim", None)
            if callable(dim_fn):
                rank = int(dim_fn())
                if rank == 1:
                    slot_count = 1
                elif rank >= 2:
                    slot_count = int(tensor.shape[-2])
            per_window_token_ranges[window_id] = (slot, slot + slot_count)
            archive_provenance[window_id] = provenance
            slot += slot_count
        residuals = GatheredResiduals(
            per_window_residuals=per_window_residuals,
            per_window_token_ranges=per_window_token_ranges,
            source_layer=int(source_layer),
            archive_provenance=archive_provenance,
        )

    token_counts = {
        str(window_id): len(token_ids)
        for window_id, token_ids in encoded_by_window.items()
    }
    metadata = {
        "source": "apollo_store",
        "used_apollo_store": True,
        "fallback_recapture": bool(recapture_windows),
        "partial_recapture": bool(recapture_windows),
        "case_metadata_source": case_metadata_source,
        "store_dir": str(store_dir),
        "manifest_path": str(manifest_path),
        "source_layer": int(source_layer),
        "target_layer": int(source_layer) + 1,
        "selected_window_ids": selected_ids,
        "selected_window_count": len(selected_ids),
        "apollo_window_ids": apollo_window_ids,
        "apollo_window_count": len(apollo_window_ids),
        "recaptured_window_ids": [int(window.window_id) for window in recapture_windows],
        "recaptured_window_count": len(recapture_windows),
        "recapture": recapture_meta,
        "residual_mode": residual_mode,
        "stream_missing_window_ids": missing_streams,
        "window_token_counts": token_counts,
        "artifact_token_count": int(sum(token_counts.values())),
        "manifest_num_windows": int(num_windows),
        "store_num_windows": int(store.num_windows),
        "row_alignment": manifest.get(
            "row_alignment",
            (manifest.get("activation_routes") or {}).get("row_alignment", ""),
        ),
    }
    return residuals, encoded_by_window, metadata


def load_or_recapture_phase3_residuals(
    *,
    args: argparse.Namespace,
    runtime: Any,
    tokenizer: Any,
    hot_windows: list[DependencyWindow],
) -> tuple[Any, dict[int, list[int]], dict[str, Any]]:
    if not hot_windows:
        raise Phase3Unavailable("Phase 3 KV-direct requested, but no hot windows were selected.")
    strict_apollo = _phase3_bool_arg(args, "require_phase3_apollo_store")
    case_metadata, case_metadata_source = _phase3_apollo_case_metadata(args)
    apollo_error = ""
    if case_metadata is not None:
        try:
            return _load_selected_apollo_residuals(
                args=args,
                runtime=runtime,
                tokenizer=tokenizer,
                hot_windows=hot_windows,
                case_metadata=case_metadata,
                case_metadata_source=case_metadata_source,
            )
        except Phase3Unavailable as exc:
            apollo_error = str(exc)
            if strict_apollo:
                raise
    elif strict_apollo:
        raise Phase3Unavailable(
            "Ready Apollo residual store is required, but no case metadata was found: "
            f"{case_metadata_source}"
        )

    residuals, encoded_by_window, metadata = recapture_layer13_residuals(
        args=args,
        runtime=runtime,
        tokenizer=tokenizer,
        hot_windows=hot_windows,
    )
    metadata = {
        **metadata,
        "source": "recapture",
        "used_apollo_store": False,
        "fallback_recapture": True,
        "apollo_case_metadata_source": case_metadata_source,
        "apollo_unavailable_reason": apollo_error
        or "apollo_case_metadata_unavailable",
    }
    return residuals, encoded_by_window, metadata


def kv_direct_prediction(
    case: LoCoBenchCase,
    args: argparse.Namespace,
    *,
    hot_windows: list[DependencyWindow],
    anchored_query: str,
    prompt: str,
    dialect_plan: dict[str, Any] | None = None,
) -> Phase3Prediction:
    from chuk_lazarus.inference.backends._torch_residual_bounded import materialize_kv_direct
    from chuk_lazarus.inference.backends.torch_runtime import WarmPenaltyConfig
    from chuk_lazarus.inference.generation import GenerationConfig
    from chuk_lazarus.session_retrieval.tier_policy import TierLabel

    tokenizer, _model, runtime = get_phase3_runtime(args)
    injected_windows = list(hot_windows)
    started = time.perf_counter()
    residuals, encoded_by_window, residual_meta = load_or_recapture_phase3_residuals(
        args=args,
        runtime=runtime,
        tokenizer=tokenizer,
        hot_windows=injected_windows,
    )
    source_layer = int(residuals.source_layer)
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
        +
        (
            f"Apollo residual store loaded for {len(hot_windows)} Hot Windows. "
            if residual_meta.get("used_apollo_store")
            else f"JIT Re-Capture fallback successful for {len(hot_windows)} Hot Windows. "
        )
        +
        f"Materializing Layer {source_layer + 1} Injector.",
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
    except Exception as exc:  # noqa: BLE001 - fail closed; protocol must stay system-only.
        raise RuntimeError(
            "Tokenizer chat template with a system role is required for KV-direct benchmark generation."
        ) from exc
    readout_path, readout_span, readout_authority_meta = selected_source_identifier_authority(
        case,
        hot_windows,
    )
    readout_identifier_authority = dict(
        readout_authority_meta.get("identifier_authority", {})
    )
    readout_identifiers = set(readout_identifier_authority.get("identifiers", []))
    readout_identifiers.update(readout_identifier_authority.get("path_identifiers", []))
    injected_identifiers = identifiers_from_windows(hot_windows)
    planned_paths = [
        str(path)
        for path in (dialect_plan or {}).get("paths", {})
        if str(path) in case.files
    ]
    target_paths = list(dict.fromkeys([*planned_paths, *phase3_target_paths(case, hot_windows)]))
    if readout_path and readout_path not in target_paths:
        target_paths.insert(0, readout_path)
    target_path = target_paths[0] if target_paths else ""
    structural_search_block = (
        f"{target_path}\n<<<< SEARCH\n{readout_span.strip()}\n==== REPLACE\n"
        if target_path and readout_span.strip()
        else ""
    )
    emitted_structural_prefix = structural_search_block
    forced_start = "" if emitted_structural_prefix else (
        f"{target_path}\n<<<< SEARCH\n"
        if target_path
        else "<<<< SEARCH\n"
    )
    forced_prefix_ids = [
        int(token_id)
        for token_id in tokenizer.encode(forced_start, add_special_tokens=False)
    ]
    structural_trigger_active = bool(emitted_structural_prefix)
    replacement_budget_tokens = max(0, int(args.kv_max_new_tokens))
    replacement_spillover_tokens = max(
        0,
        int(
            getattr(
                args,
                "kv_replacement_spillover_tokens",
                DEFAULT_KV_REPLACEMENT_SPILLOVER_TOKENS,
            )
        ),
    )
    forced_prefix_token_count = len(forced_prefix_ids)
    emitted_prefix_token_count = (
        len(tokenizer.encode(emitted_structural_prefix, add_special_tokens=False))
        if emitted_structural_prefix
        else 0
    )
    requested_payload_tokens = replacement_budget_tokens + replacement_spillover_tokens
    decode_payload_token_cap = max(
        0,
        int(
            getattr(
                args,
                "kv_decode_payload_token_cap",
                DEFAULT_KV_DECODE_PAYLOAD_TOKEN_CAP,
            )
        ),
    )
    effective_payload_tokens = (
        min(requested_payload_tokens, decode_payload_token_cap)
        if decode_payload_token_cap > 0
        else requested_payload_tokens
    )
    generation_max_new_tokens = forced_prefix_token_count + effective_payload_tokens
    decode_timeout_seconds = max(
        0,
        int(
            getattr(
                args,
                "kv_decode_timeout_seconds",
                DEFAULT_KV_DECODE_TIMEOUT_SECONDS,
            )
        ),
    )
    decode_stall_seconds = max(
        0,
        int(
            getattr(
                args,
                "kv_decode_stall_seconds",
                DEFAULT_KV_DECODE_STALL_SECONDS,
            )
        ),
    )
    decode_heartbeat_seconds = max(
        0,
        int(
            getattr(
                args,
                "kv_decode_heartbeat_seconds",
                DEFAULT_KV_DECODE_HEARTBEAT_SECONDS,
            )
        ),
    )
    target_source_text = case.files.get(target_path, "") or readout_span
    target_nearby_texts = [
        window.text for window in hot_windows if not target_path or window.path == target_path
    ]
    if readout_span:
        target_nearby_texts.append(readout_span)
    dialect_spec = infer_dialect_lock_for_source(
        target_path,
        target_source_text,
        nearby_texts=target_nearby_texts,
    )
    dialect_token_ids = token_ids_for_dialect_lock(tokenizer, dialect_spec)
    dialect_token_ids_by_path: dict[str, set[int]] = {}
    dialect_specs_by_path: dict[str, DialectLockSpec | None] = {}
    for path in target_paths:
        path_nearby_texts = [window.text for window in hot_windows if window.path == path]
        path_spec = infer_dialect_lock_for_source(
            path,
            case.files.get(path, ""),
            nearby_texts=path_nearby_texts,
        )
        dialect_specs_by_path[path] = path_spec
        dialect_token_ids_by_path[path] = token_ids_for_dialect_lock(tokenizer, path_spec)
    if target_path and target_path not in dialect_specs_by_path:
        dialect_specs_by_path[target_path] = dialect_spec
    lock_safe_identifiers = readout_identifiers | injected_identifiers
    style_token_ids = (
        token_ids_for_c11_case_lock(tokenizer, lock_safe_identifiers)
        if dialect_spec is not None and dialect_spec.enforce_c11_case
        else set()
    )
    if dialect_spec is not None:
        print(
            "PHASE 4.3 DIALECT LOCK: "
            f"target={target_paths[0] if target_paths else 'unknown'} "
            f"dialect={dialect_spec.dialect} "
            f"inference={dialect_spec.inference_mode} "
            f"confidence={dialect_spec.confidence:.3f} "
            f"forbidden_token_ids={len(dialect_token_ids)} "
            f"style_token_ids={len(style_token_ids)}",
            flush=True,
        )
    logit_lock = LogitLockedDiffProcessor(
        tokenizer=tokenizer,
        safe_token_ids=safe_token_ids_from_windows(
            tokenizer,
            hot_windows,
            extra_texts=(readout_path, readout_span),
            extra_identifiers=readout_identifiers | injected_identifiers,
        ),
        forced_prefix_ids=forced_prefix_ids,
        authority_identifiers=lock_safe_identifiers,
        banned_token_ids=token_ids_for_literals(
            tokenizer,
            (
                "...",
                " ...",
                "…",
                "// ...",
                "# ...",
                "other includes",
                "other code",
            ),
        ),
        pre_replace_token_ids=token_ids_for_literals(
            tokenizer,
            (
                "/*",
                "/**",
                "//",
                '"""',
                "'''",
                "```",
            ),
        ),
        dialect_token_ids=dialect_token_ids,
        dialect_token_ids_by_path=dialect_token_ids_by_path,
        style_token_ids=style_token_ids,
        protocol_file_paths=set(target_paths) | ({readout_path} if readout_path else set()),
        dialect_replace_region_only=(
            dialect_spec.replace_region_only if dialect_spec is not None else True
        ),
        forced_token_count=len(forced_prefix_ids),
    )
    steering_family_token_ids_by_path = {
        path: family_token_ids
        for path, family_token_ids in (
            (path, dialect_family_token_ids_for_spec(tokenizer, spec))
            for path, spec in dialect_specs_by_path.items()
        )
        if family_token_ids
    }
    steering_dialects_by_path = {
        path: spec.dialect
        for path, spec in dialect_specs_by_path.items()
        if spec is not None
    }
    steering_anchor_token_ids_by_dialect = {
        dialect: dialect_target_anchor_token_ids(tokenizer, dialect)
        for dialect in set(steering_dialects_by_path.values())
    }
    decoder_steering = DecoderDialectSteeringState.from_args(
        args,
        tokenizer=tokenizer,
        model=_model,
        layer_index=source_layer + 1,
        default_path=target_path,
        dialects_by_path=steering_dialects_by_path,
        family_token_ids_by_path=steering_family_token_ids_by_path,
        anchor_token_ids_by_dialect=steering_anchor_token_ids_by_dialect,
    )
    if decoder_steering.requested_enabled:
        print(
            "PHASE 4.6 DECODER DIALECT STEERING: "
            f"status={decoder_steering.status} "
            f"layer={source_layer + 1} "
            f"vectors={len(decoder_steering.vectors)} "
            f"paths={len(steering_family_token_ids_by_path)}",
            flush=True,
        )
    original_sampler = runtime._sample_next_token
    close_marker_ids = [
        int(token_id)
        for token_id in tokenizer.encode("\n>>>>", add_special_tokens=False)
    ]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    repetition_close_ids = list(close_marker_ids)
    if eos_token_id is not None:
        repetition_close_ids.append(int(eos_token_id))
    sample_step: dict[str, Any] = {
        "value": 0,
        "generated_ids": [],
        "forced_bridge_ids": [],
        "forced_close_ids": [],
        "magnetized_bridge": {
            "active": False,
            "trigger_count": 0,
        },
        "repetition_guard": {"active": False},
        "started_at": time.perf_counter(),
        "last_token_at": time.perf_counter(),
        "last_text_at": time.perf_counter(),
        "last_text_len": 0,
        "watchdog_abort_reason": "",
    }
    decode_safety_meta: dict[str, Any] = {
        "active": bool(
            decode_timeout_seconds
            or decode_stall_seconds
            or decode_heartbeat_seconds
            or (
                decode_payload_token_cap > 0
                and requested_payload_tokens > effective_payload_tokens
            )
        ),
        "wall_timeout_seconds": int(decode_timeout_seconds),
        "stall_seconds": int(decode_stall_seconds),
        "heartbeat_seconds": int(decode_heartbeat_seconds),
        "requested_payload_tokens": int(requested_payload_tokens),
        "payload_token_cap": int(decode_payload_token_cap),
        "effective_payload_tokens": int(effective_payload_tokens),
        "payload_cap_applied": bool(requested_payload_tokens > effective_payload_tokens),
        "forced_prefix_token_count": int(forced_prefix_token_count),
        "emitted_prefix_token_count": int(emitted_prefix_token_count),
        "generation_max_new_tokens": int(generation_max_new_tokens),
        "heartbeat_count": 0,
        "triggered": False,
        "reason": "",
        "generated_token_count": 0,
        "generated_char_count": 0,
        "synthetic_close_marker_appended": False,
    }
    watchdog_stop = threading.Event()

    def mark_decode_safety_triggered(reason: str) -> None:
        if decode_safety_meta.get("triggered"):
            return
        now = time.perf_counter()
        generated_ids = list(sample_step.get("generated_ids", []))
        try:
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
        except Exception:  # noqa: BLE001 - metadata only.
            generated_text = ""
        decode_safety_meta.update(
            {
                "triggered": True,
                "reason": reason,
                "elapsed_seconds": now - float(sample_step["started_at"]),
                "generated_token_count": len(generated_ids),
                "generated_char_count": len(generated_text),
            }
        )
        sample_step["watchdog_abort_reason"] = reason

    def emitted_completion_text(payload: str, *, close_if_payload: bool) -> str:
        text = f"{emitted_structural_prefix}{payload or ''}"
        if not emitted_structural_prefix:
            return text
        has_close_after_replace = bool(
            re.search(r"====\s*REPLACE(?:(?!>>>>).)*>>>>", text, flags=re.IGNORECASE | re.DOTALL)
        )
        if close_if_payload and (payload or "").strip() and not has_close_after_replace:
            decode_safety_meta["synthetic_close_marker_appended"] = True
            return text.rstrip() + "\n>>>>"
        return text

    def close_partial_generation_text() -> str:
        generated_ids = list(sample_step.get("generated_ids", []))
        try:
            payload = tokenizer.decode(generated_ids, skip_special_tokens=True)
        except Exception:  # noqa: BLE001 - tokenizer decode should not hide failure.
            payload = ""
        text = emitted_completion_text(payload, close_if_payload=True)
        needs_close = (
            bool(text)
            and not emitted_structural_prefix
            and ">>>>" not in text
            and (
                bool(re.search(r"<<<<\s*SEARCH", text, flags=re.IGNORECASE))
                or "====" in text
            )
        )
        if needs_close:
            text = text.rstrip() + "\n>>>>"
            decode_safety_meta["synthetic_close_marker_appended"] = True
        return text

    def emit_decode_heartbeat(reason: str) -> None:
        if decode_heartbeat_seconds <= 0:
            return
        now = time.perf_counter()
        elapsed = now - float(sample_step["started_at"])
        generated_ids = list(sample_step.get("generated_ids", []))
        heartbeat = {
            "reason": reason,
            "elapsed_seconds": elapsed,
            "generated_tokens": len(generated_ids),
            "max_new_tokens": int(generation_max_new_tokens),
            "seconds_since_token": now - float(sample_step["last_token_at"]),
            "seconds_since_text": now - float(sample_step["last_text_at"]),
        }
        decode_safety_meta["heartbeat_count"] = (
            int(decode_safety_meta.get("heartbeat_count", 0)) + 1
        )
        decode_safety_meta["last_heartbeat"] = heartbeat
        print(
            "PHASE 4.5 DECODE HEARTBEAT: "
            f"reason={reason} tokens={heartbeat['generated_tokens']}/"
            f"{heartbeat['max_new_tokens']} elapsed={elapsed:.1f}s "
            f"since_token={heartbeat['seconds_since_token']:.1f}s "
            f"since_text={heartbeat['seconds_since_text']:.1f}s",
            flush=True,
        )

    def watchdog_loop() -> None:
        if (
            decode_timeout_seconds <= 0
            and decode_stall_seconds <= 0
            and decode_heartbeat_seconds <= 0
        ):
            return
        next_heartbeat = float(sample_step["started_at"]) + max(
            1,
            decode_heartbeat_seconds,
        )
        while not watchdog_stop.wait(1.0):
            now = time.perf_counter()
            elapsed = now - float(sample_step["started_at"])
            if decode_heartbeat_seconds > 0 and now >= next_heartbeat:
                emit_decode_heartbeat("watchdog")
                next_heartbeat = now + decode_heartbeat_seconds
            if decode_timeout_seconds > 0 and elapsed >= decode_timeout_seconds:
                mark_decode_safety_triggered("wall_clock_timeout")
                print(
                    "PHASE 4.5 DECODE SAFETY: "
                    f"trigger=wall_clock_timeout elapsed={elapsed:.1f}s "
                    f"limit={decode_timeout_seconds}s",
                    flush=True,
                )
                _thread.interrupt_main()
                return
            if (
                decode_stall_seconds > 0
                and int(sample_step["value"]) > 0
                and now - float(sample_step["last_text_at"]) >= decode_stall_seconds
            ):
                mark_decode_safety_triggered("decode_stall")
                print(
                    "PHASE 4.5 DECODE SAFETY: "
                    f"trigger=decode_stall since_text="
                    f"{now - float(sample_step['last_text_at']):.1f}s "
                    f"limit={decode_stall_seconds}s",
                    flush=True,
                )
                _thread.interrupt_main()
                return

    def locked_sampler(logits: Any, config: Any) -> Any:
        try:
            generated_payload_text = tokenizer.decode(
                list(sample_step["generated_ids"]),
                skip_special_tokens=True,
            )
        except Exception:  # noqa: BLE001 - tokenizer decode should not stop generation.
            generated_payload_text = ""
        generated_text = emitted_completion_text(
            generated_payload_text,
            close_if_payload=False,
        )
        now = time.perf_counter()
        if len(generated_payload_text) > int(sample_step["last_text_len"]):
            sample_step["last_text_len"] = len(generated_payload_text)
            sample_step["last_text_at"] = now
        elapsed = now - float(sample_step["started_at"])
        if decode_timeout_seconds > 0 and elapsed >= decode_timeout_seconds:
            mark_decode_safety_triggered("wall_clock_timeout")
            raise KVDecodeSafetyTriggered("wall_clock_timeout")
        forced_bridge_ids = list(sample_step.get("forced_bridge_ids", []))
        if not forced_bridge_ids:
            bridge_decision = magnetized_search_bridge_decision(
                generated_text,
                target_path=target_path,
                target_span=readout_span,
                tokenizer=tokenizer,
            )
            if bridge_decision.get("active"):
                forced_bridge_ids = list(bridge_decision.get("force_transition_token_ids", []))
                sample_step["forced_bridge_ids"] = forced_bridge_ids
                sample_step["magnetized_bridge"] = {
                    **bridge_decision,
                    "trigger_count": int(
                        sample_step.get("magnetized_bridge", {}).get("trigger_count", 0)
                    )
                    + 1,
                    "step": int(sample_step["value"]),
                }
        forced_close_ids = list(sample_step.get("forced_close_ids", []))
        if (
            not forced_bridge_ids
            and
            not forced_close_ids
            and repetition_close_ids
            and decode_stall_seconds > 0
            and int(sample_step["value"]) > 0
            and now - float(sample_step["last_text_at"]) >= decode_stall_seconds
        ):
            mark_decode_safety_triggered("decode_stall")
            forced_close_ids = list(repetition_close_ids)
        if not forced_bridge_ids and not forced_close_ids and repetition_close_ids:
            repetition_meta = repeated_replacement_suffix_metadata(generated_text)
            if repetition_meta is not None:
                sample_step["repetition_guard"] = {
                    **repetition_meta,
                    "forced_close_token_count": len(repetition_close_ids),
                    "step": int(sample_step["value"]),
                }
                forced_close_ids = list(repetition_close_ids)
        steering_skip_reason = ""
        if forced_bridge_ids:
            forced_id = int(forced_bridge_ids.pop(0))
            sample_step["forced_bridge_ids"] = forced_bridge_ids
            scores = logits.new_full(logits.shape, float("-inf"))
            if 0 <= forced_id < int(logits.shape[-1]):
                scores[..., forced_id] = 0.0
                steering_skip_reason = "forced_bridge"
            else:
                scores = logit_lock(
                    logits,
                    step=int(sample_step["value"]),
                    generated_text=generated_text,
                )
        elif forced_close_ids:
            forced_id = int(forced_close_ids.pop(0))
            sample_step["forced_close_ids"] = forced_close_ids
            scores = logits.new_full(logits.shape, float("-inf"))
            if 0 <= forced_id < int(logits.shape[-1]):
                scores[..., forced_id] = 0.0
                steering_skip_reason = "forced_close"
            else:
                scores = logit_lock(
                    logits,
                    step=int(sample_step["value"]),
                    generated_text=generated_text,
                )
        else:
            scores = logit_lock(
                logits,
                step=int(sample_step["value"]),
                generated_text=generated_text,
            )
        next_token = original_sampler(scores, config)
        if steering_skip_reason:
            decoder_steering.observe_clean(steering_skip_reason)
        else:
            decoder_steering.observe(
                logits,
                scores,
                next_token,
                generated_text=generated_text,
                step=int(sample_step["value"]),
                replace_region_active=logit_lock._replace_region_active(generated_text),
                active_path=logit_lock.active_dialect_path or target_path,
            )
        sample_step["value"] += 1
        try:
            token_id = int(next_token.detach().to("cpu").reshape(-1)[0].item())
        except Exception:  # noqa: BLE001 - sampler return types vary.
            try:
                token_id = int(next_token)
            except Exception:  # noqa: BLE001
                token_id = -1
        if token_id >= 0:
            sample_step["generated_ids"].append(token_id)
            sample_step["last_token_at"] = time.perf_counter()
        return next_token

    runtime._sample_next_token = locked_sampler
    watchdog = threading.Thread(
        target=watchdog_loop,
        name="kv-direct-decode-watchdog",
        daemon=True,
    )
    watchdog.start()
    generated = None
    safety_generated_text: str | None = None
    decoder_steering.install(runtime)
    try:
        try:
            generated = runtime.generate_with_kv_direct_materialization(
                generation_prompt,
                GenerationConfig(
                    max_new_tokens=int(generation_max_new_tokens),
                    temperature=0.0,
                    top_p=1.0,
                ),
                materialization=materialization,
                per_window_token_ranges=per_window_ranges,
                tier_assignments=tier_map,
                warm_config=WarmPenaltyConfig(),
                source_layer=source_layer,
                query_id=f"locobench-row-{getattr(args, '_current_row_index', 'unknown')}",
            )
        except KeyboardInterrupt:
            reason = str(sample_step.get("watchdog_abort_reason") or "")
            if not reason:
                raise
            safety_generated_text = close_partial_generation_text()
        except KVDecodeSafetyTriggered as exc:
            mark_decode_safety_triggered(str(exc) or "decode_safety_triggered")
            safety_generated_text = close_partial_generation_text()
    finally:
        decoder_steering.remove()
        watchdog_stop.set()
        watchdog.join(timeout=2.0)
        runtime._sample_next_token = original_sampler
    if generated is None and safety_generated_text is None:
        safety_generated_text = close_partial_generation_text()
    context_whitelist_meta = logit_lock.context_whitelist_metadata()
    decoder_steering_meta = decoder_steering.telemetry()
    family_masked_but_not_tempting_count = int(
        decoder_steering_meta.get("masked_but_not_tempting_count", 0)
    )
    hard_masked_candidate_count = int(
        context_whitelist_meta.get("hard_mask", {}).get("masked_token_total", 0)
        if isinstance(context_whitelist_meta.get("hard_mask", {}), dict)
        else 0
    )
    decoder_trigger_count = int(decoder_steering_meta.get("trigger_count", 0))
    decoder_steering_meta["family_masked_but_not_tempting_count"] = (
        family_masked_but_not_tempting_count
    )
    decoder_steering_meta["hard_masked_candidate_count"] = hard_masked_candidate_count
    decoder_steering_meta["masked_but_not_tempting_count"] = max(
        family_masked_but_not_tempting_count,
        hard_masked_candidate_count - decoder_trigger_count,
    )
    decoder_candidate_step_count = int(decoder_steering_meta.get("candidate_step_count", 0))
    decoder_applied_count = int(decoder_steering_meta.get("applied_count", 0))
    decoder_step_denominator = max(1, decoder_candidate_step_count)
    decoder_steering_meta["temptation_rate_per_candidate_step"] = (
        float(decoder_trigger_count) / float(decoder_step_denominator)
    )
    decoder_steering_meta["steering_application_rate_per_candidate_step"] = (
        float(decoder_applied_count) / float(decoder_step_denominator)
    )
    decoder_steering_meta["hard_masked_candidates_per_candidate_step"] = (
        float(hard_masked_candidate_count) / float(decoder_step_denominator)
    )
    decoder_steering_meta["masked_but_not_tempting_ratio"] = (
        0.0
        if hard_masked_candidate_count <= 0
        else float(decoder_steering_meta["masked_but_not_tempting_count"])
        / float(hard_masked_candidate_count)
    )
    decoder_steering_meta["learning_score_hint"] = (
        "Compare temptation_rate_per_candidate_step across matched same-dialect cases; "
        "hard_masked_candidate_count is protocol pruning and should not be treated as "
        "the learned-drift score."
    )
    decoder_steering_meta["masked_but_not_tempting_basis"] = (
        "max(family_non_tempting_candidates, hard_masked_candidates_minus_decoder_triggers)"
    )
    context_whitelist_meta["decoder_dialect_steering"] = decoder_steering_meta
    if context_whitelist_meta["active"]:
        print(
            "PHASE 4.4 CONTEXT WHITELIST: "
            f"blocked {context_whitelist_meta['intervention_count']} "
            "zero-context token candidates.",
            flush=True,
        )
    if decoder_steering_meta.get("requested"):
        print(
            "PHASE 4.6 DECODER DIALECT STEERING: "
            f"triggers={decoder_steering_meta.get('trigger_count', 0)} "
            f"applied={decoder_steering_meta.get('applied_count', 0)} "
            "masked_but_not_tempting="
            f"{decoder_steering_meta.get('masked_but_not_tempting_count', 0)}",
            flush=True,
        )
    final_payload_text = stringify(getattr(generated, "text", "")) if generated is not None else ""
    final_text = (
        safety_generated_text
        if safety_generated_text is not None
        else emitted_completion_text(final_payload_text, close_if_payload=True)
    )
    decode_safety_meta.update(
        {
            "completed": not bool(decode_safety_meta.get("triggered")),
            "generated_token_count": len(sample_step.get("generated_ids", [])),
            "generated_char_count": len(final_text),
        }
    )
    metadata = dict(getattr(generated, "metadata", {}) or {}) if generated is not None else {}
    metadata.update(
        {
            "engine": "runtime.generate_with_kv_direct_materialization",
            "recapture": residual_meta,
            "residual_acquisition": residual_meta,
            "phase4": {
                "zero_prose_prompt": True,
                "pattern_window_id": None,
                "pattern_window_tier": "disabled_system_prompt_only",
                "forced_start": forced_start,
                "emitted_structural_prefix": emitted_structural_prefix,
                "forced_prefix_token_ids": forced_prefix_ids,
                "structural_trigger": {
                    "active": structural_trigger_active,
                    "mode": (
                        "emitted_verbatim_search_then_replace_separator"
                        if structural_trigger_active
                        else "path_search_prefix_only"
                    ),
                    "forced_search_block_chars": len(structural_search_block),
                    "forced_search_block_lines": (
                        len(structural_search_block.splitlines())
                        if structural_search_block
                        else 0
                    ),
                    "replacement_budget_tokens": replacement_budget_tokens,
                    "replacement_spillover_tokens": replacement_spillover_tokens,
                    "requested_payload_tokens": int(requested_payload_tokens),
                    "decode_payload_token_cap": int(decode_payload_token_cap),
                    "effective_payload_tokens": int(effective_payload_tokens),
                    "payload_cap_applied": bool(
                        requested_payload_tokens > effective_payload_tokens
                    ),
                    "forced_prefix_token_count": forced_prefix_token_count,
                    "emitted_prefix_token_count": int(emitted_prefix_token_count),
                    "generation_max_new_tokens": int(generation_max_new_tokens),
                },
                "unsafe_token_id_count": len(logit_lock.unsafe_token_ids),
                "banned_token_id_count": len(logit_lock.banned_token_ids),
                "pre_replace_token_id_count": len(logit_lock.pre_replace_token_ids),
                "safe_token_id_count": len(logit_lock.safe_token_ids),
                "dialect_lock": {
                    **dialect_lock_telemetry(dialect_spec),
                    "target_path": target_paths[0] if target_paths else "",
                    "forbidden_token_id_count": len(logit_lock.dialect_token_ids),
                    "style_token_id_count": len(logit_lock.style_token_ids),
                },
                "context_whitelist": context_whitelist_meta,
                "decoder_dialect_steering": decoder_steering_meta,
                "lsp_bias": context_whitelist_meta.get("lsp_bias", {}),
                "hard_mask": context_whitelist_meta.get("hard_mask", {}),
                "decode_safety": decode_safety_meta,
                "magnetized_state_bridge": sample_step.get("magnetized_bridge", {}),
                "repetition_guard": sample_step.get("repetition_guard", {}),
                "path_start_enforced": bool(target_paths),
                "selected_source_identifier_authority": readout_authority_meta,
                "injected_identifier_count": len(injected_identifiers),
                "injected_identifier_authority": sorted(injected_identifiers),
            },
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
        text=final_text,
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
        overlap_tokens=int(args.activation_overlap_tokens),
        activation_route_dir=args.jit_output_dir / "locobench_source_activation_routes",
        case_id=int(case.row_index),
        routing_backend=getattr(args, "routing_backend", "native"),
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
                        "start_line": window.start_line,
                        "end_line": window.end_line,
                        "start_token": window.start_token,
                        "end_token": window.end_token,
                        "activation_route_id": window.activation_route_id,
                        "lexical_score": window.lexical_score,
                        "activation_score": window.activation_score,
                        "route_source": window.route_source,
                        "exact_scope_match": window.exact_scope_match,
                        "recursive_map_window_id": window.recursive_map_window_id,
                        "recursive_pass": window.recursive_pass,
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
    raw_generated = generated
    canonicalized_generated = canonicalize_search_replace_output(generated)
    generated, verbatim_bridge = verbatim_readout_bridge(
        case,
        hot_windows,
        canonicalized_generated,
    )
    patch_result = apply_search_replace_blocks(case.files, generated)
    diff_diagnostic = diff_format_diagnostic(generated, patch_result)
    bridge_identifier_authority = dict(
        verbatim_bridge.get("identifier_authority", {})
        if verbatim_bridge.get("active")
        else {}
    )
    if bridge_identifier_authority:
        dependency_safe_identifiers = set(bridge_identifier_authority.get("identifiers", []))
        dependency_safe_identifiers.update(
            bridge_identifier_authority.get("path_identifiers", [])
        )
        dependency_safe_identifiers.update(identifiers_from_windows(hot_windows))
    else:
        dependency_safe_identifiers = set(safe_identifiers)
    target_path = (
        str(bridge_identifier_authority.get("path", "") or "")
        or (patch_result.changed_files[0] if patch_result.changed_files else "")
        or (phase3_target_paths(case, hot_windows)[0] if phase3_target_paths(case, hot_windows) else "")
    )
    target_source_text = case.files.get(target_path, "")
    target_nearby_texts = [
        window.text for window in hot_windows if not target_path or window.path == target_path
    ]
    dialect_spec = infer_dialect_lock_for_source(
        target_path,
        target_source_text,
        nearby_texts=target_nearby_texts,
    )
    unsafe_identifiers = unsafe_identifier_violations(
        generated,
        dependency_safe_identifiers,
    )
    dialect_violations = dialect_literal_violations(generated, dialect_spec)
    c11_style_violations_found = c11_case_style_violations(
        generated,
        dependency_safe_identifiers,
        dialect_spec,
    )
    dependency_constraint_pass = not (
        unsafe_identifiers or dialect_violations or c11_style_violations_found
    )
    passed = bool(patch_result.applied and dependency_constraint_pass)
    phase2_map_windows = [
        window for window in hot_windows if window.recursive_pass == "pass1_map"
    ]
    phase2_hop_windows = [
        window for window in hot_windows if window.recursive_pass == "pass2_hop"
    ]
    return {
        "row_index": int(case.row_index),
        "instruction": case.instruction,
        "generated": generated,
        "raw_generated": raw_generated,
        "format_canonicalized": canonicalized_generated != raw_generated,
        "verbatim_readout_bridge": verbatim_bridge,
        "expected_patch_present": bool(case.expected_patch),
        "pass_at_1": passed,
        "patch_applied": bool(patch_result.applied),
        "patch_block_count": int(patch_result.block_count),
        "changed_files": list(patch_result.changed_files),
        "patch_failures": list(patch_result.failures),
        "diff_format": diff_diagnostic,
        "dependency_constraint_pass": bool(dependency_constraint_pass),
        "unsafe_identifiers": unsafe_identifiers,
        "dialect_lock": {
            **dialect_lock_telemetry(dialect_spec),
            "target_path": target_path,
            "literal_violations": dialect_violations,
            "style_violations": c11_style_violations_found,
        },
        "ttft_ms": float(ttft_ms),
        "dynamic_max_k": int(max_k),
        "hot_window_ids": [int(window.window_id) for window in hot_windows],
        "hot_window_paths": [window.path for window in hot_windows],
        "hot_window_route_sources": [window.route_source for window in hot_windows],
        "router_backend": str(getattr(args, "routing_backend", "native")),
        "phase2_multi_hop": {
            "active": bool(phase2_map_windows and phase2_hop_windows),
            "blend": {"query_weight": 0.4, "map_weight": 0.6, "normalized": True},
            "map_window_ids": [int(window.window_id) for window in phase2_map_windows],
            "map_paths": [window.path for window in phase2_map_windows],
            "hop_window_ids": [int(window.window_id) for window in phase2_hop_windows],
            "hop_paths": [window.path for window in phase2_hop_windows],
            "injected_identifier_count": len(identifiers_from_windows(hot_windows)),
        },
        "hot_window_spans": [
            {
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
            for window in hot_windows
        ],
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
    args._jit_case_index_by_row = {
        int(case.row_index): int(case_index)
        for case_index, case in enumerate(cases)
    }
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
