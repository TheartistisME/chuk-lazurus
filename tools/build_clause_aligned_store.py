#!/usr/bin/env python3
"""Build a clause-aligned torch knowledge store from any directory of per-clause JSON files.

Each input JSON must contain: standard_id, standard_title, clause_id, clause_title, clause_content.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chuk_lazarus.cli.commands.context.prefill._torch_sidecar import (
    TORCH_PREFILL_FILE,
    TORCH_PREFILL_VERSION,
    TORCH_STORE_DIR,
    _load_model_and_tokenizer,
    _resolve_architecture_config,
)
from chuk_lazarus.inference.context.knowledge.route import TFIDFRouter, _extract_keywords_from_text
from chuk_lazarus.inference.context.knowledge.torch_build import (
    ACTIVATION_ROUTES_DIR,
    ACTIVATION_ROUTES_FILE,
    BOUNDARIES_DIR,
    BOUNDARY_RESIDUAL_FILE,
    ENTRIES_FILE,
    IDF_FILE,
    KEYWORDS_FILE,
    MANIFEST_FILE,
    RESIDUAL_STREAMS_DIR,
    STORE_VERSION,
    WINDOW_TOKEN_LISTS_FILE,
    WINDOW_TOKENS_FILE,
    _EntryRecord,
    _entries_to_numpy,
    _entry_coefficient,
    _encode_token_ids,
    _model_id_from_model,
    _save_npz_mapping,
    _select_targets,
    _write_activation_routes,
)
from chuk_lazarus.inference.context.knowledge.torch_capture import capture_window_boundaries
from chuk_lazarus.inference.context.knowledge.torch_store import TorchKnowledgeStore


DEFAULT_MODEL = Path(
    "/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/"
    "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
)
WINDOW_METADATA_FILE = "window_metadata.json"
BUILD_MANIFEST_FILE = "clause_aligned_build_manifest.json"

ZERO_WIDTH_CHARS = {
    "\ufeff",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
}

TEXT_REPLACEMENTS = str.maketrans(
    {
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": " - ",
        "\u2026": "...",
    }
)

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
BULLET_RE = re.compile(r"^(\s*)[\-*•●◦○▪■]\s+")
MULTISPACE_RE = re.compile(r" {2,}")


@dataclass(frozen=True)
class ClauseRecord:
    source_file: str
    standard_id: str
    standard_title: str
    clause_id: str
    clause_title: str
    clause_content: str
    content_was_empty: bool = False


@dataclass(frozen=True)
class ClauseWindow:
    clause_id: str
    clause_title: str
    source_file: str
    part_index: int
    part_count: int
    token_ids: list[int]
    token_count: int
    keywords: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a clause-aligned torch knowledge store from per-clause JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing per-clause JSON files "
            "(each with standard_id/clause_id/clause_title/clause_content)"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Base Gemma model path",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Output checkpoint directory (will be created)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="Maximum tokens per clause-aligned window",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=64,
        help="Overlap between subchunks when a clause exceeds the window budget",
    )
    parser.add_argument(
        "--entries-per-window",
        type=int,
        default=8,
        help="Minimum injection entries per window",
    )
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=12,
        help="Keywords stored per window",
    )
    parser.add_argument(
        "--topic-expansion-tokens",
        type=int,
        default=0,
        help="Optional model topic-expansion tokens to add to retrieval metadata",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for model loading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output checkpoint if it already exists",
    )
    return parser


def natural_sort_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.name.lower())
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)


def strip_bad_controls(text: str) -> str:
    chars: list[str] = []
    for char in text:
        if char in ZERO_WIDTH_CHARS:
            continue
        if char == "\n":
            chars.append(char)
            continue
        if char == "\t":
            chars.append("    ")
            continue
        category = unicodedata.category(char)
        if category in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            continue
        chars.append(char)
    return "".join(chars)


def normalize_line(line: str) -> str:
    line = line.rstrip()
    if not line:
        return ""

    line = HEADING_RE.sub("", line)
    line = BULLET_RE.sub(r"\1- ", line)

    indent_width = len(line) - len(line.lstrip(" "))
    indent = line[:indent_width]
    body = line[indent_width:]
    body = MULTISPACE_RE.sub(" ", body).strip()
    return f"{indent}{body}".rstrip()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("<br>", "; ").replace("<br/>", "; ").replace("<br />", "; ")
    text = strip_bad_controls(text)
    text = text.translate(TEXT_REPLACEMENTS)
    normalized_lines = [normalize_line(line) for line in text.split("\n")]

    collapsed: list[str] = []
    blank_run = 0
    for line in normalized_lines:
        if not line:
            blank_run += 1
            if blank_run <= 1:
                collapsed.append("")
            continue
        blank_run = 0
        collapsed.append(line)

    return "\n".join(collapsed).strip()


def load_clause_record(path: Path) -> ClauseRecord:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON, found {type(data).__name__}")

    standard_id = normalize_text(str(data.get("standard_id", "")).strip())
    standard_title = normalize_text(str(data.get("standard_title", "")).strip())
    clause_id = normalize_text(str(data.get("clause_id", "")).strip())
    clause_title = normalize_text(str(data.get("clause_title", "")).strip())
    raw_clause_content = normalize_text(str(data.get("clause_content", "")).strip())
    clause_content = raw_clause_content or clause_title

    if not all((standard_id, standard_title, clause_id, clause_title)):
        raise ValueError(
            "Missing one or more required fields: "
            "standard_id, standard_title, clause_id, clause_title"
        )

    return ClauseRecord(
        source_file=path.name,
        standard_id=standard_id,
        standard_title=standard_title,
        clause_id=clause_id,
        clause_title=clause_title,
        clause_content=clause_content,
        content_was_empty=not bool(raw_clause_content),
    )


def load_clause_records(input_dir: Path) -> list[ClauseRecord]:
    records: list[ClauseRecord] = []
    for path in sorted(input_dir.glob("*.json"), key=natural_sort_key):
        if path.name.endswith("_metadata.json"):
            continue
        records.append(load_clause_record(path))
    return records


def clause_header_lines(record: ClauseRecord) -> list[str]:
    return [
        f"clause_title: {record.clause_title}",
        f"standard_id: {record.standard_id}",
        f"standard_title: {record.standard_title}",
        f"clause_id: {record.clause_id}",
    ]


def full_clause_text(record: ClauseRecord) -> str:
    return "\n".join(clause_header_lines(record) + [f"clause_content: {record.clause_content}"]).strip()


def split_content_ids(content_ids: list[int], *, max_content_tokens: int, overlap_tokens: int) -> list[list[int]]:
    if max_content_tokens <= 0:
        raise ValueError("max_content_tokens must be positive")
    if overlap_tokens >= max_content_tokens:
        overlap_tokens = max(0, max_content_tokens // 4)

    parts: list[list[int]] = []
    start = 0
    while start < len(content_ids):
        end = min(start + max_content_tokens, len(content_ids))
        parts.append(content_ids[start:end])
        if end >= len(content_ids):
            break
        next_start = end - overlap_tokens
        if next_start <= start:
            next_start = end
        start = next_start
    return parts


def singularize_word(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses") or word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def unique_preserving(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value.strip())
    return result


def build_metadata_keywords(record: ClauseRecord, *, max_keywords: int) -> list[str]:
    title_lower = record.clause_title.lower()
    title_words = _extract_keywords_from_text(title_lower, max_keywords=max_keywords)
    singular_title = " ".join(singularize_word(word) for word in title_words)

    candidates: list[str] = [
        record.clause_id,
        record.clause_title,
        title_lower,
    ]
    if singular_title and singular_title != title_lower:
        candidates.append(singular_title)

    for word in title_words:
        candidates.append(word)
        singular_word = singularize_word(word)
        if singular_word != word:
            candidates.append(singular_word)

    content_keywords = _extract_keywords_from_text(record.clause_content, max_keywords=max_keywords * 2)
    candidates.extend(content_keywords)
    return unique_preserving(candidates)[:max_keywords]


def build_metadata_aliases(record: ClauseRecord) -> list[str]:
    title_lower = record.clause_title.lower()
    title_words = _extract_keywords_from_text(title_lower, max_keywords=20)
    singular_words = [singularize_word(word) for word in title_words]
    singular_title = " ".join(singular_words)

    aliases = [
        record.clause_id,
        f" {record.clause_id}",
        f"clause {record.clause_id}",
        f"clause_id: {record.clause_id}",
        record.clause_title,
        f" {record.clause_title}",
        title_lower,
        f" {title_lower}",
        f"clause_title: {record.clause_title}",
        f"clause_title: {title_lower}",
    ]

    if singular_title and singular_title != title_lower:
        aliases.extend([singular_title, f" {singular_title}"])

    for word in title_words:
        aliases.extend([word, f" {word}"])
    for word in singular_words:
        aliases.extend([word, f" {word}"])

    return unique_preserving(aliases)


def build_clause_windows(
    record: ClauseRecord,
    tokenizer: Any,
    *,
    window_size: int,
    overlap_tokens: int,
    max_keywords: int,
) -> list[ClauseWindow]:
    header_text = "\n".join(clause_header_lines(record)) + "\nclause_content: "
    header_ids = _encode_token_ids(tokenizer, header_text, add_special_tokens=False)
    content_ids = _encode_token_ids(tokenizer, record.clause_content, add_special_tokens=False)

    available = window_size - len(header_ids)
    if available <= 0:
        raise ValueError(
            f"Window size {window_size} is too small for AUS3000 metadata header "
            f"for clause {record.clause_id}"
        )

    content_parts = split_content_ids(
        content_ids,
        max_content_tokens=available,
        overlap_tokens=overlap_tokens,
    )
    keywords = build_metadata_keywords(record, max_keywords=max_keywords)

    windows: list[ClauseWindow] = []
    part_count = len(content_parts)
    for idx, part_ids in enumerate(content_parts, start=1):
        token_ids = header_ids + list(part_ids)
        windows.append(
            ClauseWindow(
                clause_id=record.clause_id,
                clause_title=record.clause_title,
                source_file=record.source_file,
                part_index=idx,
                part_count=part_count,
                token_ids=token_ids,
                token_count=len(token_ids),
                keywords=keywords,
            )
        )
    return windows


def maybe_add_topic_expansion(
    *,
    model: Any,
    tokenizer: Any,
    window_tokens: set[int],
    keywords: list[str],
    token_ids: list[int],
    expansion_token_budget: int,
) -> None:
    if expansion_token_budget <= 0:
        return

    from chuk_lazarus.inference.context.knowledge.torch_build import (
        _apply_topic_expansion,
        _generate_topic_expansion_ids,
    )

    expansion_ids = _generate_topic_expansion_ids(
        model,
        tokenizer,
        token_ids,
        n_tokens=expansion_token_budget,
    )
    _apply_topic_expansion(
        tokenizer=tokenizer,
        window_tokens=window_tokens,
        keywords=keywords,
        expansion_ids=expansion_ids,
    )


def add_metadata_alias_tokens(
    *,
    tokenizer: Any,
    record: ClauseRecord,
    window_tokens: set[int],
    keywords: list[str],
) -> None:
    aliases = build_metadata_aliases(record)
    for alias in aliases:
        for token_id in _encode_token_ids(tokenizer, alias, add_special_tokens=False):
            window_tokens.add(int(token_id))

    for keyword in build_metadata_keywords(record, max_keywords=max(12, len(keywords))):
        lowered = keyword.lower()
        if lowered not in {value.lower() for value in keywords}:
            keywords.append(lowered)
        for variant in (keyword, keyword.lower(), f" {keyword}", f" {keyword.lower()}"):
            for token_id in _encode_token_ids(tokenizer, variant, add_special_tokens=False):
                window_tokens.add(int(token_id))


def build_store(
    *,
    model: Any,
    tokenizer: Any,
    records: list[ClauseRecord],
    output_path: Path,
    arch_config: Any,
    overlap_tokens: int,
    max_keywords: int,
    topic_expansion_tokens: int,
) -> tuple[TorchKnowledgeStore, list[ClauseWindow], int]:
    windows: list[ClauseWindow] = []
    for record in records:
        windows.extend(
            build_clause_windows(
                record,
                tokenizer,
                window_size=arch_config.window_size,
                overlap_tokens=overlap_tokens,
                max_keywords=max_keywords,
            )
        )

    token_windows = [window.token_ids for window in windows]
    num_windows = len(token_windows)
    num_tokens = sum(len(window.token_ids) for window in windows)

    window_tokens: dict[int, set[int]] = {}
    window_token_lists: dict[int, list[int]] = {}
    keywords: dict[int, list[str]] = {}

    for wid, window in enumerate(windows):
        window_tokens[wid] = set(int(token_id) for token_id in window.token_ids)
        window_token_lists[wid] = list(window.token_ids)
        keywords[wid] = list(window.keywords)

    idf = TFIDFRouter.compute_idf(window_tokens)

    entries: list[_EntryRecord] = []
    fact_id = 0
    for wid, window in enumerate(windows):
        targets = _select_targets(
            window.token_ids,
            idf,
            min_k=arch_config.entries_per_window,
        )
        for token_id, pos_in_window in targets:
            entries.append(
                _EntryRecord(
                    token_id=token_id,
                    coefficient=_entry_coefficient(model, token_id, arch_config.inject_coefficient),
                    window_id=wid,
                    position_in_window=pos_in_window,
                    fact_id=fact_id,
                )
            )
            fact_id += 1

    boundaries, final_boundary, residual_streams = capture_window_boundaries(
        model,
        token_windows,
        crystal_layer=arch_config.crystal_layer,
        return_streams=True,
    )

    record_lookup = {record.clause_id: record for record in records}
    for wid, window in enumerate(windows):
        record = record_lookup[window.clause_id]
        add_metadata_alias_tokens(
            tokenizer=tokenizer,
            record=record,
            window_tokens=window_tokens[wid],
            keywords=keywords[wid],
        )
        maybe_add_topic_expansion(
            model=model,
            tokenizer=tokenizer,
            window_tokens=window_tokens[wid],
            keywords=keywords[wid],
            token_ids=window.token_ids,
            expansion_token_budget=topic_expansion_tokens,
        )

    idf = TFIDFRouter.compute_idf(window_tokens)

    if entries:
        np_save_entries = _entries_to_numpy(entries)
    else:
        np_save_entries = _entries_to_numpy([])
    import numpy as np

    np.savez(str(output_path / ENTRIES_FILE), entries=np_save_entries)
    _save_npz_mapping(
        output_path / WINDOW_TOKENS_FILE,
        window_tokens,
        dtype=np.uint32,
        sort_values=True,
    )
    _save_npz_mapping(
        output_path / WINDOW_TOKEN_LISTS_FILE,
        window_token_lists,
        dtype=np.uint32,
        sort_values=False,
    )
    (output_path / IDF_FILE).write_text(json.dumps({str(k): v for k, v in idf.items()}, indent=1) + "\n")
    (output_path / KEYWORDS_FILE).write_text(
        json.dumps({str(k): v for k, v in keywords.items()}, indent=1) + "\n"
    )

    if boundaries:
        boundary_dir = output_path / BOUNDARIES_DIR
        boundary_dir.mkdir(exist_ok=True)
        for wid, boundary in boundaries.items():
            boundary_np = np.asarray(boundary, dtype=np.float32).reshape(-1)
            np.save(str(boundary_dir / f"window_{wid:03d}.npy"), boundary_np)

    if residual_streams:
        stream_dir = output_path / RESIDUAL_STREAMS_DIR
        stream_dir.mkdir(exist_ok=True)
        for wid, stream in residual_streams.items():
            stream_np = np.asarray(stream, dtype=np.float32)
            np.save(str(stream_dir / f"window_{wid:03d}.npy"), stream_np)

    activation_routes_meta = _write_activation_routes(
        output_path,
        residual_streams,
        num_windows=num_windows,
        layer=arch_config.crystal_layer,
        model_id=_model_id_from_model(model),
    )

    if final_boundary is not None:
        final_np = np.asarray(final_boundary, dtype=np.float32).reshape(1, 1, -1)
        np.save(str(output_path / BOUNDARY_RESIDUAL_FILE), final_np)

    window_metadata = {}
    for wid, window in enumerate(windows):
        window_metadata[str(wid)] = {
            "clause_id": window.clause_id,
            "clause_title": window.clause_title,
            "source_file": window.source_file,
            "part_index": window.part_index,
            "part_count": window.part_count,
            "token_count": window.token_count,
            "content_was_empty": record_lookup[window.clause_id].content_was_empty,
        }
    (output_path / WINDOW_METADATA_FILE).write_text(json.dumps(window_metadata, indent=2) + "\n")

    manifest = {
        "version": STORE_VERSION,
        "num_entries": len(entries),
        "num_windows": num_windows,
        "num_tokens": num_tokens,
        "entries_per_window": arch_config.entries_per_window,
        "crystal_layer": arch_config.crystal_layer,
        "window_size": arch_config.window_size,
        "arch_config": arch_config.to_dict(),
        "has_residuals": False,
        "has_residual_streams": bool(residual_streams),
        "residual_stream_count": len(residual_streams),
        "window_metadata": WINDOW_METADATA_FILE,
        "clause_aligned": True,
    }
    if activation_routes_meta is not None:
        manifest["activation_routes"] = activation_routes_meta
    (output_path / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")

    return TorchKnowledgeStore.load(output_path), windows, len(entries)


def build_prefill_metadata(
    *,
    model_id: str,
    actual_device: str,
    input_dir: Path,
    checkpoint: Path,
    store: TorchKnowledgeStore,
    record_count: int,
    split_clause_count: int,
    overlap_tokens: int,
    max_keywords: int,
    topic_expansion_tokens: int,
    standard_id: str,
    standard_title: str,
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "version": TORCH_PREFILL_VERSION,
        "kind": "torch-prefill",
        "status": "complete",
        "backend": "torch",
        "created_at": created_at,
        "model": {
            "id": model_id,
            "device": actual_device,
        },
        "source": {
            "name": f"{standard_id} Clause Aligned",
            "standard_id": standard_id,
            "standard_title": standard_title,
            "input_file": str(input_dir),
            "max_tokens": None,
        },
        "windowing": {
            "window_size": int(store.config.window_size),
            "num_tokens": int(store.num_tokens),
            "num_windows": int(store.num_windows),
        },
        "arch_config": store.config.to_dict(),
        "requested": {
            "phases": ["windows"],
            "residual_mode": "interval",
            "store_pages": False,
            "store_kv_full": False,
            "mode": "clause-aligned",
            "device": actual_device,
        },
        "written": {
            "phases": ["windows"],
            "store_pages": False,
            "store_kv_full": False,
            "mlx_checkpoint_artifacts": False,
            "torch_store_only": True,
        },
        "artifacts": {
            "sidecar": TORCH_PREFILL_FILE,
            "store_dir": TORCH_STORE_DIR,
            "store_layout": "apollo-v12",
            "store_version": STORE_VERSION,
            "manifest": f"{TORCH_STORE_DIR}/{MANIFEST_FILE}",
            "entries": f"{TORCH_STORE_DIR}/{ENTRIES_FILE}",
            "window_tokens": f"{TORCH_STORE_DIR}/{WINDOW_TOKENS_FILE}",
            "window_token_lists": f"{TORCH_STORE_DIR}/{WINDOW_TOKEN_LISTS_FILE}",
            "idf": f"{TORCH_STORE_DIR}/{IDF_FILE}",
            "keywords": f"{TORCH_STORE_DIR}/{KEYWORDS_FILE}",
            "boundaries_dir": f"{TORCH_STORE_DIR}/{BOUNDARIES_DIR}",
            "residual_streams_dir": f"{TORCH_STORE_DIR}/{RESIDUAL_STREAMS_DIR}",
            "activation_routes": f"{TORCH_STORE_DIR}/{ACTIVATION_ROUTES_FILE}",
            "activation_routes_dir": f"{TORCH_STORE_DIR}/{ACTIVATION_ROUTES_DIR}",
            "boundary_residual": f"{TORCH_STORE_DIR}/{BOUNDARY_RESIDUAL_FILE}",
            "window_metadata": f"{TORCH_STORE_DIR}/{WINDOW_METADATA_FILE}",
        },
        "generate_batch": {
            "backend": "torch",
            "checkpoint_layout": "checkpoint-relative",
            "sidecar": TORCH_PREFILL_FILE,
            "store_dir": TORCH_STORE_DIR,
            "ready": True,
        },
        "clause_aligned": {
            "enabled": True,
            "record_count": record_count,
            "split_clause_count": split_clause_count,
            "overlap_tokens": overlap_tokens,
            "max_keywords": max_keywords,
            "topic_expansion_tokens": topic_expansion_tokens,
        },
    }
    (checkpoint / TORCH_PREFILL_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    input_dir = args.input_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    store_path = checkpoint / TORCH_STORE_DIR

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    if checkpoint.exists():
        if not args.force:
            print(
                f"Checkpoint already exists: {checkpoint}\n"
                "Pass --force to rebuild it.",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(checkpoint)

    checkpoint.mkdir(parents=True, exist_ok=True)
    store_path.mkdir(parents=True, exist_ok=True)

    records = load_clause_records(input_dir)
    if not records:
        print(f"No clause JSON files found in {input_dir}", file=sys.stderr)
        return 1

    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer, actual_device = _load_model_and_tokenizer(str(args.model), args.device)
    arch_config = _resolve_architecture_config(model.config, window_size=args.window_size)
    object.__setattr__(arch_config, "entries_per_window", int(args.entries_per_window))

    print(f"Loaded {len(records)} clause records", file=sys.stderr)
    store, windows, entry_count = build_store(
        model=model,
        tokenizer=tokenizer,
        records=records,
        output_path=store_path,
        arch_config=arch_config,
        overlap_tokens=args.overlap_tokens,
        max_keywords=args.max_keywords,
        topic_expansion_tokens=args.topic_expansion_tokens,
    )

    stub_clause_count = sum(1 for record in records if record.content_was_empty)
    split_clause_count = sum(1 for window in windows if window.part_count > 1)
    split_clause_ids = sorted({window.clause_id for window in windows if window.part_count > 1})
    build_manifest = {
        "input_dir": str(input_dir),
        "checkpoint": str(checkpoint),
        "store_path": str(store_path),
        "model": str(args.model),
        "device": actual_device,
        "window_size": arch_config.window_size,
        "overlap_tokens": args.overlap_tokens,
        "entries_per_window": arch_config.entries_per_window,
        "max_keywords": args.max_keywords,
        "topic_expansion_tokens": args.topic_expansion_tokens,
        "record_count": len(records),
        "stub_clause_count": stub_clause_count,
        "window_count": len(windows),
        "split_clause_count": len(split_clause_ids),
        "split_clause_ids": split_clause_ids,
        "num_tokens": store.num_tokens,
        "num_windows": store.num_windows,
        "num_entries": entry_count,
    }
    (checkpoint / BUILD_MANIFEST_FILE).write_text(json.dumps(build_manifest, indent=2) + "\n")
    build_prefill_metadata(
        model_id=str(args.model),
        actual_device=actual_device,
        input_dir=input_dir,
        checkpoint=checkpoint,
        store=store,
        record_count=len(records),
        split_clause_count=len(split_clause_ids),
        overlap_tokens=args.overlap_tokens,
        max_keywords=args.max_keywords,
        topic_expansion_tokens=args.topic_expansion_tokens,
        standard_id=records[0].standard_id,
        standard_title=records[0].standard_title,
    )

    print(
        f"Clause-aligned checkpoint built at {checkpoint}\n"
        f"  records={len(records)}  stubs={stub_clause_count}  windows={store.num_windows}  tokens={store.num_tokens}\n"
        f"  split_clauses={len(split_clause_ids)}  device={actual_device}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
