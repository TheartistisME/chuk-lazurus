#!/usr/bin/env python3
"""Build a Lazarus-ready plain-text corpus from TradeGuru-style JSON pages.

The Lazarus knowledge builders consume a single UTF-8 text file, not a JSON
directory. This tool extracts only the fields that matter for the corpus
(`Title` and `Info`), normalizes noisy formatting, and emits a deterministic
plain-text file that is ready for `knowledge build`.

Default output is intentionally compact to match the low-overhead style used by
the Apollo examples: title line, normalized body, blank line separator.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("/mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/json")
DEFAULT_OUTPUT_PATH = Path(
    "/mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_lazarus_corpus.txt"
)
DEFAULT_MANIFEST_PATH = Path(
    "/mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_lazarus_manifest.json"
)

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

CHECKBOX_REPLACEMENTS = {
    "☐": "[ ]",
    "☑": "[x]",
    "☒": "[x]",
}

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
BULLET_RE = re.compile(r"^(\s*)[\-*•●◦○▪■]\s+")
MULTISPACE_RE = re.compile(r" {2,}")
TABLE_SEPARATOR_CELL_RE = re.compile(r"^[-:]+$")


@dataclass(frozen=True)
class Record:
    source_file: str
    title: str
    info: str


@dataclass(frozen=True)
class CorpusModeConfig:
    name: str
    required_keys: tuple[str, ...]
    default_record_format: str
    skip_metadata_files: bool = False


MODE_CONFIGS: dict[str, CorpusModeConfig] = {
    "tradeguru": CorpusModeConfig(
        name="tradeguru",
        required_keys=("Title", "Info"),
        default_record_format="compact",
    ),
    "aus3000": CorpusModeConfig(
        name="aus3000",
        required_keys=(
            "standard_id",
            "standard_title",
            "clause_id",
            "clause_title",
        ),
        default_record_format="compact",
        skip_metadata_files=True,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Title/Info fields from JSON pages into a Lazarus-ready corpus text file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing source JSON files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Corpus output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Manifest output path (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--record-format",
        choices=("compact", "labeled", "tagged"),
        default=None,
        help="How to render each record in the text corpus",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_CONFIGS),
        default="tradeguru",
        help="Extraction mode / source schema profile",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional local model path or HF model ID for tokenizer-based token counting",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional token budget. Requires --model and stops before the budget is exceeded.",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        help="Optional directory to copy unreadable or invalid JSON files into",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit successfully even if some files fail to parse",
    )
    return parser


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name.lower())
    key: list[Any] = []
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

    if line.count("|") >= 2:
        parts = [part.strip() for part in line.split("|")]
        parts = [part for part in parts if part]
        if parts and all(TABLE_SEPARATOR_CELL_RE.fullmatch(part) for part in parts):
            return ""
        if len(parts) >= 2:
            line = " - ".join(parts)

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
    for src, dst in CHECKBOX_REPLACEMENTS.items():
        text = text.replace(src, dst)

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


def render_record(record: Record, record_format: str) -> str:
    title = record.title.strip()
    info = record.info.strip()

    if record_format == "compact":
        parts = [title]
        if info:
            parts.append(info)
        return "\n".join(parts).strip()

    if record_format == "labeled":
        parts = [f"Title: {title}"]
        if info:
            parts.extend(["", info])
        return "\n".join(parts).strip()

    parts = ["[RECORD]", f"Title: {title}"]
    if info:
        parts.extend(["", info])
    parts.append("[/RECORD]")
    return "\n".join(parts).strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_tradeguru_record(path: Path, data: dict[str, Any]) -> Record:
    title = normalize_text(str(data.get("Title", "")).strip())
    info = normalize_text(str(data.get("Info", "")).strip())
    if not title:
        raise ValueError("Missing or empty Title")
    if not info:
        raise ValueError("Missing or empty Info")
    return Record(source_file=path.name, title=title, info=info)


def _load_aus3000_record(path: Path, data: dict[str, Any]) -> Record:
    standard_id = normalize_text(str(data.get("standard_id", "")).strip())
    standard_title = normalize_text(str(data.get("standard_title", "")).strip())
    clause_id = normalize_text(str(data.get("clause_id", "")).strip())
    clause_title = normalize_text(str(data.get("clause_title", "")).strip())
    clause_content = normalize_text(str(data.get("clause_content", "")).strip())

    if not all((standard_id, standard_title, clause_id, clause_title)):
        raise ValueError(
            "Missing one or more required aus3000 fields: "
            "standard_id, standard_title, clause_id, clause_title"
        )

    info = "\n".join(
        [
            f"standard_id: {standard_id}",
            f"standard_title: {standard_title}",
            f"clause_id: {clause_id}",
            f"clause_content: {clause_content}",
        ]
    ).strip()
    return Record(source_file=path.name, title=f"clause_title: {clause_title}", info=info)


def load_record(path: Path, mode: str) -> Record:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, found {type(data).__name__}")

    if mode == "tradeguru":
        return _load_tradeguru_record(path, data)
    if mode == "aus3000":
        return _load_aus3000_record(path, data)
    raise ValueError(f"Unsupported mode: {mode}")


def copy_to_quarantine(path: Path, quarantine_dir: Path) -> None:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    (quarantine_dir / path.name).write_bytes(path.read_bytes())


def build_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def count_tokens(tokenizer: Any, text: str) -> int:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return len(token_ids)


def join_records(blocks: list[str]) -> str:
    if not blocks:
        return ""
    return "\n\n".join(blocks).strip() + "\n"


def build_corpus_text(
    records: list[Record],
    *,
    record_format: str,
    tokenizer: Any = None,
    max_tokens: int | None = None,
) -> tuple[str, list[Record], int | None]:
    rendered_blocks: list[str] = []
    included_records: list[Record] = []
    token_count: int | None = None

    for record in records:
        rendered = render_record(record, record_format)
        candidate_text = join_records(rendered_blocks + [rendered])

        if tokenizer is not None and max_tokens is not None:
            candidate_tokens = count_tokens(tokenizer, candidate_text)
            if candidate_tokens > max_tokens:
                break
            token_count = candidate_tokens

        rendered_blocks.append(rendered)
        included_records.append(record)

    corpus_text = join_records(rendered_blocks)
    if tokenizer is not None and token_count is None:
        token_count = count_tokens(tokenizer, corpus_text)
    return corpus_text, included_records, token_count


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_tokens is not None and not args.model:
        parser.error("--max-tokens requires --model so the tokenizer can enforce the budget")

    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    mode_config = MODE_CONFIGS[args.mode]
    record_format = args.record_format or mode_config.default_record_format

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    json_files = sorted(input_dir.glob("*.json"), key=natural_sort_key)
    if not json_files:
        print(f"No JSON files found in: {input_dir}", file=sys.stderr)
        return 1

    tokenizer = build_tokenizer(args.model) if args.model else None

    records: list[Record] = []
    failures: list[dict[str, str]] = []
    skipped_files: list[dict[str, str]] = []

    for path in json_files:
        if mode_config.skip_metadata_files and path.name.endswith("_metadata.json"):
            skipped_files.append({"file": path.name, "reason": "metadata sidecar skipped by mode"})
            continue
        try:
            records.append(load_record(path, args.mode))
        except Exception as exc:  # pragma: no cover - exercised via real files
            failures.append({"file": path.name, "error": str(exc)})
            if args.quarantine_dir is not None:
                copy_to_quarantine(path, args.quarantine_dir.resolve())

    corpus_text, included_records, token_count = build_corpus_text(
        records,
        record_format=record_format,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(corpus_text, encoding="utf-8")

    manifest = {
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "mode": args.mode,
        "record_format": record_format,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "source_file_count": len(json_files),
        "skipped_file_count": len(skipped_files),
        "skipped_files": skipped_files,
        "parsed_record_count": len(records),
        "included_record_count": len(included_records),
        "excluded_due_to_budget_count": len(records) - len(included_records),
        "failure_count": len(failures),
        "failures": failures,
        "included_files": [record.source_file for record in included_records],
        "output_chars": len(corpus_text),
        "output_bytes_utf8": len(corpus_text.encode("utf-8")),
        "token_count": token_count,
        "estimated_windows_512": math.ceil(token_count / 512) if token_count else None,
        "normalization_rules": [
            "read UTF-8 with BOM tolerance",
            "remove zero-width and invalid control characters",
            "normalize line endings to LF",
            "replace NBSP with space",
            "normalize smart quotes and punctuation to plain text where reasonable",
            "normalize checkbox glyphs to [ ] / [x]",
            "normalize markdown headings by stripping leading # markers",
            "normalize bullet glyphs to hyphen bullets",
            "convert simple pipe tables to plain text rows",
            "collapse repeated blank lines",
            "emit compact title+info records in natural filename order",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("TradeGuru corpus build")
    print(f"Input dir: {input_dir}")
    print(f"Output: {output_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Source JSON files: {len(json_files)}")
    print(f"Skipped files: {len(skipped_files)}")
    print(f"Included records: {len(included_records)}")
    print(f"Failures: {len(failures)}")
    print(f"Output chars: {len(corpus_text)}")
    if token_count is not None:
        print(f"Tokens: {token_count}")
        print(f"Estimated 512-token windows: {math.ceil(token_count / 512) if token_count else 0}")
    if args.max_tokens is not None:
        print(f"Token budget: {args.max_tokens}")
        print(f"Excluded due to budget: {len(records) - len(included_records)}")

    if failures and not args.allow_failures:
        print("One or more JSON files failed to parse. See manifest for details.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
