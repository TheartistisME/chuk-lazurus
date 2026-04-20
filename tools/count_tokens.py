#!/usr/bin/env python3
"""Count tokens for raw text, stdin, or one or more files."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass(frozen=True)
class InputRecord:
    source: str
    text: str


@dataclass(frozen=True)
class CountRecord:
    source: str
    tokens: int
    chars: int
    bytes_utf8: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count tokens for raw text, stdin, or one or more files.",
        epilog=(
            "Token counts are exact for the tokenizer you choose. "
            "There is no universal token count across models."
        ),
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        required=True,
        help="Tokenizer name, model ID, local path, or tiktoken encoding.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Literal text to count. Repeat to add more text inputs.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read one extra input from standard input.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input files. Default: utf-8.",
    )
    parser.add_argument(
        "--special-tokens",
        action="store_true",
        help="Include special tokens when the tokenizer supports it.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON.")
    output_group.add_argument(
        "--total-only",
        action="store_true",
        help="Print only the combined token total.",
    )
    parser.add_argument("files", nargs="*", help="File paths to read. Use '-' for stdin.")
    return parser


def load_counter(tokenizer_name: str):
    from chuk_lazarus.utils.tokenizer_loader import load_tokenizer

    return load_tokenizer(tokenizer_name)


def should_read_stdin(args: argparse.Namespace) -> bool:
    if args.stdin or "-" in args.files:
        return True
    return not sys.stdin.isatty() and not args.text and not args.files


def collect_inputs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[InputRecord]:
    if args.files.count("-") > 1:
        parser.error("stdin may only be provided once")

    records: list[InputRecord] = []
    for index, text in enumerate(args.text, start=1):
        records.append(InputRecord(source=f"text[{index}]", text=text))

    if should_read_stdin(args):
        records.append(InputRecord(source="<stdin>", text=sys.stdin.read()))

    for file_arg in args.files:
        if file_arg == "-":
            continue
        path = Path(file_arg)
        try:
            text = path.read_text(encoding=args.encoding)
        except OSError as exc:
            parser.error(f"could not read {path}: {exc}")
        records.append(InputRecord(source=str(path), text=text))

    if not records:
        parser.error("provide --text, one or more files, or pipe data on stdin")
    return records


def encode_text(tokenizer, text: str, add_special_tokens: bool) -> int:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        token_ids = tokenizer.encode(text)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return len(token_ids)


def count_inputs(tokenizer, inputs: Sequence[InputRecord], add_special_tokens: bool) -> list[CountRecord]:
    results: list[CountRecord] = []
    for item in inputs:
        results.append(
            CountRecord(
                source=item.source,
                tokens=encode_text(tokenizer, item.text, add_special_tokens),
                chars=len(item.text),
                bytes_utf8=len(item.text.encode("utf-8")),
            )
        )
    return results


def summarize(results: Sequence[CountRecord]) -> CountRecord:
    return CountRecord(
        source="TOTAL",
        tokens=sum(item.tokens for item in results),
        chars=sum(item.chars for item in results),
        bytes_utf8=sum(item.bytes_utf8 for item in results),
    )


def emit_json(tokenizer_name: str, results: Sequence[CountRecord]) -> None:
    total = summarize(results)
    payload = {
        "tokenizer": tokenizer_name,
        "inputs": [asdict(item) for item in results],
        "total": asdict(total),
    }
    print(json.dumps(payload, indent=2))


def emit_table(tokenizer_name: str, results: Sequence[CountRecord]) -> None:
    total = summarize(results)
    rows = [*results, total]
    source_width = max(len("SOURCE"), *(len(row.source) for row in rows))
    token_width = max(len("TOKENS"), *(len(str(row.tokens)) for row in rows))
    char_width = max(len("CHARS"), *(len(str(row.chars)) for row in rows))
    byte_width = max(len("BYTES"), *(len(str(row.bytes_utf8)) for row in rows))

    header = (
        f"{'TOKENS':>{token_width}}  "
        f"{'CHARS':>{char_width}}  "
        f"{'BYTES':>{byte_width}}  "
        f"{'SOURCE':<{source_width}}"
    )
    print(f"Tokenizer: {tokenizer_name}")
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row.tokens:>{token_width}}  "
            f"{row.chars:>{char_width}}  "
            f"{row.bytes_utf8:>{byte_width}}  "
            f"{row.source:<{source_width}}"
        )
    print("-" * len(header))
    print(
        f"{total.tokens:>{token_width}}  "
        f"{total.chars:>{char_width}}  "
        f"{total.bytes_utf8:>{byte_width}}  "
        f"{total.source:<{source_width}}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = collect_inputs(args, parser)

    try:
        tokenizer = load_counter(args.tokenizer)
    except Exception as exc:
        print(f"error: failed to load tokenizer '{args.tokenizer}': {exc}", file=sys.stderr)
        return 2

    results = count_inputs(tokenizer, inputs, add_special_tokens=args.special_tokens)
    if args.total_only:
        print(summarize(results).tokens)
        return 0
    if args.as_json:
        emit_json(args.tokenizer, results)
        return 0

    emit_table(args.tokenizer, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
