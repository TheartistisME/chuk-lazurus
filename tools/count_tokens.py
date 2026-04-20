#!/usr/bin/env python3
"""Count tokens for raw text, stdin, or one or more files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib import error, request

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ANTHROPIC_API = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_CODEX_MODEL = "gpt-5-codex"


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


@dataclass(frozen=True)
class CounterSpec:
    target: str
    label: str
    count: Callable[[str], int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count tokens for raw text, stdin, or one or more files.",
        epilog="Use --to codex or --to anthropic for model-aware counting.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-t", "--tokenizer", help="Direct tokenizer/model/encoding/path.")
    source.add_argument("--to", choices=["codex", "anthropic"], help="Target AI family.")
    parser.add_argument("--model", help="Model to count against. Optional with --to.")
    parser.add_argument(
        "--role",
        choices=["user", "system"],
        default="user",
        help="Prompt role for API-style counting. Default: user.",
    )
    parser.add_argument("--text", action="append", default=[], help="Literal text. Repeatable.")
    parser.add_argument("--stdin", action="store_true", help="Read one extra input from stdin.")
    parser.add_argument("--encoding", default="utf-8", help="File encoding. Default: utf-8.")
    parser.add_argument(
        "--special-tokens",
        action="store_true",
        help="Only applies with --tokenizer.",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON.")
    output.add_argument("--total-only", action="store_true", help="Print only total tokens.")
    parser.add_argument("files", nargs="*", help="File paths to read. Use '-' for stdin.")
    return parser


def should_read_stdin(args: argparse.Namespace) -> bool:
    if args.stdin or "-" in args.files:
        return True
    return not sys.stdin.isatty() and not args.text and not args.files


def collect_inputs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[InputRecord]:
    if args.files.count("-") > 1:
        parser.error("stdin may only be provided once")
    records = [InputRecord(source=f"text[{i}]", text=text) for i, text in enumerate(args.text, 1)]
    if should_read_stdin(args):
        records.append(InputRecord(source="<stdin>", text=sys.stdin.read()))
    for raw_path in args.files:
        if raw_path == "-":
            continue
        path = Path(raw_path)
        try:
            text = path.read_text(encoding=args.encoding)
        except OSError as exc:
            parser.error(f"could not read {path}: {exc}")
        records.append(InputRecord(source=str(path), text=text))
    if not records:
        parser.error("provide --text, one or more files, or pipe data on stdin")
    return records


def encode_with_tokenizer(tokenizer, text: str, add_special_tokens: bool) -> int:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return len(token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids)


def build_direct_tokenizer_counter(name: str, add_special_tokens: bool) -> CounterSpec:
    from chuk_lazarus.utils.tokenizer_loader import load_tokenizer

    tokenizer = load_tokenizer(name)
    return CounterSpec(
        target="tokenizer",
        label=name,
        count=lambda text: encode_with_tokenizer(tokenizer, text, add_special_tokens),
    )


def build_codex_counter(model: str | None) -> CounterSpec:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is required for --to codex. Run via `uv run --extra openai ...`."
        ) from exc
    model_name = model or DEFAULT_CODEX_MODEL
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    return CounterSpec(
        target="codex",
        label=f"{model_name} ({encoding.name})",
        count=lambda text: len(encoding.encode(text)),
    )


def anthropic_request(path: str, api_key: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{ANTHROPIC_API}{path}",
        data=data,
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API error {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Anthropic API request failed: {exc.reason}") from exc


def resolve_latest_anthropic_model(api_key: str) -> str:
    payload = anthropic_request("/models", api_key)
    models = payload.get("data") or []
    if not models:
        raise RuntimeError("Anthropic API returned no models")
    return models[0]["id"]


def build_anthropic_counter(model: str | None, role: str) -> CounterSpec:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for --to anthropic")
    resolved_model = model or resolve_latest_anthropic_model(api_key)

    def count(text: str) -> int:
        payload = {"model": resolved_model, "messages": [] if role == "system" else [{"role": "user", "content": text}]}
        if role == "system":
            payload["system"] = text
        response = anthropic_request("/messages/count_tokens", api_key, payload)
        return int(response["input_tokens"])

    return CounterSpec(target="anthropic", label=resolved_model, count=count)


def resolve_counter(args: argparse.Namespace, parser: argparse.ArgumentParser) -> CounterSpec:
    if args.to and args.special_tokens:
        parser.error("--special-tokens only works with --tokenizer")
    if args.to == "codex":
        return build_codex_counter(args.model)
    if args.to == "anthropic":
        return build_anthropic_counter(args.model, args.role)
    return build_direct_tokenizer_counter(args.tokenizer, args.special_tokens)


def count_inputs(counter: CounterSpec, inputs: Sequence[InputRecord]) -> list[CountRecord]:
    return [
        CountRecord(
            source=item.source,
            tokens=counter.count(item.text),
            chars=len(item.text),
            bytes_utf8=len(item.text.encode("utf-8")),
        )
        for item in inputs
    ]


def summarize(results: Sequence[CountRecord]) -> CountRecord:
    return CountRecord(
        source="TOTAL",
        tokens=sum(item.tokens for item in results),
        chars=sum(item.chars for item in results),
        bytes_utf8=sum(item.bytes_utf8 for item in results),
    )


def emit_json(counter: CounterSpec, role: str, results: Sequence[CountRecord]) -> None:
    print(
        json.dumps(
            {
                "target": counter.target,
                "counter": counter.label,
                "role": role,
                "inputs": [asdict(item) for item in results],
                "total": asdict(summarize(results)),
            },
            indent=2,
        )
    )


def emit_table(counter: CounterSpec, results: Sequence[CountRecord]) -> None:
    rows = [*results, summarize(results)]
    widths = {
        "source": max(len("SOURCE"), *(len(row.source) for row in rows)),
        "tokens": max(len("TOKENS"), *(len(str(row.tokens)) for row in rows)),
        "chars": max(len("CHARS"), *(len(str(row.chars)) for row in rows)),
        "bytes": max(len("BYTES"), *(len(str(row.bytes_utf8)) for row in rows)),
    }
    header = (
        f"{'TOKENS':>{widths['tokens']}}  "
        f"{'CHARS':>{widths['chars']}}  "
        f"{'BYTES':>{widths['bytes']}}  "
        f"{'SOURCE':<{widths['source']}}"
    )
    print(f"Target: {counter.target}")
    print(f"Counter: {counter.label}")
    print(header)
    print("-" * len(header))
    for row in rows[:-1]:
        print(
            f"{row.tokens:>{widths['tokens']}}  {row.chars:>{widths['chars']}}  "
            f"{row.bytes_utf8:>{widths['bytes']}}  {row.source:<{widths['source']}}"
        )
    print("-" * len(header))
    total = rows[-1]
    print(
        f"{total.tokens:>{widths['tokens']}}  {total.chars:>{widths['chars']}}  "
        f"{total.bytes_utf8:>{widths['bytes']}}  {total.source:<{widths['source']}}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = collect_inputs(args, parser)
    try:
        counter = resolve_counter(args, parser)
        results = count_inputs(counter, inputs)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.total_only:
        print(summarize(results).tokens)
        return 0
    if args.as_json:
        emit_json(counter, args.role, results)
        return 0
    emit_table(counter, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
