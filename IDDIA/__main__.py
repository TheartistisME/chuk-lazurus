"""Command line interface for IDDIA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_DDIA_URL,
    STAGE_LENSES,
    STAGES,
    build_context_package,
    ingest_ddia,
    next_stage,
    print_ingest_result,
)


def _write_stdout_utf8(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m IDDIA",
        description="IDDIA: build DDIA-backed context packages for agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    ingest = subparsers.add_parser(
        "ingest-ddia",
        help="Download DDIA, convert each page with MarkItDown, chunk, and vectorize with zvec",
    )
    ingest.add_argument("--url", default=DEFAULT_DDIA_URL, help="PDF URL")
    ingest.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="Root for source, markdown, chunks, vectors, and packages",
    )
    ingest.add_argument("--force-download", action="store_true", help="Re-download the PDF")
    ingest.add_argument("--force-extract", action="store_true", help="Rebuild page Markdown/chunks")
    ingest.add_argument(
        "--force-vectorize", action="store_true", help="Rebuild the zvec collection"
    )
    ingest.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Extract only the first N pages; intended for smoke tests",
    )

    package = subparsers.add_parser(
        "package",
        help="Create a bounded context package for an agent task and lifecycle stage",
    )
    package.add_argument("--task", required=True, help="Agent task or mission statement")
    package.add_argument("--stage", required=True, choices=STAGES, help="Lifecycle stage")
    package.add_argument("--next-steps", default="", help="Known next steps or constraints")
    package.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="Root containing chunks and zvec vectors",
    )
    package.add_argument("--top-k", type=int, default=8, dest="top_k", help="Chunks to retrieve")
    package.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Package output format",
    )
    package.add_argument(
        "--max-snippet-chars",
        type=int,
        default=1200,
        dest="max_snippet_chars",
        help="Maximum characters per retrieved snippet",
    )
    package.add_argument("--output", default=None, help="Optional output file. Defaults to stdout.")

    subparsers.add_parser(
        "stages",
        help="Print the chainable lifecycle stages and their retrieval lenses",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "ingest-ddia":
        result = ingest_ddia(
            url=args.url,
            artifact_root=Path(args.artifact_root),
            force_download=args.force_download,
            force_extract=args.force_extract,
            force_vectorize=args.force_vectorize,
            max_pages=args.max_pages,
        )
        print_ingest_result(result)
        return 0

    if args.command == "package":
        output = Path(args.output) if args.output else None
        rendered = build_context_package(
            task=args.task,
            stage=args.stage,
            next_steps=args.next_steps,
            artifact_root=Path(args.artifact_root),
            top_k=args.top_k,
            output=output,
            output_format=args.format,
            max_snippet_chars=args.max_snippet_chars,
        )
        if output is None:
            _write_stdout_utf8(rendered)
        else:
            print(f"wrote={output}", file=sys.stderr)
        print(f"next_stage={next_stage(args.stage)}", file=sys.stderr)
        return 0

    if args.command == "stages":
        for stage, lens in STAGE_LENSES.items():
            print(f"{stage} -> {next_stage(stage)}")
            for item in lens:
                print(f"  - {item}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
