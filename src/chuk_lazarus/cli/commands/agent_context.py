"""agent-context CLI commands."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path


def _write_stdout_utf8(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(text)


def agent_context_ingest_cmd(args: Namespace) -> None:
    from ...agent_context import DEFAULT_DDIA_URL, ingest_ddia
    from ...agent_context.ddia import print_ingest_result

    result = ingest_ddia(
        url=args.url or DEFAULT_DDIA_URL,
        artifact_root=Path(args.artifact_root),
        force_download=args.force_download,
        force_extract=args.force_extract,
        force_vectorize=args.force_vectorize,
        max_pages=args.max_pages,
    )
    print_ingest_result(result)


def agent_context_package_cmd(args: Namespace) -> None:
    from ...agent_context import build_context_package, next_stage

    output = Path(args.output) if args.output else None
    rendered = build_context_package(
        task=args.task,
        stage=args.stage,
        next_steps=args.next_steps or "",
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


def agent_context_stages_cmd(args: Namespace) -> None:
    from ...agent_context.ddia import STAGE_LENSES, next_stage

    for stage, lens in STAGE_LENSES.items():
        print(f"{stage} -> {next_stage(stage)}")
        for item in lens:
            print(f"  - {item}")
