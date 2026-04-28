"""Agent context command parsers."""

from __future__ import annotations

from ..commands.agent_context import (
    agent_context_ingest_cmd,
    agent_context_package_cmd,
    agent_context_stages_cmd,
)


def register_agent_context_parsers(subparsers):
    from ...agent_context import DEFAULT_ARTIFACT_ROOT, DEFAULT_DDIA_URL, STAGES

    parser = subparsers.add_parser(
        "agent-context",
        help="Build and query agent context packages from a zvec-backed DDIA index",
    )
    ac_subparsers = parser.add_subparsers(
        dest="agent_context_command",
        help="Agent context commands",
    )

    ingest = ac_subparsers.add_parser(
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
    ingest.set_defaults(func=agent_context_ingest_cmd)

    package = ac_subparsers.add_parser(
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
    package.add_argument(
        "--output",
        default=None,
        help="Optional output file. Defaults to stdout.",
    )
    package.set_defaults(func=agent_context_package_cmd)

    stages = ac_subparsers.add_parser(
        "stages",
        help="Print the chainable lifecycle stages and their retrieval lenses",
    )
    stages.set_defaults(func=agent_context_stages_cmd)

    return parser
