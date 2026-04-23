"""Thin subprocess wrapper around the ``larql`` CLI.

Lets the chuk-lazurus Python backend drive LARQL (Lazarus Query Language)
for knowledge-graph operations on vindex/graph artifacts produced by the
Rust ``larql`` binary (https://github.com/cronos3k/larql).

Typical uses:

* Validate or stat a `.larql.json` knowledge graph exported alongside a
  chat session's boundary residuals.
* Issue entity / subject lookups (``describe`` / ``query``) to provide
  deterministic, zero-GPU fact retrieval alongside the KV-direct path.
* Drive ``larql build`` to bake entity-relation-target triples extracted
  from chat turns into model weights (the research-only "LARQL baking"
  step, kept out of the live kv_direct REPL for now).

The wrapper intentionally keeps shell-out shape (``larql <subcommand>
[--flag value] <positional>``) so users retain direct CLI access for
debugging. Failures surface the full stderr as :class:`LarqlError`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LarqlError(RuntimeError):
    """Raised when the ``larql`` CLI exits non-zero."""


@dataclass(frozen=True)
class LarqlStats:
    """Parsed output of ``larql stats <GRAPH>``."""

    graph_path: str
    entities: int
    edges: int
    relations: int
    components: int
    avg_degree: float
    avg_conf: float
    sources: dict[str, int]


def _resolve_binary(path: str | None = None) -> str:
    """Find the ``larql`` binary.

    Order: explicit ``path`` → ``$PATH`` lookup → common cargo bin.
    """
    if path:
        return path
    found = shutil.which("larql")
    if found:
        return found
    for candidate in (
        Path.home() / ".cargo" / "bin" / "larql",
        Path("/usr/local/bin/larql"),
    ):
        if candidate.is_file():
            return str(candidate)
    raise LarqlError(
        "Could not locate the `larql` binary. Install it with `cargo build "
        "--release -p larql-cli` or put it on $PATH."
    )


def _run(
    args: list[str],
    *,
    check: bool = True,
    binary: str | None = None,
    input_text: str | None = None,
    timeout: int | None = 120,
) -> subprocess.CompletedProcess[str]:
    """Invoke the ``larql`` binary and return the CompletedProcess.

    Raises :class:`LarqlError` on non-zero exit when ``check`` is True.
    """
    cmd = [_resolve_binary(binary), *args]
    proc = subprocess.run(
        cmd,
        text=True,
        input=input_text,
        capture_output=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise LarqlError(
            f"larql {' '.join(args)!r} exited {proc.returncode}; "
            f"stderr={proc.stderr.strip()!r}"
        )
    return proc


def version(*, binary: str | None = None) -> str:
    """Return the ``larql`` CLI version string (e.g. ``'larql 0.1.0'``)."""
    return _run(["--version"], binary=binary).stdout.strip()


def validate(graph: str | Path, *, binary: str | None = None) -> str:
    """Run ``larql validate <GRAPH>``; return raw stdout (success line).

    Raises :class:`LarqlError` if the graph is malformed or missing.
    """
    return _run(["validate", str(graph)], binary=binary).stdout.strip()


def stats(graph: str | Path, *, binary: str | None = None) -> LarqlStats:
    """Parse the output of ``larql stats <GRAPH>`` into :class:`LarqlStats`."""
    out = _run(["stats", str(graph)], binary=binary).stdout
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    header = next((ln for ln in lines if ln.startswith("Graph:")), None)
    graph_path = header.split("Graph:", 1)[1].strip() if header else str(graph)
    flat: dict[str, str] = {}
    in_sources = False
    sources: dict[str, int] = {}
    for ln in lines:
        if ln.startswith("Sources:"):
            in_sources = True
            continue
        if in_sources:
            parts = ln.split()
            if len(parts) == 2 and parts[1].isdigit():
                sources[parts[0]] = int(parts[1])
            continue
        if ":" in ln:
            key, _, value = ln.partition(":")
            flat[key.strip().lower()] = value.strip()
    try:
        return LarqlStats(
            graph_path=graph_path,
            entities=int(flat.get("entities", "0")),
            edges=int(flat.get("edges", "0")),
            relations=int(flat.get("relations", "0")),
            components=int(flat.get("components", "0")),
            avg_degree=float(flat.get("avg degree", "0")),
            avg_conf=float(flat.get("avg conf", "0")),
            sources=sources,
        )
    except ValueError as exc:  # noqa: BLE001 - surface with context
        raise LarqlError(f"stats parse failed: {exc!r}; raw={out!r}") from exc


def describe(
    graph: str | Path, entity: str, *, binary: str | None = None
) -> str:
    """Run ``larql describe --graph <GRAPH> <ENTITY>``; return raw stdout."""
    return _run(
        ["describe", "--graph", str(graph), entity], binary=binary
    ).stdout


def query(
    graph: str | Path,
    subject: str,
    relation: str | None = None,
    *,
    binary: str | None = None,
) -> str:
    """Run ``larql query --graph <GRAPH> <SUBJECT> [RELATION]``.

    Returns raw stdout (pretty-printed edge list).
    """
    args = ["query", "--graph", str(graph), subject]
    if relation is not None:
        args.append(relation)
    return _run(args, binary=binary).stdout


def lql(statement: str, *, binary: str | None = None) -> str:
    """Execute a single LQL statement via ``larql lql``.

    Example::

        lql('USE "model.vindex"; DESCRIBE "France";')
    """
    return _run(["lql", statement], binary=binary).stdout


def emit_session_graph(
    edges: list[dict[str, Any]],
    *,
    model: str = "google/gemma-4-E2B-it",
    extraction_method: str = "chuk-lazurus-session",
    relations_schema: list[dict[str, Any]] | None = None,
    output_path: str | Path,
) -> Path:
    """Write a `.larql.json` graph file compatible with ``larql validate``.

    Designed to let chuk-lazurus export archived chat memories as a
    knowledge graph that the LARQL CLI can query / describe / bake.

    Parameters
    ----------
    edges:
        List of ``{"s": subject, "r": relation, "o": object, "c": conf,
        "src": source}`` records. Missing ``c``/``src`` default to
        ``1.0`` / ``"chuk-lazurus"``.
    model:
        HuggingFace model id to record in the graph metadata (so the
        downstream ``larql build`` knows which weights to patch).
    extraction_method:
        Free-form label for the graph metadata.
    relations_schema:
        Optional ``schema.relations`` list. When omitted a minimal
        self-describing schema is derived from the relations seen in
        ``edges``.
    output_path:
        Destination ``.larql.json`` file.
    """
    from collections import OrderedDict

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized: list[dict[str, Any]] = []
    rels_seen: "OrderedDict[str, None]" = OrderedDict()
    for edge in edges:
        if not {"s", "r", "o"}.issubset(edge.keys()):
            raise LarqlError(
                f"edge missing required key(s) {{'s','r','o'}}; got {edge!r}"
            )
        normalized.append(
            {
                "s": str(edge["s"]),
                "r": str(edge["r"]),
                "o": str(edge["o"]),
                "c": float(edge.get("c", 1.0)),
                "src": str(edge.get("src", "chuk-lazurus")),
            }
        )
        rels_seen[str(edge["r"])] = None

    if relations_schema is None:
        relations_schema = [
            {
                "name": rel,
                "subject_types": ["entity"],
                "object_types": ["entity"],
                "reversible": False,
            }
            for rel in rels_seen
        ]

    document = {
        "larql_version": "0.1.0",
        "metadata": {
            "model": model,
            "extraction_method": extraction_method,
        },
        "schema": {
            "relations": relations_schema,
            "type_rules": [],
        },
        "edges": normalized,
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def verify_vindex(vindex_dir: str | Path, *, binary: str | None = None) -> str:
    """Run ``larql verify <VINDEX_DIR>`` (SHA256 integrity check). Return stdout."""
    return _run(["verify", str(vindex_dir)], binary=binary, timeout=600).stdout


def walk_vindex(
    vindex_dir: str | Path,
    prompt: str,
    *,
    top_k: int | None = None,
    binary: str | None = None,
    timeout: int = 300,
) -> str:
    """Walk a prompt through the vindex as a local vector index.

    Returns raw stdout — the per-layer top-K feature activation listing.
    """
    args = ["walk", "--index", str(vindex_dir), "--prompt", prompt]
    if top_k is not None:
        args.extend(["--top-k", str(top_k)])
    return _run(args, binary=binary, timeout=timeout).stdout


def infer_vindex(
    vindex_dir: str | Path,
    prompt: str,
    *,
    top: int = 5,
    binary: str | None = None,
    timeout: int = 300,
) -> str:
    """Run an LQL ``INFER`` statement against a vindex.

    Example::

        infer_vindex("gemma4-e2b.vindex", "The capital of France is", top=3)
    """
    # Quote the prompt safely — LQL uses standard SQL-style double quotes.
    safe = prompt.replace('"', '\\"')
    stmt = f'USE "{vindex_dir}"; INFER "{safe}" TOP {int(top)};'
    return _run(["lql", stmt], binary=binary, timeout=timeout).stdout


def vindex_metadata(vindex_dir: str | Path) -> dict[str, Any]:
    """Return the parsed ``index.json`` of a vindex directory.

    Zero-subprocess — just JSON-reads the manifest that ships inside
    every vindex. Useful for surface checks (layer count, family, extract
    level, layer bands) before invoking heavier CLI commands.
    """
    path = Path(vindex_dir) / "index.json"
    if not path.is_file():
        raise LarqlError(f"vindex has no index.json at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def is_available(binary: str | None = None) -> bool:
    """True iff the larql binary is callable on this system."""
    try:
        _run(["--version"], binary=binary, timeout=10)
    except (LarqlError, subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return True


__all__ = [
    "LarqlError",
    "LarqlStats",
    "describe",
    "emit_session_graph",
    "infer_vindex",
    "is_available",
    "lql",
    "query",
    "stats",
    "validate",
    "verify_vindex",
    "version",
    "vindex_metadata",
    "walk_vindex",
]
