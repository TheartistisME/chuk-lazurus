"""DDIA-derived agent context pipeline.

The on-disk layout mirrors the book's operating principles without requiring
agents to hold the whole book in context: immutable source, page Markdown as a
replayable log, zvec as a derived materialized index, and bounded context
packages as query-time views.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DDIA_URL = (
    "https://0-lucas.github.io/digital-garden/99.-Books/"
    "Martin-Kleppmann---Designing-Data-Intensive-Applications_-"
    "O%E2%80%99Reilly-Media-(2017).pdf"
)
TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = TOOL_ROOT / "artifacts" / "ddia"
DEFAULT_EMBED_DIM = 384

STAGES = ("onboard", "plan", "build", "verify", "handoff", "exit")
NEXT_STAGE = {
    "onboard": "plan",
    "plan": "build",
    "build": "verify",
    "verify": "handoff",
    "handoff": "exit",
    "exit": "onboard",
}

STAGE_LENSES = {
    "onboard": [
        "establish source of truth",
        "identify system boundaries",
        "read manifests before derived indexes",
        "prefer explicit schema and provenance",
    ],
    "plan": [
        "separate commands from materialized views",
        "choose consistency and durability contracts",
        "partition work into replayable stages",
        "define observability before execution",
    ],
    "build": [
        "make writes idempotent",
        "append durable facts before deriving indexes",
        "treat caches and vector indexes as rebuildable state",
        "keep bounded interfaces between stages",
    ],
    "verify": [
        "test failure modes and recovery",
        "validate schema versions and invariants",
        "compare derived data to immutable sources",
        "surface staleness and incomplete indexes",
    ],
    "handoff": [
        "package provenance with decisions",
        "record open questions as explicit follow-up",
        "summarize state transitions and next commands",
        "make downstream reads monotonic and bounded",
    ],
    "exit": [
        "close loops with durable status",
        "sync sources and derived manifests",
        "leave replay instructions",
        "make abandoned work visible",
    ],
}

PRINCIPLE_KEYWORDS = {
    "source-of-truth": ("log", "record", "database", "source", "truth", "system of record"),
    "derived-index": ("index", "view", "cache", "materialized", "secondary"),
    "schema-evolution": ("schema", "version", "migration", "compatibility", "encoding"),
    "consistency": ("consistency", "linearizable", "serializable", "transaction", "isolation"),
    "durability": ("durable", "replication", "recovery", "failure", "snapshot"),
    "partitioning": ("partition", "shard", "replica", "leader", "quorum"),
    "batch-stream": ("batch", "stream", "event", "window", "incremental"),
    "observability": ("monitor", "metric", "debug", "trace", "audit"),
}


@dataclass(frozen=True)
class IngestResult:
    artifact_root: Path
    pdf_path: Path
    markdown_dir: Path
    chunks_path: Path
    vector_path: Path
    pages: int
    chunks: int


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float
    page: int
    source_path: str
    principle_tags: tuple[str, ...]
    stage_tags: tuple[str, ...]
    text: str


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def safe_slug(value: str, fallback: str = "package") -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or fallback


def next_stage(stage: str) -> str:
    stage = normalize_stage(stage)
    return NEXT_STAGE[stage]


def normalize_stage(stage: str) -> str:
    normalized = stage.strip().lower()
    if normalized not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}")
    return normalized


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_within(parent: Path, child: Path) -> None:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve()
    if parent_resolved == child_resolved:
        return
    if parent_resolved not in child_resolved.parents:
        raise ValueError(f"refusing to operate outside {parent_resolved}: {child_resolved}")


def download_pdf(url: str, pdf_path: Path, *, force: bool = False) -> Path:
    if pdf_path.exists() and not force:
        return pdf_path

    ensure_parent(pdf_path)
    tmp_path = pdf_path.with_suffix(".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "iddia-agent-context"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    tmp_path.replace(pdf_path)
    return pdf_path


def write_source_manifest(artifact_root: Path, pdf_url: str, pdf_path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "artifact": "ddia-source",
        "url": pdf_url,
        "pdf_path": str(pdf_path.as_posix()),
        "pdf_sha256": sha256_path(pdf_path),
        "recorded_at": utc_now(),
        "principle": "source files are immutable inputs; every derived view stores provenance",
    }
    path = artifact_root / "source" / "manifest.json"
    ensure_parent(path)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_page_markdown(
    pdf_path: Path,
    markdown_dir: Path,
    *,
    force: bool = False,
    max_pages: int | None = None,
) -> int:
    """Split a PDF into single-page PDFs and run MarkItDown on each page."""

    try:
        from markitdown import MarkItDown
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:  # pragma: no cover - exercised by CLI users.
        raise RuntimeError(
            "PDF extraction needs optional dependencies: "
            "python -m pip install 'markitdown[pdf]' pypdf"
        ) from exc

    markdown_dir.mkdir(parents=True, exist_ok=True)
    single_page_dir = markdown_dir.parent / "_single_page_pdf"
    single_page_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    limit = min(total_pages, max_pages) if max_pages else total_pages
    pdf_hash = sha256_path(pdf_path)
    converter = MarkItDown(enable_plugins=False)

    for page_index in range(limit):
        page_no = page_index + 1
        out_path = markdown_dir / f"page_{page_no:04d}.md"
        if out_path.exists() and not force:
            continue

        writer = PdfWriter()
        writer.add_page(reader.pages[page_index])
        page_pdf = single_page_dir / f"page_{page_no:04d}.pdf"
        with page_pdf.open("wb") as handle:
            writer.write(handle)

        result = converter.convert(str(page_pdf))
        text = result.text_content.strip()
        front_matter = {
            "schema_version": 1,
            "source": "Designing Data-Intensive Applications",
            "source_pdf": str(pdf_path.as_posix()),
            "source_pdf_sha256": pdf_hash,
            "page": page_no,
            "generated_by": "microsoft/markitdown",
            "generated_at": utc_now(),
        }
        out_path.write_text(
            "---\n"
            + "\n".join(f"{key}: {json.dumps(value)}" for key, value in front_matter.items())
            + "\n---\n\n"
            + text
            + "\n",
            encoding="utf-8",
        )

    manifest = {
        "schema_version": 1,
        "artifact": "page-markdown-log",
        "source_pdf": str(pdf_path.as_posix()),
        "source_pdf_sha256": pdf_hash,
        "markdown_dir": str(markdown_dir.as_posix()),
        "pages_total": total_pages,
        "pages_extracted": limit,
        "generated_by": "microsoft/markitdown",
        "generated_at": utc_now(),
        "principle": "page files are an ordered log; vector data is a rebuildable index over it",
    }
    manifest_path = markdown_dir.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return limit


def strip_front_matter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end >= 0:
            return text[end + len("\n---\n") :].lstrip()
    return text


def chunk_markdown_text(
    text: str, *, target_chars: int = 1800, overlap_chars: int = 180
) -> list[str]:
    text = strip_front_matter(text).strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
        current = []
        current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > target_chars:
            flush()
            start = 0
            while start < len(paragraph):
                end = min(start + target_chars, len(paragraph))
                chunks.append(paragraph[start:end].strip())
                if end == len(paragraph):
                    break
                start = max(end - overlap_chars, start + 1)
            continue

        extra_len = len(paragraph) + (2 if current else 0)
        if current and current_len + extra_len > target_chars:
            flush()
        current.append(paragraph)
        current_len += extra_len

    flush()
    return [chunk for chunk in chunks if chunk]


def infer_tags(text: str, page: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    lower = text.lower()
    principles = [
        tag
        for tag, keywords in PRINCIPLE_KEYWORDS.items()
        if any(keyword in lower for keyword in keywords)
    ]
    stages = [
        stage
        for stage, lens_terms in STAGE_LENSES.items()
        if any(term in lower for term in lens_terms)
    ]
    if page <= 20 and "onboard" not in stages:
        stages.append("onboard")
    return tuple(principles), tuple(stages)


def build_chunks(markdown_dir: Path, chunks_path: Path, *, force: bool = False) -> int:
    if chunks_path.exists() and not force:
        return sum(1 for _ in chunks_path.open("r", encoding="utf-8"))

    ensure_parent(chunks_path)
    page_paths = sorted(markdown_dir.glob("page_*.md"))
    count = 0
    with chunks_path.open("w", encoding="utf-8") as handle:
        for page_path in page_paths:
            match = re.search(r"page_(\d+)\.md$", page_path.name)
            page = int(match.group(1)) if match else 0
            text = page_path.read_text(encoding="utf-8", errors="replace")
            for chunk_index, chunk in enumerate(chunk_markdown_text(text)):
                principle_tags, stage_tags = infer_tags(chunk, page)
                text_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                record = {
                    "schema_version": 1,
                    "id": f"ddia-p{page:04d}-c{chunk_index:02d}",
                    "source": "ddia",
                    "page": page,
                    "chunk_index": chunk_index,
                    "source_path": str(page_path.as_posix()),
                    "text_hash": text_hash,
                    "char_count": len(chunk),
                    "principle_tags": list(principle_tags),
                    "stage_tags": list(stage_tags),
                    "text": chunk,
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    return count


def tokenize_for_embedding(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_'-]*", text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


def embed_hash(text: str, *, dim: int = DEFAULT_EMBED_DIM) -> list[float]:
    """Deterministic feature-hashing embedding for local zvec indexing."""

    vector = [0.0] * dim
    tokens = tokenize_for_embedding(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=12).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        weight = 1.0 + min(len(token), 24) / 24.0
        vector[index] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_chunks(chunks_path: Path) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in iter_jsonl(chunks_path)}


def remove_rebuildable_path(path: Path, artifact_root: Path) -> None:
    ensure_within(artifact_root, path)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def vectorize_chunks(
    artifact_root: Path,
    chunks_path: Path,
    vector_path: Path,
    *,
    force: bool = False,
    dim: int = DEFAULT_EMBED_DIM,
) -> int:
    try:
        import zvec
    except ImportError as exc:  # pragma: no cover - exercised by CLI users.
        raise RuntimeError("zvec is required: python -m pip install zvec") from exc

    manifest_path = vector_path.parent / "manifest.json"
    if vector_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return int(manifest.get("chunks_indexed", 0))

    if force:
        remove_rebuildable_path(vector_path, artifact_root)
    vector_path.parent.mkdir(parents=True, exist_ok=True)

    schema = zvec.CollectionSchema(
        name="ddia_agent_context",
        fields=[
            zvec.FieldSchema("source", zvec.DataType.STRING),
            zvec.FieldSchema("page", zvec.DataType.UINT32),
            zvec.FieldSchema("chunk_index", zvec.DataType.UINT32),
            zvec.FieldSchema("source_path", zvec.DataType.STRING),
            zvec.FieldSchema("principle_tags", zvec.DataType.ARRAY_STRING),
            zvec.FieldSchema("stage_tags", zvec.DataType.ARRAY_STRING),
            zvec.FieldSchema("text_hash", zvec.DataType.STRING),
        ],
        vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, dim),
    )
    collection = zvec.create_and_open(str(vector_path), schema=schema)
    count = 0
    batch = []
    for record in iter_jsonl(chunks_path):
        embedding_text = " ".join(
            [
                record["text"],
                " ".join(record.get("principle_tags", [])),
                " ".join(record.get("stage_tags", [])),
            ]
        )
        batch.append(
            zvec.Doc(
                id=record["id"],
                vectors={"embedding": embed_hash(embedding_text, dim=dim)},
                fields={
                    "source": record["source"],
                    "page": int(record["page"]),
                    "chunk_index": int(record["chunk_index"]),
                    "source_path": record["source_path"],
                    "principle_tags": list(record.get("principle_tags", [])),
                    "stage_tags": list(record.get("stage_tags", [])),
                    "text_hash": record["text_hash"],
                },
            )
        )
        count += 1
        if len(batch) >= 128:
            collection.insert(batch)
            batch = []
    if batch:
        collection.insert(batch)
    collection.optimize()

    manifest = {
        "schema_version": 1,
        "artifact": "zvec-materialized-index",
        "vector_path": str(vector_path.as_posix()),
        "chunks_path": str(chunks_path.as_posix()),
        "chunks_sha256": sha256_path(chunks_path),
        "chunks_indexed": count,
        "embedding": "deterministic-feature-hash",
        "embedding_dimension": dim,
        "created_at": utc_now(),
        "principle": "the vector store is a rebuildable materialized view over chunk records",
    }
    ensure_parent(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return count


def ingest_ddia(
    *,
    url: str = DEFAULT_DDIA_URL,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    force_download: bool = False,
    force_extract: bool = False,
    force_vectorize: bool = False,
    max_pages: int | None = None,
) -> IngestResult:
    artifact_root = Path(artifact_root)
    pdf_path = artifact_root / "source" / "pdf" / "ddia.pdf"
    markdown_dir = artifact_root / "markdown" / "pages"
    chunks_path = artifact_root / "chunks" / "chunks.jsonl"
    vector_path = artifact_root / "vectors" / "zvec"

    download_pdf(url, pdf_path, force=force_download)
    write_source_manifest(artifact_root, url, pdf_path)
    pages = extract_page_markdown(
        pdf_path,
        markdown_dir,
        force=force_extract,
        max_pages=max_pages,
    )
    chunks = build_chunks(markdown_dir, chunks_path, force=force_extract)
    indexed = vectorize_chunks(
        artifact_root,
        chunks_path,
        vector_path,
        force=force_vectorize or force_extract,
    )
    return IngestResult(
        artifact_root=artifact_root,
        pdf_path=pdf_path,
        markdown_dir=markdown_dir,
        chunks_path=chunks_path,
        vector_path=vector_path,
        pages=pages,
        chunks=indexed or chunks,
    )


def build_context_query(task: str, stage: str, next_steps: str) -> str:
    stage = normalize_stage(stage)
    lens = "\n".join(f"- {term}" for term in STAGE_LENSES[stage])
    return (
        f"Agent task:\n{task.strip()}\n\n"
        f"Building stage: {stage}\n"
        f"Stage lens:\n{lens}\n\n"
        f"Next steps:\n{next_steps.strip()}\n\n"
        "Retrieve database design principles, failure modes, data layout advice, "
        "indexing guidance, consistency contracts, and operational lifecycle knowledge "
        "that should shape this agent's immediate work."
    )


def search_context(
    *,
    artifact_root: Path,
    task: str,
    stage: str,
    next_steps: str,
    top_k: int = 8,
    dim: int = DEFAULT_EMBED_DIM,
) -> list[SearchHit]:
    try:
        import zvec
    except ImportError as exc:  # pragma: no cover - exercised by CLI users.
        raise RuntimeError("zvec is required: python -m pip install zvec") from exc

    artifact_root = Path(artifact_root)
    chunks_path = artifact_root / "chunks" / "chunks.jsonl"
    vector_path = artifact_root / "vectors" / "zvec"
    if not chunks_path.exists() or not vector_path.exists():
        raise FileNotFoundError(
            f"missing context index under {artifact_root}; run 'python -m IDDIA ingest-ddia'"
        )

    chunks = load_chunks(chunks_path)
    query = build_context_query(task, stage, next_steps)
    collection = zvec.open(str(vector_path))
    results = collection.query(
        zvec.VectorQuery("embedding", vector=embed_hash(query, dim=dim)),
        topk=top_k,
    )

    hits: list[SearchHit] = []
    for result in results:
        chunk_id = result["id"] if isinstance(result, dict) else result.id
        score = float(result.get("score", 0.0) if isinstance(result, dict) else result.score)
        record = chunks.get(chunk_id)
        if record is None:
            continue
        hits.append(
            SearchHit(
                chunk_id=record["id"],
                score=score,
                page=int(record["page"]),
                source_path=record["source_path"],
                principle_tags=tuple(record.get("principle_tags", [])),
                stage_tags=tuple(record.get("stage_tags", [])),
                text=record["text"],
            )
        )
    return hits


def stage_questions(stage: str) -> list[str]:
    stage = normalize_stage(stage)
    questions = {
        "onboard": [
            "What is the durable source of truth?",
            "Which derived artifacts can be rebuilt?",
            "Which schema or manifest must be read first?",
        ],
        "plan": [
            "What are the write path, read path, and materialized views?",
            "Where can stale derived state appear?",
            "Which command is idempotent enough to retry?",
        ],
        "build": [
            "What is append-only or atomic in this change?",
            "How will partial failure be detected and resumed?",
            "Which index or cache must carry provenance?",
        ],
        "verify": [
            "Can the derived output be checked against the source?",
            "What invariant proves the stage is complete?",
            "What failure mode should be represented in tests?",
        ],
        "handoff": [
            "What changed, what was verified, and what remains?",
            "Which command should the next agent run first?",
            "What state is durable versus local cache?",
        ],
        "exit": [
            "Are issues, commits, and remote state synchronized?",
            "Can a fresh agent replay the lifecycle from manifests?",
            "Is any incomplete work visible as an explicit follow-up?",
        ],
    }
    return questions[stage]


def trim_snippet(text: str, limit: int) -> str:
    compact = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 20)].rstrip() + "\n\n[snippet truncated]"


def render_context_package_markdown(
    *,
    task: str,
    stage: str,
    next_steps: str,
    hits: list[SearchHit],
    top_k: int,
    max_snippet_chars: int,
) -> str:
    stage = normalize_stage(stage)
    next_name = next_stage(stage)
    lens = STAGE_LENSES[stage]
    questions = stage_questions(stage)
    retrieved = "\n".join(
        [
            f"- {hit.chunk_id}: page {hit.page}, score {hit.score:.4f}, "
            f"tags={','.join(hit.principle_tags or hit.stage_tags or ('untagged',))}"
            for hit in hits
        ]
    )
    if not retrieved:
        retrieved = "- No hits returned."

    sections = [
        "# Agent Context Package",
        "",
        "## Contract",
        "",
        f"- Stage: {stage}",
        f"- Next stage: {next_name}",
        f"- Task: {task.strip()}",
        f"- Next steps: {next_steps.strip() or 'not supplied'}",
        f"- Retrieved chunks: {len(hits)} of requested {top_k}",
        f"- Generated at: {utc_now()}",
        "",
        "## Stage Lens",
        "",
        *[f"- {item}" for item in lens],
        "",
        "## Questions To Keep Live",
        "",
        *[f"- {item}" for item in questions],
        "",
        "## Retrieved Evidence",
        "",
        retrieved,
        "",
    ]

    for index, hit in enumerate(hits, start=1):
        sections.extend(
            [
                f"### Hit {index}: {hit.chunk_id}",
                "",
                f"- Page: {hit.page}",
                f"- Score: {hit.score:.4f}",
                f"- Source path: {hit.source_path}",
                f"- Principle tags: {', '.join(hit.principle_tags) or 'none'}",
                "",
                trim_snippet(hit.text, max_snippet_chars),
                "",
            ]
        )

    sections.extend(
        [
            "## Chain",
            "",
            f'- Recommended next package: `python -m IDDIA package --stage {next_name} --task "..." --next-steps "..."`',
            f"- Slash command next hop: `/agent-context:{next_name}`",
            "",
        ]
    )
    return "\n".join(sections).rstrip() + "\n"


def build_context_package(
    *,
    task: str,
    stage: str,
    next_steps: str,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    top_k: int = 8,
    output: Path | None = None,
    output_format: str = "markdown",
    max_snippet_chars: int = 1200,
) -> str:
    stage = normalize_stage(stage)
    hits = search_context(
        artifact_root=artifact_root,
        task=task,
        stage=stage,
        next_steps=next_steps,
        top_k=top_k,
    )
    output_format = output_format.lower()
    if output_format == "json":
        payload: dict[str, Any] = {
            "schema_version": 1,
            "stage": stage,
            "next_stage": next_stage(stage),
            "task": task,
            "next_steps": next_steps,
            "generated_at": utc_now(),
            "stage_lens": STAGE_LENSES[stage],
            "questions": stage_questions(stage),
            "hits": [
                {
                    "chunk_id": hit.chunk_id,
                    "score": hit.score,
                    "page": hit.page,
                    "source_path": hit.source_path,
                    "principle_tags": list(hit.principle_tags),
                    "stage_tags": list(hit.stage_tags),
                    "text": trim_snippet(hit.text, max_snippet_chars),
                }
                for hit in hits
            ],
        }
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    elif output_format in {"md", "markdown"}:
        rendered = render_context_package_markdown(
            task=task,
            stage=stage,
            next_steps=next_steps,
            hits=hits,
            top_k=top_k,
            max_snippet_chars=max_snippet_chars,
        )
    else:
        raise ValueError("output_format must be 'markdown' or 'json'")

    if output is not None:
        ensure_parent(output)
        output.write_text(rendered, encoding="utf-8")
    return rendered


def print_ingest_result(result: IngestResult) -> None:
    print(f"artifact_root={result.artifact_root}", file=sys.stderr)
    print(f"pdf={result.pdf_path}", file=sys.stderr)
    print(f"markdown_dir={result.markdown_dir}", file=sys.stderr)
    print(f"chunks={result.chunks_path}", file=sys.stderr)
    print(f"vectors={result.vector_path}", file=sys.stderr)
    print(f"pages={result.pages}", file=sys.stderr)
    print(f"chunks_indexed={result.chunks}", file=sys.stderr)
