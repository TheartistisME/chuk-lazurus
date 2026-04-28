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
import time
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
REFERENCE_CHUNK_PENALTY = 0.12
NOISE_CHUNK_PENALTIES = {
    "reference_like": REFERENCE_CHUNK_PENALTY,
    "table_of_contents": 0.18,
    "front_matter": 0.10,
    "chapter_opener": 0.08,
    "index_like": 0.16,
}
NOISE_FILTER_FLAGS = frozenset(
    {"table_of_contents", "front_matter", "chapter_opener", "index_like"}
)
ZVEC_OPEN_RETRY_ATTEMPTS = 8
ZVEC_OPEN_RETRY_DELAY_SECONDS = 0.15

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "before",
        "after",
        "be",
        "between",
        "by",
        "can",
        "check",
        "could",
        "each",
        "enough",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "made",
        "make",
        "needs",
        "new",
        "of",
        "old",
        "on",
        "one",
        "or",
        "our",
        "should",
        "that",
        "the",
        "their",
        "them",
        "this",
        "through",
        "to",
        "use",
        "used",
        "using",
        "with",
        "within",
        "without",
        "would",
        "will",
        "what",
        "when",
        "where",
        "which",
        "why",
        "was",
        "were",
        "while",
        "agent",
        "building",
        "context",
        "next",
        "package",
        "retrieve",
        "stage",
        "steps",
        "task",
    }
)

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

CONCEPT_KEYWORDS = {
    "atomic": ("atomic", "atomicity", "atomic commit", "compare-and-set", "cas"),
    "batch": ("batch", "batch processing", "bulk", "mapreduce"),
    "checkpoint": ("checkpoint", "checkpoints", "checkpointing", "savepoint"),
    "consistency": (
        "consistency",
        "consistent",
        "linearizable",
        "linearizability",
        "serializable",
        "serializability",
        "isolation",
        "tenant isolation",
        "contamination",
    ),
    "deterministic": (
        "deterministic",
        "determinism",
        "repeatable",
        "reproducible",
        "fingerprint",
        "drift",
        "same input",
    ),
    "durability": ("durability", "durable", "fsync", "recovery", "recoverable"),
    "event-log": (
        "event log",
        "event-log",
        "events",
        "append-only log",
        "commit log",
        "log-structured",
        "journal",
    ),
    "manifest": ("manifest", "manifests", "metadata file", "catalog"),
    "materialized-view": (
        "materialized view",
        "projection",
        "derived view",
        "view maintenance",
        "secondary index",
        "cache",
    ),
    "partition": ("partition", "partitioning", "shard", "sharding", "split"),
    "provenance-lineage": (
        "provenance",
        "lineage",
        "audit",
        "causality",
        "causal",
        "explain",
        "explainability",
        "parent",
        "selection",
        "selected",
        "policy",
        "history",
        "random seed",
        "seed",
    ),
    "replay": ("replay", "replaying", "rebuild", "recompute", "backfill"),
    "failure-recovery": (
        "crash",
        "crashes",
        "restart",
        "restarts",
        "outage",
        "node outage",
        "partial failure",
        "failed",
        "failure",
        "stale",
        "cleanup",
    ),
    "schema": (
        "schema",
        "schema evolution",
        "migration",
        "encoding",
        "compatibility",
        "backward compatibility",
        "forward compatibility",
        "avro",
        "protocol buffers",
        "thrift",
        "json",
        "xml",
    ),
    "snapshot": ("snapshot", "snapshotting", "point-in-time", "copy-on-write"),
    "source-of-truth": (
        "source of truth",
        "source-of-truth",
        "system of record",
        "authoritative",
        "canonical",
    ),
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
    concept_tags: tuple[str, ...] = ()
    chapter_title: str = ""
    section_title: str = ""
    vector_score: float = 0.0
    lexical_score: float = 0.0
    query_boost: float = 0.0
    noise_penalty: float = 0.0
    matched_terms: tuple[str, ...] = ()
    matched_tags: tuple[str, ...] = ()
    noise_flags: tuple[str, ...] = ()
    why_this_hit: dict[str, Any] | None = None


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


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = phrase.lower()
    if re.search(r"[\s-]", phrase):
        pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"[\s-]+") + r"\b"
        return re.search(pattern, text) is not None
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def infer_concept_tags(text: str) -> tuple[str, ...]:
    lower = text.lower()
    return tuple(
        tag
        for tag, keywords in CONCEPT_KEYWORDS.items()
        if any(_contains_phrase(lower, keyword) for keyword in keywords)
    )


def tokenize_lexical_terms(text: str) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]
    phrases = [
        " ".join(parts)
        for size in (2, 3)
        for parts in zip(*(tokens[offset:] for offset in range(size)))
    ]
    return tokens + phrases


def unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return tuple(ordered)


def extract_section_labels(text: str) -> dict[str, str]:
    """Best-effort labels from page Markdown or a chunk.

    Existing artifacts may not have labels, so these are deliberately optional
    hints rather than required schema fields.
    """

    body = strip_front_matter(text)
    raw_lines = body.splitlines()
    lines = [line.strip(" #\t") for line in raw_lines]
    labels: dict[str, str] = {}

    for index, (raw_line, line) in enumerate(zip(raw_lines, lines, strict=False)):
        if not line or len(line) > 120:
            continue
        is_markdown_heading = raw_line.lstrip().startswith("#")
        compact = re.sub(r"\s+", " ", line).strip(":- ")
        lower = compact.lower()
        if re.match(r"^(part|chapter)\s+([0-9ivxlcdm]+)\b", lower):
            title = compact
            for candidate in lines[index + 1 : index + 4]:
                candidate = re.sub(r"\s+", " ", candidate).strip(":- ")
                if candidate and len(candidate) <= 100:
                    title = f"{compact}: {candidate}"
                    break
            labels["chapter_title"] = title
            continue
        if re.match(r"^\d+(\.\d+){0,3}\s+[A-Z][A-Za-z]", compact) and not re.match(
            r"^\d+\s*$", compact
        ):
            labels.setdefault("section_title", compact)
            continue
        if is_markdown_heading and not lower.startswith(("figure", "table")):
            labels.setdefault("section_title", compact)

    return labels


def build_chunks(markdown_dir: Path, chunks_path: Path, *, force: bool = False) -> int:
    if chunks_path.exists() and not force:
        return sum(1 for _ in chunks_path.open("r", encoding="utf-8"))

    ensure_parent(chunks_path)
    page_paths = sorted(markdown_dir.glob("page_*.md"))
    count = 0
    current_chapter_title = ""
    current_section_title = ""
    with chunks_path.open("w", encoding="utf-8") as handle:
        for page_path in page_paths:
            match = re.search(r"page_(\d+)\.md$", page_path.name)
            page = int(match.group(1)) if match else 0
            text = page_path.read_text(encoding="utf-8", errors="replace")
            page_labels = extract_section_labels(text)
            if page_labels.get("chapter_title"):
                current_chapter_title = page_labels["chapter_title"]
                current_section_title = ""
            if page_labels.get("section_title"):
                current_section_title = page_labels["section_title"]
            for chunk_index, chunk in enumerate(chunk_markdown_text(text)):
                principle_tags, stage_tags = infer_tags(chunk, page)
                concept_tags = infer_concept_tags(chunk)
                chunk_labels = extract_section_labels(chunk)
                chapter_title = chunk_labels.get("chapter_title", current_chapter_title)
                section_title = chunk_labels.get("section_title", current_section_title)
                if chunk_labels.get("chapter_title"):
                    current_chapter_title = chunk_labels["chapter_title"]
                    current_section_title = ""
                if chunk_labels.get("section_title"):
                    current_section_title = chunk_labels["section_title"]
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
                    "concept_tags": list(concept_tags),
                    "chapter_title": chapter_title,
                    "section_title": section_title,
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


def is_reference_like_chunk(text: str) -> bool:
    """Detect citation-list chunks that are usually weaker agent context."""

    compact = " ".join(text.split()).lower()
    if not compact:
        return False
    if compact.startswith(("references", "bibliography")):
        return True

    citation_markers = len(re.findall(r"\[\d+\]", text))
    line_citations = len(re.findall(r"(?m)^\s*\[\d+\]", text))
    reference_terms = sum(
        term in compact
        for term in (
            "doi:",
            "proceedings",
            "conference",
            "symposium",
            "ieee",
            "acm",
            "arxiv",
            "volume",
        )
    )
    if line_citations >= 2 or citation_markers >= 4:
        return True
    if citation_markers >= 1 and reference_terms >= 2:
        return True
    return bool(
        reference_terms >= 3
        and re.match(r"^(and\s+)?[a-z ,.-]+,\s+(volume|number|pages|march|january)", compact)
    )


def is_table_of_contents_like_chunk(text: str) -> bool:
    compact = " ".join(text.split()).lower()
    if not compact:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_lines = " ".join(lines[:4]).lower()
    if first_lines.startswith(("contents", "table of contents")):
        return True
    dotted_lines = sum(bool(re.search(r"\.{2,}\s*\d{1,4}$", line)) for line in lines[:40])
    right_aligned_page_lines = sum(
        bool(re.search(r"\S.{8,}\s{2,}\d{1,4}$", line)) for line in lines[:50]
    )
    chapter_lines = sum(
        bool(re.search(r"\b(chapter|part)\s+[0-9ivxlcdm]+\b", line.lower())) for line in lines[:40]
    )
    numbered_section_lines = sum(
        bool(re.search(r"^\d+(\.\d+)*\s+.+\s+\d{1,4}$", line)) for line in lines[:40]
    )
    return (
        dotted_lines >= 3
        or right_aligned_page_lines >= 6
        or chapter_lines >= 4
        or numbered_section_lines >= 5
    )


def is_front_matter_like_chunk(text: str, page: int = 0) -> bool:
    compact = " ".join(text.split()).lower()
    if not compact:
        return False
    front_terms = (
        "copyright",
        "isbn",
        "oreilly media",
        "all rights reserved",
        "preface",
        "foreword",
        "acknowledgments",
        "about the author",
    )
    if compact.startswith(front_terms):
        return True
    if page and page > 30:
        return False
    return sum(term in compact for term in front_terms) >= 2


def is_chapter_opener_like_chunk(text: str) -> bool:
    body = strip_front_matter(text).strip()
    if not body:
        return False
    compact = " ".join(body.split())
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", compact)
    if len(words) > 90:
        return False
    starts_with_chapter = re.match(r"^(part|chapter)\s+([0-9ivxlcdm]+)\b", compact.lower())
    has_sparse_quote = compact.count('"') >= 2 or compact.count("“") + compact.count("”") >= 2
    short_heading_page = len(words) <= 45 and bool(starts_with_chapter)
    return short_heading_page or (len(words) <= 55 and has_sparse_quote and "\n\n" not in body)


def is_index_like_chunk(text: str) -> bool:
    compact = " ".join(text.split()).lower()
    if not compact:
        return False
    if compact.startswith("index"):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    index_lines = sum(
        bool(re.search(r"[a-z][a-z -]+,\s*\d{1,4}([,-]\s*\d{1,4}){1,}", line.lower()))
        for line in lines[:40]
    )
    return index_lines >= max(4, len(lines[:40]) // 3)


def classify_chunk_noise(text: str, page: int = 0) -> tuple[str, ...]:
    flags: list[str] = []
    if is_reference_like_chunk(text):
        flags.append("reference_like")
    if is_table_of_contents_like_chunk(text):
        flags.append("table_of_contents")
    if is_front_matter_like_chunk(text, page):
        flags.append("front_matter")
    if is_chapter_opener_like_chunk(text):
        flags.append("chapter_opener")
    if is_index_like_chunk(text):
        flags.append("index_like")
    return tuple(flags)


def chunk_noise_penalty(noise_flags: Iterable[str]) -> float:
    return sum(NOISE_CHUNK_PENALTIES.get(flag, 0.0) for flag in noise_flags)


def adjusted_retrieval_score(raw_score: float, text: str, *, page: int = 0) -> float:
    return float(raw_score) - chunk_noise_penalty(classify_chunk_noise(text, page))


def open_zvec_read_only_with_retry(
    zvec: Any,
    vector_path: Path,
    *,
    attempts: int = ZVEC_OPEN_RETRY_ATTEMPTS,
    initial_delay_seconds: float = ZVEC_OPEN_RETRY_DELAY_SECONDS,
    sleep: Any = time.sleep,
) -> Any:
    option = zvec.CollectionOption(read_only=True)
    for attempt in range(max(1, attempts)):
        try:
            return zvec.open(str(vector_path), option)
        except RuntimeError as exc:
            is_lock_error = "lock" in str(exc).lower()
            is_last_attempt = attempt >= max(1, attempts) - 1
            if not is_lock_error or is_last_attempt:
                raise
            sleep(initial_delay_seconds * (attempt + 1))
    raise RuntimeError(f"failed to open zvec collection at {vector_path}")


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
            zvec.FieldSchema("concept_tags", zvec.DataType.ARRAY_STRING),
            zvec.FieldSchema("chapter_title", zvec.DataType.STRING),
            zvec.FieldSchema("section_title", zvec.DataType.STRING),
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
                " ".join(record.get("concept_tags", [])),
                record.get("chapter_title", ""),
                record.get("section_title", ""),
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
                    "concept_tags": list(record.get("concept_tags", [])),
                    "chapter_title": record.get("chapter_title", ""),
                    "section_title": record.get("section_title", ""),
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


def infer_principle_tags_from_query(text: str) -> tuple[str, ...]:
    lower = text.lower()
    return tuple(
        sorted(
            tag
            for tag, keywords in PRINCIPLE_KEYWORDS.items()
            if any(_contains_phrase(lower, keyword) for keyword in keywords)
        )
    )


def build_query_profile(task: str, stage: str, next_steps: str) -> dict[str, Any]:
    stage = normalize_stage(stage)
    user_text = " ".join([task, next_steps])
    stage_text = " ".join([stage, *STAGE_LENSES[stage]])
    user_terms = unique_in_order(tokenize_lexical_terms(user_text))
    stage_terms = tuple(
        term
        for term in unique_in_order(tokenize_lexical_terms(stage_text))
        if term not in user_terms
    )
    user_concept_tags = tuple(sorted(set(infer_concept_tags(user_text))))
    stage_concept_tags = tuple(sorted(set(infer_concept_tags(stage_text))))
    user_principle_tags = infer_principle_tags_from_query(user_text)
    stage_principle_tags = infer_principle_tags_from_query(stage_text)
    return {
        "text": " ".join([user_text, stage_text]),
        "user_text": user_text,
        "stage_text": stage_text,
        "terms": (*user_terms, *stage_terms),
        "user_terms": user_terms,
        "stage_terms": stage_terms,
        "concept_tags": tuple(sorted(set(user_concept_tags) | set(stage_concept_tags))),
        "principle_tags": tuple(sorted(set(user_principle_tags) | set(stage_principle_tags))),
        "user_concept_tags": user_concept_tags,
        "stage_concept_tags": stage_concept_tags,
        "user_principle_tags": user_principle_tags,
        "stage_principle_tags": stage_principle_tags,
    }


def lexical_match_score(query_terms: Iterable[str], text: str) -> tuple[float, tuple[str, ...]]:
    text_lower = text.lower()
    text_terms = set(tokenize_lexical_terms(text))
    matched: list[str] = []
    score = 0.0
    for term in query_terms:
        if " " in term:
            if _contains_phrase(text_lower, term):
                matched.append(term)
                score += 0.035
        elif term in text_terms:
            matched.append(term)
            score += 0.014
    return min(score, 0.28), tuple(matched[:14])


def weighted_lexical_match_score(
    query_profile: dict[str, Any], text: str
) -> tuple[float, tuple[str, ...]]:
    text_lower = text.lower()
    text_terms = set(tokenize_lexical_terms(text))
    matched: list[str] = []
    score = 0.0

    def add_matches(terms: Iterable[str], *, phrase_weight: float, token_weight: float) -> None:
        nonlocal score
        for term in terms:
            if term in matched:
                continue
            if " " in term:
                if _contains_phrase(text_lower, term):
                    matched.append(term)
                    score += phrase_weight
            elif term in text_terms:
                matched.append(term)
                score += token_weight

    add_matches(query_profile["user_terms"], phrase_weight=0.055, token_weight=0.020)
    add_matches(query_profile["stage_terms"], phrase_weight=0.014, token_weight=0.005)
    return min(score, 0.42), tuple(matched[:16])


def query_tag_boost(
    query_profile: dict[str, Any],
    record_tags: Iterable[str],
) -> tuple[float, tuple[str, ...]]:
    record_tag_set = set(record_tags)
    user_tags = set(query_profile["user_concept_tags"]) | set(query_profile["user_principle_tags"])
    stage_tags = set(query_profile["stage_concept_tags"]) | set(
        query_profile["stage_principle_tags"]
    )
    matched_user = tuple(sorted(user_tags & record_tag_set))
    matched_stage = tuple(sorted((stage_tags & record_tag_set) - set(matched_user)))
    boost = min(0.055 * len(matched_user), 0.28) + min(0.015 * len(matched_stage), 0.06)
    return boost, (*matched_user, *matched_stage)


def query_affinity_boost(
    query_profile: dict[str, Any],
    *,
    text: str,
    chapter_title: str,
    section_title: str,
) -> tuple[float, tuple[str, ...]]:
    """Small chapter/section nudges for task-specific DDIA neighborhoods."""

    user_tags = set(query_profile["user_concept_tags"]) | set(query_profile["user_principle_tags"])
    user_text = str(query_profile["user_text"]).lower()
    location = f"{chapter_title} {section_title}".lower()
    lower_text = text.lower()
    boost = 0.0
    reasons: list[str] = []

    schema_heavy = "schema" in user_tags and any(
        term in user_text for term in ("migration", "migrate", "compatibility", "encoding")
    )
    tenant_heavy = "tenant" in user_text
    crash_heavy = any(
        term in user_text
        for term in (
            "crash",
            "dies",
            "restart",
            "stale",
            "pid",
            "supervisor",
            "mid-run",
            "process dies",
        )
    )
    drift_heavy = any(
        term in user_text
        for term in (
            "drift",
            "fingerprint",
            "batchplan",
            "checkpoint",
            "deterministic",
            "reproducible",
        )
    )
    experiment_lineage_heavy = any(
        term in user_text
        for term in (
            "experiment",
            "parent",
            "selected",
            "selection",
            "policy",
            "research round",
            "random seed",
        )
    )

    if schema_heavy and "encoding and evolution" in location:
        boost += 0.20
        reasons.append("affinity:encoding-evolution")

    if drift_heavy and ("batch processing" in location or "stream processing" in location):
        boost += 0.18
        reasons.append("affinity:deterministic-replay")
        if any(term in lower_text for term in ("recompute", "recomputation", "snapshot")):
            boost += 0.04
            reasons.append("affinity:rebuild-state")

    if crash_heavy and (
        "replication" in location
        or "stream processing" in location
        or any(
            term in lower_text
            for term in (
                "node outage",
                "outage",
                "crash",
                "failed batch",
                "retrying",
                "cascading failure",
            )
        )
    ):
        boost += 0.16
        reasons.append("affinity:failure-recovery")
        if any(term in lower_text for term in ("node outage", "failed batch", "snapshot")):
            boost += 0.04
            reasons.append("affinity:recoverable-state")

    if experiment_lineage_heavy and (
        "stream processing" in location
        or any(term in lower_text for term in ("event sourcing", "causality"))
    ):
        boost += 0.12
        reasons.append("affinity:lineage-log")
        if "event sourcing" in lower_text:
            boost += 0.04
            reasons.append("affinity:event-history")

    if tenant_heavy and (
        "transactions" in location or "isolation" in lower_text or "serializable" in lower_text
    ):
        boost += 0.12
        reasons.append("affinity:tenant-isolation")

    return min(boost, 0.26), tuple(reasons)


def explain_hit(
    *,
    vector_score: float,
    lexical_score: float,
    query_boost: float,
    noise_penalty: float,
    matched_terms: tuple[str, ...],
    matched_tags: tuple[str, ...],
    noise_flags: tuple[str, ...],
    chapter_title: str,
    section_title: str,
) -> dict[str, Any]:
    parts: list[str] = [f"vector score {vector_score:.4f}"]
    if lexical_score:
        parts.append(f"lexical overlap +{lexical_score:.4f}")
    if query_boost:
        parts.append(f"query boost +{query_boost:.4f}")
    if noise_penalty:
        parts.append(f"noise penalty -{noise_penalty:.4f}")
    if matched_terms:
        parts.append("matched terms: " + ", ".join(matched_terms[:8]))
    if matched_tags:
        parts.append("matched tags: " + ", ".join(matched_tags))
    if chapter_title or section_title:
        location = " / ".join(part for part in (chapter_title, section_title) if part)
        parts.append(f"section: {location}")
    if noise_flags:
        parts.append("downranked as " + ", ".join(noise_flags))

    return {
        "summary": "; ".join(parts),
        "matched_terms": list(matched_terms),
        "matched_tags": list(matched_tags),
        "noise_flags": list(noise_flags),
        "scores": {
            "vector": vector_score,
            "lexical": lexical_score,
            "query_boost": query_boost,
            "noise_penalty": noise_penalty,
        },
    }


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
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    chunks = load_chunks(chunks_path)
    query = build_context_query(task, stage, next_steps)
    query_profile = build_query_profile(task, stage, next_steps)
    collection = open_zvec_read_only_with_retry(zvec, vector_path)
    candidate_top_k = min(len(chunks), max(top_k, top_k * 12, top_k + 32, 60))
    results = collection.query(
        zvec.VectorQuery("embedding", vector=embed_hash(query, dim=dim)),
        topk=candidate_top_k,
    )

    vector_scores: dict[str, float] = {}
    for result in results:
        chunk_id = result["id"] if isinstance(result, dict) else result.id
        vector_scores[chunk_id] = float(
            result.get("score", 0.0) if isinstance(result, dict) else result.score
        )

    lexical_candidates: list[tuple[float, str]] = []
    for chunk_id, record in chunks.items():
        text = record["text"]
        page = int(record["page"])
        principle_tags = tuple(record.get("principle_tags", []))
        stage_tags = tuple(record.get("stage_tags", []))
        concept_tags = unique_in_order(
            (*tuple(record.get("concept_tags", [])), *infer_concept_tags(text))
        )
        lexical_score, _matched_terms = weighted_lexical_match_score(query_profile, text)
        tag_boost, _matched_tags = query_tag_boost(
            query_profile, (*principle_tags, *stage_tags, *concept_tags)
        )
        affinity_boost, _affinity_tags = query_affinity_boost(
            query_profile,
            text=text,
            chapter_title=str(record.get("chapter_title", "") or ""),
            section_title=str(record.get("section_title", "") or ""),
        )
        noise_penalty = chunk_noise_penalty(classify_chunk_noise(text, page))
        lexical_candidate_score = lexical_score + tag_boost + affinity_boost - noise_penalty
        if lexical_candidate_score > 0:
            lexical_candidates.append((lexical_candidate_score, chunk_id))

    lexical_candidate_top_k = min(len(chunks), max(top_k * 30, top_k + 60, 100))
    for _score, chunk_id in sorted(lexical_candidates, reverse=True)[:lexical_candidate_top_k]:
        vector_scores.setdefault(chunk_id, 0.0)

    hits: list[SearchHit] = []
    for chunk_id, raw_score in vector_scores.items():
        record = chunks.get(chunk_id)
        if record is None:
            continue
        text = record["text"]
        page = int(record["page"])
        principle_tags = tuple(record.get("principle_tags", []))
        stage_tags = tuple(record.get("stage_tags", []))
        concept_tags = unique_in_order(
            (*tuple(record.get("concept_tags", [])), *infer_concept_tags(text))
        )
        all_tags = (*principle_tags, *stage_tags, *concept_tags)
        lexical_score, matched_terms = weighted_lexical_match_score(query_profile, text)
        tag_boost, matched_tags = query_tag_boost(query_profile, all_tags)
        chapter_title = str(record.get("chapter_title", "") or "")
        section_title = str(record.get("section_title", "") or "")
        affinity_boost, affinity_tags = query_affinity_boost(
            query_profile,
            text=text,
            chapter_title=chapter_title,
            section_title=section_title,
        )
        query_boost = tag_boost + affinity_boost
        matched_tags = (*matched_tags, *affinity_tags)
        noise_flags = classify_chunk_noise(text, page)
        noise_penalty = chunk_noise_penalty(noise_flags)
        score = raw_score + lexical_score + query_boost - noise_penalty
        why_this_hit = explain_hit(
            vector_score=raw_score,
            lexical_score=lexical_score,
            query_boost=query_boost,
            noise_penalty=noise_penalty,
            matched_terms=matched_terms,
            matched_tags=matched_tags,
            noise_flags=noise_flags,
            chapter_title=chapter_title,
            section_title=section_title,
        )
        hits.append(
            SearchHit(
                chunk_id=record["id"],
                score=score,
                page=page,
                source_path=record["source_path"],
                principle_tags=principle_tags,
                stage_tags=stage_tags,
                text=text,
                concept_tags=concept_tags,
                chapter_title=chapter_title,
                section_title=section_title,
                vector_score=raw_score,
                lexical_score=lexical_score,
                query_boost=query_boost,
                noise_penalty=noise_penalty,
                matched_terms=matched_terms,
                matched_tags=matched_tags,
                noise_flags=noise_flags,
                why_this_hit=why_this_hit,
            )
        )
    hits.sort(key=lambda hit: (-hit.score, hit.page, hit.chunk_id))
    clean_hits = [hit for hit in hits if not (set(hit.noise_flags) & NOISE_FILTER_FLAGS)]
    if len(clean_hits) >= top_k:
        hits = clean_hits
    return hits[:top_k]


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
            f"tags={','.join(hit.principle_tags or hit.concept_tags or hit.stage_tags or ('untagged',))}; "
            f"why={hit.why_this_hit.get('summary', '') if hit.why_this_hit else 'not explained'}"
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
                f"- Vector score: {hit.vector_score:.4f}",
                f"- Lexical score: {hit.lexical_score:.4f}",
                f"- Query boost: {hit.query_boost:.4f}",
                f"- Noise penalty: {hit.noise_penalty:.4f}",
                f"- Source path: {hit.source_path}",
                f"- Location: {' / '.join(part for part in (hit.chapter_title, hit.section_title) if part) or 'not labeled'}",
                f"- Principle tags: {', '.join(hit.principle_tags) or 'none'}",
                f"- Concept tags: {', '.join(hit.concept_tags) or 'none'}",
                f"- Why this hit: {hit.why_this_hit.get('summary', 'not explained') if hit.why_this_hit else 'not explained'}",
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
                    "concept_tags": list(hit.concept_tags),
                    "chapter_title": hit.chapter_title,
                    "section_title": hit.section_title,
                    "vector_score": hit.vector_score,
                    "lexical_score": hit.lexical_score,
                    "query_boost": hit.query_boost,
                    "noise_penalty": hit.noise_penalty,
                    "matched_terms": list(hit.matched_terms),
                    "matched_tags": list(hit.matched_tags),
                    "noise_flags": list(hit.noise_flags),
                    "why_this_hit": hit.why_this_hit or {},
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
