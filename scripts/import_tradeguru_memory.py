#!/usr/bin/env python3
"""Import TradeGuru electrical documents into a Lazarus checkpoint store.

This importer is intentionally checkpoint-backed: each source document becomes
one deterministic Lazarus memory session under:

    <store-root>/<session_id>/torch_store/

Windows are captured through LiveIndexer, so the resulting stores include
boundary residuals, residual streams, token lists, keywords, and metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_STORE_ROOT = "/tmp/lazarus-tradeguru-fault-memory"
DEFAULT_EXTENSIONS = (".txt", ".md", ".json")
DEFAULT_WINDOW_SIZE = 512
DEFAULT_OVERLAP_TOKENS = 64

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
_TXT_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 /&:.,'()\-]{8,120}$")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_\-]{2,}", re.IGNORECASE)


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    source_root: Path
    relative_path: str
    folder: str
    category: str
    doc_title: str
    text: str


@dataclass(frozen=True)
class Section:
    title: str
    text: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a dedicated TradeGuru fault-finding Lazarus memory store "
            "from .txt/.md/readable .json documents."
        )
    )
    parser.add_argument(
        "--store-root",
        default=DEFAULT_STORE_ROOT,
        help="Output checkpoint root. Default: %(default)s",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source directory or file. Repeat for electricalhowto, electricalhowto - 2, etc.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Torch device for residual capture.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Tokenizer window size per memory chunk.",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=DEFAULT_OVERLAP_TOKENS,
        help="Token overlap between adjacent windows.",
    )
    parser.add_argument(
        "--extension",
        action="append",
        dest="extensions",
        help="Readable extension to import. Repeat to override defaults.",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Optional smoke-test cap on source documents.",
    )
    parser.add_argument(
        "--limit-windows-per-doc",
        type=int,
        default=None,
        help="Optional smoke-test cap on captured windows per document.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep an existing store root and skip already-complete document sessions.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing store root before import.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Pass local_files_only=True to transformers loaders.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to transformers loaders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report import plan without loading the model or writing stores.",
    )
    parser.add_argument(
        "--manifest-name",
        default="tradeguru_import_manifest.json",
        help="Root-level manifest filename.",
    )
    return parser


def resolve_host_path(raw_path: str) -> Path:
    """Resolve WSL paths while accepting Windows drive paths from the user."""
    raw_path = str(raw_path).strip().strip('"')
    match = _WINDOWS_DRIVE_RE.match(raw_path)
    if match and os.name != "nt":
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}").expanduser()
    return Path(raw_path).expanduser()


def _slug(text: str) -> str:
    parts = _WORD_RE.findall(text.lower())
    return "_".join(parts[:12]) or "document"


def category_for_path(path: Path, source_root: Path) -> str:
    folder = source_root.name.lower()
    rel = str(path.relative_to(source_root) if path != source_root else path.name).lower()
    joined = f"{folder}/{rel}"
    if "fault" in joined:
        return "fault_finding_guide"
    if "electricalhowto - 2" in joined or "electricalhowto-2" in joined:
        return "electrical_howto_markdown"
    if "electricalhowto" in joined:
        return "electrical_howto_text"
    if "noise" in joined or "distractor" in joined:
        return "noise"
    return _slug(folder)


def _read_text_with_fallback(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _json_to_lines(value: Any, *, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).strip()
            child_prefix = f"{prefix}.{key_text}" if prefix else key_text
            if isinstance(item, str):
                if item.strip():
                    lines.append(f"{child_prefix}: {item.strip()}")
            elif isinstance(item, (int, float, bool)) or item is None:
                lines.append(f"{child_prefix}: {item}")
            else:
                lines.extend(_json_to_lines(item, prefix=child_prefix))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            lines.extend(_json_to_lines(item, prefix=f"{prefix}[{idx}]"))
    elif isinstance(value, str) and value.strip():
        lines.append(f"{prefix}: {value.strip()}" if prefix else value.strip())
    elif prefix:
        lines.append(f"{prefix}: {value}")
    return lines


def readable_json_text(raw_text: str) -> tuple[str, str | None]:
    data = json.loads(raw_text)
    title: str | None = None
    if isinstance(data, dict):
        for key in ("title", "name", "doc_title", "heading"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                title = value.strip()
                break
    return "\n".join(_json_to_lines(data)), title


def normalize_text(text: str) -> str:
    text = _CONTROL_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    normalized: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank:
                normalized.append("")
            blank = True
            continue
        normalized.append(line)
        blank = False
    return "\n".join(normalized).strip()


def title_from_text(path: Path, text: str, explicit_title: str | None = None) -> str:
    if explicit_title:
        return explicit_title[:180]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading = _MARKDOWN_HEADING_RE.match(stripped)
        if heading:
            return heading.group(1).strip()[:180]
        if len(stripped) <= 180:
            return stripped
        break
    return path.stem.replace("_", " ").replace("-", " ")[:180]


def load_source_document(path: Path, source_root: Path) -> SourceDocument | None:
    try:
        raw_text = _read_text_with_fallback(path)
    except OSError as exc:
        print(f"[scan] skip unreadable {path}: {exc}", file=sys.stderr)
        return None

    explicit_title = None
    if path.suffix.lower() == ".json":
        try:
            raw_text, explicit_title = readable_json_text(raw_text)
        except json.JSONDecodeError as exc:
            print(f"[scan] skip invalid json {path}: {exc}", file=sys.stderr)
            return None

    text = normalize_text(raw_text)
    if not text:
        return None

    try:
        relative_path = path.relative_to(source_root).as_posix()
    except ValueError:
        relative_path = path.name

    return SourceDocument(
        path=path,
        source_root=source_root,
        relative_path=relative_path,
        folder=source_root.name,
        category=category_for_path(path, source_root),
        doc_title=title_from_text(path, text, explicit_title),
        text=text,
    )


def iter_source_paths(sources: list[Path], extensions: set[str]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"source does not exist: {source}")
        if source.is_file():
            if source.suffix.lower() in extensions:
                pairs.append((source, source.parent))
            continue
        for child in sorted(source.rglob("*")):
            if child.is_file() and child.suffix.lower() in extensions:
                pairs.append((child, source))
    pairs.sort(key=lambda item: (item[1].as_posix().lower(), item[0].as_posix().lower()))
    return pairs


def split_sections(doc: SourceDocument) -> list[Section]:
    sections: list[Section] = []
    current_title = doc.doc_title
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(Section(title=current_title, text=text))
        current_lines = []

    for line in doc.text.splitlines():
        stripped = line.strip()
        heading_match = _MARKDOWN_HEADING_RE.match(stripped)
        is_txt_heading = bool(_TXT_HEADING_RE.match(stripped)) and len(stripped.split()) <= 14
        if (heading_match or is_txt_heading) and current_lines:
            flush()
            current_title = (
                heading_match.group(1).strip() if heading_match else stripped.title()
            )[:180]
        current_lines.append(line)
    flush()
    return sections or [Section(title=doc.doc_title, text=doc.text)]


def encode_token_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(token_id) for token_id in token_ids]


def iter_token_windows(
    token_ids: list[int],
    *,
    window_size: int,
    overlap_tokens: int,
) -> list[tuple[int, int, list[int]]]:
    if window_size <= 0:
        raise ValueError("--window-size must be positive")
    if overlap_tokens < 0:
        raise ValueError("--overlap-tokens must be >= 0")
    if overlap_tokens >= window_size:
        raise ValueError("--overlap-tokens must be smaller than --window-size")
    if not token_ids:
        return []

    stride = window_size - overlap_tokens
    windows: list[tuple[int, int, list[int]]] = []
    start = 0
    while start < len(token_ids):
        end = min(start + window_size, len(token_ids))
        windows.append((start, end, token_ids[start:end]))
        if end >= len(token_ids):
            break
        start += stride
    return windows


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def _extra_keyword_candidates(*texts: str) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for text in texts:
        for word in _WORD_RE.findall(text.lower()):
            if word in seen:
                continue
            seen.add(word)
            keywords.append(word)
            if len(keywords) >= 8:
                return keywords
    return keywords


def session_id_for_document(doc: SourceDocument) -> str:
    digest_source = f"{doc.folder}/{doc.relative_path}".replace("\\", "/")
    return hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:32]


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _text_config(model_config: Any) -> Any:
    nested = _cfg_get(model_config, "text_config", None)
    return nested if nested is not None else model_config


def architecture_config_dict(model: Any, *, window_size: int) -> dict[str, Any]:
    from chuk_lazarus.inference.context.knowledge.config import ArchitectureConfig

    text_config = _text_config(model.config)
    base = ArchitectureConfig.from_model_config(model.config)
    hidden_dim = int(
        _cfg_get(text_config, "hidden_size", None)
        or _cfg_get(text_config, "hidden_dim", None)
        or _cfg_get(model.config, "hidden_size", 0)
    )
    num_heads = int(
        _cfg_get(text_config, "num_attention_heads", None)
        or _cfg_get(text_config, "n_head", None)
        or 0
    )
    head_dim = int(_cfg_get(text_config, "head_dim", 0) or 0)
    if head_dim <= 0 and hidden_dim > 0 and num_heads > 0:
        head_dim = hidden_dim // num_heads

    num_kv_heads = int(_cfg_get(text_config, "num_key_value_heads", num_heads) or num_heads or 1)
    groups_per_kv = max(1, num_heads // max(1, num_kv_heads)) if num_heads else 1
    kv_head = int(base.query_head // groups_per_kv)

    config = ArchitectureConfig(
        retrieval_layer=base.retrieval_layer,
        query_head=base.query_head,
        injection_layer=base.injection_layer,
        kv_head=kv_head,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        k_dim=head_dim,
        threshold_multiplier=base.threshold_multiplier,
        inject_coefficient=base.inject_coefficient,
        entries_per_window=base.entries_per_window,
        crystal_layer=base.crystal_layer,
        window_size=int(window_size),
    )
    return config.to_dict()


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    loader_kwargs = {
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **loader_kwargs)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **loader_kwargs,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _is_complete_session(session_dir: Path) -> bool:
    manifest = session_dir / "torch_store" / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("in_flight") is not True and int(data.get("num_entries", 0)) >= 1


def import_document(
    doc: SourceDocument,
    *,
    doc_index: int,
    store_root: Path,
    model: Any,
    tokenizer: Any,
    arch_config: dict[str, Any],
    window_size: int,
    overlap_tokens: int,
    limit_windows_per_doc: int | None,
    import_run_id: str,
    force_session: bool,
) -> dict[str, Any]:
    from chuk_lazarus.inference.context.knowledge.route import extract_window_keywords
    from chuk_lazarus.session_store.live_indexer import LiveIndexer

    session_id = session_id_for_document(doc)
    session_dir = store_root / session_id
    torch_store_dir = session_dir / "torch_store"
    if force_session and session_dir.exists():
        shutil.rmtree(session_dir)
    if _is_complete_session(session_dir):
        return {
            "session_id": session_id,
            "status": "skipped_complete",
            "source_path": doc.path.as_posix(),
            "relative_path": doc.relative_path,
            "folder": doc.folder,
            "category": doc.category,
            "doc_title": doc.doc_title,
            "num_windows": None,
        }

    if session_dir.exists():
        shutil.rmtree(session_dir)
    torch_store_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "input_records").mkdir(parents=True, exist_ok=True)

    indexer = LiveIndexer(
        model=model,
        tokenizer=tokenizer,
        checkpoint_dir=torch_store_dir,
        crystal_layer=int(arch_config["crystal_layer"]),
        window_size=window_size,
        arch_config=arch_config,
        forward_semaphore=threading.Semaphore(1),
    )

    window_records: list[dict[str, Any]] = []
    window_id = 0
    sections = split_sections(doc)
    try:
        for section_index, section in enumerate(sections):
            section_text = section.text
            token_ids = encode_token_ids(tokenizer, section_text)
            for token_start, token_end, window_tokens in iter_token_windows(
                token_ids,
                window_size=window_size,
                overlap_tokens=overlap_tokens,
            ):
                if limit_windows_per_doc is not None and window_id >= limit_windows_per_doc:
                    break
                decoded_window = _decode_token_ids(tokenizer, window_tokens)
                keywords = extract_window_keywords(window_tokens, tokenizer, max_keywords=8)
                for extra in _extra_keyword_candidates(
                    doc.doc_title,
                    section.title,
                    doc.category,
                    doc.relative_path,
                ):
                    if extra not in keywords:
                        keywords.append(extra)
                    if len(keywords) >= 14:
                        break

                metadata = {
                    "session_id": session_id,
                    "source_path": doc.path.as_posix(),
                    "source_relpath": doc.relative_path,
                    "folder": doc.folder,
                    "category": doc.category,
                    "doc_title": doc.doc_title,
                    "section_title": section.title,
                    "section_index": int(section_index),
                    "chunk_index": int(window_id),
                    "token_start": int(token_start),
                    "token_end": int(token_end),
                    "token_count": int(len(window_tokens)),
                    "clause_id": f"{doc_index}.{window_id}",
                    "clause_title": f"{doc.doc_title} / {section.title} / chunk {window_id}",
                    "part_index": int(window_id + 1),
                    "shard_id": 0,
                    "import_run_id": import_run_id,
                    "source_kind": "tradeguru_fault_eval",
                }
                indexer.enqueue(window_id, window_tokens, keywords, metadata)
                record = {
                    "window_id": int(window_id),
                    "text_preview": " ".join(decoded_window.split())[:500],
                    "keywords": keywords,
                    "metadata": metadata,
                }
                window_records.append(record)
                _write_json(
                    session_dir / "input_records" / f"window_{window_id:03d}.json",
                    record,
                )
                window_id += 1
            if limit_windows_per_doc is not None and window_id >= limit_windows_per_doc:
                break

        if window_id == 0:
            indexer.stop()
            shutil.rmtree(session_dir, ignore_errors=True)
            return {
                "session_id": session_id,
                "status": "skipped_empty",
                "source_path": doc.path.as_posix(),
                "relative_path": doc.relative_path,
                "folder": doc.folder,
                "category": doc.category,
                "doc_title": doc.doc_title,
                "num_windows": 0,
            }

        indexer.flush_and_close()
    except Exception:
        try:
            indexer.stop()
        finally:
            shutil.rmtree(session_dir, ignore_errors=True)
        raise

    payload = {
        "session_id": session_id,
        "status": "imported",
        "source_path": doc.path.as_posix(),
        "relative_path": doc.relative_path,
        "folder": doc.folder,
        "category": doc.category,
        "doc_title": doc.doc_title,
        "num_windows": int(window_id),
        "input_record_count": len(window_records),
    }
    _write_json(session_dir / "document_manifest.json", payload)
    return payload


def prepare_store_root(store_root: Path, *, force: bool, append: bool) -> None:
    if force and append:
        raise ValueError("--force and --append are mutually exclusive")
    if force and store_root.exists():
        shutil.rmtree(store_root)
    if store_root.exists() and any(store_root.iterdir()) and not append:
        raise FileExistsError(
            f"{store_root} already exists and is not empty; use --append or --force"
        )
    store_root.mkdir(parents=True, exist_ok=True)


def scan_documents(args: argparse.Namespace) -> list[SourceDocument]:
    extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (args.extensions or DEFAULT_EXTENSIONS)
    }
    sources = [resolve_host_path(source) for source in args.source]
    pairs = iter_source_paths(sources, extensions)
    if args.limit_docs is not None:
        pairs = pairs[: int(args.limit_docs)]
    docs: list[SourceDocument] = []
    for path, source_root in pairs:
        doc = load_source_document(path, source_root)
        if doc is not None:
            docs.append(doc)
    return docs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store_root = resolve_host_path(args.store_root)
    import_run_id = hashlib.sha256(f"{time.time_ns()}:{store_root}".encode()).hexdigest()[:16]

    docs = scan_documents(args)
    if not docs:
        print("[import] no readable documents found", file=sys.stderr)
        return 2

    by_category: dict[str, int] = {}
    for doc in docs:
        by_category[doc.category] = by_category.get(doc.category, 0) + 1
    print(f"[import] documents={len(docs)} categories={by_category}")

    if args.dry_run:
        for doc in docs[:20]:
            print(f"[dry-run] {doc.category} {doc.relative_path} :: {doc.doc_title}")
        if len(docs) > 20:
            print(f"[dry-run] ... {len(docs) - 20} more")
        return 0

    prepare_store_root(store_root, force=args.force, append=args.append)
    model, tokenizer, actual_device = load_model_and_tokenizer(args)
    arch_config = architecture_config_dict(model, window_size=int(args.window_size))
    print(
        "[import] model="
        f"{args.model} device={actual_device} crystal_layer={arch_config['crystal_layer']} "
        f"window_size={args.window_size} overlap={args.overlap_tokens}"
    )

    imported: list[dict[str, Any]] = []
    total_windows = 0
    for doc_index, doc in enumerate(docs, start=1):
        print(f"[import] {doc_index}/{len(docs)} {doc.relative_path}")
        result = import_document(
            doc,
            doc_index=doc_index,
            store_root=store_root,
            model=model,
            tokenizer=tokenizer,
            arch_config=arch_config,
            window_size=int(args.window_size),
            overlap_tokens=int(args.overlap_tokens),
            limit_windows_per_doc=args.limit_windows_per_doc,
            import_run_id=import_run_id,
            force_session=bool(args.force),
        )
        imported.append(result)
        if isinstance(result.get("num_windows"), int):
            total_windows += int(result["num_windows"])

    manifest = {
        "schema_version": 1,
        "kind": "tradeguru_fault_memory_import",
        "import_run_id": import_run_id,
        "created_at": time.time(),
        "store_root": store_root.as_posix(),
        "model": args.model,
        "device": actual_device,
        "window_size": int(args.window_size),
        "overlap_tokens": int(args.overlap_tokens),
        "arch_config": arch_config,
        "source_count": len(args.source),
        "document_count": len(imported),
        "total_windows": int(total_windows),
        "categories": by_category,
        "documents": imported,
    }
    _write_json(store_root / args.manifest_name, manifest)
    print(f"[import] wrote {store_root / args.manifest_name}")
    print(f"[import] complete: documents={len(imported)} windows={total_windows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
