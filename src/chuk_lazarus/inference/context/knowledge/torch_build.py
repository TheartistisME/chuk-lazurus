"""Torch-native Apollo knowledge-store builder."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ArchitectureConfig
from .route import TFIDFRouter, extract_window_keywords
from .torch_capture import capture_window_boundaries
from .torch_store import TorchKnowledgeStore

STORE_VERSION = 12
MANIFEST_FILE = "manifest.json"
ENTRIES_FILE = "entries.npz"
WINDOW_TOKENS_FILE = "window_tokens.npz"
WINDOW_TOKEN_LISTS_FILE = "window_token_lists.npz"
IDF_FILE = "idf.json"
KEYWORDS_FILE = "keywords.json"
BOUNDARY_RESIDUAL_FILE = "boundary_residual.npy"
BOUNDARIES_DIR = "boundaries"
RESIDUAL_STREAMS_DIR = "residual_streams"

_ENTRY_DTYPE = np.dtype(
    [
        ("token_id", np.uint32),
        ("coefficient", np.float32),
        ("window_id", np.uint16),
        ("position_in_window", np.uint16),
        ("fact_id", np.uint16),
    ]
)


@dataclass(frozen=True)
class _EntryRecord:
    token_id: int
    coefficient: float
    window_id: int
    position_in_window: int
    fact_id: int


def _encode_token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(t) for t in token_ids]


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(list(token_ids), skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(list(token_ids))


def _chunk_tokens(token_ids: Sequence[int], window_size: int) -> list[list[int]]:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    windows: list[list[int]] = []
    for start in range(0, len(token_ids), window_size):
        windows.append(list(token_ids[start : start + window_size]))
    return windows


def _save_npz_mapping(
    path: Path,
    mapping: dict[int, Sequence[int]],
    *,
    dtype: Any,
    sort_values: bool,
) -> None:
    arrays = {}
    for key, value in mapping.items():
        values = sorted(value) if sort_values else list(value)
        arrays[str(key)] = np.asarray(values, dtype=dtype)
    np.savez(str(path), **arrays)


def _entries_to_numpy(entries: Sequence[_EntryRecord]) -> np.ndarray:
    arr = np.empty(len(entries), dtype=_ENTRY_DTYPE)
    for idx, entry in enumerate(entries):
        arr[idx] = (
            entry.token_id,
            entry.coefficient,
            entry.window_id,
            entry.position_in_window,
            entry.fact_id,
        )
    return arr


def _get_embedding_matrix(model: Any):
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        embeddings = getter()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight

    for attr in ("embed_tokens", "embedding", "embeddings"):
        candidate = getattr(model, attr, None)
        if candidate is not None and hasattr(candidate, "weight"):
            return candidate.weight

    nested = getattr(getattr(model, "model", None), "embed_tokens", None)
    if nested is not None and hasattr(nested, "weight"):
        return nested.weight

    return None


def _model_vocab_size(model: Any) -> int | None:
    embeddings = _get_embedding_matrix(model)
    if embeddings is not None and hasattr(embeddings, "shape") and len(embeddings.shape) >= 1:
        return int(embeddings.shape[0])

    for candidate in (
        getattr(model, "vocab_size", None),
        getattr(getattr(model, "config", None), "vocab_size", None),
    ):
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _filter_supported_token_ids(model: Any, token_ids: Sequence[int]) -> list[int]:
    vocab_size = _model_vocab_size(model)
    if vocab_size is None:
        return [int(token_id) for token_id in token_ids]
    return [int(token_id) for token_id in token_ids if 0 <= int(token_id) < vocab_size]


def _entry_coefficient(model: Any, token_id: int, inject_coefficient: float) -> float:
    embeddings = _get_embedding_matrix(model)
    if embeddings is None:
        return float(inject_coefficient)

    if token_id < 0 or token_id >= int(embeddings.shape[0]):
        return float(inject_coefficient)

    vector = embeddings[token_id].detach()
    natural_coeff = float(vector.norm().item())
    return float(inject_coefficient * natural_coeff)


def _model_device(model: Any):
    for container_name in ("parameters", "buffers"):
        container = getattr(model, container_name, None)
        if not callable(container):
            continue
        try:
            item = next(container())
        except (StopIteration, TypeError):
            continue
        return item.device
    return None


def _next_token_logits(model: Any, input_ids, attention_mask):
    outputs = None
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except TypeError:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = getattr(outputs, "logits", None)
    if logits is not None:
        return logits
    return None


def _generate_topic_expansion_ids(
    model: Any,
    tokenizer: Any,
    window_token_ids: Sequence[int],
    *,
    n_tokens: int = 5,
) -> list[int]:
    if n_tokens <= 0:
        return []

    try:
        import torch
    except Exception:  # pragma: no cover - exercised only on hosts without torch
        return []

    window_text = _decode_token_ids(tokenizer, window_token_ids)
    prompt = f"{window_text}\n\nTopics:"
    prompt_ids = _filter_supported_token_ids(
        model,
        _encode_token_ids(tokenizer, prompt, add_special_tokens=True),
    )
    if not prompt_ids:
        prompt_ids = _filter_supported_token_ids(
            model,
            _encode_token_ids(tokenizer, prompt, add_special_tokens=False),
        )
    if not prompt_ids:
        return []

    device = _model_device(model) or torch.device("cpu")
    input_ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    expansion_ids: list[int] = []
    model.eval()
    with torch.inference_mode():
        for _ in range(n_tokens):
            logits = _next_token_logits(model, input_ids, attention_mask)
            if logits is None or logits.ndim != 3:
                return expansion_ids

            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            expansion_ids.append(next_token)

            next_token_tensor = torch.as_tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat((input_ids, next_token_tensor), dim=1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token_tensor, dtype=torch.long)),
                dim=1,
            )

    return expansion_ids


def _apply_topic_expansion(
    *,
    tokenizer: Any,
    window_tokens: set[int],
    keywords: list[str],
    expansion_ids: Sequence[int],
) -> None:
    for token_id in expansion_ids:
        window_tokens.add(int(token_id))
        kw_text = _decode_token_ids(tokenizer, [int(token_id)]).strip()
        if not kw_text or len(kw_text) < 2:
            continue

        kw_lower = kw_text.lower()
        if kw_lower not in keywords:
            keywords.append(kw_lower)
        for variant in (kw_lower, f" {kw_lower}", kw_text, f" {kw_text}"):
            for variant_id in _encode_token_ids(tokenizer, variant, add_special_tokens=False):
                window_tokens.add(int(variant_id))


def _fill_entity_gaps(
    selected_positions: set[int],
    chunk_ids: Sequence[int],
    *,
    max_gap: int = 2,
) -> set[int]:
    filled = set(selected_positions)
    sorted_pos = sorted(selected_positions)

    for idx in range(len(sorted_pos) - 1):
        gap = sorted_pos[idx + 1] - sorted_pos[idx]
        if 1 < gap <= max_gap + 1:
            for pos in range(sorted_pos[idx] + 1, sorted_pos[idx + 1]):
                if 0 < pos < len(chunk_ids):
                    filled.add(pos)

    return filled


def _select_targets(
    chunk_ids: Sequence[int],
    idf: dict[int, float],
    *,
    min_k: int,
) -> list[tuple[int, int]]:
    if not chunk_ids:
        return []

    seen: set[int] = set()
    scored: list[tuple[float, int, int]] = []

    for pos in range(1, len(chunk_ids)):
        tid = int(chunk_ids[pos])
        if tid >= 236700:
            continue
        if tid in seen:
            continue
        seen.add(tid)
        token_idf = idf.get(tid, 0.0)
        if token_idf > 0:
            scored.append((token_idf, tid, pos))

    scored.sort(reverse=True)

    if scored:
        max_idf = scored[0][0]
        num_df1 = sum(1 for score, _, _ in scored if abs(score - max_idf) < 0.01)
        k = max(min_k, num_df1)
    else:
        k = min_k

    selected = scored[:k]
    selected_positions = {pos for _, _, pos in selected}
    filled = _fill_entity_gaps(selected_positions, chunk_ids, max_gap=2)

    result: list[tuple[int, int]] = []
    for pos in sorted(filled):
        result.append((int(chunk_ids[pos]), pos))
    return result


def build_knowledge_store_torch(
    model: Any,
    tokenizer: Any,
    raw_text: str,
    config: ArchitectureConfig,
    output_path: Path | str,
    *,
    max_tokens: int | None = None,
    progress_fn: Callable[[int, int], None] | None = None,
) -> TorchKnowledgeStore:
    """Build and persist a v12 Apollo knowledge store using torch only."""

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    token_ids = _encode_token_ids(tokenizer, raw_text, add_special_tokens=False)
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]

    windows = _chunk_tokens(token_ids, config.window_size)
    num_windows = len(windows)

    window_tokens: dict[int, set[int]] = {}
    window_token_lists: dict[int, list[int]] = {}
    keywords: dict[int, list[str]] = {}

    for wid, chunk_ids in enumerate(windows):
        window_tokens[wid] = set(int(t) for t in chunk_ids)
        window_token_lists[wid] = [int(t) for t in chunk_ids]
        keywords[wid] = extract_window_keywords(chunk_ids, tokenizer)

    idf = TFIDFRouter.compute_idf(window_tokens)

    entries: list[_EntryRecord] = []
    fact_id = 0
    for wid, chunk_ids in enumerate(windows):
        targets = _select_targets(chunk_ids, idf, min_k=config.entries_per_window)
        for token_id, pos_in_window in targets:
            entries.append(
                _EntryRecord(
                    token_id=token_id,
                    coefficient=_entry_coefficient(model, token_id, config.inject_coefficient),
                    window_id=wid,
                    position_in_window=pos_in_window,
                    fact_id=fact_id,
                )
            )
            fact_id += 1

        if progress_fn is not None:
            progress_fn(wid, num_windows)

    boundaries, final_boundary, residual_streams = capture_window_boundaries(
        model,
        windows,
        crystal_layer=config.crystal_layer,
        return_streams=True,
    )

    for wid, chunk_ids in enumerate(windows):
        expansion_ids = _generate_topic_expansion_ids(model, tokenizer, chunk_ids)
        _apply_topic_expansion(
            tokenizer=tokenizer,
            window_tokens=window_tokens[wid],
            keywords=keywords[wid],
            expansion_ids=expansion_ids,
        )

    if window_tokens:
        idf = TFIDFRouter.compute_idf(window_tokens)

    if entries:
        np.savez(str(output_path / ENTRIES_FILE), entries=_entries_to_numpy(entries))
    else:
        np.savez(str(output_path / ENTRIES_FILE), entries=np.array([], dtype=_ENTRY_DTYPE))

    _save_npz_mapping(
        output_path / WINDOW_TOKENS_FILE,
        window_tokens,
        dtype=np.uint32,
        sort_values=True,
    )
    _save_npz_mapping(
        output_path / WINDOW_TOKEN_LISTS_FILE,
        window_token_lists,
        dtype=np.uint32,
        sort_values=False,
    )

    (output_path / IDF_FILE).write_text(json.dumps({str(k): v for k, v in idf.items()}, indent=1) + "\n")
    (output_path / KEYWORDS_FILE).write_text(
        json.dumps({str(k): v for k, v in keywords.items()}, indent=1) + "\n"
    )

    if boundaries:
        boundary_dir = output_path / BOUNDARIES_DIR
        boundary_dir.mkdir(exist_ok=True)
        for wid, boundary in boundaries.items():
            boundary_np = np.asarray(boundary, dtype=np.float32).reshape(-1)
            np.save(str(boundary_dir / f"window_{wid:03d}.npy"), boundary_np)

    if residual_streams:
        stream_dir = output_path / RESIDUAL_STREAMS_DIR
        stream_dir.mkdir(exist_ok=True)
        for wid, stream in residual_streams.items():
            stream_np = np.asarray(stream, dtype=np.float32)
            np.save(str(stream_dir / f"window_{wid:03d}.npy"), stream_np)

    if final_boundary is not None:
        final_np = np.asarray(final_boundary, dtype=np.float32).reshape(1, 1, -1)
        np.save(str(output_path / BOUNDARY_RESIDUAL_FILE), final_np)

    manifest = {
        "version": STORE_VERSION,
        "num_entries": len(entries),
        "num_windows": num_windows,
        "num_tokens": len(token_ids),
        "entries_per_window": config.entries_per_window,
        "crystal_layer": config.crystal_layer,
        "window_size": config.window_size,
        "arch_config": config.to_dict(),
        "has_residuals": False,
        "has_residual_streams": bool(residual_streams),
        "residual_stream_count": len(residual_streams),
    }
    (output_path / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")

    return TorchKnowledgeStore.load(output_path)


__all__ = [
    "build_knowledge_store_torch",
    "_select_targets",
    "_fill_entity_gaps",
]
