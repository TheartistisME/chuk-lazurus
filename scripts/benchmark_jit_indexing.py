#!/usr/bin/env python3
"""Benchmark-local high-throughput JIT window indexing helpers.

The benchmark fast path intentionally bypasses ``LiveIndexer``. Benchmark
documents are already complete before inference, so the fastest route is to
tokenize the whole document, create overlapping token windows with
``torch.Tensor.unfold``, run those windows through one batched Layer-12 hook,
and write only the benchmark artifacts needed by the query path:

* ``activation_routes.npy``: memmapped ``[num_windows, 128]`` route matrix.
* ``window_tokens.npz``: window token ids, keyed by window id.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LAYER = 12
DEFAULT_DIM = 128
DEFAULT_OVERLAP_TOKENS = 128
DEFAULT_CUDA_BATCH_SIZE = 128
DEFAULT_CPU_BATCH_SIZE = 4


@dataclass(frozen=True)
class JitIndexResult:
    window_count: int
    activation_matrix_path: str
    activation_route_dir: str
    metadata_path: str
    layer: int
    mode: str
    token_count: int = 0
    tokens_per_second: float = 0.0
    memory_usage_percent: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_count": int(self.window_count),
            "activation_matrix_path": self.activation_matrix_path,
            "activation_route_dir": self.activation_route_dir,
            "metadata_path": self.metadata_path,
            "layer": int(self.layer),
            "mode": self.mode,
            "token_count": int(self.token_count),
            "tokens_per_second": float(self.tokens_per_second),
            "memory_usage_percent": float(self.memory_usage_percent),
        }


@dataclass(frozen=True)
class FastCaseIndexResult:
    case_index: int
    token_count: int
    window_count: int
    activation_matrix_path: str
    window_tokens_path: str
    store_dir: str
    tokens_per_second: float
    memory_usage_percent: float
    batch_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_index": int(self.case_index),
            "token_count": int(self.token_count),
            "window_count": int(self.window_count),
            "activation_matrix_path": self.activation_matrix_path,
            "window_tokens_path": self.window_tokens_path,
            "store_dir": self.store_dir,
            "tokens_per_second": float(self.tokens_per_second),
            "memory_usage_percent": float(self.memory_usage_percent),
            "batch_size": int(self.batch_size),
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def release_cuda_memory() -> None:
    """Best-effort hard fence between indexing and query/inference phases."""

    try:
        import gc

        gc.collect()
    except Exception:  # noqa: BLE001 - cleanup must never fail benchmark flow.
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:  # noqa: BLE001 - unavailable on some CUDA builds.
                pass
    except Exception:  # noqa: BLE001 - cleanup is best effort.
        pass


def _model_hidden_dim(model: Any) -> int:
    for source in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        value = getattr(source, "hidden_size", None)
        if value:
            return int(value)
    try:
        return int(model.get_input_embeddings().embedding_dim)
    except Exception:  # noqa: BLE001
        return 0


def _head_dim(model: Any, hidden_dim: int) -> int:
    config = getattr(model, "config", None)
    for source in (config, getattr(config, "text_config", None)):
        value = getattr(source, "head_dim", None)
        if value:
            return int(value)
        heads = getattr(source, "num_attention_heads", None)
        if heads and hidden_dim:
            return max(1, int(hidden_dim) // int(heads))
    return 256


def _arch_config(model: Any, *, window_size: int) -> dict[str, Any]:
    hidden_dim = _model_hidden_dim(model) or 1536
    return {
        "retrieval_layer": LAYER,
        "query_head": 7,
        "injection_layer": 13,
        "hidden_dim": int(hidden_dim),
        "head_dim": int(_head_dim(model, hidden_dim)),
        "crystal_layer": LAYER,
        "window_size": int(window_size),
    }


def _load_real_gemma(model_path: str | None, device: str | None) -> tuple[Any, Any]:
    from chuk_lazarus.chat_loop.cli import load_gemma

    tokenizer, model = load_gemma(model_path, device=device)
    try:
        import torch

        model_device = _model_device(model, torch)
        if model_device.type == "cuda":
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model.to(device=model_device, dtype=target_dtype)
        config = getattr(model, "config", None)
        if config is not None:
            config.use_cache = False
    except Exception:  # noqa: BLE001 - dtype optimization is best effort.
        pass
    model.eval()
    return tokenizer, model


def _model_device(model: Any, torch_mod: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return torch_mod.device(device)
    try:
        return next(model.parameters()).device
    except Exception:  # noqa: BLE001
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")


def _pad_token_id(tokenizer: Any) -> int:
    for name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    return 0


def _encoded_input_ids(tokenizer: Any, text: str) -> Any:
    encoded = tokenizer(
        str(text),
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
    )
    ids = getattr(encoded, "input_ids", None)
    if ids is None:
        ids = encoded.get("input_ids")
    if ids.ndim == 2:
        ids = ids[0]
    return ids.to(dtype=ids.dtype)


def _tensorized_windows(
    token_ids: Any,
    *,
    window_size: int,
    overlap_tokens: int,
    pad_token_id: int,
    device: Any,
    torch_mod: Any,
) -> tuple[Any, Any, int]:
    window = max(1, int(window_size))
    overlap = max(0, min(int(overlap_tokens), window - 1))
    stride = max(1, window - overlap)
    tokens = token_ids.to(device=device, dtype=torch_mod.long, non_blocking=True)
    token_count = int(tokens.numel())
    if token_count <= 0:
        tokens = torch_mod.tensor([int(pad_token_id)], dtype=torch_mod.long, device=device)
        token_count = 1

    if token_count <= window:
        num_windows = 1
    else:
        num_windows = int(math.ceil((token_count - window) / stride)) + 1
    padded_length = int((num_windows - 1) * stride + window)
    pad_length = max(0, padded_length - token_count)
    if pad_length:
        pad = torch_mod.full((pad_length,), int(pad_token_id), dtype=torch_mod.long, device=device)
        tokens = torch_mod.cat((tokens, pad), dim=0)

    windows = tokens.unfold(0, window, stride)
    starts = torch_mod.arange(num_windows, dtype=torch_mod.long, device=device).unsqueeze(1) * stride
    offsets = torch_mod.arange(window, dtype=torch_mod.long, device=device).unsqueeze(0)
    attention_mask = (starts + offsets < token_count).to(dtype=torch_mod.long)
    return windows, attention_mask, token_count


def _nested_getattr(root: Any, path: str) -> Any | None:
    current = root
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _layer_module(model: Any, layer: int) -> Any:
    layer_paths = (
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "text_model.layers",
        "text_model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "backbone.layers",
    )
    for path in layer_paths:
        layers = _nested_getattr(model, path)
        if layers is None:
            continue
        try:
            if len(layers) > int(layer):
                return layers[int(layer)]
        except TypeError:
            continue
    raise RuntimeError(f"Could not locate transformer layer {int(layer)} for {type(model).__name__}")


def _hook_hidden(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states
    last_hidden = getattr(output, "last_hidden_state", None)
    if last_hidden is not None:
        return last_hidden
    return output


def _project_routes_gpu(pooled: Any, *, dim: int, torch_mod: Any) -> Any:
    route_dim = int(dim)
    if int(pooled.shape[-1]) >= route_dim:
        projected = pooled[..., :route_dim]
    else:
        pad_shape = (*pooled.shape[:-1], route_dim - int(pooled.shape[-1]))
        pad = torch_mod.zeros(pad_shape, dtype=pooled.dtype, device=pooled.device)
        projected = torch_mod.cat((pooled, pad), dim=-1)
    return torch_mod.nn.functional.normalize(
        projected.to(dtype=torch_mod.float32),
        p=2.0,
        dim=-1,
        eps=1e-12,
    )


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: memory allocation" in message


def _memory_usage_percent(torch_mod: Any, device: Any) -> float:
    if getattr(device, "type", "") != "cuda" or not torch_mod.cuda.is_available():
        return 0.0
    try:
        free_bytes, total_bytes = torch_mod.cuda.mem_get_info(device)
        used_bytes = int(total_bytes) - int(free_bytes)
        return float(used_bytes / max(1, int(total_bytes)) * 100.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _fast_batch_size(torch_mod: Any, device: Any, requested: int | None, num_windows: int) -> int:
    if requested and int(requested) > 0:
        return max(1, min(int(requested), int(num_windows)))
    env_value = os.environ.get("LAZARUS_FAST_INDEX_BATCH_SIZE", "").strip()
    if env_value:
        try:
            return max(1, min(int(env_value), int(num_windows)))
        except ValueError:
            pass
    default = DEFAULT_CUDA_BATCH_SIZE if getattr(device, "type", "") == "cuda" else DEFAULT_CPU_BATCH_SIZE
    return max(1, min(default, int(num_windows)))


def _write_window_tokens_npz(path: Path, windows: Any, attention_mask: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    windows_cpu = windows.detach().to("cpu", dtype=windows.dtype).contiguous()
    lengths_cpu = attention_mask.sum(dim=1).detach().to("cpu", dtype=attention_mask.dtype)
    arrays: dict[str, np.ndarray] = {}
    for window_id in range(int(windows_cpu.shape[0])):
        length = int(lengths_cpu[window_id].item())
        arrays[str(window_id)] = (
            windows_cpu[window_id, :length]
            .numpy()
            .astype(np.int64, copy=False)
        )
    np.savez(str(path), **arrays)


def _window_tokens_complete(path: Path, *, expected_windows: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as zf:
            return len(zf.files) == int(expected_windows)
    except Exception:  # noqa: BLE001 - treat unreadable stores as incomplete.
        return False


def _route_fill_count(path: Path, *, expected_windows: int, dim: int) -> int | None:
    if not path.exists():
        return None
    try:
        routes = np.load(path, mmap_mode="r")
    except Exception:  # noqa: BLE001 - treat unreadable stores as incomplete.
        return None
    if tuple(routes.shape) != (int(expected_windows), int(dim)):
        return None
    norms = np.linalg.norm(np.asarray(routes, dtype=np.float32), axis=1)
    first_unfilled = next((idx for idx, norm in enumerate(norms) if float(norm) <= 1e-6), None)
    if first_unfilled is None:
        return int(expected_windows)
    return int(first_unfilled)


def _clean_fast_store(store_dir: Path) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    keep = {"activation_routes.npy", "window_tokens.npz"}
    for child in store_dir.iterdir():
        if child.name in keep:
            continue
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            for nested in sorted(child.rglob("*"), reverse=True):
                if nested.is_file() or nested.is_symlink():
                    nested.unlink()
                elif nested.is_dir():
                    nested.rmdir()
            child.rmdir()


class FastBatchIndexer:
    """Layer-12 batched indexer for complete benchmark documents."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        layer: int = LAYER,
        dim: int = DEFAULT_DIM,
        window_size: int = 512,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        batch_size: int | None = None,
    ) -> None:
        import torch

        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.layer = int(layer)
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.overlap_tokens = int(overlap_tokens)
        self.device = _model_device(model, torch)
        self.pad_token_id = _pad_token_id(tokenizer)
        self.requested_batch_size = batch_size

    def index_text(
        self,
        text: str,
        *,
        case_index: int,
        store_dir: Path,
    ) -> FastCaseIndexResult:
        from chuk_lazarus.inference.context.knowledge.activation_routes import mean_pool_hidden

        torch = self.torch
        _clean_fast_store(store_dir)

        token_ids = _encoded_input_ids(self.tokenizer, text)
        windows, attention_mask, token_count = _tensorized_windows(
            token_ids,
            window_size=self.window_size,
            overlap_tokens=self.overlap_tokens,
            pad_token_id=self.pad_token_id,
            device=self.device,
            torch_mod=torch,
        )
        num_windows = int(windows.shape[0])

        window_tokens_path = store_dir / "window_tokens.npz"
        if not _window_tokens_complete(window_tokens_path, expected_windows=num_windows):
            _write_window_tokens_npz(window_tokens_path, windows, attention_mask)

        activation_path = store_dir / "activation_routes.npy"
        filled_rows = _route_fill_count(
            activation_path,
            expected_windows=num_windows,
            dim=self.dim,
        )
        batch_size = _fast_batch_size(torch, self.device, self.requested_batch_size, num_windows)
        if filled_rows == num_windows:
            memory_percent = _memory_usage_percent(torch, self.device)
            print(
                "FAST-PATH INDEXING ACTIVE: "
                f"reusing completed case={int(case_index)} "
                f"windows={num_windows}/{num_windows} | {memory_percent:.1f}%",
                flush=True,
            )
            return FastCaseIndexResult(
                case_index=int(case_index),
                token_count=int(token_count),
                window_count=int(num_windows),
                activation_matrix_path=str(activation_path),
                window_tokens_path=str(window_tokens_path),
                store_dir=str(store_dir),
                tokens_per_second=0.0,
                memory_usage_percent=float(memory_percent),
                batch_size=int(batch_size),
            )
        if filled_rows is None:
            routes = np.lib.format.open_memmap(
                activation_path,
                mode="w+",
                dtype=np.float16,
                shape=(num_windows, self.dim),
            )
            current = 0
        else:
            routes = np.lib.format.open_memmap(
                activation_path,
                mode="r+",
                dtype=np.float16,
                shape=(num_windows, self.dim),
            )
            current = int(filled_rows)
            print(
                "FAST-PATH INDEXING ACTIVE: "
                f"resuming case={int(case_index)} from window {current}/{num_windows}",
                flush=True,
            )

        captured: dict[str, Any] = {}

        def capture_hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured["hidden"] = _hook_hidden(output)

        handle = _layer_module(self.model, self.layer).register_forward_hook(capture_hook)
        processed_tokens = int(attention_mask[:current].sum().item()) if current > 0 else 0
        started = time.perf_counter()
        max_memory_percent = _memory_usage_percent(torch, self.device)

        try:
            while current < num_windows:
                end = min(current + batch_size, num_windows)
                batch_input = windows[current:end].contiguous()
                batch_mask = attention_mask[current:end].contiguous()
                try:
                    captured.clear()
                    with torch.inference_mode():
                        kwargs = {
                            "input_ids": batch_input,
                            "attention_mask": batch_mask,
                            "use_cache": False,
                        }
                        try:
                            self.model(**kwargs)
                        except TypeError:
                            kwargs.pop("use_cache", None)
                            self.model(**kwargs)
                    hidden = captured.get("hidden")
                    if hidden is None:
                        raise RuntimeError(f"Layer {self.layer} hook did not capture hidden states")
                    pooled = mean_pool_hidden(
                        hidden,
                        batch_mask,
                        normalize=True,
                        out_dtype=torch.float32,
                    )
                    projected = _project_routes_gpu(pooled, dim=self.dim, torch_mod=torch)
                    routes[current:end] = (
                        projected.detach()
                        .to("cpu", dtype=torch.float16)
                        .numpy()
                    )
                    batch_tokens = int(batch_mask.sum().item())
                    processed_tokens += batch_tokens
                    current = end
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    elapsed = max(1e-9, time.perf_counter() - started)
                    memory_percent = _memory_usage_percent(torch, self.device)
                    max_memory_percent = max(max_memory_percent, memory_percent)
                    print(
                        "FAST-PATH INDEXING ACTIVE: "
                        f"{processed_tokens / elapsed:.1f} tokens/sec | "
                        f"{memory_percent:.1f}% "
                        f"(case={int(case_index)} windows={current}/{num_windows} batch={batch_size})",
                        flush=True,
                    )
                except RuntimeError as exc:
                    if batch_size <= 1 or not _is_cuda_oom(exc):
                        raise
                    batch_size = max(1, batch_size // 2)
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    print(
                        f"FAST-PATH INDEXING OOM: reducing batch_size to {batch_size}",
                        flush=True,
                    )
                    continue
        finally:
            handle.remove()
            routes.flush()
            del routes

        elapsed = max(1e-9, time.perf_counter() - started)
        tokens_per_second = float(processed_tokens / elapsed)
        print(
            "FAST-PATH INDEXING COMPLETE: "
            f"{tokens_per_second:.1f} tokens/sec | "
            f"{max_memory_percent:.1f}% peak "
            f"(case={int(case_index)} tokens={token_count} windows={num_windows})",
            flush=True,
        )
        return FastCaseIndexResult(
            case_index=int(case_index),
            token_count=int(token_count),
            window_count=int(num_windows),
            activation_matrix_path=str(activation_path),
            window_tokens_path=str(window_tokens_path),
            store_dir=str(store_dir),
            tokens_per_second=tokens_per_second,
            memory_usage_percent=float(max_memory_percent),
            batch_size=int(batch_size),
        )


def jit_index_dataset_windows(
    *,
    benchmark_slug: str,
    raw_texts: Iterable[str],
    output_dir: Path,
    tokens_per_window: int,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    dim: int = DEFAULT_DIM,
    model_path: str | None = None,
    device: str | None = None,
    reuse_existing: bool = True,
    batch_size: int | None = None,
) -> JitIndexResult:
    """Materialize Layer-12 activation routes using the benchmark fast path."""

    window_size = max(1, int(tokens_per_window))
    overlap = max(0, min(int(overlap_tokens), window_size - 1))
    route_root = output_dir / f"{benchmark_slug}_activation_routes"
    metadata_path = output_dir / f"{benchmark_slug}_jit_index_metadata.json"

    if reuse_existing and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            first_matrix_path = str(metadata.get("activation_matrix_path", ""))
            window_count = int(metadata.get("window_count", 0))
            token_count = int(metadata.get("token_count", 0))
            mode = str(metadata.get("mode", "unknown"))
            case_matrices = list(metadata.get("case_matrices", []))
        except Exception:  # noqa: BLE001 - fall through and rebuild.
            first_matrix_path = ""
        else:
            all_case_paths_exist = bool(case_matrices) and all(
                Path(str(case.get("activation_matrix_path", ""))).exists()
                and Path(str(case.get("window_tokens_path", ""))).exists()
                for case in case_matrices
                if isinstance(case, dict)
            )
            if (
                mode == "real_gemma4_fast_batch_layer12"
                and first_matrix_path
                and Path(first_matrix_path).exists()
                and route_root.exists()
                and all_case_paths_exist
            ):
                print(
                    "FAST-PATH INDEXING ACTIVE: "
                    f"{float(metadata.get('tokens_per_second', 0.0)):.1f} tokens/sec | "
                    f"{float(metadata.get('memory_usage_percent', 0.0)):.1f}% "
                    "(reused)",
                    flush=True,
                )
                return JitIndexResult(
                    window_count=int(window_count),
                    activation_matrix_path=first_matrix_path,
                    activation_route_dir=str(route_root),
                    metadata_path=str(metadata_path),
                    layer=LAYER,
                    mode=mode,
                    token_count=int(token_count),
                    tokens_per_second=float(metadata.get("tokens_per_second", 0.0)),
                    memory_usage_percent=float(metadata.get("memory_usage_percent", 0.0)),
                )

    route_root.mkdir(parents=True, exist_ok=True)

    tokenizer, model = _load_real_gemma(model_path, device)
    arch_config = _arch_config(model, window_size=window_size)
    indexer = FastBatchIndexer(
        tokenizer=tokenizer,
        model=model,
        layer=LAYER,
        dim=int(dim),
        window_size=window_size,
        overlap_tokens=overlap,
        batch_size=batch_size,
    )

    mode = "real_gemma4_fast_batch_layer12"
    case_metadata: list[dict[str, Any]] = []
    first_matrix_path = ""
    total_windows = 0
    total_tokens = 0
    weighted_token_seconds = 0.0
    max_memory_percent = 0.0

    try:
        for case_index, text in enumerate(raw_texts):
            store_dir = route_root / f"case_{case_index:05d}" / "fast_store"
            result = indexer.index_text(str(text), case_index=case_index, store_dir=store_dir)
            result_dict = result.as_dict()
            if not first_matrix_path:
                first_matrix_path = result.activation_matrix_path
            total_windows += int(result.window_count)
            total_tokens += int(result.token_count)
            if result.tokens_per_second > 0.0:
                weighted_token_seconds += int(result.token_count) / float(result.tokens_per_second)
            max_memory_percent = max(max_memory_percent, float(result.memory_usage_percent))
            case_metadata.append(result_dict)
    finally:
        try:
            del indexer
        except Exception:  # noqa: BLE001 - cleanup best effort.
            pass
        try:
            del model
        except Exception:  # noqa: BLE001 - cleanup best effort.
            pass
        try:
            del tokenizer
        except Exception:  # noqa: BLE001 - cleanup best effort.
            pass
        release_cuda_memory()

    aggregate_tps = (
        float(total_tokens / weighted_token_seconds)
        if weighted_token_seconds > 0.0
        else 0.0
    )
    metadata = {
        "benchmark_slug": benchmark_slug,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "layer": LAYER,
        "window_count": int(total_windows),
        "token_count": int(total_tokens),
        "case_count": len(case_metadata),
        "activation_route_dir": str(route_root),
        "activation_matrix_path": first_matrix_path,
        "case_matrices": case_metadata,
        "dim": int(dim),
        "dtype": "float16",
        "window_tokens": int(window_size),
        "overlap_tokens": int(overlap),
        "stride_tokens": int(window_size - overlap),
        "mode": mode,
        "tokens_per_second": float(aggregate_tps),
        "memory_usage_percent": float(max_memory_percent),
        "arch_config": arch_config,
        "fast_path": {
            "tensorized_windowing": "torch.Tensor.unfold",
            "batched_forward": True,
            "layer_hook": LAYER,
            "gpu_mean_pooling": True,
            "projection_dim": int(dim),
            "activation_routes": "np.lib.format.open_memmap(mode='w+')",
            "bypassed_files": [
                "boundaries/window_*.npy",
                "residual_streams/window_*.npy",
                "activation_routes/window_*.npy",
                "keywords.json",
                "idf.json",
            ],
        },
    }
    _write_json(metadata_path, metadata)
    print(
        "FAST-PATH INDEXING ACTIVE: "
        f"{aggregate_tps:.1f} tokens/sec | {max_memory_percent:.1f}% "
        "- proceeding to first query.",
        flush=True,
    )
    return JitIndexResult(
        window_count=int(total_windows),
        activation_matrix_path=first_matrix_path,
        activation_route_dir=str(route_root),
        metadata_path=str(metadata_path),
        layer=LAYER,
        mode=mode,
        token_count=int(total_tokens),
        tokens_per_second=float(aggregate_tps),
        memory_usage_percent=float(max_memory_percent),
    )
