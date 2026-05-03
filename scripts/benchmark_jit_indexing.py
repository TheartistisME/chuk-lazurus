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
APOLLO_LAYER = LAYER + 1
DEFAULT_DIM = 128
DEFAULT_OVERLAP_TOKENS = 128
DEFAULT_CUDA_BATCH_SIZE = 128
DEFAULT_CPU_BATCH_SIZE = 4
APOLLO_MANIFEST_NAME = "manifest.json"
APOLLO_BOUNDARY_RESIDUAL_NAME = "boundary_residual.npy"
APOLLO_BOUNDARIES_DIR = "boundaries"
APOLLO_RESIDUAL_STREAMS_DIR = "residual_streams"
APOLLO_SEMANTICS = "input_ids_seeded_boundary_prehook"


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
    apollo_ready: bool = False

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
            "apollo_ready": bool(self.apollo_ready),
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
    apollo_manifest_path: str = ""
    boundary_residual_path: str = ""
    boundaries_dir: str = ""
    residual_streams_dir: str = ""
    apollo_window_count: int = 0
    apollo_hidden_dim: int = 0
    apollo_tokens_per_second: float = 0.0
    apollo_reused: bool = False
    apollo_ready: bool = False

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
            "apollo_manifest_path": self.apollo_manifest_path,
            "boundary_residual_path": self.boundary_residual_path,
            "boundaries_dir": self.boundaries_dir,
            "residual_streams_dir": self.residual_streams_dir,
            "apollo_window_count": int(self.apollo_window_count),
            "apollo_hidden_dim": int(self.apollo_hidden_dim),
            "apollo_tokens_per_second": float(self.apollo_tokens_per_second),
            "apollo_reused": bool(self.apollo_reused),
            "apollo_ready": bool(self.apollo_ready),
        }


@dataclass(frozen=True)
class ApolloResidualPassResult:
    manifest_path: str
    boundary_residual_path: str
    boundaries_dir: str
    residual_streams_dir: str
    window_count: int
    token_count: int
    hidden_dim: int
    tokens_per_second: float
    reused: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "boundary_residual_path": self.boundary_residual_path,
            "boundaries_dir": self.boundaries_dir,
            "residual_streams_dir": self.residual_streams_dir,
            "window_count": int(self.window_count),
            "token_count": int(self.token_count),
            "hidden_dim": int(self.hidden_dim),
            "tokens_per_second": float(self.tokens_per_second),
            "reused": bool(self.reused),
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


def _load_window_tokens_npz(path: Path) -> dict[int, list[int]]:
    with np.load(str(path), allow_pickle=False) as zf:
        token_lists: dict[int, list[int]] = {}
        for key in sorted(zf.files, key=lambda item: int(item)):
            token_lists[int(key)] = [int(token) for token in zf[key].tolist()]
    return token_lists


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


def _clear_apollo_artifacts(store_dir: Path) -> None:
    for child in (
        store_dir / APOLLO_BOUNDARIES_DIR,
        store_dir / APOLLO_RESIDUAL_STREAMS_DIR,
    ):
        if not child.exists():
            continue
        for nested in sorted(child.rglob("*"), reverse=True):
            if nested.is_file() or nested.is_symlink():
                nested.unlink()
            elif nested.is_dir():
                nested.rmdir()
        child.rmdir()
    for child in (
        store_dir / APOLLO_BOUNDARY_RESIDUAL_NAME,
        store_dir / APOLLO_MANIFEST_NAME,
    ):
        if child.exists() or child.is_symlink():
            child.unlink()


def _model_identity(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    candidates = (config, text_config)
    identity: dict[str, Any] = {
        "model_class": type(model).__name__,
    }
    for field in ("name_or_path", "_name_or_path", "model_type", "architectures"):
        for source in candidates:
            value = getattr(source, field, None)
            if value:
                identity[field.lstrip("_")] = value
                break
    hidden_dim = _model_hidden_dim(model)
    if hidden_dim:
        identity["hidden_dim"] = int(hidden_dim)
    return identity


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "tokenizer_class": type(tokenizer).__name__,
    }
    for field in ("name_or_path", "_name_or_path", "vocab_size", "model_max_length"):
        value = getattr(tokenizer, field, None)
        if value is not None:
            identity[field.lstrip("_")] = value
    return identity


def _call_model_for_apollo(model: Any, kwargs: dict[str, Any]) -> None:
    try:
        model(**kwargs)
    except TypeError:
        kwargs = dict(kwargs)
        kwargs.pop("use_cache", None)
        model(**kwargs)


def _boundary_as_prefix(boundary: Any, *, like: Any, torch_mod: Any) -> Any:
    prefix = boundary
    if not torch_mod.is_tensor(prefix):
        prefix = torch_mod.as_tensor(prefix)
    prefix = prefix.to(device=like.device, dtype=like.dtype)
    if prefix.ndim == 1:
        prefix = prefix.view(1, 1, -1)
    elif prefix.ndim == 2:
        prefix = prefix.unsqueeze(0)
    elif prefix.ndim != 3:
        raise ValueError(f"Unsupported boundary residual rank: {int(prefix.ndim)}")
    if int(prefix.shape[0]) != int(like.shape[0]):
        if int(prefix.shape[0]) == 1:
            prefix = prefix.expand(int(like.shape[0]), -1, -1)
        else:
            raise ValueError("Boundary residual batch size does not match window batch")
    if int(prefix.shape[-1]) != int(like.shape[-1]):
        raise ValueError(
            "Boundary residual hidden dim does not match model embeddings: "
            f"{int(prefix.shape[-1])} != {int(like.shape[-1])}"
        )
    return prefix.contiguous()


def _boundary_seed_ids(input_ids: Any, *, prefix_len: int, torch_mod: Any) -> Any:
    seed = torch_mod.zeros(
        (int(input_ids.shape[0]), int(prefix_len)),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    return torch_mod.cat((seed, input_ids), dim=1)


def _forward_window_to_boundary_layer(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    layer: int,
    initial_residual: Any | None,
    torch_mod: Any,
) -> Any:
    """Forward one window and capture post-boundary-layer residual stream."""

    captured: dict[str, Any] = {}

    def capture_hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["hidden"] = _hook_hidden(output)

    boundary_layer = _layer_module(model, layer)
    capture_handle = boundary_layer.register_forward_hook(capture_hook)
    prepend_handle = None
    try:
        with torch_mod.inference_mode():
            if initial_residual is None:
                kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                }
                _call_model_for_apollo(model, kwargs)
            else:
                # Gemma4 cannot safely run this path with inputs_embeds only:
                # it reverse-maps embeddings against the full vocab to recover
                # token ids for per-layer inputs. Keep the input_ids path live
                # and replace the dummy prefix hidden state before layer 0.
                reference = model.get_input_embeddings()(input_ids[:, :1])
                prefix = _boundary_as_prefix(initial_residual, like=reference, torch_mod=torch_mod)
                prefix_len = int(prefix.shape[1])
                seeded_input_ids = _boundary_seed_ids(
                    input_ids,
                    prefix_len=prefix_len,
                    torch_mod=torch_mod,
                )
                prefix_mask = torch_mod.ones(
                    (int(attention_mask.shape[0]), int(prefix_len)),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                extended_mask = torch_mod.cat((prefix_mask, attention_mask), dim=1)

                def prepend_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                    if not inputs:
                        return inputs
                    hidden = inputs[0]
                    if not torch_mod.is_tensor(hidden) or int(hidden.ndim) != 3:
                        return inputs
                    prefix_local = prefix.to(device=hidden.device, dtype=hidden.dtype)
                    if int(prefix_local.shape[0]) != int(hidden.shape[0]):
                        if int(prefix_local.shape[0]) == 1:
                            prefix_local = prefix_local.expand(int(hidden.shape[0]), -1, -1)
                        else:
                            raise ValueError("Boundary residual batch size does not match hidden batch")
                    if int(prefix_local.shape[1]) > int(hidden.shape[1]):
                        raise ValueError("Boundary residual prefix is longer than seeded hidden states")
                    adjusted = hidden.clone()
                    adjusted[:, : int(prefix_local.shape[1]), :] = prefix_local
                    return (adjusted, *inputs[1:])

                prepend_handle = _layer_module(model, 0).register_forward_pre_hook(prepend_hook)
                kwargs = {
                    "input_ids": seeded_input_ids,
                    "attention_mask": extended_mask,
                    "use_cache": False,
                }
                _call_model_for_apollo(model, kwargs)
        hidden = captured.get("hidden")
        if hidden is None:
            raise RuntimeError(f"Layer {int(layer)} hook did not capture hidden states")
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        return hidden
    finally:
        if prepend_handle is not None:
            prepend_handle.remove()
        capture_handle.remove()


def _save_float32_array(path: Path, tensor: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().float().to("cpu").contiguous().numpy()
    np.save(str(path), array.astype(np.float32, copy=False))


def _apollo_manifest_path(store_dir: Path) -> Path:
    return store_dir / APOLLO_MANIFEST_NAME


def _apollo_boundaries_dir(store_dir: Path) -> Path:
    return store_dir / APOLLO_BOUNDARIES_DIR


def _apollo_residual_streams_dir(store_dir: Path) -> Path:
    return store_dir / APOLLO_RESIDUAL_STREAMS_DIR


def _apollo_residual_complete(
    store_dir: Path,
    *,
    expected_windows: int,
    layer: int = APOLLO_LAYER,
) -> bool:
    manifest_path = _apollo_manifest_path(store_dir)
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    if manifest.get("kind") != "benchmark_jit_apollo_sequential_residual":
        return False
    if manifest.get("status") != "ready" or manifest.get("apollo_ready") is not True:
        return False
    if int(manifest.get("num_windows", -1)) != int(expected_windows):
        return False
    if int(manifest.get("layer", -1)) != int(layer):
        return False
    if manifest.get("semantics") != APOLLO_SEMANTICS:
        return False
    if manifest.get("row_alignment") != "window_id":
        return False
    document_order = manifest.get("document_order")
    if not isinstance(document_order, list) or len(document_order) != int(expected_windows):
        return False
    final_boundary = store_dir / str(manifest.get("boundary_residual_path", APOLLO_BOUNDARY_RESIDUAL_NAME))
    if not final_boundary.exists():
        return False
    boundaries_dir = store_dir / APOLLO_BOUNDARIES_DIR
    streams_dir = store_dir / APOLLO_RESIDUAL_STREAMS_DIR
    hidden_dim = int(manifest.get("hidden_dim", 0) or 0)
    for raw_window_id in document_order:
        try:
            window_id = int(raw_window_id)
        except Exception:  # noqa: BLE001
            return False
        boundary_path = boundaries_dir / f"window_{window_id:03d}.npy"
        stream_path = streams_dir / f"window_{window_id:03d}.npy"
        if not boundary_path.exists() or not stream_path.exists():
            return False
        if hidden_dim > 0:
            try:
                boundary = np.load(str(boundary_path), mmap_mode="r", allow_pickle=False)
                stream = np.load(str(stream_path), mmap_mode="r", allow_pickle=False)
            except Exception:  # noqa: BLE001
                return False
            if tuple(boundary.shape) != (hidden_dim,):
                return False
            if int(stream.ndim) != 2 or int(stream.shape[-1]) != hidden_dim:
                return False
    if hidden_dim > 0:
        try:
            final = np.load(str(final_boundary), mmap_mode="r", allow_pickle=False)
        except Exception:  # noqa: BLE001
            return False
        if tuple(final.shape) != (1, 1, hidden_dim):
            return False
    return True


class ApolloSequentialResidualPass:
    """Sequential residual sidecar writer for benchmark JIT stores."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        layer: int = APOLLO_LAYER,
        window_size: int = 512,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ) -> None:
        import torch

        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.layer = int(layer)
        self.window_size = int(window_size)
        self.overlap_tokens = int(overlap_tokens)
        self.device = _model_device(model, torch)

    def ensure(
        self,
        *,
        store_dir: Path,
        activation_path: Path,
        window_tokens_path: Path,
        token_count: int,
        expected_windows: int,
        force: bool = False,
    ) -> ApolloResidualPassResult:
        if not force and _apollo_residual_complete(
            store_dir,
            expected_windows=expected_windows,
            layer=self.layer,
        ):
            manifest = json.loads(_apollo_manifest_path(store_dir).read_text(encoding="utf-8"))
            return ApolloResidualPassResult(
                manifest_path=str(_apollo_manifest_path(store_dir)),
                boundary_residual_path=str(store_dir / APOLLO_BOUNDARY_RESIDUAL_NAME),
                boundaries_dir=str(_apollo_boundaries_dir(store_dir)),
                residual_streams_dir=str(_apollo_residual_streams_dir(store_dir)),
                window_count=int(expected_windows),
                token_count=int(token_count),
                hidden_dim=int(manifest.get("hidden_dim", 0) or 0),
                tokens_per_second=0.0,
                reused=True,
            )

        return self.build(
            store_dir=store_dir,
            activation_path=activation_path,
            window_tokens_path=window_tokens_path,
            token_count=token_count,
            expected_windows=expected_windows,
        )

    def build(
        self,
        *,
        store_dir: Path,
        activation_path: Path,
        window_tokens_path: Path,
        token_count: int,
        expected_windows: int,
    ) -> ApolloResidualPassResult:
        torch = self.torch
        token_lists = _load_window_tokens_npz(window_tokens_path)
        document_order = sorted(token_lists)
        if len(document_order) != int(expected_windows):
            raise RuntimeError(
                "Cannot build Apollo residual sidecars: "
                f"expected {int(expected_windows)} token windows, found {len(document_order)}"
            )

        _clear_apollo_artifacts(store_dir)
        boundaries_dir = _apollo_boundaries_dir(store_dir)
        residual_streams_dir = _apollo_residual_streams_dir(store_dir)
        boundaries_dir.mkdir(parents=True, exist_ok=True)
        residual_streams_dir.mkdir(parents=True, exist_ok=True)

        boundary = None
        hidden_dim = _model_hidden_dim(self.model)
        started = time.perf_counter()
        processed_tokens = 0
        for window_id in document_order:
            token_ids = token_lists[int(window_id)] or [_pad_token_id(self.tokenizer)]
            input_ids = torch.tensor(
                [token_ids],
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
            hidden = _forward_window_to_boundary_layer(
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer=self.layer,
                initial_residual=boundary,
                torch_mod=torch,
            )
            valid_len = int(input_ids.shape[1])
            stream = hidden[:, -valid_len:, :]
            boundary = hidden[:, -1:, :].detach()
            hidden_dim = int(boundary.shape[-1])
            _save_float32_array(boundaries_dir / f"window_{int(window_id):03d}.npy", boundary[0, 0, :])
            _save_float32_array(
                residual_streams_dir / f"window_{int(window_id):03d}.npy",
                stream[0, :, :],
            )
            processed_tokens += valid_len

        if boundary is None:
            raise RuntimeError("Cannot build Apollo residual sidecars without at least one window")
        boundary_residual_path = store_dir / APOLLO_BOUNDARY_RESIDUAL_NAME
        _save_float32_array(boundary_residual_path, boundary.detach().float().to("cpu"))

        activation_shape: list[int] = []
        activation_dtype = "float16"
        if activation_path.exists():
            routes = np.load(str(activation_path), mmap_mode="r", allow_pickle=False)
            activation_shape = [int(axis) for axis in routes.shape]
            activation_dtype = str(routes.dtype)
        stride_tokens = max(1, int(self.window_size) - max(0, int(self.overlap_tokens)))
        model_identity = _model_identity(self.model)
        tokenizer_identity = _tokenizer_identity(self.tokenizer)
        arch_config = dict(_arch_config(self.model, window_size=self.window_size))
        arch_config["retrieval_layer"] = int(self.layer)
        arch_config["crystal_layer"] = int(self.layer)
        arch_config["injection_layer"] = int(self.layer)
        manifest = {
            "version": 1,
            "kind": "benchmark_jit_apollo_sequential_residual",
            "status": "ready",
            "apollo_ready": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "layer": int(self.layer),
            "crystal_layer": int(self.layer),
            "source_layer": int(self.layer),
            "target_layer": int(self.layer) + 1,
            "arch_config": arch_config,
            "model_id": model_identity.get("name_or_path")
            or model_identity.get("model_type")
            or type(self.model).__name__,
            "tokenizer_id": tokenizer_identity.get("name_or_path") or type(self.tokenizer).__name__,
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
            "num_windows": int(expected_windows),
            "num_tokens": int(token_count),
            "processed_tokens": int(processed_tokens),
            "window_tokens": int(self.window_size),
            "window_size": int(self.window_size),
            "overlap": int(self.overlap_tokens),
            "overlap_tokens": int(self.overlap_tokens),
            "stride": int(stride_tokens),
            "stride_tokens": int(stride_tokens),
            "document_order": [int(window_id) for window_id in document_order],
            "semantics": APOLLO_SEMANTICS,
            "boundary_source": "last_post_crystal_hidden",
            "input_policy": "window_tokens_row_in_document_order",
            "stream_policy": "valid_window_tokens_excludes_prefix_boundary",
            "dtype": "float32",
            "hidden_dim": int(hidden_dim),
            "row_alignment": "window_id",
            "activation_routes": {
                "path": "activation_routes.npy",
                "dtype": activation_dtype,
                "shape": activation_shape,
                "count": int(expected_windows),
                "row_alignment": "window_id",
                "query_path": True,
            },
            "window_tokens_path": "window_tokens.npz",
            "boundary_residual_path": APOLLO_BOUNDARY_RESIDUAL_NAME,
            "boundaries_dir": APOLLO_BOUNDARIES_DIR,
            "residual_streams_dir": APOLLO_RESIDUAL_STREAMS_DIR,
            "artifacts": {
                "final_boundary": APOLLO_BOUNDARY_RESIDUAL_NAME,
                "boundaries_dir": APOLLO_BOUNDARIES_DIR,
                "residual_streams_dir": APOLLO_RESIDUAL_STREAMS_DIR,
            },
        }
        _write_json(_apollo_manifest_path(store_dir), manifest)
        elapsed = max(1e-9, time.perf_counter() - started)
        tokens_per_second = float(processed_tokens / elapsed)
        print(
            "APOLLO RESIDUAL PASS COMPLETE: "
            f"{tokens_per_second:.1f} tokens/sec "
            f"(windows={int(expected_windows)} layer={int(self.layer)})",
            flush=True,
        )
        return ApolloResidualPassResult(
            manifest_path=str(_apollo_manifest_path(store_dir)),
            boundary_residual_path=str(boundary_residual_path),
            boundaries_dir=str(boundaries_dir),
            residual_streams_dir=str(residual_streams_dir),
            window_count=int(expected_windows),
            token_count=int(token_count),
            hidden_dim=int(hidden_dim),
            tokens_per_second=tokens_per_second,
            reused=False,
        )


def _clean_fast_store(store_dir: Path) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    keep = {
        "activation_routes.npy",
        "window_tokens.npz",
        APOLLO_MANIFEST_NAME,
        APOLLO_BOUNDARY_RESIDUAL_NAME,
        APOLLO_BOUNDARIES_DIR,
        APOLLO_RESIDUAL_STREAMS_DIR,
    }
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

    def _ensure_apollo_sidecars(
        self,
        *,
        store_dir: Path,
        activation_path: Path,
        window_tokens_path: Path,
        token_count: int,
        expected_windows: int,
    ) -> ApolloResidualPassResult:
        pass_runner = ApolloSequentialResidualPass(
            tokenizer=self.tokenizer,
            model=self.model,
            layer=self.layer + 1,
            window_size=self.window_size,
            overlap_tokens=self.overlap_tokens,
        )
        return pass_runner.ensure(
            store_dir=store_dir,
            activation_path=activation_path,
            window_tokens_path=window_tokens_path,
            token_count=token_count,
            expected_windows=expected_windows,
        )

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
            if not _apollo_residual_complete(
                store_dir,
                expected_windows=num_windows,
                layer=self.layer + 1,
            ):
                print(
                    "FAST JIT FOUND: Apollo sidecars missing. "
                    f"Running Apollo upgrader only for case={int(case_index)} windows={num_windows}.",
                    flush=True,
                )
            apollo_result = self._ensure_apollo_sidecars(
                store_dir=store_dir,
                activation_path=activation_path,
                window_tokens_path=window_tokens_path,
                token_count=token_count,
                expected_windows=num_windows,
            )
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
                apollo_manifest_path=apollo_result.manifest_path,
                boundary_residual_path=apollo_result.boundary_residual_path,
                boundaries_dir=apollo_result.boundaries_dir,
                residual_streams_dir=apollo_result.residual_streams_dir,
                apollo_window_count=apollo_result.window_count,
                apollo_hidden_dim=apollo_result.hidden_dim,
                apollo_tokens_per_second=apollo_result.tokens_per_second,
                apollo_reused=apollo_result.reused,
                apollo_ready=True,
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
        apollo_result = self._ensure_apollo_sidecars(
            store_dir=store_dir,
            activation_path=activation_path,
            window_tokens_path=window_tokens_path,
            token_count=token_count,
            expected_windows=num_windows,
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
            apollo_manifest_path=apollo_result.manifest_path,
            boundary_residual_path=apollo_result.boundary_residual_path,
            boundaries_dir=apollo_result.boundaries_dir,
            residual_streams_dir=apollo_result.residual_streams_dir,
            apollo_window_count=apollo_result.window_count,
            apollo_hidden_dim=apollo_result.hidden_dim,
            apollo_tokens_per_second=apollo_result.tokens_per_second,
            apollo_reused=apollo_result.reused,
            apollo_ready=True,
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
            all_fast_case_paths_exist = bool(case_matrices)
            all_apollo_case_paths_exist = bool(case_matrices)
            for case in case_matrices:
                if not isinstance(case, dict):
                    all_fast_case_paths_exist = False
                    all_apollo_case_paths_exist = False
                    break
                store_dir = Path(str(case.get("store_dir", "")))
                expected_windows = int(case.get("window_count", 0) or 0)
                activation_matrix_path = Path(str(case.get("activation_matrix_path", "")))
                window_tokens_path = Path(str(case.get("window_tokens_path", "")))
                fast_jit_ready = (
                    _route_fill_count(
                        activation_matrix_path,
                        expected_windows=expected_windows,
                        dim=int(dim),
                    )
                    == expected_windows
                    and _window_tokens_complete(
                        window_tokens_path,
                        expected_windows=expected_windows,
                    )
                )
                if not fast_jit_ready:
                    all_fast_case_paths_exist = False
                    all_apollo_case_paths_exist = False
                    break
                if not _apollo_residual_complete(
                    store_dir,
                    expected_windows=expected_windows,
                    layer=APOLLO_LAYER,
                ):
                    all_apollo_case_paths_exist = False
            fast_cache_ready = (
                mode == "real_gemma4_fast_batch_layer12"
                and first_matrix_path
                and Path(first_matrix_path).exists()
                and route_root.exists()
                and all_fast_case_paths_exist
            )
            if fast_cache_ready and all_apollo_case_paths_exist:
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
                    apollo_ready=True,
                )
            if fast_cache_ready:
                print(
                    "FAST-PATH INDEXING ACTIVE: "
                    "reusing fast routes; Apollo sidecars missing - running Apollo upgrader only.",
                    flush=True,
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
    apollo_total_tokens = sum(int(case.get("token_count", 0) or 0) for case in case_metadata)
    apollo_weighted_token_seconds = sum(
        int(case.get("token_count", 0) or 0) / float(case.get("apollo_tokens_per_second", 0.0))
        for case in case_metadata
        if float(case.get("apollo_tokens_per_second", 0.0)) > 0.0
    )
    apollo_tps = (
        float(apollo_total_tokens / apollo_weighted_token_seconds)
        if apollo_weighted_token_seconds > 0.0
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
            "route_query_path": "activation_routes.npy",
            "window_tokens_path": "window_tokens.npz",
            "bypassed_files": [
                "activation_routes/window_*.npy",
                "keywords.json",
                "idf.json",
            ],
        },
        "apollo_residual_pass": {
            "enabled": True,
            "status": "ready" if all(case.get("apollo_ready") for case in case_metadata) else "incomplete",
            "semantics": APOLLO_SEMANTICS,
            "row_alignment": "window_id",
            "layer": LAYER,
            "crystal_layer": APOLLO_LAYER,
            "source_layer": APOLLO_LAYER,
            "case_count": len(case_metadata),
            "window_count": int(total_windows),
            "token_count": int(apollo_total_tokens),
            "tokens_per_second": float(apollo_tps),
            "query_path_preserved": "activation_routes.npy",
            "case_manifests": [
                str(case.get("apollo_manifest_path", ""))
                for case in case_metadata
                if case.get("apollo_manifest_path")
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
        apollo_ready=True,
    )
