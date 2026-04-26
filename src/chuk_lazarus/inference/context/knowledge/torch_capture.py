"""Torch-native residual capture helpers for Apollo knowledge builds."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

try:  # Torch is required for the build path, but keep the module import-safe.
    import torch
except Exception:  # pragma: no cover - exercised only on hosts without torch
    torch = None


def _require_torch():
    if torch is None:  # pragma: no cover - local safety net
        raise RuntimeError("Torch is required for torch-native knowledge capture")
    return torch


def _infer_model_device(model: Any):
    torch_mod = _require_torch()
    for container in (getattr(model, "parameters", None), getattr(model, "buffers", None)):
        if callable(container):
            try:
                item = next(container())
            except StopIteration:
                continue
            except TypeError:
                continue
            return item.device
    return torch_mod.device("cpu")


def _resolve_transformer_layers(model: Any) -> list[Any]:
    candidates = [
        # VLM wrappers — language backbone nested under .language_model.
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        # Plain causal-LM shapes.
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        (None, "layers"),
    ]

    for path in candidates:
        target = model
        *outer, inner_name = path
        ok = True
        for step in outer:
            if step is None:
                continue
            target = getattr(target, step, None)
            if target is None:
                ok = False
                break
        if not ok:
            continue

        layers = getattr(target, inner_name, None)
        if layers is None:
            continue
        if not hasattr(layers, "__iter__") or not hasattr(layers, "__len__"):
            continue
        try:
            length = len(layers)
        except TypeError:
            continue
        if length == 0:
            continue
        return list(layers)

    raise ValueError(
        "Cannot resolve transformer layers for residual capture. "
        "Expected model.model.layers, model.model.language_model.layers, "
        "model.language_model.model.layers, model.transformer.h, "
        "gpt_neox.layers, or model.layers."
    )


def _extract_boundary_tensor(output: Any):
    torch_mod = _require_torch()

    hidden = output[0] if isinstance(output, tuple) else output
    if not torch_mod.is_tensor(hidden):
        hidden = torch_mod.as_tensor(hidden)

    if hidden.ndim == 3:
        boundary = hidden[0, -1, :]
    elif hidden.ndim == 2:
        boundary = hidden[-1, :]
    elif hidden.ndim == 1:
        boundary = hidden
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"Unsupported residual tensor rank: {hidden.ndim}")

    return boundary.detach().to("cpu", dtype=torch_mod.float32).contiguous()


def _extract_stream_tensor(output: Any):
    torch_mod = _require_torch()

    hidden = output[0] if isinstance(output, tuple) else output
    if not torch_mod.is_tensor(hidden):
        hidden = torch_mod.as_tensor(hidden)

    if hidden.ndim == 3:
        stream = hidden[0, :, :]
    elif hidden.ndim == 2:
        stream = hidden
    elif hidden.ndim == 1:
        stream = hidden.unsqueeze(0)
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"Unsupported residual tensor rank: {hidden.ndim}")

    return stream.detach().to("cpu", dtype=torch_mod.float32).contiguous()


def _coerce_boundary_residual(initial_residual: Any, *, hidden_size: int):
    torch_mod = _require_torch()
    boundary = torch_mod.as_tensor(initial_residual, dtype=torch_mod.float32)
    if boundary.ndim == 3:
        boundary = boundary[:, -1, :]
    elif boundary.ndim == 2:
        boundary = boundary[-1:, :]
    elif boundary.ndim == 1:
        boundary = boundary.unsqueeze(0)
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"Unsupported boundary residual rank: {boundary.ndim}")

    if int(boundary.shape[-1]) != hidden_size:
        raise ValueError(
            f"Boundary residual hidden size {int(boundary.shape[-1])} "
            f"does not match model hidden size {hidden_size}"
        )

    return boundary


def capture_post_crystal_boundary(
    model: Any,
    token_ids: Sequence[int],
    *,
    crystal_layer: int,
    device: str | Any | None = None,
    initial_residual: Any | None = None,
    return_stream: bool = False,
):
    """Capture the post-crystal-layer boundary vector for one token window."""

    from chuk_lazarus import tracing

    _trace_t0 = time.perf_counter()

    torch_mod = _require_torch()
    layers = _resolve_transformer_layers(model)
    if crystal_layer < 0 or crystal_layer >= len(layers):
        raise ValueError(
            f"crystal_layer={crystal_layer} is outside the model's layer range (0..{len(layers) - 1})"
        )

    _token_ids_list = list(token_ids)
    if tracing.is_enabled("a1"):
        # A1 probe: function entry metadata.
        tracing.emit(
            "a1",
            "capture.begin",
            {
                "crystal_layer": int(crystal_layer),
                "layers_total": int(len(layers)),
                "layer_module": type(layers[crystal_layer]).__name__,
                "token_count": int(len(_token_ids_list)),
                "token_ids_head": [int(t) for t in _token_ids_list[:8]],
                "token_ids_tail": [int(t) for t in _token_ids_list[-8:]],
                "has_initial_residual": initial_residual is not None,
            },
        )

    model_device = torch_mod.device(device) if device is not None else _infer_model_device(model)
    input_ids = torch_mod.as_tensor(_token_ids_list, dtype=torch_mod.long, device=model_device)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    attention_mask = torch_mod.ones_like(input_ids, dtype=torch_mod.long)

    captured: dict[str, Any] = {}

    def _hook(_module, _args, output):
        captured["tensor"] = _extract_boundary_tensor(output)
        if return_stream:
            captured["stream"] = _extract_stream_tensor(output)

    handle = layers[crystal_layer].register_forward_hook(_hook)
    inject_handle = None
    if initial_residual is not None:
        boundary = _coerce_boundary_residual(
            initial_residual,
            hidden_size=int(model.get_input_embeddings().embedding_dim),
        )
        if tracing.is_enabled("a1"):
            # A1 probe: coerced initial residual stats.
            tracing.emit("a1", "capture.initial_residual", tracing.tensor_stats(boundary))

        def _inject_hook(_module, inputs):
            if not inputs:
                return inputs

            hidden_states = inputs[0]
            adjusted = hidden_states.clone()
            boundary_hidden = boundary.to(device=hidden_states.device, dtype=hidden_states.dtype)
            adjusted[:, -1, :] = adjusted[:, -1, :] + boundary_hidden
            return (adjusted, *inputs[1:])

        inject_handle = layers[crystal_layer].register_forward_pre_hook(_inject_hook)
    try:
        model.eval()
        with torch_mod.inference_mode():
            try:
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            except TypeError:
                model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        if inject_handle is not None:
            inject_handle.remove()
        handle.remove()

    boundary = captured.get("tensor")
    if boundary is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to capture boundary for layer {crystal_layer}")
    if tracing.is_enabled("a1"):
        # A1 probe: final captured tensor + wall time.
        tracing.emit(
            "a1",
            "capture.done",
            {
                "stats": tracing.tensor_stats(boundary),
                "elapsed_sec": float(time.perf_counter() - _trace_t0),
            },
        )
    if return_stream:
        stream = captured.get("stream")
        if stream is None:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to capture residual stream for layer {crystal_layer}")
        return boundary, stream
    return boundary


def capture_window_boundaries(
    model: Any,
    windows: Sequence[Sequence[int]],
    *,
    crystal_layer: int,
    device: str | Any | None = None,
    return_streams: bool = False,
) -> tuple[dict[int, Any], Any | None] | tuple[dict[int, Any], Any | None, dict[int, Any]]:
    """Capture one post-crystal boundary per window."""

    from chuk_lazarus import tracing

    _trace_t0 = time.perf_counter()

    boundaries: dict[int, Any] = {}
    streams: dict[int, Any] = {}
    final_boundary = None
    running_boundary = None

    for wid, token_ids in enumerate(windows):
        captured = capture_post_crystal_boundary(
            model,
            token_ids,
            crystal_layer=crystal_layer,
            device=device,
            initial_residual=running_boundary,
            return_stream=return_streams,
        )
        if return_streams:
            boundary, stream = captured
            streams[wid] = stream
        else:
            boundary = captured
        boundaries[wid] = boundary
        final_boundary = boundary
        running_boundary = boundary

    if tracing.is_enabled("a1"):
        # A1 probe: summary across all windows.
        _final_stats = tracing.tensor_stats(final_boundary)
        tracing.emit(
            "a1",
            "capture_windows.summary",
            {
                "n_windows": int(len(boundaries)),
                "final_boundary_sha256": str(_final_stats.get("sha256", "")),
                "elapsed_sec": float(time.perf_counter() - _trace_t0),
            },
        )

    if return_streams:
        return boundaries, final_boundary, streams
    return boundaries, final_boundary


__all__ = ["capture_post_crystal_boundary", "capture_window_boundaries"]
