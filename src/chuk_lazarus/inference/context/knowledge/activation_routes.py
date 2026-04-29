"""Activation-similarity helpers for native mean-pooled routing vectors."""

from __future__ import annotations

from typing import Any

try:  # Keep imports safe on hosts where torch is unavailable.
    import torch
except Exception:  # pragma: no cover - exercised only on torchless hosts
    torch = None  # type: ignore[assignment]


def _require_torch():
    if torch is None:  # pragma: no cover - local safety net
        raise RuntimeError("Torch is required for activation route utilities")
    return torch


def mean_pool_hidden(
    hidden_states: Any,
    attention_mask: Any,
    *,
    normalize: bool = True,
    out_dtype: Any | None = None,
):
    """Mean-pool hidden states over non-padding tokens and return ``[B, H]``.

    ``hidden_states`` may be ``[B, S, H]`` or ``[S, H]``. ``attention_mask``
    may be ``[B, S]`` or ``[S]`` and is treated as truthy for kept tokens.
    """

    torch_mod = _require_torch()
    if out_dtype is None:
        out_dtype = torch_mod.float16

    hidden = hidden_states if torch_mod.is_tensor(hidden_states) else torch_mod.as_tensor(hidden_states)
    mask = attention_mask if torch_mod.is_tensor(attention_mask) else torch_mod.as_tensor(attention_mask)

    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(0)
    elif hidden.ndim != 3:
        raise ValueError(f"hidden_states must have shape [B,S,H] or [S,H], got rank {hidden.ndim}")

    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    elif mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape [B,S] or [S], got rank {mask.ndim}")

    if int(mask.shape[0]) != int(hidden.shape[0]) or int(mask.shape[1]) != int(hidden.shape[1]):
        raise ValueError(
            "attention_mask shape must match hidden_states batch/sequence dimensions: "
            f"mask={tuple(mask.shape)} hidden={tuple(hidden.shape)}"
        )

    work = hidden.to(dtype=torch_mod.float32)
    mask_f = mask.to(device=work.device, dtype=torch_mod.float32).unsqueeze(-1)
    pooled = (work * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    pooled = pooled / denom

    if normalize:
        pooled = torch_mod.nn.functional.normalize(pooled, p=2.0, dim=-1, eps=1e-12)

    return pooled.to(dtype=out_dtype)


def activation_topk(query_vec: Any, window_matrix: Any, top_k: int):
    """Return sorted top-k ``(indices, values)`` using cosine similarity."""

    torch_mod = _require_torch()
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    query = query_vec if torch_mod.is_tensor(query_vec) else torch_mod.as_tensor(query_vec)
    matrix = window_matrix if torch_mod.is_tensor(window_matrix) else torch_mod.as_tensor(window_matrix)

    if query.ndim == 2 and int(query.shape[0]) == 1:
        query = query.squeeze(0)
    if query.ndim != 1:
        raise ValueError(f"query_vec must have shape [H] or [1,H], got {tuple(query.shape)}")
    if matrix.ndim != 2:
        raise ValueError(f"window_matrix must have shape [N,H], got {tuple(matrix.shape)}")
    if int(query.shape[0]) != int(matrix.shape[1]):
        raise ValueError(
            "query_vec hidden dimension must match window_matrix columns: "
            f"query={int(query.shape[0])} matrix={int(matrix.shape[1])}"
        )

    query_f = torch_mod.nn.functional.normalize(
        query.to(dtype=torch_mod.float32),
        p=2.0,
        dim=-1,
        eps=1e-12,
    )
    matrix_f = torch_mod.nn.functional.normalize(
        matrix.to(dtype=torch_mod.float32),
        p=2.0,
        dim=-1,
        eps=1e-12,
    )
    scores = torch_mod.matmul(matrix_f, query_f)
    k = min(int(top_k), int(scores.shape[0]))
    values, indices = torch_mod.topk(scores, k=k, largest=True, sorted=True)
    return indices, values


__all__ = ["activation_topk", "mean_pool_hidden"]
