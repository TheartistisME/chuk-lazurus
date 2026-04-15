"""
Advantage estimation for policy gradient methods.

Dual-backend. For the loop-based GAE/returns accumulation we fall back to
numpy internally to avoid MLX-specific ``.at[...].add`` semantics that have
no direct torch equivalent; the result is converted back to the caller's
backend tensor type. This keeps torch/MLX parity within atol=1e-5, rtol=1e-4.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chuk_lazarus.training._backend_math import detect_backend, xp_for


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if type(x).__module__.startswith("torch"):
        return x.detach().cpu().numpy()
    # mlx
    return np.asarray(x)


def _from_numpy(x: np.ndarray, backend: str, like: Any) -> Any:
    if backend == "torch":
        import torch

        return torch.as_tensor(x, dtype=like.dtype, device=like.device)
    import mlx.core as mx

    return mx.array(x.astype(np.float32))


def compute_returns(rewards: Any, dones: Any, gamma: float = 0.99) -> Any:
    """Compute discounted returns (reward-to-go)."""
    bk = detect_backend(rewards)
    r = _to_numpy(rewards).astype(np.float32)
    d = _to_numpy(dones).astype(np.float32)
    batch_size, timesteps = r.shape
    out = np.zeros_like(r)
    running = np.zeros(batch_size, dtype=np.float32)
    for t in range(timesteps - 1, -1, -1):
        running = r[:, t] + gamma * running * (1 - d[:, t])
        out[:, t] = running
    return _from_numpy(out, bk, rewards)


def compute_gae(
    rewards: Any,
    values: Any,
    dones: Any,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[Any, Any]:
    """Compute Generalized Advantage Estimation (GAE)."""
    bk = detect_backend(rewards)
    r = _to_numpy(rewards).astype(np.float32)
    v = _to_numpy(values).astype(np.float32)
    d = _to_numpy(dones).astype(np.float32)
    batch_size, timesteps = r.shape
    adv = np.zeros_like(r)
    last_gae = np.zeros(batch_size, dtype=np.float32)
    last_value = np.zeros(batch_size, dtype=np.float32)
    for t in range(timesteps - 1, -1, -1):
        next_value = last_value if t == timesteps - 1 else v[:, t + 1]
        delta = r[:, t] + gamma * next_value * (1 - d[:, t]) - v[:, t]
        last_gae = delta + gamma * lam * (1 - d[:, t]) * last_gae
        adv[:, t] = last_gae
    returns = adv + v
    return (
        _from_numpy(adv, bk, rewards),
        _from_numpy(returns, bk, rewards),
    )


def normalize_advantages(advantages: Any, eps: float = 1e-8) -> Any:
    """Normalize advantages to zero mean, unit variance."""
    bk = detect_backend(advantages)
    xp = xp_for(bk)
    mean = xp.mean(advantages)
    std = xp.sqrt(xp.var(advantages) + eps)
    return (advantages - mean) / std
