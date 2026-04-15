"""Torch vs MLX parity tests for EWS-9 losses.

Tolerances: atol=1e-5, rtol=1e-4 (per EWS-9 acceptance).

All tests skip gracefully if either framework is unavailable on the host.
"""

from __future__ import annotations

import numpy as np
import pytest

mlx_core = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

ATOL = 1e-5
RTOL = 1e-4

RNG = np.random.default_rng(0)


def _to_mlx(x):
    return mlx_core.array(np.ascontiguousarray(x).astype(np.float32))


def _to_torch(x):
    return torch.as_tensor(np.ascontiguousarray(x).astype(np.float32))


def _np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _close(a, b, label: str) -> None:
    np.testing.assert_allclose(_np(a), _np(b), atol=ATOL, rtol=RTOL, err_msg=label)


def test_sft_loss_parity():
    from chuk_lazarus.training.losses.sft_loss import sft_loss

    b, s, v = 2, 5, 7
    logits = RNG.standard_normal((b, s, v)).astype(np.float32)
    labels = RNG.integers(0, v, (b, s)).astype(np.int64)
    mask = (RNG.random((b, s)) > 0.2).astype(np.float32)

    lm, mm = sft_loss(_to_mlx(logits), mlx_core.array(labels.astype(np.int32)), _to_mlx(mask))
    lt, mt = sft_loss(_to_torch(logits), torch.as_tensor(labels), _to_torch(mask))
    _close(lm, lt, "sft loss")
    _close(mm["perplexity"], mt["perplexity"], "sft perplexity")


def test_log_probs_parity():
    from chuk_lazarus.training.utils.log_probs import compute_log_probs_from_logits

    b, s, v = 2, 4, 6
    logits = RNG.standard_normal((b, s, v)).astype(np.float32)
    actions = RNG.integers(0, v, (b, s)).astype(np.int64)

    lp_m = compute_log_probs_from_logits(_to_mlx(logits), mlx_core.array(actions.astype(np.int32)))
    lp_t = compute_log_probs_from_logits(_to_torch(logits), torch.as_tensor(actions))
    _close(lp_m, lp_t, "log_probs")


def test_kl_divergence_parity():
    from chuk_lazarus.training.utils.kl_divergence import (
        compute_approx_kl,
        compute_kl_divergence,
    )

    p = RNG.standard_normal((3, 5)).astype(np.float32)
    q = RNG.standard_normal((3, 5)).astype(np.float32)
    mask = (RNG.random((3, 5)) > 0.1).astype(np.float32)

    _close(
        compute_kl_divergence(_to_mlx(p), _to_mlx(q), _to_mlx(mask)),
        compute_kl_divergence(_to_torch(p), _to_torch(q), _to_torch(mask)),
        "kl",
    )
    _close(
        compute_approx_kl(_to_mlx(p), _to_mlx(q), _to_mlx(mask)),
        compute_approx_kl(_to_torch(p), _to_torch(q), _to_torch(mask)),
        "approx_kl",
    )


def test_advantage_parity():
    from chuk_lazarus.training.utils.advantage import (
        compute_gae,
        compute_returns,
        normalize_advantages,
    )

    rewards = RNG.standard_normal((2, 6)).astype(np.float32)
    values = RNG.standard_normal((2, 6)).astype(np.float32)
    dones = (RNG.random((2, 6)) > 0.8).astype(np.float32)

    _close(
        compute_returns(_to_mlx(rewards), _to_mlx(dones)),
        compute_returns(_to_torch(rewards), _to_torch(dones)),
        "returns",
    )
    am, rm = compute_gae(_to_mlx(rewards), _to_mlx(values), _to_mlx(dones))
    at, rt = compute_gae(_to_torch(rewards), _to_torch(values), _to_torch(dones))
    _close(am, at, "gae adv")
    _close(rm, rt, "gae ret")
    _close(
        normalize_advantages(_to_mlx(rewards.reshape(-1))),
        normalize_advantages(_to_torch(rewards.reshape(-1))),
        "norm_adv",
    )


def test_ppo_loss_parity():
    from chuk_lazarus.training.losses.ppo_loss import ppo_loss

    n = 8
    log_probs = RNG.standard_normal(n).astype(np.float32)
    old_log_probs = log_probs + 0.01 * RNG.standard_normal(n).astype(np.float32)
    advantages = RNG.standard_normal(n).astype(np.float32)
    values = RNG.standard_normal(n).astype(np.float32)
    returns = values + 0.1 * RNG.standard_normal(n).astype(np.float32)
    entropy = np.abs(RNG.standard_normal(n)).astype(np.float32)

    lm, _ = ppo_loss(
        _to_mlx(log_probs), _to_mlx(old_log_probs), _to_mlx(advantages),
        _to_mlx(values), _to_mlx(returns), _to_mlx(entropy),
    )
    lt, _ = ppo_loss(
        _to_torch(log_probs), _to_torch(old_log_probs), _to_torch(advantages),
        _to_torch(values), _to_torch(returns), _to_torch(entropy),
    )
    _close(lm, lt, "ppo loss")


def test_grpo_loss_parity():
    from chuk_lazarus.training.losses.grpo_loss import grpo_loss

    gs = 4
    n = 2 * gs
    lp = RNG.standard_normal(n).astype(np.float32)
    rlp = lp + 0.01 * RNG.standard_normal(n).astype(np.float32)
    rw = RNG.standard_normal(n).astype(np.float32)

    lm, _ = grpo_loss(_to_mlx(lp), _to_mlx(rlp), _to_mlx(rw), gs)
    lt, _ = grpo_loss(_to_torch(lp), _to_torch(rlp), _to_torch(rw), gs)
    _close(lm, lt, "grpo loss")
