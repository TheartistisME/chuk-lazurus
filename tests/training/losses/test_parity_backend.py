"""Torch vs MLX parity tests for EWS-9 losses.

Tolerances: atol=1e-5, rtol=1e-4 (per EWS-9 acceptance).

All tests skip gracefully if either framework is unavailable on the host.

Every test in this module is marked ``@pytest.mark.parity`` and invokes
``mlx_snapshot_assert`` so each comparison writes a residual entry under
``tests/_helpers/mlx_snapshots/residuals.json`` (see M-x1 /
ve-ins-0mo7014jk000083602f axis-3 baseline).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow direct execution with "pytest tests/..." without src-layout editable
# install side effects.  Resolves tests/_helpers import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _helpers.mlx_snapshots import mlx_snapshot_assert  # noqa: E402

mlx_core = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.parity

ATOL = 1e-5
RTOL = 1e-4

RNG = np.random.default_rng(0)


def _to_mlx(x):
    return mlx_core.array(np.ascontiguousarray(x).astype(np.float32))


def _to_torch(x):
    return torch.as_tensor(np.ascontiguousarray(x).astype(np.float32))


def _assert(op, mlx_val, torch_val, inputs=None, atol=ATOL, rtol=RTOL):
    mlx_snapshot_assert(
        f"training.losses.{op}",
        mlx_val,
        torch_val,
        atol=atol,
        rtol=rtol,
        inputs=inputs,
    )


@pytest.mark.parity
def test_sft_loss_parity():
    from chuk_lazarus.training.losses.sft_loss import sft_loss

    b, s, v = 2, 5, 7
    logits = RNG.standard_normal((b, s, v)).astype(np.float32)
    labels = RNG.integers(0, v, (b, s)).astype(np.int64)
    mask = (RNG.random((b, s)) > 0.2).astype(np.float32)

    lm, mm = sft_loss(_to_mlx(logits), mlx_core.array(labels.astype(np.int32)), _to_mlx(mask))
    lt, mt = sft_loss(_to_torch(logits), torch.as_tensor(labels), _to_torch(mask))
    _assert("sft_loss", lm, lt, inputs=[logits, labels, mask])
    _assert("sft_perplexity", mm["perplexity"], mt["perplexity"], inputs=[logits, labels, mask])


@pytest.mark.parity
def test_log_probs_parity():
    from chuk_lazarus.training.utils.log_probs import compute_log_probs_from_logits

    b, s, v = 2, 4, 6
    logits = RNG.standard_normal((b, s, v)).astype(np.float32)
    actions = RNG.integers(0, v, (b, s)).astype(np.int64)

    lp_m = compute_log_probs_from_logits(_to_mlx(logits), mlx_core.array(actions.astype(np.int32)))
    lp_t = compute_log_probs_from_logits(_to_torch(logits), torch.as_tensor(actions))
    _assert("log_probs_from_logits", lp_m, lp_t, inputs=[logits, actions])


@pytest.mark.parity
def test_kl_divergence_parity():
    from chuk_lazarus.training.utils.kl_divergence import (
        compute_approx_kl,
        compute_kl_divergence,
    )

    p = RNG.standard_normal((3, 5)).astype(np.float32)
    q = RNG.standard_normal((3, 5)).astype(np.float32)
    mask = (RNG.random((3, 5)) > 0.1).astype(np.float32)

    _assert(
        "kl_divergence",
        compute_kl_divergence(_to_mlx(p), _to_mlx(q), _to_mlx(mask)),
        compute_kl_divergence(_to_torch(p), _to_torch(q), _to_torch(mask)),
        inputs=[p, q, mask],
    )
    _assert(
        "approx_kl",
        compute_approx_kl(_to_mlx(p), _to_mlx(q), _to_mlx(mask)),
        compute_approx_kl(_to_torch(p), _to_torch(q), _to_torch(mask)),
        inputs=[p, q, mask],
    )


@pytest.mark.parity
def test_advantage_parity():
    from chuk_lazarus.training.utils.advantage import (
        compute_gae,
        compute_returns,
        normalize_advantages,
    )

    rewards = RNG.standard_normal((2, 6)).astype(np.float32)
    values = RNG.standard_normal((2, 6)).astype(np.float32)
    dones = (RNG.random((2, 6)) > 0.8).astype(np.float32)

    _assert(
        "compute_returns",
        compute_returns(_to_mlx(rewards), _to_mlx(dones)),
        compute_returns(_to_torch(rewards), _to_torch(dones)),
        inputs=[rewards, dones],
    )
    am, rm = compute_gae(_to_mlx(rewards), _to_mlx(values), _to_mlx(dones))
    at, rt = compute_gae(_to_torch(rewards), _to_torch(values), _to_torch(dones))
    _assert("compute_gae.advantages", am, at, inputs=[rewards, values, dones])
    _assert("compute_gae.returns", rm, rt, inputs=[rewards, values, dones])
    _assert(
        "normalize_advantages",
        normalize_advantages(_to_mlx(rewards.reshape(-1))),
        normalize_advantages(_to_torch(rewards.reshape(-1))),
        inputs=[rewards],
    )


@pytest.mark.parity
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
    _assert(
        "ppo_loss",
        lm,
        lt,
        inputs=[log_probs, old_log_probs, advantages, values, returns, entropy],
    )


@pytest.mark.parity
def test_grpo_loss_parity():
    from chuk_lazarus.training.losses.grpo_loss import grpo_loss

    gs = 4
    n = 2 * gs
    lp = RNG.standard_normal(n).astype(np.float32)
    rlp = lp + 0.01 * RNG.standard_normal(n).astype(np.float32)
    rw = RNG.standard_normal(n).astype(np.float32)

    lm, _ = grpo_loss(_to_mlx(lp), _to_mlx(rlp), _to_mlx(rw), gs)
    lt, _ = grpo_loss(_to_torch(lp), _to_torch(rlp), _to_torch(rw), gs)
    _assert("grpo_loss", lm, lt, inputs=[lp, rlp, rw, np.array(gs)])
