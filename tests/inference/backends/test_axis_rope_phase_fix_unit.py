"""Unit tests for archived-prefix RoPE phase dynamism (axis-rope-phase-fix run 2).

This module exercises ``Gemma4TextRotaryEmbedding.forward`` *directly* (not
through the patched attention forward) to lock in the invariant the
production fix at ``torch_runtime.py:1907-1919`` relies on:

    Calling the model's rotary module with ``position_ids = arange(N)`` must
    return per-position dynamic ``cos`` / ``sin`` tensors. If two distinct
    positions yielded identical cos/sin, the rotary itself would be broken
    and no production-side fix could re-phase archived K to its absolute
    positions {0..N-1}.

If this UNIT test PASSES but the integration parity test
``test_kv_materialisation_parity_layers_27_to_32`` still FAILS, the
production fix is broken (not the rotary). Conversely, if this test FAILS,
the rotary contract assumed by ``_prepare_archived_prefix`` does not hold
on the pinned snapshot and the fix design must be revisited.

Shape note (re bug-spec acceptance test (2)):
    The bug-spec brief reads ``cos shape (B, 1, N, head_dim)``, but the raw
    output of ``Gemma4TextRotaryEmbedding.forward`` (per the HF source) is
    ``(B, N, D)`` where ``D = 2 * inv_freq.shape[0]`` — the 4D shape only
    appears *inside* ``apply_rotary_pos_emb`` after the ``unsqueeze_dim``
    step. We assert the empirically correct 3D shape here and document the
    discrepancy.

    Layer-type asymmetry: on Gemma-4-E2B-it, ``sliding_attention`` rotary
    returns shape ``(B, N, 256)`` (matches ``head_dim`` coincidentally)
    while ``full_attention`` returns ``(B, N, 512)``
    (= ``2 * full_attention_inv_freq.shape[0]``). The shape assertion is
    computed from the rotary module's ``inv_freq``, not ``config.head_dim``.

GPU-only. CUDA bf16, RTX 5090. Real Gemma-4-E2B-it weights, pinned snapshot
``b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf``. No CPU fallback. No mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chuk_lazarus.inference.backends import TorchInferenceRuntime


GEMMA4_SNAPSHOT = Path(
    "/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/"
    "snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
)


requires_cuda_gemma4 = pytest.mark.skipif(
    not (torch.cuda.is_available() and GEMMA4_SNAPSHOT.is_dir()),
    reason="requires CUDA + local Gemma-4-E2B-it snapshot",
)


@pytest.fixture(scope="module")
def real_runtime():
    """Load Gemma-4-E2B-it once for the whole module (mirrors the parity suite)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(GEMMA4_SNAPSHOT))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(GEMMA4_SNAPSHOT), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    return TorchInferenceRuntime(model, tokenizer, device="cuda")


def _rotary_at_arange(real_runtime, n: int, layer_type: str):
    """Invoke the model's rotary module at ``arange(n)`` for ``layer_type``.

    Returns ``(cos, sin, head_dim, rotary_module)``. Both tensors live on
    CUDA in bf16. ``rotary_module`` is returned so callers can read the
    layer-type-specific ``inv_freq`` attribute to compute the expected
    last-dim (= ``2 * inv_freq.shape[0]``), which is *not* always equal to
    ``head_dim`` (see module docstring for layer-type asymmetry).
    """
    rotary_module = real_runtime._model.model.language_model.rotary_emb
    head_dim = int(real_runtime._model.config.get_text_config().head_dim)

    # ``apply_rotary_pos_emb`` only reads positional info from cos/sin, so the
    # hidden tensor handed to ``rotary_module`` is purely a device/dtype carrier.
    dummy_hidden = torch.zeros(1, n, 1, device="cuda", dtype=torch.bfloat16)
    position_ids = torch.arange(n, device="cuda", dtype=torch.long).unsqueeze(0)

    cos, sin = rotary_module(dummy_hidden, position_ids, layer_type=layer_type)
    return cos, sin, head_dim, rotary_module


@requires_cuda_gemma4
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_rotary_forward_returns_3d_shape(real_runtime, layer_type):
    """``Gemma4TextRotaryEmbedding.forward`` returns ``(B, N, 2 * inv_freq.shape[0])``.

    The bug-spec brief mis-states this as ``(B, 1, N, head_dim)``; that 4D
    shape is the *post-unsqueeze* form used inside ``apply_rotary_pos_emb``.
    The raw forward output is 3D with last-dim ``2 * inv_freq.shape[0]``
    (per ``emb = torch.cat((freqs, freqs), dim=-1)`` in the HF source).

    On Gemma-4-E2B-it this is *layer-type asymmetric*:
      - ``sliding_attention`` → ``(1, N, 256)`` (matches ``head_dim``
        coincidentally; ``sliding_attention_inv_freq.shape[0] == 128``).
      - ``full_attention``    → ``(1, N, 512)`` (does **not** match
        ``head_dim=256``; ``full_attention_inv_freq.shape[0] == 256``).
    The full 512-dim cos/sin is consumed by partial-RoPE slicing inside
    the real attention forward; for the rotary MODULE OUTPUT directly,
    we must compute the expected dim from the layer-type's ``inv_freq``,
    not from ``config.head_dim``.
    """
    n = 4
    cos, sin, head_dim, rotary_module = _rotary_at_arange(
        real_runtime, n, layer_type
    )

    expected_dim = 2 * int(
        getattr(rotary_module, f"{layer_type}_inv_freq").shape[0]
    )

    assert cos.ndim == 3, f"expected 3D cos (B, N, D), got ndim={cos.ndim}"
    assert sin.ndim == 3, f"expected 3D sin (B, N, D), got ndim={sin.ndim}"
    assert tuple(cos.shape) == (1, n, expected_dim), (
        f"cos shape {tuple(cos.shape)} != expected (1, {n}, {expected_dim}) "
        f"for layer_type={layer_type!r} "
        f"(head_dim={head_dim}, inv_freq.shape[0]={expected_dim // 2})"
    )
    assert tuple(sin.shape) == (1, n, expected_dim), (
        f"sin shape {tuple(sin.shape)} != expected (1, {n}, {expected_dim}) "
        f"for layer_type={layer_type!r} "
        f"(head_dim={head_dim}, inv_freq.shape[0]={expected_dim // 2})"
    )
    assert cos.device.type == "cuda" and sin.device.type == "cuda"


@requires_cuda_gemma4
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
@pytest.mark.parametrize("n_offset", [1, 2, 3])
def test_rotary_cos_per_position_dynamism(real_runtime, layer_type, n_offset):
    """``cos[:, n, :] != cos[:, 0, :]`` for every n>0, both layer types.

    This is the core invariant: if cos at position n equals cos at position
    0, the production fix's call to ``rotary_module(... arange(N) ...)`` is
    indistinguishable from the broken slice ``cos[:, 0:1, :].expand(...)``,
    and archived K would remain RoPE-identity.
    """
    n_total = 4
    cos, _, _, _ = _rotary_at_arange(real_runtime, n_total, layer_type)

    cos_zero = cos[:, 0, :]
    cos_n = cos[:, n_offset, :]
    assert not torch.allclose(cos_n, cos_zero, atol=1e-6), (
        f"cos at position {n_offset} matches cos at position 0 within 1e-6 "
        f"for layer_type={layer_type!r}; rotary is position-invariant."
    )


@requires_cuda_gemma4
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
@pytest.mark.parametrize("n_offset", [1, 2, 3])
def test_rotary_sin_per_position_dynamism(real_runtime, layer_type, n_offset):
    """``sin[:, n, :] != sin[:, 0, :]`` for every n>0, both layer types."""
    n_total = 4
    _, sin, _, _ = _rotary_at_arange(real_runtime, n_total, layer_type)

    sin_zero = sin[:, 0, :]
    sin_n = sin[:, n_offset, :]
    assert not torch.allclose(sin_n, sin_zero, atol=1e-6), (
        f"sin at position {n_offset} matches sin at position 0 within 1e-6 "
        f"for layer_type={layer_type!r}; rotary is position-invariant."
    )


@requires_cuda_gemma4
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_rotary_position_zero_is_identity(real_runtime, layer_type):
    """Sanity: at position 0, cos==1 and sin==0 (RoPE identity).

    This is the *correct* behaviour at position 0 only; the bug was applying
    this identity to positions {1..N-1} as well.
    """
    n = 4
    cos, sin, _, _ = _rotary_at_arange(real_runtime, n, layer_type)

    # Cast to fp32 for tolerance budgeting; bf16 has ~3 decimal digits.
    cos_zero = cos[:, 0, :].to(torch.float32)
    sin_zero = sin[:, 0, :].to(torch.float32)

    assert torch.allclose(
        cos_zero, torch.ones_like(cos_zero), atol=1e-2
    ), f"cos at position 0 is not ~1 for layer_type={layer_type!r}"
    assert torch.allclose(
        sin_zero, torch.zeros_like(sin_zero), atol=1e-2
    ), f"sin at position 0 is not ~0 for layer_type={layer_type!r}"
