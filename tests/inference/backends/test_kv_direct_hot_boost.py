"""Focused HOT-tier additive-boost coverage for KV-direct attention."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.backends.torch_runtime import (
    AttentionTierMaskInputs,
    HotBoostConfig,
    WarmPenaltyConfig,
    apply_tier_attention_mask,
)
from chuk_lazarus.inference.generation import GenerationConfig
from chuk_lazarus.session_retrieval.tier_policy import TierLabel
from tests.inference.backends import test_kv_direct_materialized_real_gemma4 as _shared

_build_materialization = _shared._build_materialization
real_runtime = _shared.real_runtime
requires_cuda_gemma4 = _shared.requires_cuda_gemma4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test requires a CUDA GPU")
def test_hot_boost_zero_is_identity() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    baseline = torch.randn((1, 2, 4, 8), device="cuda", dtype=torch.bfloat16)

    inputs = AttentionTierMaskInputs(
        per_window_token_ranges={0: (0, 4), 1: (4, 8)},
        tier_assignments={0: TierLabel.HOT, 1: TierLabel.HOT},
        warm_config=WarmPenaltyConfig(),
        hot_config=HotBoostConfig(boost_value=0.0),
    )

    masked = apply_tier_attention_mask(baseline, inputs)

    assert torch.equal(masked, baseline)
    assert masked is not baseline


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test requires a CUDA GPU")
def test_hot_boost_applies_additive_offset() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    baseline = torch.randn((1, 1, 3, 8), device="cuda", dtype=torch.bfloat16)

    inputs = AttentionTierMaskInputs(
        per_window_token_ranges={0: (0, 4), 1: (4, 8)},
        tier_assignments={0: TierLabel.HOT, 1: TierLabel.HOT},
        warm_config=WarmPenaltyConfig(),
        hot_config=HotBoostConfig(boost_value=2.0),
    )

    masked = apply_tier_attention_mask(baseline, inputs)
    expected = baseline + 2.0

    torch.testing.assert_close(masked, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test requires a CUDA GPU")
def test_hot_boost_does_not_touch_warm_or_cold() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    baseline = torch.randn((1, 1, 2, 9), device="cuda")

    inputs = AttentionTierMaskInputs(
        per_window_token_ranges={0: (0, 3), 1: (3, 6), 2: (6, 9)},
        tier_assignments={0: TierLabel.HOT, 1: TierLabel.WARM, 2: TierLabel.COLD},
        warm_config=WarmPenaltyConfig(penalty_value=4.0),
        hot_config=HotBoostConfig(boost_value=5.0),
    )

    masked = apply_tier_attention_mask(baseline, inputs)

    torch.testing.assert_close(masked[..., 0:3], baseline[..., 0:3] + 5.0)
    torch.testing.assert_close(masked[..., 3:6], baseline[..., 3:6] - 4.0)
    assert torch.isneginf(masked[..., 6:9]).all()


@requires_cuda_gemma4
def test_metadata_surfaces_hot_boost(real_runtime) -> None:
    materialization = _build_materialization(real_runtime, n_slots=2)

    result = real_runtime.generate_with_kv_direct_materialization(
        prompt="<start_of_turn>user\nHello\n<end_of_turn>\n<start_of_turn>model\n",
        config=GenerationConfig(max_new_tokens=2, temperature=0.0),
        materialization=materialization,
        per_window_token_ranges={0: (0, 1), 1: (1, 2)},
        tier_assignments={0: TierLabel.HOT, 1: TierLabel.HOT},
        warm_config=WarmPenaltyConfig(),
        hot_config=HotBoostConfig(boost_value=2.0),
        source_layer=13,
    )

    assert result.metadata["hot_boost_applied"] is True
    assert result.metadata["hot_boost_value"] == pytest.approx(2.0)
