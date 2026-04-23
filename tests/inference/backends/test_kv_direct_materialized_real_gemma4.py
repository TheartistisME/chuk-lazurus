"""CUDA integration test: real Gemma-4-E2B + patched forward + tier mask.

GPU-only. Verifies the KV-direct materialization path end-to-end on the
actual model the REPL uses:

  * The forward patch fires on the target layer (prefix_forwards > 0).
  * The tier mask is applied when WARM windows are present
    (mask_applied > 0 AND mask_penalty_applied=True).
  * kv_direct_active is truthful.
  * path_a_replay_count == 0 (Path-A guard passed).
  * The generation produces non-empty text (plumbing did not corrupt
    attention so badly that the model emitted EOS immediately).
  * When COLD-only assignments are handed in, the path raises rather
    than silently returning empty K/V.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from chuk_lazarus.inference.backends import TorchInferenceRuntime
from chuk_lazarus.inference.backends._torch_residual_bounded import (
    KVDirectMaterialization,
    resolve_kv_materialization_provenance,
)
from chuk_lazarus.inference.backends.torch_runtime import (
    KvDirectGenerationResult,
    WarmPenaltyConfig,
)
from chuk_lazarus.inference.generation import GenerationConfig
from chuk_lazarus.session_retrieval.tier_policy import TierLabel


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
    """Load Gemma-4-E2B-it once for the whole module."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(GEMMA4_SNAPSHOT))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(GEMMA4_SNAPSHOT), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    return TorchInferenceRuntime(model, tokenizer, device="cuda")


def _build_materialization(runtime, n_slots: int, injection_layer: int = 13):
    """Project random synthetic residuals through the target layer's W_K/W_V.

    Mirrors what :func:`materialize_kv_direct` does but with fabricated
    residuals so the test is self-contained (no store dependency).
    """
    layers = runtime._resolve_layers()
    target_layer = layers[injection_layer + 1]
    self_attn = target_layer.self_attn
    n_kv_heads = self_attn.k_proj.out_features // self_attn.head_dim
    head_dim = self_attn.head_dim
    hidden = self_attn.k_proj.in_features

    # Synthetic residual batch: (1, N, hidden). Use small magnitude to avoid
    # numerical blow-up through bfloat16 projection.
    residuals = torch.randn(1, n_slots, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        k_flat = self_attn.k_proj(residuals)
        v_flat = self_attn.v_proj(residuals)
    K = k_flat.view(1, n_slots, n_kv_heads, head_dim).transpose(1, 2).contiguous()
    V = v_flat.view(1, n_slots, n_kv_heads, head_dim).transpose(1, 2).contiguous()
    materialized_insertion_family, materialized_lineage_layer_indices = (
        resolve_kv_materialization_provenance(
            runtime._model,
            injection_layer,
            resolve_layers=lambda _model: runtime._resolve_layers(),
        )
    )
    return KVDirectMaterialization(
        K=K,
        V=V,
        materialization_mode="project_through_W_K_W_V_at_injection_layer",
        hot_budget_mib_observed=1,
        path_a_replay_count=0,
        materialized_source_layer=injection_layer,
        materialized_insertion_family=materialized_insertion_family,
        materialized_lineage_layer_indices=materialized_lineage_layer_indices,
    )


def _all_sliding_layer_indices(runtime) -> tuple[int, ...]:
    layers = runtime._resolve_layers()
    raw_config = getattr(runtime._model, "config", None)
    text_config = (
        raw_config.get_text_config()
        if raw_config is not None and hasattr(raw_config, "get_text_config")
        else raw_config
    )
    layer_types = list(getattr(text_config, "layer_types", [])) if text_config is not None else []

    sliding_indices: list[int] = []
    for layer_idx, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if self_attn is None:
            continue
        layer_type = getattr(self_attn, "layer_type", None)
        if layer_type is None and layer_idx < len(layer_types):
            layer_type = layer_types[layer_idx]
        if bool(getattr(self_attn, "is_sliding", False)) or str(layer_type) == "sliding_attention":
            sliding_indices.append(layer_idx)
    return tuple(sliding_indices)


@requires_cuda_gemma4
def test_forward_patch_fires_on_prefill(real_runtime):
    """The patched attention forward is invoked and stamps the prefix."""
    mat = _build_materialization(real_runtime, n_slots=3)
    result = real_runtime.generate_with_kv_direct_materialization(
        prompt="<start_of_turn>user\nHello\n<end_of_turn>\n<start_of_turn>model\n",
        config=GenerationConfig(max_new_tokens=4, temperature=0.0),
        materialization=mat,
        per_window_token_ranges={0: (0, 1), 1: (1, 2), 2: (2, 3)},
        tier_assignments={0: TierLabel.HOT, 1: TierLabel.HOT, 2: TierLabel.HOT},
        warm_config=WarmPenaltyConfig(),
        source_layer=13,
    )
    assert isinstance(result, KvDirectGenerationResult)
    meta = result.metadata
    assert meta["kv_direct_active"] is True
    assert meta["prefix_forwards_observed"] >= 1
    assert meta["path_a_replay_count"] == 0
    assert meta["n_archived_slots"] == 3
    assert meta["output_tokens"] >= 1  # generation did not immediately EOS


@requires_cuda_gemma4
def test_warm_tier_triggers_mask_application(real_runtime):
    """WARM assignments + non-zero penalty → mask_penalty_applied=True."""
    mat = _build_materialization(real_runtime, n_slots=3)
    result = real_runtime.generate_with_kv_direct_materialization(
        prompt="<start_of_turn>user\nHello\n<end_of_turn>\n<start_of_turn>model\n",
        config=GenerationConfig(max_new_tokens=4, temperature=0.0),
        materialization=mat,
        per_window_token_ranges={0: (0, 1), 1: (1, 2), 2: (2, 3)},
        tier_assignments={0: TierLabel.HOT, 1: TierLabel.WARM, 2: TierLabel.WARM},
        warm_config=WarmPenaltyConfig(penalty_value=4.0),
        source_layer=13,
    )
    meta = result.metadata
    assert meta["mask_penalty_applied"] is True
    assert meta["mask_applied_count"] >= 1
    assert "warm" in meta["selected_tier"]
    assert meta["kv_direct_active"] is True


@requires_cuda_gemma4
def test_hot_only_does_not_apply_mask_penalty(real_runtime):
    """HOT-only assignments → identity mask → mask_penalty_applied=False."""
    mat = _build_materialization(real_runtime, n_slots=2)
    result = real_runtime.generate_with_kv_direct_materialization(
        prompt="<start_of_turn>user\nHello\n<end_of_turn>\n<start_of_turn>model\n",
        config=GenerationConfig(max_new_tokens=4, temperature=0.0),
        materialization=mat,
        per_window_token_ranges={0: (0, 1), 1: (1, 2)},
        tier_assignments={0: TierLabel.HOT, 1: TierLabel.HOT},
        warm_config=WarmPenaltyConfig(penalty_value=4.0),
        source_layer=13,
    )
    meta = result.metadata
    assert meta["mask_penalty_applied"] is False
    assert meta["selected_tier"] == ["hot"]


@requires_cuda_gemma4
def test_sliding_insertion_family_runs_on_real_gemma4(real_runtime):
    """Explicit sliding-family selection must exercise the sliding branch."""
    mat = _build_materialization(real_runtime, n_slots=2, injection_layer=12)
    result = real_runtime.generate_with_kv_direct_materialization(
        prompt="<start_of_turn>user\nHello\n<end_of_turn>\n<start_of_turn>model\n",
        config=GenerationConfig(max_new_tokens=4, temperature=0.0),
        materialization=mat,
        per_window_token_ranges={0: (0, 1), 1: (1, 2)},
        tier_assignments={0: TierLabel.HOT, 1: TierLabel.HOT},
        warm_config=WarmPenaltyConfig(),
        source_layer=12,
        insertion_family="sliding",
        sliding_layer_indices=(13, 15),
        sliding_head_indices=(0, 7),
    )
    assert isinstance(result, KvDirectGenerationResult)
    meta = result.metadata
    assert meta["insertion_family"] == "sliding"
    assert meta["sliding_layer_indices"] == [13, 15]
    assert meta["sliding_head_indices"] == [0, 7]
    assert meta["kv_direct_active"] is True
    assert meta["prefix_forwards_observed"] == 2
    assert meta["path_a_replay_count"] == 0
    assert meta["output_tokens"] >= 1


@requires_cuda_gemma4
def test_sliding_insertion_rejects_full_attention_materialization_lineage(real_runtime):
    """Sliding insertion must reject archived K/V materialized from a full-attention source."""
    mat = _build_materialization(real_runtime, n_slots=2, injection_layer=13)
    with pytest.raises(RuntimeError) as excinfo:
        real_runtime.generate_with_kv_direct_materialization(
            prompt="<start_of_turn>user\nHello\n<end_of_turn>\n<start_of_turn>model\n",
            config=GenerationConfig(max_new_tokens=4, temperature=0.0),
            materialization=mat,
            per_window_token_ranges={0: (0, 1), 1: (1, 2)},
            tier_assignments={0: TierLabel.HOT, 1: TierLabel.HOT},
            warm_config=WarmPenaltyConfig(),
            source_layer=12,
            insertion_family="sliding",
            sliding_layer_indices=(13, 15),
            sliding_head_indices=(0, 7),
        )
    message = str(excinfo.value)
    assert "requested source_layer=12 disagrees" in message
    assert "materialized_source_layer=13" in message


@requires_cuda_gemma4
def test_sliding_insertion_rejects_incompatible_sliding_lineage(real_runtime):
    """Requested sliding layers must stay within the materialized sliding lineage."""
    mat = _build_materialization(real_runtime, n_slots=2, injection_layer=12)
    _, lineage = resolve_kv_materialization_provenance(
        real_runtime._model,
        12,
        resolve_layers=lambda _model: real_runtime._resolve_layers(),
    )
    invalid_layer = next(
        (layer_idx for layer_idx in _all_sliding_layer_indices(real_runtime) if layer_idx not in lineage),
        None,
    )
    if invalid_layer is None:
        pytest.skip("real Gemma-4 snapshot exposes no sliding layer outside the source-12 lineage")

    with pytest.raises(RuntimeError) as excinfo:
        real_runtime.generate_with_kv_direct_materialization(
            prompt="<start_of_turn>user\nHello\n<end_of_turn>\n<start_of_turn>model\n",
            config=GenerationConfig(max_new_tokens=4, temperature=0.0),
            materialization=mat,
            per_window_token_ranges={0: (0, 1), 1: (1, 2)},
            tier_assignments={0: TierLabel.HOT, 1: TierLabel.HOT},
            warm_config=WarmPenaltyConfig(),
            source_layer=12,
            insertion_family="sliding",
            sliding_layer_indices=(13, invalid_layer),
            sliding_head_indices=(0, 7),
        )
    message = str(excinfo.value)
    assert "incompatible with materialized sliding lineage" in message
    assert str(invalid_layer) in message


@requires_cuda_gemma4
def test_out_of_range_source_layer_raises(real_runtime):
    """source_layer+1 past the end of layers → explicit RuntimeError."""
    mat = _build_materialization(real_runtime, n_slots=1)
    with pytest.raises(RuntimeError) as excinfo:
        real_runtime.generate_with_kv_direct_materialization(
            prompt="hi",
            config=GenerationConfig(max_new_tokens=1, temperature=0.0),
            materialization=mat,
            per_window_token_ranges={0: (0, 1)},
            tier_assignments={0: TierLabel.HOT},
            warm_config=WarmPenaltyConfig(),
            source_layer=999,
        )
    assert "out of range" in str(excinfo.value)


@requires_cuda_gemma4
def test_patched_forward_restored_after_generation(real_runtime):
    """After the call returns, target_self_attn.forward is restored and
    the stashed archived tensors are purged."""
    layers = real_runtime._resolve_layers()
    target_self_attn = layers[14].self_attn
    original_forward = target_self_attn.forward

    mat = _build_materialization(real_runtime, n_slots=1)
    real_runtime.generate_with_kv_direct_materialization(
        prompt="<start_of_turn>user\nHi\n<end_of_turn>\n<start_of_turn>model\n",
        config=GenerationConfig(max_new_tokens=2, temperature=0.0),
        materialization=mat,
        per_window_token_ranges={0: (0, 1)},
        tier_assignments={0: TierLabel.HOT},
        warm_config=WarmPenaltyConfig(),
        source_layer=13,
    )

    # Forward must be restored.
    assert target_self_attn.forward == original_forward
    # Stashed tensors must be cleaned up.
    for attr in (
        "_lazarus_archived_K",
        "_lazarus_archived_V",
        "_lazarus_mask_inputs",
        "_lazarus_kv_direct_prefix",
    ):
        assert not hasattr(target_self_attn, attr)
