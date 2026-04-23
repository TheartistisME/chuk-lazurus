"""
Qwen-specific torch loader helpers.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def compressed_tensors_quant_method(config: Any) -> str | None:
    """Best-effort extraction of a Hugging Face quant_method label."""
    quant_config = getattr(config, "quantization_config", None)
    if quant_config is None:
        return None
    if isinstance(quant_config, dict):
        quant_method = quant_config.get("quant_method")
    else:
        quant_method = getattr(quant_config, "quant_method", None)
    return str(quant_method) if quant_method is not None else None


def install_qwen3_5_moe_per_expert_patch() -> str:
    """Patch Qwen3.5-MoE sparse blocks to match the NVFP4 checkpoint layout."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen_mod

    if getattr(qwen_mod, "_QWEN35_PER_EXPERT_PATCHED", False):
        return "already_patched"

    mlp_cls = qwen_mod.Qwen3_5MoeMLP
    sparse_block_cls = qwen_mod.Qwen3_5MoeSparseMoeBlock
    original_block_init = sparse_block_cls.__init__

    class PerExpertQwen3_5MoeExperts(nn.ModuleList):
        """Per-expert MLP modules matching the checkpoint tensor names exactly."""

        def __init__(self, config):
            super().__init__(
                [
                    mlp_cls(config, intermediate_size=config.moe_intermediate_size)
                    for _ in range(config.num_experts)
                ]
            )
            self.config = config
            self.num_experts = config.num_experts

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(
                    top_k_index, num_classes=self.num_experts
                ).permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                expert_mlp = self[int(expert_idx)]
                current_hidden_states = expert_mlp(current_state)
                current_hidden_states = current_hidden_states * top_k_weights[
                    token_idx, top_k_pos, None
                ]
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    current_hidden_states.to(final_hidden_states.dtype),
                )
            return final_hidden_states

    def patched_block_init(self, config):
        nn.Module.__init__(self)
        self.gate = qwen_mod.Qwen3_5MoeTopKRouter(config)
        self.experts = PerExpertQwen3_5MoeExperts(config)
        self.shared_expert = qwen_mod.Qwen3_5MoeMLP(
            config,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    sparse_block_cls.__init__ = patched_block_init
    qwen_mod._QWEN35_PER_EXPERT_PATCHED = True
    qwen_mod._QWEN35_ORIGINAL_SPARSE_INIT = original_block_init
    return "installed:per_expert_module_list_via_sparse_block_init_override"


def install_blackwell_causal_conv1d_workaround() -> str:
    """Disable causal-conv1d on Blackwell consumer GPUs while keeping FLA alive."""
    if not torch.cuda.is_available():
        return "skipped:no_cuda"

    capability = torch.cuda.get_device_capability(0)
    if capability < (12, 0):
        return f"skipped:sm_{capability[0]}{capability[1]}"

    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen_mod

    qwen_mod.causal_conv1d_fn = None
    qwen_mod.causal_conv1d_update = None
    if (
        qwen_mod.chunk_gated_delta_rule is not None
        and qwen_mod.fused_recurrent_gated_delta_rule is not None
    ):
        qwen_mod.is_fast_path_available = True
        return "installed:disable_causal_conv1d_on_sm120_keep_fla"

    qwen_mod.is_fast_path_available = False
    return "installed:disable_causal_conv1d_on_sm120"


__all__ = [
    "compressed_tensors_quant_method",
    "install_blackwell_causal_conv1d_workaround",
    "install_qwen3_5_moe_per_expert_patch",
]
