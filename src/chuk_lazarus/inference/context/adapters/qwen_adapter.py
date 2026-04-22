"""
Qwen adapter implementing TransformerLayerProtocol / ModelBackboneProtocol.

Wraps Qwen3ForCausalLM (and its blocks) without modifying any existing model
code.

Qwen-specific details handled here:
- 2 RMSNorm layers per block with Llama-like residual adds
- per-head q_norm / k_norm before RoPE
- no embedding scale
- no sliding-window attention in the current Qwen3 family

Top-level ``mlx`` imports are intentionally absent. Every method that touches
``mlx.core`` / ``mlx.nn`` imports it lazily inside the function body so that
``import chuk_lazarus.inference.context.adapters.qwen_adapter`` succeeds under
``CHUK_BACKEND=torch`` without pulling ``libmlx.so``. The adapter algebra is
unchanged -- every call site still runs identical MLX ops when MLX is
available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .llama_adapter import LlamaBackboneAdapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401


class QwenLayerAdapter:
    """
    Adapts a single Qwen3Block to TransformerLayerProtocol.

    Qwen3Block norms:
        input_layernorm          -> pre_attn_norm
        post_attention_layernorm -> pre_ffn_norm (applied to full residual)

    Residual adds are plain addition like Llama, but attention applies
    per-head q_norm / k_norm before RoPE.
    """

    __slots__ = ("_block",)

    def __init__(self, block) -> None:
        self._block = block

    # --- Attention ---

    def pre_attn_norm(self, h: Any) -> Any:
        return self._block.input_layernorm(h)

    def project_qkv(
        self, x: Any, B: int, S: int, offset: int
    ) -> tuple[Any, Any, Any]:
        attn = self._block.self_attn
        nq = attn.num_heads
        nkv = attn.num_kv_heads
        dh = attn.head_dim

        q = attn.q_proj(x).reshape(B, S, nq, dh)
        k = attn.k_proj(x).reshape(B, S, nkv, dh)
        v = attn.v_proj(x).reshape(B, S, nkv, dh)

        q = attn.q_norm(q)
        k = attn.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if attn.rope is not None:
            q = attn.rope(q, offset=offset)
            k = attn.rope(k, offset=offset)
        return q, k, v

    def project_qkv_pre_rope(
        self, x: Any, B: int, S: int
    ) -> tuple[Any, Any, Any]:
        """Project Q, K, V with norms but WITHOUT RoPE."""
        attn = self._block.self_attn
        nq = attn.num_heads
        nkv = attn.num_kv_heads
        dh = attn.head_dim

        q = attn.q_proj(x).reshape(B, S, nq, dh)
        k = attn.k_proj(x).reshape(B, S, nkv, dh)
        v = attn.v_proj(x).reshape(B, S, nkv, dh)

        q = attn.q_norm(q)
        k = attn.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        return q, k, v

    def apply_rope(self, x: Any, offset: int) -> Any:
        """Apply RoPE to pre-RoPE Q or K at the desired position offset."""
        attn = self._block.self_attn
        if attn.rope is not None:
            return attn.rope(x, offset=offset)
        return x

    def head_output_projection(self, head_out: Any, head_idx: int) -> Any:
        import mlx.core as mx

        o_weight = self._block.self_attn.o_proj.weight  # (D, nq*dh)
        dh = self._block.self_attn.head_dim
        return mx.matmul(head_out, o_weight[:, head_idx * dh : (head_idx + 1) * dh].T)

    def output_project(self, attn_result: Any) -> Any:
        return self._block.self_attn.o_proj(attn_result)

    def residual_add_attn(self, h: Any, attn_out: Any) -> Any:
        return h + attn_out

    # --- FFN ---

    def pre_ffn_norm(self, h: Any) -> Any:
        return self._block.post_attention_layernorm(h)

    def ffn(self, x: Any) -> Any:
        return self._block.mlp(x)

    def residual_add_ffn(self, h: Any, ffn_out: Any) -> Any:
        return h + ffn_out

    # --- Dimensions ---

    @property
    def num_heads(self) -> int:
        return self._block.self_attn.num_heads

    @property
    def num_kv_heads(self) -> int:
        return self._block.self_attn.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self._block.self_attn.head_dim

    @property
    def n_rep(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def attn_scale(self) -> float:
        return self._block.self_attn.scale


class QwenBackboneAdapter(LlamaBackboneAdapter):
    """
    Adapts a Qwen3ForCausalLM to ModelBackboneProtocol.

    Qwen3 shares Llama's backbone-level semantics (embedding, masking, final
    norm, plain residual adds), but its attention blocks require q_norm /
    k_norm handling before RoPE.
    """

    def __init__(self, causal_lm) -> None:
        self._model = causal_lm
        self._backbone = causal_lm.model

        self._adapted: list[QwenLayerAdapter] = [
            QwenLayerAdapter(block) for block in self._backbone.layers
        ]

    @property
    def adapted_layers(self) -> list[QwenLayerAdapter]:
        return self._adapted


__all__ = ["QwenBackboneAdapter", "QwenLayerAdapter"]
