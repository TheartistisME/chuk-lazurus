"""Qwen KV-direct evidence tests for the unified engine surface.

The host used for verification does not have ``libmlx.so`` available, so the
KV-direct algorithm is exercised with a small NumPy-backed MLX shim. That lets
the tests cover the real ``KVDirectGenerator`` execution path without
fabricating a green ``Qwen3`` auto-discovery path that does not exist yet.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from chuk_lazarus.inference.context import kv_generator as kv_generator_mod
from chuk_lazarus.inference.unified import EngineMode, UnifiedPipeline, UnifiedPipelineConfig
from chuk_lazarus.models_v2.families import FamilyInfo, ModelFamilyType


class _FakeMxFast:
    @staticmethod
    def scaled_dot_product_attention(q, k, v, scale, mask=None):
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
        if mask is not None:
            scores = scores + mask
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        return np.matmul(weights, v)


class _FakeMx:
    float32 = np.float32
    bfloat16 = np.float32
    fast = _FakeMxFast()

    @staticmethod
    def array(value, dtype=None):
        return np.array(value, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype=None):
        return np.zeros(shape, dtype=dtype or np.float32)

    @staticmethod
    def full(shape, value, dtype=None):
        return np.full(shape, value, dtype=dtype or np.float32)

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)

    @staticmethod
    def repeat(value, repeats, axis=0):
        return np.repeat(value, repeats, axis=axis)

    @staticmethod
    def triu(value, k=0):
        return np.triu(value, k=k)

    @staticmethod
    def eval(*_values):
        return None

    @staticmethod
    def compile(fn, shapeless=True):
        del shapeless
        return fn

    @staticmethod
    def softmax(value, axis=-1):
        shifted = value - np.max(value, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    @staticmethod
    def argmax(value, axis=-1):
        return np.argmax(value, axis=axis)

    @staticmethod
    def expand_dims(value, axis=-1):
        return np.expand_dims(value, axis=axis)

    @staticmethod
    def where(condition, x, y):
        return np.where(condition, x, y)

    @staticmethod
    def any(value):
        return np.any(value)

    @staticmethod
    def sum(value, axis=None):
        return np.sum(value, axis=axis)

    @staticmethod
    def matmul(lhs, rhs):
        return np.matmul(lhs, rhs)

    @staticmethod
    def sigmoid(value):
        return 1.0 / (1.0 + np.exp(-value))


class _Linear:
    def __init__(self, weight: np.ndarray):
        self.weight = np.asarray(weight, dtype=np.float32)

    def __call__(self, x):
        return np.matmul(x, self.weight)


class _Identity:
    def __call__(self, x):
        return np.asarray(x, dtype=np.float32)


class _FakeEmbedding:
    def __init__(self, vocab_size: int, hidden_size: int):
        rows = []
        for token_id in range(vocab_size):
            rows.append(np.full((hidden_size,), float(token_id + 1), dtype=np.float32))
        self.weight = np.stack(rows, axis=0)

    def __call__(self, input_ids):
        return self.weight[np.asarray(input_ids, dtype=np.int64)]


class _FakeLMHead:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, hidden_states):
        shape = hidden_states.shape[:-1] + (self.vocab_size,)
        logits = np.zeros(shape, dtype=np.float32)
        logits[..., 1] = 1.0
        return logits


class _FakeRope:
    def __call__(self, x, offset=0):
        del offset
        return x


class _FakeSelfAttention:
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.rope = _FakeRope()
        self.q_norm = _Identity()
        self.k_norm = _Identity()

        self.q_proj = _Linear(np.eye(hidden_size, dtype=np.float32))
        self.k_proj = _Linear(np.eye(hidden_size, num_kv_heads * head_dim, dtype=np.float32))
        self.v_proj = _Linear(np.eye(hidden_size, num_kv_heads * head_dim, dtype=np.float32))
        self.o_proj = _Linear(np.eye(hidden_size, dtype=np.float32))


class _FakeBlock:
    def __init__(self, hidden_size: int):
        self.input_layernorm = _Identity()
        self.post_attention_layernorm = _Identity()
        self.self_attn = _FakeSelfAttention(
            hidden_size=hidden_size,
            num_heads=4,
            num_kv_heads=1,
            head_dim=2,
        )

    def mlp(self, x):
        return np.zeros_like(x, dtype=np.float32)


class _FakeQwenBackbone:
    def __init__(self, num_layers: int, vocab_size: int, hidden_size: int):
        self.layers = [_FakeBlock(hidden_size) for _ in range(num_layers)]
        self.embed_tokens = _FakeEmbedding(vocab_size, hidden_size)
        self.norm = _Identity()


class Qwen3ForCausalLM:
    """Qwen-shaped model stub with the same surface that the adapter expects."""

    def __init__(self, num_layers: int = 2, vocab_size: int = 8, hidden_size: int = 8):
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=2,
            sliding_window=None,
            rope_theta=1.0,
        )
        self.model = _FakeQwenBackbone(num_layers=num_layers, vocab_size=vocab_size, hidden_size=hidden_size)
        self.lm_head = _FakeLMHead(vocab_size=vocab_size)


def _causal_mask(seq_len: int, dtype=np.float32):
    mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=dtype), k=1)
    return mask[None, None, :, :]


def _make_family_info() -> FamilyInfo:
    return FamilyInfo(
        family_type=ModelFamilyType.QWEN3,
        config_class=type("Qwen3Config", (), {}),
        model_class=Qwen3ForCausalLM,
        model_types=["qwen2", "qwen3"],
        architectures=["Qwen2ForCausalLM", "Qwen3ForCausalLM"],
    )


def test_qwen3_class_name_auto_dispatches_to_kv_direct_generator(monkeypatch):
    monkeypatch.setattr(kv_generator_mod, "mx", _FakeMx())

    model = Qwen3ForCausalLM()

    generator = kv_generator_mod.make_kv_generator(model)

    assert isinstance(generator, kv_generator_mod.KVDirectGenerator)
    assert type(generator.backbone).__name__ == "QwenBackboneAdapter"


def test_unified_pipeline_make_engine_returns_qwen_kv_direct_generator(monkeypatch):
    monkeypatch.setattr(kv_generator_mod, "mx", _FakeMx())

    model = Qwen3ForCausalLM()
    pipeline = UnifiedPipeline(
        model=model,
        tokenizer=SimpleNamespace(),
        model_config=SimpleNamespace(hidden_size=8, num_hidden_layers=2),
        family_info=_make_family_info(),
        pipeline_config=UnifiedPipelineConfig(engine=EngineMode.KV_DIRECT),
        runtime=MagicMock(),
    )

    engine = pipeline.make_engine()

    assert isinstance(engine, kv_generator_mod.KVDirectGenerator)
    assert type(engine.backbone).__name__ == "QwenBackboneAdapter"


def test_kv_direct_generator_prefill_step_slide_and_residual_capture_on_qwen_like_backbone(
    monkeypatch,
):
    monkeypatch.setattr(kv_generator_mod, "mx", _FakeMx())

    model = Qwen3ForCausalLM()
    generator = kv_generator_mod.make_kv_generator(model)
    generator.backbone.prefill_mask = lambda layer_idx, h: _causal_mask(h.shape[1], dtype=h.dtype)
    generator.backbone.is_global_layer = lambda _layer_idx: True
    input_ids = np.array([[1, 2, 3]], dtype=np.int64)

    logits, kv_store = generator.prefill(input_ids)
    logits_with_residual, kv_store_with_residual, residual = generator.prefill_with_residual(input_ids)

    assert logits.shape == (1, 3, 8)
    assert logits_with_residual.shape == (1, 3, 8)
    assert np.array_equal(logits, logits_with_residual)
    assert residual.shape == (1, 1, 8)
    assert len(kv_store) == 2
    assert kv_store[0][0].shape == (1, 1, 3, 2)
    assert kv_store[0][1].shape == (1, 1, 3, 2)
    assert int(np.argmax(logits[0, -1, :])) == 1

    next_logits, stepped = generator.step(np.array([[4]], dtype=np.int64), kv_store, seq_len=3)
    assert next_logits.shape == (1, 1, 8)
    assert stepped[0][0].shape == (1, 1, 4, 2)
    assert stepped[0][1].shape == (1, 1, 4, 2)

    slid = generator.slide(stepped, 2)
    assert slid[0][0].shape == (1, 1, 2, 2)
    assert slid[0][1].shape == (1, 1, 2, 2)

    extend_logits, extended = generator.extend(np.array([[5, 6]], dtype=np.int64), slid, abs_start=4)
    assert extend_logits.shape == (1, 2, 8)
    assert extended[0][0].shape == (1, 1, 4, 2)
    assert extended[0][1].shape == (1, 1, 4, 2)

    extend_with_residual_logits, extended_with_residual, residual_last = generator.extend_with_residual(
        np.array([[7]], dtype=np.int64),
        slid,
        abs_start=4,
    )
    assert extend_with_residual_logits.shape == (1, 1, 8)
    assert extended_with_residual[0][0].shape == (1, 1, 3, 2)
    assert residual_last.shape == (1, 1, 8)

    assert generator.kv_bytes(seq_len=10) == 160
    assert generator.residual_equivalent_bytes(seq_len=10) == 320
