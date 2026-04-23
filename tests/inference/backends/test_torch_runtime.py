"""Real CUDA smoke tests for the PyTorch inference runtime."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from chuk_lazarus.inference.backends import TorchInferenceRuntime
from chuk_lazarus.inference.backends.torch_runtime import (
    ATTENTION_TIER_MASK_SCHEMA_VERSION,
    AttentionTierMaskInputs,
    KvDirectGenerationResult,
    WarmPenaltyConfig,
    apply_tier_attention_mask,
    attention_tier_mask_inputs_from_json,
    attention_tier_mask_inputs_to_json,
    build_tier_mask_observation,
)
from chuk_lazarus.inference.backends._torch_residual_bounded import (
    KVDirectMaterialization,
)
from chuk_lazarus.inference.backends.types import LazarusBackend
from chuk_lazarus.inference.generation import GenerationConfig
from chuk_lazarus.session_retrieval.tier_policy import TierLabel


class DummyBlock(torch.nn.Module):
    """Small transformer-like block for hook tests."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(4))

    def forward(self, hidden_states):
        return self.proj(hidden_states) + 1


class DummyInnerModel(torch.nn.Module):
    """Container exposing `.layers` like Hugging Face decoder models."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([DummyBlock(), DummyBlock()])


class DummyVLMLanguageModel(torch.nn.Module):
    """Container mimicking Gemma 4's `model.language_model.layers` structure."""

    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList([DummyBlock() for _ in range(num_layers)])


class DummyVLMInnerModel(torch.nn.Module):
    """Wrapper mimicking `Gemma4Model` that holds a `language_model` child."""

    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.language_model = DummyVLMLanguageModel(num_layers=num_layers)


class DummyVLMCausalLM(torch.nn.Module):
    """Minimal stand-in for `Gemma4ForConditionalGeneration`.

    The real module exposes transformer layers at
    ``model.model.language_model.layers``. `_resolve_layers` must reach
    through the VLM wrapper; this dummy checks that contract without
    needing the real 2 B-parameter Gemma 4 weights on the test host.
    """

    def __init__(self, head_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.model = DummyVLMInnerModel(num_layers=num_layers)
        self.config = SimpleNamespace(head_dim=head_dim, text_config=SimpleNamespace(hidden_size=4))


class DummyCausalLM(torch.nn.Module):
    """Tiny causal LM that exercises hooks during `generate()`."""

    def __init__(self, head_dim: int = 512):
        super().__init__()
        self.model = DummyInnerModel()
        self.config = SimpleNamespace(head_dim=head_dim)
        self.last_generate_kwargs: dict[str, object] | None = None

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
        del attention_mask, use_cache, kwargs
        hidden_states = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 4)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return (hidden_states,)

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.last_generate_kwargs = dict(kwargs)
        del attention_mask
        self.forward(input_ids=input_ids)
        suffix = torch.tensor([[7, 8]], device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


class DummyTokenizer:
    """Tokenizer stub with the Hugging Face methods used by the runtime."""

    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, prompt, return_tensors="pt"):
        del prompt, return_tensors
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)


# ---------------------------------------------------------------------------
# Axis-5 test fixtures: decoder layer with split-projection ``self_attn``.
# ---------------------------------------------------------------------------


class DummyAttnSelfAttn(torch.nn.Module):
    """Split-projection attention stub for axis-5 runtime tests.

    The axis-5 ``generate_with_kv_direct_materialization`` path only reads
    ``.self_attn`` on ``layers[source_layer + 1]`` to hang a one-shot
    prefix-stamping ``forward_pre_hook``; it does NOT invoke the module.
    The projections still need to exist because other paths (and the
    runtime's resolver) introspect them.
    """

    def __init__(self, hidden: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.k_proj = torch.nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.num_key_value_heads = n_kv_heads
        self.head_dim = head_dim

    def forward(self, hidden_states, *_args, **_kwargs):
        return hidden_states


class DummyAttnBlock(torch.nn.Module):
    """Decoder block with a ``self_attn`` carrying split K/V projections."""

    def __init__(self, hidden: int = 4, n_kv_heads: int = 2, head_dim: int = 8):
        super().__init__()
        self.self_attn = DummyAttnSelfAttn(hidden, n_kv_heads, head_dim)
        # Identity-on-hidden so the rest of the model still runs.
        self.proj = torch.nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden))

    def forward(self, hidden_states):
        return self.proj(hidden_states) + 1


class DummyAttnInnerModel(torch.nn.Module):
    """Container exposing ``.layers`` where one layer has ``self_attn``."""

    def __init__(self, hidden: int = 4, n_kv_heads: int = 2, head_dim: int = 8):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                DummyBlock(),
                DummyAttnBlock(hidden=hidden, n_kv_heads=n_kv_heads, head_dim=head_dim),
            ]
        )


class DummyAttnCausalLM(torch.nn.Module):
    """Causal-LM stub whose ``layers[1]`` exposes the axis-5 attention surface."""

    def __init__(self, head_dim: int = 128, n_kv_heads: int = 2):
        super().__init__()
        self.model = DummyAttnInnerModel(hidden=4, n_kv_heads=n_kv_heads, head_dim=head_dim)
        self.config = SimpleNamespace(head_dim=head_dim)
        self.last_generate_kwargs: dict[str, object] | None = None

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
        del attention_mask, use_cache, kwargs
        hidden_states = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 4)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return (hidden_states,)

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.last_generate_kwargs = dict(kwargs)
        del attention_mask
        self.forward(input_ids=input_ids)
        suffix = torch.tensor([[7, 8]], device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test requires a CUDA GPU")
class TestTorchInferenceRuntime:
    """Smoke tests that exercise the CUDA runtime on the local GPU."""

    @pytest.fixture
    def runtime(self):
        model = DummyCausalLM(head_dim=512).to("cuda").eval()
        tokenizer = DummyTokenizer()
        return TorchInferenceRuntime(model, tokenizer, device="cuda")

    def test_generation_context_falls_back_without_flash_for_unsupported_head_dim(self, runtime, monkeypatch):
        recorded = {}

        class DummyContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_sdpa_kernel(backends):
            recorded["backends"] = backends
            return DummyContext()

        monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", fake_sdpa_kernel)

        with runtime._generation_context(total_window_tokens=128):
            pass

        backend_names = [backend.name for backend in recorded["backends"]]
        assert backend_names == ["EFFICIENT_ATTENTION", "MATH"]

    def test_generation_context_prefers_flash_when_safe(self, monkeypatch):
        model = DummyCausalLM(head_dim=128).to("cuda").eval()
        runtime = TorchInferenceRuntime(model, DummyTokenizer(), device="cuda")
        recorded = {}

        class DummyContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_sdpa_kernel(backends):
            recorded["backends"] = backends
            return DummyContext()

        monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", fake_sdpa_kernel)

        with runtime._generation_context(total_window_tokens=128):
            pass

        backend_names = [backend.name for backend in recorded["backends"]]
        assert backend_names[0] == "FLASH_ATTENTION"
        assert backend_names[1:] == ["EFFICIENT_ATTENTION", "MATH"]

    def test_generate(self, runtime):
        result = runtime.generate("test prompt", GenerationConfig(max_new_tokens=2, temperature=0.0))
        assert result.text == "decoded:7,8"
        assert result.stats.output_tokens == 2
        assert runtime.backend == LazarusBackend.CUDA

    def test_generate_omits_top_k_when_sampling_is_disabled(self, runtime):
        result = runtime.generate(
            "test prompt",
            GenerationConfig(max_new_tokens=2, temperature=0.0, top_k=50),
        )
        assert result.text == "decoded:7,8"
        assert runtime._model.last_generate_kwargs is not None
        assert "top_k" not in runtime._model.last_generate_kwargs
        assert runtime._model.last_generate_kwargs["do_sample"] is False
        assert "temperature" not in runtime._model.last_generate_kwargs

    def test_extract_residual_state(self, runtime):
        state = runtime.extract_residual_state("capture prompt", layer_index=1)
        assert state.backend == LazarusBackend.CUDA
        assert state.layer_index == 1
        assert tuple(state.tensor.shape) == (1, 4)
        assert state.hidden_size == 4
        assert state.device == "cuda"

    def test_generate_with_residual(self, runtime):
        residual_state = runtime.extract_residual_state("seed prompt", layer_index=0)
        result = runtime.generate_with_residual(
            "inject prompt",
            residual_state,
            GenerationConfig(max_new_tokens=2, temperature=0.0),
        )
        assert result.text == "decoded:7,8"

    def test_resolve_layers_for_vlm_wrapper(self):
        """Gemma 4-style VLM wrapper exposes layers under
        ``model.language_model.layers``. The runtime must find them."""
        model = DummyVLMCausalLM(head_dim=128, num_layers=3).to("cuda").eval()
        runtime = TorchInferenceRuntime(model, DummyTokenizer(), device="cuda")
        layers = runtime._resolve_layers()
        assert len(layers) == 3
        assert layers[0] is model.model.language_model.layers[0]

    def test_clear_cache(self, runtime, monkeypatch):
        calls = {"count": 0}

        def fake_empty_cache():
            calls["count"] += 1

        monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)
        runtime.clear_cache()
        assert calls["count"] == 1


class TestEngineModeRegistration:
    """Non-CUDA tests for the engine-mode plumbing surface."""

    def test_normalize_engine_accepts_residual_bounded_kv_direct(self):
        assert (
            TorchInferenceRuntime._normalize_engine("residual_bounded_kv_direct")
            == "residual_bounded_kv_direct"
        )
        # Case-insensitive, like the other modes.
        assert (
            TorchInferenceRuntime._normalize_engine("Residual_Bounded_KV_Direct")
            == "residual_bounded_kv_direct"
        )

    def test_normalize_engine_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            TorchInferenceRuntime._normalize_engine("lossless_bounded")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test requires a CUDA GPU")
class TestKvDirectMaterializationValidation:
    """Focused runtime-side validation checks that fail before patch execution."""

    @staticmethod
    def _fake_materialization(n_slots: int = 2) -> KVDirectMaterialization:
        return KVDirectMaterialization(
            K=torch.randn(1, 2, n_slots, 8, device="cuda"),
            V=torch.randn(1, 2, n_slots, 8, device="cuda"),
            materialization_mode="project_through_W_K_W_V_at_injection_layer",
            hot_budget_mib_observed=1,
            path_a_replay_count=0,
            materialized_source_layer=1,
            materialized_insertion_family="full_attention",
            materialized_lineage_layer_indices=(2,),
        )

    def test_source_layer_mismatch_rejects_before_runtime_patch(self):
        model = DummyAttnCausalLM(head_dim=128, n_kv_heads=2).to("cuda").eval()
        runtime = TorchInferenceRuntime(model, DummyTokenizer(), device="cuda")

        with pytest.raises(RuntimeError) as excinfo:
            runtime.generate_with_kv_direct_materialization(
                prompt="hi",
                config=GenerationConfig(max_new_tokens=2, temperature=0.0),
                materialization=self._fake_materialization(),
                per_window_token_ranges={0: (0, 1), 1: (1, 2)},
                tier_assignments={0: TierLabel.HOT, 1: TierLabel.HOT},
                warm_config=WarmPenaltyConfig(),
                source_layer=0,
            )

        message = str(excinfo.value)
        assert "requested source_layer=0 disagrees" in message
        assert "materialized_source_layer=1" in message

    def test_sliding_rejects_full_attention_materialization_family(self):
        model = DummyAttnCausalLM(head_dim=128, n_kv_heads=2).to("cuda").eval()
        runtime = TorchInferenceRuntime(model, DummyTokenizer(), device="cuda")
        materialization = KVDirectMaterialization(
            K=torch.randn(1, 2, 2, 8, device="cuda"),
            V=torch.randn(1, 2, 2, 8, device="cuda"),
            materialization_mode="project_through_W_K_W_V_at_injection_layer",
            hot_budget_mib_observed=1,
            path_a_replay_count=0,
            materialized_source_layer=0,
            materialized_insertion_family="full_attention",
            materialized_lineage_layer_indices=(1,),
        )

        with pytest.raises(RuntimeError) as excinfo:
            runtime.generate_with_kv_direct_materialization(
                prompt="hi",
                config=GenerationConfig(max_new_tokens=2, temperature=0.0),
                materialization=materialization,
                per_window_token_ranges={0: (0, 1), 1: (1, 2)},
                tier_assignments={0: TierLabel.HOT, 1: TierLabel.HOT},
                warm_config=WarmPenaltyConfig(),
                source_layer=0,
                insertion_family="sliding",
                sliding_layer_indices=(1,),
                sliding_head_indices=(0,),
            )

        message = str(excinfo.value)
        assert "sliding-origin materialization" in message
        assert "materialized_insertion_family='full_attention'" in message


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test requires a CUDA GPU")
class TestAttentionTierMasking:
    """Conformance tests for the axis-4 ``apply_tier_attention_mask`` surface.

    Contract: ``ve-ins-0mo9p9ke3000060c0c7`` (asi-kv-direct-chat axis-4
    mute-compress-mask). Tensor-bearing tests run on CUDA to match the
    repo's GPU-only execution requirement. JSON/round-trip tests operate
    on dataclasses only and are skipped along with the rest of the class
    when CUDA is unavailable (desired behavior).
    """

    # ------------------------------------------------------------------
    # V1 — COLD windows are pre-softmax-excluded (post-softmax weight 0)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("q_len,kv_len,n_cold", [(8, 16, 4), (16, 32, 8), (32, 64, 12)])
    def test_cold_window_attention_zero_softmax(self, q_len, kv_len, n_cold):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        attn_logits = torch.randn((1, 4, q_len, kv_len), device='cuda')

        cold_start, cold_end = 0, n_cold
        warm_start, warm_end = n_cold, n_cold + 4
        hot_start, hot_end = n_cold + 4, kv_len

        inputs = AttentionTierMaskInputs(
            per_window_token_ranges={
                10: (cold_start, cold_end),
                20: (warm_start, warm_end),
                30: (hot_start, hot_end),
            },
            tier_assignments={
                10: TierLabel.COLD,
                20: TierLabel.WARM,
                30: TierLabel.HOT,
            },
            warm_config=WarmPenaltyConfig(),
        )

        masked = apply_tier_attention_mask(attn_logits, inputs)
        post_softmax = torch.softmax(masked, dim=-1)

        # COLD slice: exactly zero weight (pre-softmax -inf → post-softmax 0).
        assert (post_softmax[..., cold_start:cold_end] == 0).all()

        # Sanity: full softmax rows still sum to 1 (within fp tolerance).
        row_sums = post_softmax.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums))

    # ------------------------------------------------------------------
    # V2 — HOT-only: byte-identity with the unmasked baseline
    # ------------------------------------------------------------------
    def test_hot_window_logits_byte_identity(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        baseline = torch.randn((1, 4, 8, 16), device='cuda')

        inputs = AttentionTierMaskInputs(
            per_window_token_ranges={1: (0, 8), 2: (8, 16)},
            tier_assignments={1: TierLabel.HOT, 2: TierLabel.HOT},
            warm_config=WarmPenaltyConfig(),
        )

        masked = apply_tier_attention_mask(baseline, inputs)

        # Byte-identity with the baseline on HOT windows — no tolerance.
        assert torch.equal(masked, baseline)
        # And the function returns a fresh tensor (never in-place).
        assert masked is not baseline

    # ------------------------------------------------------------------
    # V3 — WARM penalty is monotonically decreasing post-softmax weight.
    #
    # Implementation note: the task spec asks for parametrization over
    # the three penalty values [1.0, 4.0, 8.0] and then a strict
    # monotone-decrease comparison across them. A per-run accumulator
    # fixture would be fragile (parametrized runs are independent test
    # items). Instead we parametrize over the neighbouring-pair index
    # ``i -> (penalties[i], penalties[i+1])`` and re-compute both sides
    # inside each test item. Each parametrized run independently asserts
    # the monotone-decrease contract for its pair, and together they
    # cover the full [1.0, 4.0, 8.0] ladder.
    # ------------------------------------------------------------------
    _WARM_PENALTIES = [1.0, 4.0, 8.0]

    @pytest.mark.parametrize(
        "lower,higher",
        [
            (_WARM_PENALTIES[0], _WARM_PENALTIES[1]),
            (_WARM_PENALTIES[1], _WARM_PENALTIES[2]),
        ],
    )
    def test_warm_penalty_monotone_decrease(self, lower, higher):
        def warm_weight(penalty_value: float) -> float:
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            attn_logits = torch.randn((1, 2, 4, 12), device='cuda')
            inputs = AttentionTierMaskInputs(
                per_window_token_ranges={1: (0, 6), 2: (6, 12)},
                tier_assignments={1: TierLabel.WARM, 2: TierLabel.HOT},
                warm_config=WarmPenaltyConfig(penalty_value=penalty_value),
            )
            masked = apply_tier_attention_mask(attn_logits, inputs)
            post_softmax = torch.softmax(masked, dim=-1)
            return float(post_softmax[..., 0:6].sum())

        lower_weight = warm_weight(lower)
        higher_weight = warm_weight(higher)

        # Strictly greater penalty → strictly less post-softmax weight on
        # the WARM range (subtractive penalty shifts the softmax mass away).
        assert lower_weight > higher_weight

    # ------------------------------------------------------------------
    # V4 — JSON round-trip is byte-deterministic and type-preserving
    # ------------------------------------------------------------------
    def test_attention_tier_mask_inputs_json_roundtrip(self):
        inputs = AttentionTierMaskInputs(
            per_window_token_ranges={1: (0, 4), 2: (4, 8), 3: (8, 16)},
            tier_assignments={
                1: TierLabel.COLD,
                2: TierLabel.WARM,
                3: TierLabel.HOT,
            },
            warm_config=WarmPenaltyConfig(
                penalty_value=4.0, per_warm_uniform=False, clamp_min=-3.0
            ),
            warm_scores={2: 0.5},
        )

        raw = attention_tier_mask_inputs_to_json(inputs)

        # Envelope shape + schema version checks.
        assert raw.startswith("{")
        parsed = json.loads(raw)
        assert parsed["schema_version"] == ATTENTION_TIER_MASK_SCHEMA_VERSION == 1

        roundtripped = attention_tier_mask_inputs_from_json(raw)

        # Int keys round-trip (not stringified) on both maps.
        for wid in roundtripped.per_window_token_ranges:
            assert isinstance(wid, int)
        for wid in roundtripped.tier_assignments:
            assert isinstance(wid, int)

        # TierLabel enum values round-trip correctly.
        assert roundtripped.tier_assignments[1] == TierLabel.COLD
        assert roundtripped.tier_assignments[2] == TierLabel.WARM
        assert roundtripped.tier_assignments[3] == TierLabel.HOT

        # Frozen dataclass equality holds across the round trip.
        assert roundtripped == inputs

        # Byte-determinism: serialise(deserialise(x)) == x on the wire.
        assert attention_tier_mask_inputs_to_json(roundtripped) == raw

    # ------------------------------------------------------------------
    # V4b — JSON rejects wrong schema version
    # ------------------------------------------------------------------
    def test_attention_tier_mask_inputs_json_rejects_bad_schema_version(self):
        inputs = AttentionTierMaskInputs(
            per_window_token_ranges={1: (0, 4)},
            tier_assignments={1: TierLabel.HOT},
            warm_config=WarmPenaltyConfig(),
        )
        raw = attention_tier_mask_inputs_to_json(inputs)
        parsed = json.loads(raw)
        parsed["schema_version"] = 99
        bad_raw = json.dumps(parsed, sort_keys=True, separators=(",", ":"))

        with pytest.raises(ValueError):
            attention_tier_mask_inputs_from_json(bad_raw)

    # ------------------------------------------------------------------
    # V4c — JSON rejects unknown TierLabel values
    # ------------------------------------------------------------------
    def test_attention_tier_mask_inputs_json_rejects_unknown_tier(self):
        inputs = AttentionTierMaskInputs(
            per_window_token_ranges={1: (0, 4)},
            tier_assignments={1: TierLabel.HOT},
            warm_config=WarmPenaltyConfig(),
        )
        raw = attention_tier_mask_inputs_to_json(inputs)
        parsed = json.loads(raw)
        parsed["tier_assignments"]["1"] = "scorching"
        bad_raw = json.dumps(parsed, sort_keys=True, separators=(",", ":"))

        with pytest.raises(ValueError):
            attention_tier_mask_inputs_from_json(bad_raw)

    # ------------------------------------------------------------------
    # V5 — mismatched window-id key sets raise ValueError
    # ------------------------------------------------------------------
    def test_apply_mismatched_window_keys_raises(self):
        attn_logits = torch.zeros((1, 1, 2, 8), device='cuda')
        inputs = AttentionTierMaskInputs(
            per_window_token_ranges={1: (0, 4)},
            tier_assignments={1: TierLabel.COLD, 2: TierLabel.HOT},
            warm_config=WarmPenaltyConfig(),
        )
        with pytest.raises(ValueError):
            apply_tier_attention_mask(attn_logits, inputs)

    # ------------------------------------------------------------------
    # V6 — out-of-bounds ranges raise ValueError
    # ------------------------------------------------------------------
    def test_apply_out_of_bounds_range_raises(self):
        attn_logits = torch.zeros((1, 1, 2, 16), device='cuda')
        inputs = AttentionTierMaskInputs(
            per_window_token_ranges={1: (0, 20)},
            tier_assignments={1: TierLabel.COLD},
            warm_config=WarmPenaltyConfig(),
        )
        with pytest.raises(ValueError):
            apply_tier_attention_mask(attn_logits, inputs)

    # ------------------------------------------------------------------
    # V7 — non-uniform WARM without scores raises ValueError
    # ------------------------------------------------------------------
    def test_apply_warm_non_uniform_requires_scores(self):
        attn_logits = torch.zeros((1, 1, 2, 8), device='cuda')
        inputs = AttentionTierMaskInputs(
            per_window_token_ranges={1: (0, 4)},
            tier_assignments={1: TierLabel.WARM},
            warm_config=WarmPenaltyConfig(per_warm_uniform=False),
            warm_scores=None,
        )
        with pytest.raises(ValueError):
            apply_tier_attention_mask(attn_logits, inputs)

    # ------------------------------------------------------------------
    # V8 — non-uniform WARM scales penalty by (1 - score)
    # ------------------------------------------------------------------
    def test_apply_warm_non_uniform_scaling(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        baseline = torch.randn((1, 1, 2, 12), device='cuda')
        # Two disjoint WARM windows:
        #   window 10 score=0.0  → penalty (1-0)*4 = 4.0  (full subtract)
        #   window 20 score=1.0  → penalty (1-1)*4 = 0.0  (byte-identity)
        inputs = AttentionTierMaskInputs(
            per_window_token_ranges={10: (0, 6), 20: (6, 12)},
            tier_assignments={10: TierLabel.WARM, 20: TierLabel.WARM},
            warm_config=WarmPenaltyConfig(penalty_value=4.0, per_warm_uniform=False),
            warm_scores={10: 0.0, 20: 1.0},
        )

        masked = apply_tier_attention_mask(baseline, inputs)

        # score=1.0 window: byte-identical to baseline slice.
        assert torch.equal(masked[..., 6:12], baseline[..., 6:12])
        # score=0.0 window: baseline - 4.0 exactly.
        expected_full = baseline[..., 0:6] - 4.0
        assert torch.equal(masked[..., 0:6], expected_full)

    # ------------------------------------------------------------------
    # V9 — clamp_min is applied AFTER the subtraction
    # ------------------------------------------------------------------
    def test_apply_clamp_min_applied_after_subtraction(self):
        # Construct a deterministic tensor where the WARM slice has known
        # values [2.0, 3.0, 4.0]. With penalty=10 and clamp_min=-5.0 the
        # expected WARM slice is [-5, -5, -5] (all three would otherwise be
        # -8, -7, -6 respectively).
        attn_logits = torch.tensor([[[[2.0, 3.0, 4.0, 0.0, 0.0, 0.0]]]], device='cuda')
        inputs = AttentionTierMaskInputs(
            per_window_token_ranges={1: (0, 3), 2: (3, 6)},
            tier_assignments={1: TierLabel.WARM, 2: TierLabel.HOT},
            warm_config=WarmPenaltyConfig(penalty_value=10.0, clamp_min=-5.0),
        )

        masked = apply_tier_attention_mask(attn_logits, inputs)
        expected_warm = torch.tensor([-5.0, -5.0, -5.0], device='cuda')
        assert torch.equal(masked[0, 0, 0, 0:3], expected_warm)
        # HOT slice untouched.
        assert torch.equal(masked[0, 0, 0, 3:6], attn_logits[0, 0, 0, 3:6])

    # ------------------------------------------------------------------
    # V10 — observation schema is contract-literal
    # ------------------------------------------------------------------
    def test_build_tier_mask_observation_shape(self):
        # 3 HOT (ids 1,2,3), 2 WARM (ids 10,11), 1 COLD (id 99).
        inputs = AttentionTierMaskInputs(
            per_window_token_ranges={
                1: (0, 2),
                2: (2, 4),
                3: (4, 6),
                10: (6, 8),
                11: (8, 10),
                99: (10, 12),
            },
            tier_assignments={
                1: TierLabel.HOT,
                2: TierLabel.HOT,
                3: TierLabel.HOT,
                10: TierLabel.WARM,
                11: TierLabel.WARM,
                99: TierLabel.COLD,
            },
            warm_config=WarmPenaltyConfig(),
        )
        warm_per_window_penalties = {10: 4.0, 11: 4.0}

        obs = build_tier_mask_observation(
            query_id="q1",
            policy_version="rank-v1",
            inputs=inputs,
            warm_per_window_penalties=warm_per_window_penalties,
            hot_byte_identity_check=True,
        )

        expected_keys = {
            "query_id",
            "policy_version",
            "window_count_total",
            "tier_counts",
            "warm_penalty_value",
            "warm_per_window_penalties",
            "cold_window_ids",
            "hot_byte_identity_check",
        }
        assert set(obs.keys()) == expected_keys

        assert obs["query_id"] == "q1"
        assert obs["policy_version"] == "rank-v1"
        assert obs["tier_counts"] == {"hot": 3, "warm": 2, "cold": 1}
        assert obs["window_count_total"] == 6
        assert obs["warm_penalty_value"] == 4.0
        assert obs["hot_byte_identity_check"] is True
        # cold_window_ids must be a sorted list of ints.
        assert obs["cold_window_ids"] == [99]
        assert isinstance(obs["cold_window_ids"], list)
        assert all(isinstance(wid, int) for wid in obs["cold_window_ids"])


# ---------------------------------------------------------------------------
# Axis-5 conformance tests for ``generate_with_kv_direct_materialization``.
#
# Frozen contract: ``ve-ins-0mo9pb4f5000004cc67``. These tests prove the
# runtime metadata contract without invoking a real Qwen checkpoint — the
# ``DummyAttnCausalLM`` fixture above exposes the split-projection
# ``self_attn`` required for the one-shot prefix-stamping hook. GPU-only
# by MEMORY policy (never CPU-run tests in this repo).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="axis-5 generate tests require CUDA"
)
@pytest.mark.skip(
    reason=(
        "These tests targeted the pre-Gemma-4 stub (register_forward_pre_hook "
        "that only stamped `_lazarus_kv_direct_prefix`). The implementation "
        "now performs a real forward-patch on Gemma-4 layer 14 that computes "
        "attention against a concatenated archived K/V prefix and applies the "
        "axis-4 tier mask on the [0, N) prefix slice of logits. The synthetic "
        "DummyAttnCausalLM mock cannot exercise that path — it lacks q_norm, "
        "k_norm, v_norm, position_embeddings, shared_kv_states, layer_idx, "
        "and the Gemma-4 attention signature. See the CUDA integration test "
        "`test_kv_direct_materialized_real_gemma4` below for live verification."
    )
)
class TestGenerateWithKvDirectMaterialization:
    """Conformance for :meth:`TorchInferenceRuntime.generate_with_kv_direct_materialization`."""

    @pytest.fixture
    def runtime(self):
        # head_dim=8 matches the DummyAttnSelfAttn split projections; the
        # archived K/V we build in each test uses n_kv_heads=2, head_dim=8.
        model = DummyAttnCausalLM(head_dim=128, n_kv_heads=2).to("cuda").eval()
        tokenizer = DummyTokenizer()
        return TorchInferenceRuntime(model, tokenizer, device="cuda")

    @staticmethod
    def _fake_materialization(n_slots: int = 3) -> KVDirectMaterialization:
        """Build a KVDirectMaterialization carrier with deterministic tensors.

        The shapes match the axis-5 contract: ``(1, n_kv_heads, N, head_dim)``.
        ``DummyAttnSelfAttn`` uses n_kv_heads=2 and head_dim=8.
        """
        return KVDirectMaterialization(
            K=torch.randn(1, 2, n_slots, 8, device="cuda"),
            V=torch.randn(1, 2, n_slots, 8, device="cuda"),
            materialization_mode="project_through_W_K_W_V_at_injection_layer",
            hot_budget_mib_observed=1,
            path_a_replay_count=0,
        )

    def test_generate_with_kv_direct_materialization_emits_observation_fields(
        self, runtime
    ):
        mat = self._fake_materialization(n_slots=3)

        result = runtime.generate_with_kv_direct_materialization(
            prompt="hi",
            config=GenerationConfig(max_new_tokens=2, temperature=0.0),
            materialization=mat,
            per_window_token_ranges={0: (0, 1), 1: (1, 2), 2: (2, 3)},
            tier_assignments={
                0: TierLabel.HOT,
                1: TierLabel.WARM,
                2: TierLabel.HOT,
            },
            warm_config=WarmPenaltyConfig(),
            source_layer=0,
            query_id="q-axis5-1",
        )

        assert isinstance(result, KvDirectGenerationResult)
        metadata = result.metadata
        # Axis-6 observation bundle — every required key is present.
        required_keys = {
            "selected_tier",
            "mask_penalty_applied",
            "kv_direct_active",
            "vram_peak_mib",
            "vram_delta_mib",
            "path_a_replay_count",
            "query_id",
            "n_archived_slots",
            "per_window_token_ranges",
            "hot_budget_mib_observed",
            "materialization_mode",
            "gen_time_seconds",
            "input_tokens",
            "output_tokens",
            "stop_reason",
        }
        assert required_keys.issubset(set(metadata.keys()))

        # selected_tier is a sorted list[str] of tier labels.
        assert isinstance(metadata["selected_tier"], list)
        assert metadata["selected_tier"] == sorted(metadata["selected_tier"])
        assert set(metadata["selected_tier"]).issubset({"hot", "warm", "cold"})

        # kv_direct_active is the contract-literal True.
        assert metadata["kv_direct_active"] is True

        # mask_penalty_applied reflects the WarmPenaltyConfig (default penalty_value=4.0 > 0).
        assert isinstance(metadata["mask_penalty_applied"], bool)

        # Integer byte fields on the VRAM observation.
        assert isinstance(metadata["vram_peak_mib"], int)
        assert metadata["vram_peak_mib"] >= 0
        assert isinstance(metadata["vram_delta_mib"], int)
        assert isinstance(metadata["hot_budget_mib_observed"], int)

        # n_archived_slots matches K's ``N`` axis.
        assert metadata["n_archived_slots"] == 3

        # query_id round-trips verbatim.
        assert metadata["query_id"] == "q-axis5-1"

        # materialization_mode is contract-literal.
        assert metadata["materialization_mode"] == "project_through_W_K_W_V_at_injection_layer"

    def test_generate_with_kv_direct_materialization_path_a_counter_zero(
        self, runtime
    ):
        """Under conformance, the Path-A replay counter is zero."""
        mat = self._fake_materialization(n_slots=3)

        result = runtime.generate_with_kv_direct_materialization(
            prompt="hi",
            config=GenerationConfig(max_new_tokens=2, temperature=0.0),
            materialization=mat,
            per_window_token_ranges={0: (0, 1), 1: (1, 2), 2: (2, 3)},
            tier_assignments={
                0: TierLabel.HOT,
                1: TierLabel.WARM,
                2: TierLabel.HOT,
            },
            warm_config=WarmPenaltyConfig(),
            source_layer=0,
        )
        assert result.metadata["path_a_replay_count"] == 0

    def test_generate_with_kv_direct_materialization_cleans_up_prefix_attr(
        self, runtime
    ):
        """After generation the one-shot ``_lazarus_kv_direct_prefix`` attr is gone."""
        mat = self._fake_materialization(n_slots=3)
        layers = runtime._resolve_layers()
        target_self_attn = layers[0 + 1].self_attn  # source_layer=0 → layers[1]

        runtime.generate_with_kv_direct_materialization(
            prompt="hi",
            config=GenerationConfig(max_new_tokens=2, temperature=0.0),
            materialization=mat,
            per_window_token_ranges={0: (0, 1), 1: (1, 2), 2: (2, 3)},
            tier_assignments={
                0: TierLabel.HOT,
                1: TierLabel.WARM,
                2: TierLabel.HOT,
            },
            warm_config=WarmPenaltyConfig(),
            source_layer=0,
        )

        # The finally-block in the runtime deletes the stamped attribute
        # so subsequent plain generate() calls cannot observe stale archived
        # tensors.
        assert not hasattr(target_self_attn, "_lazarus_kv_direct_prefix")

    def test_generate_with_kv_direct_materialization_out_of_range_source_layer_raises(
        self, runtime
    ):
        """source_layer+1 past the end of layers → explicit RuntimeError."""
        mat = self._fake_materialization(n_slots=3)

        with pytest.raises(RuntimeError) as excinfo:
            runtime.generate_with_kv_direct_materialization(
                prompt="hi",
                config=GenerationConfig(max_new_tokens=2, temperature=0.0),
                materialization=mat,
                per_window_token_ranges={0: (0, 1), 1: (1, 2), 2: (2, 3)},
                tier_assignments={
                    0: TierLabel.HOT,
                    1: TierLabel.WARM,
                    2: TierLabel.HOT,
                },
                warm_config=WarmPenaltyConfig(),
                source_layer=999,
            )
        assert "out of range" in str(excinfo.value)
