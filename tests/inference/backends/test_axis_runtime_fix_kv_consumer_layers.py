"""axis-runtime-fix — patched_forward correctness on Gemma-4-E2B-it KV-consumer layers.

References
----------
* mission .................. chuk-lazurus-3h1
* lead manifest ............ ve-ins-0moe0c0jb0000475268
* failure-finding (the bug)  ve-ins-0moe02apx00003fe3b6
* supervisor decision (alpha) ve-ins-0moe06p5g00002de88a
* axis-A fixture ........... GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS = {4,9,14,19,24,29,34}
* axis-E canary (READ ONLY)  test_axis_E_kv_direct_e2e_apollo.py::test_kv_direct_synthetic_smoke_e2e_layer_29
* AMD 14 ................... CUDA-only execution (no CPU fallbacks)

These tests prove the fix in torch_runtime.py for KV-consumer target layers.
The axis-E smoke remains the load-bearing canary; these tests narrow the
coverage to two specific consumer-target cases (target_idx=30 sliding,
target_idx=34 full-attention boundary) plus a producer-target regression
(target_idx=14) that MUST stay GREEN to prove the fix did not regress
the historical layer-14 path.
"""

from __future__ import annotations

import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA required (AMD 14)", allow_module_level=True)


GEMMA4_SNAPSHOT_ID = "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
GEMMA4_SNAPSHOT = Path(
    f"/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/"
    f"snapshots/{GEMMA4_SNAPSHOT_ID}"
)


requires_cuda_gemma4 = pytest.mark.skipif(
    not (torch.cuda.is_available() and GEMMA4_SNAPSHOT.is_dir()),
    reason="requires CUDA + local Gemma-4-E2B-it snapshot (AMD 14)",
)


# ── Diagnostic JSONL sink ──────────────────────────────────────────────
_DIAG_LINES: list[dict] = []
_DIAG_PATH: Path | None = None


def _diag_path() -> Path:
    global _DIAG_PATH
    if _DIAG_PATH is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short = secrets.token_hex(4)
        _DIAG_PATH = (
            Path("prod/validation")
            / f"diagnostic_axis_runtime_fix_{ts}-{short}.jsonl"
        ).resolve()
        _DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DIAG_LINES.append(
            {
                "event": "meta",
                "axis": "runtime-fix",
                "mission": "chuk-lazurus-3h1",
                "lead_manifest": "ve-ins-0moe0c0jb0000475268",
                "failure_finding": "ve-ins-0moe02apx00003fe3b6",
                "supervisor_decision": "ve-ins-0moe06p5g00002de88a",
                "model_snapshot_id": GEMMA4_SNAPSHOT_ID,
                "device": "cuda",
                "dtype": "bf16",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _flush()
    return _DIAG_PATH


def _flush() -> None:
    if _DIAG_PATH is None:
        return
    with _DIAG_PATH.open("w", encoding="utf-8") as fh:
        for entry in _DIAG_LINES:
            fh.write(json.dumps(entry) + "\n")


def _append(event: dict) -> None:
    _diag_path()
    _DIAG_LINES.append(event)
    _flush()


# ── Module-scoped real Gemma-4-E2B-it runtime ──────────────────────────
@pytest.fixture(scope="module")
def real_runtime():
    """Load Gemma-4-E2B-it once for the whole module on CUDA bf16.

    Mirrors test_axis_E_kv_direct_e2e_apollo.py and
    test_kv_direct_materialized_real_gemma4.py fixture pattern verbatim.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from chuk_lazarus.inference.backends import TorchInferenceRuntime

    tokenizer = AutoTokenizer.from_pretrained(str(GEMMA4_SNAPSHOT))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(GEMMA4_SNAPSHOT),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .to("cuda")
        .eval()
    )
    return TorchInferenceRuntime(model, tokenizer, device="cuda")


# ── Helpers ────────────────────────────────────────────────────────────


def _resolve_int_attr(cfg, attr: str) -> int:
    top = getattr(cfg, attr, None)
    if isinstance(top, int) and top > 0:
        return top
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        sub = getattr(text_cfg, attr, None)
        if isinstance(sub, int) and sub > 0:
            return sub
    raise AssertionError(
        f"axis-runtime-fix: could not resolve positive int '{attr}' from "
        f"model.config or model.config.text_config; got "
        f"top={getattr(cfg, attr, None)!r}, "
        f"text_config_attr="
        f"{getattr(getattr(cfg, 'text_config', None), attr, None)!r}."
    )


def _build_synthetic_kv_pages(n_kv_heads: int, n_facts: int, head_dim: int):
    """Pure-synthetic K/V pages, shape (1, n_kv_heads, n_facts, head_dim)."""
    K = torch.randn(1, n_kv_heads, n_facts, head_dim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, n_kv_heads, n_facts, head_dim, dtype=torch.bfloat16, device="cuda")
    return K.contiguous(), V.contiguous()


def _resolve_producer_kv_dims(runtime, source_layer: int) -> tuple[int, int]:
    """Return (n_kv_heads, head_dim) of the producer module that prefix
    normalisation will route through.

    The runtime resolves target_idx = source_layer + 1. If target is a
    KV-consumer (no k_proj), the prefix is normalised through the
    producer at target.kv_shared_layer_index. If target is a producer
    (e.g. layer 14), the producer IS target itself.
    """
    layers = runtime._resolve_layers()
    target_idx = source_layer + 1
    target_self_attn = getattr(layers[target_idx], "self_attn", None) or getattr(
        layers[target_idx], "attention", None
    )
    if target_self_attn is None:
        raise AssertionError(
            f"axis-runtime-fix: target layer {target_idx} has no self_attn"
        )
    if hasattr(target_self_attn, "k_proj"):
        producer_self_attn = target_self_attn
    else:
        producer_idx = getattr(target_self_attn, "kv_shared_layer_index", None)
        if producer_idx is None:
            raise AssertionError(
                f"axis-runtime-fix: target layer {target_idx} is consumer "
                "but exposes no kv_shared_layer_index"
            )
        producer_self_attn = getattr(layers[int(producer_idx)], "self_attn", None) or getattr(
            layers[int(producer_idx)], "attention", None
        )
    head_dim = int(producer_self_attn.head_dim)
    n_kv_heads = int(producer_self_attn.k_proj.out_features) // head_dim
    return n_kv_heads, head_dim


def _materialization_at(runtime, K, V, *, source_layer: int):
    """Build a KVDirectMaterialization at injection_layer=source_layer.

    Mirrors `_materialization_from_adapter_pages` from the axis-E smoke test
    inline (that helper lives in a READ-ONLY test module).
    """
    from chuk_lazarus.inference.backends._torch_residual_bounded import (
        KVDirectMaterialization,
        resolve_kv_materialization_provenance,
    )

    insertion_family, lineage = resolve_kv_materialization_provenance(
        runtime._model,
        source_layer,
        resolve_layers=lambda _model: runtime._resolve_layers(),
    )
    return KVDirectMaterialization(
        K=K,
        V=V,
        materialization_mode="project_through_W_K_W_V_at_injection_layer",
        hot_budget_mib_observed=1,
        path_a_replay_count=0,
        materialized_source_layer=source_layer,
        materialized_insertion_family=insertion_family,
        materialized_lineage_layer_indices=lineage,
    )


def _project_kv_via_layer(runtime, *, injection_layer: int, n_slots: int):
    """Project synthetic residuals through layers[injection_layer+1].self_attn k_proj/v_proj.

    Used ONLY for the producer-target regression case (injection_layer=13 →
    target=14, which has k_proj/v_proj). Mirrors _build_materialization
    pattern from test_kv_direct_materialized_real_gemma4.py.
    """
    layers = runtime._resolve_layers()
    target_layer = layers[injection_layer + 1]
    self_attn = target_layer.self_attn
    n_kv_heads = self_attn.k_proj.out_features // self_attn.head_dim
    head_dim = self_attn.head_dim
    hidden = self_attn.k_proj.in_features

    residuals = torch.randn(
        1, n_slots, hidden, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    with torch.no_grad():
        k_flat = self_attn.k_proj(residuals)
        v_flat = self_attn.v_proj(residuals)
    K = k_flat.view(1, n_slots, n_kv_heads, head_dim).transpose(1, 2).contiguous()
    V = v_flat.view(1, n_slots, n_kv_heads, head_dim).transpose(1, 2).contiguous()
    return K, V


# ── Tests ──────────────────────────────────────────────────────────────


@requires_cuda_gemma4
def test_patched_forward_consumer_at_target_idx_30_via_source_layer_29(real_runtime):
    """source_layer=29 → target_idx=30 (sliding KV-consumer). Pre-fix this raised
    AttributeError: 'Gemma4TextAttention' object has no attribute 'k_proj'.
    Post-fix the patched_forward routes through the shared-KV branch using
    the producer at kv_shared_layer_index (layer 28 for sliding).
    """
    from chuk_lazarus.inference.backends.torch_runtime import (
        KvDirectGenerationResult,
        WarmPenaltyConfig,
    )
    from chuk_lazarus.inference.context.knowledge.gemma4_e2b_it_layers import (
        GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS,
    )
    from chuk_lazarus.inference.generation import GenerationConfig
    from chuk_lazarus.session_retrieval.tier_policy import TierLabel

    SOURCE_LAYER = 29
    assert SOURCE_LAYER in GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS

    runtime = real_runtime
    n_kv_heads, head_dim = _resolve_producer_kv_dims(runtime, SOURCE_LAYER)

    n_facts = 2
    K, V = _build_synthetic_kv_pages(n_kv_heads, n_facts, head_dim)
    mat = _materialization_at(runtime, K, V, source_layer=SOURCE_LAYER)

    config = GenerationConfig(max_new_tokens=8, temperature=0.0)
    t0 = time.perf_counter()
    gen_result = runtime.generate_with_kv_direct_materialization(
        prompt="Hello",
        config=config,
        materialization=mat,
        per_window_token_ranges={0: (0, n_facts)},
        tier_assignments={0: TierLabel.HOT},
        warm_config=WarmPenaltyConfig(),
        source_layer=SOURCE_LAYER,
    )
    duration_seconds = float(time.perf_counter() - t0)

    assert isinstance(gen_result, KvDirectGenerationResult)
    meta = gen_result.metadata
    output_tokens = int(meta.get("output_tokens", 0))
    text = (gen_result.text or "").strip()
    assert output_tokens >= 1 or len(text) > 0, (
        f"axis-runtime-fix: target_idx=30 produced no output — "
        f"output_tokens={output_tokens}, text={gen_result.text!r}"
    )
    assert bool(meta.get("kv_direct_active")) is True

    _append(
        {
            "event": "consumer_target_idx_30",
            "test": "test_patched_forward_consumer_at_target_idx_30_via_source_layer_29",
            "verdict": "PASS",
            "source_layer": SOURCE_LAYER,
            "target_idx": SOURCE_LAYER + 1,
            "n_facts": n_facts,
            "synthetic_kv_shape": [1, n_kv_heads, n_facts, head_dim],
            "duration_seconds": duration_seconds,
            "output_tokens": output_tokens,
        }
    )


@requires_cuda_gemma4
def test_patched_forward_consumer_at_target_idx_34_via_source_layer_33(real_runtime):
    """source_layer=33 → target_idx=34 (full-attention KV-consumer at boundary).
    Layer 34 is the final layer; kv_shared_layer_index=24 (full producer).
    Pre-fix this raises AttributeError on layer 34. Post-fix routes through
    shared-KV branch with producer=24.
    """
    from chuk_lazarus.inference.backends.torch_runtime import (
        KvDirectGenerationResult,
        WarmPenaltyConfig,
    )
    from chuk_lazarus.inference.generation import GenerationConfig
    from chuk_lazarus.session_retrieval.tier_policy import TierLabel

    SOURCE_LAYER = 33
    runtime = real_runtime
    n_kv_heads, head_dim = _resolve_producer_kv_dims(runtime, SOURCE_LAYER)

    n_facts = 2
    K, V = _build_synthetic_kv_pages(n_kv_heads, n_facts, head_dim)
    mat = _materialization_at(runtime, K, V, source_layer=SOURCE_LAYER)

    config = GenerationConfig(max_new_tokens=8, temperature=0.0)
    t0 = time.perf_counter()
    gen_result = runtime.generate_with_kv_direct_materialization(
        prompt="Hello",
        config=config,
        materialization=mat,
        per_window_token_ranges={0: (0, n_facts)},
        tier_assignments={0: TierLabel.HOT},
        warm_config=WarmPenaltyConfig(),
        source_layer=SOURCE_LAYER,
    )
    duration_seconds = float(time.perf_counter() - t0)

    assert isinstance(gen_result, KvDirectGenerationResult)
    meta = gen_result.metadata
    output_tokens = int(meta.get("output_tokens", 0))
    text = (gen_result.text or "").strip()
    assert output_tokens >= 1 or len(text) > 0
    assert bool(meta.get("kv_direct_active")) is True

    _append(
        {
            "event": "consumer_target_idx_34",
            "test": "test_patched_forward_consumer_at_target_idx_34_via_source_layer_33",
            "verdict": "PASS",
            "source_layer": SOURCE_LAYER,
            "target_idx": SOURCE_LAYER + 1,
            "n_facts": n_facts,
            "synthetic_kv_shape": [1, n_kv_heads, n_facts, head_dim],
            "duration_seconds": duration_seconds,
            "output_tokens": output_tokens,
        }
    )


@requires_cuda_gemma4
def test_patched_forward_producer_target_idx_14_unchanged(real_runtime):
    """source_layer=13 → target_idx=14 (producer; has k_proj/v_proj/k_norm/v_norm).
    The fix MUST NOT regress this path. K/V are projected through layer 14's
    real W_K/W_V (mirroring test_kv_direct_materialized_real_gemma4.py).
    """
    from chuk_lazarus.inference.backends.torch_runtime import (
        KvDirectGenerationResult,
        WarmPenaltyConfig,
    )
    from chuk_lazarus.inference.generation import GenerationConfig
    from chuk_lazarus.session_retrieval.tier_policy import TierLabel

    SOURCE_LAYER = 13
    runtime = real_runtime

    n_facts = 2
    K, V = _project_kv_via_layer(runtime, injection_layer=SOURCE_LAYER, n_slots=n_facts)
    mat = _materialization_at(runtime, K, V, source_layer=SOURCE_LAYER)

    config = GenerationConfig(max_new_tokens=8, temperature=0.0)
    t0 = time.perf_counter()
    gen_result = runtime.generate_with_kv_direct_materialization(
        prompt="Hello",
        config=config,
        materialization=mat,
        per_window_token_ranges={0: (0, n_facts)},
        tier_assignments={0: TierLabel.HOT},
        warm_config=WarmPenaltyConfig(),
        source_layer=SOURCE_LAYER,
    )
    duration_seconds = float(time.perf_counter() - t0)

    assert isinstance(gen_result, KvDirectGenerationResult)
    meta = gen_result.metadata
    output_tokens = int(meta.get("output_tokens", 0))
    text = (gen_result.text or "").strip()
    assert output_tokens >= 1 or len(text) > 0
    assert bool(meta.get("kv_direct_active")) is True

    _append(
        {
            "event": "producer_target_idx_14_regression",
            "test": "test_patched_forward_producer_target_idx_14_unchanged",
            "verdict": "PASS",
            "source_layer": SOURCE_LAYER,
            "target_idx": SOURCE_LAYER + 1,
            "n_facts": n_facts,
            "duration_seconds": duration_seconds,
            "output_tokens": output_tokens,
        }
    )
