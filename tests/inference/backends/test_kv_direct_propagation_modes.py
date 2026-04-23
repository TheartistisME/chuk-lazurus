"""CUDA integration coverage for KV-direct propagation modes on Gemma-4-E2B."""

from __future__ import annotations

import contextlib
import types
from collections.abc import Iterable

import pytest
import torch

from chuk_lazarus.inference.backends.torch_runtime import WarmPenaltyConfig
from chuk_lazarus.inference.generation import GenerationConfig
from chuk_lazarus.session_retrieval.tier_policy import TierLabel
from tests.inference.backends import test_kv_direct_materialized_real_gemma4 as _shared_tests

PROMPT = "<start_of_turn>user\nHello\n<end_of_turn>\n<start_of_turn>model\n"
SHARED_LAYER_INDICES = (19, 24, 29, 34)
CLEANUP_LAYER_INDICES = (14, 19, 24, 29, 34)
PROPAGATION_ENV_VAR = "LAZARUS_KV_PROPAGATION_MODE"

_build_materialization = _shared_tests._build_materialization
requires_cuda_gemma4 = _shared_tests.requires_cuda_gemma4
real_runtime = _shared_tests.real_runtime


def _generation_kwargs(materialization):
    n_slots = int(materialization.K.shape[-2])
    return {
        "prompt": PROMPT,
        "config": GenerationConfig(max_new_tokens=1, temperature=0.0),
        "materialization": materialization,
        "per_window_token_ranges": {
            window_id: (window_id, window_id + 1) for window_id in range(n_slots)
        },
        "tier_assignments": dict.fromkeys(range(n_slots), TierLabel.HOT),
        "warm_config": WarmPenaltyConfig(),
        "source_layer": 13,
    }


@contextlib.contextmanager
def _force_eager_attention(runtime):
    config = runtime._model.config
    original_impl = getattr(config, "_attn_implementation", None)
    config._attn_implementation = "eager"
    try:
        yield
    finally:
        if original_impl is None:
            try:
                delattr(config, "_attn_implementation")
            except AttributeError:
                pass
        else:
            config._attn_implementation = original_impl


@contextlib.contextmanager
def _capture_prefill_attn(runtime, layer_indices: Iterable[int], *, capture_tensors=()):
    layers = runtime._resolve_layers()
    captures: dict[int, dict[str, object]] = {}
    originals: list[tuple[object, object]] = []
    tensor_layers = {int(layer_idx) for layer_idx in capture_tensors}

    try:
        for layer_idx in layer_indices:
            self_attn = layers[int(layer_idx)].self_attn
            original_forward = self_attn.forward
            originals.append((self_attn, original_forward))

            def wrapped_forward(
                module_self,
                *args,
                __layer_idx=int(layer_idx),
                __original=original_forward,
                **kwargs,
            ):
                hidden_states = args[0] if args else kwargs["hidden_states"]
                outputs = __original(*args, **kwargs)
                if int(hidden_states.shape[1]) > 1 and __layer_idx not in captures:
                    attn_weights = outputs[1]
                    if attn_weights is None:
                        raise AssertionError(
                            f"layer {__layer_idx} returned no attention weights during prefill"
                        )
                    record: dict[str, object] = {
                        "q_len": int(hidden_states.shape[1]),
                        "attn_shape": tuple(int(dim) for dim in attn_weights.shape),
                    }
                    if __layer_idx in tensor_layers:
                        record["attn_weights"] = (
                            attn_weights.detach().to(dtype=torch.float32).cpu()
                        )
                    captures[__layer_idx] = record
                return outputs

            self_attn.forward = types.MethodType(wrapped_forward, self_attn)

        yield captures
    finally:
        for self_attn, original_forward in reversed(originals):
            self_attn.forward = original_forward


def _run_and_capture(
    runtime,
    monkeypatch: pytest.MonkeyPatch,
    *,
    materialization,
    mode: str | None,
    layer_indices: Iterable[int],
    capture_tensors=(),
):
    if mode is None:
        monkeypatch.delenv(PROPAGATION_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(PROPAGATION_ENV_VAR, mode)

    with _force_eager_attention(runtime):
        with _capture_prefill_attn(
            runtime, layer_indices, capture_tensors=capture_tensors
        ) as captures:
            runtime.generate_with_kv_direct_materialization(
                **_generation_kwargs(materialization)
            )

    runtime.clear_cache()
    return captures


@requires_cuda_gemma4
def test_unset_mode_defaults_to_mode_a_propagation(real_runtime, monkeypatch):
    materialization = _build_materialization(real_runtime, n_slots=3)
    n_slots = int(materialization.K.shape[-2])
    captures = _run_and_capture(
        real_runtime,
        monkeypatch,
        materialization=materialization,
        mode=None,
        layer_indices=SHARED_LAYER_INDICES,
    )

    assert set(captures) == set(SHARED_LAYER_INDICES)
    for layer_idx in SHARED_LAYER_INDICES:
        q_len = captures[layer_idx]["q_len"]
        assert captures[layer_idx]["attn_shape"][-1] == q_len + n_slots


@requires_cuda_gemma4
@pytest.mark.parametrize(
    ("mode", "expects_prefix"),
    [("a", True), ("b", True), ("ab", True), ("none", False)],
)
def test_prefill_shared_layer_kv_len_matches_propagation_mode(
    real_runtime, monkeypatch, mode, expects_prefix
):
    materialization = _build_materialization(real_runtime, n_slots=3)
    captures = _run_and_capture(
        real_runtime,
        monkeypatch,
        materialization=materialization,
        mode=mode,
        layer_indices=SHARED_LAYER_INDICES,
    )
    n_slots = int(materialization.K.shape[-2])

    assert set(captures) == set(SHARED_LAYER_INDICES)
    for layer_idx in SHARED_LAYER_INDICES:
        q_len = captures[layer_idx]["q_len"]
        expected_kv_len = q_len + n_slots if expects_prefix else q_len
        assert captures[layer_idx]["attn_shape"][-1] == expected_kv_len


@requires_cuda_gemma4
def test_mode_ab_matches_mode_a_at_layer_19_within_bfloat16_tolerance(
    real_runtime, monkeypatch
):
    materialization = _build_materialization(real_runtime, n_slots=3)
    n_slots = int(materialization.K.shape[-2])

    captures_a = _run_and_capture(
        real_runtime,
        monkeypatch,
        materialization=materialization,
        mode="a",
        layer_indices=(19,),
        capture_tensors=(19,),
    )
    captures_ab = _run_and_capture(
        real_runtime,
        monkeypatch,
        materialization=materialization,
        mode="ab",
        layer_indices=(19,),
        capture_tensors=(19,),
    )

    attn_a = captures_a[19]["attn_weights"]
    attn_ab = captures_ab[19]["attn_weights"]
    assert isinstance(attn_a, torch.Tensor)
    assert isinstance(attn_ab, torch.Tensor)
    assert attn_ab.shape == attn_a.shape
    assert attn_a.shape[-1] == captures_a[19]["q_len"] + n_slots
    assert attn_ab.shape[-1] == captures_ab[19]["q_len"] + n_slots

    mean_abs_diff = (attn_ab - attn_a).abs().mean().item()
    assert mean_abs_diff < 1e-2


@requires_cuda_gemma4
def test_all_modes_cleanup_lazarus_attrs_after_each_run(real_runtime, monkeypatch):
    layers = real_runtime._resolve_layers()
    materialization = _build_materialization(real_runtime, n_slots=3)

    for mode in ("a", "b", "ab", "none"):
        _run_and_capture(
            real_runtime,
            monkeypatch,
            materialization=materialization,
            mode=mode,
            layer_indices=(),
        )
        for layer_idx in CLEANUP_LAYER_INDICES:
            leaked_attrs = sorted(
                name
                for name in vars(layers[layer_idx].self_attn)
                if name.startswith("_lazarus_")
            )
            assert leaked_attrs == [], (
                f"mode={mode} layer={layer_idx} leaked Lazarus attrs: {leaked_attrs}"
            )
