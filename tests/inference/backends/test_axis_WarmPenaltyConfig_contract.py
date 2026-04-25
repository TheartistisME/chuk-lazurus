"""Axis-2 cross-file contract tests for ``WarmPenaltyConfig`` (run-4).

Acceptance criterion: enforce the script-kwargs <-> dataclass-fields contract
between ``scripts/interactive_memory_chat.py`` (the ``/kv_query`` REPL
construction site) and
``chuk_lazarus.inference.backends.torch_runtime.WarmPenaltyConfig``. Pure
unit tests on the dataclass alone do NOT catch the run-3 ``/kv_query`` REPL
crash class (a kwarg added at the call site that the dataclass does not
accept yields ``TypeError`` at construction); only construction-site
contract tests do. See critical-learning record ve-ins-0moe7opep00003ef31f
and the run-3 failure record ve-ins-0moe7elql0000afaa2b.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from chuk_lazarus.inference.backends.torch_runtime import (
    ATTENTION_TIER_MASK_SCHEMA_VERSION,
    AttentionTierMaskInputs,
    WarmPenaltyConfig,
    apply_tier_attention_mask,
    attention_tier_mask_inputs_from_dict,
    attention_tier_mask_inputs_to_dict,
)

try:  # torch is the project runtime; import here for the CUDA-gated test.
    import torch
except Exception:  # pragma: no cover - torch is always present in this repo.
    torch = None  # type: ignore[assignment]

from chuk_lazarus.session_retrieval.tier_policy import TierLabel


# This file lives at tests/inference/backends/<this>.py — three parents up
# is the repo root (which contains the scripts/ directory).
REPO_ROOT = Path(__file__).resolve().parents[3]
CHAT_SCRIPT = REPO_ROOT / "scripts" / "interactive_memory_chat.py"


requires_cuda = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="requires CUDA",
)


# ---------------------------------------------------------------------------
# 1. Construction with the new ``hot_bonus_value`` kwarg.
# ---------------------------------------------------------------------------


def test_construct_with_hot_bonus_value():
    # Full positional/explicit construction must accept hot_bonus_value.
    cfg = WarmPenaltyConfig(
        penalty_value=4.0,
        per_warm_uniform=True,
        clamp_min=None,
        hot_bonus_value=2.0,
    )
    assert cfg.hot_bonus_value == 2.0

    # Mirrors the chat-script call shape at scripts/interactive_memory_chat.py:1132
    # (kwarg-only; other fields default).
    cfg_kwarg_only = WarmPenaltyConfig(hot_bonus_value=2.0)
    assert cfg_kwarg_only.penalty_value == 4.0  # documented default
    assert cfg_kwarg_only.per_warm_uniform is True  # documented default
    assert cfg_kwarg_only.clamp_min is None  # documented default
    assert cfg_kwarg_only.hot_bonus_value == 2.0


# ---------------------------------------------------------------------------
# 2. Backward-compat: legacy 3-field construction must keep working.
# ---------------------------------------------------------------------------


def test_construct_without_hot_bonus_value():
    # All-default construction: hot_bonus_value defaults to None (identity HOT).
    cfg_defaults = WarmPenaltyConfig()
    assert cfg_defaults.hot_bonus_value is None

    # Legacy run-1/run-2/run-3 caller shape (3 fields, no hot_bonus_value).
    cfg_legacy = WarmPenaltyConfig(
        penalty_value=2.0, per_warm_uniform=False, clamp_min=-1.0
    )
    assert cfg_legacy.hot_bonus_value is None
    assert cfg_legacy.penalty_value == 2.0
    assert cfg_legacy.per_warm_uniform is False
    assert cfg_legacy.clamp_min == -1.0


# ---------------------------------------------------------------------------
# 3. STATIC AST contract: every kwarg the chat script passes must be a
#    dataclass field. This is the test that would have CAUGHT the run-3
#    REPL crash (script kwarg ``hot_bonus_value`` not in dataclass fields ->
#    TypeError -> /kv_query REPL crash).
# ---------------------------------------------------------------------------


def _collect_warm_penalty_config_kwargs(source_text: str) -> list[set[str]]:
    """Walk the AST for every ``WarmPenaltyConfig(...)`` call site.

    Returns a list of kwarg-name sets, one per call site. Detects both
    bare-name calls (``WarmPenaltyConfig(...)``) and attribute-style calls
    (``module.WarmPenaltyConfig(...)``). Positional args are ignored — the
    contract only spans the kwarg surface.
    """
    tree = ast.parse(source_text)
    found: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        target_name: str | None = None
        if isinstance(func, ast.Name):
            target_name = func.id
        elif isinstance(func, ast.Attribute):
            target_name = func.attr
        if target_name != "WarmPenaltyConfig":
            continue
        kwarg_names: set[str] = set()
        for keyword in node.keywords:
            if keyword.arg is None:
                # Skip ``**mapping`` splats — cannot statically validate.
                continue
            kwarg_names.add(keyword.arg)
        found.append(kwarg_names)
    return found


def test_chat_script_construction_path_kwarg_alignment():
    # WHY: this is the only test shape that catches "script adds new kwarg
    # that dataclass does not accept" -> TypeError at construction. See
    # ve-ins-0moe7opep00003ef31f (critical learning) and
    # ve-ins-0moe7elql0000afaa2b (run-3 /kv_query REPL crash record).
    if not CHAT_SCRIPT.is_file():
        pytest.skip(
            f"chat script not present at {CHAT_SCRIPT} - nothing to contract-test"
        )

    source_text = CHAT_SCRIPT.read_text(encoding="utf-8")
    call_kwarg_sets = _collect_warm_penalty_config_kwargs(source_text)
    if not call_kwarg_sets:
        pytest.skip(
            "WarmPenaltyConfig() call not present in chat script - "
            "nothing to contract-test"
        )

    valid_field_names = {f.name for f in dataclasses.fields(WarmPenaltyConfig)}

    for call_idx, kwarg_names in enumerate(call_kwarg_sets):
        unknown = kwarg_names - valid_field_names
        assert not unknown, (
            f"WarmPenaltyConfig call site #{call_idx} in {CHAT_SCRIPT.name} "
            f"passes kwarg(s) {sorted(unknown)!r} not present in dataclass "
            f"fields {sorted(valid_field_names)!r}. This is the run-3 "
            "/kv_query REPL crash class - either add the field to "
            "WarmPenaltyConfig or remove the kwarg from the call site."
        )


# ---------------------------------------------------------------------------
# 4. Tensor behaviour: HOT bonus is additive on the HOT slice; WARM penalty
#    is unaffected by the HOT-bonus toggle.
# ---------------------------------------------------------------------------


@requires_cuda
def test_bonus_arithmetic_applied_when_hot():
    # WHY: locks the documented HOT-bonus contract (logit += hot_bonus_value)
    # and proves WARM behaviour is independent of the HOT-bonus toggle.
    # Per AMD 14 + user feedback memory: tensors live on device='cuda'.
    baseline = torch.zeros(1, 1, 1, 4, dtype=torch.float32, device="cuda")

    per_window_token_ranges = {0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 4)}
    tier_assignments = {
        0: TierLabel.HOT,
        1: TierLabel.HOT,
        2: TierLabel.WARM,
        3: TierLabel.WARM,
    }

    # Call A: identity-HOT baseline (no bonus).
    inputs_a = AttentionTierMaskInputs(
        per_window_token_ranges=per_window_token_ranges,
        tier_assignments=tier_assignments,
        warm_config=WarmPenaltyConfig(),  # hot_bonus_value defaults to None
    )
    out_a = apply_tier_attention_mask(baseline, inputs_a)

    # Call B: same wiring but HOT bonus = 2.0.
    inputs_b = AttentionTierMaskInputs(
        per_window_token_ranges=per_window_token_ranges,
        tier_assignments=tier_assignments,
        warm_config=WarmPenaltyConfig(hot_bonus_value=2.0),
    )
    out_b = apply_tier_attention_mask(baseline, inputs_b)

    # HOT slice (indices 0..2): A is identity (zero), B has +2.0 bonus.
    assert torch.equal(out_a[..., 0:2], torch.zeros_like(out_a[..., 0:2]))
    assert torch.equal(
        out_b[..., 0:2], torch.full_like(out_b[..., 0:2], 2.0)
    )

    # WARM slice (indices 2..4): both A and B subtract penalty_value (default
    # 4.0) and must therefore agree byte-for-byte.
    assert torch.equal(out_a[..., 2:4], out_b[..., 2:4])
    # Sanity: the WARM penalty actually fired (logits should equal -4.0).
    assert torch.equal(
        out_a[..., 2:4], torch.full_like(out_a[..., 2:4], -4.0)
    )


# ---------------------------------------------------------------------------
# 5. JSON envelope round-trip: hot_bonus_value preserved; legacy envelopes
#    (no hot_bonus_value key) deserialize cleanly with default None. Schema
#    version stays at 1 (no bump).
# ---------------------------------------------------------------------------


def _make_envelope_inputs(hot_bonus_value: float | None) -> AttentionTierMaskInputs:
    return AttentionTierMaskInputs(
        per_window_token_ranges={0: (0, 2), 1: (2, 4)},
        tier_assignments={0: TierLabel.HOT, 1: TierLabel.WARM},
        warm_config=WarmPenaltyConfig(hot_bonus_value=hot_bonus_value),
    )


def test_json_envelope_round_trip_preserves_hot_bonus_value():
    # Round-trip with a concrete HOT bonus.
    inputs_with_bonus = _make_envelope_inputs(hot_bonus_value=3.5)
    envelope = attention_tier_mask_inputs_to_dict(inputs_with_bonus)
    assert envelope["schema_version"] == ATTENTION_TIER_MASK_SCHEMA_VERSION
    assert envelope["warm_config"]["hot_bonus_value"] == 3.5
    restored = attention_tier_mask_inputs_from_dict(envelope)
    assert restored.warm_config.hot_bonus_value == 3.5

    # Round-trip with the default (None) preserves None.
    inputs_default = _make_envelope_inputs(hot_bonus_value=None)
    envelope_default = attention_tier_mask_inputs_to_dict(inputs_default)
    assert envelope_default["warm_config"]["hot_bonus_value"] is None
    restored_default = attention_tier_mask_inputs_from_dict(envelope_default)
    assert restored_default.warm_config.hot_bonus_value is None

    # Legacy envelope (run-1/2/3 shape: no hot_bonus_value key) must
    # deserialize cleanly with hot_bonus_value defaulting to None.
    legacy_envelope = {
        "schema_version": ATTENTION_TIER_MASK_SCHEMA_VERSION,
        "per_window_token_ranges": {"0": [0, 2], "1": [2, 4]},
        "tier_assignments": {"0": "hot", "1": "warm"},
        "warm_config": {
            "penalty_value": 4.0,
            "per_warm_uniform": True,
            "clamp_min": None,
            # NOTE: hot_bonus_value key intentionally omitted to mimic legacy.
        },
        "warm_scores": None,
    }
    restored_legacy = attention_tier_mask_inputs_from_dict(legacy_envelope)
    assert restored_legacy.warm_config.hot_bonus_value is None
    assert restored_legacy.warm_config.penalty_value == 4.0
    assert restored_legacy.warm_config.per_warm_uniform is True
    assert restored_legacy.warm_config.clamp_min is None
