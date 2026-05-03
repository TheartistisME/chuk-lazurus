"""DFlash planning and verification utilities.

This module intentionally stops short of defining a trainable Gemma draft
checkpoint. DFlash needs model-family-specific draft weights; these helpers
capture the parts we can make deterministic now: target hidden-layer selection,
Gemma-4 E2B calibration defaults, context feature extraction, and lossless
speculative prefix acceptance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class DFlashCalibration:
    """Target-model calibration values used by a DFlash drafter."""

    route_layer: int | None
    boundary_layer: int | None
    kv_source_layer: int | None
    kv_target_layer: int | None
    insertion_family: str | None
    confidence: str | None = None
    model_revision_or_hash: str | None = None


@dataclass(frozen=True)
class DFlashPlan:
    """Serializable config needed to build/train a DFlash draft model."""

    model_family: str
    num_target_layers: int
    target_hidden_size: int
    draft_num_layers: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    mask_token_id: int | None
    calibration: DFlashCalibration
    notes: tuple[str, ...] = ()

    def as_config(self) -> dict[str, Any]:
        """Return a draft-model config fragment."""
        return {
            "block_size": self.block_size,
            "num_target_layers": self.num_target_layers,
            "dflash_config": {
                "target_layer_ids": list(self.target_layer_ids),
                "mask_token_id": self.mask_token_id,
                "calibration": {
                    "route_layer": self.calibration.route_layer,
                    "boundary_layer": self.calibration.boundary_layer,
                    "kv_source_layer": self.calibration.kv_source_layer,
                    "kv_target_layer": self.calibration.kv_target_layer,
                    "insertion_family": self.calibration.insertion_family,
                    "confidence": self.calibration.confidence,
                    "model_revision_or_hash": self.calibration.model_revision_or_hash,
                },
                "notes": list(self.notes),
            },
        }


@dataclass(frozen=True)
class DFlashAcceptance:
    """Longest verified speculative prefix and the target bonus token."""

    accepted_length: int
    accepted_token_ids: tuple[int, ...]
    bonus_token_id: int | None


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> tuple[int, ...]:
    """Match the official DFlash uniform hidden-feature rule.

    The Qwen implementation samples from layer 1 through layer N-3 inclusive.
    Hidden-state tensors returned by Hugging Face include the embedding state
    at index 0, so callers should add one when indexing `output.hidden_states`.
    """
    if num_target_layers <= 0:
        raise ValueError(f"num_target_layers must be positive, got {num_target_layers}")
    if num_draft_layers <= 0:
        raise ValueError(f"num_draft_layers must be positive, got {num_draft_layers}")
    if num_draft_layers == 1:
        return (num_target_layers // 2,)

    start = 1
    end = max(start, num_target_layers - 3)
    span = end - start
    return tuple(
        int(round(start + (idx * span) / (num_draft_layers - 1)))
        for idx in range(num_draft_layers)
    )


def _dedupe_preserving_order(layer_ids: Sequence[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    result: list[int] = []
    for layer_id in layer_ids:
        value = int(layer_id)
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def build_gemma4_e2b_dflash_plan(
    *,
    mask_token_id: int | None = None,
    draft_num_layers: int = 5,
    block_size: int = 16,
    calibrated_context: bool = True,
) -> DFlashPlan:
    """Return the current Gemma-4 E2B DFlash plan.

    The default context layers intentionally include the locally measured
    13->14 handoff. The uniform Qwen rule for a 35-layer target would be
    (1, 9, 17, 24, 32), which is useful as an ablation but ignores the measured
    full-attention seam.
    """
    uniform_layers = build_target_layer_ids(35, draft_num_layers)
    if calibrated_context and draft_num_layers == 5:
        target_layer_ids = (4, 9, 13, 14, 29)
    else:
        target_layer_ids = uniform_layers

    calibration = DFlashCalibration(
        route_layer=13,
        boundary_layer=13,
        kv_source_layer=13,
        kv_target_layer=14,
        insertion_family="full_attention",
        confidence="high",
        model_revision_or_hash="b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf",
    )

    notes = (
        "Gemma-4 E2B has 35 text decoder layers, hidden size 1536, 8 Q heads, 1 KV head.",
        "Full-attention layers are 4,9,14,19,24,29,34; local calibration prefers 13->14.",
        f"Uniform DFlash baseline would select {uniform_layers}.",
        "mask_token_id is unset until a Gemma tokenizer strategy is chosen.",
    )
    return DFlashPlan(
        model_family="gemma4_e2b",
        num_target_layers=35,
        target_hidden_size=1536,
        draft_num_layers=draft_num_layers,
        block_size=block_size,
        target_layer_ids=_dedupe_preserving_order(target_layer_ids),
        mask_token_id=mask_token_id,
        calibration=calibration,
        notes=notes,
    )


def _cat_tensors(tensors: Sequence[Any], *, dim: int) -> Any:
    if not tensors:
        raise ValueError("cannot concatenate an empty tensor sequence")
    first = tensors[0]
    module = type(first).__module__
    if module.startswith("torch"):
        import torch

        return torch.cat(list(tensors), dim=dim)
    if module.startswith("mlx"):
        import mlx.core as mx

        return mx.concatenate(list(tensors), axis=dim)
    if module.startswith("numpy"):
        import numpy as np

        return np.concatenate(list(tensors), axis=dim)
    raise TypeError(f"unsupported tensor type for DFlash context feature: {type(first)!r}")


def extract_context_feature(
    hidden_states: Sequence[Any],
    layer_ids: Sequence[int],
    *,
    hidden_state_offset: int = 1,
) -> Any:
    """Concatenate selected target hidden states along the hidden dimension."""
    if not layer_ids:
        raise ValueError("layer_ids must not be empty")
    selected = []
    for layer_id in layer_ids:
        index = int(layer_id) + int(hidden_state_offset)
        if index < 0 or index >= len(hidden_states):
            raise IndexError(
                f"hidden state index {index} for target layer {layer_id} is outside "
                f"available hidden_states length {len(hidden_states)}"
            )
        selected.append(hidden_states[index])
    return _cat_tensors(selected, dim=-1)


def accept_speculated_prefix(
    draft_token_ids: Sequence[int],
    posterior_token_ids: Sequence[int],
) -> DFlashAcceptance:
    """Verify a DFlash block using the target posterior tokens.

    `draft_token_ids` should be the proposed block beginning with the clean
    target token from the previous step. `posterior_token_ids` should be the
    target model's sampled/argmax token for each draft position. DFlash accepts
    draft positions 1..k while they equal posterior positions 0..k-1, then uses
    posterior[k] as the target bonus token at the first mismatch.
    """
    draft = tuple(int(token_id) for token_id in draft_token_ids)
    posterior = tuple(int(token_id) for token_id in posterior_token_ids)
    if not draft:
        raise ValueError("draft_token_ids must contain at least the clean anchor token")
    if len(posterior) < len(draft):
        raise ValueError(
            "posterior_token_ids must be at least as long as draft_token_ids "
            f"({len(posterior)} < {len(draft)})"
        )

    matched_mask_tokens = 0
    for draft_idx in range(1, len(draft)):
        posterior_idx = draft_idx - 1
        if draft[draft_idx] != posterior[posterior_idx]:
            break
        matched_mask_tokens += 1

    accepted_length = matched_mask_tokens + 1
    bonus_index = matched_mask_tokens
    bonus_token_id = posterior[bonus_index] if bonus_index < len(posterior) else None
    return DFlashAcceptance(
        accepted_length=accepted_length,
        accepted_token_ids=draft[:accepted_length],
        bonus_token_id=bonus_token_id,
    )


def load_dflash_calibration(path: str | Path) -> DFlashCalibration:
    """Load David/get_model_config.py JSON output as a DFlash calibration."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    candidate = data.get("adapter_config_candidate") or {}
    model = data.get("model") or {}
    return DFlashCalibration(
        route_layer=candidate.get("route_layer"),
        boundary_layer=candidate.get("boundary_layer"),
        kv_source_layer=candidate.get("kv_source_layer"),
        kv_target_layer=candidate.get("kv_target_layer"),
        insertion_family=candidate.get("insertion_family"),
        confidence=candidate.get("confidence"),
        model_revision_or_hash=candidate.get("model_revision_or_hash")
        or model.get("model_revision_or_hash"),
    )
