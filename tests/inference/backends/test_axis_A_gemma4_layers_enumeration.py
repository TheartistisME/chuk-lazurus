"""Axis-A §3.7 Step 0 — Gemma-4-E2B-it layer-type enumeration regression.

CUDA-only. Reloads the pinned snapshot in bf16 and verifies that the
hand-frozen constants in
``chuk_lazarus.inference.context.knowledge.gemma4_e2b_it_layers`` agree
byte-for-byte with ``model.config.get_text_config().layer_types`` AND with
the prod/validation JSONL fixture (D2). This is the D5 cross-artefact
agreement check guarding axis-BC PROP K.0 (sliding-window-hazard gating).

Per AMD 14 the skip reason names the precise dependency
(``torch.cuda.is_available()``); we never frame missing CUDA as
``hardware-absence``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "torch.cuda.is_available()==False; axis-A capture validated on CUDA only",
        allow_module_level=True,
    )

from chuk_lazarus.inference.context.knowledge.gemma4_e2b_it_layers import (
    GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS,
    GEMMA4_E2B_IT_LAYER_TYPES,
    LAYER_COUNT,
    MODEL_SNAPSHOT_ID,
)


GEMMA4_SNAPSHOT_PATH = Path(
    "/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/"
    "snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
)

PROD_VALIDATION_JSONL = (
    Path(__file__).resolve().parents[3]
    / "prod/validation/gemma4_e2b_it_layer_types_b4a60110.jsonl"
)


@pytest.fixture(scope="module")
def gemma4_model():
    """Load Gemma-4-E2B-it once per module on CUDA bf16, then free."""
    if not GEMMA4_SNAPSHOT_PATH.is_dir():
        pytest.skip(
            "gemma-4-e2b-it-snapshot-missing-locally: "
            f"expected directory at {GEMMA4_SNAPSHOT_PATH}"
        )

    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            str(GEMMA4_SNAPSHOT_PATH),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .to("cuda")
        .eval()
    )
    try:
        yield model
    finally:
        del model
        torch.cuda.empty_cache()


def _runtime_layer_types(model) -> list[str]:
    cfg = model.config
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return list(text_cfg.layer_types)


def test_snapshot_id_matches_constant():
    assert MODEL_SNAPSHOT_ID == "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"


def test_layer_count_matches_runtime(gemma4_model):
    runtime_types = _runtime_layer_types(gemma4_model)
    assert LAYER_COUNT == len(runtime_types) == len(GEMMA4_E2B_IT_LAYER_TYPES) == 35


def test_layer_types_byte_for_byte(gemma4_model):
    runtime_types = _runtime_layer_types(gemma4_model)
    # Element-wise equality: every position must match the frozen constant.
    assert runtime_types == GEMMA4_E2B_IT_LAYER_TYPES


def test_global_attention_indices_match(gemma4_model):
    runtime_types = _runtime_layer_types(gemma4_model)
    # The transformers Gemma-4 schema uses the literal "full_attention" for
    # the non-sliding (global) layer type. We accept either that or the
    # legacy "global_attention" alias for forward-compat.
    GLOBAL_ALIASES = {"full_attention", "global_attention"}
    derived = frozenset(
        i for i, t in enumerate(runtime_types) if t in GLOBAL_ALIASES
    )
    assert derived == GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS
    # Concrete check against the captured set (axis-BC PROP K.0 input).
    assert derived == frozenset({4, 9, 14, 19, 24, 29, 34})


def test_jsonl_fixture_agrees_with_module():
    assert PROD_VALIDATION_JSONL.is_file(), (
        f"D2 prod/validation JSONL missing at {PROD_VALIDATION_JSONL}"
    )
    raw = PROD_VALIDATION_JSONL.read_text(encoding="utf-8").strip()
    record = json.loads(raw)

    assert record["model_snapshot_id"] == MODEL_SNAPSHOT_ID
    assert record["layer_count"] == LAYER_COUNT
    assert record["layer_types"] == GEMMA4_E2B_IT_LAYER_TYPES
    assert record["device"] == "cuda"
    assert record["dtype"] == "torch.bfloat16"
    assert record["axis"] == "A"
    assert record["mission"] == "chuk-lazurus-164"
    assert record["run"] == 1
    assert record["capture_method"] == "model.config.layer_types"

    # D5 byte-for-byte agreement: JSONL global indices == module frozenset.
    jsonl_indices = sorted(record["global_attention_layer_indices"])
    module_indices = sorted(GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS)
    assert jsonl_indices == module_indices
