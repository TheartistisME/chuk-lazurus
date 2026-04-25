"""Static layer-type enumeration for Gemma-4-E2B-it (axis-A §3.7 Step 0).

Hand-written, frozen constants captured from
``model.config.get_text_config().layer_types`` on the pinned snapshot
``b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf``, loaded on CUDA in
``torch.bfloat16``. This module is the source-of-truth for the global-
attention layer index set used by axis-BC PROP K.0 (sliding-window-hazard
gating during KV-direct propagation).

References:
    * Recipe ............ ``ve-ins-0modtwi7v0000ff6d88``
    * End-state ......... ``ve-ins-0moduw60z00000872b2``
    * Capture diagnostic  ``prod/validation/gemma4_e2b_it_layer_types_b4a60110.jsonl``
    * AMD 11 ............ sliding-window-hazard precondition for PROP K.0

The Gemma-4 transformers schema names the non-sliding layer type
``"full_attention"``; that is the literal stored in
``GEMMA4_E2B_IT_LAYER_TYPES`` below. The set
``GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS`` enumerates the indices where the
attention type is global/full (i.e. not sliding-window).

NEVER load the model at import time of this module. The constants are
captured upstream and verified for byte-for-byte agreement with the JSONL
fixture by ``tests/inference/backends/test_axis_A_gemma4_layers_enumeration.py``.
"""

from __future__ import annotations

from typing import Final

MODEL_SNAPSHOT_ID: Final[str] = "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"

GEMMA4_E2B_IT_LAYER_TYPES: Final[list[str]] = [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
]

LAYER_COUNT: Final[int] = 35

GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS: Final[frozenset[int]] = frozenset(
    {4, 9, 14, 19, 24, 29, 34}
)

__all__ = [
    "MODEL_SNAPSHOT_ID",
    "GEMMA4_E2B_IT_LAYER_TYPES",
    "LAYER_COUNT",
    "GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS",
]
