"""Loader weight-name conversion parity test.

M-x1 extended-coverage deliverable (see ve-ins-0mo7014jk000083602f
axis-3 baseline). Pins the ``StandardWeightConverter`` mapping so a
CUDA-only refactor cannot silently rewire a weight name that the MLX
arm relies on. Written as a deterministic pure-python test (no
framework imports), so it runs on every host regardless of backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _helpers.mlx_snapshots import _write_residual  # noqa: E402

pytestmark = pytest.mark.parity


CASES = [
    ("model.embed_tokens.weight", "model.embed_tokens.weight.weight", False),
    ("model.norm.weight", "model.norm.weight", False),
    ("lm_head.weight", "lm_head.lm_head.weight", False),
    ("lm_head.weight", None, True),  # tied
    ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight", False),
    ("model.layers.5.rotary_emb.inv_freq", None, False),  # dropped
    ("not.a.known.pattern", None, False),
]


def test_standard_weight_converter_parity() -> None:
    from chuk_lazarus.inference.loader import StandardWeightConverter

    mismatches: list[tuple[str, object, object]] = []
    for hf_name, expected, tied in CASES:
        conv = StandardWeightConverter(tie_word_embeddings=tied)
        got = conv.convert(hf_name)
        if got != expected:
            mismatches.append((hf_name, expected, got))

    _write_residual(
        {
            "op": "inference.loader.StandardWeightConverter",
            "max_abs_err": 0.0 if not mismatches else float("inf"),
            "rtol_violation_count": len(mismatches),
            "shape": [len(CASES)],
            "dtype": "str",
            "input_hash": "",
            "atol": 0.0,
            "rtol": 0.0,
            "timestamp": "",
        }
    )

    assert not mismatches, f"weight-name converter drift: {mismatches}"
