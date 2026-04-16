"""Deterministic fixture builders (Validation matrix Appendix A).

Every builder exposes ``build(out_dir: Path, *, model: str | None = None,
seed: int = 0) -> Path`` returning the generated artifact path.  Builders
are intentionally tiny and dependency-light so the dispatcher can run on a
CUDA-only host without MLX installed (and vice versa).
"""

from __future__ import annotations

BUILDERS = (
    "build_vec",
    "build_corpus",
    "build_probe",
    "build_ds",
    "build_vector",
    "build_exp",
    "build_cls",
    "build_sft",
    "build_pairs",
    "build_grpo",
    "build_raw",
)

__all__ = ["BUILDERS"]
