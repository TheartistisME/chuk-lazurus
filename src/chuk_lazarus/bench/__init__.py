"""Cross-backend performance harness utilities.

This package hosts the perf-harness comparator used by the M-x2
deliverable. The pipeline is::

    gpt-oss-lite-v2/dual_backend_benchmark.py --backend mlx  --out-dir ...
    gpt-oss-lite-v2/dual_backend_benchmark.py --backend torch --out-dir ...
    python -m chuk_lazarus.bench.perf_compare \
        gpt-oss-lite-v2/benchmark_results/mlx.mps.json \
        gpt-oss-lite-v2/benchmark_results/torch.cuda_0.json

The comparator reads two JSON artifacts (schema documented in
``docs/refactor/dual-backend-cuda-all-buckets/03-workstreams.md`` §9
Perf-harness contract) and emits a per-op ratio + PASS/FAIL verdict.
"""

from __future__ import annotations

__all__ = ["perf_compare"]
