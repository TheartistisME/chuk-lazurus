#!/usr/bin/env python3
"""Dual-backend (MLX vs torch/CUDA) benchmark for GPT-OSS-Lite variants.

M-x2 deliverable. Companion to ``benchmark_simple.py`` (MLX-only memory
+ file-size baseline). Emits JSON artifacts into
``gpt-oss-lite-v2/benchmark_results/<backend>.<device>.json`` using the
perf-harness schema (see axis-4 baseline ve-ins-0mo7015fp0000871fbe and
``src/chuk_lazarus/bench/perf_compare.py``).

Usage::

    python gpt-oss-lite-v2/dual_backend_benchmark.py --backend mlx
    python gpt-oss-lite-v2/dual_backend_benchmark.py --backend torch --device cpu
    python gpt-oss-lite-v2/dual_backend_benchmark.py --backend torch --device cuda:0

Design choices (documented in
``docs/refactor/dual-backend-cuda-all-buckets/03-workstreams.md`` §9
Perf-harness contract):

1. Drives the SAME op for both backends so the resulting JSON files are
   directly comparable by ``perf_compare``.
2. Each op repeats ``N`` times; the reported ``ms_per_op`` is the median
   (resilient to outliers) and ``tokens_per_second`` is derived from the
   *total* tokens touched divided by total wall time.
3. Uses fixed RNG seeds; both backends see bitwise-identical input
   numpy arrays so the only difference is the backend math path.
4. The driver intentionally exercises small synthetic ops (matmul,
   softmax, layernorm-style reduction) rather than a full model — the
   goal is a stable op-level regression surface, not end-to-end
   generation fidelity.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


OPS = [
    # (op_name, shape, dtype)
    ("matmul.1024x1024", (1024, 1024), "float32"),
    ("matmul.2048x2048", (2048, 2048), "float32"),
    ("softmax.8x512x512", (8, 512, 512), "float32"),
    ("reduce_mean.8x4096", (8, 4096), "float32"),
]


def _rng_inputs(shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(seed=2026_04_20)
    return rng.standard_normal(shape).astype(np.float32)


def _time_op(fn, repeats: int, warmup: int) -> tuple[float, list[float]]:
    """Run ``fn`` ``warmup`` + ``repeats`` times, return (total_wall, per_run_ms)."""

    for _ in range(warmup):
        fn()
    per_run_ms: list[float] = []
    total_start = time.perf_counter()
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        per_run_ms.append((time.perf_counter() - t0) * 1000.0)
    total = time.perf_counter() - total_start
    return total, per_run_ms


def _tokens_touched(shape: tuple[int, ...]) -> int:
    """Elements in the output tensor — a proxy for ``tokens_per_second``."""

    n = 1
    for s in shape:
        n *= s
    return n


def _run_mlx(op: str, shape: tuple[int, ...], dtype: str, repeats: int, warmup: int) -> dict[str, Any]:
    import mlx.core as mx  # type: ignore

    arr = _rng_inputs(shape)
    x = mx.array(arr)
    if op.startswith("matmul."):
        y = mx.array(_rng_inputs(shape))

        def _fn():
            out = mx.matmul(x, y)
            mx.eval(out)

    elif op.startswith("softmax."):

        def _fn():
            out = mx.softmax(x, axis=-1)
            mx.eval(out)

    elif op.startswith("reduce_mean."):

        def _fn():
            out = mx.mean(x, axis=-1)
            mx.eval(out)

    else:
        raise ValueError(f"unknown op: {op}")

    total_wall, per_run = _time_op(_fn, repeats=repeats, warmup=warmup)
    return {
        "op": op,
        "ms_per_op": median(per_run),
        "tokens_per_second": (repeats * _tokens_touched(shape)) / total_wall if total_wall > 0 else 0.0,
        "wall_time_seconds": total_wall,
        "input_shape": list(shape),
        "dtype": dtype,
        "repeats": repeats,
        "per_run_ms_min": min(per_run),
        "per_run_ms_max": max(per_run),
    }


def _run_torch(
    op: str, shape: tuple[int, ...], dtype: str, device: str, repeats: int, warmup: int
) -> dict[str, Any]:
    import torch

    tdtype = torch.float32 if dtype == "float32" else torch.float32
    arr = _rng_inputs(shape)
    x = torch.as_tensor(arr, dtype=tdtype, device=device)

    if op.startswith("matmul."):
        y = torch.as_tensor(_rng_inputs(shape), dtype=tdtype, device=device)

        def _fn():
            out = torch.matmul(x, y)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            return out

    elif op.startswith("softmax."):

        def _fn():
            out = torch.softmax(x, dim=-1)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            return out

    elif op.startswith("reduce_mean."):

        def _fn():
            out = torch.mean(x, dim=-1)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            return out

    else:
        raise ValueError(f"unknown op: {op}")

    total_wall, per_run = _time_op(_fn, repeats=repeats, warmup=warmup)
    return {
        "op": op,
        "ms_per_op": median(per_run),
        "tokens_per_second": (repeats * _tokens_touched(shape)) / total_wall if total_wall > 0 else 0.0,
        "wall_time_seconds": total_wall,
        "input_shape": list(shape),
        "dtype": dtype,
        "repeats": repeats,
        "per_run_ms_min": min(per_run),
        "per_run_ms_max": max(per_run),
    }


def run(backend: str, device: str | None, repeats: int, warmup: int, out_dir: Path) -> Path:
    run_id = uuid.uuid4().hex[:12]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if backend == "mlx":
        dev = device or "mps"
        entries = [_run_mlx(op, shape, dtype, repeats, warmup) for op, shape, dtype in OPS]
    elif backend == "torch":
        import torch

        dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        entries = [
            _run_torch(op, shape, dtype, dev, repeats, warmup) for op, shape, dtype in OPS
        ]
    else:
        raise SystemExit(f"unknown backend: {backend}")

    payload = {
        "backend": backend,
        "device": dev,
        "run_id": run_id,
        "timestamp": ts,
        "repeats": repeats,
        "warmup": warmup,
        "entries": [
            {
                "backend": backend,
                "device": dev,
                "op": e["op"],
                "input_shape": e["input_shape"],
                "dtype": e["dtype"],
                "ms_per_op": e["ms_per_op"],
                "tokens_per_second": e["tokens_per_second"],
                "wall_time_seconds": e["wall_time_seconds"],
                "run_id": run_id,
                "timestamp": ts,
                "per_run_ms_min": e["per_run_ms_min"],
                "per_run_ms_max": e["per_run_ms_max"],
                "repeats": e["repeats"],
            }
            for e in entries
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    device_slug = dev.replace("/", "_").replace(":", "_")
    out_path = out_dir / f"{backend}.{device_slug}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MLX-vs-torch/CUDA op-level benchmark driver.")
    p.add_argument("--backend", choices=("mlx", "torch"), required=True)
    p.add_argument(
        "--device",
        default=None,
        help="Override device (cpu / cuda:0 / mps); defaults to backend default.",
    )
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent / "benchmark_results"),
        help="Directory to write <backend>.<device>.json into.",
    )
    args = p.parse_args(argv)

    out_path = run(
        backend=args.backend,
        device=args.device,
        repeats=args.repeats,
        warmup=args.warmup,
        out_dir=Path(args.out_dir),
    )
    gc.collect()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
