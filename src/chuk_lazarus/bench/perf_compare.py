"""Perf-harness comparator: MLX vs torch/CUDA, op-by-op.

Reads two JSON artifacts produced by
``gpt-oss-lite-v2/dual_backend_benchmark.py`` (or any tool that emits
the same schema — ``chuk-lazarus gym benchmark --json <path>`` also
produces a compatible single-op entry) and prints / returns an op-by-op
comparison with a PASS/FAIL verdict.

The verdict is controlled by a single config knob:

    cuda_to_mlx_ratio_floor: float (default 1.0)

        We PASS op when ``cuda_ms <= mlx_ms * cuda_to_mlx_ratio_floor``.
        A floor of 1.0 means "CUDA must be at least as fast as MLX".
        Raising to 1.25 admits up to 25% regression; set >1 only during
        bring-up. The config can also be passed inline via ``--factor``.

Artifacts:
    Each artifact is a dict with ``entries: [ {op, ms_per_op,
    tokens_per_second, input_shape, dtype, ...} ]``. We match entries
    across the two artifacts by ``op`` (string equality).

CLI::

    python -m chuk_lazarus.bench.perf_compare A.json B.json
    python -m chuk_lazarus.bench.perf_compare A.json B.json --factor 1.25
    python -m chuk_lazarus.bench.perf_compare A.json B.json --json out.json

Tolerance contract anchor:
    docs/refactor/dual-backend-cuda-all-buckets/03-workstreams.md
    §9 (Perf-harness contract) — axis-4 baseline
    ve-ins-0mo7015fp0000871fbe.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_RATIO_FLOOR = 1.0


@dataclasses.dataclass
class OpComparison:
    op: str
    mlx_ms: float | None
    cuda_ms: float | None
    ratio: float | None  # cuda_ms / mlx_ms — lower is better for CUDA
    verdict: str  # PASS | FAIL | MISSING
    message: str


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _index_entries(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = artifact.get("entries") or []
    if not entries and {"op", "ms_per_op"}.issubset(artifact.keys()):
        # Single-op payload (e.g. `chuk-lazarus gym benchmark --json`).
        entries = [artifact]
    return {e["op"]: e for e in entries}


def _classify_side(artifact: dict[str, Any]) -> str:
    """Return 'mlx' or 'cuda' based on backend/device fields.

    Torch on CPU is treated as the "cuda" side for comparison purposes
    when no dedicated CUDA artifact is available — the real contract
    gate lives in downstream CI which knows whether CUDA is present.
    """

    backend = (artifact.get("backend") or "").lower()
    if backend == "mlx":
        return "mlx"
    if backend == "torch":
        return "cuda"
    return "unknown"


def compare(
    a: dict[str, Any],
    b: dict[str, Any],
    ratio_floor: float = DEFAULT_RATIO_FLOOR,
) -> list[OpComparison]:
    """Return one ``OpComparison`` per op present in either artifact.

    The artifacts may be passed in any order; the classifier maps each
    to "mlx" or "cuda" using backend metadata.
    """

    side_a = _classify_side(a)
    side_b = _classify_side(b)
    if side_a == side_b:
        raise ValueError(
            f"both artifacts report backend={side_a}; perf_compare needs one "
            "mlx and one torch/cuda artifact."
        )
    mlx_art, cuda_art = (a, b) if side_a == "mlx" else (b, a)

    mlx_ops = _index_entries(mlx_art)
    cuda_ops = _index_entries(cuda_art)
    all_ops = sorted(set(mlx_ops) | set(cuda_ops))

    out: list[OpComparison] = []
    for op in all_ops:
        m = mlx_ops.get(op)
        c = cuda_ops.get(op)
        mlx_ms = float(m["ms_per_op"]) if m and "ms_per_op" in m else None
        cuda_ms = float(c["ms_per_op"]) if c and "ms_per_op" in c else None

        if mlx_ms is None or cuda_ms is None:
            out.append(
                OpComparison(
                    op=op,
                    mlx_ms=mlx_ms,
                    cuda_ms=cuda_ms,
                    ratio=None,
                    verdict="MISSING",
                    message="op missing from one artifact",
                )
            )
            continue

        ratio = cuda_ms / mlx_ms if mlx_ms > 0 else float("inf")
        if ratio <= ratio_floor:
            verdict = "PASS"
            msg = f"cuda {cuda_ms:.3f}ms <= mlx {mlx_ms:.3f}ms × {ratio_floor}"
        else:
            verdict = "FAIL"
            msg = (
                f"cuda {cuda_ms:.3f}ms > mlx {mlx_ms:.3f}ms × {ratio_floor} "
                f"(ratio={ratio:.3f})"
            )
        out.append(
            OpComparison(
                op=op, mlx_ms=mlx_ms, cuda_ms=cuda_ms, ratio=ratio,
                verdict=verdict, message=msg,
            )
        )
    return out


def print_table(results: list[OpComparison]) -> None:
    header = f"{'op':<32} {'mlx_ms':>10} {'cuda_ms':>10} {'ratio':>8} {'verdict':<8}"
    print(header)
    print("-" * len(header))
    for r in results:
        mlx_str = f"{r.mlx_ms:.3f}" if r.mlx_ms is not None else "-"
        cuda_str = f"{r.cuda_ms:.3f}" if r.cuda_ms is not None else "-"
        ratio_str = f"{r.ratio:.3f}" if r.ratio is not None else "-"
        print(f"{r.op:<32} {mlx_str:>10} {cuda_str:>10} {ratio_str:>8} {r.verdict:<8}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("artifact_a", type=Path, help="First JSON artifact (mlx or torch).")
    p.add_argument("artifact_b", type=Path, help="Second JSON artifact (the other backend).")
    p.add_argument(
        "--factor",
        type=float,
        default=float(os.environ.get("CHUK_PERF_RATIO_FLOOR", DEFAULT_RATIO_FLOOR)),
        help=f"cuda_to_mlx ratio floor (default: {DEFAULT_RATIO_FLOOR}).",
    )
    p.add_argument(
        "--json",
        dest="json_out",
        type=Path,
        default=None,
        help="Write comparison results as JSON to this path.",
    )
    args = p.parse_args(argv)

    a = _load(args.artifact_a)
    b = _load(args.artifact_b)
    results = compare(a, b, ratio_floor=args.factor)

    print_table(results)

    if args.json_out is not None:
        payload = {
            "artifact_a": str(args.artifact_a),
            "artifact_b": str(args.artifact_b),
            "ratio_floor": args.factor,
            "results": [dataclasses.asdict(r) for r in results],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nwrote {args.json_out}")

    failures = [r for r in results if r.verdict == "FAIL"]
    missing = [r for r in results if r.verdict == "MISSING"]
    print(
        f"\n{len(results)} ops compared, {len(failures)} FAIL, {len(missing)} MISSING"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
