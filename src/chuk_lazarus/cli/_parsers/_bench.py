"""Bench command parser.

EWS-15 owns only the parser registration here; the handler
``bench_pipeline`` is owned by EWS-14 in ``cli/commands/gym/benchmark.py``.
The shared ``--backend``/``--device`` flags are registered via
``add_backend_flags`` (EWS-0).
"""

from ..commands._base import add_backend_flags
from ..commands.gym import bench_pipeline


def register_bench_parser(subparsers):
    """Register the bench subcommand."""
    bench_parser = subparsers.add_parser(
        "bench",
        help="Benchmark the batching pipeline",
        description="Run comprehensive benchmarks on tokenization, batching, packing, and efficiency.",
    )
    add_backend_flags(bench_parser)
    bench_parser.add_argument(
        "-d",
        "--dataset",
        help="JSONL dataset file (optional - uses synthetic data if not provided)",
    )
    bench_parser.add_argument(
        "-t",
        "--tokenizer",
        default="gpt2",
        help="Tokenizer to use (default: gpt2)",
    )
    bench_parser.add_argument(
        "--bucket-edges",
        default="128,256,512,1024",
        help="Bucket edge lengths (comma-separated, default: 128,256,512,1024)",
    )
    bench_parser.add_argument(
        "--token-budget",
        type=int,
        default=4096,
        help="Token budget per microbatch (default: 4096)",
    )
    bench_parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    bench_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to process from dataset",
    )
    bench_parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples (when no dataset, default: 1000)",
    )
    bench_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    bench_parser.add_argument(
        "--json",
        dest="json_output",
        metavar="PATH",
        default=None,
        help=(
            "Emit a JSON artifact at PATH using the perf-harness schema "
            "(backend/device/op/input_shape/dtype/ms_per_op/"
            "tokens_per_second/wall_time_seconds/run_id/timestamp). "
            "Consumed by src/chuk_lazarus/bench/perf_compare.py."
        ),
    )
    bench_parser.set_defaults(func=bench_pipeline)
