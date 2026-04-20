"""Type definitions for gym CLI commands.

This module contains Pydantic models and enums for gym commands.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from pydantic import Field

from .._base import CommandConfig, CommandResult, OutputMixin


class GymRunConfig(CommandConfig):
    """Configuration for gym run command.

    Attributes:
        tokenizer: Path or name of the tokenizer
        mock: Whether to use mock gym stream
        host: Server host
        port: Server port
        transport: Transport protocol
        output_mode: Output mode
        num_episodes: Number of episodes for mock
        steps_per_episode: Steps per episode for mock
        difficulty_min: Minimum difficulty
        difficulty_max: Maximum difficulty
        success_rate: Target success rate for mock
        buffer_size: Replay buffer size
        max_samples: Maximum samples to collect
        output: Output file path
        timeout: Connection timeout
        retries: Maximum retries
        seed: Random seed
    """

    tokenizer: str = Field(..., description="Tokenizer path or name")
    mock: bool = Field(default=True, description="Use mock gym stream")
    host: str = Field(default="localhost", description="Server host")
    port: int = Field(default=8023, ge=1, le=65535, description="Server port")
    transport: str = Field(default="telnet", description="Transport protocol")
    output_mode: str = Field(default="json", description="Output mode")
    num_episodes: int = Field(default=10, ge=1, description="Number of episodes")
    steps_per_episode: int = Field(default=20, ge=1, description="Steps per episode")
    difficulty_min: float = Field(default=0.0, ge=0.0, le=1.0, description="Min difficulty")
    difficulty_max: float = Field(default=1.0, ge=0.0, le=1.0, description="Max difficulty")
    success_rate: float = Field(default=0.7, ge=0.0, le=1.0, description="Target success rate")
    buffer_size: int = Field(default=10000, ge=1, description="Buffer size")
    max_samples: int | None = Field(default=None, ge=1, description="Max samples")
    output: Path | None = Field(default=None, description="Output file")
    timeout: float = Field(default=10.0, gt=0, description="Connection timeout")
    retries: int = Field(default=3, ge=0, description="Max retries")
    seed: int | None = Field(default=None, description="Random seed")
    backend: str | None = Field(default=None, description="Runtime backend (mlx|torch)")
    device: str | None = Field(default=None, description="Device override (e.g. cuda:0, mps, cpu)")

    @classmethod
    def from_args(cls, args: Namespace) -> GymRunConfig:
        """Create config from argparse Namespace."""
        return cls(
            tokenizer=args.tokenizer,
            mock=getattr(args, "mock", True),
            host=getattr(args, "host", "localhost"),
            port=getattr(args, "port", 8023),
            transport=getattr(args, "transport", "telnet"),
            output_mode=getattr(args, "output_mode", "json"),
            num_episodes=getattr(args, "num_episodes", 10),
            steps_per_episode=getattr(args, "steps_per_episode", 20),
            difficulty_min=getattr(args, "difficulty_min", 0.0),
            difficulty_max=getattr(args, "difficulty_max", 1.0),
            success_rate=getattr(args, "success_rate", 0.7),
            buffer_size=getattr(args, "buffer_size", 10000),
            max_samples=getattr(args, "max_samples", None),
            output=Path(args.output) if getattr(args, "output", None) else None,
            timeout=getattr(args, "timeout", 10.0),
            retries=getattr(args, "retries", 3),
            seed=getattr(args, "seed", None),
            backend=getattr(args, "backend", None),
            device=getattr(args, "device", None),
        )


class BenchmarkConfig(CommandConfig):
    """Configuration for benchmark command.

    Attributes:
        tokenizer: Tokenizer path or name
        dataset: Dataset path (optional)
        num_samples: Number of synthetic samples
        max_samples: Maximum samples to process
        max_length: Maximum sequence length
        token_budget: Token budget per batch
        bucket_edges: Bucket edge sizes
        seed: Random seed
    """

    tokenizer: str | None = Field(default=None, description="Tokenizer path or name")
    dataset: Path | None = Field(default=None, description="Dataset path")
    num_samples: int = Field(default=10000, ge=1, description="Synthetic samples")
    max_samples: int | None = Field(default=None, ge=1, description="Max samples")
    max_length: int = Field(default=2048, ge=1, description="Max sequence length")
    token_budget: int = Field(default=8192, ge=1, description="Token budget")
    bucket_edges: str = Field(default="128,256,512,1024", description="Bucket edges")
    seed: int = Field(default=42, description="Random seed")
    backend: str | None = Field(default=None, description="Runtime backend (mlx|torch)")
    device: str | None = Field(default=None, description="Device override (e.g. cuda:0, mps, cpu)")
    json_output: Path | None = Field(default=None, description="Emit benchmark result as JSON to this path")

    @classmethod
    def from_args(cls, args: Namespace) -> BenchmarkConfig:
        """Create config from argparse Namespace."""
        return cls(
            tokenizer=getattr(args, "tokenizer", None),
            dataset=Path(args.dataset) if getattr(args, "dataset", None) else None,
            num_samples=getattr(args, "num_samples", 10000),
            max_samples=getattr(args, "max_samples", None),
            max_length=getattr(args, "max_length", 2048),
            token_budget=getattr(args, "token_budget", 8192),
            bucket_edges=getattr(args, "bucket_edges", "128,256,512,1024"),
            seed=getattr(args, "seed", 42),
            backend=getattr(args, "backend", None),
            device=getattr(args, "device", None),
            json_output=Path(args.json_output) if getattr(args, "json_output", None) else None,
        )

    def get_bucket_edges(self) -> tuple[int, ...]:
        """Parse bucket edges string into tuple."""
        return tuple(int(x.strip()) for x in self.bucket_edges.split(","))


class GymRunResult(CommandResult, OutputMixin):
    """Result of gym run command.

    Attributes:
        total_samples: Total samples collected
        total_episodes: Total episodes completed
        buffer_size: Final buffer size
        success_rate: Overall success rate
        mean_difficulty: Mean difficulty
        mean_reward: Mean reward
        output_path: Path where buffer was saved
    """

    total_samples: int = Field(default=0, ge=0, description="Total samples")
    total_episodes: int = Field(default=0, ge=0, description="Total episodes")
    buffer_size: int = Field(default=0, ge=0, description="Buffer size")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate")
    mean_difficulty: float = Field(default=0.0, ge=0.0, description="Mean difficulty")
    mean_reward: float = Field(default=0.0, description="Mean reward")
    output_path: Path | None = Field(default=None, description="Output path")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [self.format_header("Gym Run Summary")]
        lines.append(self.format_field("Total samples", self.total_samples))
        lines.append(self.format_field("Total episodes", self.total_episodes))
        lines.append(self.format_field("Buffer size", self.buffer_size))
        lines.append(self.format_field("Success rate", f"{self.success_rate:.1%}"))
        lines.append(self.format_field("Mean difficulty", f"{self.mean_difficulty:.2f}"))
        lines.append(self.format_field("Mean reward", f"{self.mean_reward:.2f}"))
        if self.output_path:
            lines.append(self.format_field("Output", str(self.output_path)))
        return "\n".join(lines)


class BenchmarkResult(CommandResult, OutputMixin):
    """Result of benchmark command.

    Attributes:
        samples: Number of samples processed
        total_tokens: Total tokens
        plan_fingerprint: Plan fingerprint
        bucket_efficiency: Overall bucket efficiency
        packing_ratio: Packing ratio
        packing_efficiency: Packing efficiency
        token_budget_utilization: Token budget utilization
        microbatches: Number of microbatches
        backend: Resolved backend name (mlx|torch)
        device: Resolved device (cpu|cuda:0|mps)
        wall_time_seconds: Wall-clock time of the benchmark run
        run_id: Unique run identifier (hex, seeded)
        timestamp: UTC timestamp when the run completed
    """

    samples: int = Field(default=0, ge=0, description="Samples processed")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens")
    plan_fingerprint: str = Field(default="", description="Plan fingerprint")
    bucket_efficiency: float = Field(default=0.0, ge=0.0, le=1.0, description="Bucket efficiency")
    packing_ratio: float = Field(default=1.0, ge=0.0, description="Packing ratio")
    packing_efficiency: float = Field(default=0.0, ge=0.0, le=1.0, description="Packing efficiency")
    token_budget_utilization: float = Field(default=0.0, ge=0.0, description="Budget utilization")
    microbatches: int = Field(default=0, ge=0, description="Number of microbatches")
    backend: str | None = Field(default=None, description="Resolved backend name")
    device: str | None = Field(default=None, description="Resolved device")
    wall_time_seconds: float = Field(default=0.0, ge=0.0, description="Total wall-clock time")
    run_id: str = Field(default="", description="Unique run id")
    timestamp: str = Field(default="", description="UTC timestamp (ISO-8601)")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [self.format_header("Benchmark Summary")]
        lines.append(self.format_field("Samples", f"{self.samples:,}"))
        lines.append(self.format_field("Total tokens", f"{self.total_tokens:,}"))
        lines.append(self.format_field("Microbatches", f"{self.microbatches:,}"))
        lines.append(self.format_field("Bucket efficiency", f"{self.bucket_efficiency:.1%}"))
        lines.append(self.format_field("Packing ratio", f"{self.packing_ratio:.2f}x"))
        lines.append(self.format_field("Packing efficiency", f"{self.packing_efficiency:.1%}"))
        lines.append(
            self.format_field("Budget utilization", f"{self.token_budget_utilization:.1%}")
        )
        lines.append(self.format_field("Plan fingerprint", self.plan_fingerprint))
        if self.backend:
            lines.append(self.format_field("Backend", self.backend))
        if self.device:
            lines.append(self.format_field("Device", self.device))
        if self.wall_time_seconds:
            lines.append(self.format_field("Wall time", f"{self.wall_time_seconds:.3f}s"))
        return "\n".join(lines)

    def to_json_payload(self) -> dict:
        """Return the canonical JSON schema for perf-harness comparison.

        Schema matches the M-x2 deliverable (see axis-4 baseline
        ve-ins-0mo7015fp0000871fbe) and the perf_compare comparator under
        ``src/chuk_lazarus/bench/perf_compare.py``::

            {backend, device, op, input_shape, dtype, ms_per_op,
             tokens_per_second, wall_time_seconds, run_id, timestamp}

        The ``chuk-lazarus gym benchmark`` command is a *pipeline*
        benchmark (batching / packing), so we emit one entry per stage
        with ``op`` set to the stage name.
        """

        backend = self.backend or "unknown"
        device = self.device or "unknown"
        dtype = "int32"  # length values are integer sequence ids
        samples = float(self.samples) if self.samples else 0.0
        wall = float(self.wall_time_seconds) if self.wall_time_seconds else 0.0
        tokens_per_second = (
            (self.total_tokens / wall) if wall > 0 and self.total_tokens else 0.0
        )
        ms_per_op = (wall * 1000.0 / max(self.microbatches, 1)) if wall else 0.0
        return {
            "backend": backend,
            "device": device,
            "op": "gym.benchmark.pipeline",
            "input_shape": [int(self.samples), int(self.microbatches)],
            "dtype": dtype,
            "ms_per_op": ms_per_op,
            "tokens_per_second": tokens_per_second,
            "wall_time_seconds": wall,
            "run_id": self.run_id or "",
            "timestamp": self.timestamp or "",
            "metrics": {
                "bucket_efficiency": self.bucket_efficiency,
                "packing_ratio": self.packing_ratio,
                "packing_efficiency": self.packing_efficiency,
                "token_budget_utilization": self.token_budget_utilization,
                "plan_fingerprint": self.plan_fingerprint,
                "total_tokens": self.total_tokens,
                "samples": self.samples,
                "microbatches": self.microbatches,
            },
        }


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "GymRunConfig",
    "GymRunResult",
]
