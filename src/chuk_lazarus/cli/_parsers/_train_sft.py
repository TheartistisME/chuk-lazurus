"""SFT + ``generate`` parser registrars (split from legacy ``_train.py``).

EWS-0 carved these out verbatim.  Ownership transfers to EWS-10 (train
bucket) at Epic 2 merge time.
"""

import asyncio

from ..commands._base import add_backend_flags
from ..commands.train import generate_data_cmd, train_sft_cmd


def register_train_sft_parser(train_subparsers) -> None:
    """Register the ``train sft`` subcommand on the given action."""

    sft_parser = train_subparsers.add_parser("sft", help="Supervised Fine-Tuning")
    sft_parser.add_argument("--model", required=True, help="Model name or path")
    sft_parser.add_argument("--data", required=True, help="Training data path (JSONL)")
    sft_parser.add_argument("--eval-data", help="Evaluation data path (JSONL)")
    sft_parser.add_argument("--output", default="./checkpoints/sft", help="Output directory")
    sft_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    sft_parser.add_argument("--max-steps", type=int, help="Max training steps (overrides epochs)")
    sft_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    sft_parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    sft_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    sft_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    sft_parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    sft_parser.add_argument(
        "--lora-targets",
        default="q_proj,v_proj",
        help="Comma-separated LoRA target modules (default: q_proj,v_proj). "
        "Options: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    sft_parser.add_argument(
        "--freeze-layers",
        help="Layers to freeze (e.g., '0-12' or '0,1,2,3'). Frozen layers are not trained.",
    )
    sft_parser.add_argument(
        "--config",
        help="YAML config file (overrides other arguments)",
    )
    sft_parser.add_argument("--mask-prompt", action="store_true", help="Mask prompt in loss")
    sft_parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    sft_parser.add_argument("--batchplan", help="Use pre-computed batch plan directory")
    sft_parser.add_argument(
        "--bucket-edges",
        help="Bucket edges for length-based batching (e.g., 128,256,512)",
    )
    sft_parser.add_argument(
        "--token-budget",
        type=int,
        help="Token budget for dynamic batching (replaces --batch-size)",
    )
    sft_parser.add_argument("--pack", action="store_true", help="Enable sequence packing")
    sft_parser.add_argument("--pack-max-len", type=int, help="Max length for packed sequences")
    sft_parser.add_argument(
        "--pack-mode",
        choices=["first_fit", "best_fit", "greedy"],
        default="first_fit",
        help="Packing algorithm",
    )
    sft_parser.add_argument(
        "--online",
        action="store_true",
        help="Enable online training with gym stream",
    )
    sft_parser.add_argument(
        "--gym-host",
        default="localhost",
        help="Gym server host for online training",
    )
    sft_parser.add_argument(
        "--gym-port",
        type=int,
        default=8023,
        help="Gym server port for online training",
    )
    sft_parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="Replay buffer size for online training",
    )
    add_backend_flags(sft_parser)
    sft_parser.set_defaults(func=lambda args: asyncio.run(train_sft_cmd(args)))


def register_generate_parser(subparsers) -> None:
    """Register the top-level ``generate`` subcommand."""

    gen_parser = subparsers.add_parser("generate", help="Generate training data")
    gen_parser.add_argument("--type", required=True, choices=["math"], help="Data type")
    gen_parser.add_argument("--output", default="./data/generated", help="Output directory")
    gen_parser.add_argument("--sft-samples", type=int, default=10000, help="SFT samples")
    gen_parser.add_argument("--dpo-samples", type=int, default=5000, help="DPO samples")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_parser.set_defaults(func=lambda args: asyncio.run(generate_data_cmd(args)))


__all__ = ["register_generate_parser", "register_train_sft_parser"]
