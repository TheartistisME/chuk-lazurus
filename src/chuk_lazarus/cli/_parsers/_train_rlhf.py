"""DPO + GRPO parser registrars (split from legacy ``_train.py``).

EWS-0 carved these out verbatim.  Ownership transfers to EWS-10 at merge.
"""

import asyncio

from ..commands._base import add_backend_flags
from ..commands.train import train_dpo_cmd, train_grpo_cmd


def register_train_dpo_parser(train_subparsers) -> None:
    """Register the ``train dpo`` subcommand."""

    dpo_parser = train_subparsers.add_parser("dpo", help="Direct Preference Optimization")
    dpo_parser.add_argument("--model", required=True, help="Policy model name or path")
    dpo_parser.add_argument("--ref-model", help="Reference model (default: same as --model)")
    dpo_parser.add_argument("--data", required=True, help="Preference data path (JSONL)")
    dpo_parser.add_argument("--eval-data", help="Evaluation data path (JSONL)")
    dpo_parser.add_argument("--output", default="./checkpoints/dpo", help="Output directory")
    dpo_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    dpo_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    dpo_parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    dpo_parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    dpo_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    dpo_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    dpo_parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    add_backend_flags(dpo_parser)
    dpo_parser.set_defaults(func=lambda args: asyncio.run(train_dpo_cmd(args)))


def register_train_grpo_parser(train_subparsers) -> None:
    """Register the ``train grpo`` subcommand."""

    grpo_parser = train_subparsers.add_parser(
        "grpo", help="Group Relative Policy Optimization (RL with verifiable rewards)"
    )
    grpo_parser.add_argument("--model", required=True, help="Policy model name or path")
    grpo_parser.add_argument("--ref-model", help="Reference model (default: same as --model)")
    grpo_parser.add_argument("--output", default="./checkpoints/grpo", help="Output directory")
    grpo_parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    grpo_parser.add_argument(
        "--prompts-per-iteration", type=int, default=16, help="Prompts per iteration"
    )
    grpo_parser.add_argument("--group-size", type=int, default=4, help="Responses per prompt")
    grpo_parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    grpo_parser.add_argument("--kl-coef", type=float, default=0.1, help="KL penalty coefficient")
    grpo_parser.add_argument(
        "--max-response-length", type=int, default=256, help="Max response tokens"
    )
    grpo_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    grpo_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    grpo_parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    grpo_parser.add_argument(
        "--lora-targets",
        default="q_proj,v_proj",
        help="Comma-separated LoRA target modules (default: q_proj,v_proj)",
    )
    grpo_parser.add_argument(
        "--freeze-layers",
        help="Layers to freeze (e.g., '0-12' or '0,1,2,3')",
    )
    grpo_parser.add_argument(
        "--reward-script",
        required=True,
        help="Python script defining reward_fn(prompt, response) -> float and get_prompts() -> list[str]",
    )
    grpo_parser.add_argument(
        "--config",
        help="YAML config file (overrides other arguments)",
    )
    add_backend_flags(grpo_parser)
    grpo_parser.set_defaults(func=lambda args: asyncio.run(train_grpo_cmd(args)))


__all__ = ["register_train_dpo_parser", "register_train_grpo_parser"]
