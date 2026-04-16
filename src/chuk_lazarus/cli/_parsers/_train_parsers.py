"""Orchestrator that re-composes the legacy ``register_train_parsers``.

EWS-0 keeps the public entrypoint name stable by composing the split
registrars from ``_train_sft`` and ``_train_rlhf``.  The legacy ``_train.py``
shims this module with ``from ._train_parsers import *``.
"""

from ._train_rlhf import register_train_dpo_parser, register_train_grpo_parser
from ._train_sft import register_generate_parser, register_train_sft_parser


def register_train_parsers(subparsers) -> None:
    """Register the ``train`` subcommand tree and the ``generate`` subcommand."""

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_subparsers = train_parser.add_subparsers(dest="train_type", help="Training type")

    register_train_sft_parser(train_subparsers)
    register_train_dpo_parser(train_subparsers)
    register_train_grpo_parser(train_subparsers)

    register_generate_parser(subparsers)


__all__ = [
    "register_generate_parser",
    "register_train_dpo_parser",
    "register_train_grpo_parser",
    "register_train_parsers",
    "register_train_sft_parser",
]
