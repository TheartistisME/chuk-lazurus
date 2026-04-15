"""
Composable blocks for sequence models.

Blocks combine components (attention/SSM/RNN + FFN + normalization)
into reusable units that form the layers of a model.

Top-level ``import mlx.*`` is intentionally absent: re-exports are resolved
lazily via PEP 562 module ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .base import Block, BlockOutput  # noqa: F401
    from .hybrid import HybridBlock, create_hybrid_block  # noqa: F401
    from .mamba import MambaBlockWrapper, create_mamba_block_wrapper  # noqa: F401
    from .recurrent import (  # noqa: F401
        RecurrentBlockWrapper,
        RecurrentWithFFN,
        create_recurrent_block,
    )
    from .transformer import TransformerBlock, create_transformer_block  # noqa: F401

__all__ = [
    "Block",
    "BlockOutput",
    "TransformerBlock",
    "create_transformer_block",
    "MambaBlockWrapper",
    "create_mamba_block_wrapper",
    "RecurrentBlockWrapper",
    "RecurrentWithFFN",
    "create_recurrent_block",
    "HybridBlock",
    "create_hybrid_block",
]

_LAZY: dict[str, tuple[str, str]] = {
    "Block": (".base", "Block"),
    "BlockOutput": (".base", "BlockOutput"),
    "TransformerBlock": (".transformer", "TransformerBlock"),
    "create_transformer_block": (".transformer", "create_transformer_block"),
    "MambaBlockWrapper": (".mamba", "MambaBlockWrapper"),
    "create_mamba_block_wrapper": (".mamba", "create_mamba_block_wrapper"),
    "RecurrentBlockWrapper": (".recurrent", "RecurrentBlockWrapper"),
    "RecurrentWithFFN": (".recurrent", "RecurrentWithFFN"),
    "create_recurrent_block": (".recurrent", "create_recurrent_block"),
    "HybridBlock": (".hybrid", "HybridBlock"),
    "create_hybrid_block": (".hybrid", "create_hybrid_block"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name, __name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
