"""
Backbone stacks for sequence models.

Backbones combine embeddings + blocks into a complete encoder/decoder.
They produce hidden states that can be fed to different heads.

Provides:
- TransformerBackbone: Stack of transformer blocks
- MambaBackbone: Stack of Mamba blocks
- RecurrentBackbone: Stack of RNN blocks
- HybridBackbone: Mixed block types

Top-level ``from .base import ...`` etc. are intentionally absent: each
backbone submodule uses PEP 562 module ``__getattr__`` to defer ``mlx``
imports. This package ``__init__`` mirrors that pattern so that
``import chuk_lazarus.models_v2.backbones`` does not trigger ``mlx``
under ``CHUK_BACKEND=torch``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .base import Backbone, BackboneOutput  # noqa: F401
    from .hybrid import HybridBackbone, create_hybrid_backbone  # noqa: F401
    from .mamba import MambaBackbone, create_mamba_backbone  # noqa: F401
    from .recurrent import RecurrentBackbone, create_recurrent_backbone  # noqa: F401
    from .transformer import TransformerBackbone, create_transformer_backbone  # noqa: F401

__all__ = [
    "Backbone",
    "BackboneOutput",
    "TransformerBackbone",
    "create_transformer_backbone",
    "MambaBackbone",
    "create_mamba_backbone",
    "RecurrentBackbone",
    "create_recurrent_backbone",
    "HybridBackbone",
    "create_hybrid_backbone",
]

_LAZY: dict[str, tuple[str, str]] = {
    "Backbone": (".base", "Backbone"),
    "BackboneOutput": (".base", "BackboneOutput"),
    "TransformerBackbone": (".transformer", "TransformerBackbone"),
    "create_transformer_backbone": (".transformer", "create_transformer_backbone"),
    "MambaBackbone": (".mamba", "MambaBackbone"),
    "create_mamba_backbone": (".mamba", "create_mamba_backbone"),
    "RecurrentBackbone": (".recurrent", "RecurrentBackbone"),
    "create_recurrent_backbone": (".recurrent", "create_recurrent_backbone"),
    "HybridBackbone": (".hybrid", "HybridBackbone"),
    "create_hybrid_backbone": (".hybrid", "create_hybrid_backbone"),
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
