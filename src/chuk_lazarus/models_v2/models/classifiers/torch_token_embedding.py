"""
Torch-native token embedding with optional tied output projection.

Wraps :class:`torch.nn.Embedding` and exposes :meth:`as_output_projection` which
returns a :class:`torch.nn.Linear` that either shares the embedding weight
(when ``tie_weights=True``) or holds an independent projection weight.

Mirrors the constructor signature of the MLX ``TokenEmbedding`` primitive so
downstream tools can swap backends by selecting the module at import time.
This is torch-native (NOT an MLX port). No device placement is performed
inside the module. Default dtype is ``torch.float32``.
"""

from __future__ import annotations

import torch
from torch import nn


class TorchTokenEmbedding(nn.Module):
    """
    Token embedding with optional tied output projection.

    Args:
        vocab_size: Size of the token vocabulary.
        hidden_size: Embedding (hidden) dimension.
        tie_weights: If True, :meth:`as_output_projection` returns a
            :class:`torch.nn.Linear` that shares the embedding weight tensor.

    Shape:
        - Input: ``(batch, seq_len)`` of integer token ids.
        - Output: ``(batch, seq_len, hidden_size)`` of embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        tie_weights: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.tie_weights = bool(tie_weights)

        self.embedding: nn.Embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            dtype=torch.float32,
        )
        # Lazily-constructed output projection; we build it eagerly so that
        # repeated ``as_output_projection`` calls return the same module and
        # weight sharing is a single deterministic operation.
        self._output_projection: nn.Linear = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size,
            bias=False,
            dtype=torch.float32,
        )
        if self.tie_weights:
            # Share the underlying weight tensor (same storage, same Parameter).
            self._output_projection.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up token embeddings.

        Args:
            input_ids: Integer token ids of shape ``(batch, seq_len)``.

        Returns:
            Embeddings of shape ``(batch, seq_len, hidden_size)``.
        """
        return self.embedding(input_ids)

    def as_output_projection(self) -> nn.Linear:
        """
        Return an output projection ``Linear(hidden_size -> vocab_size)``.

        When ``tie_weights=True`` the returned module's ``.weight`` is the same
        :class:`torch.nn.Parameter` as :attr:`embedding.weight` (shared
        storage). When ``tie_weights=False`` the projection owns an
        independent weight tensor.
        """
        return self._output_projection
