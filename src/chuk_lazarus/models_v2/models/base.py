"""
Base model abstractions.

Defines the common interface for complete models.

Top-level ``import mlx.*`` is intentionally absent: this module is listed
in ``tests/ci/test_no_top_level_mlx.py::BACKEND_IN_SCOPE``. The real
``mlx.nn.Module``-backed ``Model`` and ``ModelOutput`` dataclass are
built lazily inside ``_build()`` and surfaced via PEP 562
``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

__all__ = ["Model", "ModelOutput"]

_built: dict[str, type] = {}


def _build() -> dict[str, type]:
    if _built:
        return _built

    from abc import ABC, abstractmethod
    from dataclasses import dataclass, field
    from typing import Any

    import mlx.core as mx
    import mlx.nn as nn

    from ..core.config import ModelConfig

    @dataclass
    class ModelOutput:
        """
        Output from a model forward pass.
        """

        loss: mx.array | None = None
        logits: mx.array | None = None
        hidden_states: tuple[mx.array, ...] | None = None
        cache: list[Any] | None = None
        aux_loss: mx.array | None = None
        aux_outputs: dict[str, Any] = field(default_factory=dict)

    class Model(nn.Module, ABC):
        """
        Abstract base class for complete models.
        """

        @property
        @abstractmethod
        def config(self) -> ModelConfig:
            """Return the model configuration."""

        @property
        @abstractmethod
        def backbone(self) -> nn.Module:
            """Return the backbone module."""

        @abstractmethod
        def __call__(
            self,
            input_ids: mx.array,
            attention_mask: mx.array | None = None,
            labels: mx.array | None = None,
            cache: list[Any] | None = None,
            output_hidden_states: bool = False,
        ) -> "ModelOutput":
            """Forward pass."""

        def init_cache(self, batch_size: int, max_seq_len: int) -> list[Any]:
            return self.backbone.init_cache(batch_size, max_seq_len)

        def get_input_embeddings(self) -> nn.Module:
            return self.backbone.get_input_embeddings()

        def set_input_embeddings(self, embeddings: nn.Module) -> None:
            self.backbone.set_input_embeddings(embeddings)

        def num_parameters(self, only_trainable: bool = True) -> int:
            import mlx.utils

            total = 0
            for name, param in mlx.utils.tree_flatten(self.parameters()):
                if isinstance(param, mx.array):
                    total += param.size
            return total

        def save_weights(self, path: str) -> None:
            import numpy as np

            weights = {}
            for name, param in self.parameters().items():
                if isinstance(param, mx.array):
                    weights[name] = np.array(param)

            np.savez(path, **weights)

        def load_weights(self, path: str) -> None:
            import numpy as np

            data = np.load(path)
            weights = {k: mx.array(v) for k, v in data.items()}
            self.update(weights)

        @classmethod
        @abstractmethod
        def from_config(cls, config: ModelConfig) -> "Model":
            """Create model from configuration."""

        @classmethod
        async def from_pretrained_async(
            cls,
            model_path: str,
            config: ModelConfig | None = None,
        ) -> "Model":
            import json
            from pathlib import Path

            import aiofiles

            path = Path(model_path)

            if config is None:
                config_path = path / "config.json"
                async with aiofiles.open(config_path) as f:
                    config_data = json.loads(await f.read())
                config = ModelConfig(**config_data)

            model = cls.from_config(config)

            weights_path = path / "model.safetensors"
            if weights_path.exists():
                try:
                    import safetensors.numpy as st

                    weights = st.load_file(str(weights_path))
                    weights = {k: mx.array(v) for k, v in weights.items()}
                    model.update(weights)
                except ImportError:
                    pass

            return model

    _built["Model"] = Model
    _built["ModelOutput"] = ModelOutput
    return _built


def _make_facade(name: str) -> type:
    """Façade class whose instantiation triggers the real class build."""

    class _Meta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, _build()[name])

        def __subclasscheck__(cls, subclass):
            return issubclass(subclass, _build()[name])

        def __getattr__(cls, attr):
            return getattr(_build()[name], attr)

    class _Facade(metaclass=_Meta):
        __qualname__ = name

        def __new__(cls, *args, **kwargs):
            return _build()[name](*args, **kwargs)

    _Facade.__name__ = name
    return _Facade


_facades: dict[str, type] = {}


def __getattr__(name: str):
    if name in ("Model", "ModelOutput"):
        if name not in _facades:
            _facades[name] = _make_facade(name)
        return _facades[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
