"""Cross-framework optimizer adapter (EWS-9).

Wires ``mlx.optimizers.AdamW`` on MLX and ``torch.optim.AdamW`` on torch.
All framework imports are function-scope so this module is lazy-import-clean
under ``CHUK_BACKEND=torch``.
"""

from __future__ import annotations

import os
from typing import Any


def _resolve_framework(framework: str | None) -> str:
    if framework:
        return framework
    return os.environ.get("CHUK_BACKEND", "mlx").lower() or "mlx"


def create_adamw(
    model_or_params: Any,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    framework: str | None = None,
) -> Any:
    """Create an AdamW optimizer appropriate for the active backend."""
    fw = _resolve_framework(framework)
    if fw == "torch":
        import torch.optim as torch_optim

        params = (
            list(model_or_params.parameters())
            if hasattr(model_or_params, "parameters")
            else list(model_or_params)
        )
        return torch_optim.AdamW(
            params,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    import mlx.optimizers as mlx_optim

    return mlx_optim.AdamW(
        learning_rate=learning_rate,
        betas=list(betas),
        eps=eps,
        weight_decay=weight_decay,
    )


class OptimizerAdapter:
    """Thin wrapper around torch/MLX optimizers with a unified step/zero_grad API.

    MLX optimizers mutate state through ``optimizer.update(model, grads)`` so the
    ``step``/``zero_grad`` methods only do work when the torch backend is active.
    """

    def __init__(self, framework: str = "mlx"):
        self.framework = _resolve_framework(framework)
        if self.framework == "torch":
            try:
                import torch.optim  # noqa: F401
            except ImportError as e:  # pragma: no cover - torch always present in CI
                raise ImportError(
                    "torch is required for framework='torch'. Install with: pip install torch"
                ) from e
        self.optimizer = None

    def create_optimizer(self, model_parameters, optimizer_name: str = "AdamW", **kwargs):
        if optimizer_name not in {"Adam", "AdamW", "SGD"}:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        if self.framework == "mlx":
            import mlx.optimizers as mlx_optim

            cls = {"Adam": mlx_optim.Adam, "AdamW": mlx_optim.AdamW, "SGD": mlx_optim.SGD}[
                optimizer_name
            ]
            self.optimizer = cls(**kwargs)
        else:
            import torch.optim as torch_optim

            cls = {"Adam": torch_optim.Adam, "AdamW": torch_optim.AdamW, "SGD": torch_optim.SGD}[
                optimizer_name
            ]
            self.optimizer = cls(model_parameters, **kwargs)
        return self.optimizer

    def step(self):
        if self.optimizer is not None and self.framework == "torch":
            self.optimizer.step()

    def zero_grad(self):
        if self.optimizer is not None and self.framework == "torch":
            self.optimizer.zero_grad()
