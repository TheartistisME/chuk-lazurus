"""EWS-9: Optimizer adapter wires torch.optim.AdamW on torch, mlx.optimizers.AdamW on MLX."""

from __future__ import annotations

import pytest


def test_create_adamw_torch():
    torch = pytest.importorskip("torch")
    from chuk_lazarus.utils.optimizer_adapter import create_adamw

    model = torch.nn.Linear(4, 2)
    opt = create_adamw(model, learning_rate=1e-3, weight_decay=0.1, framework="torch")
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.1)


def test_create_adamw_mlx():
    mlx_optim = pytest.importorskip("mlx.optimizers")
    from chuk_lazarus.utils.optimizer_adapter import create_adamw

    opt = create_adamw(model_or_params=None, learning_rate=1e-3, framework="mlx")
    assert isinstance(opt, mlx_optim.AdamW)


def test_optimizer_adapter_torch_roundtrip():
    torch = pytest.importorskip("torch")
    from chuk_lazarus.utils.optimizer_adapter import OptimizerAdapter

    model = torch.nn.Linear(3, 3)
    adapter = OptimizerAdapter(framework="torch")
    opt = adapter.create_optimizer(model.parameters(), "AdamW", lr=5e-4)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)
    adapter.zero_grad()  # no crash
