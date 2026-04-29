from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
activation_routes = pytest.importorskip(
    "chuk_lazarus.inference.context.knowledge.activation_routes"
)
activation_topk = activation_routes.activation_topk
mean_pool_hidden = activation_routes.mean_pool_hidden


def test_mean_pool_hidden_masks_padding_and_normalizes():
    hidden = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 4.0], [100.0, 100.0]],
            [[1.0, 1.0], [1.0, -1.0], [0.0, 9.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)

    pooled = mean_pool_hidden(hidden, mask, normalize=True, out_dtype=torch.float32)

    assert pooled.shape == (2, 2)
    assert pooled.dtype == torch.float32
    assert torch.allclose(pooled[0], torch.tensor([0.6, 0.8]), atol=1e-6)
    assert torch.allclose(pooled[1], torch.tensor([0.70710677, 0.70710677]), atol=1e-6)


def test_mean_pool_hidden_accepts_single_window_and_float16_output():
    hidden = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    mask = torch.tensor([1, 1], dtype=torch.long)

    pooled = mean_pool_hidden(hidden, mask)

    assert pooled.shape == (1, 2)
    assert pooled.dtype == torch.float16
    assert torch.allclose(pooled.float(), torch.tensor([[0.7071, 0.7071]]), atol=1e-3)


def test_activation_topk_returns_sorted_indices_and_values():
    query = torch.tensor([1.0, 0.0], dtype=torch.float32)
    windows = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )

    indices, values = activation_topk(query, windows, top_k=2)

    assert indices.tolist() == [1, 2]
    assert torch.allclose(values, torch.tensor([1.0, 0.6]), atol=1e-6)


def test_activation_topk_validates_shapes():
    with pytest.raises(ValueError, match="hidden dimension"):
        activation_topk(torch.ones(3), torch.ones(2, 4), top_k=1)
