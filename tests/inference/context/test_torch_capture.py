"""Unit tests for torch_capture layer resolution — wrapper-config aware."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.context.knowledge.torch_capture import (
    _resolve_transformer_layers,
)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)


class _ListHolder(torch.nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block() for _ in range(n)])


class _VLMInner(torch.nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.language_model = _ListHolder(n)


class _VLMWrapped(torch.nn.Module):
    """Mimics `model.model.language_model.layers` — Gemma 4's path."""

    def __init__(self, n: int):
        super().__init__()
        self.model = _VLMInner(n)


class _PlainWrapped(torch.nn.Module):
    """Mimics `model.model.layers` — plain Gemma 3 / Llama path."""

    def __init__(self, n: int):
        super().__init__()
        self.model = _ListHolder(n)


def test_resolve_transformer_layers_finds_vlm_language_model_layers():
    model = _VLMWrapped(n=35)
    layers = _resolve_transformer_layers(model)
    assert len(layers) == 35
    assert layers[0] is model.model.language_model.layers[0]


def test_resolve_transformer_layers_finds_plain_model_layers():
    model = _PlainWrapped(n=26)
    layers = _resolve_transformer_layers(model)
    assert len(layers) == 26
    assert layers[0] is model.model.layers[0]


def test_resolve_transformer_layers_raises_when_no_path_matches():
    class _Bare(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match="Cannot resolve transformer layers"):
        _resolve_transformer_layers(_Bare())
