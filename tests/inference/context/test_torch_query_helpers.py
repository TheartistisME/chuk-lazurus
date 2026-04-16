"""Unit tests for torch_query helpers that must unwrap VLM wrapper configs.

Gemma 4's top-level HF config is a thin wrapper: `hidden_size` /
`num_hidden_layers` live under ``config.text_config``. These tests ensure
``_infer_model_hidden_size``, ``_count_model_layers`` and
``_residual_is_compatible`` read the text-backbone dimensions correctly
(regression for the gemma 4 residual-injection path).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.context.knowledge.torch_query import (
    _count_model_layers,
    _infer_model_hidden_size,
    _residual_is_compatible,
)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):  # pragma: no cover - unused
        return self.proj(x)


class _VLMLanguageModel(torch.nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block() for _ in range(n)])


class _VLMInner(torch.nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.language_model = _VLMLanguageModel(n)


class _VLMCausalLM(torch.nn.Module):
    """Stand-in for Gemma4ForConditionalGeneration — wrapper config has no
    top-level hidden_size; dims live under ``config.text_config``."""

    def __init__(self, num_layers: int = 35, hidden_size: int = 1536):
        super().__init__()
        self.model = _VLMInner(num_layers)
        self.config = SimpleNamespace(
            model_type="gemma4",
            hidden_size=None,
            num_hidden_layers=None,
            text_config=SimpleNamespace(
                model_type="gemma4_text",
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
            ),
        )


class _PlainInner(torch.nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block() for _ in range(n)])


class _PlainCausalLM(torch.nn.Module):
    """Stand-in for a plain causal LM (Gemma 3-style, no wrapper)."""

    def __init__(self, num_layers: int = 26, hidden_size: int = 1152):
        super().__init__()
        self.model = _PlainInner(num_layers)
        self.config = SimpleNamespace(
            model_type="gemma3_text",
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
        )


class _FakeRuntime:
    """Minimal stand-in for TorchInferenceRuntime — only the parts the
    torch_query helpers touch (``_model`` and ``_resolve_layers``)."""

    def __init__(self, model):
        self._model = model

    def _resolve_layers(self):
        from chuk_lazarus.inference.backends.torch_runtime import TorchInferenceRuntime

        return TorchInferenceRuntime._resolve_layers(self)  # type: ignore[arg-type]


class TestInferHiddenSizeForWrapperConfig:
    def test_gemma4_wrapper_reads_text_config_hidden_size(self):
        """Gemma 4 hides hidden_size under config.text_config."""
        model = _VLMCausalLM(num_layers=35, hidden_size=1536)
        runtime = _FakeRuntime(model)
        assert _infer_model_hidden_size(runtime) == 1536

    def test_plain_model_reads_top_level_hidden_size(self):
        """Plain Gemma 3 config keeps hidden_size on the top-level config."""
        model = _PlainCausalLM(num_layers=26, hidden_size=1152)
        runtime = _FakeRuntime(model)
        assert _infer_model_hidden_size(runtime) == 1152

    def test_falls_back_to_embedding_dim_when_config_is_silent(self):
        """If config has no hidden_size at any level, use the embedding matrix."""

        class _ConfigSilent(SimpleNamespace):
            pass

        class _Wrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _PlainInner(3)
                self.embed = torch.nn.Embedding(100, 768)
                self.config = _ConfigSilent()

            def get_input_embeddings(self):
                return self.embed

        runtime = _FakeRuntime(_Wrapped())
        assert _infer_model_hidden_size(runtime) == 768


class TestCountModelLayersForWrapperConfig:
    def test_gemma4_counts_language_model_layers(self):
        """35-layer Gemma 4 counts via ``model.model.language_model.layers``."""
        model = _VLMCausalLM(num_layers=35)
        runtime = _FakeRuntime(model)
        assert _count_model_layers(runtime) == 35

    def test_plain_model_counts_model_layers(self):
        model = _PlainCausalLM(num_layers=26)
        runtime = _FakeRuntime(model)
        assert _count_model_layers(runtime) == 26


class TestResidualCompatibilityForGemma4:
    def test_boundary_matching_gemma4_hidden_size_is_compatible(self):
        model = _VLMCausalLM(num_layers=35, hidden_size=1536)
        runtime = _FakeRuntime(model)
        boundary = torch.zeros((1, 1536))
        ok, reason = _residual_is_compatible(runtime, boundary, layer_index=29)
        assert ok, reason

    def test_out_of_range_layer_index_is_rejected(self):
        model = _VLMCausalLM(num_layers=35, hidden_size=1536)
        runtime = _FakeRuntime(model)
        boundary = torch.zeros((1, 1536))
        ok, reason = _residual_is_compatible(runtime, boundary, layer_index=40)
        assert not ok
        assert "outside the model's 35 layers" in reason

    def test_hidden_size_mismatch_is_rejected(self):
        """A Gemma 3 4B boundary (2560) should NOT load onto Gemma 4 (1536)."""
        model = _VLMCausalLM(num_layers=35, hidden_size=1536)
        runtime = _FakeRuntime(model)
        boundary = torch.zeros((1, 2560))
        ok, reason = _residual_is_compatible(runtime, boundary, layer_index=29)
        assert not ok
        assert "2560" in reason and "1536" in reason
