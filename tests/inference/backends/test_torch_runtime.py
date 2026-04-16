"""Real CUDA smoke tests for the PyTorch inference runtime."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from chuk_lazarus.inference.backends import TorchInferenceRuntime
from chuk_lazarus.inference.backends.types import LazarusBackend
from chuk_lazarus.inference.generation import GenerationConfig


class DummyBlock(torch.nn.Module):
    """Small transformer-like block for hook tests."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(4))

    def forward(self, hidden_states):
        return self.proj(hidden_states) + 1


class DummyInnerModel(torch.nn.Module):
    """Container exposing `.layers` like Hugging Face decoder models."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([DummyBlock(), DummyBlock()])


class DummyCausalLM(torch.nn.Module):
    """Tiny causal LM that exercises hooks during `generate()`."""

    def __init__(self, head_dim: int = 512):
        super().__init__()
        self.model = DummyInnerModel()
        self.config = SimpleNamespace(head_dim=head_dim)
        self.last_generate_kwargs: dict[str, object] | None = None

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
        del attention_mask, use_cache, kwargs
        hidden_states = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 4)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return (hidden_states,)

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.last_generate_kwargs = dict(kwargs)
        del attention_mask
        self.forward(input_ids=input_ids)
        suffix = torch.tensor([[7, 8]], device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


class DummyTokenizer:
    """Tokenizer stub with the Hugging Face methods used by the runtime."""

    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, prompt, return_tensors="pt"):
        del prompt, return_tensors
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test requires a CUDA GPU")
class TestTorchInferenceRuntime:
    """Smoke tests that exercise the CUDA runtime on the local GPU."""

    @pytest.fixture
    def runtime(self):
        model = DummyCausalLM(head_dim=512).to("cuda").eval()
        tokenizer = DummyTokenizer()
        return TorchInferenceRuntime(model, tokenizer, device="cuda")

    def test_generation_context_falls_back_without_flash_for_unsupported_head_dim(self, runtime, monkeypatch):
        recorded = {}

        class DummyContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_sdpa_kernel(backends):
            recorded["backends"] = backends
            return DummyContext()

        monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", fake_sdpa_kernel)

        with runtime._generation_context(total_window_tokens=128):
            pass

        backend_names = [backend.name for backend in recorded["backends"]]
        assert backend_names == ["EFFICIENT_ATTENTION", "MATH"]

    def test_generation_context_prefers_flash_when_safe(self, monkeypatch):
        model = DummyCausalLM(head_dim=128).to("cuda").eval()
        runtime = TorchInferenceRuntime(model, DummyTokenizer(), device="cuda")
        recorded = {}

        class DummyContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_sdpa_kernel(backends):
            recorded["backends"] = backends
            return DummyContext()

        monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", fake_sdpa_kernel)

        with runtime._generation_context(total_window_tokens=128):
            pass

        backend_names = [backend.name for backend in recorded["backends"]]
        assert backend_names[0] == "FLASH_ATTENTION"
        assert backend_names[1:] == ["EFFICIENT_ATTENTION", "MATH"]

    def test_generate(self, runtime):
        result = runtime.generate("test prompt", GenerationConfig(max_new_tokens=2, temperature=0.0))
        assert result.text == "decoded:7,8"
        assert result.stats.output_tokens == 2
        assert runtime.backend == LazarusBackend.CUDA

    def test_generate_omits_top_k_when_sampling_is_disabled(self, runtime):
        result = runtime.generate(
            "test prompt",
            GenerationConfig(max_new_tokens=2, temperature=0.0, top_k=50),
        )
        assert result.text == "decoded:7,8"
        assert runtime._model.last_generate_kwargs is not None
        assert "top_k" not in runtime._model.last_generate_kwargs
        assert runtime._model.last_generate_kwargs["do_sample"] is False
        assert "temperature" not in runtime._model.last_generate_kwargs

    def test_extract_residual_state(self, runtime):
        state = runtime.extract_residual_state("capture prompt", layer_index=1)
        assert state.backend == LazarusBackend.CUDA
        assert state.layer_index == 1
        assert tuple(state.tensor.shape) == (1, 4)
        assert state.hidden_size == 4
        assert state.device == "cuda"

    def test_generate_with_residual(self, runtime):
        residual_state = runtime.extract_residual_state("seed prompt", layer_index=0)
        result = runtime.generate_with_residual(
            "inject prompt",
            residual_state,
            GenerationConfig(max_new_tokens=2, temperature=0.0),
        )
        assert result.text == "decoded:7,8"

    def test_clear_cache(self, runtime, monkeypatch):
        calls = {"count": 0}

        def fake_empty_cache():
            calls["count"] += 1

        monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)
        runtime.clear_cache()
        assert calls["count"] == 1
