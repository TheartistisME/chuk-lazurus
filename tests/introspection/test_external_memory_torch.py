"""Torch-specific tests for external_memory."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.backends import TorchInferenceRuntime
from chuk_lazarus.introspection.external_memory import ExternalMemory, MemoryConfig


class TorchIdentityLayer(torch.nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class TorchMockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([TorchIdentityLayer()]))
        self.embed = torch.nn.Embedding(16, 4)
        self.lm_head = torch.nn.Linear(4, 16, bias=False)

        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[4] = torch.tensor([0.0, 0.0, 4.0, 0.0])
            self.embed.weight[6] = torch.tensor([0.0, 0.0, 0.0, 4.0])
            self.lm_head.weight.zero_()
            self.lm_head.weight[4] = torch.tensor([0.0, 0.0, 1.0, 0.0])
            self.lm_head.weight[6] = torch.tensor([0.0, 0.0, 0.0, 1.0])

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden_states = self.embed(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(logits=logits)


class TorchMockTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, prompt, return_tensors="pt"):
        token_id = 4 if prompt == "alpha" else 6
        return {
            "input_ids": torch.tensor([[token_id]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def encode(self, prompt, return_tensors=None):
        token_id = 4 if prompt == "alpha" else 6
        tokens = [token_id]
        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long)
        return tokens

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list):
            return f"tok-{ids[0]}"
        return f"tok-{ids}"


def test_query_force_injection_uses_torch_runtime():
    model = TorchMockModel().eval()
    tokenizer = TorchMockTokenizer()
    config = SimpleNamespace(hidden_size=4, num_hidden_layers=1, embedding_scale=None)
    runtime = TorchInferenceRuntime(model, tokenizer, device="cpu")
    memory_config = MemoryConfig(
        query_layer=0,
        value_layer=0,
        inject_layer=0,
        similarity_threshold=0.99,
    )

    memory = ExternalMemory(
        model,
        tokenizer,
        config,
        memory_config,
        runtime=runtime,
    )
    memory.add_fact("alpha", "fact-alpha")

    result = memory.query("gamma", use_injection=True, force_injection=True)

    assert result.matched_entry is not None
    assert result.matched_entry.query == "alpha"
    assert result.used_injection is True
    assert result.baseline_answer == "tok-6"
    assert result.injected_answer == "tok-4"
    assert result.injected_confidence is not None


class TestFromPretrainedErrors:
    """Test error handling in from_pretrained."""

    def test_from_pretrained_unsupported_model(self, monkeypatch):
        import chuk_lazarus.inference as inference_module
        import chuk_lazarus.models_v2.core.backend as backend_module

        fake_backend = SimpleNamespace(name="mlx", device="mps")

        def mock_from_pretrained(*args, **kwargs):
            raise ValueError("Unable to detect model family. Model may not be supported yet.")

        monkeypatch.setattr(backend_module, "get_backend", lambda *args, **kwargs: fake_backend)
        monkeypatch.setattr(inference_module.UnifiedPipeline, "from_pretrained", mock_from_pretrained)

        with pytest.raises(ValueError, match="Unsupported model"):
            ExternalMemory.from_pretrained("fake/unsupported-model")

    def test_from_pretrained_auto_config(self, monkeypatch):
        import chuk_lazarus.inference as inference_module
        import chuk_lazarus.models_v2.core.backend as backend_module

        fake_backend = SimpleNamespace(name="mlx", device="mps")
        fake_runtime = SimpleNamespace(backend="mlx")
        fake_pipeline = SimpleNamespace(
            model=object(),
            tokenizer=object(),
            config=SimpleNamespace(hidden_size=64, num_hidden_layers=24),
            runtime=fake_runtime,
        )

        monkeypatch.setattr(backend_module, "get_backend", lambda *args, **kwargs: fake_backend)
        monkeypatch.setattr(
            inference_module.UnifiedPipeline,
            "from_pretrained",
            lambda *args, **kwargs: fake_pipeline,
        )

        memory = ExternalMemory.from_pretrained("fake/test-model")

        assert memory._memory_config.query_layer == 22
        assert memory._memory_config.inject_layer == 21
        assert memory._memory_config.value_layer == 22
        assert memory._runtime is fake_runtime

    def test_from_pretrained_explicit_config(self, monkeypatch):
        import chuk_lazarus.inference as inference_module
        import chuk_lazarus.models_v2.core.backend as backend_module

        fake_backend = SimpleNamespace(name="mlx", device="mps")
        fake_pipeline = SimpleNamespace(
            model=object(),
            tokenizer=object(),
            config=SimpleNamespace(hidden_size=64, num_hidden_layers=24),
            runtime=SimpleNamespace(backend="mlx"),
        )

        monkeypatch.setattr(backend_module, "get_backend", lambda *args, **kwargs: fake_backend)
        monkeypatch.setattr(
            inference_module.UnifiedPipeline,
            "from_pretrained",
            lambda *args, **kwargs: fake_pipeline,
        )

        custom_config = MemoryConfig(query_layer=10, inject_layer=9, value_layer=10)
        memory = ExternalMemory.from_pretrained("fake/test-model", memory_config=custom_config)

        assert memory._memory_config.query_layer == 10
        assert memory._memory_config.inject_layer == 9
        assert memory._memory_config.value_layer == 10

    def test_from_pretrained_threads_backend_and_device(self, monkeypatch):
        import chuk_lazarus.inference as inference_module
        import chuk_lazarus.models_v2.core.backend as backend_module

        fake_backend = SimpleNamespace(name="torch", device="cuda:1")
        fake_pipeline = SimpleNamespace(
            model=object(),
            tokenizer=object(),
            config=SimpleNamespace(hidden_size=64, num_hidden_layers=24),
            runtime=SimpleNamespace(backend="cuda"),
        )
        captured = {}

        def mock_from_pretrained(model_id, pipeline_config=None, verbose=False):
            captured["model_id"] = model_id
            captured["pipeline_config"] = pipeline_config
            captured["verbose"] = verbose
            return fake_pipeline

        monkeypatch.setattr(backend_module, "get_backend", lambda *args, **kwargs: fake_backend)
        monkeypatch.setattr(inference_module.UnifiedPipeline, "from_pretrained", mock_from_pretrained)

        memory = ExternalMemory.from_pretrained(
            "fake/test-model",
            backend="torch",
            device="cuda:1",
        )

        assert captured["model_id"] == "fake/test-model"
        assert captured["pipeline_config"].backend_name == "torch"
        assert captured["pipeline_config"].device == "cuda:1"
        assert memory._runtime.backend == "cuda"
