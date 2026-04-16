"""Torch runtime tests for introspection generation service."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.backends import TorchInferenceRuntime
from chuk_lazarus.inference.generation import GenerationConfig as RuntimeGenerationConfig
from chuk_lazarus.introspection.generation import GenerationConfig, GenerationService


TORCH_DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


class TinyTorchGenerationTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    chat_template = None

    def __init__(self) -> None:
        self._prompt_ids = {
            "2+2=": 1,
            "2+2= ": 2,
            "alpha": 3,
        }
        self._token_text = {
            0: "<pad>",
            1: "2+2=",
            2: "2+2= ",
            3: "alpha",
            4: "4",
            5: "5",
            6: "answer",
        }

    def __call__(self, prompt, return_tensors="pt"):
        token_id = self._prompt_ids[prompt]
        input_ids = torch.tensor([[token_id]], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def encode(self, text, return_tensors=None):
        direct_outputs = {"4": [4], "5": [5], "answer": [6]}
        if text in self._prompt_ids:
            tokens = [self._prompt_ids[text]]
        elif text in direct_outputs:
            tokens = direct_outputs[text]
        else:
            prompt = None
            for candidate in sorted(self._prompt_ids.keys(), key=len, reverse=True):
                if text.startswith(candidate):
                    prompt = candidate
                    break

            if prompt is None:
                tokens = []
            else:
                suffix = text[len(prompt) :]
                tokens = [self._prompt_ids[prompt]]
                if suffix == "4":
                    tokens.append(4)
                elif suffix == "5":
                    tokens.append(5)
                elif suffix == "answer":
                    tokens.append(6)

        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long)
        return tokens

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list):
            return "".join(
                self._token_text[token_id]
                for token_id in ids
                if not skip_special_tokens or token_id not in {self.pad_token_id, self.eos_token_id}
            )
        return self._token_text[ids]


class TinyTorchGenerationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(7, 3)
        self.lm_head = torch.nn.Linear(3, 7, bias=False)

        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[1] = torch.tensor([1.0, 0.0, 0.0])
            self.embed.weight[2] = torch.tensor([0.0, 1.0, 0.0])
            self.embed.weight[3] = torch.tensor([0.0, 0.0, 1.0])
            self.embed.weight[4] = torch.tensor([1.0, 0.0, 0.0])
            self.embed.weight[5] = torch.tensor([0.0, 1.0, 0.0])
            self.embed.weight[6] = torch.tensor([0.0, 0.0, 1.0])
            self.lm_head.weight.zero_()
            self.lm_head.weight[4] = torch.tensor([1.0, 0.0, 0.0])
            self.lm_head.weight[5] = torch.tensor([0.0, 1.0, 0.0])
            self.lm_head.weight[6] = torch.tensor([0.0, 0.0, 1.0])

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden_states = self.embed(input_ids)
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        *,
        input_ids,
        max_new_tokens,
        do_sample,
        pad_token_id,
        eos_token_id,
        use_cache,
        attention_mask=None,
        **kwargs,
    ):
        generated = input_ids
        current_mask = attention_mask

        for _ in range(max_new_tokens):
            outputs = self.forward(generated, attention_mask=current_mask, use_cache=use_cache)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if current_mask is not None:
                current_mask = torch.cat([current_mask, torch.ones_like(next_token)], dim=1)

        return generated


class FakePipeline:
    def __init__(self, model, tokenizer, runtime):
        self.model = model
        self.tokenizer = tokenizer
        self.runtime = runtime

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        config: RuntimeGenerationConfig | None = None,
    ):
        if config is None:
            config = RuntimeGenerationConfig(
                max_new_tokens=max_new_tokens or 100,
                temperature=temperature or 0.0,
            )
        return self.runtime.generate(prompt, config)


def _make_fake_pipeline(device: str) -> FakePipeline:
    tokenizer = TinyTorchGenerationTokenizer()
    model = TinyTorchGenerationModel().eval().to(device)
    runtime = TorchInferenceRuntime(model, tokenizer, device=device)
    return FakePipeline(model, tokenizer, runtime)


@pytest.mark.parametrize("device", TORCH_DEVICES)
@pytest.mark.asyncio
async def test_generation_service_uses_torch_runtime(monkeypatch, device):
    import chuk_lazarus.inference as inference_module
    import chuk_lazarus.models_v2.core.backend as backend_module

    fake_pipeline = _make_fake_pipeline(device)
    captured = {}

    def fake_from_pretrained(model_id, pipeline_config=None, verbose=False):
        captured["model_id"] = model_id
        captured["pipeline_config"] = pipeline_config
        return fake_pipeline

    monkeypatch.setattr(
        backend_module,
        "get_backend",
        lambda *args, **kwargs: SimpleNamespace(name="torch", device=device),
    )
    monkeypatch.setattr(inference_module.UnifiedPipeline, "from_pretrained", fake_from_pretrained)

    result = await GenerationService.generate(
        GenerationConfig(
            model="fake/test-model",
            prompt="2+2=",
            max_tokens=1,
            temperature=0.0,
            backend="torch",
            device=device,
        )
    )

    assert captured["model_id"] == "fake/test-model"
    assert captured["pipeline_config"].backend_name == "torch"
    assert captured["pipeline_config"].device == device
    assert result.generated_text == "4"
    assert result.tokens == ["4"]
    assert result.answer_found is True
    assert result.answer_onset == 0
    assert result.is_answer_first is True


@pytest.mark.asyncio
async def test_generation_service_compare_format_uses_runtime(monkeypatch):
    import chuk_lazarus.inference as inference_module
    import chuk_lazarus.models_v2.core.backend as backend_module

    fake_pipeline = _make_fake_pipeline("cpu")

    monkeypatch.setattr(
        backend_module,
        "get_backend",
        lambda *args, **kwargs: SimpleNamespace(name="torch", device="cpu"),
    )
    monkeypatch.setattr(
        inference_module.UnifiedPipeline,
        "from_pretrained",
        lambda *args, **kwargs: fake_pipeline,
    )

    result = await GenerationService.generate(
        GenerationConfig(
            model="fake/test-model",
            prompt="2+2= ",
            max_tokens=1,
            temperature=0.0,
            compare_format=True,
            backend="torch",
            device="cpu",
        )
    )

    assert [entry["has_trailing_space"] for entry in result.comparison_results] == [False, True]
    assert [entry["output"] for entry in result.comparison_results] == ["4", "5"]
