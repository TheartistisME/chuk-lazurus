"""Torch-specific tests for the ablation study runtime path."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.backends import TorchInferenceRuntime
from chuk_lazarus.introspection.ablation.adapter import ModelAdapter
from chuk_lazarus.introspection.ablation.config import AblationConfig, ComponentType
from chuk_lazarus.introspection.ablation.study import AblationStudy


class TorchMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.down_proj.weight.fill_(1.0)


class TorchAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = torch.nn.Linear(1, 1, bias=False)


class TorchLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TorchMLP()
        self.self_attn = TorchAttention()


class TorchBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([TorchLayer()])


class TorchAblationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TorchBackbone()

    def generate(self, input_ids, **kwargs):
        del kwargs
        weight_sum = float(self.model.layers[0].mlp.down_proj.weight.sum().item())
        next_token = 4 if weight_sum == 0.0 else 6
        next_ids = torch.tensor([[next_token]], device=input_ids.device, dtype=input_ids.dtype)
        return torch.cat([input_ids, next_ids], dim=1)


class FailingTorchAblationModel(TorchAblationModel):
    def generate(self, input_ids, **kwargs):
        if float(self.model.layers[0].mlp.down_proj.weight.sum().item()) == 0.0:
            raise RuntimeError("generation failed")
        return super().generate(input_ids, **kwargs)


class TorchTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, prompt, return_tensors="pt"):
        del prompt, return_tensors
        return {
            "input_ids": torch.tensor([[2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def encode(self, prompt, return_tensors=None):
        del prompt
        tokens = [2]
        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long)
        return tokens

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        token_id = ids[0] if isinstance(ids, list) else ids
        return "ablated" if token_id == 4 else "baseline"


def _make_study(model):
    tokenizer = TorchTokenizer()
    config = SimpleNamespace(hidden_size=1)
    runtime = TorchInferenceRuntime(model, tokenizer, device="cpu")
    adapter = ModelAdapter(model, tokenizer, config, runtime=runtime)
    return AblationStudy(adapter)


def test_ablate_and_generate_uses_torch_runtime_and_restores_weights():
    study = _make_study(TorchAblationModel().eval())
    config = AblationConfig(max_new_tokens=1, temperature=0.0)

    baseline = study.ablate_and_generate(
        "prompt",
        layers=[],
        component=ComponentType.MLP,
        config=config,
    )
    original_weight = study.adapter.clone_weight(study.adapter.get_mlp_down_weight(0))

    ablated = study.ablate_and_generate(
        "prompt",
        layers=[0],
        component=ComponentType.MLP,
        config=config,
    )

    assert baseline == "baseline"
    assert ablated == "ablated"
    assert torch.allclose(study.adapter.get_mlp_down_weight(0), original_weight)


def test_ablate_and_generate_restores_torch_weights_on_error():
    study = _make_study(FailingTorchAblationModel().eval())
    config = AblationConfig(max_new_tokens=1, temperature=0.0)
    original_weight = study.adapter.clone_weight(study.adapter.get_mlp_down_weight(0))

    with pytest.raises(RuntimeError, match="generation failed"):
        study.ablate_and_generate(
            "prompt",
            layers=[0],
            component=ComponentType.MLP,
            config=config,
        )

    assert torch.allclose(study.adapter.get_mlp_down_weight(0), original_weight)
