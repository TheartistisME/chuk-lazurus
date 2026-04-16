"""Tests for introspect analyze CLI commands."""

import asyncio
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest


class TestIntrospectAnalyze:
    """Tests for introspect_analyze command."""

    @pytest.fixture
    def analyze_args(self):
        """Create arguments for analyze command."""
        return Namespace(
            model="test-model",
            prompt="2+2=",
            prefix=None,
            adapter=None,
            embedding_scale=None,
            raw=False,
            layers=None,
            all_layers=False,
            layer_strategy="evenly_spaced",
            layer_step=4,
            top_k=10,
            track=None,
            steer=None,
            steer_neuron=None,
            steer_layer=None,
            strength=None,
            inject_layer=None,
            inject_token=None,
            inject_blend=1.0,
            compute_override="none",
            compute_layer=None,
            find_answer=None,
            no_find_answer=False,
            gen_tokens=30,
            expected=None,
            output=None,
        )

    def test_analyze_requires_prompt(self):
        """Test that analyze requires --prompt or --prefix."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        args = Namespace(
            model="test-model",
            prompt=None,
            prefix=None,
            adapter=None,
            embedding_scale=None,
            raw=False,
            layers=None,
            all_layers=False,
            layer_strategy="evenly_spaced",
            layer_step=4,
            top_k=10,
            track=None,
            steer=None,
            steer_neuron=None,
            steer_layer=None,
            strength=None,
            inject_layer=None,
            inject_token=None,
            inject_blend=1.0,
            compute_override="none",
            compute_layer=None,
            find_answer=None,
            no_find_answer=False,
            gen_tokens=30,
            expected=None,
            output=None,
        )

        with pytest.raises(ValueError, match="--prompt"):
            asyncio.run(introspect_analyze(args))

    def test_analyze_basic(self, analyze_args, capsys):
        """Test basic analysis."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        asyncio.run(introspect_analyze(analyze_args))

        captured = capsys.readouterr()
        assert "LOGIT" in captured.out or "ANALYSIS" in captured.out

    def test_analyze_with_prefix(self, analyze_args, capsys):
        """Test analysis with prefix mode."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        analyze_args.prompt = None
        analyze_args.prefix = "The answer is"

        asyncio.run(introspect_analyze(analyze_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_analyze_with_custom_layers(self, analyze_args, capsys):
        """Test analysis with custom layer selection."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        analyze_args.layers = "0,4,8,12"

        asyncio.run(introspect_analyze(analyze_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_analyze_with_all_layers(self, analyze_args, capsys):
        """Test analysis with all layers."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        analyze_args.all_layers = True

        asyncio.run(introspect_analyze(analyze_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_analyze_raw_mode(self, analyze_args, capsys):
        """Test analysis in raw mode (no chat template)."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        analyze_args.raw = True

        asyncio.run(introspect_analyze(analyze_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_analyze_with_output(self, analyze_args, tmp_path, capsys):
        """Test analysis with output file."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        output_file = tmp_path / "analysis.json"
        analyze_args.output = str(output_file)

        asyncio.run(introspect_analyze(analyze_args))

        captured = capsys.readouterr()
        assert "saved to" in captured.out


class TestIntrospectCompare:
    """Tests for introspect_compare command."""

    @pytest.fixture
    def compare_args(self):
        """Create arguments for compare command."""
        return Namespace(
            model1="model-a",
            model2="model-b",
            prompt="2+2=",
            top_k=10,
            track=None,
        )

    def test_compare_basic(self, compare_args, capsys):
        """Test basic model comparison."""
        from chuk_lazarus.cli.commands.introspect import introspect_compare

        asyncio.run(introspect_compare(compare_args))

        captured = capsys.readouterr()
        assert "COMPARISON" in captured.out or "Model" in captured.out


class TestIntrospectHooks:
    """Tests for introspect_hooks command."""

    @pytest.fixture
    def hooks_args(self):
        """Create arguments for hooks command."""
        return Namespace(
            model="test-model",
            prompt="2+2=",
            layers=None,
            capture_attention=False,
            last_only=False,
            no_logit_lens=False,
        )

    def test_hooks_basic(self, hooks_args, capsys):
        """Test basic hooks demonstration."""
        from chuk_lazarus.cli.commands.introspect import introspect_hooks

        asyncio.run(introspect_hooks(hooks_args))

        captured = capsys.readouterr()
        assert "HOOKS" in captured.out or "Captured" in captured.out

    def test_hooks_with_layers(self, hooks_args, capsys):
        """Test hooks with custom layers."""
        from chuk_lazarus.cli.commands.introspect import introspect_hooks

        hooks_args.layers = "0,4,8,12"

        asyncio.run(introspect_hooks(hooks_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_hooks_with_attention(self, hooks_args, capsys):
        """Test hooks with attention capture."""
        from chuk_lazarus.cli.commands.introspect import introspect_hooks

        hooks_args.capture_attention = True

        asyncio.run(introspect_hooks(hooks_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""


class TestAnalysisConfig:
    """Tests for analysis configuration types."""

    def test_delimiters(self):
        """Test delimiter constants."""
        from chuk_lazarus.cli.commands._constants import Delimiters

        assert Delimiters.LAYER_SEPARATOR == ","
        assert Delimiters.PROMPT_SEPARATOR == "|"


class TestIntrospectAnalyzeTorchRuntime:
    """Real analyzer/CLI wiring test for the torch runtime path."""

    def test_analyze_uses_real_torch_analyzer_path(self):
        pytest.importorskip("torch")

        repo_root = Path(__file__).resolve().parents[4]
        src = repo_root / "src"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")

        code = """
import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from chuk_lazarus.cli.commands.introspect.analyze import introspect_analyze
from chuk_lazarus.models_v2.core.backend import get_backend, reset_backend


class TinyTorchBlock(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.GELU()

    def forward(self, x, output_attentions: bool = False, **_):
        hidden = x + self.activation(self.linear(self.norm(x)))
        if not output_attentions:
            return hidden
        batch, seq_len, _ = hidden.shape
        attn = torch.zeros(batch, 1, seq_len, seq_len, dtype=hidden.dtype, device=hidden.device)
        return hidden, attn


class TinyTorchBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 12)
        self.layers = torch.nn.ModuleList([TinyTorchBlock(12) for _ in range(3)])
        self.norm = torch.nn.LayerNorm(12)
        self.hidden_size = 12

    def forward(self, input_ids, output_attentions: bool = False, **_):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            layer_out = layer(hidden, output_attentions=output_attentions)
            hidden = layer_out[0] if output_attentions else layer_out
        return self.norm(hidden), None


class TinyTorchForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyTorchBackbone()
        self.lm_head = torch.nn.Linear(12, 32, bias=False)
        self.tie_word_embeddings = False

    def forward(self, input_ids=None, output_attentions: bool = False, **_):
        hidden, _ = self.model(input_ids=input_ids, output_attentions=output_attentions)
        return SimpleNamespace(logits=self.lm_head(hidden))


class TinyTokenizer:
    vocab_size = 32

    def encode(self, text: str):
        encoded = [ord(ch) % self.vocab_size for ch in text[:5]]
        return encoded or [0]

    def decode(self, ids):
        return f"<{ids[0]}>"


async def main():
    reset_backend()
    get_backend("torch", device="cpu")
    args = Namespace(
        model="tiny-torch",
        prompt="2+2=",
        prefix=None,
        adapter=None,
        embedding_scale=None,
        raw=False,
        layers=None,
        all_layers=True,
        layer_strategy="evenly_spaced",
        layer_step=4,
        top_k=3,
        track=None,
        steer=None,
        steer_neuron=None,
        steer_layer=None,
        strength=None,
        inject_layer=None,
        inject_token=None,
        inject_blend=1.0,
        compute_override="none",
        compute_layer=None,
        find_answer=None,
        no_find_answer=False,
        gen_tokens=30,
        expected=None,
        output=None,
    )
    model = TinyTorchForCausalLM()
    tokenizer = TinyTokenizer()
    config = SimpleNamespace(
        num_hidden_layers=3,
        hidden_size=12,
        vocab_size=32,
        tie_word_embeddings=False,
    )
    with patch(
        "chuk_lazarus.introspection.analyzer.core._load_model_sync",
        return_value=(model, tokenizer, config),
    ):
        await introspect_analyze(args)
    reset_backend()


asyncio.run(main())
"""

        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
        assert "Logit Lens Analysis" in result.stdout
        assert "Final prediction" in result.stdout
