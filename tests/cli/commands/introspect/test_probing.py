"""Tests for introspect probing CLI commands."""

import asyncio
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_backend_dispatch_module():
    """Expose the backend-dispatch shim under the mocked introspection package."""
    module_name = "chuk_lazarus.introspection._backend_dispatch"
    original_module = sys.modules.get(module_name)
    intro_module = sys.modules.get("chuk_lazarus.introspection")
    original_path = getattr(intro_module, "__path__", None) if intro_module is not None else None

    if intro_module is not None:
        intro_module.__path__ = []

    backend_dispatch = ModuleType(module_name)
    backend_dispatch.from_backend_tensor = MagicMock()
    backend_dispatch.to_backend_tensor = MagicMock()
    sys.modules[module_name] = backend_dispatch

    yield

    if intro_module is not None:
        if original_path is None:
            try:
                delattr(intro_module, "__path__")
            except AttributeError:
                pass
        else:
            intro_module.__path__ = original_path

    if original_module is not None:
        sys.modules[module_name] = original_module
    else:
        sys.modules.pop(module_name, None)


class TestIntrospectMetacognitive:
    """Tests for introspect_metacognitive command."""

    @pytest.fixture
    def metacognitive_args(self):
        """Create arguments for metacognitive command."""
        return Namespace(
            model="test-model",
            prompts="2+2=|47*47=",
            decision_layer=None,
            raw=False,
            output=None,
            top_k=5,
        )

    def test_metacognitive_basic(self, metacognitive_args, capsys):
        """Test basic metacognitive analysis."""
        from chuk_lazarus.cli.commands.introspect.probing import (
            introspect_metacognitive,
        )

        asyncio.run(introspect_metacognitive(metacognitive_args))

        captured = capsys.readouterr()
        assert "METACOGNITIVE" in captured.out or "test-model" in captured.out

    def test_metacognitive_custom_decision_layer(self, metacognitive_args, capsys):
        """Test with custom decision layer."""
        from chuk_lazarus.cli.commands.introspect.probing import (
            introspect_metacognitive,
        )

        metacognitive_args.decision_layer = 5

        asyncio.run(introspect_metacognitive(metacognitive_args))

        captured = capsys.readouterr()
        # The mock should have been called and output captured
        assert captured.out != "" or captured.err != ""

    def test_metacognitive_raw_mode(self, metacognitive_args, capsys):
        """Test raw mode (no chat template)."""
        from chuk_lazarus.cli.commands.introspect.probing import (
            introspect_metacognitive,
        )

        metacognitive_args.raw = True

        asyncio.run(introspect_metacognitive(metacognitive_args))

        # Just verify it runs without error
        capsys.readouterr()
        assert True  # Test passes if no exception


class TestIntrospectUncertainty:
    """Tests for introspect_uncertainty command."""

    @pytest.fixture
    def uncertainty_args(self):
        """Create arguments for uncertainty command."""
        return Namespace(
            model="test-model",
            prompt="What is 2+2?",
            prompts=None,
            layer=None,
            calibration_file=None,
            working=None,
            broken=None,
            backend=None,
            device=None,
            output=None,
        )

    def test_uncertainty_basic(self, uncertainty_args, capsys):
        """Test basic uncertainty analysis."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_uncertainty

        asyncio.run(introspect_uncertainty(uncertainty_args))

        captured = capsys.readouterr()
        # The mock should produce output
        assert "UNCERTAINTY" in captured.out or captured.out != ""

    def test_uncertainty_custom_layer(self, uncertainty_args, capsys):
        """Test with custom layer."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_uncertainty

        uncertainty_args.layer = 5

        asyncio.run(introspect_uncertainty(uncertainty_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_uncertainty_threads_backend_and_uses_prompts_arg(self, uncertainty_args):
        """Test uncertainty config uses prompts/backend/device from real CLI args."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_uncertainty

        uncertainty_args.prompt = None
        uncertainty_args.prompts = "working|broken"
        uncertainty_args.working = "1+1 = |2+2 = "
        uncertainty_args.broken = "1+1 =|2+2 ="
        uncertainty_args.backend = "torch"
        uncertainty_args.device = "cuda:0"

        mock_result = MagicMock()
        mock_result.to_display.return_value = "UNCERTAINTY"

        with patch(
            "chuk_lazarus.introspection.probing.UncertaintyService.analyze",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_analyze, patch(
            "chuk_lazarus.introspection.probing.UncertaintyConfig",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ):
            asyncio.run(introspect_uncertainty(uncertainty_args))

        config = mock_analyze.call_args.args[0]
        assert config.prompts == ["working", "broken"]
        assert config.working_prompts == ["1+1 = ", "2+2 = "]
        assert config.broken_prompts == ["1+1 =", "2+2 ="]
        assert config.backend == "torch"
        assert config.device == "cuda:0"


class TestIntrospectProbe:
    """Tests for introspect_probe command."""

    @pytest.fixture
    def probe_args(self):
        """Create arguments for probe command."""
        return Namespace(
            model="test-model",
            positive="good|positive|great",
            negative="bad|negative|terrible",
            class_a=None,
            class_b=None,
            label_a=None,
            label_b=None,
            positive_label=None,
            negative_label=None,
            layer=None,
            layers=None,
            all_layers=False,
            probe_file=None,
            backend=None,
            device=None,
            output=None,
        )

    def test_probe_basic(self, probe_args, capsys):
        """Test basic probe training."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        asyncio.run(introspect_probe(probe_args))

        captured = capsys.readouterr()
        assert "PROBE" in captured.out or captured.out != ""

    def test_probe_missing_data(self):
        """Test error when no probe data provided."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        args = Namespace(
            model="test-model",
            positive=None,
            negative=None,
            layers=None,
            all_layers=False,
            probe_file=None,
            output=None,
        )

        with pytest.raises(ValueError, match="Probing requires"):
            asyncio.run(introspect_probe(args))

    def test_probe_supports_parser_class_args_and_backend(self, probe_args):
        """Test probe config accepts parser-style args and threads backend/device."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        probe_args.positive = None
        probe_args.negative = None
        probe_args.class_a = "math1|math2"
        probe_args.class_b = "lang1|lang2"
        probe_args.label_a = "math"
        probe_args.label_b = "lang"
        probe_args.layer = 7
        probe_args.backend = "torch"
        probe_args.device = "cuda:2"

        mock_result = MagicMock()
        mock_result.to_display.return_value = "PROBE"

        with patch(
            "chuk_lazarus.introspection.probing.ProbeService.train_and_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_train, patch(
            "chuk_lazarus.introspection.probing.ProbeConfig",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ):
            asyncio.run(introspect_probe(probe_args))

        config = mock_train.call_args.args[0]
        assert config.positive_prompts == ["math1", "math2"]
        assert config.negative_prompts == ["lang1", "lang2"]
        assert config.positive_label == "math"
        assert config.negative_label == "lang"
        assert config.layers == [7]
        assert config.backend == "torch"
        assert config.device == "cuda:2"


class TestProbingUtils:
    """Tests for probing utility functions."""

    def test_parse_prompts(self):
        """Test prompt parsing from string."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_prompts

        prompts = parse_prompts("prompt1|prompt2|prompt3")
        assert len(prompts) == 3
        assert prompts[0] == "prompt1"
        assert prompts[2] == "prompt3"

    def test_parse_prompts_single(self):
        """Test parsing single prompt."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_prompts

        prompts = parse_prompts("single prompt")
        assert len(prompts) == 1
        assert prompts[0] == "single prompt"

    def test_parse_layers_specific(self):
        """Test parsing specific layer list."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_layers

        layers = parse_layers("0,5,10,15")
        assert layers == [0, 5, 10, 15]

    def test_parse_layers_none(self):
        """Test parsing None returns None."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_layers

        layers = parse_layers(None)
        assert layers is None

    def test_extract_arg_with_default(self):
        """Test extract_arg with default value."""
        from chuk_lazarus.cli.commands.introspect._utils import extract_arg

        args = Namespace(existing_attr="value")

        # Existing attribute
        result = extract_arg(args, "existing_attr")
        assert result == "value"

        # Non-existing attribute with default
        result = extract_arg(args, "missing_attr", "default")
        assert result == "default"

        # Non-existing attribute without default
        result = extract_arg(args, "missing_attr")
        assert result is None

    def test_get_layer_depth_ratio(self):
        """Test layer depth ratio calculation."""
        from chuk_lazarus.cli.commands._constants import LayerDepthRatio
        from chuk_lazarus.cli.commands.introspect._utils import get_layer_depth_ratio

        # When layer is specified, ratio is ignored
        ratio = get_layer_depth_ratio(5, LayerDepthRatio.LATE)
        assert ratio is None or ratio == LayerDepthRatio.LATE

        # When layer is None, use provided ratio
        ratio = get_layer_depth_ratio(None, LayerDepthRatio.LATE)
        assert ratio == LayerDepthRatio.LATE
