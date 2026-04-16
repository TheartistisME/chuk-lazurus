"""Tests for introspect layer CLI commands."""

import sys
import tempfile
from argparse import Namespace
from types import ModuleType
from unittest.mock import MagicMock, patch

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


class TestIntrospectLayer:
    """Tests for introspect_layer command."""

    @pytest.fixture
    def layer_args(self):
        """Create arguments for layer command."""
        return Namespace(
            model="test-model",
            prompts="test1|test2",
            labels=None,
            layers=None,
            attention=False,
            output=None,
        )

    def test_layer_basic(self, layer_args, capsys):
        """Test basic layer analysis."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_cls.from_pretrained.return_value = mock_analyzer

            # Mock result
            mock_result = MagicMock()
            mock_result.layers = [0, 4, 8]
            mock_result.representations = {}
            mock_result.clusters = None
            mock_analyzer.analyze_representations.return_value = mock_result

            introspect_layer(layer_args)

            captured = capsys.readouterr()
            assert "Loading model" in captured.out

    def test_layer_from_file(self, layer_args):
        """Test loading prompts from file."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("prompt1\nprompt2\n")
            f.flush()

            layer_args.prompts = f"@{f.name}"

            with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
                mock_analyzer = MagicMock()
                mock_cls.from_pretrained.return_value = mock_analyzer

                mock_result = MagicMock()
                mock_result.layers = [0]
                mock_result.representations = {}
                mock_result.clusters = None
                mock_analyzer.analyze_representations.return_value = mock_result

                introspect_layer(layer_args)

    def test_layer_with_labels(self, layer_args, capsys):
        """Test layer analysis with labels."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        layer_args.labels = "A,B"

        with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_cls.from_pretrained.return_value = mock_analyzer

            mock_result = MagicMock()
            mock_result.layers = [0]
            mock_result.representations = {}

            # Mock cluster results
            mock_cluster = MagicMock()
            mock_cluster.within_cluster_similarity = {"A": 0.9, "B": 0.85}
            mock_cluster.between_cluster_similarity = {("A", "B"): 0.5}
            mock_cluster.separation_score = 0.35
            mock_result.clusters = {0: mock_cluster}

            mock_analyzer.analyze_representations.return_value = mock_result

            introspect_layer(layer_args)

            captured = capsys.readouterr()
            assert "Loading model" in captured.out

    def test_layer_specific_layers(self, layer_args):
        """Test specifying layers to analyze."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        layer_args.layers = "4,8,12"

        with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_cls.from_pretrained.return_value = mock_analyzer

            mock_result = MagicMock()
            mock_result.layers = [4, 8, 12]
            mock_result.representations = {}
            mock_result.clusters = None
            mock_analyzer.analyze_representations.return_value = mock_result

            introspect_layer(layer_args)

            # Check that analyze_representations was called with correct layers
            call_args = mock_analyzer.analyze_representations.call_args
            assert call_args[1]["layers"] == [4, 8, 12]

    def test_layer_with_attention(self, layer_args, capsys):
        """Test layer analysis with attention."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        layer_args.attention = True

        with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_cls.from_pretrained.return_value = mock_analyzer

            mock_result = MagicMock()
            mock_result.layers = [0]
            mock_result.representations = {}
            mock_result.clusters = None
            mock_analyzer.analyze_representations.return_value = mock_result
            mock_analyzer.analyze_attention.return_value = {}

            introspect_layer(layer_args)

            captured = capsys.readouterr()
            assert "Attention Analysis" in captured.out or "Loading" in captured.out

    def test_layer_with_attention_and_results(self, layer_args, capsys):
        """Test layer analysis with attention returning actual results (covers line 85)."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        layer_args.attention = True
        layer_args.layers = "0,4,8"

        with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_cls.from_pretrained.return_value = mock_analyzer

            mock_result = MagicMock()
            mock_result.layers = [0, 4, 8]
            mock_result.representations = {}
            mock_result.clusters = None
            mock_analyzer.analyze_representations.return_value = mock_result
            # Return attention results for layer 0 and 4
            mock_analyzer.analyze_attention.return_value = {
                0: MagicMock(),
                4: MagicMock(),
            }

            introspect_layer(layer_args)

            # Verify print_attention_comparison was called
            assert mock_analyzer.print_attention_comparison.call_count == 2

    def test_layer_label_count_mismatch(self, layer_args, capsys):
        """Test warning when label count doesn't match prompt count (covers lines 28-29)."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        layer_args.labels = "A,B,C"  # 3 labels for 2 prompts

        with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_cls.from_pretrained.return_value = mock_analyzer

            mock_result = MagicMock()
            mock_result.layers = [0]
            mock_result.representations = {}
            mock_result.clusters = None
            mock_analyzer.analyze_representations.return_value = mock_result

            introspect_layer(layer_args)

            captured = capsys.readouterr()
            assert "Warning:" in captured.out
            assert "3 labels" in captured.out
            assert "2 prompts" in captured.out

    def test_layer_low_separation_score(self, layer_args, capsys):
        """Test layer that does NOT distinguish groups (covers line 75)."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        layer_args.labels = "A,B"

        with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_cls.from_pretrained.return_value = mock_analyzer

            mock_result = MagicMock()
            mock_result.layers = [0]
            mock_result.representations = {}

            # Mock cluster with LOW separation score (< 0.02)
            mock_cluster = MagicMock()
            mock_cluster.within_cluster_similarity = {"A": 0.9, "B": 0.85}
            mock_cluster.between_cluster_similarity = {("A", "B"): 0.88}
            mock_cluster.separation_score = 0.01  # Low - does NOT distinguish
            mock_result.clusters = {0: mock_cluster}

            mock_analyzer.analyze_representations.return_value = mock_result

            introspect_layer(layer_args)

            captured = capsys.readouterr()
            assert "does NOT distinguish" in captured.out

    def test_layer_save_output(self, layer_args):
        """Test saving layer analysis results (covers lines 89-111)."""
        import json

        from chuk_lazarus.cli.commands.introspect import introspect_layer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            layer_args.output = f.name

        layer_args.labels = "A,B"

        with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_cls.from_pretrained.return_value = mock_analyzer

            mock_result = MagicMock()
            mock_result.layers = [0]
            mock_result.representations = {0: MagicMock(similarity_matrix=[[1.0, 0.5], [0.5, 1.0]])}

            # Mock cluster results
            mock_cluster = MagicMock()
            mock_cluster.within_cluster_similarity = {"A": 0.9, "B": 0.85}
            mock_cluster.between_cluster_similarity = {("A", "B"): 0.5}
            mock_cluster.separation_score = 0.35
            mock_result.clusters = {0: mock_cluster}

            mock_analyzer.analyze_representations.return_value = mock_result

            introspect_layer(layer_args)

            # Check file was created and has correct structure
            with open(layer_args.output) as f:
                data = json.load(f)
                assert "prompts" in data
                assert "layers" in data
                assert "similarity_matrices" in data
                assert "clusters" in data

    def test_layer_threads_backend_and_device_to_runtime_loader(self, layer_args):
        """Test torch backend/device use the runtime-aware analyzer loader."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        layer_args.backend = "torch"
        layer_args.device = "cuda:2"

        with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_cls.load.return_value = mock_analyzer

            mock_result = MagicMock()
            mock_result.layers = [0]
            mock_result.representations = {}
            mock_result.clusters = None
            mock_analyzer.analyze_representations.return_value = mock_result

            introspect_layer(layer_args)

            mock_cls.load.assert_called_once_with(
                "test-model",
                backend="torch",
                device="cuda:2",
                verbose=False,
            )
            mock_cls.from_pretrained.assert_not_called()


class TestIntrospectFormatSensitivity:
    """Tests for introspect_format_sensitivity command."""

    @pytest.fixture
    def format_args(self):
        """Create arguments for format sensitivity command."""
        return Namespace(
            model="test-model",
            prompts="test1|test2",
            layers=None,
        )

    def test_format_sensitivity_basic(self, format_args, capsys):
        """Test basic format sensitivity analysis."""
        from chuk_lazarus.cli.commands.introspect import introspect_format_sensitivity

        with patch("chuk_lazarus.introspection.analyze_format_sensitivity") as mock_fn:
            mock_result = MagicMock()
            mock_result.layers = [0, 4]

            mock_cluster = MagicMock()
            mock_cluster.separation_score = 0.05
            mock_result.clusters = {0: mock_cluster, 4: mock_cluster}

            mock_fn.return_value = mock_result

            introspect_format_sensitivity(format_args)

            captured = capsys.readouterr()
            assert "Format sensitivity" in captured.out

    def test_format_sensitivity_with_layers(self, format_args):
        """Test format sensitivity with specific layers."""
        from chuk_lazarus.cli.commands.introspect import introspect_format_sensitivity

        format_args.layers = "4,8"

        with patch("chuk_lazarus.introspection.analyze_format_sensitivity") as mock_fn:
            mock_result = MagicMock()
            mock_result.layers = [4, 8]
            mock_result.clusters = {}
            mock_fn.return_value = mock_result

            introspect_format_sensitivity(format_args)

            # Check layers were passed correctly
            call_args = mock_fn.call_args
            assert call_args[1]["layers"] == [4, 8]

    def test_format_sensitivity_from_file(self, format_args):
        """Test format sensitivity loading prompts from file (covers lines 120-121)."""
        from chuk_lazarus.cli.commands.introspect import introspect_format_sensitivity

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("prompt1\nprompt2 \n")  # One with trailing space to strip
            f.flush()

            format_args.prompts = f"@{f.name}"

        with patch("chuk_lazarus.introspection.analyze_format_sensitivity") as mock_fn:
            mock_result = MagicMock()
            mock_result.layers = [0]
            mock_result.clusters = {}
            mock_fn.return_value = mock_result

            introspect_format_sensitivity(format_args)

            # Check prompts were loaded from file (stripped of trailing space)
            call_args = mock_fn.call_args
            assert call_args[1]["base_prompts"] == ["prompt1", "prompt2"]

    def test_format_sensitivity_threads_backend_and_device(self, format_args):
        """Test backend/device are forwarded into the layer-analysis helper."""
        from chuk_lazarus.cli.commands.introspect import introspect_format_sensitivity

        format_args.backend = "torch"
        format_args.device = "cuda:1"

        with patch("chuk_lazarus.introspection.analyze_format_sensitivity") as mock_fn:
            mock_result = MagicMock()
            mock_result.layers = [0]
            mock_result.clusters = {}
            mock_fn.return_value = mock_result

            introspect_format_sensitivity(format_args)

            call_args = mock_fn.call_args
            assert call_args[1]["backend"] == "torch"
            assert call_args[1]["device"] == "cuda:1"
