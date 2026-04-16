"""Tests for introspect neurons CLI commands."""

import sys
import tempfile
from argparse import Namespace
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
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


class TestNeuronAnalysisConfig:
    """Tests for NeuronAnalysisConfig."""

    def test_from_args_basic(self):
        """Test creating config from args."""
        from chuk_lazarus.cli.commands.introspect._types import NeuronAnalysisConfig

        args = Namespace(
            model="test-model",
            layer=12,
            layers=None,
            prompts="2+2=|47*47=",
            neurons="100,200",
            from_direction=None,
            top_k=10,
            labels=None,
            output=None,
            steer=None,
            strength=None,
            auto_discover=False,
            neuron_names=None,
        )

        config = NeuronAnalysisConfig.from_args(args)

        assert config.model == "test-model"
        assert config.layer == 12
        assert config.prompts == "2+2=|47*47="
        assert config.neurons == "100,200"
        assert config.top_k == 10

    def test_from_args_with_layers(self):
        """Test creating config with layers string."""
        from chuk_lazarus.cli.commands.introspect._types import NeuronAnalysisConfig

        args = Namespace(
            model="test-model",
            layer=None,
            layers="4,8,12",
            prompts="test",
            neurons="100",
            from_direction=None,
            top_k=10,
            labels=None,
            output=None,
            steer=None,
            strength=None,
            auto_discover=False,
            neuron_names=None,
        )

        config = NeuronAnalysisConfig.from_args(args)

        assert config.layers == "4,8,12"

    def test_from_args_with_auto_discover(self):
        """Test creating config with auto-discover."""
        from chuk_lazarus.cli.commands.introspect._types import NeuronAnalysisConfig

        args = Namespace(
            model="test-model",
            layer=12,
            layers=None,
            prompts="easy|hard",
            neurons=None,
            from_direction=None,
            top_k=5,
            labels="easy|hard",
            output=None,
            steer=None,
            strength=None,
            auto_discover=True,
            neuron_names=None,
        )

        config = NeuronAnalysisConfig.from_args(args)

        assert config.auto_discover is True
        assert config.labels == "easy|hard"


class TestIntrospectNeurons:
    """Tests for introspect_neurons command."""

    @pytest.fixture
    def neurons_args(self):
        """Create arguments for neurons command."""
        return Namespace(
            model="test-model",
            layer=12,
            layers=None,
            prompts="2+2=|47*47=",
            neurons="100,200",
            from_direction=None,
            top_k=10,
            labels=None,
            output=None,
            steer=None,
            strength=None,
            auto_discover=False,
            neuron_names=None,
            backend=None,
            device=None,
        )

    def test_neurons_basic(self, neurons_args, capsys):
        """Test basic neuron analysis."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    def test_neurons_no_layer_specified(self, neurons_args, capsys):
        """Test error when no layer specified."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.layer = None
        neurons_args.layers = None

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "ERROR: Must specify --layer or --layers" in captured.out

    def test_neurons_with_layers_string(self, neurons_args, capsys):
        """Test neurons with multiple layers specified as comma-separated string."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.layer = None
        neurons_args.layers = "4,8,12"

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Analyzing layers: [4, 8, 12]" in captured.out

    def test_neurons_with_labels(self, neurons_args, capsys):
        """Test neuron analysis with labels."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.labels = "easy|hard"

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Loading" in captured.out

    def test_neurons_with_neuron_names(self, neurons_args, capsys):
        """Test neuron analysis with custom neuron names."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neuron_names = "carry_detector|result_encoder"

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Neuron names:" in captured.out

    def test_neurons_neuron_names_mismatch(self, neurons_args, capsys):
        """Test warning when neuron name count doesn't match neuron count."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neuron_names = "only_one_name"  # 1 name for 2 neurons

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        # Names won't be used if count doesn't match
        assert "Neuron names:" not in captured.out

    def test_neurons_no_source_error(self, neurons_args, capsys):
        """Test error when no neuron source specified."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.from_direction = None
        neurons_args.auto_discover = False
        neurons_args.labels = None

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "ERROR: Must specify --neurons, --from-direction, or --auto-discover" in captured.out

    def test_neurons_auto_discover_with_labels(self, neurons_args, capsys):
        """Test auto-discover mode with labels."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.labels = "easy|hard"
        neurons_args.auto_discover = True

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Auto-discovering discriminative neurons" in captured.out

    def test_neurons_auto_discover_inferred(self, neurons_args, capsys):
        """Test that auto-discover is inferred when labels but no neurons."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.labels = "cat1|cat2"
        # Don't set auto_discover=True - it should be inferred

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Auto-discovering discriminative neurons" in captured.out

    def test_neurons_label_count_mismatch(self, neurons_args, capsys):
        """Test warning when label count doesn't match prompt count."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.labels = "easy"  # Only 1 label for 2 prompts

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Warning: 1 labels for 2 prompts" in captured.out

    def test_neurons_threads_backend_and_device(self, neurons_args):
        """Test neuron analysis threads backend/device to the service."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neuron_module = sys.modules["chuk_lazarus.introspection.steering.neuron_service"]
        neurons_args.backend = "torch"
        neurons_args.device = "cuda:2"

        introspect_neurons(neurons_args)

        call = neuron_module.NeuronAnalysisService.analyze_neurons.call_args
        assert call.kwargs["backend"] == "torch"
        assert call.kwargs["device"] == "cuda:2"


class TestDirectionComparisonConfig:
    """Tests for DirectionComparisonConfig."""

    def test_from_args(self):
        """Test creating config from args."""
        from chuk_lazarus.cli.commands.introspect._types import (
            DirectionComparisonConfig,
        )

        args = Namespace(
            files=["dir1.npz", "dir2.npz"],
            threshold=0.1,
            output=None,
        )

        config = DirectionComparisonConfig.from_args(args)

        assert len(config.files) == 2
        assert config.threshold == 0.1


class TestIntrospectDirections:
    """Tests for introspect_directions command."""

    @pytest.fixture
    def directions_args(self):
        """Create arguments for directions command."""
        return Namespace(
            files=["dir1.npz", "dir2.npz"],
            threshold=0.1,
            output=None,
        )

    def test_directions_single_file_error(self, directions_args, capsys):
        """Test error when only one file provided."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        directions_args.files = ["single.npz"]

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "ERROR: Need at least 2 direction files" in captured.out

    def test_directions_file_not_found(self, directions_args, capsys):
        """Test error when file doesn't exist."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        directions_args.files = ["/nonexistent/path.npz", "/another/fake.npz"]

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "ERROR: File not found" in captured.out

    def test_directions_basic(self, directions_args, capsys):
        """Test comparing direction vectors."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        # Create direction files
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=np.random.randn(768).astype(np.float32), layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(f2.name, direction=np.random.randn(768).astype(np.float32), layer=12)
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "Loading" in captured.out or "COSINE SIMILARITY MATRIX" in captured.out

    def test_directions_with_labels(self, directions_args, capsys):
        """Test directions with label metadata."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(
                f1.name,
                direction=np.random.randn(768).astype(np.float32),
                layer=12,
                label_positive="positive",
                label_negative="negative",
            )
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(
                f2.name,
                direction=np.random.randn(768).astype(np.float32),
                layer=12,
                label_positive="correct",
                label_negative="wrong",
            )
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "negative->positive" in captured.out
        assert "wrong->correct" in captured.out

    def test_directions_aligned_vectors(self, directions_args, capsys):
        """Test with highly aligned direction vectors."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        base_direction = np.random.randn(768).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=base_direction, layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            # Same direction = highly aligned
            np.savez(f2.name, direction=base_direction * 0.9, layer=12)
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "Aligned" in captured.out or "HIGHLY correlated" in captured.out

    def test_directions_save_output(self, directions_args, capsys):
        """Test saving direction comparison results."""
        import json
        from pathlib import Path

        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=np.random.randn(768).astype(np.float32), layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(f2.name, direction=np.random.randn(768).astype(np.float32), layer=12)
            directions_args.files[1] = f2.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
            directions_args.output = out.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "Results saved to" in captured.out

        if Path(directions_args.output).exists():
            with open(directions_args.output) as f:
                data = json.load(f)
            assert "pairs" in data


class TestIntrospectOperandDirections:
    """Tests for operand-directions CLI command."""

    def test_operand_directions_threads_backend_to_unified_pipeline(self, capsys):
        """Test operand-directions uses UnifiedPipeline with backend/device."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_operand_directions

        class FakeRuntime:
            def extract_residual_state(self, prompt: str, layer_index: int):
                del prompt
                return SimpleNamespace(
                    tensor=np.array([[float(layer_index), 0.0]], dtype=np.float32)
                )

        fake_pipeline = SimpleNamespace(
            model=SimpleNamespace(),
            tokenizer=SimpleNamespace(),
            config=SimpleNamespace(num_hidden_layers=8),
            runtime=FakeRuntime(),
        )

        args = Namespace(
            model="test-model",
            digits="2,3",
            operation="*",
            layers="1,2",
            output=None,
            backend="torch",
            device="cuda:0",
        )

        with patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=fake_pipeline,
        ) as mock_from_pretrained:
            introspect_operand_directions(args)

        call = mock_from_pretrained.call_args
        assert call.kwargs["pipeline_config"].backend_name == "torch"
        assert call.kwargs["pipeline_config"].device == "cuda:0"
        captured = capsys.readouterr()
        assert "LAYER 1" in captured.out


class TestDirectionPairSimilarity:
    """Tests for DirectionPairSimilarity result type."""

    def test_basic_creation(self):
        """Test creating pair similarity result."""
        from chuk_lazarus.cli.commands.introspect._types import DirectionPairSimilarity

        pair = DirectionPairSimilarity(
            name_a="positive->negative",
            name_b="correct->wrong",
            cosine_similarity=0.8,
            orthogonal=False,
        )

        assert pair.name_a == "positive->negative"
        assert pair.cosine_similarity == 0.8
        assert pair.orthogonal is False


class TestParseLayers:
    """Tests for layer parsing utility."""

    def test_parse_layers_string(self):
        """Test parsing layer string."""
        from chuk_lazarus.cli.commands.introspect._types import parse_layers_string

        layers = parse_layers_string("4,8,12")
        assert layers == [4, 8, 12]

    def test_parse_layers_single(self):
        """Test parsing single layer."""
        from chuk_lazarus.cli.commands.introspect._types import parse_layers_string

        layers = parse_layers_string("6")
        assert layers == [6]

    def test_parse_layers_with_spaces(self):
        """Test parsing layers with spaces."""
        from chuk_lazarus.cli.commands.introspect._types import parse_layers_string

        layers = parse_layers_string("4, 8, 12")
        assert layers == [4, 8, 12]


class TestPrintNeuronResults:
    """Tests for _print_neuron_results helper function."""

    def test_print_single_layer_results(self, capsys):
        """Test printing results for single layer."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.neurons import _print_neuron_results

        # Create mock results
        result1 = MagicMock()
        result1.neuron_idx = 100
        result1.min_val = -5.0
        result1.max_val = 10.0
        result1.mean_val = 2.5
        result1.std_val = 3.0

        result2 = MagicMock()
        result2.neuron_idx = 200
        result2.min_val = -2.0
        result2.max_val = 8.0
        result2.mean_val = 3.0
        result2.std_val = 2.5

        results = {12: [result1, result2]}

        _print_neuron_results(
            results=results,
            neurons=[100, 200],
            prompts=["2+2=", "47*47="],
            labels=None,
            neuron_names={},
            neuron_weights={},
            neuron_stats={},
        )

        captured = capsys.readouterr()
        assert "NEURON ACTIVATION MAP AT LAYER 12" in captured.out
        assert "Neuron  100" in captured.out or "N  100" in captured.out

    def test_print_multi_layer_results(self, capsys):
        """Test printing results for multiple layers."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.neurons import _print_neuron_results

        result1_l4 = MagicMock()
        result1_l4.neuron_idx = 100
        result1_l4.min_val = -5.0
        result1_l4.max_val = 10.0
        result1_l4.mean_val = 2.5
        result1_l4.std_val = 3.0

        result1_l12 = MagicMock()
        result1_l12.neuron_idx = 100
        result1_l12.min_val = -3.0
        result1_l12.max_val = 8.0
        result1_l12.mean_val = 3.0
        result1_l12.std_val = 2.0

        results = {4: [result1_l4], 12: [result1_l12]}

        _print_neuron_results(
            results=results,
            neurons=[100],
            prompts=["2+2="],
            labels=None,
            neuron_names={},
            neuron_weights={},
            neuron_stats={},
        )

        captured = capsys.readouterr()
        assert "CROSS-LAYER NEURON TRACKING" in captured.out
        assert "L 4" in captured.out or "L4" in captured.out
        assert "L12" in captured.out or "L 12" in captured.out

    def test_print_results_with_labels(self, capsys):
        """Test printing results with labels."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.neurons import _print_neuron_results

        result1 = MagicMock()
        result1.neuron_idx = 100
        result1.min_val = -5.0
        result1.max_val = 10.0
        result1.mean_val = 2.5
        result1.std_val = 3.0

        results = {12: [result1]}

        _print_neuron_results(
            results=results,
            neurons=[100],
            prompts=["2+2=", "hard problem"],
            labels=["easy", "hard"],
            neuron_names={},
            neuron_weights={},
            neuron_stats={},
        )

        captured = capsys.readouterr()
        assert "Label" in captured.out

    def test_print_results_with_neuron_names(self, capsys):
        """Test printing results with custom neuron names."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.neurons import _print_neuron_results

        result1 = MagicMock()
        result1.neuron_idx = 100
        result1.min_val = -5.0
        result1.max_val = 10.0
        result1.mean_val = 2.5
        result1.std_val = 3.0

        results = {12: [result1]}

        _print_neuron_results(
            results=results,
            neurons=[100],
            prompts=["2+2="],
            labels=None,
            neuron_names={100: "carry"},
            neuron_weights={},
            neuron_stats={},
        )

        captured = capsys.readouterr()
        assert "carry" in captured.out

    def test_print_results_with_weights(self, capsys):
        """Test printing results with neuron weights."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.neurons import _print_neuron_results

        result1 = MagicMock()
        result1.neuron_idx = 100
        result1.min_val = -5.0
        result1.max_val = 10.0
        result1.mean_val = 2.5
        result1.std_val = 3.0

        results = {12: [result1]}

        _print_neuron_results(
            results=results,
            neurons=[100],
            prompts=["2+2="],
            labels=None,
            neuron_names={},
            neuron_weights={100: 0.5},
            neuron_stats={},
        )

        captured = capsys.readouterr()
        assert "weight" in captured.out
        assert "POSITIVE detector" in captured.out

    def test_print_results_with_negative_weight(self, capsys):
        """Test printing results with negative neuron weight."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.neurons import _print_neuron_results

        result1 = MagicMock()
        result1.neuron_idx = 100
        result1.min_val = -5.0
        result1.max_val = 10.0
        result1.mean_val = 2.5
        result1.std_val = 3.0

        results = {12: [result1]}

        _print_neuron_results(
            results=results,
            neurons=[100],
            prompts=["2+2="],
            labels=None,
            neuron_names={},
            neuron_weights={100: -0.3},
            neuron_stats={},
        )

        captured = capsys.readouterr()
        assert "NEGATIVE detector" in captured.out

    def test_print_results_with_stats(self, capsys):
        """Test printing results with neuron stats."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.neurons import _print_neuron_results

        result1 = MagicMock()
        result1.neuron_idx = 100
        result1.min_val = -5.0
        result1.max_val = 10.0
        result1.mean_val = 2.5
        result1.std_val = 3.0

        results = {12: [result1]}

        _print_neuron_results(
            results=results,
            neurons=[100],
            prompts=["2+2="],
            labels=None,
            neuron_names={},
            neuron_weights={},
            neuron_stats={100: {"separation": 0.85, "best_pair": ("easy", "hard")}},
        )

        captured = capsys.readouterr()
        assert "separation" in captured.out
        assert "easy vs hard" in captured.out


class TestPrintDirectionComparison:
    """Tests for _print_direction_comparison helper function."""

    def test_print_orthogonal_directions(self, capsys):
        """Test printing orthogonal direction comparison."""
        from chuk_lazarus.cli.commands.introspect._types import (
            DirectionComparisonResult,
            DirectionPairSimilarity,
        )
        from chuk_lazarus.cli.commands.introspect.neurons import (
            _print_direction_comparison,
        )

        pairs = [
            DirectionPairSimilarity(
                name_a="pos->neg",
                name_b="correct->wrong",
                cosine_similarity=0.05,
                orthogonal=True,
            )
        ]

        result = DirectionComparisonResult(
            files=["dir1.npz", "dir2.npz"],
            names=["pos->neg", "correct->wrong"],
            pairs=pairs,
            mean_abs_similarity=0.05,
        )

        _print_direction_comparison(result, threshold=0.1)

        captured = capsys.readouterr()
        assert "COSINE SIMILARITY MATRIX" in captured.out
        assert "Orthogonal" in captured.out
        assert "ORTHOGONAL" in captured.out or "independent" in captured.out.lower()

    def test_print_aligned_directions(self, capsys):
        """Test printing aligned direction comparison."""
        from chuk_lazarus.cli.commands.introspect._types import (
            DirectionComparisonResult,
            DirectionPairSimilarity,
        )
        from chuk_lazarus.cli.commands.introspect.neurons import (
            _print_direction_comparison,
        )

        pairs = [
            DirectionPairSimilarity(
                name_a="pos->neg",
                name_b="correct->wrong",
                cosine_similarity=0.85,
                orthogonal=False,
            )
        ]

        result = DirectionComparisonResult(
            files=["dir1.npz", "dir2.npz"],
            names=["pos->neg", "correct->wrong"],
            pairs=pairs,
            mean_abs_similarity=0.85,
        )

        _print_direction_comparison(result, threshold=0.1)

        captured = capsys.readouterr()
        assert "Aligned" in captured.out
        assert "HIGHLY correlated" in captured.out or "redundant" in captured.out.lower()

    def test_print_moderate_similarity(self, capsys):
        """Test printing directions with moderate similarity."""
        from chuk_lazarus.cli.commands.introspect._types import (
            DirectionComparisonResult,
            DirectionPairSimilarity,
        )
        from chuk_lazarus.cli.commands.introspect.neurons import (
            _print_direction_comparison,
        )

        pairs = [
            DirectionPairSimilarity(
                name_a="dir1",
                name_b="dir2",
                cosine_similarity=0.35,
                orthogonal=False,
            )
        ]

        result = DirectionComparisonResult(
            files=["dir1.npz", "dir2.npz"],
            names=["dir1", "dir2"],
            pairs=pairs,
            mean_abs_similarity=0.35,
        )

        _print_direction_comparison(result, threshold=0.1)

        captured = capsys.readouterr()
        assert "MODERATE correlation" in captured.out

    def test_print_multiple_pairs(self, capsys):
        """Test printing comparison with multiple pairs."""
        from chuk_lazarus.cli.commands.introspect._types import (
            DirectionComparisonResult,
            DirectionPairSimilarity,
        )
        from chuk_lazarus.cli.commands.introspect.neurons import (
            _print_direction_comparison,
        )

        pairs = [
            DirectionPairSimilarity(
                name_a="dir1", name_b="dir2", cosine_similarity=0.05, orthogonal=True
            ),
            DirectionPairSimilarity(
                name_a="dir1", name_b="dir3", cosine_similarity=0.75, orthogonal=False
            ),
            DirectionPairSimilarity(
                name_a="dir2", name_b="dir3", cosine_similarity=0.15, orthogonal=False
            ),
        ]

        result = DirectionComparisonResult(
            files=["dir1.npz", "dir2.npz", "dir3.npz"],
            names=["dir1", "dir2", "dir3"],
            pairs=pairs,
            mean_abs_similarity=0.32,
        )

        _print_direction_comparison(result, threshold=0.1)

        captured = capsys.readouterr()
        assert "Total pairs: 3" in captured.out
        assert "Orthogonal" in captured.out
