"""Tests for introspect embedding CLI commands."""

import importlib
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from .conftest import requires_sklearn


def _mlx_available() -> bool:
    """Return True only when ``mlx.core`` can actually import on this host."""
    try:
        importlib.import_module("mlx.core")
    except Exception:
        return False
    return True


MLX_AVAILABLE = _mlx_available()
requires_mlx = pytest.mark.skipif(not MLX_AVAILABLE, reason="mlx not available")


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


@requires_mlx
@requires_sklearn
class TestIntrospectEmbedding:
    """Tests for introspect_embedding command."""

    @pytest.fixture
    def embedding_args(self):
        """Create arguments for embedding command."""
        return Namespace(
            model="test-model",
            layers=None,
            operation=None,
            output=None,
        )

    def test_embedding_basic(self, embedding_args, mock_ablation_study, capsys):
        """Test basic embedding analysis."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
            }
            mock_hooks.forward.return_value = None
            mock_hooks_cls.return_value = mock_hooks

            # Mock embedding access
            mock_study = mock_ablation_study.from_pretrained.return_value
            mock_model = MagicMock()
            mock_model.model.embed_tokens.return_value = mx.zeros((1, 5, 768))
            mock_study.adapter.model = mock_model

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.LinearRegression") as mock_lin:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_lin.return_value = mock_reg

                    with patch("sklearn.model_selection.cross_val_score") as mock_cv:
                        mock_cv.return_value = np.array([0.9, 0.95, 0.92])

                        introspect_embedding(embedding_args)

                        captured = capsys.readouterr()
                        assert "Loading model" in captured.out
                        assert "TASK TYPE DETECTION" in captured.out

    def test_embedding_specific_layers(self, embedding_args, mock_ablation_study, capsys):
        """Test embedding analysis at specific layers."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        embedding_args.layers = "0,4,8"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            mock_study = mock_ablation_study.from_pretrained.return_value
            mock_model = MagicMock()
            mock_model.model.embed_tokens.return_value = mx.zeros((1, 5, 768))
            mock_study.adapter.model = mock_model

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.LinearRegression") as mock_lin:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_lin.return_value = mock_reg

                    with patch("sklearn.model_selection.cross_val_score") as mock_cv:
                        mock_cv.return_value = np.array([0.9, 0.95, 0.92])

                        introspect_embedding(embedding_args)

                        captured = capsys.readouterr()
                        assert "Loading model" in captured.out

    def test_embedding_specific_operation(self, embedding_args, mock_ablation_study, capsys):
        """Test embedding analysis with specific operation."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        embedding_args.operation = "mult"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            mock_study = mock_ablation_study.from_pretrained.return_value
            mock_model = MagicMock()
            mock_model.model.embed_tokens.return_value = mx.zeros((1, 5, 768))
            mock_study.adapter.model = mock_model

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.LinearRegression") as mock_lin:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_lin.return_value = mock_reg

                    with patch("sklearn.model_selection.cross_val_score") as mock_cv:
                        mock_cv.return_value = np.array([0.9, 0.95, 0.92])

                        introspect_embedding(embedding_args)

                        captured = capsys.readouterr()
                        assert "Loading model" in captured.out

    def test_embedding_save_output(self, embedding_args, mock_ablation_study):
        """Test saving embedding analysis results."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            embedding_args.output = f.name

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            mock_study = mock_ablation_study.from_pretrained.return_value
            mock_model = MagicMock()
            mock_model.model.embed_tokens.return_value = mx.zeros((1, 5, 768))
            mock_study.adapter.model = mock_model

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.LinearRegression") as mock_lin:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_lin.return_value = mock_reg

                    with patch("sklearn.model_selection.cross_val_score") as mock_cv:
                        mock_cv.return_value = np.array([0.9, 0.95, 0.92])

                        introspect_embedding(embedding_args)

                        if Path(embedding_args.output).exists():
                            import json

                            with open(embedding_args.output) as f:
                                data = json.load(f)
                                assert "results" in data


@requires_mlx
@requires_sklearn
class TestIntrospectEarlyLayers:
    """Tests for introspect_early_layers command."""

    @pytest.fixture
    def early_layers_args(self):
        """Create arguments for early layers command."""
        return Namespace(
            model="test-model",
            layers=None,
            operations=None,
            digits=None,
            analyze_positions=False,
            output=None,
        )

    def test_early_layers_basic(self, early_layers_args, mock_ablation_study, capsys):
        """Test basic early layers analysis."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
            }
            mock_hooks.forward.return_value = None
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    captured = capsys.readouterr()
                    assert "Loading model" in captured.out
                    assert "REPRESENTATION SIMILARITY" in captured.out

    def test_early_layers_specific_layers(self, early_layers_args, mock_ablation_study, capsys):
        """Test early layers analysis at specific layers."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        early_layers_args.layers = "0,2,4"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    captured = capsys.readouterr()
                    assert "Loading model" in captured.out
                    assert "0, 2, 4" in captured.out

    def test_early_layers_specific_operations(self, early_layers_args, mock_ablation_study, capsys):
        """Test early layers analysis with specific operations."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        early_layers_args.operations = "*,+"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    captured = capsys.readouterr()
                    assert "Loading model" in captured.out
                    assert "*, +" in captured.out or "Operations" in captured.out

    def test_early_layers_custom_digits(self, early_layers_args, mock_ablation_study, capsys):
        """Test early layers analysis with custom digit range."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        early_layers_args.digits = "2-5"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    captured = capsys.readouterr()
                    assert "Loading model" in captured.out
                    assert "2-5" in captured.out or "Digit" in captured.out

    def test_early_layers_with_position_analysis(
        self, early_layers_args, mock_ablation_study, capsys
    ):
        """Test early layers analysis with position analysis."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        early_layers_args.analyze_positions = True

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 5, 768)),
                1: mx.zeros((1, 5, 768)),
                2: mx.zeros((1, 5, 768)),
                4: mx.zeros((1, 5, 768)),
                8: mx.zeros((1, 5, 768)),
            }
            mock_hooks.forward.return_value = None
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    captured = capsys.readouterr()
                    assert "Loading model" in captured.out
                    assert "POSITION-WISE ANALYSIS" in captured.out

    def test_early_layers_save_output(self, early_layers_args, mock_ablation_study):
        """Test saving early layers analysis results."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            early_layers_args.output = f.name

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    if Path(early_layers_args.output).exists():
                        import json

                        with open(early_layers_args.output) as f:
                            data = json.load(f)
                            assert "probe_results" in data


@requires_mlx
@requires_sklearn
class TestIntrospectEmbeddingEdgeCases:
    """Additional tests for edge cases and error handling."""

    @pytest.fixture
    def embedding_args(self):
        """Create arguments for embedding command."""
        return Namespace(
            model="test-model",
            layers=None,
            operation=None,
            output=None,
        )

    @pytest.fixture
    def early_layers_args(self):
        """Create arguments for early layers command."""
        return Namespace(
            model="test-model",
            layers=None,
            operations=None,
            digits=None,
            analyze_positions=False,
            output=None,
        )

    def test_embedding_alternative_embed_access(self, embedding_args, mock_ablation_study, capsys):
        """Test alternative embedding layer access path (model.embed_tokens)."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            # Mock model without nested model.embed_tokens
            mock_study = mock_ablation_study.from_pretrained.return_value
            mock_model = MagicMock()
            # Remove model.model.embed_tokens, only have embed_tokens at top level
            delattr(mock_model, "model")
            mock_model.embed_tokens.return_value = mx.zeros((1, 5, 768))
            mock_study.adapter.model = mock_model

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.LinearRegression") as mock_lin:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_lin.return_value = mock_reg

                    with patch("sklearn.model_selection.cross_val_score") as mock_cv:
                        mock_cv.return_value = np.array([0.9, 0.95, 0.92])

                        introspect_embedding(embedding_args)

                        captured = capsys.readouterr()
                        assert "Loading model" in captured.out

    def test_embedding_no_embed_layer_raises_error(self, embedding_args, mock_ablation_study):
        """Test error when embedding layer cannot be found."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            # Mock model without any embed_tokens
            mock_study = mock_ablation_study.from_pretrained.return_value
            mock_model = MagicMock(spec=[])  # Empty spec, no attributes
            mock_study.adapter.model = mock_model

            with pytest.raises(AttributeError, match="Cannot find embedding layer"):
                introspect_embedding(embedding_args)

    def test_embedding_cross_val_score_value_error(
        self, embedding_args, mock_ablation_study, capsys
    ):
        """Test fallback when cross_val_score raises ValueError."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            mock_study = mock_ablation_study.from_pretrained.return_value
            mock_model = MagicMock()
            mock_model.model.embed_tokens.return_value = mx.zeros((1, 5, 768))
            mock_study.adapter.model = mock_model

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.75
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.LinearRegression") as mock_lin:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_lin.return_value = mock_reg

                    with patch("sklearn.model_selection.cross_val_score") as mock_cv:
                        # Raise ValueError to trigger fallback
                        mock_cv.side_effect = ValueError("Not enough samples")

                        introspect_embedding(embedding_args)

                        captured = capsys.readouterr()
                        assert "Loading model" in captured.out
                        # Should use probe.score fallback
                        assert mock_probe.fit.called

    def test_embedding_partial_task_encoding(self, embedding_args, mock_ablation_study, capsys):
        """Test interpretation output for partial task encoding."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            mock_study = mock_ablation_study.from_pretrained.return_value
            mock_model = MagicMock()
            mock_model.model.embed_tokens.return_value = mx.zeros((1, 5, 768))
            mock_study.adapter.model = mock_model

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                # Return low accuracy to trigger partial encoding message
                mock_probe.score.return_value = 0.65
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.LinearRegression") as mock_lin:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_lin.return_value = mock_reg

                    with patch("sklearn.model_selection.cross_val_score") as mock_cv:
                        mock_cv.return_value = np.array([0.6, 0.65, 0.7])

                        introspect_embedding(embedding_args)

                        captured = capsys.readouterr()
                        assert "Task type partially encoded" in captured.out
                        assert "May need more layer computation" in captured.out

    def test_embedding_operations_add_and_mult(self, embedding_args, mock_ablation_study, capsys):
        """Test with both add and mult operations."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        embedding_args.operation = "add"  # Will also include mult in line 45

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            mock_study = mock_ablation_study.from_pretrained.return_value
            mock_model = MagicMock()
            mock_model.model.embed_tokens.return_value = mx.zeros((1, 5, 768))
            mock_study.adapter.model = mock_model

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.LinearRegression") as mock_lin:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 10
                    mock_lin.return_value = mock_reg

                    with patch("sklearn.model_selection.cross_val_score") as mock_cv:
                        mock_cv.return_value = np.array([0.9, 0.95, 0.92])

                        introspect_embedding(embedding_args)

                        captured = capsys.readouterr()
                        assert "Loading model" in captured.out

    def test_early_layers_single_operation(self, early_layers_args, mock_ablation_study, capsys):
        """Test early layers with only one operation (triggers op_acc=1.0 branch)."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        early_layers_args.operations = "*"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    captured = capsys.readouterr()
                    # When only one operation, op_acc should be set to 1.0
                    assert "Loading model" in captured.out

    def test_early_layers_fewer_than_two_operations_fallback(
        self, early_layers_args, mock_ablation_study, capsys
    ):
        """Test fallback sample expressions when fewer than 2 operations."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        # Only one operation
        early_layers_args.operations = "*"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    captured = capsys.readouterr()
                    assert "REPRESENTATION SIMILARITY" in captured.out


@requires_sklearn
class TestTorchRuntimeEmbeddingPaths:
    """Backend-aware regression tests for the torch runtime path."""

    @pytest.fixture
    def embedding_args(self):
        return Namespace(
            model="test-model",
            layers=None,
            operation=None,
            output=None,
            backend="torch",
            device="cpu",
        )

    @pytest.fixture
    def early_layers_args(self):
        return Namespace(
            model="test-model",
            layers=None,
            operations=None,
            digits=None,
            analyze_positions=False,
            output=None,
            backend="torch",
            device="cpu",
        )

    def test_embedding_uses_pipeline_for_torch_backend(
        self, embedding_args, mock_ablation_study, capsys
    ):
        """Test embedding switches to UnifiedPipeline when backend=torch."""
        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        mock_ablation_study.from_pretrained.side_effect = AssertionError(
            "AblationStudy.from_pretrained should not run for torch backend"
        )

        embedding_dim = 8
        fake_embedding = MagicMock(
            side_effect=lambda ids: np.zeros((1, ids.shape[1], embedding_dim), dtype=np.float32)
        )
        fake_model = MagicMock()
        fake_model.get_input_embeddings.return_value = fake_embedding
        fake_pipeline = SimpleNamespace(
            model=fake_model,
            tokenizer=MagicMock(),
            config=SimpleNamespace(num_hidden_layers=12),
            runtime=SimpleNamespace(backend="cuda", _device="cpu"),
        )

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.embedding._load_pipeline",
                return_value=fake_pipeline,
            ) as mock_load,
            patch("chuk_lazarus.cli.commands.introspect.embedding._to_backend_input") as mock_input,
            patch(
                "chuk_lazarus.cli.commands.introspect.embedding._to_numpy",
                side_effect=lambda tensor, **_: np.asarray(tensor),
            ),
            patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls,
            patch("sklearn.linear_model.LogisticRegression") as mock_lr,
            patch("sklearn.linear_model.LinearRegression") as mock_lin,
            patch("sklearn.model_selection.cross_val_score") as mock_cv,
        ):
            mock_input.return_value = np.ones((1, 3), dtype=np.int64)

            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: np.zeros((1, 1, embedding_dim), dtype=np.float32),
                1: np.zeros((1, 1, embedding_dim), dtype=np.float32),
                2: np.zeros((1, 1, embedding_dim), dtype=np.float32),
            }
            mock_hooks_cls.return_value = mock_hooks

            mock_probe = MagicMock()
            mock_probe.fit.return_value = mock_probe
            mock_probe.score.return_value = 0.95
            mock_lr.return_value = mock_probe

            mock_reg = MagicMock()
            mock_reg.fit.return_value = mock_reg
            mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
            mock_lin.return_value = mock_reg

            mock_cv.return_value = np.array([0.9, 0.95, 0.92])

            introspect_embedding(embedding_args)

            captured = capsys.readouterr()
            assert "Loading model" in captured.out
            mock_load.assert_called_once_with("test-model", backend="torch", device="cpu")

    def test_early_layers_uses_pipeline_for_torch_backend(
        self, early_layers_args, mock_ablation_study, capsys
    ):
        """Test early-layers switches to UnifiedPipeline when backend=torch."""
        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        mock_ablation_study.from_pretrained.side_effect = AssertionError(
            "AblationStudy.from_pretrained should not run for torch backend"
        )

        embedding_dim = 8
        fake_pipeline = SimpleNamespace(
            model=MagicMock(),
            tokenizer=MagicMock(
                encode=MagicMock(return_value=[1, 2, 3]),
                decode=MagicMock(side_effect=lambda ids: "tok"),
            ),
            config=SimpleNamespace(num_hidden_layers=12),
            runtime=SimpleNamespace(backend="cuda", _device="cpu"),
        )

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.embedding._load_pipeline",
                return_value=fake_pipeline,
            ) as mock_load,
            patch("chuk_lazarus.cli.commands.introspect.embedding._to_backend_input") as mock_input,
            patch(
                "chuk_lazarus.cli.commands.introspect.embedding._to_numpy",
                side_effect=lambda tensor, **_: np.asarray(tensor),
            ),
            patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls,
            patch("sklearn.linear_model.LogisticRegression") as mock_lr,
            patch("sklearn.linear_model.Ridge") as mock_ridge,
        ):
            mock_input.return_value = np.ones((1, 3), dtype=np.int64)

            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: np.zeros((1, 1, embedding_dim), dtype=np.float32),
                1: np.zeros((1, 1, embedding_dim), dtype=np.float32),
                2: np.zeros((1, 1, embedding_dim), dtype=np.float32),
                4: np.zeros((1, 1, embedding_dim), dtype=np.float32),
                8: np.zeros((1, 1, embedding_dim), dtype=np.float32),
            }
            mock_hooks_cls.return_value = mock_hooks

            mock_probe = MagicMock()
            mock_probe.fit.return_value = mock_probe
            mock_probe.score.return_value = 0.95
            mock_lr.return_value = mock_probe

            mock_reg = MagicMock()
            mock_reg.fit.return_value = mock_reg
            mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
            mock_ridge.return_value = mock_reg

            introspect_early_layers(early_layers_args)

            captured = capsys.readouterr()
            assert "Loading model" in captured.out
            mock_load.assert_called_once_with("test-model", backend="torch", device="cpu")
