"""Tests for introspect ablation CLI commands."""

import asyncio
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import numpy as np

import pytest


@pytest.fixture(autouse=True)
def _stub_backend_dispatch_module():
    """Make the introspection mock package expose the backend-dispatch submodule."""
    intro = sys.modules.get("chuk_lazarus.introspection")
    original_dispatch = sys.modules.get("chuk_lazarus.introspection._backend_dispatch")
    original_path = intro.__dict__.get("__path__") if intro is not None else None

    backend_dispatch = ModuleType("chuk_lazarus.introspection._backend_dispatch")
    backend_dispatch.from_backend_tensor = lambda tensor, backend=None: tensor
    backend_dispatch.to_backend_tensor = lambda tensor, backend=None: tensor

    if intro is not None:
        intro.__dict__["__path__"] = []
        intro._backend_dispatch = backend_dispatch

    sys.modules["chuk_lazarus.introspection._backend_dispatch"] = backend_dispatch

    yield

    if original_dispatch is None:
        sys.modules.pop("chuk_lazarus.introspection._backend_dispatch", None)
    else:
        sys.modules["chuk_lazarus.introspection._backend_dispatch"] = original_dispatch

    if intro is not None:
        if original_path is None:
            intro.__dict__.pop("__path__", None)
        else:
            intro.__dict__["__path__"] = original_path
        intro._backend_dispatch = original_dispatch


class TestIntrospectAblate:
    """Tests for introspect_ablate command."""

    @pytest.fixture
    def ablate_args(self):
        """Create arguments for ablate command."""
        return Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21,22",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

    def test_ablate_no_prompt_error(self, capsys):
        """Test error when no prompt provided."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        args = Namespace(
            model="test-model",
            prompt=None,
            prompts=None,
            criterion=None,
            layers="20,21,22",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        with pytest.raises(ValueError, match="--prompt"):
            introspect_ablate(args)

    def test_ablate_prompt_without_criterion_error(self, capsys):
        """Test error when prompt without criterion."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion=None,
            layers="20,21,22",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        with pytest.raises(ValueError, match="--criterion"):
            introspect_ablate(args)


class TestAblationConfig:
    """Tests for AblationConfig model."""

    def test_from_args(self):
        """Test creating config from args."""
        from chuk_lazarus.cli.commands.introspect._types import AblationConfig

        args = Namespace(
            model="test-model",
            backend="torch",
            device="cuda:1",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21,22",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        config = AblationConfig.from_args(args)

        assert config.model == "test-model"
        assert config.backend == "torch"
        assert config.device == "cuda:1"
        assert config.prompt == "2+2="
        assert config.criterion == "4"
        assert config.component == "mlp"

    def test_from_args_with_prompts(self):
        """Test creating config with multi-prompt format."""
        from chuk_lazarus.cli.commands.introspect._types import AblationConfig

        args = Namespace(
            model="test-model",
            prompt=None,
            prompts="2+2=:4|3+3=:6",
            criterion=None,
            layers="20,21,22",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        config = AblationConfig.from_args(args)

        assert config.prompts == "2+2=:4|3+3=:6"

    def test_from_args_multi_mode(self):
        """Test creating config with multi mode."""
        from chuk_lazarus.cli.commands.introspect._types import AblationConfig

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21,22",
            component="mlp",
            multi=True,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        config = AblationConfig.from_args(args)

        assert config.multi is True


class TestParsePrompts:
    """Tests for prompt parsing utilities."""

    def test_parse_prompts(self):
        """Test parsing prompts from pipe-separated string."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_prompts

        prompts = parse_prompts("2+2=|3+3=|4+4=")
        assert len(prompts) == 3
        assert prompts[0] == "2+2="

    def test_parse_layers_range(self):
        """Test parsing layer range."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_layers

        layers = parse_layers("20-23")
        assert layers == [20, 21, 22, 23]

    def test_parse_layers_comma_separated(self):
        """Test parsing comma-separated layers."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_layers

        layers = parse_layers("20,21,22")
        assert layers == [20, 21, 22]

    def test_parse_layers_none(self):
        """Test parsing None returns None."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_layers

        layers = parse_layers(None)
        assert layers is None


class TestAblationResult:
    """Tests for AblationResult type."""

    def test_result_creation(self):
        """Test creating ablation result."""
        from chuk_lazarus.cli.commands.introspect._types import AblationResult

        result = AblationResult(
            prompt="2+2=",
            expected="4",
            ablation="L20 MLP",
            output="4",
            correct=True,
        )

        assert result.prompt == "2+2="
        assert result.correct is True

    def test_result_to_display(self):
        """Test result display format."""
        from chuk_lazarus.cli.commands.introspect._types import AblationResult

        result = AblationResult(
            prompt="2+2=",
            expected="4",
            ablation="L20 MLP",
            output="4",
            correct=True,
        )

        display = result.to_display()
        assert "PASS" in display
        assert "L20 MLP" in display


class TestPrintMultiPromptResults:
    """Tests for _print_multi_prompt_results helper function."""

    def test_print_results_basic(self, capsys):
        """Test basic multi-prompt results printing."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_prompt_results,
        )

        # Create mock results
        mock_single_result = MagicMock()
        mock_single_result.prompt = "2+2="
        mock_single_result.output = "4"
        mock_single_result.passes_criterion = True

        mock_ablation_result = MagicMock()
        mock_ablation_result.ablation_name = "L20 MLP"
        mock_ablation_result.results = [mock_single_result]

        prompt_pairs = [("2+2=", "4")]

        _print_multi_prompt_results([mock_ablation_result], prompt_pairs, verbose=False)

        captured = capsys.readouterr()
        assert "MULTI-PROMPT ABLATION TEST" in captured.out
        assert "L20 MLP" in captured.out

    def test_print_results_verbose(self, capsys):
        """Test verbose multi-prompt results printing."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_prompt_results,
        )

        mock_single_result = MagicMock()
        mock_single_result.prompt = "2+2="
        mock_single_result.output = "4"
        mock_single_result.passes_criterion = True

        mock_ablation_result = MagicMock()
        mock_ablation_result.ablation_name = "L20 MLP"
        mock_ablation_result.results = [mock_single_result]

        prompt_pairs = [("2+2=", "4")]

        _print_multi_prompt_results([mock_ablation_result], prompt_pairs, verbose=True)

        captured = capsys.readouterr()
        assert "FULL OUTPUTS" in captured.out
        assert "PASS" in captured.out

    def test_print_results_failing(self, capsys):
        """Test multi-prompt results with failures."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_prompt_results,
        )

        mock_single_result = MagicMock()
        mock_single_result.prompt = "2+2="
        mock_single_result.output = "5"
        mock_single_result.passes_criterion = False

        mock_ablation_result = MagicMock()
        mock_ablation_result.ablation_name = "L20 MLP"
        mock_ablation_result.results = [mock_single_result]

        prompt_pairs = [("2+2=", "4")]

        _print_multi_prompt_results([mock_ablation_result], prompt_pairs, verbose=True)

        captured = capsys.readouterr()
        assert "FAIL" in captured.out


class TestPrintMultiAblationResults:
    """Tests for _print_multi_ablation_results helper function."""

    def test_causal_result(self, capsys):
        """Test printing causal ablation result."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_ablation_results,
        )

        baseline = MagicMock()
        baseline.passes_criterion = True
        baseline.output = "4"

        ablated = MagicMock()
        ablated.passes_criterion = False
        ablated.output = "wrong"

        _print_multi_ablation_results("2+2=", "4", [20, 21], baseline, ablated)

        captured = capsys.readouterr()
        assert "CAUSAL" in captured.out
        assert "breaks the criterion" in captured.out

    def test_inverse_causal_result(self, capsys):
        """Test printing inverse causal ablation result."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_ablation_results,
        )

        baseline = MagicMock()
        baseline.passes_criterion = False
        baseline.output = "wrong"

        ablated = MagicMock()
        ablated.passes_criterion = True
        ablated.output = "4"

        _print_multi_ablation_results("2+2=", "4", [20, 21], baseline, ablated)

        captured = capsys.readouterr()
        assert "INVERSE CAUSAL" in captured.out
        assert "enables the criterion" in captured.out

    def test_not_causal_result(self, capsys):
        """Test printing not causal ablation result."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_ablation_results,
        )

        baseline = MagicMock()
        baseline.passes_criterion = True
        baseline.output = "4"

        ablated = MagicMock()
        ablated.passes_criterion = True
        ablated.output = "4"

        _print_multi_ablation_results("2+2=", "4", [20, 21], baseline, ablated)

        captured = capsys.readouterr()
        assert "NOT CAUSAL" in captured.out
        assert "doesn't affect" in captured.out

    def test_baseline_fails_result(self, capsys):
        """Test printing baseline fails result."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_ablation_results,
        )

        baseline = MagicMock()
        baseline.passes_criterion = False
        baseline.output = "wrong"

        ablated = MagicMock()
        ablated.passes_criterion = False
        ablated.output = "also wrong"

        _print_multi_ablation_results("2+2=", "4", [20, 21], baseline, ablated)

        captured = capsys.readouterr()
        assert "BASELINE FAILS" in captured.out


class TestAsyncIntrospectAblate:
    """Tests for _async_introspect_ablate function."""

    @pytest.mark.asyncio
    async def test_multi_prompt_mode(self, capsys):
        """Test ablation with multiple prompts."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt=None,
            prompts="2+2=:4|3+3=:6",
            criterion=None,
            layers="20,21",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24

        mock_result = MagicMock()
        mock_result.ablation_name = "Baseline"
        mock_single = MagicMock()
        mock_single.prompt = "2+2="
        mock_single.output = "4"
        mock_single.passes_criterion = True
        mock_result.results = [mock_single]

        # Need to patch at the introspection module level, not CLI level
        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.parse_prompt_pairs",
                return_value=[("2+2=", "4"), ("3+3=", "6")],
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_multi_prompt_ablation",
                new_callable=AsyncMock,
                return_value=[mock_result],
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    @pytest.mark.asyncio
    async def test_multi_layer_ablation_mode(self, capsys):
        """Test multi-layer ablation mode."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21",
            component="mlp",
            multi=True,  # Multi-layer mode
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24

        baseline = MagicMock()
        baseline.passes_criterion = True
        baseline.output = "4"

        ablated = MagicMock()
        ablated.passes_criterion = False
        ablated.output = "wrong"

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_multi_ablation",
                new_callable=AsyncMock,
                return_value=(baseline, ablated),
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Ablating layers together" in captured.out
        assert "CAUSAL" in captured.out

    @pytest.mark.asyncio
    async def test_sweep_mode(self, capsys):
        """Test ablation sweep mode."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21",
            component="mlp",
            multi=False,  # Sweep mode
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24

        mock_result = MagicMock()

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_ablation_sweep",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Sweeping layers individually" in captured.out

    @pytest.mark.asyncio
    async def test_sweep_mode_with_output(self, tmp_path, capsys):
        """Test ablation sweep mode with output file."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        output_file = str(tmp_path / "results.json")

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=output_file,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24

        mock_result = MagicMock()

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_ablation_sweep",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            await _async_introspect_ablate(args)

        capsys.readouterr()  # Clear output
        mock_study.save_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_attention_component(self, capsys):
        """Test ablation with attention component."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21",
            component="attention",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24
        mock_result = MagicMock()

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_ablation_sweep",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Component: attention" in captured.out

    @pytest.mark.asyncio
    async def test_both_component(self, capsys):
        """Test ablation with both components."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21",
            component="both",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24
        mock_result = MagicMock()

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_ablation_sweep",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Component: both" in captured.out

    @pytest.mark.asyncio
    async def test_raw_mode(self, capsys):
        """Test ablation with raw mode."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21",
            component="mlp",
            multi=False,
            raw=True,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24
        mock_result = MagicMock()

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_ablation_sweep",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Mode: RAW" in captured.out

    @pytest.mark.asyncio
    async def test_no_layers_uses_all(self, capsys):
        """Test ablation without explicit layers uses all layers."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers=None,  # No layers specified
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24
        mock_result = MagicMock()

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_ablation_sweep",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        # Should use all 24 layers
        assert "Sweeping layers individually" in captured.out

    @pytest.mark.asyncio
    async def test_prompts_without_expected_uses_criterion(self, capsys):
        """Test prompts without expected values use criterion."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt=None,
            prompts="2+2=|3+3=",  # No expected values
            criterion="4",  # Use criterion for all
            layers="20,21",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24

        mock_result = MagicMock()
        mock_result.ablation_name = "Baseline"
        mock_single = MagicMock()
        mock_single.prompt = "2+2="
        mock_single.output = "4"
        mock_single.passes_criterion = True
        mock_result.results = [mock_single]

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.parse_prompt_pairs",
                return_value=[("2+2=", ""), ("3+3=", "")],  # No expected values
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_multi_prompt_ablation",
                new_callable=AsyncMock,
                return_value=[mock_result],
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    @pytest.mark.asyncio
    async def test_prompts_without_expected_or_criterion_error(self):
        """Test prompts without expected values or criterion raises error."""
        from unittest.mock import MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt=None,
            prompts="2+2=|3+3=",  # No expected values
            criterion=None,  # No criterion either
            layers="20,21",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.parse_prompt_pairs",
                return_value=[("2+2=", ""), ("3+3=", "")],
            ),
            pytest.raises(ValueError, match="has no expected value"),
        ):
            await _async_introspect_ablate(args)

    @pytest.mark.asyncio
    async def test_threads_backend_device_and_reuses_loaded_study(self):
        """Test CLI threads backend/device and reuses the initially loaded study."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            backend="torch",
            device="cuda:1",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24
        mock_result = MagicMock()

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ) as mock_from_pretrained,
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_ablation_sweep",
                new_callable=AsyncMock,
                return_value=mock_result,
            ) as mock_run_ablation_sweep,
        ):
            await _async_introspect_ablate(args)

        mock_from_pretrained.assert_called_once_with(
            "test-model",
            backend="torch",
            device="cuda:1",
        )
        assert mock_run_ablation_sweep.await_args.kwargs["study"] is mock_study


class TestIntrospectWeightDiff:
    """Tests for introspect_weight_diff command."""

    def test_calls_asyncio_run(self):
        """Test that introspect_weight_diff calls asyncio.run."""
        from unittest.mock import patch

        from chuk_lazarus.cli.commands.introspect.ablation import introspect_weight_diff

        args = Namespace(base="test/base", finetuned="test/ft", output=None)

        def close_coro(coro):
            coro.close()

        with patch(
            "chuk_lazarus.cli.commands.introspect.ablation.asyncio.run",
            side_effect=close_coro,
        ) as mock_run:
            introspect_weight_diff(args)
            mock_run.assert_called_once()

    @pytest.mark.parametrize(
        ("backend_name", "device"),
        [
            ("torch", "cuda:1"),
            ("mlx", None),
        ],
    )
    def test_uses_backend_safe_loader(
        self,
        backend_name,
        device,
        capsys,
    ):
        """Test weight diff loads both models through the runtime-safe ablation seam."""
        from unittest.mock import MagicMock, call, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_weight_diff,
        )

        args = Namespace(
            base="test/base",
            finetuned="test/ft",
            backend=backend_name,
            device=device,
            output=None,
        )
        resolved_backend = SimpleNamespace(name=backend_name, device=device)

        def make_study(mlp_weights, attn_weights):
            adapter = MagicMock()
            adapter.num_layers = len(mlp_weights)
            adapter.get_mlp_down_weight.side_effect = lambda layer: mlp_weights[layer]
            adapter.get_attn_o_weight.side_effect = lambda layer: attn_weights[layer]
            study = MagicMock()
            study.adapter = adapter
            return study

        base_study = make_study(
            [
                np.array([1.0, 0.0]),
                np.array([0.5, 0.5]),
            ],
            [
                np.array([0.0, 1.0]),
                np.array([1.0, 1.0]),
            ],
        )
        ft_study = make_study(
            [
                np.array([1.2, 0.0]),
                np.array([0.5, 0.8]),
            ],
            [
                np.array([0.0, 1.3]),
                np.array([0.7, 1.0]),
            ],
        )

        with (
            patch(
                "chuk_lazarus.models_v2.core.backend.get_backend",
                return_value=resolved_backend,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                side_effect=[base_study, ft_study],
            ) as mock_from_pretrained,
            patch(
                "chuk_lazarus.introspection._backend_dispatch.from_backend_tensor",
                side_effect=lambda tensor, backend=None: tensor,
            ) as mock_from_backend_tensor,
        ):
            asyncio.run(_async_introspect_weight_diff(args))

        expected_device = device if backend_name == "torch" else None
        mock_from_pretrained.assert_has_calls(
            [
                call(
                    "test/base",
                    backend=backend_name,
                    device=expected_device,
                ),
                call(
                    "test/ft",
                    backend=backend_name,
                    device=expected_device,
                ),
            ]
        )
        assert mock_from_backend_tensor.call_count == 8

        captured = capsys.readouterr()
        assert "Comparing 2 layers" in captured.out
        assert "mlp_down" in captured.out
        assert "attn_o" in captured.out


class TestIntrospectActivationDiff:
    """Tests for introspect_activation_diff command."""

    def test_calls_asyncio_run(self):
        """Test that introspect_activation_diff calls asyncio.run."""
        from unittest.mock import patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            introspect_activation_diff,
        )

        args = Namespace(base="test/base", finetuned="test/ft", prompts="test", output=None)

        def close_coro(coro):
            coro.close()

        with patch(
            "chuk_lazarus.cli.commands.introspect.ablation.asyncio.run",
            side_effect=close_coro,
        ) as mock_run:
            introspect_activation_diff(args)
            mock_run.assert_called_once()

    def test_threads_backend_and_tensor_conversion_through_hooks(self, capsys):
        """Test activation diff uses backend-aware loading and tensor conversion."""
        from unittest.mock import MagicMock, call, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_activation_diff,
        )

        args = Namespace(
            base="test/base",
            finetuned="test/ft",
            prompts="alpha,beta",
            backend="torch",
            device="cuda:1",
            output=None,
        )
        resolved_backend = SimpleNamespace(name="torch", device="cuda:1")
        backend_input = object()
        hook_instances = []

        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda prompt, return_tensors="np": np.array(
            [[len(prompt), 1]],
            dtype=np.int64,
        )

        base_model = SimpleNamespace(
            hidden_by_layer={
                0: np.array([[[1.0, 0.0]]], dtype=np.float32),
                1: np.array([[[1.0, 1.0]]], dtype=np.float32),
            }
        )
        ft_model = SimpleNamespace(
            hidden_by_layer={
                0: np.array([[[0.0, 1.0]]], dtype=np.float32),
                1: np.array([[[1.0, 1.0]]], dtype=np.float32),
            }
        )

        def make_study(model, config_name):
            adapter = MagicMock()
            adapter.model = model
            adapter.config = SimpleNamespace(name=config_name)
            adapter.num_layers = 2
            adapter.tokenizer = tokenizer
            study = MagicMock()
            study.adapter = adapter
            return study

        class FakeHooks:
            def __init__(self, model, model_config=None):
                self.model = model
                self.model_config = model_config
                self.state = SimpleNamespace(hidden_states={})
                hook_instances.append(self)

            def configure(self, config):
                self.config = config
                return self

            def forward(self, input_ids):
                self.forward_input_ids = input_ids
                self.state.hidden_states = dict(self.model.hidden_by_layer)
                return None

        base_study = make_study(base_model, "base-config")
        ft_study = make_study(ft_model, "ft-config")

        with (
            patch(
                "chuk_lazarus.models_v2.core.backend.get_backend",
                return_value=resolved_backend,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                side_effect=[base_study, ft_study],
            ) as mock_from_pretrained,
            patch(
                "chuk_lazarus.introspection.ModelHooks",
                FakeHooks,
            ),
            patch(
                "chuk_lazarus.introspection._backend_dispatch.to_backend_tensor",
                return_value=backend_input,
            ) as mock_to_backend_tensor,
            patch(
                "chuk_lazarus.introspection._backend_dispatch.from_backend_tensor",
                side_effect=lambda tensor, backend=None: tensor,
            ),
        ):
            asyncio.run(_async_introspect_activation_diff(args))

        mock_from_pretrained.assert_has_calls(
            [
                call(
                    "test/base",
                    backend="torch",
                    device="cuda:1",
                ),
                call(
                    "test/ft",
                    backend="torch",
                    device="cuda:1",
                ),
            ]
        )
        assert mock_to_backend_tensor.call_count == 2
        assert all(
            hook.forward_input_ids is backend_input
            for hook in hook_instances
        )
        assert tokenizer.encode.call_count == 2

        captured = capsys.readouterr()
        assert "Testing 2 prompts" in captured.out
        assert "Avg Cos Sim" in captured.out
        assert "1.0000" in captured.out
