"""Tests for ablation loader module."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from chuk_lazarus.inference import UnifiedPipelineConfig
from chuk_lazarus.introspection.ablation.loader import load_model_for_ablation


class TestLoadModelForAblation:
    """Tests for load_model_for_ablation function."""

    @patch("chuk_lazarus.inference.UnifiedPipeline.from_pretrained")
    def test_load_model_success(self, mock_from_pretrained):
        """Test successful model loading."""
        pipeline = SimpleNamespace(
            model=Mock(),
            tokenizer=Mock(),
            config=Mock(),
            runtime=Mock(),
        )
        mock_from_pretrained.return_value = pipeline

        adapter = load_model_for_ablation("test-model")

        assert adapter.model is pipeline.model
        assert adapter.tokenizer is pipeline.tokenizer
        assert adapter.config is pipeline.config
        assert adapter.runtime is pipeline.runtime
        call = mock_from_pretrained.call_args
        assert call.args == ("test-model",)
        assert isinstance(call.kwargs["pipeline_config"], UnifiedPipelineConfig)
        assert call.kwargs["pipeline_config"].backend_name is None
        assert call.kwargs["pipeline_config"].device is None
        assert call.kwargs["verbose"] is False

    @patch("chuk_lazarus.inference.UnifiedPipeline.from_pretrained")
    def test_load_threads_backend_and_device(self, mock_from_pretrained):
        """Test backend/device overrides flow into UnifiedPipeline."""
        pipeline = SimpleNamespace(
            model=Mock(),
            tokenizer=Mock(),
            config=Mock(),
            runtime=Mock(),
        )
        mock_from_pretrained.return_value = pipeline

        load_model_for_ablation("test-model", backend="torch", device="cuda:1")

        pipeline_config = mock_from_pretrained.call_args.kwargs["pipeline_config"]
        assert pipeline_config.backend_name == "torch"
        assert pipeline_config.device == "cuda:1"
