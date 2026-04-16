"""Tests for ExpertRouter async router manipulation."""

import types
from unittest.mock import MagicMock, patch

import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from chuk_lazarus.inference.backends.torch_runtime import TorchInferenceRuntime
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.expert_router import ExpertRouter
from chuk_lazarus.introspection.moe.models import (
    CoactivationAnalysis,
    ExpertChatResult,
    ExpertComparisonResult,
    GenerationStats,
    LayerRouterWeights,
    MoEModelInfo,
    TopKVariationResult,
)


TORCH_DEVICES = ["cpu"] + (["cuda:0"] if torch is not None and torch.cuda.is_available() else [])


if torch is not None:

    class TinyTorchTokenizer:
        """Minimal tokenizer for exercising the torch ExpertRouter path."""

        pad_token_id = 0
        eos_token_id = 0

        def __init__(self) -> None:
            self._vocab = {
                "<pad>": 0,
                "alpha": 1,
                "beta": 2,
                "expert0": 3,
                "expert1": 4,
            }
            self._inv_vocab = {idx: token for token, idx in self._vocab.items()}

        def encode(self, text: str, return_tensors: str | None = None):
            token_ids = [self._vocab[text]]
            if return_tensors == "pt":
                return torch.tensor([token_ids], dtype=torch.long)
            return token_ids

        def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            elif hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()

            tokens = [
                self._inv_vocab[token_id]
                for token_id in token_ids
                if not skip_special_tokens or token_id not in {self.pad_token_id, self.eos_token_id}
            ]
            return " ".join(tokens)

        def __call__(self, text: str, return_tensors: str = "pt"):
            input_ids = self.encode(text, return_tensors=return_tensors)
            attention_mask = torch.ones_like(input_ids)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }


    class TinyTorchRouter(nn.Module):
        """Simple router that selects between two experts from the hidden state."""

        def __init__(self) -> None:
            super().__init__()
            self.num_experts = 2
            self.num_experts_per_tok = 1
            self.weight = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            self.bias = nn.Parameter(torch.zeros(2))

        def forward(self, hidden_states):
            logits = hidden_states @ self.weight.T
            logits = logits + self.bias
            top_logits, top_indices = logits.topk(k=self.num_experts_per_tok, dim=-1)
            top_weights = top_logits.softmax(dim=-1)
            return top_weights, top_indices


    class TinyTorchMlp(nn.Module):
        """Two-expert MLP whose output is determined by router choice."""

        def __init__(self) -> None:
            super().__init__()
            self.router = TinyTorchRouter()
            self.num_experts_per_tok = self.router.num_experts_per_tok
            self.expert_values = nn.Parameter(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))

        def forward(self, hidden_states):
            expert_weights, expert_indices = self.router(hidden_states)
            selected_values = self.expert_values[expert_indices]
            return (selected_values * expert_weights.unsqueeze(-1)).sum(dim=-2)


    class TinyTorchLayer(nn.Module):
        """Single transformer-like layer that delegates to the MoE MLP."""

        def __init__(self) -> None:
            super().__init__()
            self.mlp = TinyTorchMlp()

        def forward(self, hidden_states, *args, **kwargs):
            return self.mlp(hidden_states)


    class TinyTorchMoEModel(nn.Module):
        """Minimal causal LM that routes through a single MoE layer."""

        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(5, 2)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([TinyTorchLayer()])
            self.lm_head = nn.Linear(2, 5, bias=False)
            self._init_weights()

        def _init_weights(self) -> None:
            with torch.no_grad():
                self.embed.weight.zero_()
                self.embed.weight[1] = torch.tensor([2.0, 0.0])
                self.embed.weight[2] = torch.tensor([0.0, 2.0])
                self.embed.weight[3] = torch.tensor([2.0, 0.0])
                self.embed.weight[4] = torch.tensor([0.0, 2.0])
                self.lm_head.weight.zero_()
                self.lm_head.weight[3] = torch.tensor([1.0, 0.0])
                self.lm_head.weight[4] = torch.tensor([0.0, 1.0])

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            hidden_states = self.embed(input_ids)
            for layer in self.model.layers:
                hidden_states = layer(hidden_states)
            logits = self.lm_head(hidden_states)
            return types.SimpleNamespace(logits=logits)

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


def build_torch_router(device: str) -> ExpertRouter:
    """Create a tiny torch-backed ExpertRouter for runtime-path testing."""
    if torch is None:
        pytest.skip("torch is not available")

    tokenizer = TinyTorchTokenizer()
    model = TinyTorchMoEModel().to(device)
    runtime = TorchInferenceRuntime(model, tokenizer, device=device)
    model_info = MoEModelInfo(
        moe_layers=(0,),
        num_experts=2,
        num_experts_per_tok=1,
        total_layers=1,
        architecture=MoEArchitecture.GENERIC,
        has_shared_expert=False,
    )
    return ExpertRouter(
        model,
        tokenizer,
        model_info,
        runtime=runtime,
        backend_name="torch",
    )


class TestExpertRouterInit:
    """Tests for ExpertRouter initialization."""

    def test_init_with_valid_moe_model(self, mock_mlx_model, mock_tokenizer, mock_moe_model_info):
        """Test initialization with valid MoE model."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)
        assert router.info == mock_moe_model_info
        assert router.tokenizer == mock_tokenizer

    def test_init_raises_for_non_moe_model(self, mock_mlx_model, mock_tokenizer):
        """Test initialization raises for non-MoE model."""
        non_moe_info = MoEModelInfo(
            moe_layers=(),  # Empty - no MoE layers
            num_experts=0,
            num_experts_per_tok=0,
            total_layers=8,
        )
        with pytest.raises(ValueError, match="no MoE layers"):
            ExpertRouter(mock_mlx_model, mock_tokenizer, non_moe_info)

    def test_info_property(self, mock_mlx_model, mock_tokenizer, mock_moe_model_info):
        """Test info property returns model info."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)
        assert router.info.num_experts == 32
        assert router.info.num_experts_per_tok == 4

    def test_tokenizer_property(self, mock_mlx_model, mock_tokenizer, mock_moe_model_info):
        """Test tokenizer property."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)
        assert router.tokenizer.vocab_size == 32000


class TestExpertRouterContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_mlx_model, mock_tokenizer, mock_moe_model_info):
        """Test async context manager enter and exit."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)
        async with router as r:
            assert r is router
        # Should not raise on exit


class TestExpertRouterMoETypeDetection:
    """Tests for MoE type detection."""

    def test_detect_gpt_oss_batched(self, mock_tokenizer, mock_moe_model_info):
        """Test detection of GPT-OSS batched style."""
        # Create a model with GPT-OSS batched structure
        mock_model = MagicMock()
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            layer.mlp.router = MagicMock()
            # GPT-OSS batched has experts.gate_up_proj
            layer.mlp.experts = MagicMock()
            layer.mlp.experts.gate_up_proj = MagicMock()
            layers.append(layer)
        mock_model.model.layers = layers

        router = ExpertRouter(mock_model, mock_tokenizer, mock_moe_model_info)
        assert router._moe_type == "gpt_oss_batched"

    def test_detect_standard(self, mock_tokenizer, mock_moe_model_info):
        """Test detection of standard MoE style."""
        mock_model = MagicMock()
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            layer.mlp.router = MagicMock()
            # Standard MoE has experts list but no gate_up_proj
            layer.mlp.experts = [MagicMock() for _ in range(8)]
            # Remove the attribute to make hasattr return False
            del layer.mlp.experts
            layer.mlp.experts = MagicMock(spec=[])  # Empty spec means no gate_up_proj
            layers.append(layer)
        mock_model.model.layers = layers

        router = ExpertRouter(mock_model, mock_tokenizer, mock_moe_model_info)
        assert router._moe_type == "standard"


class TestExpertRouterGeneration:
    """Tests for generation methods."""

    @pytest.mark.asyncio
    async def test_chat_with_expert_returns_result(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test chat_with_expert returns ExpertChatResult."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        # Mock the sync method
        mock_stats = GenerationStats(
            expert_idx=6,
            tokens_generated=10,
            layers_modified=8,
            moe_type="gpt_oss_batched",
        )
        router._generate_with_forced_expert_sync = MagicMock(return_value=("11303", mock_stats))

        result = await router.chat_with_expert("127 * 89 = ", expert_idx=6)

        assert isinstance(result, ExpertChatResult)
        assert result.prompt == "127 * 89 = "
        assert result.response == "11303"
        assert result.expert_idx == 6

    @pytest.mark.asyncio
    async def test_compare_experts_returns_comparison(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test compare_experts returns ExpertComparisonResult."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        # Mock the sync method
        mock_stats = GenerationStats(
            expert_idx=6,
            tokens_generated=10,
            layers_modified=8,
            moe_type="gpt_oss_batched",
        )
        router._generate_with_forced_expert_sync = MagicMock(return_value=("response", mock_stats))

        result = await router.compare_experts("Test", expert_indices=[6, 7, 20])

        assert isinstance(result, ExpertComparisonResult)
        assert result.prompt == "Test"
        assert len(result.expert_results) == 3

    @pytest.mark.asyncio
    async def test_generate_with_ablation_returns_tuple(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test generate_with_ablation returns response and stats."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        mock_stats = GenerationStats(
            expert_idx=-1,
            tokens_generated=15,
            layers_modified=8,
            moe_type="gpt_oss_batched",
        )
        router._generate_with_ablation_sync = MagicMock(return_value=("ablated output", mock_stats))

        text, stats = await router.generate_with_ablation("Test prompt", expert_indices=[6, 7])

        assert text == "ablated output"
        assert isinstance(stats, GenerationStats)
        assert stats.expert_idx == -1

    @pytest.mark.asyncio
    async def test_generate_with_topk_returns_result(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test generate_with_topk returns TopKVariationResult."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        router._generate_normal_sync = MagicMock(return_value="normal response")
        router._generate_with_topk_sync = MagicMock(return_value="topk response")

        result = await router.generate_with_topk("Test prompt", k=2)

        assert isinstance(result, TopKVariationResult)
        assert result.k_value == 2
        assert result.default_k == 4
        assert result.normal_response == "normal response"
        assert result.response == "topk response"


class TestExpertRouterAnalysis:
    """Tests for analysis methods."""

    @pytest.mark.asyncio
    async def test_capture_router_weights_returns_list(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test capture_router_weights returns list of LayerRouterWeights."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        # Mock the sync method
        from chuk_lazarus.introspection.moe.models import RouterWeightCapture

        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="Hello",
                        expert_indices=(6, 7, 20, 1),
                        weights=(0.4, 0.3, 0.2, 0.1),
                    ),
                ),
            )
        ]
        router._capture_router_weights_sync = MagicMock(return_value=mock_weights)

        result = await router.capture_router_weights("Hello world")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], LayerRouterWeights)

    @pytest.mark.asyncio
    async def test_analyze_coactivation_returns_analysis(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test analyze_coactivation returns CoactivationAnalysis."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        mock_analysis = CoactivationAnalysis(
            layer_idx=0,
            total_activations=100,
            generalist_experts=(6, 7),
        )
        router._analyze_coactivation_sync = MagicMock(return_value=mock_analysis)

        result = await router.analyze_coactivation(["Test 1", "Test 2"])

        assert isinstance(result, CoactivationAnalysis)
        assert result.layer_idx == 0
        assert result.total_activations == 100


class TestExpertRouterSampling:
    """Tests for token sampling."""

    @pytest.mark.skipif(mx is None, reason="MLX is not available on this host")
    def test_sample_token_greedy(self, mock_mlx_model, mock_tokenizer, mock_moe_model_info):
        """Test greedy sampling (temperature=0)."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        # Create logits with clear maximum
        logits = mx.array([[[0.1, 0.2, 0.9, 0.3]]])  # Token 2 has highest logit

        token = router._sample_token(logits, temperature=0.0)
        assert token == 2

    @pytest.mark.skipif(mx is None, reason="MLX is not available on this host")
    def test_sample_token_with_temperature(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test sampling with temperature > 0."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        logits = mx.array([[[0.1, 0.2, 0.9, 0.3]]])

        # With temperature, should still return a valid token index
        token = router._sample_token(logits, temperature=1.0)
        assert 0 <= token < 4


class TestExpertRouterExtractInfo:
    """Tests for MoE info extraction."""

    def test_extract_moe_info_gpt_oss(self):
        """Test extracting info from GPT-OSS style model."""
        mock_model = MagicMock()
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            layer.mlp.router = MagicMock()
            layer.mlp.router.num_experts = 32
            layer.mlp.router.num_experts_per_tok = 4
            # No shared_expert
            layer.mlp.shared_expert = None
            del layer.mlp.shared_expert
            layers.append(layer)
        mock_model.model.layers = layers

        info = ExpertRouter._extract_moe_info(mock_model)

        assert info.num_experts == 32
        assert info.num_experts_per_tok == 4
        assert info.total_layers == 8
        assert len(info.moe_layers) == 8
        assert info.architecture == MoEArchitecture.GPT_OSS

    def test_extract_moe_info_mixtral(self):
        """Test extracting info from Mixtral style model."""
        mock_model = MagicMock()
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            layer.mlp.router = MagicMock()
            layer.mlp.router.num_experts = 8
            layer.mlp.router.num_experts_per_tok = 2
            del layer.mlp.shared_expert
            layers.append(layer)
        mock_model.model.layers = layers

        info = ExpertRouter._extract_moe_info(mock_model)

        assert info.num_experts == 8
        assert info.num_experts_per_tok == 2
        assert info.architecture == MoEArchitecture.MIXTRAL

    def test_extract_moe_info_with_shared_expert(self):
        """Test extracting info from model with shared expert."""
        mock_model = MagicMock()
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            layer.mlp.router = MagicMock()
            layer.mlp.router.num_experts = 16
            layer.mlp.router.num_experts_per_tok = 2
            layer.mlp.shared_expert = MagicMock()  # Has shared expert
            layers.append(layer)
        mock_model.model.layers = layers

        info = ExpertRouter._extract_moe_info(mock_model)

        assert info.has_shared_expert is True
        assert info.architecture == MoEArchitecture.LLAMA4

    def test_extract_moe_info_no_moe_layers(self):
        """Test extracting info from model without MoE."""
        mock_model = MagicMock()
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            # No router attribute
            del layer.mlp.router
            layer.mlp.router = None
            layers.append(layer)
        mock_model.model.layers = layers

        # Need to patch hasattr to return False for router
        original_hasattr = hasattr

        def custom_hasattr(obj, name):
            if name == "router" and hasattr(obj, "__class__") and "MagicMock" in str(type(obj)):
                return False
            return original_hasattr(obj, name)

        with patch("builtins.hasattr", custom_hasattr):
            info = ExpertRouter._extract_moe_info(mock_model)

        assert info.moe_layers == ()
        assert info.is_moe is False


class TestExpertRouterFromPretrained:
    """Tests for from_pretrained class method."""

    @pytest.mark.asyncio
    async def test_from_pretrained_calls_load_sync(self):
        """Test that from_pretrained calls _load_model_sync."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2, 3, 4, 5, 6, 7),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=8,
            architecture=MoEArchitecture.GPT_OSS,
        )

        # Create proper mock model structure
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            layer.mlp.router = MagicMock()
            layer.mlp.experts = MagicMock()
            layer.mlp.experts.gate_up_proj = MagicMock()
            layers.append(layer)
        mock_model.model.layers = layers

        with patch.object(
            ExpertRouter,
            "_load_model_sync",
            return_value=(mock_model, mock_tokenizer, mock_info),
        ):
            router = await ExpertRouter.from_pretrained("test/model")

            assert isinstance(router, ExpertRouter)
            assert router.info.num_experts == 32

    @pytest.mark.asyncio
    async def test_from_pretrained_threads_backend_device(self):
        """Test backend/device arguments are forwarded to the sync loader."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=2,
            num_experts_per_tok=1,
            total_layers=1,
            architecture=MoEArchitecture.GENERIC,
        )
        mock_runtime = MagicMock()
        mock_pipeline = MagicMock()

        layers = []
        layer = MagicMock()
        layer.mlp = MagicMock()
        layer.mlp.router = MagicMock()
        layers.append(layer)
        mock_model.model.layers = layers

        with patch.object(
            ExpertRouter,
            "_load_model_sync",
            return_value=(mock_model, mock_tokenizer, mock_info, mock_runtime, mock_pipeline, "torch"),
        ) as mock_loader:
            router = await ExpertRouter.from_pretrained(
                "test/model",
                backend="torch",
                device="cuda:0",
            )

        mock_loader.assert_called_once_with("test/model", "torch", "cuda:0")
        assert router._runtime is mock_runtime
        assert router._pipeline is mock_pipeline
        assert router._backend_name == "torch"


@pytest.mark.skipif(torch is None, reason="torch is not available")
class TestExpertRouterTorchRuntime:
    """Tests for the real torch runtime path inside ExpertRouter."""

    @pytest.mark.parametrize("device", TORCH_DEVICES)
    @pytest.mark.asyncio
    async def test_torch_generation_and_capture_paths(self, device):
        """Forced routing, ablation, top-k, and router capture work on torch."""
        router = build_torch_router(device)

        assert router._generate_normal_sync("alpha", 1) == "expert0"

        forced = await router.chat_with_expert(
            "alpha",
            expert_idx=1,
            max_tokens=1,
            apply_chat_template=False,
        )
        assert forced.response == "expert1"
        assert forced.stats.layers_modified == 1

        ablated_text, ablated_stats = await router.generate_with_ablation(
            "alpha",
            expert_indices=[0],
            max_tokens=1,
        )
        assert ablated_text == "expert1"
        assert ablated_stats.layers_modified == 1

        topk_result = await router.generate_with_topk("alpha", k=2, max_tokens=1)
        assert topk_result.response == "expert0"
        assert topk_result.normal_response == "expert0"

        weights = await router.capture_router_weights("alpha")
        assert len(weights) == 1
        assert weights[0].layer_idx == 0
        assert len(weights[0].positions) == 1
        assert weights[0].positions[0].token == "alpha"
        assert weights[0].positions[0].expert_indices == (0,)
        assert weights[0].positions[0].weights == pytest.approx((1.0,))
