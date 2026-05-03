"""Unit tests for the residual-backed bounded KV-direct helpers.

These tests exercise the small helpers that plumb the Gemma research
methodology onto the torch Qwen path without materialising a large model.
The full residual-seeded generation path is integration-tested by pointing
the Pi launcher at a real Qwen checkpoint — see manual commands in the
PIR attached to this change.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


from chuk_lazarus.inference.backends._torch_residual_bounded import (  # noqa: E402
    GatheredResiduals,
    KVDirectMaterialization,
    PathAReplayGuard,
    ResidualBoundedState,
    _BoundaryResidualHook,
    _dtype_byte_size,
    _Layer0ResidualSeedHook,
    _resolve_attention_projections,
    _resolve_default_layers,
    _resolve_materialization_projection_layer,
    _text_config,
    estimate_kv_bytes_per_token,
    gather_selected_residuals,
    install_boundary_residual_capture_hook,
    install_layer0_residual_seed_hook,
    materialize_kv_direct,
)


class _FakeConfig:
    def __init__(
        self,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 8,
        num_key_value_heads: int | None = 4,
        hidden_size: int = 64,
        head_dim: int | None = None,
        text_config=None,
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.text_config = text_config


class _FakeModel:
    def __init__(self, config, dtype=torch.bfloat16) -> None:
        self.config = config
        self.dtype = dtype
        self._param = torch.zeros(1, dtype=dtype)

    def parameters(self):
        return iter([self._param])


def test_dtype_byte_size_covers_common_transformers_dtypes() -> None:
    assert _dtype_byte_size("torch.bfloat16") == 2
    assert _dtype_byte_size("float16") == 2
    assert _dtype_byte_size("torch.float32") == 4
    # Unknown dtype falls back to 2 to avoid NaN budgets on quantised runtimes.
    assert _dtype_byte_size("quint4x2") == 2


def test_text_config_returns_text_backbone_when_top_level_missing_fields() -> None:
    inner = _FakeConfig(num_hidden_layers=32)
    outer = SimpleNamespace(text_config=inner, num_hidden_layers=None)
    assert _text_config(outer) is inner

    standalone = _FakeConfig()
    assert _text_config(standalone) is standalone


def test_estimate_kv_bytes_per_token_uses_gqa_kv_heads() -> None:
    config = _FakeConfig(
        num_hidden_layers=4,
        num_attention_heads=32,
        num_kv_heads=None,  # unused path
        hidden_size=128,
        head_dim=8,
        num_key_value_heads=4,
    ) if False else _FakeConfig(
        num_hidden_layers=4,
        num_attention_heads=32,
        num_key_value_heads=4,
        hidden_size=128,
        head_dim=8,
    )
    model = _FakeModel(config, dtype=torch.bfloat16)
    # 2 * 4 (kv heads) * 8 (head_dim) * 4 (layers) * 2 (bf16) = 512 bytes/token
    assert estimate_kv_bytes_per_token(model) == 512


def test_estimate_kv_bytes_per_token_falls_back_when_head_dim_missing() -> None:
    config = _FakeConfig(
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_size=128,
        head_dim=None,
    )
    model = _FakeModel(config, dtype=torch.bfloat16)
    # head_dim = 128 // 16 = 8; 2 * 16 * 8 * 2 * 2 = 1024
    assert estimate_kv_bytes_per_token(model) == 1024


def test_estimate_kv_bytes_per_token_unwraps_vlm_text_config() -> None:
    text_backbone = _FakeConfig(
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_size=128,
        head_dim=64,
    )
    # VLM-style wrapper: top-level config omits the text dims.
    wrapper = SimpleNamespace(
        text_config=text_backbone,
        num_hidden_layers=None,
        num_attention_heads=None,
        hidden_size=None,
    )
    model = SimpleNamespace(
        config=wrapper,
        dtype=torch.float16,
        parameters=lambda: iter([torch.zeros(1, dtype=torch.float16)]),
    )
    # 2 * 2 * 64 * 8 * 2 = 4096
    assert estimate_kv_bytes_per_token(model) == 4096


def _load_loco_benchmark_module():
    repo_root = Path(__file__).resolve().parents[3]
    scripts_dir = repo_root / "scripts"
    for path in (repo_root, repo_root / "src", scripts_dir):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)
    module_name = "run_locobench_benchmark"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        scripts_dir / "run_locobench_benchmark.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _CpuRuntime:
    def __init__(self, layer_count: int = 4) -> None:
        self._model = torch.nn.Linear(1, 1)
        self._layer_count = int(layer_count)

    def _resolve_layers(self):
        return [object() for _ in range(self._layer_count)]


class _TinyTokenizer:
    def encode(self, text, **_kwargs):
        return [ord(ch) % 97 for ch in str(text)] or [0]


def _write_apollo_phase3_store(
    root: Path,
    *,
    source_layer: int = 1,
    num_windows: int = 2,
    hidden_dim: int = 3,
) -> Path:
    np = pytest.importorskip("numpy")
    store_dir = root / "case_00000" / "fast_store"
    (store_dir / "boundaries").mkdir(parents=True)
    (store_dir / "residual_streams").mkdir()
    np.savez(
        str(store_dir / "window_token_lists.npz"),
        **{
            "0": np.array([11, 12], dtype=np.int64),
            "1": np.array([21, 22, 23], dtype=np.int64),
        },
    )
    for window_id in range(num_windows):
        np.save(
            str(store_dir / "boundaries" / f"window_{window_id:03d}.npy"),
            np.full((hidden_dim,), float(window_id + 1), dtype=np.float32),
        )
        np.save(
            str(store_dir / "residual_streams" / f"window_{window_id:03d}.npy"),
            np.full((window_id + 2, hidden_dim), float(window_id + 1), dtype=np.float32),
        )
    manifest = {
        "kind": "benchmark_jit_apollo_sequential_residual",
        "status": "ready",
        "apollo_ready": True,
        "source_layer": int(source_layer),
        "layer": int(source_layer),
        "crystal_layer": int(source_layer),
        "target_layer": int(source_layer) + 1,
        "arch_config": {
            "retrieval_layer": 0,
            "query_head": 0,
            "injection_layer": int(source_layer),
            "crystal_layer": int(source_layer),
            "hidden_dim": int(hidden_dim),
            "head_dim": 1,
            "window_size": 8,
        },
        "num_windows": int(num_windows),
        "num_tokens": 5,
        "window_tokens": 8,
        "row_alignment": "window_id",
        "residual_streams_dir": "residual_streams",
        "boundaries_dir": "boundaries",
        "activation_routes": {"row_alignment": "window_id"},
    }
    (store_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return store_dir


def test_loco_phase3_uses_ready_apollo_store_without_recapture(tmp_path, monkeypatch) -> None:
    loco = _load_loco_benchmark_module()
    store_dir = _write_apollo_phase3_store(tmp_path)
    metadata_path = tmp_path / "jit_index_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "case_matrices": [
                    {
                        "case_index": 0,
                        "store_dir": str(store_dir),
                        "apollo_manifest_path": str(store_dir / "manifest.json"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        window_tokens=8,
        jit_indexing={"metadata_path": str(metadata_path)},
        _current_row_index=42,
        _jit_case_index_by_row={42: 0},
    )
    hot_windows = [
        loco.DependencyWindow(window_id=0, path="a.py", text="alpha", score=1.0),
        loco.DependencyWindow(window_id=1, path="b.py", text="beta", score=1.0),
    ]

    def _fail_recapture(**_kwargs):
        raise AssertionError("recapture should not run when Apollo store is ready")

    monkeypatch.setattr(loco, "recapture_layer13_residuals", _fail_recapture)

    residuals, encoded_by_window, metadata = loco.load_or_recapture_phase3_residuals(
        args=args,
        runtime=_CpuRuntime(layer_count=4),
        tokenizer=_TinyTokenizer(),
        hot_windows=hot_windows,
    )

    assert metadata["used_apollo_store"] is True
    assert metadata["residual_mode"] == "stream"
    assert residuals.source_layer == 1
    assert set(residuals.per_window_residuals) == {0, 1}
    assert tuple(residuals.per_window_residuals[0].shape) == (2, 3)
    assert tuple(residuals.per_window_residuals[1].shape) == (3, 3)
    assert encoded_by_window == {0: [11, 12], 1: [21, 22, 23]}


def test_loco_phase3_mixes_apollo_with_synthetic_recapture(
    tmp_path,
    monkeypatch,
) -> None:
    loco = _load_loco_benchmark_module()
    store_dir = _write_apollo_phase3_store(tmp_path)
    metadata_path = tmp_path / "jit_index_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "case_matrices": [
                    {
                        "case_index": 0,
                        "store_dir": str(store_dir),
                        "apollo_manifest_path": str(store_dir / "manifest.json"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        window_tokens=8,
        jit_indexing={"metadata_path": str(metadata_path)},
        _current_row_index=42,
        _jit_case_index_by_row={42: 0},
    )
    hot_windows = [
        loco.DependencyWindow(window_id=0, path="a.py", text="alpha", score=1.0),
        loco.DependencyWindow(window_id=-200000, path="target.py", text="synthetic", score=1.0),
        loco.DependencyWindow(window_id=1, path="b.py", text="beta", score=1.0),
    ]

    def _fake_recapture(**kwargs):
        assert [int(window.window_id) for window in kwargs["hot_windows"]] == [-200000]
        assert int(kwargs["source_layer"]) == 1
        sentinel = GatheredResiduals(
            per_window_residuals={-200000: torch.full((4, 3), 9.0)},
            per_window_token_ranges={-200000: (0, 4)},
            source_layer=1,
            archive_provenance={-200000: "jit_recapture://layer1/in_vram"},
        )
        return sentinel, {-200000: [31, 32, 33, 34]}, {"source_layer": 1, "storage": "in_vram_only"}

    monkeypatch.setattr(loco, "recapture_layer13_residuals", _fake_recapture)

    residuals, encoded_by_window, metadata = loco.load_or_recapture_phase3_residuals(
        args=args,
        runtime=_CpuRuntime(layer_count=4),
        tokenizer=_TinyTokenizer(),
        hot_windows=hot_windows,
    )

    assert metadata["used_apollo_store"] is True
    assert metadata["partial_recapture"] is True
    assert metadata["recaptured_window_ids"] == [-200000]
    assert list(residuals.per_window_residuals) == [0, -200000, 1]
    assert residuals.source_layer == 1
    assert residuals.per_window_token_ranges == {
        0: (0, 2),
        -200000: (2, 6),
        1: (6, 9),
    }
    assert encoded_by_window == {
        0: [11, 12],
        -200000: [31, 32, 33, 34],
        1: [21, 22, 23],
    }


def test_loco_phase3_missing_apollo_falls_back_unless_strict(monkeypatch) -> None:
    loco = _load_loco_benchmark_module()
    args = SimpleNamespace(
        window_tokens=8,
        jit_indexing={},
        _current_row_index=0,
    )
    hot_windows = [
        loco.DependencyWindow(window_id=0, path="a.py", text="alpha", score=1.0),
    ]
    sentinel = GatheredResiduals(
        per_window_residuals={0: torch.zeros(1, 3)},
        per_window_token_ranges={0: (0, 1)},
        source_layer=1,
        archive_provenance={0: "recapture://sentinel"},
    )
    called = {"value": False}

    def _fake_recapture(**_kwargs):
        called["value"] = True
        return sentinel, {0: [99]}, {"source_layer": 1}

    monkeypatch.setattr(loco, "recapture_layer13_residuals", _fake_recapture)
    residuals, encoded_by_window, metadata = loco.load_or_recapture_phase3_residuals(
        args=args,
        runtime=_CpuRuntime(layer_count=4),
        tokenizer=_TinyTokenizer(),
        hot_windows=hot_windows,
    )

    assert called["value"] is True
    assert residuals is sentinel
    assert encoded_by_window == {0: [99]}
    assert metadata["used_apollo_store"] is False
    assert metadata["fallback_recapture"] is True

    strict_args = SimpleNamespace(
        window_tokens=8,
        jit_indexing={},
        _current_row_index=0,
        require_phase3_apollo_store=True,
    )
    with pytest.raises(loco.Phase3Unavailable, match="required"):
        loco.load_or_recapture_phase3_residuals(
            args=strict_args,
            runtime=_CpuRuntime(layer_count=4),
            tokenizer=_TinyTokenizer(),
            hot_windows=hot_windows,
        )


def test_loco_phase3_incompatible_apollo_falls_back_unless_strict(
    tmp_path,
    monkeypatch,
) -> None:
    loco = _load_loco_benchmark_module()
    store_dir = _write_apollo_phase3_store(tmp_path, source_layer=1)
    args = SimpleNamespace(
        window_tokens=8,
        jit_indexing={},
        _current_row_index=0,
        phase3_apollo_store_dir=str(store_dir),
    )
    hot_windows = [
        loco.DependencyWindow(window_id=0, path="a.py", text="alpha", score=1.0),
    ]
    sentinel = GatheredResiduals(
        per_window_residuals={0: torch.zeros(1, 3)},
        per_window_token_ranges={0: (0, 1)},
        source_layer=1,
        archive_provenance={0: "recapture://sentinel"},
    )

    def _fake_recapture(**_kwargs):
        return sentinel, {0: [77]}, {"source_layer": 1}

    monkeypatch.setattr(loco, "recapture_layer13_residuals", _fake_recapture)
    residuals, encoded_by_window, metadata = loco.load_or_recapture_phase3_residuals(
        args=args,
        runtime=_CpuRuntime(layer_count=2),
        tokenizer=_TinyTokenizer(),
        hot_windows=hot_windows,
    )

    assert residuals is sentinel
    assert encoded_by_window == {0: [77]}
    assert metadata["used_apollo_store"] is False
    assert "incompatible with runtime layer stack" in metadata["apollo_unavailable_reason"]

    strict_args = SimpleNamespace(
        window_tokens=8,
        jit_indexing={},
        _current_row_index=0,
        phase3_apollo_store_dir=str(store_dir),
        require_phase3_apollo_store=True,
    )
    with pytest.raises(loco.Phase3Unavailable, match="incompatible with runtime layer stack"):
        loco.load_or_recapture_phase3_residuals(
            args=strict_args,
            runtime=_CpuRuntime(layer_count=2),
            tokenizer=_TinyTokenizer(),
            hot_windows=hot_windows,
        )


def test_resolve_attention_projections_prefers_direct_attention_metadata() -> None:
    self_attn = _DummySelfAttn(hidden=16, n_kv_heads=2, head_dim=8)
    self_attn.config = SimpleNamespace(
        num_key_value_heads=99,
        head_dim=99,
        num_attention_heads=99,
        hidden_size=99 * 99,
    )
    layer = SimpleNamespace(self_attn=self_attn)

    _k_proj, _v_proj, n_kv_heads, head_dim = _resolve_attention_projections(layer)

    assert n_kv_heads == 2
    assert head_dim == 8


def test_resolve_attention_projections_falls_back_to_attention_config() -> None:
    self_attn = _DummySelfAttn(hidden=16, n_kv_heads=2, head_dim=8)
    delattr(self_attn, "num_key_value_heads")
    delattr(self_attn, "head_dim")
    self_attn.config = SimpleNamespace(
        num_key_value_heads=2,
        head_dim=8,
        num_attention_heads=4,
        hidden_size=32,
    )
    layer = SimpleNamespace(self_attn=self_attn)

    _k_proj, _v_proj, n_kv_heads, head_dim = _resolve_attention_projections(layer)

    assert n_kv_heads == 2
    assert head_dim == 8


def test_resolve_attention_projections_derives_head_dim_from_attention_config() -> None:
    self_attn = _DummySelfAttn(hidden=16, n_kv_heads=2, head_dim=8)
    delattr(self_attn, "num_key_value_heads")
    delattr(self_attn, "head_dim")
    self_attn.config = SimpleNamespace(
        num_key_value_heads=2,
        head_dim=None,
        num_attention_heads=4,
        hidden_size=32,
    )
    layer = SimpleNamespace(self_attn=self_attn)

    _k_proj, _v_proj, n_kv_heads, head_dim = _resolve_attention_projections(layer)

    assert n_kv_heads == 2
    assert head_dim == 8


def test_resolve_materialization_projection_layer_walks_kv_shared_source() -> None:
    producer = SimpleNamespace(
        self_attn=SimpleNamespace(k_proj=object(), v_proj=object())
    )
    consumer = SimpleNamespace(
        self_attn=SimpleNamespace(is_kv_shared_layer=True, kv_shared_layer_index=0)
    )

    resolved_layer, resolved_idx = _resolve_materialization_projection_layer(
        [producer, consumer],
        1,
    )

    assert resolved_layer is producer
    assert resolved_idx == 0


def test_boundary_residual_hook_captures_last_position() -> None:
    hook = _BoundaryResidualHook()
    # Simulate a forward output: (B=1, S=5, H=8).
    output = torch.arange(1 * 5 * 8, dtype=torch.float32).view(1, 5, 8)
    hook(None, None, output)
    assert hook.tensor is not None
    assert hook.tensor.shape == (1, 1, 8)
    assert torch.equal(hook.tensor[0, 0], output[0, -1])


def test_boundary_residual_hook_ignores_scalar_outputs() -> None:
    hook = _BoundaryResidualHook()
    # (B, S) — only 2 dims, must be skipped.
    hook(None, None, torch.zeros(1, 3))
    assert hook.tensor is None
    # Also ignore bare ints/tensors that obviously are not hidden states.
    hook(None, None, 42)
    assert hook.tensor is None


def test_boundary_residual_hook_also_handles_tuple_output() -> None:
    hook = _BoundaryResidualHook()
    output = (
        torch.arange(1 * 4 * 3, dtype=torch.float32).view(1, 4, 3),
        None,
    )
    hook(None, None, output)
    assert hook.tensor is not None
    assert hook.tensor.shape == (1, 1, 3)


def test_layer0_seed_hook_injects_on_multi_position_forward_only() -> None:
    boundary = torch.tensor([[[1.0, 2.0, 3.0]]])  # (1, 1, 3)
    hook = _Layer0ResidualSeedHook(boundary)
    # Multi-position prefill — should inject at position 0.
    hidden = torch.zeros(1, 4, 3)
    (new_args, _), _kwargs = hook(None, (hidden,), {}), None  # prime tuple destruction
    # Call again to re-invoke properly via the real call path.
    # Reset and run properly:
    hook = _Layer0ResidualSeedHook(boundary)
    new_args, new_kwargs = hook(None, (hidden,), {})
    assert hook.injected is True
    modified = new_args[0]
    assert torch.equal(modified[0, 0], torch.tensor([1.0, 2.0, 3.0]))
    # Other positions must be untouched.
    assert torch.equal(modified[0, 1:], torch.zeros(3, 3))

    # Decode step (S=1) — must NOT latch.
    hook2 = _Layer0ResidualSeedHook(boundary)
    decode_hidden = torch.zeros(1, 1, 3)
    out_args, _out_kwargs = hook2(None, (decode_hidden,), {})
    assert hook2.injected is False
    # Decode input is returned unchanged (identity tuple).
    assert out_args[0] is decode_hidden


def test_layer0_seed_hook_latches_once() -> None:
    boundary = torch.ones(1, 1, 2)
    hook = _Layer0ResidualSeedHook(boundary)
    hidden_first = torch.zeros(1, 3, 2)
    hook(None, (hidden_first,), {})
    assert hook.injected is True
    # Second multi-position call should be a no-op because we latched.
    hidden_second = torch.full((1, 3, 2), 7.0)
    out_args, _ = hook(None, (hidden_second,), {})
    assert torch.equal(out_args[0], hidden_second)


def test_install_helpers_return_usable_handles() -> None:
    layer = torch.nn.Linear(8, 8)
    hook, handle = install_boundary_residual_capture_hook(layer)
    try:
        out = layer(torch.zeros(1, 3, 8))
        # Linear returns (B, S, H); promote to (B, S, H) by the hook's capture.
        hook(layer, None, out)
        assert hook.tensor is not None
    finally:
        handle.remove()

    boundary = torch.ones(1, 1, 8)
    seed_hook, seed_handle = install_layer0_residual_seed_hook(layer, boundary)
    try:
        assert seed_hook.injected is False
    finally:
        seed_handle.remove()


def test_residual_bounded_state_defaults_are_fresh_per_instance() -> None:
    a = ResidualBoundedState()
    b = ResidualBoundedState()
    a.cold_token_ids.append(1)
    assert b.cold_token_ids == []
    assert a.boundary_residual is None
    assert a.boundary_residual_position == -1
    assert a.rehydrate_count == 0
    assert a.prefill_was_split is False


# ---------------------------------------------------------------------------
# Axis-5 conformance tests — archived-residual K/V materialization.
#
# Frozen contract: ``ve-ins-0mo9pb4f5000004cc67``. Tests here prove each
# proof obligation on a minimal CUDA DummyModel without needing a real Qwen
# checkpoint. GPU-only by policy — see MEMORY "GPU-only test execution".
# ---------------------------------------------------------------------------


class _DummySelfAttn(torch.nn.Module):
    """Split-projection attention stub with the axis-5 required surface."""

    def __init__(self, hidden: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.k_proj = torch.nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.num_key_value_heads = n_kv_heads
        self.head_dim = head_dim


class _DummyDecoderLayer(torch.nn.Module):
    """Minimal decoder layer whose ``self_attn`` carries split K/V projections."""

    def __init__(self, hidden: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.self_attn = _DummySelfAttn(hidden, n_kv_heads, head_dim)

    def forward(self, hidden_states):
        return hidden_states


class _DummySharedKvConsumerAttn(torch.nn.Module):
    """Attention stub that mirrors Gemma-4 shared-KV consumer metadata."""

    def __init__(self, producer_idx: int):
        super().__init__()
        self.is_kv_shared_layer = True
        self.kv_shared_layer_index = int(producer_idx)


class _DummyInner(torch.nn.Module):
    """Container exposing ``.layers`` like HF decoder models."""

    def __init__(self, hidden: int, n_kv_heads: int, head_dim: int, num_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_DummyDecoderLayer(hidden, n_kv_heads, head_dim) for _ in range(num_layers)]
        )


class _DummyAxis5Model(torch.nn.Module):
    """Tiny causal LM whose layers expose split K/V projections for axis-5."""

    def __init__(
        self,
        hidden: int = 16,
        n_kv_heads: int = 2,
        head_dim: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        self.model = _DummyInner(hidden, n_kv_heads, head_dim, num_layers)
        self.config = SimpleNamespace(
            hidden_size=hidden,
            head_dim=head_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=n_kv_heads,
            num_key_value_heads=n_kv_heads,
        )


class _StubTorchKnowledgeStore:
    """Stub for :class:`TorchKnowledgeStore` exposing the surface gather uses.

    The axis-5 gather path only reads ``store.config.injection_layer`` and
    ``store.load_boundary(wid, device=...)``. A real store is not needed
    for these conformance tests, and the scope binding permits stubbing.
    """

    def __init__(self, hidden: int, injection_layer: int, *, seed: int = 0) -> None:
        self.config = SimpleNamespace(
            injection_layer=int(injection_layer),
            crystal_layer=int(injection_layer),
        )
        self._hidden = int(hidden)
        self._seed = int(seed)

    def load_boundary(self, window_id: int, *, device: str = "cuda") -> torch.Tensor:
        # Deterministic per-window tensor so eviction tests can verify which
        # residuals survived.
        generator = torch.Generator(device=device)
        generator.manual_seed(self._seed + int(window_id))
        return torch.randn(self._hidden, device=device, generator=generator)


def _make_tier_assignment(window_id: int, tier):
    """Build a lightweight object with the attrs ``gather_selected_residuals`` reads."""
    return SimpleNamespace(
        candidate=SimpleNamespace(window_id=int(window_id)),
        tier=tier,
        rank=0,
        policy_version="rank-v1",
        policy_params={},
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="axis-5 KV-direct tests require CUDA"
)
class TestAxis5GatherAndMaterialize:
    """Conformance suite for axis-5 ``asi-kv-direct-chat`` leg-3.

    Proves the proof obligations for :func:`gather_selected_residuals`,
    :func:`materialize_kv_direct`, and :class:`PathAReplayGuard`.
    """

    # ------------------------------------------------------------------
    # Dataclass frozen-contract assertions.
    # ------------------------------------------------------------------
    def test_gathered_residuals_is_frozen_dataclass(self) -> None:
        residuals = GatheredResiduals(
            per_window_residuals={0: torch.zeros(4, device="cuda")},
            per_window_token_ranges={0: (0, 1)},
            source_layer=1,
            archive_provenance={0: "/tmp/x/boundaries/window_000.npy"},
        )
        field_names = {f.name for f in dataclasses.fields(residuals)}
        assert field_names == {
            "per_window_residuals",
            "per_window_token_ranges",
            "source_layer",
            "archive_provenance",
        }
        with pytest.raises(dataclasses.FrozenInstanceError):
            residuals.source_layer = 99  # type: ignore[misc]

    def test_kv_direct_materialization_is_frozen_dataclass(self) -> None:
        mat = KVDirectMaterialization(
            K=torch.zeros(1, 2, 2, 8, device="cuda"),
            V=torch.zeros(1, 2, 2, 8, device="cuda"),
            materialization_mode="project_through_W_K_W_V_at_injection_layer",
            hot_budget_mib_observed=1,
            path_a_replay_count=0,
        )
        field_names = {f.name for f in dataclasses.fields(mat)}
        assert field_names == {
            "K",
            "V",
            "materialization_mode",
            "hot_budget_mib_observed",
            "path_a_replay_count",
            "materialized_source_layer",
            "materialized_insertion_family",
            "materialized_lineage_layer_indices",
            "per_window_token_ranges",
        }
        with pytest.raises(dataclasses.FrozenInstanceError):
            mat.path_a_replay_count = 99  # type: ignore[misc]

    # ------------------------------------------------------------------
    # gather_selected_residuals: tier filtering + provenance + ranges.
    # ------------------------------------------------------------------
    def test_gather_skips_cold_and_keeps_hot_and_warm(self, tmp_path) -> None:
        from chuk_lazarus.session_retrieval.tier_policy import TierLabel

        hidden = 16
        injection_layer = 1
        store = _StubTorchKnowledgeStore(hidden, injection_layer, seed=1234)

        # Checkpoint handle stub — gather only reads ``checkpoint_dir``.
        checkpoint_handle = SimpleNamespace(
            checkpoint_dir=tmp_path,
            torch_store_dir=tmp_path / "torch_store",
            session_id="test",
            manifest={},
            original_input_dir=None,
        )

        assignments = [
            _make_tier_assignment(0, TierLabel.HOT),
            _make_tier_assignment(1, TierLabel.WARM),
            _make_tier_assignment(2, TierLabel.COLD),
        ]

        result = gather_selected_residuals(
            assignments,
            checkpoint_handle=checkpoint_handle,
            store=store,
            source_layer=injection_layer,
            device="cuda",
        )

        # COLD excluded; HOT + WARM kept.
        assert set(result.per_window_residuals.keys()) == {0, 1}
        # source_layer round-trips the injection_layer.
        assert result.source_layer == injection_layer
        # per_window_token_ranges is a single-slot range per window.
        for wid, (start, end) in result.per_window_token_ranges.items():
            assert end == start + 1
        # Slot ids are contiguous in the gather order [HOT, WARM].
        assert result.per_window_token_ranges[0] == (0, 1)
        assert result.per_window_token_ranges[1] == (1, 2)
        # Archive provenance is rooted at the live checkpoint dir.
        assert result.archive_provenance[0].endswith("boundaries/window_000.npy")
        assert result.archive_provenance[1].endswith("boundaries/window_001.npy")
        # Every residual is on CUDA.
        for tensor in result.per_window_residuals.values():
            assert tensor.device.type == "cuda"

    def test_gather_rejects_mismatched_source_layer(self, tmp_path) -> None:
        from chuk_lazarus.session_retrieval.tier_policy import TierLabel

        hidden = 16
        injection_layer = 1
        store = _StubTorchKnowledgeStore(hidden, injection_layer)

        checkpoint_handle = SimpleNamespace(
            checkpoint_dir=tmp_path,
            torch_store_dir=tmp_path / "torch_store",
            session_id="test",
            manifest={},
            original_input_dir=None,
        )
        assignments = [_make_tier_assignment(0, TierLabel.HOT)]

        with pytest.raises(ValueError):
            gather_selected_residuals(
                assignments,
                checkpoint_handle=checkpoint_handle,
                store=store,
                source_layer=injection_layer + 99,
                device="cuda",
            )

    def test_materialize_kv_direct_source_layer_equals_injection_layer(
        self, tmp_path
    ) -> None:
        """Round-trip: gather result's ``source_layer`` matches store config."""
        from chuk_lazarus.session_retrieval.tier_policy import TierLabel

        hidden = 16
        injection_layer = 1
        store = _StubTorchKnowledgeStore(hidden, injection_layer)

        checkpoint_handle = SimpleNamespace(
            checkpoint_dir=tmp_path,
            torch_store_dir=tmp_path / "torch_store",
            session_id="test",
            manifest={},
            original_input_dir=None,
        )
        assignments = [_make_tier_assignment(0, TierLabel.HOT)]

        result = gather_selected_residuals(
            assignments,
            checkpoint_handle=checkpoint_handle,
            store=store,
            source_layer=injection_layer,
            device="cuda",
        )
        assert result.source_layer == int(store.config.injection_layer)

    # ------------------------------------------------------------------
    # materialize_kv_direct: shape, budget, eviction, replay counter.
    # ------------------------------------------------------------------
    def _build_runtime_and_residuals(
        self,
        *,
        hidden: int = 16,
        n_kv_heads: int = 2,
        head_dim: int = 8,
        num_layers: int = 4,
        injection_layer: int = 1,
        num_windows: int = 2,
    ):
        """Construct a fresh DummyModel-backed runtime and GatheredResiduals."""
        from chuk_lazarus.inference.backends import TorchInferenceRuntime

        model = _DummyAxis5Model(
            hidden=hidden,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_layers=num_layers,
        ).to("cuda").eval()

        class _TinyTokenizer:
            eos_token_id = 99
            pad_token_id = 0

        runtime = TorchInferenceRuntime(model, _TinyTokenizer(), device="cuda")

        per_window_residuals: dict[int, torch.Tensor] = {}
        per_window_token_ranges: dict[int, tuple[int, int]] = {}
        archive_provenance: dict[int, str] = {}
        for slot, wid in enumerate(range(num_windows)):
            generator = torch.Generator(device="cuda")
            generator.manual_seed(4242 + wid)
            per_window_residuals[wid] = torch.randn(
                hidden, device="cuda", generator=generator
            )
            per_window_token_ranges[wid] = (slot, slot + 1)
            archive_provenance[wid] = f"/tmp/x/boundaries/window_{wid:03d}.npy"

        residuals = GatheredResiduals(
            per_window_residuals=per_window_residuals,
            per_window_token_ranges=per_window_token_ranges,
            source_layer=injection_layer,
            archive_provenance=archive_provenance,
        )
        return runtime, residuals, model

    def test_materialize_kv_direct_shape_matches_num_selected_windows(self) -> None:
        runtime, residuals, _ = self._build_runtime_and_residuals(
            hidden=16, n_kv_heads=2, head_dim=8, num_windows=2, injection_layer=1
        )
        mat = materialize_kv_direct(residuals, runtime, hot_budget_mib=1024)

        # Axis-5 contract: (batch=1, n_kv_heads, sink + N, head_dim).
        assert tuple(mat.K.shape) == (1, 2, 3, 8)
        assert tuple(mat.V.shape) == (1, 2, 3, 8)
        assert torch.count_nonzero(mat.K[:, :, 0, :]) == 0
        assert torch.count_nonzero(mat.V[:, :, 0, :]) == 0
        assert mat.K.device.type == "cuda"
        assert mat.V.device.type == "cuda"
        assert mat.materialization_mode == "project_through_W_K_W_V_at_injection_layer"

    def test_materialize_kv_direct_path_a_replay_count_is_zero(self) -> None:
        """Under normal construction no Path-A replay happens → counter == 0."""
        runtime, residuals, _ = self._build_runtime_and_residuals(num_windows=2)
        mat = materialize_kv_direct(residuals, runtime, hot_budget_mib=1024)
        assert mat.path_a_replay_count == 0

    def test_materialize_kv_direct_uses_shared_kv_producer_projection(self) -> None:
        """Gemma-4 KV-consumer layers materialize through their producer W_K/W_V."""
        hidden = 16
        n_kv_heads = 2
        head_dim = 8
        runtime, residuals, model = self._build_runtime_and_residuals(
            hidden=hidden,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_windows=1,
            injection_layer=1,
        )
        layers = _resolve_default_layers(model)
        producer_idx = 1
        target_idx = 2
        layers[target_idx].self_attn = _DummySharedKvConsumerAttn(producer_idx)

        mat = materialize_kv_direct(residuals, runtime, hot_budget_mib=1024)

        producer_k_proj = layers[producer_idx].self_attn.k_proj
        producer_v_proj = layers[producer_idx].self_attn.v_proj
        first_param = next(model.parameters())
        residual = residuals.per_window_residuals[0].to(
            device=first_param.device,
            dtype=first_param.dtype,
        )
        stacked = residual.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            expected_k = producer_k_proj(stacked)
            expected_v = producer_v_proj(stacked)
        expected_k = (
            expected_k.view(1, 1, n_kv_heads, head_dim).transpose(1, 2).contiguous()
        )
        expected_v = (
            expected_v.view(1, 1, n_kv_heads, head_dim).transpose(1, 2).contiguous()
        )
        expected_k = torch.cat([torch.zeros_like(expected_k), expected_k], dim=-2)
        expected_v = torch.cat([torch.zeros_like(expected_v), expected_v], dim=-2)

        torch.testing.assert_close(mat.K, expected_k)
        torch.testing.assert_close(mat.V, expected_v)

    def test_path_a_guard_raises_on_layer_preinjection_forward(self) -> None:
        """Directly verify the guard catches pre-injection layer forwards."""
        hidden = 16
        model = _DummyAxis5Model(
            hidden=hidden, n_kv_heads=2, head_dim=8, num_layers=4
        ).to("cuda").eval()

        injection_layer = 1
        guard = PathAReplayGuard(
            model, injection_layer=injection_layer, resolve_layers=_resolve_default_layers
        )
        with pytest.raises(RuntimeError) as excinfo:
            with guard:
                # Trigger layer 0 (< injection_layer) — must raise.
                model.model.layers[0](torch.randn(1, 1, hidden, device="cuda"))
        assert "axis-5 violated: Path A replay detected at layer 0" in str(excinfo.value)
        # guard.count is incremented before the hook raises.
        assert guard.count >= 1

    def test_materialize_kv_direct_honors_hot_budget_mib(self) -> None:
        """Observed MiB must fit under the configured ceiling.

        The attention sink is a real K/V slot, so a budget that cannot hold
        both the reserved sink and at least one memory slot is rejected before
        projection.
        """
        hidden = 16
        n_kv_heads = 2
        head_dim = 8
        runtime, residuals, _ = self._build_runtime_and_residuals(
            hidden=hidden,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_windows=2,
            injection_layer=1,
        )

        with pytest.raises(RuntimeError, match="reserved attention sink plus one memory slot"):
            materialize_kv_direct(residuals, runtime, hot_budget_mib=0)

    def test_materialize_kv_direct_evicts_warm_before_hot_under_budget_pressure(
        self,
    ) -> None:
        """With 3 windows [HOT, HOT, WARM] + ``tier_assignments``, WARM evicts
        first even when the budget cannot hold all of HOT.

        The reserved sink counts against the K/V budget. This fixture makes
        a 3 MiB budget hold exactly sink + two memory slots, so the WARM
        window is dropped while both HOT windows survive.
        """
        from chuk_lazarus.session_retrieval.tier_policy import TierLabel

        hidden = 1
        n_kv_heads = 1
        head_dim = 131072
        runtime, residuals, model = self._build_runtime_and_residuals(
            hidden=hidden,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_windows=3,
            injection_layer=1,
        )

        # Stamp deterministic distinct values per window so we can verify
        # which archived residual survived post-eviction.
        distinct_vals = {
            0: torch.full((hidden,), 1.0, device="cuda"),
            1: torch.full((hidden,), 2.0, device="cuda"),
            2: torch.full((hidden,), 3.0, device="cuda"),
        }
        # Rebuild GatheredResiduals with the distinct values.
        distinct_residuals = GatheredResiduals(
            per_window_residuals=dict(distinct_vals),
            per_window_token_ranges={
                0: (0, 1),
                1: (1, 2),
                2: (2, 3),
            },
            source_layer=1,
            archive_provenance={
                0: "/tmp/x/boundaries/window_000.npy",
                1: "/tmp/x/boundaries/window_001.npy",
                2: "/tmp/x/boundaries/window_002.npy",
            },
        )

        assignments = [
            _make_tier_assignment(0, TierLabel.HOT),
            _make_tier_assignment(1, TierLabel.HOT),
            _make_tier_assignment(2, TierLabel.WARM),
        ]

        mat = materialize_kv_direct(
            distinct_residuals,
            runtime,
            hot_budget_mib=3,
            tier_assignments=assignments,
        )

        assert mat.path_a_replay_count == 0
        # Sink + both HOT windows survive; WARM is evicted.
        assert mat.K.shape[-2] == 3
        assert mat.per_window_token_ranges == {0: (1, 2), 1: (2, 3)}
        # Verify the two surviving slots correspond to HOT window ids 0 and 1
        # (not the WARM window id 2). Project wid=0 and wid=1 residuals
        # through the same k_proj and compare byte-for-byte.
        layers = _resolve_default_layers(model)
        target = layers[1 + 1]  # injection_layer + 1 == 2
        k_proj = target.self_attn.k_proj

        def _project(res_vec: torch.Tensor) -> torch.Tensor:
            first_param = next(model.parameters())
            stacked = (
                res_vec.to(device=first_param.device, dtype=first_param.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
            )  # (1, 1, hidden)
            with torch.no_grad():
                flat = k_proj(stacked)
            return (
                flat.view(1, 1, n_kv_heads, head_dim).transpose(1, 2).contiguous()
            )

        expected_w0 = _project(distinct_vals[0])
        expected_w1 = _project(distinct_vals[1])
        assert torch.count_nonzero(mat.K[:, :, 0:1, :]) == 0
        torch.testing.assert_close(mat.K[:, :, 1:2, :], expected_w0)
        torch.testing.assert_close(mat.K[:, :, 2:3, :], expected_w1)
