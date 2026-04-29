"""Tests for ArchitectureConfig — registry, serialisation, user config."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from chuk_lazarus.inference.context.arch_config import (
    ArchitectureConfig,
    ArchitectureNotCalibrated,
)


class TestKnownRegistry:
    """Validated configs for known models."""

    def test_gemma_4b_known(self):
        """Gemma 3 4B (34 layers) should resolve to L29 H4."""
        config = _fake_config("gemma3", 34)
        ac = ArchitectureConfig.from_model_config(config)
        assert ac.retrieval_layer == 29
        assert ac.query_head == 4
        assert ac.injection_layer == 30

    def test_gemma_1b_known(self):
        """Gemma 1B (26 layers) should resolve to L17 H0."""
        config = _fake_config("gemma", 26)
        ac = ArchitectureConfig.from_model_config(config)
        assert ac.retrieval_layer == 17
        assert ac.query_head == 0
        assert ac.injection_layer == 18

    def test_gemma_variant_spellings(self):
        """All Gemma variant names normalise to 'gemma'."""
        for name in ("gemma", "gemma2", "gemma3", "gemma3_text"):
            config = _fake_config(name, 34)
            ac = ArchitectureConfig.from_model_config(config)
            assert ac.retrieval_layer == 29

    def test_gemma3_wrapper_uses_nested_text_config(self):
        config = _fake_wrapper_config("gemma3", "gemma3_text", 34)
        ac = ArchitectureConfig.from_model_config(config)
        assert ac.retrieval_layer == 29
        assert ac.query_head == 4
        assert ac.injection_layer == 30

    def test_gemma4_text_35_layers_known(self):
        """Gemma 4 E2B (35 layers) resolves to native L13 activation defaults."""
        config = _fake_config("gemma4_text", 35)
        ac = ArchitectureConfig.from_model_config(config)
        assert ac.retrieval_layer == 12
        assert ac.query_head == 7
        assert ac.injection_layer == 13
        assert ac.crystal_layer == 13

    def test_gemma4_wrapper_uses_nested_text_config(self):
        """Gemma 4 wrapper config (model_type=gemma4 with nested gemma4_text)
        resolves via the nested text_config, not the outer wrapper."""
        config = _fake_wrapper_config("gemma4", "gemma4_text", 35)
        ac = ArchitectureConfig.from_model_config(config)
        assert ac.retrieval_layer == 12
        assert ac.query_head == 7
        assert ac.injection_layer == 13
        assert ac.crystal_layer == 13

    def test_gemma4_does_not_alias_to_gemma3(self):
        """Gemma 4 values must NOT match any Gemma 3 calibration — per
        operator constraint, the Gemma 4 bucket is independent."""
        gemma4 = ArchitectureConfig.from_model_config(
            _fake_wrapper_config("gemma4", "gemma4_text", 35)
        )
        for gemma_layers in (26, 34):
            gemma3 = ArchitectureConfig.from_model_config(_fake_config("gemma", gemma_layers))
            assert (
                gemma4.retrieval_layer,
                gemma4.query_head,
                gemma4.injection_layer,
            ) != (
                gemma3.retrieval_layer,
                gemma3.query_head,
                gemma3.injection_layer,
            ), f"Gemma 4 aliases Gemma 3 ({gemma_layers} layers)"

    def test_gemma4_raises_for_unknown_layer_count(self):
        """Only the validated (gemma4, 35) entry is registered; other layer
        counts must surface ArchitectureNotCalibrated, not fall back to Gemma 3."""
        with pytest.raises(ArchitectureNotCalibrated) as exc_info:
            ArchitectureConfig.from_model_config(
                _fake_wrapper_config("gemma4", "gemma4_text", 27)
            )
        assert exc_info.value.model_type == "gemma4"
        assert exc_info.value.num_layers == 27

    def test_llama_32_layers(self):
        """SmolLM2 (llama, 32 layers) should be in registry."""
        config = _fake_config("llama", 32)
        ac = ArchitectureConfig.from_model_config(config)
        assert ac.retrieval_layer == 30
        assert ac.query_head == 6

    def test_unknown_raises(self):
        config = _fake_config("unknown_model", 99)
        with pytest.raises(ArchitectureNotCalibrated) as exc_info:
            ArchitectureConfig.from_model_config(config)
        assert "unknown_model" in str(exc_info.value)
        assert exc_info.value.model_type == "unknown_model"
        assert exc_info.value.num_layers == 99


class TestForModel:
    """Graceful fallback via for_model()."""

    def test_known_returns_config(self):
        config = _fake_config("gemma", 34)
        ac = ArchitectureConfig.for_model(config)
        assert ac is not None
        assert ac.retrieval_layer == 29

    def test_unknown_returns_none(self):
        config = _fake_config("imaginary", 50)
        ac = ArchitectureConfig.for_model(config)
        assert ac is None


class TestSerialisation:
    def test_to_dict_roundtrip(self):
        ac = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
            kv_head=1,
            head_dim=320,
            hidden_dim=2560,
        )
        d = ac.to_dict()
        ac2 = ArchitectureConfig.from_dict(d)
        assert ac2.retrieval_layer == ac.retrieval_layer
        assert ac2.query_head == ac.query_head
        assert ac2.injection_layer == ac.injection_layer
        assert ac2.kv_head == ac.kv_head
        assert ac2.head_dim == ac.head_dim
        assert ac2.hidden_dim == ac.hidden_dim

    def test_to_dict_minimal(self):
        """Only core fields + non-default inject_coefficient serialised."""
        ac = ArchitectureConfig(
            retrieval_layer=17,
            query_head=0,
            injection_layer=18,
        )
        d = ac.to_dict()
        # inject_coefficient=10.0 (non-default) is always serialised
        assert d["retrieval_layer"] == 17
        assert d["query_head"] == 0
        assert d["injection_layer"] == 18
        assert "kv_head" not in d
        assert "head_dim" not in d

    def test_threshold_multiplier_serialised_when_non_default(self):
        ac = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
            threshold_multiplier=3.0,
        )
        d = ac.to_dict()
        assert d["threshold_multiplier"] == 3.0

    def test_threshold_multiplier_omitted_when_default(self):
        ac = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
        )
        d = ac.to_dict()
        assert "threshold_multiplier" not in d

    def test_inject_coefficient_serialised_when_non_default(self):
        ac = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
            inject_coefficient=3.0,
        )
        d = ac.to_dict()
        assert d["inject_coefficient"] == 3.0

    def test_inject_coefficient_present_when_non_2(self):
        """Default is 10.0, so it serialises (it's non-2.0)."""
        ac = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
        )
        d = ac.to_dict()
        assert d["inject_coefficient"] == 10.0

    def test_entries_per_window_serialised_when_non_default(self):
        ac = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
            entries_per_window=16,
        )
        d = ac.to_dict()
        assert d["entries_per_window"] == 16

    def test_entries_per_window_omitted_when_default(self):
        ac = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
        )
        d = ac.to_dict()
        assert "entries_per_window" not in d

    def test_inject_coefficient_roundtrip(self):
        ac = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
            inject_coefficient=1.5,
            entries_per_window=16,
        )
        d = ac.to_dict()
        ac2 = ArchitectureConfig.from_dict(d)
        assert ac2.inject_coefficient == 1.5
        assert ac2.entries_per_window == 16

    def test_with_geometry(self):
        ac = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
        )
        ac2 = ac.with_geometry(kv_head=1, head_dim=320, hidden_dim=2560)
        assert ac2.kv_head == 1
        assert ac2.head_dim == 320
        assert ac2.hidden_dim == 2560
        assert ac2.k_dim == 320  # defaults to head_dim


class TestValidation:
    def test_negative_retrieval_layer(self):
        with pytest.raises(ValueError, match="retrieval_layer"):
            ArchitectureConfig(retrieval_layer=-1, query_head=0, injection_layer=0)

    def test_negative_query_head(self):
        with pytest.raises(ValueError, match="query_head"):
            ArchitectureConfig(retrieval_layer=0, query_head=-1, injection_layer=0)

    def test_negative_injection_layer(self):
        with pytest.raises(ValueError, match="injection_layer"):
            ArchitectureConfig(retrieval_layer=0, query_head=0, injection_layer=-1)

    def test_k_dim_defaults_to_head_dim(self):
        ac = ArchitectureConfig(
            retrieval_layer=0,
            query_head=0,
            injection_layer=1,
            head_dim=256,
        )
        assert ac.k_dim == 256


class TestUserConfigFile:
    def test_save_and_load(self, tmp_path):
        config_file = tmp_path / "arch_configs.json"
        config_dir = tmp_path

        with (
            patch("chuk_lazarus.inference.context.knowledge.config._USER_ARCH_FILE", config_file),
            patch("chuk_lazarus.inference.context.knowledge.config._USER_CONFIG_DIR", config_dir),
        ):
            ac = ArchitectureConfig(
                retrieval_layer=15,
                query_head=2,
                injection_layer=16,
            )
            ac.save_to_user_config("custom_model", 24)

            # Should be loadable
            loaded = ArchitectureConfig._load_user_config("custom_model", 24)
            assert loaded is not None
            assert loaded.retrieval_layer == 15
            assert loaded.query_head == 2

    def test_load_nonexistent_returns_none(self, tmp_path):
        config_file = tmp_path / "nonexistent.json"
        with patch("chuk_lazarus.inference.context.knowledge.config._USER_ARCH_FILE", config_file):
            assert ArchitectureConfig._load_user_config("foo", 10) is None

    def test_load_corrupt_file_returns_none(self, tmp_path):
        config_file = tmp_path / "arch_configs.json"
        config_file.write_text("not valid json{{{")
        with patch("chuk_lazarus.inference.context.knowledge.config._USER_ARCH_FILE", config_file):
            assert ArchitectureConfig._load_user_config("foo", 10) is None

    def test_from_model_config_uses_user_file(self, tmp_path):
        config_file = tmp_path / "arch_configs.json"
        config_dir = tmp_path

        data = {
            "newmodel:20": {
                "retrieval_layer": 10,
                "query_head": 3,
                "injection_layer": 11,
            }
        }
        config_file.write_text(json.dumps(data))

        with (
            patch("chuk_lazarus.inference.context.knowledge.config._USER_ARCH_FILE", config_file),
            patch("chuk_lazarus.inference.context.knowledge.config._USER_CONFIG_DIR", config_dir),
        ):
            config = _fake_config("newmodel", 20)
            ac = ArchitectureConfig.from_model_config(config)
            assert ac.retrieval_layer == 10
            assert ac.query_head == 3

    def test_from_model_config_uses_gemma4_wrapper_user_file(self, tmp_path):
        """User config file still provides values for Gemma 4 layer counts
        that are NOT in the built-in registry. The normalised key is
        ``gemma4:<N>`` (both ``gemma4`` and ``gemma4_text`` normalise to
        ``gemma4``)."""
        config_file = tmp_path / "arch_configs.json"
        config_dir = tmp_path

        # 36 is deliberately chosen to avoid colliding with the built-in
        # ("gemma4", 35) entry so we exercise the user-file fallback path.
        data = {
            "gemma4:36": {
                "retrieval_layer": 30,
                "query_head": 0,
                "injection_layer": 31,
            }
        }
        config_file.write_text(json.dumps(data))

        with (
            patch("chuk_lazarus.inference.context.knowledge.config._USER_ARCH_FILE", config_file),
            patch("chuk_lazarus.inference.context.knowledge.config._USER_CONFIG_DIR", config_dir),
        ):
            config = _fake_wrapper_config("gemma4", "gemma4_text", 36)
            ac = ArchitectureConfig.from_model_config(config)
            assert ac.retrieval_layer == 30
            assert ac.query_head == 0
            assert ac.injection_layer == 31

    def test_builtin_registry_wins_over_user_config_for_gemma4_35(self, tmp_path):
        """Built-in ``(gemma4, 35)`` values must take precedence over any
        stale user config file. This protects operators who upgrade from an
        older version that required a manual arch_configs.json entry."""
        config_file = tmp_path / "arch_configs.json"
        config_dir = tmp_path

        # Stale user config tries to override the built-in Gemma 4 entry.
        data = {
            "gemma4:35": {
                "retrieval_layer": 99,
                "query_head": 99,
                "injection_layer": 99,
            }
        }
        config_file.write_text(json.dumps(data))

        with (
            patch("chuk_lazarus.inference.context.knowledge.config._USER_ARCH_FILE", config_file),
            patch("chuk_lazarus.inference.context.knowledge.config._USER_CONFIG_DIR", config_dir),
        ):
            config = _fake_wrapper_config("gemma4", "gemma4_text", 35)
            ac = ArchitectureConfig.from_model_config(config)
            assert ac.retrieval_layer == 12
            assert ac.query_head == 7
            assert ac.injection_layer == 13
            assert ac.crystal_layer == 13

    def test_save_merges_with_existing(self, tmp_path):
        config_file = tmp_path / "arch_configs.json"
        config_dir = tmp_path

        existing = {"existing:10": {"retrieval_layer": 5, "query_head": 1, "injection_layer": 6}}
        config_file.write_text(json.dumps(existing))

        with (
            patch("chuk_lazarus.inference.context.knowledge.config._USER_ARCH_FILE", config_file),
            patch("chuk_lazarus.inference.context.knowledge.config._USER_CONFIG_DIR", config_dir),
        ):
            ac = ArchitectureConfig(
                retrieval_layer=15,
                query_head=2,
                injection_layer=16,
            )
            ac.save_to_user_config("new", 20)

            data = json.loads(config_file.read_text())
            assert "existing:10" in data
            assert "new:20" in data


class TestRepr:
    def test_basic_repr(self):
        ac = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
        )
        r = repr(ac)
        assert "retrieval_layer=29" in r
        assert "query_head=4" in r
        assert "injection_layer=30" in r

    def test_repr_with_geometry(self):
        ac = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
            kv_head=1,
            hidden_dim=2560,
            k_dim=320,
        )
        r = repr(ac)
        assert "kv_head=1" in r
        assert "hidden_dim=2560" in r


# ── Helpers ───────────────────────────────────────────────────────────


class _FakeConfig:
    def __init__(
        self,
        model_type: str,
        num_hidden_layers: int | None,
        *,
        text_config: _FakeConfig | None = None,
    ):
        self.model_type = model_type
        self.num_hidden_layers = num_hidden_layers
        if text_config is not None:
            self.text_config = text_config


def _fake_config(model_type: str, num_layers: int) -> _FakeConfig:
    return _FakeConfig(model_type, num_layers)


def _fake_wrapper_config(wrapper_type: str, text_type: str, text_layers: int) -> _FakeConfig:
    return _FakeConfig(
        wrapper_type,
        None,
        text_config=_FakeConfig(text_type, text_layers),
    )
