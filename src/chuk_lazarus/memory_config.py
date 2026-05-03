"""Central memory recall configuration profiles.

The REPL and proof harnesses share one resolved configuration object so normal
chat can use the strongest bounded recall path without requiring users to know
the older KV-direct flags. Environment variables remain supported for harnesses
and power users.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

MEMORY_MODES = ("auto", "topical", "entity_mention", "vec_inject", "kv_direct", "off")
MEMORY_PROFILES = ("plug_and_play", "proof", "smoke", "nightly", "diagnostic")


@dataclass(frozen=True)
class DynamicAllocatorConfig:
    cognitive_ceiling_tokens: int = 4096
    allocation_quantum_tokens: int = 512
    output_buffers: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.output_buffers is None:
            object.__setattr__(
                self,
                "output_buffers",
                {
                    "DETECTIVE": 100,
                    "PARSER": 512,
                    "BUILDER": 1280,
                    "REASONER": 2048,
                },
            )

    @classmethod
    def calculate_max_k(cls, task_type: str, prompt_length: int) -> int:
        config = cls()
        normalized_task = str(task_type or "BUILDER").strip().upper()
        output_buffers = config.output_buffers or {}
        output_buffer = int(
            output_buffers.get(normalized_task, output_buffers["BUILDER"])
        )
        available_tokens = (
            int(config.cognitive_ceiling_tokens)
            - int(prompt_length)
            - output_buffer
        )
        return max(1, int(available_tokens // int(config.allocation_quantum_tokens)))


@dataclass(frozen=True)
class MemoryRecallConfig:
    profile: str = "plug_and_play"
    enabled: bool = True
    mode: str = "auto"
    recall_backend: str = "kv_direct"
    active_generation_engine: str = "residual_bounded_kv_direct"
    route_candidate_pool: int = 64
    candidate_pool: int = 12
    k_hot: int = 4
    k_warm: int = 4
    selector_policy: str = "utility-v2"
    dense_scoring: str = "deterministic"
    rrf_k: float = 60.0
    mmr_lambda: float = 0.75
    dedupe_sessions: bool = True
    residual_mode: str = "stream"
    insertion_family: str = "full_attention"
    max_total_inject_tokens: int = 4096
    hot_budget_mib: int = 32
    semantic_prefix_enabled: bool = True
    semantic_prefix_tokens: int = 4096
    include_cold_in_kv: bool = True
    allow_cold_attention: bool = False
    evidence_enabled: bool = True
    telemetry_enabled: bool = True
    fallback_policy: str = "normal_chat_if_no_relevant_memory"
    diagnostics_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROFILE_CONFIGS: dict[str, MemoryRecallConfig] = {
    "plug_and_play": MemoryRecallConfig(),
    "proof": MemoryRecallConfig(
        profile="proof",
        mode="kv_direct",
        route_candidate_pool=64,
        candidate_pool=12,
        k_hot=4,
        k_warm=4,
        max_total_inject_tokens=65536,
        semantic_prefix_tokens=4096,
    ),
    "smoke": MemoryRecallConfig(
        profile="smoke",
        mode="auto",
        route_candidate_pool=16,
        candidate_pool=8,
        k_hot=2,
        k_warm=2,
        max_total_inject_tokens=4096,
        semantic_prefix_tokens=1024,
    ),
    "nightly": MemoryRecallConfig(
        profile="nightly",
        mode="auto",
        route_candidate_pool=256,
        candidate_pool=64,
        k_hot=8,
        k_warm=8,
        max_total_inject_tokens=65536,
        hot_budget_mib=64,
        semantic_prefix_tokens=8192,
    ),
    "diagnostic": MemoryRecallConfig(
        profile="diagnostic",
        mode="kv_direct",
        route_candidate_pool=64,
        candidate_pool=32,
        k_hot=4,
        k_warm=4,
        max_total_inject_tokens=65536,
        diagnostics_enabled=True,
    ),
}


def _parse_bool(raw_value: object) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    text = str(raw_value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected boolean value, got {raw_value!r}")


def _parse_int(raw_value: object) -> int:
    return int(str(raw_value).strip())


def _parse_float(raw_value: object) -> float:
    return float(str(raw_value).strip())


def _parse_str(raw_value: object) -> str:
    return str(raw_value).strip()


_ENV_FIELD_PARSERS: dict[str, tuple[str, Any]] = {
    "LAZARUS_MEMORY_MODE": ("mode", _parse_str),
    "LAZARUS_GENERATION_ENGINE": ("active_generation_engine", _parse_str),
    "LAZARUS_KV_CANDIDATE_POOL": ("candidate_pool", _parse_int),
    "LAZARUS_KV_ROUTE_CANDIDATE_POOL": ("route_candidate_pool", _parse_int),
    "LAZARUS_KV_K_HOT": ("k_hot", _parse_int),
    "LAZARUS_KV_K_WARM": ("k_warm", _parse_int),
    "LAZARUS_ASI_SELECTOR_POLICY": ("selector_policy", _parse_str),
    "LAZARUS_ASI_DENSE_SCORING": ("dense_scoring", _parse_str),
    "LAZARUS_ASI_RRF_K": ("rrf_k", _parse_float),
    "LAZARUS_ASI_MMR_LAMBDA": ("mmr_lambda", _parse_float),
    "LAZARUS_KV_DEDUP_SESSION": ("dedupe_sessions", _parse_bool),
    "LAZARUS_MAX_TOTAL_INJECT_TOKENS": ("max_total_inject_tokens", _parse_int),
    "LAZARUS_KV_HOT_BUDGET_MIB": ("hot_budget_mib", _parse_int),
    "LAZARUS_HOT_BUDGET_MIB": ("hot_budget_mib", _parse_int),
    "LAZARUS_KV_SEMANTIC_PREFIX": ("semantic_prefix_enabled", _parse_bool),
    "LAZARUS_KV_SEMANTIC_PREFIX_TOKENS": ("semantic_prefix_tokens", _parse_int),
}


def get_memory_profile_names() -> tuple[str, ...]:
    return MEMORY_PROFILES


def get_memory_profile(name: str | None) -> MemoryRecallConfig:
    profile_name = (name or "plug_and_play").strip().lower().replace("-", "_")
    try:
        return _PROFILE_CONFIGS[profile_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown memory profile {name!r}; expected one of: " + ", ".join(MEMORY_PROFILES)
        ) from exc


def _normalized(config: MemoryRecallConfig) -> MemoryRecallConfig:
    mode = str(config.mode).strip().lower().replace("-", "_")
    if mode not in MEMORY_MODES:
        raise ValueError(
            f"unknown memory mode {config.mode!r}; expected one of: " + ", ".join(MEMORY_MODES)
        )
    enabled = bool(config.enabled) and mode != "off"
    return replace(
        config,
        profile=str(config.profile).strip().lower().replace("-", "_"),
        enabled=enabled,
        mode="off" if not enabled else mode,
        recall_backend=str(config.recall_backend).strip().lower(),
        active_generation_engine=str(config.active_generation_engine).strip().lower(),
        route_candidate_pool=max(1, int(config.route_candidate_pool)),
        candidate_pool=max(1, int(config.candidate_pool)),
        k_hot=max(0, int(config.k_hot)),
        k_warm=max(0, int(config.k_warm)),
        selector_policy=str(config.selector_policy).strip().lower().replace("_", "-"),
        dense_scoring=str(config.dense_scoring).strip().lower().replace("-", "_"),
        rrf_k=max(1.0, float(config.rrf_k)),
        mmr_lambda=min(1.0, max(0.0, float(config.mmr_lambda))),
        max_total_inject_tokens=max(0, int(config.max_total_inject_tokens)),
        hot_budget_mib=max(1, int(config.hot_budget_mib)),
        semantic_prefix_tokens=max(0, int(config.semantic_prefix_tokens)),
    )


def resolve_memory_config(
    *,
    profile: str | None = None,
    base: MemoryRecallConfig | None = None,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> MemoryRecallConfig:
    env = os.environ if environ is None else environ
    profile_name = profile or env.get("LAZARUS_MEMORY_PROFILE")
    config = get_memory_profile(profile_name) if base is None else base

    env_updates: dict[str, Any] = {}
    for env_key, (field_name, parser) in _ENV_FIELD_PARSERS.items():
        if env_key in env:
            env_updates[field_name] = parser(env[env_key])
    if env_updates:
        config = replace(config, **env_updates)

    clean_overrides = {
        key: value for key, value in dict(overrides or {}).items() if value is not None
    }
    if clean_overrides:
        config = replace(config, **clean_overrides)

    return _normalized(config)


def memory_config_env(config: MemoryRecallConfig) -> dict[str, str]:
    """Runtime env keys still consumed by lower-level retrieval code."""
    return {
        "LAZARUS_MAX_TOTAL_INJECT_TOKENS": str(int(config.max_total_inject_tokens)),
        "LAZARUS_KV_SEMANTIC_PREFIX": "1" if config.semantic_prefix_enabled else "0",
        "LAZARUS_KV_SEMANTIC_PREFIX_TOKENS": str(int(config.semantic_prefix_tokens)),
        "LAZARUS_ASI_SELECTOR_POLICY": str(config.selector_policy),
        "LAZARUS_ASI_DENSE_SCORING": str(config.dense_scoring),
        "LAZARUS_ASI_RRF_K": str(float(config.rrf_k)),
        "LAZARUS_ASI_MMR_LAMBDA": str(float(config.mmr_lambda)),
    }


__all__ = [
    "MEMORY_MODES",
    "MEMORY_PROFILES",
    "DynamicAllocatorConfig",
    "MemoryRecallConfig",
    "get_memory_profile",
    "get_memory_profile_names",
    "memory_config_env",
    "resolve_memory_config",
]
