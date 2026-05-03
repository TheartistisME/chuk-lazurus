#!/usr/bin/env python3
"""Persistent decoder dialect prior store.

This module is intentionally stdlib-only and independent from the router. It is
small enough for benchmark harnesses to use now, while remaining suitable for a
future centralized decoding controller.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_NAME = "lazarus.decoder_dialect_prior_store"
SCHEMA_VERSION = 1
DEFAULT_MODE = "same_dialect_bounded_seed"
DEFAULT_STEERING_VERSION = "lm_head_logit_lens_v1"
DEFAULT_MAX_SEED_ALPHA = 0.18
DEFAULT_PEAK_TO_SEED = 0.35
DEFAULT_CROSS_CASE_DECAY = 0.62
DEFAULT_ALPHA_FLOOR = 0.03


@dataclass(frozen=True)
class DecoderPriorScope:
    """Compatibility boundary for a decoder prior store."""

    model_key: str = ""
    tokenizer_key: str = ""
    layer_index: int = 14
    steering_version: str = DEFAULT_STEERING_VERSION
    benchmark: str = ""
    task_type: str = ""
    runner: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "tokenizer_key": self.tokenizer_key,
            "layer_index": int(self.layer_index),
            "steering_version": self.steering_version,
            "benchmark": self.benchmark,
            "task_type": self.task_type,
            "runner": self.runner,
        }


@dataclass(frozen=True)
class DecoderPriorConfig:
    """Policy knobs for turning observed temptation into future seed alpha."""

    max_seed_alpha: float = DEFAULT_MAX_SEED_ALPHA
    peak_to_seed: float = DEFAULT_PEAK_TO_SEED
    cross_case_decay: float = DEFAULT_CROSS_CASE_DECAY
    alpha_floor: float = DEFAULT_ALPHA_FLOOR
    mode: str = DEFAULT_MODE

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_seed_alpha": clamp_unit(self.max_seed_alpha),
            "peak_to_seed": clamp_unit(self.peak_to_seed),
            "cross_case_decay": clamp_unit(self.cross_case_decay),
            "alpha_floor": max(0.0, float(self.alpha_floor)),
            "mode": self.mode,
        }


def clamp_unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def new_decoder_dialect_prior_state(
    *,
    scope: dict[str, Any] | DecoderPriorScope | None = None,
    config: dict[str, Any] | DecoderPriorConfig | None = None,
) -> dict[str, Any]:
    if isinstance(scope, DecoderPriorScope):
        scope_dict = scope.as_dict()
    else:
        scope_dict = dict(scope or {})
    if isinstance(config, DecoderPriorConfig):
        config_dict = config.as_dict()
    else:
        config_dict = {**DecoderPriorConfig().as_dict(), **dict(config or {})}
    return {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "mode": str(config_dict.get("mode", DEFAULT_MODE)),
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "seen_cases": 0,
        "accepted_cases": 0,
        "max_seed_alpha": clamp_unit(config_dict.get("max_seed_alpha", DEFAULT_MAX_SEED_ALPHA)),
        "peak_to_seed": clamp_unit(config_dict.get("peak_to_seed", DEFAULT_PEAK_TO_SEED)),
        "cross_case_decay": clamp_unit(
            config_dict.get("cross_case_decay", DEFAULT_CROSS_CASE_DECAY)
        ),
        "alpha_floor": max(0.0, float(config_dict.get("alpha_floor", DEFAULT_ALPHA_FLOOR))),
        "scope": json_safe(scope_dict),
        "dialects": {},
    }


def normalize_decoder_dialect_prior_state(
    prior: dict[str, Any] | None,
    *,
    scope: dict[str, Any] | DecoderPriorScope | None = None,
    config: dict[str, Any] | DecoderPriorConfig | None = None,
) -> dict[str, Any]:
    if not isinstance(prior, dict):
        return new_decoder_dialect_prior_state(scope=scope, config=config)
    fresh = new_decoder_dialect_prior_state(scope=scope, config=config)
    normalized = {**fresh, **prior}
    normalized["schema"] = SCHEMA_NAME
    normalized["version"] = SCHEMA_VERSION
    if scope is not None:
        normalized["scope"] = (
            scope.as_dict() if isinstance(scope, DecoderPriorScope) else dict(scope)
        )
    normalized["seen_cases"] = int(normalized.get("seen_cases", 0) or 0)
    normalized["accepted_cases"] = int(normalized.get("accepted_cases", 0) or 0)
    normalized["max_seed_alpha"] = clamp_unit(normalized.get("max_seed_alpha"))
    normalized["peak_to_seed"] = clamp_unit(normalized.get("peak_to_seed"))
    normalized["cross_case_decay"] = clamp_unit(normalized.get("cross_case_decay"))
    normalized["alpha_floor"] = max(0.0, float(normalized.get("alpha_floor", DEFAULT_ALPHA_FLOOR)))
    if not isinstance(normalized.get("dialects"), dict):
        normalized["dialects"] = {}
    return json_safe(normalized)


def scopes_compatible(
    existing_scope: dict[str, Any] | None,
    expected_scope: dict[str, Any] | DecoderPriorScope | None,
) -> tuple[bool, list[str]]:
    if expected_scope is None:
        return True, []
    expected = expected_scope.as_dict() if isinstance(expected_scope, DecoderPriorScope) else dict(expected_scope)
    existing = dict(existing_scope or {})
    strict_keys = ("model_key", "tokenizer_key", "layer_index", "steering_version")
    mismatches = [
        key
        for key in strict_keys
        if str(existing.get(key, "")) != str(expected.get(key, ""))
    ]
    return not mismatches, mismatches


def dialects_from_dialect_plan(dialect_plan: dict[str, Any]) -> list[str]:
    paths = dialect_plan.get("paths", {}) if isinstance(dialect_plan, dict) else {}
    if not isinstance(paths, dict):
        return []
    dialects = {
        str(spec.get("target_dialect", ""))
        for spec in paths.values()
        if isinstance(spec, dict) and str(spec.get("target_dialect", ""))
    }
    return sorted(dialects)


def decoder_dialect_prior_plan(
    prior: dict[str, Any] | None,
    dialect_plan: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(prior, dict):
        return {"active": False, "reason": "no_cross_case_prior"}
    current_dialects = dialects_from_dialect_plan(dialect_plan)
    if not current_dialects:
        return {"active": False, "reason": "no_current_dialect"}
    dialect_entries = prior.get("dialects", {})
    if not isinstance(dialect_entries, dict):
        return {"active": False, "reason": "malformed_prior"}
    max_seed_alpha = clamp_unit(prior.get("max_seed_alpha", DEFAULT_MAX_SEED_ALPHA))
    matched: dict[str, dict[str, Any]] = {}
    seed_by_family: dict[str, float] = {}
    for dialect in current_dialects:
        entry = dialect_entries.get(dialect, {})
        if not isinstance(entry, dict):
            continue
        families = entry.get("families", {})
        if not isinstance(families, dict):
            families = {}
        family_seeds: dict[str, float] = {}
        for family, family_entry in families.items():
            if not isinstance(family_entry, dict):
                continue
            seed_alpha = min(max_seed_alpha, clamp_unit(family_entry.get("seed_alpha", 0.0)))
            if seed_alpha <= 0.0:
                continue
            family_key = str(family)
            family_seeds[family_key] = seed_alpha
            seed_by_family[family_key] = max(seed_by_family.get(family_key, 0.0), seed_alpha)
        if int(entry.get("seen_cases", 0) or 0) or family_seeds:
            matched[dialect] = {
                "seen_cases": int(entry.get("seen_cases", 0) or 0),
                "accepted_case_count": int(entry.get("accepted_case_count", 0) or 0),
                "successful_case_streak": int(entry.get("successful_case_streak", 0) or 0),
                "last_row_index": entry.get("last_row_index"),
                "last_instance_id": entry.get("last_instance_id", ""),
                "last_repo": entry.get("last_repo", ""),
                "seed_alpha_by_family": dict(sorted(family_seeds.items())),
            }
    return {
        "active": bool(matched),
        "mode": str(prior.get("mode", DEFAULT_MODE)),
        "scope": dict(prior.get("scope", {}) or {}),
        "current_dialects": current_dialects,
        "matched_dialects": sorted(matched),
        "matched": matched,
        "seen_cases": int(prior.get("seen_cases", 0) or 0),
        "accepted_cases": int(prior.get("accepted_cases", 0) or 0),
        "max_seed_alpha": max_seed_alpha,
        "seed_alpha_by_family": dict(sorted(seed_by_family.items())),
    }


def decay_decoder_dialect_prior(prior: dict[str, Any]) -> None:
    decay = clamp_unit(prior.get("cross_case_decay", DEFAULT_CROSS_CASE_DECAY))
    floor = max(0.0, float(prior.get("alpha_floor", DEFAULT_ALPHA_FLOOR)))
    dialects = prior.get("dialects", {})
    if not isinstance(dialects, dict):
        return
    for entry in dialects.values():
        if not isinstance(entry, dict):
            continue
        families = entry.get("families", {})
        if not isinstance(families, dict):
            continue
        for family_entry in families.values():
            if not isinstance(family_entry, dict):
                continue
            seed_alpha = float(family_entry.get("seed_alpha", 0.0) or 0.0) * decay
            family_entry["seed_alpha"] = 0.0 if seed_alpha < floor else seed_alpha


def update_decoder_dialect_prior_state(
    prior: dict[str, Any],
    *,
    dialect_plan: dict[str, Any],
    steering: dict[str, Any],
    accepted: bool,
    case_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_decoder_dialect_prior_state(prior)
    prior.clear()
    prior.update(normalized)
    decay_decoder_dialect_prior(prior)
    prior["seen_cases"] = int(prior.get("seen_cases", 0) or 0) + 1
    prior["updated_at_utc"] = utc_now()
    if accepted:
        prior["accepted_cases"] = int(prior.get("accepted_cases", 0) or 0) + 1
    case_meta = dict(case_metadata or {})
    dialects = dialects_from_dialect_plan(dialect_plan)
    peak_by_family = steering.get("alpha", {}).get("peak_by_family", {})
    if not isinstance(peak_by_family, dict):
        peak_by_family = {}
    events = steering.get("events", [])
    if not isinstance(events, list):
        events = []
    event_count_by_family: dict[str, int] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        family = str(event.get("offense_family", ""))
        if family:
            event_count_by_family[family] = event_count_by_family.get(family, 0) + 1
    dialect_entries = prior.setdefault("dialects", {})
    max_seed_alpha = clamp_unit(prior.get("max_seed_alpha", DEFAULT_MAX_SEED_ALPHA))
    peak_to_seed = clamp_unit(prior.get("peak_to_seed", DEFAULT_PEAK_TO_SEED))
    for dialect in dialects:
        entry = dialect_entries.setdefault(
            dialect,
            {
                "seen_cases": 0,
                "accepted_case_count": 0,
                "successful_case_streak": 0,
                "families": {},
            },
        )
        entry["seen_cases"] = int(entry.get("seen_cases", 0) or 0) + 1
        entry["last_row_index"] = case_meta.get("row_index")
        entry["last_instance_id"] = str(case_meta.get("instance_id", ""))
        entry["last_repo"] = str(case_meta.get("repo", ""))
        entry["last_update_accepted"] = bool(accepted)
        entry["last_updated_at_utc"] = prior["updated_at_utc"]
        if accepted:
            entry["accepted_case_count"] = int(entry.get("accepted_case_count", 0) or 0) + 1
            entry["successful_case_streak"] = int(entry.get("successful_case_streak", 0) or 0) + 1
        else:
            entry["successful_case_streak"] = 0
        entry["hard_masked_candidate_count"] = int(
            entry.get("hard_masked_candidate_count", 0) or 0
        ) + int(steering.get("hard_masked_candidate_count", 0) or 0)
        entry["masked_but_not_tempting_count"] = int(
            entry.get("masked_but_not_tempting_count", 0) or 0
        ) + int(steering.get("masked_but_not_tempting_count", 0) or 0)
        entry["trigger_count"] = int(entry.get("trigger_count", 0) or 0) + int(
            steering.get("trigger_count", 0) or 0
        )
        entry["applied_count"] = int(entry.get("applied_count", 0) or 0) + int(
            steering.get("applied_count", 0) or 0
        )
        families = entry.setdefault("families", {})
        if not isinstance(families, dict):
            families = {}
            entry["families"] = families
        for family, peak in peak_by_family.items():
            family_key = str(family)
            peak_alpha = clamp_unit(peak)
            family_entry = families.setdefault(
                family_key,
                {
                    "peak_alpha": 0.0,
                    "seed_alpha": 0.0,
                    "retained_event_count": 0,
                },
            )
            family_entry["peak_alpha"] = max(
                float(family_entry.get("peak_alpha", 0.0) or 0.0),
                peak_alpha,
            )
            family_entry["last_peak_alpha"] = peak_alpha
            family_entry["retained_event_count"] = int(
                family_entry.get("retained_event_count", 0) or 0
            ) + int(event_count_by_family.get(family_key, 0))
            family_entry["last_updated_at_utc"] = prior["updated_at_utc"]
            if accepted and peak_alpha > 0.0:
                family_entry["seed_alpha"] = max(
                    float(family_entry.get("seed_alpha", 0.0) or 0.0),
                    min(max_seed_alpha, peak_alpha * peak_to_seed),
                )
    return prior


def load_decoder_dialect_prior_store(
    path: str | Path | None,
    *,
    scope: dict[str, Any] | DecoderPriorScope | None = None,
    config: dict[str, Any] | DecoderPriorConfig | None = None,
    reset: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None:
        prior = new_decoder_dialect_prior_state(scope=scope, config=config)
        return prior, {"active": False, "reason": "no_store_path", "loaded": False}
    store_path = Path(path)
    if reset or not store_path.exists():
        prior = new_decoder_dialect_prior_state(scope=scope, config=config)
        return prior, {
            "active": True,
            "path": str(store_path),
            "loaded": False,
            "reset": bool(reset),
            "reason": "reset_requested" if reset else "store_missing",
        }
    try:
        raw = json.loads(store_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - caller can continue with fresh memory.
        prior = new_decoder_dialect_prior_state(scope=scope, config=config)
        return prior, {
            "active": True,
            "path": str(store_path),
            "loaded": False,
            "reason": "load_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    compatible, mismatches = scopes_compatible(raw.get("scope", {}), scope)
    if not compatible:
        prior = new_decoder_dialect_prior_state(scope=scope, config=config)
        return prior, {
            "active": True,
            "path": str(store_path),
            "loaded": False,
            "reason": "scope_mismatch",
            "mismatched_scope_keys": mismatches,
            "existing_scope": raw.get("scope", {}),
            "expected_scope": scope.as_dict() if isinstance(scope, DecoderPriorScope) else dict(scope or {}),
        }
    prior = normalize_decoder_dialect_prior_state(raw, scope=scope, config=config)
    return prior, {
        "active": True,
        "path": str(store_path),
        "loaded": True,
        "reason": "loaded",
        "seen_cases": int(prior.get("seen_cases", 0) or 0),
        "accepted_cases": int(prior.get("accepted_cases", 0) or 0),
    }


def save_decoder_dialect_prior_store(
    path: str | Path | None,
    prior: dict[str, Any],
) -> dict[str, Any]:
    if path is None:
        return {"active": False, "saved": False, "reason": "no_store_path"}
    store_path = Path(path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_decoder_dialect_prior_state(prior)
    normalized["updated_at_utc"] = utc_now()
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{store_path.name}.",
        suffix=".tmp",
        dir=str(store_path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_safe(normalized), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, store_path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass
    return {
        "active": True,
        "saved": True,
        "path": str(store_path),
        "seen_cases": int(normalized.get("seen_cases", 0) or 0),
        "accepted_cases": int(normalized.get("accepted_cases", 0) or 0),
        "saved_at_unix": int(time.time()),
    }


__all__ = [
    "DEFAULT_ALPHA_FLOOR",
    "DEFAULT_CROSS_CASE_DECAY",
    "DEFAULT_MAX_SEED_ALPHA",
    "DEFAULT_MODE",
    "DEFAULT_PEAK_TO_SEED",
    "DEFAULT_STEERING_VERSION",
    "DecoderPriorConfig",
    "DecoderPriorScope",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "decoder_dialect_prior_plan",
    "decay_decoder_dialect_prior",
    "dialects_from_dialect_plan",
    "load_decoder_dialect_prior_store",
    "new_decoder_dialect_prior_state",
    "normalize_decoder_dialect_prior_state",
    "save_decoder_dialect_prior_store",
    "scopes_compatible",
    "update_decoder_dialect_prior_state",
]
