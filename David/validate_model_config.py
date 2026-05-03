#!/usr/bin/env python3
"""
Standalone validator for get_model_config.py reports.

This script is intentionally isolated from Lazarus runtime plumbing. It consumes
a model config calibration report, validates candidate layer geometry, and can
run a real tensor projection check against the model. The output is a validation
artifact for harness startup, not a registry mutation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_NAME = "lazarus.model_config_validation_report"
SCHEMA_VERSION = 1
CONFIG_REPORT_SCHEMA = "lazarus.model_config_calibration_report"

PROBES: list[dict[str, str]] = [
    {
        "name": "voltara",
        "doc": "Calibration fact: The archive code for Voltara is Dravenport.",
        "cloze_prefix": "Calibration fact: The archive code for Voltara is",
        "answer": "Dravenport",
    },
    {
        "name": "mirevale",
        "doc": "Calibration fact: The routing key for Mirevale is Quenlan.",
        "cloze_prefix": "Calibration fact: The routing key for Mirevale is",
        "answer": "Quenlan",
    },
    {
        "name": "selkirk",
        "doc": "Calibration fact: The memory token for Selkirk is Fendora.",
        "cloze_prefix": "Calibration fact: The memory token for Selkirk is",
        "answer": "Fendora",
    },
    {
        "name": "orris",
        "doc": "Calibration fact: The checkpoint label for Orris is Halvex.",
        "cloze_prefix": "Calibration fact: The checkpoint label for Orris is",
        "answer": "Halvex",
    },
]

BASE_PROJECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "q": ("q_proj", "query", "query_proj", "q", "c_attn_q"),
    "k": ("k_proj", "key", "key_proj", "k", "c_attn_k"),
    "v": ("v_proj", "value", "value_proj", "v", "c_attn_v"),
    "o": ("o_proj", "out_proj", "dense", "output_proj", "proj", "c_proj"),
}

FAMILY_PROJECTION_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "gemma": {"o": ("o_proj", "out_proj", "dense")},
    "gemma3": {"o": ("o_proj", "out_proj", "dense")},
    "gemma4": {"o": ("o_proj", "out_proj", "dense")},
    "qwen": {"o": ("o_proj", "out_proj", "dense")},
    "qwen3": {"o": ("o_proj", "out_proj", "dense")},
    "llama": {"o": ("o_proj", "out_proj")},
    "smollm": {"o": ("o_proj", "out_proj")},
}


def eprint(*parts: Any) -> None:
    print(*parts, file=sys.stderr, flush=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            return str(value)
    return value


def section(status: str, reason: str | None = None, **fields: Any) -> dict[str, Any]:
    out = {"status": status}
    if reason:
        out["reason"] = reason
    out.update(fields)
    return json_safe(out)


def iso_utc(timestamp: float | None = None) -> str:
    moment = datetime.fromtimestamp(time.time() if timestamp is None else timestamp, timezone.utc)
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def adapter_family_from_model_type(*values: Any) -> str:
    text = " ".join(str(v or "").lower() for v in values)
    if "gemma4" in text or "gemma-4" in text:
        return "gemma4"
    if "gemma3" in text or "gemma-3" in text:
        return "gemma3"
    if "gemma" in text:
        return "gemma"
    if "qwen3" in text or "qwen3_5" in text or "qwen3.5" in text:
        return "qwen3"
    if "qwen" in text:
        return "qwen"
    if "smollm" in text:
        return "smollm"
    if "llama" in text:
        return "llama"
    return "fallback"


def projection_aliases(adapter_family: str, role: str) -> tuple[str, ...]:
    aliases: list[str] = []
    for alias in FAMILY_PROJECTION_ALIASES.get(adapter_family, {}).get(role, ()):
        if alias not in aliases:
            aliases.append(alias)
    for alias in BASE_PROJECTION_ALIASES.get(role, ()):
        if alias not in aliases:
            aliases.append(alias)
    return tuple(aliases)


def resolve_attr(root: Any, dotted: str) -> Any | None:
    current = root
    for part in dotted.split("."):
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def resolve_attention_projection(
    attn: Any, role: str, adapter_family: str
) -> tuple[Any | None, str | None, tuple[str, ...]]:
    checked = projection_aliases(adapter_family, role)
    for alias in checked:
        if hasattr(attn, alias):
            value = getattr(attn, alias)
            if value is not None:
                return value, alias, checked
    return None, None, checked


def projection_like_attrs(attn: Any) -> list[str]:
    if attn is None:
        return []
    tokens = ("proj", "projection", "query", "key", "value", "dense", "attn")
    attrs = []
    for name in dir(attn):
        low = name.lower()
        if not name.startswith("_") and any(token in low for token in tokens):
            attrs.append(name)
    return sorted(attrs)


def get_text_config(config: Any) -> Any:
    if config is None:
        return None
    if hasattr(config, "get_text_config"):
        try:
            return config.get_text_config()
        except Exception:
            pass
    return getattr(config, "text_config", None) or config


def config_value(config: Any, names: tuple[str, ...], default: Any = None) -> Any:
    text_config = get_text_config(config)
    for source in (text_config, config):
        if source is None:
            continue
        for name in names:
            if hasattr(source, name):
                value = getattr(source, name)
                if value is not None:
                    return value
    return default


def resolve_layers(model: Any) -> list[Any]:
    paths = (
        "model.layers",
        "model.decoder.layers",
        "model.transformer.h",
        "model.gpt_neox.layers",
        "model.language_model.layers",
        "model.language_model.model.layers",
        "model.text_model.layers",
        "model.text_model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "decoder.layers",
        "layers",
    )
    for path in paths:
        layers = resolve_attr(model, path)
        if layers is None:
            continue
        try:
            layers_list = list(layers)
        except TypeError:
            continue
        if layers_list:
            return layers_list
    raise RuntimeError("Could not resolve decoder layers for validation.")


def resolve_self_attn(layer: Any) -> Any | None:
    for name in ("self_attn", "attention", "attn", "self_attention"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def module_out_features(module: Any) -> int | None:
    value = getattr(module, "out_features", None)
    if value is not None:
        return int(value)
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape and len(shape) >= 1:
        return int(shape[0])
    return None


def module_in_features(module: Any) -> int | None:
    value = getattr(module, "in_features", None)
    if value is not None:
        return int(value)
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape and len(shape) >= 2:
        return int(shape[1])
    return None


def encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    limit = len(haystack) - len(needle) + 1
    for idx in range(max(0, limit)):
        if haystack[idx : idx + len(needle)] == needle:
            return idx
    return None


def answer_tokens(tokenizer: Any, answer: str) -> list[int]:
    for variant in (f" {answer}", answer):
        ids = encode(tokenizer, variant, add_special_tokens=False)
        if ids:
            return ids
    raise RuntimeError(f"Could not tokenize answer {answer!r}")


def locate_answer(tokenizer: Any, doc: str, answer: str) -> tuple[list[int], int, int]:
    doc_ids = encode(tokenizer, doc, add_special_tokens=True)
    for variant in (f" {answer}", answer, f"{answer}."):
        ids = encode(tokenizer, variant, add_special_tokens=False)
        start = find_subsequence(doc_ids, ids)
        if start is not None and ids:
            return doc_ids, start, ids[0]
    answer_first = answer_tokens(tokenizer, answer)[0]
    for idx, token_id in enumerate(doc_ids):
        if int(token_id) == int(answer_first):
            return doc_ids, idx, answer_first
    raise RuntimeError(f"Could not locate answer {answer!r} in tokenized probe doc.")


def run_model(model: Any, torch: Any, input_ids: Any, attention_mask: Any | None = None) -> Any:
    kwargs = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    try:
        kwargs["use_cache"] = False
        return model(**kwargs)
    except TypeError:
        kwargs.pop("use_cache", None)
        return model(**kwargs)


def capture_layer_vectors(
    model: Any,
    torch: Any,
    layers: list[Any],
    token_ids: list[int],
    target_pos: int,
    layer_indices: list[int],
    device: str,
) -> dict[int, Any]:
    captures: dict[int, Any] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden is None:
                return
            pos = min(target_pos, int(hidden.shape[1]) - 1)
            captures[layer_idx] = hidden[0, pos, :].detach().to("cpu", dtype=torch.float32)

        return hook

    for layer_idx in layer_indices:
        handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
    try:
        input_ids = torch.as_tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            run_model(model, torch, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    return captures


def tensor_stats(torch: Any, tensor: Any) -> dict[str, Any]:
    flat = tensor.detach().to(dtype=torch.float32)
    return {
        "shape": [int(v) for v in tensor.shape],
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "finite": bool(torch.isfinite(flat).all().item()),
        "norm": float(torch.linalg.vector_norm(flat).item()),
        "mean_abs": float(flat.abs().mean().item()),
        "zero_fraction": float((flat == 0).to(dtype=torch.float32).mean().item()),
    }


def read_report(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def candidate_key(candidate: dict[str, Any]) -> tuple[int | None, int | None]:
    source = candidate.get("kv_source_layer")
    target = candidate.get("kv_target_layer")
    return (int(source) if source is not None else None, int(target) if target is not None else None)


def topology_layer_info(report: dict[str, Any], layer_idx: int | None) -> dict[str, Any] | None:
    if layer_idx is None:
        return None
    for info in (report.get("topology") or {}).get("layers") or []:
        if info.get("layer") == layer_idx:
            return info
    return None


def behavior_cache_layer_for_candidate(report: dict[str, Any], candidate: dict[str, Any]) -> int | None:
    target = candidate.get("kv_target_layer")
    if target is None:
        return None
    target = int(target)
    target_info = topology_layer_info(report, target)
    if target_info and target_info.get("kv_shared_layer_index") is not None:
        return int(target_info["kv_shared_layer_index"])
    return int(candidate.get("projection_producer_layer", target))


def normalize_candidate(
    report: dict[str, Any],
    candidate: dict[str, Any],
    role: str,
    rank: int = 0,
) -> dict[str, Any]:
    out = dict(candidate)
    out["candidate_role"] = role
    out["candidate_rank"] = rank
    if out.get("boundary_layer") is None and out.get("kv_source_layer") is not None:
        out["boundary_layer"] = int(out["kv_source_layer"])
    if out.get("injection_layer") is None and out.get("kv_target_layer") is not None:
        out["injection_layer"] = int(out["kv_target_layer"])
    if out.get("behavior_cache_layer") is None:
        out["behavior_cache_layer"] = behavior_cache_layer_for_candidate(report, out)
    return out


def collect_candidates(report: dict[str, Any], max_candidates: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int | None, int | None]] = set()

    def add(candidate: dict[str, Any] | None, role: str, rank: int = 0) -> None:
        if not candidate:
            return
        if candidate.get("kv_source_layer") is None or candidate.get("kv_target_layer") is None:
            return
        normalized = normalize_candidate(report, candidate, role, rank)
        key = candidate_key(normalized)
        if key in seen:
            for item in candidates:
                if candidate_key(item) == key and role not in item["candidate_role"].split("+"):
                    item["candidate_role"] += f"+{role}"
                    for field in (
                        "candidate_score",
                        "materialization_safe",
                        "projection_producer_layer",
                        "behavior_cache_layer",
                        "source_avg_dot",
                        "target_avg_dot",
                        "projection_status",
                    ):
                        if item.get(field) is None and normalized.get(field) is not None:
                            item[field] = normalized[field]
                    if normalized.get("candidate_score") is not None:
                        current_score = item.get("candidate_score")
                        if current_score is None or float(normalized["candidate_score"]) > float(current_score):
                            item["candidate_score"] = normalized["candidate_score"]
            return
        seen.add(key)
        candidates.append(normalized)

    adapter_candidate = report.get("adapter_config_candidate") or {}
    add(adapter_candidate, "adapter_config_candidate", 0)
    recommended = report.get("recommended") or {}
    add(recommended, "recommended", 0)

    legacy = report.get("legacy_vec_inject") or {}
    if legacy.get("legacy_retrieval_layer") is not None and legacy.get("legacy_injection_layer") is not None:
        add(
            {
                "kv_source_layer": int(legacy["legacy_retrieval_layer"]),
                "kv_target_layer": int(legacy["legacy_injection_layer"]),
                "boundary_layer": int(legacy["legacy_retrieval_layer"]),
                "injection_layer": int(legacy["legacy_injection_layer"]),
                "insertion_family": "legacy_vec_inject_candidate",
                "candidate_score": None,
                "materialization_safe": True,
            },
            "legacy_vec_inject",
            0,
        )

    for idx, candidate in enumerate(report.get("kv_candidates") or []):
        if len(candidates) >= max_candidates:
            break
        add(candidate, f"kv_candidate_{idx}", idx)

    return candidates[:max_candidates]


def report_model_path(report: dict[str, Any]) -> str | None:
    for section_name in ("adapter_config_candidate", "model"):
        section_data = report.get(section_name) or {}
        value = section_data.get("model")
        if value:
            return str(value)
    return None


def validate_report_integrity(report: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    warnings = []
    if report.get("schema_name") != CONFIG_REPORT_SCHEMA:
        warnings.append(f"Unexpected report schema_name {report.get('schema_name')!r}.")
    if not candidates:
        warnings.append("No candidate layers found in config report.")
    status = "ok" if not warnings else "needs_review"
    return section(status, warnings=warnings, candidate_count=len(candidates))


def validate_topology(report: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    topology = report.get("topology") or {}
    num_layers = topology.get("num_layers")
    warnings = []
    results = []
    if num_layers is None:
        return section("unavailable", "config report has no topology.num_layers")
    for candidate in candidates:
        source, target = candidate_key(candidate)
        reasons = []
        if source is None or target is None:
            reasons.append("missing source or target layer")
        else:
            if not (0 <= source < int(num_layers)):
                reasons.append("source layer out of range")
            if not (0 <= target < int(num_layers)):
                reasons.append("target layer out of range")
            if target <= source:
                reasons.append("target layer must be after source layer")
        results.append(
            {
                "candidate_role": candidate.get("candidate_role"),
                "kv_source_layer": source,
                "kv_target_layer": target,
                "status": "ok" if not reasons else "failed",
                "reasons": reasons,
            }
        )
        warnings.extend(f"{candidate.get('candidate_role')}: {reason}" for reason in reasons)
    return section("ok" if not warnings else "failed", warnings=warnings, results=results)


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_dtype(torch: Any, requested: str, device: str) -> tuple[Any, str]:
    if requested == "float32" or device == "cpu":
        return torch.float32, "float32"
    if requested == "float16":
        return torch.float16, "float16"
    if requested == "bfloat16":
        return torch.bfloat16, "bfloat16"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bfloat16"
    if device == "cuda":
        return torch.float16, "float16"
    return torch.float32, "float32"


def load_model_and_tokenizer(args: argparse.Namespace, model_path: str) -> tuple[Any, Any, Any, str, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = choose_device(torch, args.device)
    dtype, dtype_name = choose_dtype(torch, args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "dtype": dtype,
        "local_files_only": bool(args.local_files_only),
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.attn_implementation != "default":
        model_kwargs["attn_implementation"] = args.attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except TypeError:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    if device != "cpu":
        model.to(device)
    return torch, model, tokenizer, device, dtype_name


def validate_model_identity(report: dict[str, Any], model: Any, model_path: str, device: str, dtype_name: str) -> dict[str, Any]:
    config = getattr(model, "config", None)
    text_config = get_text_config(config)
    family = adapter_family_from_model_type(
        getattr(config, "model_type", None),
        getattr(text_config, "model_type", None),
        report.get("model", {}).get("adapter_family"),
    )
    report_model = report.get("model") or {}
    warnings = []
    reported_revision = report_model.get("model_revision_or_hash")
    actual_revision = None
    path = Path(model_path)
    if path.exists() and path.parent.name == "snapshots":
        actual_revision = path.name
    if reported_revision and actual_revision and reported_revision != actual_revision:
        warnings.append("model revision/hash differs from config report")
    return section(
        "ok" if not warnings else "needs_review",
        warnings=warnings,
        adapter_family=family,
        wrapper_model_type=getattr(config, "model_type", None),
        text_model_type=getattr(text_config, "model_type", None),
        model_revision_or_hash=actual_revision,
        hidden_size=config_value(config, ("hidden_size", "n_embd", "d_model")),
        num_attention_heads=config_value(config, ("num_attention_heads", "n_head", "num_heads")),
        num_key_value_heads=config_value(config, ("num_key_value_heads", "num_kv_heads")),
        device=device,
        dtype=dtype_name,
    )


def validate_projection_gate(
    args: argparse.Namespace,
    report: dict[str, Any],
    candidates: list[dict[str, Any]],
    torch: Any,
    model: Any,
    tokenizer: Any,
    device: str,
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    text_config = get_text_config(config)
    adapter_family = adapter_family_from_model_type(
        getattr(config, "model_type", None),
        getattr(text_config, "model_type", None),
        report.get("model", {}).get("adapter_family"),
    )
    layers = resolve_layers(model)
    probes = PROBES[: max(1, int(args.max_probes))]
    source_layers = sorted({int(c["kv_source_layer"]) for c in candidates if c.get("kv_source_layer") is not None})
    captures_by_source: dict[int, Any] = {}
    probe_status = []

    for probe in probes:
        try:
            token_ids, answer_pos, _answer_token_id = locate_answer(tokenizer, probe["doc"], probe["answer"])
            target_pos = max(0, int(answer_pos) - 1)
            captures = capture_layer_vectors(model, torch, layers, token_ids, target_pos, source_layers, device)
            for layer_idx, vector in captures.items():
                captures_by_source.setdefault(layer_idx, vector)
            probe_status.append({"name": probe["name"], "status": "used"})
        except Exception as exc:
            probe_status.append({"name": probe["name"], "status": "skipped", "reason": str(exc)})

    first_param = next(model.parameters())
    model_dtype = first_param.dtype
    candidate_results = []
    warnings = []
    for candidate in candidates:
        source, target = candidate_key(candidate)
        role = candidate.get("candidate_role")
        if source is None or target is None or target >= len(layers):
            result = section("failed", "candidate layer ids invalid", candidate_role=role)
            candidate_results.append(result)
            warnings.append(f"{role}: invalid layer ids")
            continue
        residual = captures_by_source.get(source)
        if residual is None:
            result = section("failed", "no captured residual for source layer", candidate_role=role)
            candidate_results.append(result)
            warnings.append(f"{role}: no captured residual for L{source}")
            continue
        producer_idx = candidate.get("projection_producer_layer")
        if producer_idx is None:
            producer_idx = target
        producer_layer = layers[int(producer_idx)]
        target_layer = layers[target]
        attn = resolve_self_attn(producer_layer)
        if attn is None:
            result = section(
                "failed",
                "producer layer has no attention module",
                candidate_role=role,
                producer_layer=producer_idx,
            )
            candidate_results.append(result)
            warnings.append(f"{role}: producer layer has no attention")
            continue
        k_proj, k_alias, k_checked = resolve_attention_projection(attn, "k", adapter_family)
        v_proj, v_alias, v_checked = resolve_attention_projection(attn, "v", adapter_family)
        if k_proj is None or v_proj is None:
            result = section(
                "failed",
                "producer layer lacks K/V projection module",
                candidate_role=role,
                available_projection_like_attrs=projection_like_attrs(attn),
                aliases_checked={"k": list(k_checked), "v": list(v_checked)},
            )
            candidate_results.append(result)
            warnings.append(f"{role}: missing K/V projections")
            continue

        residual_tensor = residual.to(device=device, dtype=model_dtype, non_blocking=True).reshape(1, 1, -1)
        variants = [("raw_residual", residual_tensor)]
        input_layernorm = getattr(target_layer, "input_layernorm", None)
        if input_layernorm is not None:
            with torch.inference_mode():
                variants.append(("input_layernorm_then_kv_proj", input_layernorm(residual_tensor)))

        projection_scores: dict[str, Any] = {}
        statuses = []
        with torch.inference_mode():
            for variant_name, variant_residual in variants:
                k_flat = k_proj(variant_residual)
                v_flat = v_proj(variant_residual)
                k_stats = tensor_stats(torch, k_flat)
                v_stats = tensor_stats(torch, v_flat)
                ok = bool(k_stats["finite"] and v_stats["finite"] and k_stats["norm"] > 0 and v_stats["norm"] > 0)
                statuses.append(ok)
                projection_scores[variant_name] = {"K": k_stats, "V": v_stats, "ok": ok}

        discovery_score = candidate.get("candidate_score")
        validation_score = 0.0
        if any(statuses):
            validation_score += 10.0
        if candidate.get("materialization_safe") is not False:
            validation_score += 1.0
        if discovery_score is not None:
            validation_score += min(5.0, max(-5.0, float(discovery_score)))
        if "adapter_config_candidate" in str(role) or "recommended" in str(role):
            validation_score += 0.5
        if "legacy_vec_inject" in str(role):
            validation_score += 0.25

        result = section(
            "ok" if any(statuses) else "failed",
            candidate_role=role,
            kv_source_layer=source,
            kv_target_layer=target,
            injection_layer=target,
            boundary_layer=candidate.get("boundary_layer", source),
            projection_producer_layer=int(producer_idx),
            behavior_cache_layer=candidate.get("behavior_cache_layer", target),
            projection_aliases={"k": k_alias, "v": v_alias},
            projection_scores=projection_scores,
            discovery_score=discovery_score,
            validation_score=float(validation_score),
        )
        candidate_results.append(result)
        if result["status"] != "ok":
            warnings.append(f"{role}: projection check failed")

    passed = [item for item in candidate_results if item.get("status") == "ok"]
    if not passed:
        status = "failed"
        reason = "no candidate passed projection validation"
    else:
        status = "ok"
        reason = None
    ranked = sorted(passed, key=lambda item: float(item.get("validation_score") or 0.0), reverse=True)
    return section(
        status,
        reason,
        adapter_family=adapter_family,
        probes=probe_status,
        candidates=candidate_results,
        ranked_candidates=ranked,
        warnings=warnings,
    )


def extract_logits_and_cache(outputs: Any) -> tuple[Any, Any]:
    if hasattr(outputs, "logits"):
        return outputs.logits, getattr(outputs, "past_key_values", None)
    if isinstance(outputs, tuple) and outputs:
        logits = outputs[0]
        cache = outputs[1] if len(outputs) > 1 else None
        return logits, cache
    raise RuntimeError("Model outputs did not contain logits.")


def cache_seq_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        seq_len = past_key_values.get_seq_length()
        if seq_len is not None:
            return int(seq_len)
    if hasattr(past_key_values, "key_cache") and past_key_values.key_cache:
        first = past_key_values.key_cache[0]
        if hasattr(first, "shape") and len(first.shape) >= 3:
            return int(first.shape[-2])
    if hasattr(past_key_values, "layers") and past_key_values.layers:
        layer = past_key_values.layers[0]
        for attr in ("keys", "key_cache", "k"):
            value = getattr(layer, attr, None)
            if hasattr(value, "shape") and len(value.shape) >= 3:
                return int(value.shape[-2])
    if isinstance(past_key_values, (list, tuple)) and past_key_values:
        first_layer = past_key_values[0]
        if isinstance(first_layer, (list, tuple)) and first_layer:
            first_tensor = first_layer[0]
            if hasattr(first_tensor, "shape") and len(first_tensor.shape) >= 3:
                return int(first_tensor.shape[-2])
    return 0


def clone_cache(past_key_values: Any, torch: Any) -> Any:
    if past_key_values is None:
        return None
    if torch.is_tensor(past_key_values):
        return past_key_values.clone()
    if isinstance(past_key_values, tuple):
        return tuple(clone_cache(item, torch) for item in past_key_values)
    if isinstance(past_key_values, list):
        return [clone_cache(item, torch) for item in past_key_values]
    if isinstance(past_key_values, dict):
        return {key: clone_cache(value, torch) for key, value in past_key_values.items()}
    try:
        return copy.deepcopy(past_key_values)
    except Exception:
        return past_key_values


def zero_cache_value(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return torch.zeros_like(value)
    if isinstance(value, tuple):
        return tuple(zero_cache_value(item, torch) for item in value)
    if isinstance(value, list):
        return [zero_cache_value(item, torch) for item in value]
    if isinstance(value, dict):
        return {key: zero_cache_value(item, torch) for key, item in value.items()}
    for attr in ("keys", "values", "key_cache", "value_cache", "k", "v"):
        if hasattr(value, attr):
            current = getattr(value, attr)
            try:
                setattr(value, attr, zero_cache_value(current, torch))
            except Exception:
                pass
    return value


def ablate_cache_layer(past_key_values: Any, layer_idx: int, torch: Any) -> tuple[Any, str]:
    cache = clone_cache(past_key_values, torch)
    if cache is None:
        raise RuntimeError("Cannot ablate a missing KV cache.")
    if hasattr(cache, "layers") and len(cache.layers) > layer_idx:
        cache.layers[layer_idx] = zero_cache_value(cache.layers[layer_idx], torch)
        return cache, "cache.layers"
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        if len(cache.key_cache) > layer_idx and len(cache.value_cache) > layer_idx:
            cache.key_cache[layer_idx] = zero_cache_value(cache.key_cache[layer_idx], torch)
            cache.value_cache[layer_idx] = zero_cache_value(cache.value_cache[layer_idx], torch)
            return cache, "cache.key_cache/value_cache"
    if isinstance(cache, tuple) and len(cache) > layer_idx:
        layers = list(cache)
        layers[layer_idx] = zero_cache_value(layers[layer_idx], torch)
        return tuple(layers), "tuple_layer"
    if isinstance(cache, list) and len(cache) > layer_idx:
        cache[layer_idx] = zero_cache_value(cache[layer_idx], torch)
        return cache, "list_layer"
    raise RuntimeError(f"Unsupported cache type for layer ablation: {type(cache).__name__}")


def answer_probability_from_logits(torch: Any, logits: Any, answer_token_id: int) -> float:
    probs = torch.softmax(logits[0, -1, :].to(dtype=torch.float32), dim=-1)
    return float(probs[int(answer_token_id)].item())


def forward_answer_probability(
    model: Any,
    torch: Any,
    input_ids: Any,
    attention_mask: Any,
    answer_token_id: int,
    *,
    past_key_values: Any | None = None,
    cache_len: int = 0,
) -> float:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
        kwargs["use_cache"] = True
        positions = torch.arange(
            cache_len,
            cache_len + int(input_ids.shape[1]),
            device=input_ids.device,
            dtype=torch.long,
        )
        kwargs["position_ids"] = positions.view(1, -1)
    with torch.inference_mode():
        outputs = model(**kwargs)
        logits, _cache = extract_logits_and_cache(outputs)
    return answer_probability_from_logits(torch, logits, answer_token_id)


def prefill_memory_cache(model: Any, torch: Any, memory_ids: Any, attention_mask: Any) -> Any:
    with torch.inference_mode():
        outputs = model(
            input_ids=memory_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    _logits, cache = extract_logits_and_cache(outputs)
    if cache is None:
        raise RuntimeError("Model did not return past_key_values during memory prefill.")
    return cache


def validate_behavior_gate(
    args: argparse.Namespace,
    report: dict[str, Any],
    projection: dict[str, Any],
    torch: Any,
    model: Any,
    tokenizer: Any,
    device: str,
) -> dict[str, Any]:
    ranked = projection.get("ranked_candidates") or []
    if not ranked:
        return section("skipped", "projection gate produced no ranked candidates")
    candidates = ranked[: max(1, min(int(args.max_behavior_candidates), len(ranked)))]
    probes = PROBES[: max(1, int(args.max_behavior_probes))]
    candidate_results: dict[str, dict[str, Any]] = {
        str(candidate.get("candidate_role")): {
            "candidate_role": candidate.get("candidate_role"),
            "kv_source_layer": candidate.get("kv_source_layer"),
            "kv_target_layer": candidate.get("kv_target_layer"),
            "injection_layer": candidate.get("injection_layer", candidate.get("kv_target_layer")),
            "boundary_layer": candidate.get("boundary_layer", candidate.get("kv_source_layer")),
            "behavior_cache_layer": candidate.get("behavior_cache_layer", candidate.get("kv_target_layer")),
            "insertion_family": candidate.get("insertion_family"),
            "probe_results": [],
            "mean_ablation_drop": 0.0,
            "mean_full_delta": 0.0,
            "behavior_score": 0.0,
        }
        for candidate in candidates
    }
    warnings: list[str] = []
    probe_summaries = []

    for probe in probes:
        try:
            answer_id = int(answer_tokens(tokenizer, probe["answer"])[0])
            query_ids = torch.as_tensor(
                [encode(tokenizer, probe["cloze_prefix"], add_special_tokens=True)],
                dtype=torch.long,
                device=device,
            )
            query_mask = torch.ones_like(query_ids, dtype=torch.long, device=device)
            baseline_prob = forward_answer_probability(
                model,
                torch,
                query_ids,
                query_mask,
                answer_id,
            )

            memory_text = probe["doc"] + "\n\n"
            memory_ids = torch.as_tensor(
                [encode(tokenizer, memory_text, add_special_tokens=True)],
                dtype=torch.long,
                device=device,
            )
            memory_mask = torch.ones_like(memory_ids, dtype=torch.long, device=device)
            memory_cache = prefill_memory_cache(model, torch, memory_ids, memory_mask)
            prefix_len = cache_seq_length(memory_cache) or int(memory_ids.shape[1])
            cached_attention_mask = torch.ones(
                (1, prefix_len + int(query_ids.shape[1])),
                dtype=torch.long,
                device=device,
            )
            full_prob = forward_answer_probability(
                model,
                torch,
                query_ids,
                cached_attention_mask,
                answer_id,
                past_key_values=clone_cache(memory_cache, torch),
                cache_len=prefix_len,
            )
            full_delta = float(full_prob - baseline_prob)
            probe_summaries.append(
                {
                    "name": probe["name"],
                    "baseline_answer_probability": baseline_prob,
                    "full_cache_answer_probability": full_prob,
                    "full_cache_delta": full_delta,
                }
            )

            for candidate in candidates:
                role = str(candidate.get("candidate_role"))
                layer = int(candidate.get("behavior_cache_layer", candidate.get("kv_target_layer")))
                try:
                    ablated_cache, ablation_strategy = ablate_cache_layer(memory_cache, layer, torch)
                    ablated_prob = forward_answer_probability(
                        model,
                        torch,
                        query_ids,
                        cached_attention_mask,
                        answer_id,
                        past_key_values=ablated_cache,
                        cache_len=prefix_len,
                    )
                    ablation_drop = float(full_prob - ablated_prob)
                    candidate_results[role]["probe_results"].append(
                        {
                            "name": probe["name"],
                            "baseline_answer_probability": baseline_prob,
                            "full_cache_answer_probability": full_prob,
                            "ablated_answer_probability": ablated_prob,
                            "full_cache_delta": full_delta,
                            "ablation_drop": ablation_drop,
                            "ablation_strategy": ablation_strategy,
                            "behavior_cache_layer": layer,
                        }
                    )
                except Exception as exc:
                    candidate_results[role]["probe_results"].append(
                        {"name": probe["name"], "status": "failed", "reason": str(exc)}
                    )
                    warnings.append(f"{role}/{probe['name']}: {exc}")
        except Exception as exc:
            warnings.append(f"{probe['name']}: behavioral probe failed: {exc}")

    ranked_behavior = []
    for role, result in candidate_results.items():
        used = [probe for probe in result["probe_results"] if probe.get("ablation_drop") is not None]
        if used:
            mean_drop = sum(float(probe["ablation_drop"]) for probe in used) / len(used)
            mean_delta = sum(float(probe["full_cache_delta"]) for probe in used) / len(used)
            pass_rate = sum(
                1
                for probe in used
                if float(probe["full_cache_delta"]) >= float(args.min_behavior_delta)
                and float(probe["ablation_drop"]) >= float(args.min_ablation_drop)
            ) / len(used)
            score = mean_delta + mean_drop + pass_rate
            result.update(
                {
                    "status": "ok" if pass_rate > 0 else "needs_review",
                    "mean_ablation_drop": float(mean_drop),
                    "mean_full_delta": float(mean_delta),
                    "pass_rate": float(pass_rate),
                    "behavior_score": float(score),
                    "used_probe_count": len(used),
                }
            )
        else:
            result.update(
                {
                    "status": "failed",
                    "reason": "no behavioral probes completed for candidate",
                    "behavior_score": 0.0,
                    "used_probe_count": 0,
                }
            )
        ranked_behavior.append(result)

    ranked_behavior.sort(key=lambda item: float(item.get("behavior_score") or 0.0), reverse=True)
    best = ranked_behavior[0] if ranked_behavior else None
    if not best or best.get("status") == "failed":
        return section(
            "failed",
            "no candidate passed behavioral KV validation",
            probes=probe_summaries,
            candidates=ranked_behavior,
            warnings=warnings,
        )

    status = "ok" if best.get("pass_rate", 0.0) > 0 else "needs_review"
    if best.get("mean_full_delta", 0.0) < float(args.min_behavior_delta):
        status = "needs_review"
        warnings.append("best candidate full-cache improvement is below threshold")
    if best.get("mean_ablation_drop", 0.0) < float(args.min_ablation_drop):
        status = "needs_review"
        warnings.append("best candidate ablation drop is below threshold")

    return section(
        status,
        mode="prefix_cache_layer_ablation",
        probes=probe_summaries,
        candidates=ranked_behavior,
        selected_candidate=best,
        warnings=list(dict.fromkeys(warnings)),
        thresholds={
            "min_behavior_delta": float(args.min_behavior_delta),
            "min_ablation_drop": float(args.min_ablation_drop),
        },
    )


def decide_validation(
    report: dict[str, Any],
    integrity: dict[str, Any],
    topology: dict[str, Any],
    projection: dict[str, Any],
    behavior: dict[str, Any],
    *,
    require_behavioral_kv: bool,
    min_validation_margin: float,
) -> dict[str, Any]:
    warnings: list[str] = []
    warnings.extend(integrity.get("warnings") or [])
    warnings.extend(topology.get("warnings") or [])
    warnings.extend(projection.get("warnings") or [])
    warnings.extend(behavior.get("warnings") or [])
    behavior_selected = behavior.get("selected_candidate") or {}
    ranked = behavior.get("candidates") or projection.get("ranked_candidates") or []
    selected = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None
    margin = None
    if selected and second:
        score_key = "behavior_score" if selected.get("behavior_score") is not None else "validation_score"
        margin = float(selected.get(score_key) or 0.0) - float(second.get(score_key) or 0.0)
    elif selected:
        margin = None

    if require_behavioral_kv:
        if behavior.get("status") != "ok":
            warnings.append("behavioral KV validation was required but did not pass")

    if topology.get("status") == "failed" or projection.get("status") == "failed":
        validation_status = "rejected"
        confidence = "low"
    elif behavior.get("status") == "failed":
        validation_status = "rejected"
        confidence = "low"
    elif require_behavioral_kv and behavior.get("status") != "ok":
        validation_status = "needs_review"
        confidence = "low"
    elif selected is None:
        validation_status = "needs_review"
        confidence = "low"
    elif margin is not None and margin < min_validation_margin:
        validation_status = "needs_review"
        confidence = "low"
        warnings.append(f"validation score margin {margin:.3f} is below threshold {min_validation_margin}")
    else:
        validation_status = "accepted"
        confidence = "high" if behavior.get("status") == "ok" else "medium"

    discovery_review = report.get("recommendation_review") or {}
    discovery_status = discovery_review.get("status")
    if discovery_status in {"needs_review", "review_required"}:
        warnings.append("source config report was review_required; validator selected by validation evidence")
        if validation_status == "accepted":
            confidence = "high" if behavior.get("status") == "ok" else "medium"

    auto_load_allowed = validation_status == "accepted" and behavior.get("status") == "ok" and confidence == "high"
    harness_load_policy = "allow_auto_load" if auto_load_allowed else "manual_or_validation_required"
    selected_config = None
    if selected is not None:
        role = selected.get("candidate_role") or behavior_selected.get("candidate_role")
        selected_config = {
            "route_layer": (report.get("recommended") or {}).get("route_layer"),
            "route_query_head": (report.get("recommended") or {}).get("route_query_head"),
            "boundary_layer": selected.get("boundary_layer"),
            "kv_source_layer": selected.get("kv_source_layer"),
            "kv_target_layer": selected.get("kv_target_layer"),
            "injection_layer": selected.get("injection_layer", selected.get("kv_target_layer")),
            "behavior_cache_layer": selected.get("behavior_cache_layer"),
            "insertion_family": (report.get("recommended") or {}).get("insertion_family"),
            "candidate_role": role,
        }

    return section(
        validation_status,
        confidence=confidence,
        validation_level="topology_projection_behavior",
        auto_load_allowed=auto_load_allowed,
        harness_load_policy=harness_load_policy,
        selected_config=selected_config,
        validation_score_margin=margin,
        behavior_gate=behavior,
        warnings=list(dict.fromkeys(warnings)),
    )


def provenance(args: argparse.Namespace, started: float, model_path: str | None) -> dict[str, Any]:
    return json_safe(
        {
            "generated_at_unix": int(started),
            "generated_at_utc": iso_utc(started),
            "tool": Path(__file__).as_posix(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "argv": sys.argv[1:],
            "config_report": args.config_report,
            "model": model_path,
            "dry_run": bool(args.dry_run),
            "skip_behavioral_kv": bool(args.skip_behavioral_kv),
            "require_behavioral_kv": bool(args.require_behavioral_kv),
            "max_behavior_probes": int(args.max_behavior_probes),
            "max_behavior_candidates": int(args.max_behavior_candidates),
        }
    )


def emit(report: dict[str, Any], json_out: str | None) -> None:
    text = json.dumps(json_safe(report), indent=2, sort_keys=True)
    print(text)
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(text + "\n", encoding="utf-8")
        eprint(f"Wrote report to {json_out}")


def failure_report(args: argparse.Namespace, started: float, stage: str, error: BaseException | str) -> dict[str, Any]:
    error_text = str(error)
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "validation_status": "failed",
        "confidence": "low",
        "provenance": provenance(args, started, args.model),
        "failure": section(
            "failed",
            error_text,
            stage=stage,
            error_type=error.__class__.__name__ if isinstance(error, BaseException) else "Error",
        ),
        "warnings": [f"{stage}: {error_text}"],
        "candidate_results": [],
        "selected_config": None,
        "auto_load_allowed": False,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a get_model_config.py report before harness adapter loading."
    )
    parser.add_argument("--config-report", required=True, help="Path to get_model_config.py JSON report")
    parser.add_argument("--model", default=None, help="Model path/id. Defaults to report model field.")
    parser.add_argument("--json-out", default=None, help="Optional JSON validation report path")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, cuda:0, etc.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype. CPU forces float32.",
    )
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate report/topology only; do not load model.")
    parser.add_argument("--max-probes", type=int, default=len(PROBES))
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--min-validation-margin", type=float, default=0.25)
    parser.add_argument("--max-behavior-probes", type=int, default=2)
    parser.add_argument("--max-behavior-candidates", type=int, default=4)
    parser.add_argument("--min-behavior-delta", type=float, default=0.01)
    parser.add_argument("--min-ablation-drop", type=float, default=0.001)
    parser.add_argument(
        "--skip-behavioral-kv",
        action="store_true",
        help="Skip Phase 2 prefix-cache behavioral validation.",
    )
    parser.add_argument(
        "--require-behavioral-kv",
        action="store_true",
        help="Require Phase 2 behavioral KV validation to pass before accepting a config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    started = time.time()
    try:
        source_report = read_report(args.config_report)
        candidates = collect_candidates(source_report, max_candidates=max(1, int(args.max_candidates)))
        model_path = args.model or report_model_path(source_report)
        integrity = validate_report_integrity(source_report, candidates)
        topology = validate_topology(source_report, candidates)

        model_identity = section("skipped", "dry-run")
        projection = section("skipped", "dry-run")
        behavior = section("skipped", "dry-run")
        if not args.dry_run:
            if not model_path:
                raise RuntimeError("No model path supplied and config report has no model field.")
            eprint(f"Loading model for validation: {model_path}")
            torch, model, tokenizer, device, dtype_name = load_model_and_tokenizer(args, model_path)
            model_identity = validate_model_identity(source_report, model, model_path, device, dtype_name)
            projection = validate_projection_gate(args, source_report, candidates, torch, model, tokenizer, device)
            if args.skip_behavioral_kv:
                behavior = section("skipped", "requested")
            else:
                behavior = validate_behavior_gate(args, source_report, projection, torch, model, tokenizer, device)

        decision = decide_validation(
            source_report,
            integrity,
            topology,
            projection,
            behavior,
            require_behavioral_kv=bool(args.require_behavioral_kv),
            min_validation_margin=float(args.min_validation_margin),
        )
        report = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "generated_at_unix": int(started),
            "provenance": provenance(args, started, model_path),
            "source_report_summary": {
                "schema_name": source_report.get("schema_name"),
                "schema_version": source_report.get("schema_version"),
                "recommendation_review_status": (source_report.get("recommendation_review") or {}).get("status"),
                "adapter_config_candidate_status": (source_report.get("adapter_config_candidate") or {}).get("status"),
            },
            "validation_status": decision["status"],
            "confidence": decision["confidence"],
            "validation_level": decision["validation_level"],
            "auto_load_allowed": decision["auto_load_allowed"],
            "harness_load_policy": decision["harness_load_policy"],
            "selected_config": decision["selected_config"],
            "warnings": decision["warnings"],
            "report_integrity": integrity,
            "topology_gate": topology,
            "model_identity_gate": model_identity,
            "projection_gate": projection,
            "behavior_gate": decision["behavior_gate"],
            "candidate_results": behavior.get("candidates") or projection.get("candidates", []),
            "runtime_seconds": round(time.time() - started, 3),
        }
        emit(report, args.json_out)
        return 0 if decision["status"] in {"accepted", "needs_review"} else 2
    except Exception as exc:
        report = failure_report(args, started, "validation", exc)
        emit(report, args.json_out)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
