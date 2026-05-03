#!/usr/bin/env python3
"""
Standalone open-weight model architecture calibrator.

This script intentionally does not import Lazarus runtime or the older
calibrate_arch examples. It loads a Hugging Face causal LM directly, reports
model topology, preserves the legacy vec-inject/copy-head calibration, and
adds KV-direct source/target candidate discovery for adapter configuration.

The generated report is a measurement artifact, not a registry mutation. Use
it to decide model adapter fields such as route layer, query head, boundary
layer, KV source layer, KV target layer, insertion family, and candidate
confidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_NAME = "lazarus.model_config_calibration_report"
SCHEMA_VERSION = 2
LOW_CONFIDENCE_EXIT_CODE = 2


BASE_PROJECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "q": ("q_proj", "query_proj", "query", "q"),
    "k": ("k_proj", "key_proj", "key", "k"),
    "v": ("v_proj", "value_proj", "value", "v"),
    "o": ("o_proj", "out_proj", "output_proj", "dense", "wo", "o"),
}

FAMILY_PROJECTION_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "gemma": {
        "q": ("q_proj",),
        "k": ("k_proj",),
        "v": ("v_proj",),
        "o": ("o_proj",),
    },
    "gemma3": {
        "q": ("q_proj",),
        "k": ("k_proj",),
        "v": ("v_proj",),
        "o": ("o_proj",),
    },
    "gemma4": {
        "q": ("q_proj",),
        "k": ("k_proj",),
        "v": ("v_proj",),
        "o": ("o_proj",),
    },
    "llama": {
        "q": ("q_proj",),
        "k": ("k_proj",),
        "v": ("v_proj",),
        "o": ("o_proj",),
    },
    "qwen": {
        "q": ("q_proj",),
        "k": ("k_proj",),
        "v": ("v_proj",),
        "o": ("o_proj",),
    },
    "qwen3": {
        "q": ("q_proj",),
        "k": ("k_proj",),
        "v": ("v_proj",),
        "o": ("o_proj",),
    },
    "smollm": {
        "q": ("q_proj",),
        "k": ("k_proj",),
        "v": ("v_proj",),
        "o": ("o_proj",),
    },
}


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


def section_status(status: str, reason: str | None = None, **fields: Any) -> dict[str, Any]:
    section = {"status": status}
    if reason:
        section["reason"] = reason
    section.update(fields)
    return json_safe(section)


def iso_utc(timestamp: float | None = None) -> str:
    moment = datetime.fromtimestamp(time.time() if timestamp is None else timestamp, timezone.utc)
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_provenance(args: argparse.Namespace, started: float, device: str, dtype_name: str) -> dict[str, Any]:
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
            "loader_options": {
                "model": args.model,
                "device": device,
                "dtype": dtype_name,
                "requested_device": args.device,
                "requested_dtype": args.dtype,
                "attn_implementation": args.attn_implementation,
                "trust_remote_code": bool(args.trust_remote_code),
                "local_files_only": bool(args.local_files_only),
                "default_local_files_only": bool(args.default_local_files_only),
            },
            "probe_options": {
                "inspect_only": bool(args.inspect_only),
                "skip_head_scan": bool(args.skip_head_scan),
                "max_probes": int(args.max_probes),
                "top_layers": int(args.top_layers),
                "scan_layers": args.scan_layers,
                "max_heads": args.max_heads,
                "coverage_threshold": float(args.coverage_threshold),
                "candidate_targets": args.candidate_targets,
                "target_layers": args.target_layers,
                "allow_sliding": bool(args.allow_sliding),
                "max_candidates": int(args.max_candidates),
            },
            "review_options": {
                "fail_on_low_confidence": bool(args.fail_on_low_confidence),
                "min_score_margin": float(args.min_score_margin),
            },
        }
    )


def build_failure_provenance(args: argparse.Namespace, started: float) -> dict[str, Any]:
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
            "loader_options": {
                "model": args.model,
                "requested_device": args.device,
                "requested_dtype": args.dtype,
                "attn_implementation": args.attn_implementation,
                "trust_remote_code": bool(args.trust_remote_code),
                "local_files_only": bool(args.local_files_only),
                "default_local_files_only": bool(args.default_local_files_only),
            },
        }
    )


def failure_report(
    args: argparse.Namespace,
    started: float,
    *,
    stage: str,
    error: BaseException | str,
) -> dict[str, Any]:
    error_text = str(error)
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": int(started),
        "provenance": build_failure_provenance(args, started),
        "status": "failed",
        "failure": section_status(
            "failed",
            error_text,
            stage=stage,
            error_type=error.__class__.__name__ if isinstance(error, BaseException) else "Error",
        ),
        "warnings": [f"{stage}: {error_text}"],
        "model": {"model": args.model},
        "topology": section_status("unavailable", f"{stage} failed before topology extraction"),
        "legacy_vec_inject": section_status("unavailable", f"{stage} failed before calibration"),
        "head_ablation": section_status("unavailable", f"{stage} failed before calibration"),
        "kv_candidates": [],
        "recommended": {},
        "recommendation_review": section_status("unavailable", f"{stage} failed"),
        "adapter_config_candidate": section_status("review_required", f"{stage} failed"),
        "runtime_seconds": round(time.time() - started, 3),
    }


def emit_report(report: dict[str, Any], json_out: str | None) -> None:
    text = json.dumps(json_safe(report), indent=2, sort_keys=True)
    print(text)
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(text + "\n", encoding="utf-8")
        eprint(f"Wrote report to {json_out}")


def score_margin(items: list[dict[str, Any]], score_key: str = "candidate_score") -> dict[str, Any]:
    scored = [
        (idx, float(item[score_key]), item)
        for idx, item in enumerate(items)
        if item.get(score_key) is not None
    ]
    if not scored:
        return section_status("unavailable", "no scored items", score_key=score_key)
    ranked = sorted(scored, key=lambda row: row[1], reverse=True)
    best_idx, best_score, best_item = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else None
    margin = None if second_score is None else float(best_score - second_score)
    return section_status(
        "ok",
        score_key=score_key,
        best_index=int(best_idx),
        best_score=float(best_score),
        second_best_score=second_score,
        margin=margin,
        best_kv_target_layer=best_item.get("kv_target_layer"),
    )


def resolve_attr(root: Any, dotted: str) -> Any | None:
    current = root
    for part in dotted.split("."):
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def get_text_config(config: Any) -> Any:
    if config is None:
        return None
    if hasattr(config, "get_text_config"):
        try:
            return config.get_text_config()
        except Exception:
            pass
    return getattr(config, "text_config", None) or config


def adapter_family_from_config(config: Any) -> str:
    text_config = get_text_config(config)
    candidates = [
        getattr(text_config, "model_type", None),
        getattr(config, "model_type", None),
        getattr(text_config, "architectures", None),
        getattr(config, "architectures", None),
    ]
    normalized_parts: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple)):
            normalized_parts.extend(str(item).lower() for item in candidate)
        else:
            normalized_parts.append(str(candidate).lower())
    normalized = " ".join(normalized_parts).replace("-", "_")
    if "gemma4" in normalized:
        return "gemma4"
    if "gemma3" in normalized:
        return "gemma3"
    if "gemma" in normalized:
        return "gemma"
    if "qwen3" in normalized:
        return "qwen3"
    if "qwen" in normalized:
        return "qwen"
    if "smollm" in normalized or "smol_lm" in normalized:
        return "smollm"
    if "llama" in normalized:
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


def all_projection_aliases(adapter_family: str) -> dict[str, list[str]]:
    return {
        role: list(projection_aliases(adapter_family, role))
        for role in ("q", "k", "v", "o")
    }


def resolve_attention_projection(
    attn: Any, role: str, adapter_family: str
) -> tuple[Any | None, str | None, tuple[str, ...]]:
    checked = projection_aliases(adapter_family, role)
    if attn is None:
        return None, None, checked
    for alias in checked:
        try:
            value = getattr(attn, alias)
        except Exception:
            continue
        if value is not None:
            return value, alias, checked
    return None, None, checked


def projection_like_attrs(attn: Any) -> list[str]:
    if attn is None:
        return []
    names: set[str] = set()
    modules = getattr(attn, "_modules", None)
    if isinstance(modules, dict):
        names.update(str(name) for name, value in modules.items() if value is not None)
    try:
        names.update(str(name) for name in vars(attn).keys())
    except TypeError:
        pass
    if not names:
        try:
            names.update(str(name) for name in dir(attn))
        except Exception:
            pass
    tokens = ("proj", "projection", "query", "key", "value", "dense", "attn")
    return sorted(
        name
        for name in names
        if not name.startswith("_") and any(token in name.lower() for token in tokens)
    )


def projection_diagnostics(
    config: Any, attn: Any, roles: tuple[str, ...] = ("q", "k", "v", "o")
) -> dict[str, Any]:
    family = adapter_family_from_config(config)
    resolved: dict[str, str | None] = {}
    checked: dict[str, list[str]] = {}
    for role in roles:
        _module, alias, aliases = resolve_attention_projection(attn, role, family)
        resolved[role] = alias
        checked[role] = list(aliases)
    return {
        "adapter_family": family,
        "attention_class": attn.__class__.__name__ if attn is not None else None,
        "available_projection_like_attrs": projection_like_attrs(attn),
        "aliases_checked": checked,
        "resolved_aliases": resolved,
        "missing_projections": [role for role, alias in resolved.items() if alias is None],
    }


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
    raise RuntimeError(
        "Could not resolve decoder layers. Add this model family to resolve_layers()."
    )


def resolve_self_attn(layer: Any) -> Any | None:
    for name in ("self_attn", "attention", "attn", "self_attention"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


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


def infer_head_dim(attn: Any, config: Any) -> int | None:
    for name in ("head_dim", "attention_head_size"):
        value = getattr(attn, name, None)
        if value:
            return int(value)
    q_proj, _alias, _checked = resolve_attention_projection(
        attn, "q", adapter_family_from_config(config)
    )
    out_features = getattr(q_proj, "out_features", None)
    heads = config_value(config, ("num_attention_heads", "n_head", "num_heads"))
    if out_features and heads:
        return int(out_features) // int(heads)
    hidden = config_value(config, ("hidden_size", "n_embd", "d_model"))
    if hidden and heads:
        return int(hidden) // int(heads)
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


def layer_family(layer_type: str | None, attn: Any) -> str:
    normalized = (layer_type or "").lower()
    if bool(getattr(attn, "is_sliding", False)):
        return "sliding_attention"
    if normalized in {"sliding_attention", "sliding", "local_attention"}:
        return "sliding_attention"
    if normalized in {"full_attention", "global_attention", "full", "global"}:
        return "full_attention"
    # Most open-weight causal LMs without explicit layer_types use global KV.
    return "full_attention"


def layer_topology(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    text_config = get_text_config(config)
    adapter_family = adapter_family_from_config(config)
    layers = resolve_layers(model)
    layer_types = list(getattr(text_config, "layer_types", []) or [])
    infos: list[dict[str, Any]] = []

    for idx, layer in enumerate(layers):
        attn = resolve_self_attn(layer)
        declared_type = None
        if attn is not None:
            declared_type = getattr(attn, "layer_type", None)
        if declared_type is None and idx < len(layer_types):
            declared_type = layer_types[idx]
        family = layer_family(str(declared_type) if declared_type is not None else None, attn)

        q_proj, q_alias, _q_checked = resolve_attention_projection(attn, "q", adapter_family)
        k_proj, k_alias, _k_checked = resolve_attention_projection(attn, "k", adapter_family)
        v_proj, v_alias, _v_checked = resolve_attention_projection(attn, "v", adapter_family)
        o_proj, o_alias, _o_checked = resolve_attention_projection(attn, "o", adapter_family)
        head_dim = infer_head_dim(attn, config) if attn is not None else None
        k_out = module_out_features(k_proj) if k_proj is not None else None
        q_out = module_out_features(q_proj) if q_proj is not None else None
        num_kv_heads = (
            int(k_out) // int(head_dim)
            if k_out is not None and head_dim not in (None, 0)
            else config_value(config, ("num_key_value_heads", "num_kv_heads"))
        )
        num_heads = (
            int(q_out) // int(head_dim)
            if q_out is not None and head_dim not in (None, 0)
            else config_value(config, ("num_attention_heads", "n_head", "num_heads"))
        )

        infos.append(
            {
                "layer": idx,
                "declared_type": str(declared_type) if declared_type is not None else None,
                "family": family,
                "has_self_attn": attn is not None,
                "has_q_proj": q_proj is not None,
                "has_k_proj": k_proj is not None,
                "has_v_proj": v_proj is not None,
                "has_o_proj": o_proj is not None,
                "projection_aliases": {
                    "q": q_alias,
                    "k": k_alias,
                    "v": v_alias,
                    "o": o_alias,
                },
                "has_input_layernorm": hasattr(layer, "input_layernorm"),
                "has_k_norm": hasattr(attn, "k_norm") if attn is not None else False,
                "has_v_norm": hasattr(attn, "v_norm") if attn is not None else False,
                "store_full_length_kv": bool(getattr(attn, "store_full_length_kv", False))
                if attn is not None
                else False,
                "is_kv_shared_layer": bool(getattr(attn, "is_kv_shared_layer", False))
                if attn is not None
                else False,
                "kv_shared_layer_index": (
                    int(getattr(attn, "kv_shared_layer_index"))
                    if attn is not None and getattr(attn, "kv_shared_layer_index", None) is not None
                    else None
                ),
                "head_dim": int(head_dim) if head_dim is not None else None,
                "num_heads": int(num_heads) if num_heads is not None else None,
                "num_key_value_heads": int(num_kv_heads) if num_kv_heads is not None else None,
                "q_proj_out": q_out,
                "k_proj_out": k_out,
                "k_proj_in": module_in_features(k_proj) if k_proj is not None else None,
                "v_proj_out": module_out_features(v_proj) if v_proj is not None else None,
            }
        )

    full_layers = [info["layer"] for info in infos if info["family"] == "full_attention"]
    sliding_layers = [info["layer"] for info in infos if info["family"] == "sliding_attention"]
    return {
        "num_layers": len(layers),
        "adapter_family": adapter_family,
        "projection_aliases": all_projection_aliases(adapter_family),
        "hook_capabilities": {
            "attention_projection_resolution": True,
            "o_projection_pre_hook_head_ablation": any(
                info["has_o_proj"] for info in infos
            ),
            "kv_projection_scoring": any(
                info["has_k_proj"] and info["has_v_proj"] for info in infos
            ),
        },
        "full_attention_layers": full_layers,
        "sliding_attention_layers": sliding_layers,
        "explicit_layer_types": layer_types,
        "layers": infos,
    }


def resolve_projection_producer(
    topology: dict[str, Any], target_layer: int
) -> tuple[int | None, str | None]:
    layers = topology["layers"]
    if target_layer < 0 or target_layer >= len(layers):
        return None, "target_out_of_range"
    target = layers[target_layer]
    if target["has_k_proj"] and target["has_v_proj"]:
        return target_layer, None
    shared_idx = target.get("kv_shared_layer_index")
    if shared_idx is None:
        return None, "target_lacks_kv_projection_and_shared_index"
    if shared_idx < 0 or shared_idx >= len(layers):
        return None, "kv_shared_layer_index_out_of_range"
    producer = layers[shared_idx]
    if producer["has_k_proj"] and producer["has_v_proj"]:
        return int(shared_idx), None
    return None, "kv_shared_producer_lacks_kv_projection"


def materialization_lineage(topology: dict[str, Any], target_layer: int) -> list[int]:
    layers = topology["layers"]
    if target_layer < 0 or target_layer >= len(layers):
        return []
    target = layers[target_layer]
    family = target["family"]
    lineage = [target_layer]
    for info in layers[target_layer + 1 :]:
        same_family = info["family"] == family
        shares_from_target = info.get("kv_shared_layer_index") == target_layer
        if same_family or shares_from_target:
            lineage.append(int(info["layer"]))
    return list(dict.fromkeys(lineage))


def tokenizer_file_hash(model_arg: str, tokenizer: Any) -> dict[str, Any]:
    path = Path(model_arg)
    if not path.exists():
        name = getattr(tokenizer, "name_or_path", model_arg)
        return {"tokenizer_id": str(name), "tokenizer_hash": None}
    digest = hashlib.sha256()
    seen = []
    for filename in (
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "spiece.model",
    ):
        candidate = path / filename
        if candidate.exists():
            digest.update(filename.encode("utf-8"))
            digest.update(candidate.read_bytes())
            seen.append(filename)
    return {
        "tokenizer_id": str(path),
        "tokenizer_files": seen,
        "tokenizer_hash": digest.hexdigest() if seen else None,
    }


def model_identity(model_arg: str, model: Any, tokenizer: Any, device: str, dtype_name: str) -> dict[str, Any]:
    config = getattr(model, "config", None)
    text_config = get_text_config(config)
    adapter_family = adapter_family_from_config(config)
    commit_hash = getattr(config, "_commit_hash", None) or getattr(text_config, "_commit_hash", None)
    path = Path(model_arg)
    if commit_hash is None and path.exists() and path.parent.name == "snapshots":
        commit_hash = path.name
    identity = {
        "model": model_arg,
        "wrapper_model_type": getattr(config, "model_type", None),
        "text_model_type": getattr(text_config, "model_type", None),
        "adapter_family": adapter_family,
        "projection_aliases": all_projection_aliases(adapter_family),
        "model_revision_or_hash": commit_hash,
        "hidden_size": config_value(config, ("hidden_size", "n_embd", "d_model")),
        "num_attention_heads": config_value(config, ("num_attention_heads", "n_head", "num_heads")),
        "num_key_value_heads": config_value(config, ("num_key_value_heads", "num_kv_heads")),
        "device": device,
        "dtype": dtype_name,
        "tokenizer_class": tokenizer.__class__.__name__,
    }
    identity.update(tokenizer_file_hash(model_arg, tokenizer))
    return identity


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
    variants = [f" {answer}", answer]
    for variant in variants:
        ids = encode(tokenizer, variant, add_special_tokens=False)
        if ids:
            return ids
    raise RuntimeError(f"Could not tokenize answer {answer!r}")


def locate_answer(tokenizer: Any, doc: str, answer: str) -> tuple[list[int], int, int]:
    doc_ids = encode(tokenizer, doc, add_special_tokens=True)
    variants = [f" {answer}", answer, f"{answer}."]
    for variant in variants:
        ids = encode(tokenizer, variant, add_special_tokens=False)
        start = find_subsequence(doc_ids, ids)
        if start is not None and ids:
            return doc_ids, start, ids[0]
    # Slow fallback for tokenizers that split preceding whitespace differently.
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


def embedding_vector(model: Any, torch: Any, token_id: int) -> Any:
    embeddings = model.get_input_embeddings()
    return embeddings.weight[int(token_id)].detach().to("cpu", dtype=torch.float32)


def next_token_probability(
    model: Any,
    tokenizer: Any,
    torch: Any,
    prompt: str,
    answer: str,
    device: str,
) -> float:
    ans_id = int(answer_tokens(tokenizer, answer)[0])
    ids = encode(tokenizer, prompt, add_special_tokens=True)
    input_ids = torch.as_tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        outputs = run_model(model, torch, input_ids, attention_mask)
        logits = outputs.logits[0, -1, :].to(dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
    return float(probs[ans_id].item())


def scan_legacy_vec_inject(
    model: Any,
    tokenizer: Any,
    torch: Any,
    layers: list[Any],
    probes: list[dict[str, str]],
    layer_indices: list[int],
    device: str,
) -> dict[str, Any]:
    per_layer_scores: dict[int, list[float]] = {idx: [] for idx in layer_indices}
    sample_residuals: dict[int, Any] = {}
    probe_status: list[dict[str, Any]] = []

    for probe in probes:
        try:
            token_ids, answer_pos, answer_token_id = locate_answer(
                tokenizer, probe["doc"], probe["answer"]
            )
        except RuntimeError as exc:
            probe_status.append({"name": probe["name"], "status": "skipped", "reason": str(exc)})
            continue
        target_pos = max(0, int(answer_pos) - 1)
        embed_vec = embedding_vector(model, torch, answer_token_id)
        captures = capture_layer_vectors(
            model, torch, layers, token_ids, target_pos, layer_indices, device
        )
        for layer_idx, vector in captures.items():
            score = float(torch.dot(vector, embed_vec).item())
            per_layer_scores[layer_idx].append(score)
            sample_residuals.setdefault(layer_idx, vector)
        probe_status.append({"name": probe["name"], "status": "used"})

    avg_scores = {
        str(layer_idx): float(sum(scores) / len(scores))
        for layer_idx, scores in per_layer_scores.items()
        if scores
    }
    if not avg_scores:
        return {
            "status": "unavailable",
            "reason": "No probes produced layer scores.",
            "avg_dot_by_layer": {},
            "sample_residuals": {},
            "probe_status": probe_status,
        }
    best_layer = max(avg_scores, key=lambda key: avg_scores[key])
    legacy_injection_layer = int(best_layer)
    legacy_retrieval_layer = max(0, legacy_injection_layer - 1)
    return {
        "status": "ok",
        "avg_dot_by_layer": avg_scores,
        "legacy_injection_layer": legacy_injection_layer,
        "legacy_retrieval_layer": legacy_retrieval_layer,
        "sample_residuals": sample_residuals,
        "probe_status": probe_status,
    }


def scan_query_heads(
    model: Any,
    tokenizer: Any,
    torch: Any,
    layers: list[Any],
    probes: list[dict[str, str]],
    retrieval_layer: int,
    device: str,
    max_heads: int | None,
    coverage_threshold: float,
) -> dict[str, Any]:
    if retrieval_layer < 0 or retrieval_layer >= len(layers):
        return {"status": "unavailable", "reason": "retrieval_layer out of range"}
    config = getattr(model, "config", None)
    attn = resolve_self_attn(layers[retrieval_layer])
    adapter_family = adapter_family_from_config(config)
    o_proj, o_alias, o_checked = resolve_attention_projection(attn, "o", adapter_family)
    if attn is None or o_proj is None:
        return {
            "status": "unavailable",
            "reason": "attention module has no output projection for per-head ablation",
            "diagnostics": projection_diagnostics(config, attn, ("o",)),
        }
    o_in = module_in_features(o_proj)
    if not o_in:
        return {
            "status": "unavailable",
            "reason": "could not infer output projection input width",
            "diagnostics": projection_diagnostics(config, attn, ("o",)),
        }

    num_heads = int(
        getattr(attn, "num_heads", None)
        or config_value(config, ("num_attention_heads", "n_head", "num_heads"), 0)
        or 0
    )
    if num_heads <= 0:
        return {
            "status": "unavailable",
            "reason": "could not infer attention head count",
            "diagnostics": projection_diagnostics(config, attn, ("o",)),
        }
    head_dim = int(o_in) // int(num_heads)
    if head_dim <= 0 or head_dim * num_heads != int(o_in):
        return {
            "status": "unavailable",
            "reason": "o_proj input width does not divide by inferred head count",
            "num_heads": num_heads,
            "o_proj_in": o_in,
            "diagnostics": projection_diagnostics(config, attn, ("o",)),
        }

    heads_to_scan = list(range(num_heads))
    if max_heads is not None:
        heads_to_scan = heads_to_scan[: max(1, int(max_heads))]

    baselines = [
        next_token_probability(model, tokenizer, torch, p["cloze_prefix"], p["answer"], device)
        for p in probes
    ]

    head_results: dict[str, Any] = {}
    for head_idx in heads_to_scan:
        start = int(head_idx) * int(head_dim)
        end = start + int(head_dim)

        def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if not inputs:
                return inputs
            x = inputs[0]
            ablated = x.clone()
            ablated[..., start:end] = 0
            return (ablated, *inputs[1:])

        handle = o_proj.register_forward_pre_hook(pre_hook)
        try:
            deltas = []
            for probe, baseline in zip(probes, baselines):
                p_ablated = next_token_probability(
                    model, tokenizer, torch, probe["cloze_prefix"], probe["answer"], device
                )
                deltas.append(float(baseline - p_ablated))
        finally:
            handle.remove()

        mean_delta = float(sum(deltas) / len(deltas)) if deltas else 0.0
        coverage = (
            float(sum(1 for delta in deltas if delta >= coverage_threshold) / len(deltas))
            if deltas
            else 0.0
        )
        score = float(mean_delta * coverage)
        head_results[str(head_idx)] = {
            "score": score,
            "mean_delta": mean_delta,
            "coverage": coverage,
            "deltas": deltas,
        }

    if not head_results:
        return {"status": "unavailable", "reason": "no heads were scanned"}
    best_head = max(head_results, key=lambda key: head_results[key]["score"])
    return {
        "status": "ok",
        "retrieval_layer": int(retrieval_layer),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "adapter_family": adapter_family,
        "o_projection_alias": o_alias,
        "projection_aliases_checked": {"o": list(o_checked)},
        "heads_scanned": heads_to_scan,
        "best_head": int(best_head),
        "routing_score": float(head_results[best_head]["score"]),
        "head_results": head_results,
        "baseline_answer_probabilities": baselines,
    }


def parse_int_list(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def build_kv_candidates(
    topology: dict[str, Any],
    legacy: dict[str, Any] | None,
    *,
    candidate_targets: str,
    explicit_targets: set[int] | None,
    allow_sliding: bool,
) -> list[dict[str, Any]]:
    layers = topology["layers"]
    candidate_target_indices: list[int] = []
    if explicit_targets is not None:
        candidate_target_indices = sorted(explicit_targets)
    else:
        for info in layers:
            target = int(info["layer"])
            if target == 0:
                continue
            is_full = info["family"] == "full_attention"
            is_sliding = info["family"] == "sliding_attention"
            if candidate_targets == "all":
                include = True
            elif candidate_targets == "legacy":
                include = legacy is not None and target == legacy.get("legacy_injection_layer")
            else:
                include = is_full or (allow_sliding and is_sliding)
            if include:
                candidate_target_indices.append(target)

        if legacy is not None and legacy.get("legacy_injection_layer") is not None:
            legacy_target = int(legacy["legacy_injection_layer"])
            if 0 < legacy_target < len(layers):
                candidate_target_indices.append(legacy_target)

    avg_dot = {}
    if legacy is not None:
        avg_dot = {int(k): float(v) for k, v in legacy.get("avg_dot_by_layer", {}).items()}

    candidates: list[dict[str, Any]] = []
    for target in sorted(set(candidate_target_indices)):
        source = target - 1
        if source < 0 or target >= len(layers):
            continue
        target_info = layers[target]
        source_info = layers[source]
        family = target_info["family"]
        if family == "sliding_attention" and not allow_sliding and candidate_targets != "all":
            continue
        producer, unsafe_reason = resolve_projection_producer(topology, target)
        lineage = materialization_lineage(topology, target)
        safe = unsafe_reason is None and producer is not None
        legacy_target_match = bool(
            legacy is not None and target == legacy.get("legacy_injection_layer")
        )
        legacy_source_match = bool(
            legacy is not None and source == legacy.get("legacy_retrieval_layer")
        )
        source_dot = avg_dot.get(source)
        target_dot = avg_dot.get(target)
        candidate_score = 0.0
        if source_dot is not None:
            candidate_score += source_dot
        if target_dot is not None:
            candidate_score += 0.25 * target_dot
        if legacy_target_match:
            candidate_score += 1.0
        if family == "full_attention":
            candidate_score += 0.5

        candidates.append(
            {
                "kv_source_layer": int(source),
                "kv_target_layer": int(target),
                "boundary_layer": int(source),
                "source_family": source_info["family"],
                "insertion_family": "sliding" if family == "sliding_attention" else "full_attention",
                "target_declared_type": target_info["declared_type"],
                "projection_producer_layer": producer,
                "materialization_safe": bool(safe),
                "unsafe_reason": unsafe_reason,
                "lineage": lineage,
                "legacy_target_match": legacy_target_match,
                "legacy_source_match": legacy_source_match,
                "source_avg_dot": source_dot,
                "target_avg_dot": target_dot,
                "candidate_score": float(candidate_score),
                "path_a_replay_expected": 0,
            }
        )

    return sorted(
        candidates,
        key=lambda item: (
            not item["materialization_safe"],
            -float(item.get("candidate_score") or 0.0),
            int(item["kv_target_layer"]),
        ),
    )


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


def score_projection_variants(
    model: Any,
    torch: Any,
    layers: list[Any],
    topology: dict[str, Any],
    candidates: list[dict[str, Any]],
    sample_residuals: dict[int, Any],
    device: str,
    max_candidates: int,
) -> list[dict[str, Any]]:
    first_param = next(model.parameters())
    model_dtype = first_param.dtype
    config = getattr(model, "config", None)
    adapter_family = adapter_family_from_config(config)
    scored: list[dict[str, Any]] = []
    for candidate in candidates[: max(0, int(max_candidates))]:
        result = dict(candidate)
        source = int(candidate["kv_source_layer"])
        target = int(candidate["kv_target_layer"])
        producer = candidate.get("projection_producer_layer")
        if not candidate.get("materialization_safe"):
            result["projection_scores"] = {}
            result["projection_status"] = "unsafe"
            scored.append(result)
            continue
        if source not in sample_residuals:
            result["projection_scores"] = {}
            result["projection_status"] = "no_sample_residual_for_source"
            scored.append(result)
            continue

        producer_layer = layers[int(producer)]
        producer_attn = resolve_self_attn(producer_layer)
        target_layer = layers[target]
        if producer_attn is None:
            result["projection_scores"] = {}
            result["projection_status"] = "producer_has_no_attention"
            result["projection_diagnostics"] = projection_diagnostics(
                config, producer_attn, ("k", "v")
            )
            scored.append(result)
            continue

        k_proj, k_alias, k_checked = resolve_attention_projection(
            producer_attn, "k", adapter_family
        )
        v_proj, v_alias, v_checked = resolve_attention_projection(
            producer_attn, "v", adapter_family
        )
        if k_proj is None or v_proj is None:
            result["projection_scores"] = {}
            result["projection_status"] = "producer_lacks_kv_projection"
            result["projection_diagnostics"] = projection_diagnostics(
                config, producer_attn, ("k", "v")
            )
            scored.append(result)
            continue

        residual = (
            sample_residuals[source]
            .to(device=device, dtype=model_dtype, non_blocking=True)
            .reshape(1, 1, -1)
        )
        projection_scores: dict[str, Any] = {}
        variants: list[tuple[str, Any]] = [("raw_residual", residual)]
        input_layernorm = getattr(target_layer, "input_layernorm", None)
        if input_layernorm is not None:
            with torch.inference_mode():
                variants.append(("input_layernorm_then_kv_proj", input_layernorm(residual)))

        with torch.inference_mode():
            for variant_name, variant_residual in variants:
                k_flat = k_proj(variant_residual)
                v_flat = v_proj(variant_residual)
                projection_scores[variant_name] = {
                    "K": tensor_stats(torch, k_flat),
                    "V": tensor_stats(torch, v_flat),
                }

        if "raw_residual" in projection_scores and "input_layernorm_then_kv_proj" in projection_scores:
            raw_k_norm = projection_scores["raw_residual"]["K"]["norm"]
            norm_k_norm = projection_scores["input_layernorm_then_kv_proj"]["K"]["norm"]
            raw_v_norm = projection_scores["raw_residual"]["V"]["norm"]
            norm_v_norm = projection_scores["input_layernorm_then_kv_proj"]["V"]["norm"]
            projection_scores["variant_delta"] = {
                "k_norm_ratio_input_layernorm_over_raw": (
                    float(norm_k_norm / raw_k_norm) if raw_k_norm else None
                ),
                "v_norm_ratio_input_layernorm_over_raw": (
                    float(norm_v_norm / raw_v_norm) if raw_v_norm else None
                ),
            }

        result["projection_scores"] = projection_scores
        result["projection_status"] = "ok"
        result["projection_aliases"] = {"k": k_alias, "v": v_alias}
        result["projection_aliases_checked"] = {
            "k": list(k_checked),
            "v": list(v_checked),
        }
        scored.append(result)
    return scored


def make_recommendation(
    legacy: dict[str, Any] | None,
    head_scan: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    safe_candidates = [c for c in candidates if c.get("materialization_safe")]
    kv_choice = safe_candidates[0] if safe_candidates else (candidates[0] if candidates else None)
    route_layer = None
    route_query_head = None
    routing_score = None
    if legacy and legacy.get("legacy_retrieval_layer") is not None:
        route_layer = int(legacy["legacy_retrieval_layer"])
    if head_scan and head_scan.get("status") == "ok":
        route_query_head = int(head_scan["best_head"])
        routing_score = float(head_scan["routing_score"])

    recommendation = {
        "route_layer": route_layer,
        "route_query_head": route_query_head,
        "routing_score": routing_score,
        "boundary_layer": None,
        "kv_source_layer": None,
        "kv_target_layer": None,
        "insertion_family": None,
        "confidence": "unmeasured",
        "notes": [],
    }
    if kv_choice is not None:
        recommendation.update(
            {
                "boundary_layer": int(kv_choice["boundary_layer"]),
                "kv_source_layer": int(kv_choice["kv_source_layer"]),
                "kv_target_layer": int(kv_choice["kv_target_layer"]),
                "insertion_family": kv_choice["insertion_family"],
                "confidence": "materialization_safe"
                if kv_choice.get("materialization_safe")
                else "unsafe_candidate_only",
            }
        )
    if legacy is not None and kv_choice is not None:
        legacy_injection = legacy.get("legacy_injection_layer")
        if legacy_injection == kv_choice.get("kv_target_layer"):
            recommendation["notes"].append(
                "Legacy injection layer was interpreted as KV target layer, not KV source layer."
            )
        elif legacy_injection is not None:
            recommendation["notes"].append(
                "Legacy vec-inject peak differs from recommended KV target; compare candidates before registry update."
            )
    return recommendation


def review_recommendation(
    recommendation: dict[str, Any],
    legacy: dict[str, Any] | None,
    head_scan: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    *,
    min_score_margin: float,
) -> dict[str, Any]:
    safe_candidates = [c for c in candidates if c.get("materialization_safe")]
    margin = score_margin(safe_candidates or candidates)
    warnings: list[str] = []
    checks: dict[str, bool] = {
        "has_kv_choice": recommendation.get("kv_source_layer") is not None
        and recommendation.get("kv_target_layer") is not None,
        "has_materialization_safe_candidate": bool(safe_candidates),
        "legacy_scan_ok": bool(legacy and legacy.get("status") == "ok"),
        "head_scan_ok": bool(head_scan and head_scan.get("status") == "ok"),
        "legacy_target_agrees": True,
        "score_margin_ok": False,
    }
    if legacy and legacy.get("status") == "ok":
        legacy_injection = legacy.get("legacy_injection_layer")
        kv_target = recommendation.get("kv_target_layer")
        if legacy_injection is not None and kv_target is not None:
            checks["legacy_target_agrees"] = int(legacy_injection) == int(kv_target)
    if margin.get("status") == "ok":
        margin_value = margin.get("margin")
        checks["score_margin_ok"] = margin_value is None or float(margin_value) >= min_score_margin
    else:
        warnings.append("Candidate score margin unavailable.")

    if not checks["has_kv_choice"]:
        warnings.append("No KV candidate was selected.")
    if not checks["has_materialization_safe_candidate"]:
        warnings.append("No materialization-safe KV candidate was available.")
    if not checks["legacy_scan_ok"]:
        warnings.append("Legacy vec-inject scan did not complete successfully.")
    head_scan_status = head_scan.get("status") if head_scan else None
    if not checks["head_scan_ok"] and head_scan_status not in {"unavailable", "skipped"}:
        warnings.append("Head ablation did not complete successfully; route_query_head may be unset.")
    if not checks["legacy_target_agrees"]:
        warnings.append(
            "Legacy vec-inject injection layer disagrees with the recommended KV target layer."
        )
    if not checks["score_margin_ok"]:
        warnings.append(
            f"Best candidate score margin is below required threshold {min_score_margin}."
        )

    needs_review = bool(warnings)
    if not checks["legacy_target_agrees"]:
        confidence = "review_required"
    elif checks["has_materialization_safe_candidate"] and checks["score_margin_ok"] and checks["head_scan_ok"]:
        confidence = "high"
    elif checks["has_materialization_safe_candidate"] and checks["score_margin_ok"]:
        confidence = "medium"
    else:
        confidence = "low"

    return section_status(
        "needs_review" if needs_review else "ok",
        confidence=confidence,
        confidence_level=confidence,
        meets_minimum_confidence=confidence not in {"low", "review_required"},
        needs_review=needs_review,
        min_score_margin=float(min_score_margin),
        score_margin=margin,
        checks=checks,
        warnings=warnings,
    )


def adapter_config_candidate(
    model_identity_data: dict[str, Any],
    recommendation: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    required_fields = (
        "route_layer",
        "boundary_layer",
        "kv_source_layer",
        "kv_target_layer",
        "insertion_family",
    )
    missing = [field for field in required_fields if recommendation.get(field) is None]
    status = (
        "ready"
        if not missing
        and review.get("meets_minimum_confidence")
        and not review.get("needs_review")
        else "review_required"
    )
    return section_status(
        status,
        reason=", ".join(f"missing {field}" for field in missing) if missing else None,
        schema_name="lazarus.adapter_config_candidate",
        schema_version=1,
        model=model_identity_data.get("model"),
        model_revision_or_hash=model_identity_data.get("model_revision_or_hash"),
        text_model_type=model_identity_data.get("text_model_type"),
        adapter_family=model_identity_data.get("adapter_family"),
        route_layer=recommendation.get("route_layer"),
        route_query_head=recommendation.get("route_query_head"),
        routing_score=recommendation.get("routing_score"),
        boundary_layer=recommendation.get("boundary_layer"),
        kv_source_layer=recommendation.get("kv_source_layer"),
        kv_target_layer=recommendation.get("kv_target_layer"),
        injection_layer=recommendation.get("kv_target_layer"),
        insertion_family=recommendation.get("insertion_family"),
        confidence=review.get("confidence"),
        review_status=review.get("status"),
        score_margin=review.get("score_margin"),
        notes=recommendation.get("notes", []),
    )


def load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "This script requires torch and transformers. Run with the repo environment, "
            "for example: uv run python David/get_model_config.py --model <model>"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


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


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, Any, str, str]:
    torch, AutoModelForCausalLM, AutoTokenizer = load_dependencies()
    device = choose_device(torch, args.device)
    dtype, dtype_name = choose_dtype(torch, args.dtype, device)
    tokenizer_kwargs = {
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
    }
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
    }
    if args.attn_implementation != "default":
        model_kwargs["attn_implementation"] = args.attn_implementation

    eprint(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    eprint(f"Loading model: {args.model} ({dtype_name} on {device})")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        except TypeError:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    if device != "cpu":
        model.to(device)
    return torch, model, tokenizer, device, dtype_name


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover legacy routing and KV-direct adapter candidates for an open-weight model."
    )
    parser.add_argument("--model", required=True, help="HF model id or local model path")
    parser.add_argument("--json-out", default=None, help="Optional path for JSON report")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, cuda:0, etc.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype. CPU forces float32.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="HF attention implementation. Use 'default' to omit this kwarg.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--default-local-files-only",
        action="store_true",
        help="Opt into local-files-only loading as the production default for this invocation.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only load model and emit identity/topology/KV candidates; no forward passes.",
    )
    parser.add_argument(
        "--skip-head-scan",
        action="store_true",
        help="Run layer scan but skip causal per-head ablation.",
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=len(PROBES),
        help="Number of built-in synthetic probes to use.",
    )
    parser.add_argument(
        "--top-layers",
        type=int,
        default=0,
        help="Number of final layers to scan. Default 0 scans the top third.",
    )
    parser.add_argument(
        "--scan-layers",
        default=None,
        help="Comma-separated exact layer indices to scan for legacy vec-inject.",
    )
    parser.add_argument(
        "--max-heads",
        type=int,
        default=None,
        help="Optional limit on heads scanned during causal ablation.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.05,
        help="Delta probability threshold counted as head-ablation coverage.",
    )
    parser.add_argument(
        "--candidate-targets",
        choices=["auto", "all", "legacy"],
        default="auto",
        help="auto=full/global targets plus legacy target, all=every layer, legacy=legacy peak only.",
    )
    parser.add_argument(
        "--target-layers",
        default=None,
        help="Comma-separated exact KV target layers. Overrides --candidate-targets.",
    )
    parser.add_argument(
        "--allow-sliding",
        action="store_true",
        help="Allow sliding-attention targets as experimental KV candidates.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="Projection-score this many sorted candidates.",
    )
    parser.add_argument(
        "--fail-on-low-confidence",
        action="store_true",
        help=f"Exit with {LOW_CONFIDENCE_EXIT_CODE} when recommendation_review confidence is low.",
    )
    parser.add_argument(
        "--min-score-margin",
        type=float,
        default=0.25,
        help="Minimum score gap between the top two candidates for recommendation review.",
    )
    args = parser.parse_args(argv)
    if args.default_local_files_only:
        args.local_files_only = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    started = time.time()
    try:
        torch, model, tokenizer, device, dtype_name = load_model_and_tokenizer(args)
    except Exception as exc:
        report = failure_report(args, started, stage="model_load", error=exc)
        emit_report(report, args.json_out)
        return 1
    layers = resolve_layers(model)
    topology = layer_topology(model)
    probes = PROBES[: max(1, int(args.max_probes))]
    identity = model_identity(args.model, model, tokenizer, device, dtype_name)

    report: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": int(started),
        "provenance": build_provenance(args, started, device, dtype_name),
        "warnings": [],
        "model": identity,
        "topology": topology,
        "legacy_vec_inject": section_status("skipped", "inspect-only"),
        "head_ablation": section_status("skipped", "inspect-only"),
        "kv_candidates": [],
        "recommended": {},
        "recommendation_review": section_status("pending"),
        "adapter_config_candidate": section_status("pending"),
        "runtime_seconds": None,
    }

    legacy: dict[str, Any] | None = None
    head_scan: dict[str, Any] | None = None
    sample_residuals: dict[int, Any] = {}

    if not args.inspect_only:
        exact_scan_layers = parse_int_list(args.scan_layers)
        if exact_scan_layers is not None:
            layer_indices = sorted(idx for idx in exact_scan_layers if 0 <= idx < len(layers))
        else:
            top_layers = int(args.top_layers) if args.top_layers else max(1, len(layers) // 3)
            first_layer = max(0, len(layers) - top_layers)
            layer_indices = list(range(first_layer, len(layers)))
        eprint(f"Running legacy layer scan over {len(layer_indices)} layer(s): {layer_indices}")
        legacy = scan_legacy_vec_inject(
            model, tokenizer, torch, layers, probes, layer_indices, device
        )
        sample_residuals = legacy.pop("sample_residuals", {}) if legacy is not None else {}
        report["legacy_vec_inject"] = legacy

        if legacy.get("status") == "ok" and not args.skip_head_scan:
            eprint(f"Running head ablation at L{legacy['legacy_retrieval_layer']}")
            head_scan = scan_query_heads(
                model,
                tokenizer,
                torch,
                layers,
                probes,
                int(legacy["legacy_retrieval_layer"]),
                device,
                args.max_heads,
                float(args.coverage_threshold),
            )
            report["head_ablation"] = head_scan
        else:
            report["head_ablation"] = {
                "status": "skipped",
                "reason": "requested" if args.skip_head_scan else "legacy scan unavailable",
            }
    else:
        legacy = None

    candidates = build_kv_candidates(
        topology,
        legacy,
        candidate_targets=args.candidate_targets,
        explicit_targets=parse_int_list(args.target_layers),
        allow_sliding=bool(args.allow_sliding),
    )
    if not args.inspect_only and sample_residuals:
        eprint(f"Projection-scoring up to {args.max_candidates} KV candidate(s)")
        scored = score_projection_variants(
            model,
            torch,
            layers,
            topology,
            candidates,
            sample_residuals,
            device,
            max_candidates=int(args.max_candidates),
        )
        scored_by_target = {item["kv_target_layer"]: item for item in scored}
        candidates = [
            scored_by_target.get(item["kv_target_layer"], item)
            for item in candidates
        ]

    report["kv_candidates"] = candidates
    recommendation = make_recommendation(legacy, head_scan, candidates)
    review = review_recommendation(
        recommendation,
        legacy,
        head_scan,
        candidates,
        min_score_margin=float(args.min_score_margin),
    )
    report["recommended"] = recommendation
    report["recommendation_review"] = review
    report["adapter_config_candidate"] = adapter_config_candidate(identity, recommendation, review)
    report["warnings"] = list(review.get("warnings", []))
    report["runtime_seconds"] = round(time.time() - started, 3)
    emit_report(report, args.json_out)
    if args.fail_on_low_confidence and review.get("confidence") == "low":
        return LOW_CONFIDENCE_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
