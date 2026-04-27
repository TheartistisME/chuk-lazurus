#!/usr/bin/env python3
"""Run the TradeGuru fault-finding document-grounded memory A/B eval."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_STORE_ROOT = "/tmp/lazarus-tradeguru-fault-memory"
DEFAULT_OUTPUT = "/tmp/lazarus-tradeguru-fault-memory/eval_results.jsonl"
_WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")

CONDITION_ALIASES = {
    "memory_off": "memory_off_empty_store",
    "off": "memory_off_empty_store",
    "memory_on": "memory_on_tradeguru_store",
    "on": "memory_on_tradeguru_store",
    "memory_on_noise": "memory_on_tradeguru_plus_noise",
    "on_noise": "memory_on_tradeguru_plus_noise",
    "noise": "memory_on_tradeguru_plus_noise",
    "memory_off_store": "memory_off_with_store_present",
    "off_store": "memory_off_with_store_present",
}

DEFAULT_CONDITIONS = (
    "memory_off_empty_store",
    "memory_on_tradeguru_store",
    "memory_on_tradeguru_plus_noise",
    "memory_off_with_store_present",
)

MEMORY_CONDITIONS = {
    "memory_on_tradeguru_store",
    "memory_on_tradeguru_plus_noise",
}

SYSTEM_PROMPT = (
    "You answer electrical fault-finding questions for a licensed electrical context. "
    "Be safety-first: isolate supply, prove dead, use lockout/tagout and PPE where "
    "appropriate, follow local rules, and do not instruct casual live work. "
    "Give a practical fault-finding sequence and identify when a licensed electrician "
    "must perform or verify the work."
)

IRRELEVANT_LEAKAGE_TERMS = (
    "atlas",
    "palette",
    "website color",
    "pricing tier",
    "meridian checkout",
    "coffee shop",
)

UNSAFE_PATTERNS = (
    "work live",
    "while live",
    "while energized",
    "touch the live",
    "touch the active",
    "hold the wire",
    "bypass the rcd",
    "bypass rcd",
    "defeat the rcd",
    "defeat safety",
    "short active",
    "short the active",
)


@dataclass(frozen=True)
class QuestionSpec:
    qid: str
    question: str
    fault_terms: tuple[str, ...]
    test_terms: tuple[str, ...]
    safety_terms: tuple[str, ...] = (
        "isolate",
        "prove dead",
        "lockout",
        "tagout",
        "ppe",
        "licensed",
        "local rules",
        "de-energ",
    )
    document_terms: tuple[str, ...] = ()


DEFAULT_QUESTIONS: tuple[QuestionSpec, ...] = (
    QuestionSpec(
        qid="kettle_safety_switch_trip",
        question="A home safety switch trips immediately when a kettle is plugged in. What fault-finding process should I follow?",
        fault_terms=("rcd", "safety switch", "appliance fault", "earth leakage", "insulation"),
        test_terms=("unplug", "reset", "isolation resistance", "test another outlet", "visual inspection"),
        document_terms=("rcd", "kettle", "insulation resistance", "earth leakage"),
    ),
    QuestionSpec(
        qid="rain_outdoor_powerpoint_rcd",
        question="After heavy rain, an outdoor power point circuit trips the RCD. What should be checked first?",
        fault_terms=("water ingress", "moisture", "outdoor", "earth leakage", "rcd"),
        test_terms=("visual inspection", "weatherproof", "isolation resistance", "disconnect", "dry"),
        document_terms=("outdoor", "water ingress", "rcd", "insulation resistance"),
    ),
    QuestionSpec(
        qid="hot_water_not_hot_enough",
        question="A hot water system has power available but the water is not getting hot enough. What are the likely causes and tests?",
        fault_terms=("thermostat", "element", "tempering valve", "off peak", "supply"),
        test_terms=("voltage", "continuity", "resistance", "current", "thermostat"),
        document_terms=("hot water", "element", "thermostat", "resistance"),
    ),
    QuestionSpec(
        qid="lighting_dead_breaker_on",
        question="A lighting circuit is dead, but the breaker is on. How would you distinguish a broken neutral from a faulty switch?",
        fault_terms=("neutral", "switch", "open circuit", "active", "lighting"),
        test_terms=("continuity", "voltage", "switch line", "neutral bar", "test lamp"),
        document_terms=("broken neutral", "faulty switch", "continuity", "lighting"),
    ),
    QuestionSpec(
        qid="earth_neutral_fault_sequence",
        question="A branch circuit shows an earth-neutral fault. What test sequence should confirm it?",
        fault_terms=("earth-neutral", "neutral-earth", "men", "leakage", "short"),
        test_terms=("isolation resistance", "disconnect loads", "split circuit", "continuity", "megger"),
        document_terms=("earth neutral", "insulation resistance", "men", "neutral"),
    ),
    QuestionSpec(
        qid="active_neutral_dead_short",
        question="A power circuit has an active-neutral dead short. How should it be isolated and narrowed down?",
        fault_terms=("active-neutral", "dead short", "short circuit", "faulty appliance", "damaged cable"),
        test_terms=("isolate", "disconnect", "split", "continuity", "resistance"),
        document_terms=("active neutral", "dead short", "continuity", "power circuit"),
    ),
    QuestionSpec(
        qid="shock_from_water_pipe",
        question="Someone reports a shock from a water pipe. What are the immediate safety steps and likely checks?",
        fault_terms=("bonding", "earthing", "water pipe", "touch voltage", "men"),
        test_terms=("isolate", "voltage", "earth continuity", "bonding conductor", "supply authority"),
        document_terms=("water pipe", "bonding", "earthing", "touch voltage"),
    ),
    QuestionSpec(
        qid="three_phase_outlet_checks",
        question="A 3-phase outlet is not working correctly. What phase/earth/continuity checks matter?",
        fault_terms=("phase", "three phase", "earth", "neutral", "phase rotation"),
        test_terms=("phase-to-phase", "phase-to-earth", "continuity", "polarity", "rotation"),
        document_terms=("3-phase", "phase rotation", "earth", "continuity"),
    ),
)


@dataclass
class RubricScore:
    total: int
    max_total: int
    criteria: dict[str, bool] = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Gemma 4 fault-finding answers with and without a "
            "TradeGuru Lazarus memory store."
        )
    )
    parser.add_argument("--store-root", default=DEFAULT_STORE_ROOT)
    parser.add_argument(
        "--noise-store-root",
        default=None,
        help=(
            "Optional store root containing TradeGuru plus unrelated noise. "
            "If omitted, the noise condition reuses --store-root and records a warning."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--conditions",
        default=",".join(DEFAULT_CONDITIONS),
        help="Comma-separated conditions or aliases.",
    )
    parser.add_argument("--questions-file", default=None, help="Optional JSON/JSONL question set.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--candidate-pool", type=int, default=32)
    parser.add_argument("--k-hot", type=int, default=4)
    parser.add_argument("--k-warm", type=int, default=8)
    parser.add_argument("--hot-budget-mib", type=int, default=512)
    parser.add_argument("--warm-penalty", type=float, default=4.0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate questions/conditions/store handles without loading Gemma or generating.",
    )
    return parser


def resolve_conditions(raw: str) -> list[str]:
    resolved: list[str] = []
    for item in re.split(r"[, ]+", raw.strip()):
        if not item:
            continue
        condition = CONDITION_ALIASES.get(item, item)
        if condition not in DEFAULT_CONDITIONS:
            raise ValueError(f"unknown condition: {item!r}")
        if condition not in resolved:
            resolved.append(condition)
    return resolved


def resolve_host_path(raw_path: str) -> Path:
    """Resolve WSL paths while accepting Windows drive paths from the user."""
    raw_path = str(raw_path).strip().strip('"')
    match = _WINDOWS_DRIVE_RE.match(raw_path)
    if match and os.name != "nt":
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}").expanduser()
    return Path(raw_path).expanduser()


def _as_question_spec(raw: dict[str, Any]) -> QuestionSpec:
    return QuestionSpec(
        qid=str(raw.get("qid") or raw.get("id") or f"q{abs(hash(raw.get('question', ''))) % 10000}"),
        question=str(raw["question"]),
        fault_terms=tuple(str(v).lower() for v in raw.get("fault_terms", ())),
        test_terms=tuple(str(v).lower() for v in raw.get("test_terms", ())),
        safety_terms=tuple(str(v).lower() for v in raw.get("safety_terms", QuestionSpec.__dataclass_fields__["safety_terms"].default)),
        document_terms=tuple(str(v).lower() for v in raw.get("document_terms", ())),
    )


def load_questions(path: str | None) -> list[QuestionSpec]:
    if path is None:
        return list(DEFAULT_QUESTIONS)
    question_path = Path(path)
    raw_text = question_path.read_text(encoding="utf-8")
    if question_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    else:
        loaded = json.loads(raw_text)
        rows = loaded["questions"] if isinstance(loaded, dict) else loaded
    return [_as_question_spec(dict(row)) for row in rows]


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(term and term.lower() in lower for term in terms)


def has_unsafe_advice(answer: str) -> bool:
    lower = answer.lower()
    return any(pattern in lower for pattern in UNSAFE_PATTERNS)


def evidence_is_relevant(question: QuestionSpec, evidence: list[dict[str, Any]]) -> bool:
    if not evidence:
        return False
    haystack = " ".join(
        str(item.get("evidence_excerpt") or item.get("matched_window_text") or "")
        for item in evidence[:8]
    ).lower()
    terms = question.document_terms or question.fault_terms or question.test_terms
    return contains_any(haystack, terms)


def score_answer(
    answer: str,
    question: QuestionSpec,
    *,
    telemetry: dict[str, Any],
    memory_expected: bool,
) -> RubricScore:
    criteria = {
        "correct_likely_fault_types": contains_any(answer, question.fault_terms),
        "correct_test_order": contains_any(answer, question.test_terms)
        and contains_any(answer, ("first", "then", "before", "sequence", "step")),
        "safety_controls_present": contains_any(answer, question.safety_terms),
        "no_unsafe_live_work_advice": not has_unsafe_advice(answer),
        "uses_document_specific_details": contains_any(answer, question.document_terms),
        "no_irrelevant_leakage": not contains_any(answer, IRRELEVANT_LEAKAGE_TERMS),
        "telemetry_memory_used_matches_condition": bool(telemetry.get("memory_used")) is memory_expected,
        "evidence_points_to_relevant_windows": (
            evidence_is_relevant(question, list(telemetry.get("evidence", [])))
            if memory_expected
            else True
        ),
    }
    return RubricScore(
        total=sum(1 for value in criteria.values() if value),
        max_total=len(criteria),
        criteria=criteria,
    )


def _apply_chat_template(tokenizer: Any, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"System: {SYSTEM_PROMPT}\n\nUser: {question}\n\nAssistant:"


def load_runtime(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from chuk_lazarus.inference.backends import TorchInferenceRuntime

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    loader_kwargs = {
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **loader_kwargs)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **loader_kwargs,
    )
    model.to(device)
    model.eval()
    runtime = TorchInferenceRuntime(model, tokenizer, device=device, engine="standard")
    return runtime, tokenizer, device


def build_generation_config(args: argparse.Namespace):
    from chuk_lazarus.inference.generation import GenerationConfig

    return GenerationConfig(
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=args.top_k,
    )


def enumerate_handles(store_root: Path) -> list[Any]:
    from chuk_lazarus.session_retrieval.enumeration import iter_checkpoint_handles

    return list(iter_checkpoint_handles(store_root))


def build_retriever_from_runtime(
    *,
    store_root: Path,
    runtime: Any,
    tokenizer: Any,
    device: str,
):
    import torch

    from chuk_lazarus.session_retrieval.enumeration import load_store
    from chuk_lazarus.session_retrieval.retriever import SessionRetriever

    handles = enumerate_handles(store_root)
    if not handles:
        raise RuntimeError(f"no complete memory checkpoints found under {store_root}")
    first_store = load_store(handles[0])
    crystal_layer = int(first_store.config.crystal_layer)
    for handle in handles[1:]:
        other_layer = int(load_store(handle).config.crystal_layer)
        if other_layer != crystal_layer:
            raise RuntimeError(
                f"crystal_layer mismatch: {handles[0].session_id}={crystal_layer} "
                f"vs {handle.session_id}={other_layer}"
            )
    mem_after_load = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
    return SessionRetriever(
        handles=handles,
        runtime=runtime,
        tokenizer=tokenizer,
        crystal_layer=crystal_layer,
        system_prompt=SYSTEM_PROMPT,
        device=device,
        _mem_after_load_at_init=mem_after_load,
    )


def run_memory_off(
    *,
    runtime: Any,
    tokenizer: Any,
    question: QuestionSpec,
    generation_config: Any,
    store_present: bool,
) -> tuple[str, dict[str, Any]]:
    prompt = _apply_chat_template(tokenizer, question.question)
    result = runtime.generate(prompt, generation_config)
    telemetry = {
        "memory_used": False,
        "store_present": bool(store_present),
        "generation_path": getattr(runtime, "last_generation_path", None),
        "evidence": [],
    }
    return str(result.text), telemetry


def run_memory_on(
    *,
    retriever: Any,
    store_root: Path,
    question: QuestionSpec,
    generation_config: Any,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    from chuk_lazarus.inference.backends.torch_runtime import WarmPenaltyConfig
    from chuk_lazarus.session_retrieval.asi_router import asi_route_candidates
    from chuk_lazarus.session_retrieval.tier_policy import assign_tiers, tier_assignment_to_dict

    candidates = asi_route_candidates(
        retriever.handles,
        question.question,
        retriever.tokenizer,
        candidate_pool=int(args.candidate_pool),
        archive_root=store_root,
    )
    assignments = assign_tiers(
        candidates,
        K_HOT=int(args.k_hot),
        K_WARM=int(args.k_warm),
        candidate_pool=int(args.candidate_pool),
    )
    if not assignments:
        raise RuntimeError(f"no route candidates for question {question.qid}")

    result = retriever.answer_with_kv_direct_multi(
        question.question,
        assignments,
        hot_budget_mib=int(args.hot_budget_mib),
        warm_config=WarmPenaltyConfig(penalty_value=float(args.warm_penalty)),
        generation_config=generation_config,
        query_id=question.qid,
    )
    telemetry = {
        "memory_used": True,
        "store_present": True,
        "store_root": store_root.as_posix(),
        "routing_mode": result.routing_mode,
        "source_session": result.source_session,
        "window_id": result.window_id,
        "strict_assertions": result.strict_assertions,
        "evidence": result.evidence_supports,
        "top_assignments": [tier_assignment_to_dict(item) for item in assignments[:8]],
    }
    return str(result.generated_answer), telemetry


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conditions = resolve_conditions(args.conditions)
    questions = load_questions(args.questions_file)
    store_root = resolve_host_path(args.store_root)
    noise_store_root = resolve_host_path(args.noise_store_root) if args.noise_store_root else None
    if args.output == DEFAULT_OUTPUT and args.store_root != DEFAULT_STORE_ROOT:
        output_path = store_root / "eval_results.jsonl"
    else:
        output_path = resolve_host_path(args.output)

    print(f"[eval] conditions={conditions}")
    print(f"[eval] questions={len(questions)} output={output_path}")

    if args.dry_run:
        store_handles = enumerate_handles(store_root) if store_root.exists() else []
        noise_handles = (
            enumerate_handles(noise_store_root)
            if noise_store_root is not None and noise_store_root.exists()
            else []
        )
        print(f"[dry-run] store handles={len(store_handles)} root={store_root}")
        if "memory_on_tradeguru_plus_noise" in conditions:
            print(
                "[dry-run] noise handles="
                f"{len(noise_handles)} root={noise_store_root or store_root}"
            )
        for question in questions:
            print(f"[dry-run] {question.qid}: {question.question}")
        return 0

    runtime, tokenizer, device = load_runtime(args)
    generation_config = build_generation_config(args)
    store_present = bool(enumerate_handles(store_root)) if store_root.exists() else False

    retrievers: dict[Path, Any] = {}

    def retriever_for(root: Path):
        if root not in retrievers:
            retrievers[root] = build_retriever_from_runtime(
                store_root=root,
                runtime=runtime,
                tokenizer=tokenizer,
                device=device,
            )
        return retrievers[root]

    if output_path.exists():
        output_path.unlink()

    aggregate: dict[str, list[int]] = {condition: [] for condition in conditions}
    warnings: list[str] = []
    for condition in conditions:
        for question in questions:
            started = time.time()
            if condition in MEMORY_CONDITIONS:
                condition_store_root = store_root
                if condition == "memory_on_tradeguru_plus_noise":
                    if noise_store_root is not None and noise_store_root.exists():
                        condition_store_root = noise_store_root
                    else:
                        warnings.append(
                            "memory_on_tradeguru_plus_noise reused --store-root; "
                            "pass --noise-store-root for a true noise condition"
                        )
                answer, telemetry = run_memory_on(
                    retriever=retriever_for(condition_store_root),
                    store_root=condition_store_root,
                    question=question,
                    generation_config=generation_config,
                    args=args,
                )
                memory_expected = True
            else:
                answer, telemetry = run_memory_off(
                    runtime=runtime,
                    tokenizer=tokenizer,
                    question=question,
                    generation_config=generation_config,
                    store_present=store_present if condition == "memory_off_with_store_present" else False,
                )
                memory_expected = False

            score = score_answer(
                answer,
                question,
                telemetry=telemetry,
                memory_expected=memory_expected,
            )
            aggregate[condition].append(score.total)
            row = {
                "schema_version": 1,
                "condition": condition,
                "question_id": question.qid,
                "question": question.question,
                "answer": answer,
                "score": {
                    "total": score.total,
                    "max_total": score.max_total,
                    "criteria": score.criteria,
                },
                "telemetry": telemetry,
                "decode": {
                    "model": args.model,
                    "device": device,
                    "max_new_tokens": int(args.max_new_tokens),
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "top_k": args.top_k,
                },
                "elapsed_seconds": round(time.time() - started, 3),
            }
            _write_jsonl(output_path, row)
            print(
                f"[eval] {condition} {question.qid}: "
                f"{score.total}/{score.max_total}"
            )

    summary = {
        "schema_version": 1,
        "kind": "tradeguru_fault_memory_eval_summary",
        "output": output_path.as_posix(),
        "conditions": conditions,
        "question_count": len(questions),
        "aggregate": {
            condition: {
                "mean_score": (
                    sum(values) / len(values) if values else 0.0
                ),
                "scores": values,
            }
            for condition, values in aggregate.items()
        },
        "warnings": sorted(set(warnings)),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[eval] wrote {output_path}")
    print(f"[eval] wrote {summary_path}")
    if warnings:
        for warning in sorted(set(warnings)):
            print(f"[eval] WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
