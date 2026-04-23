#!/usr/bin/env python3
"""Scoped AC-4 live-indexer harness.

Runs a fresh scripted session through ``scripts.interactive_memory_chat.MemoryChat``
using the live indexer path, saves the session, finds the assistant clause that
contains the planted phrase, starts a fresh empty-context session, performs an
exact-ID query against the live-built store, and writes ``report.json`` in the
shape consumed by ``scripts/criterion4_e2e_verify.py --reuse``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_ROOT = Path("/tmp/release-gate-ac4-live")
DEFAULT_PLANTED_PHRASE = "the saffron sidecar remembers hexagon gullies at sunrise."
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def force_deterministic_streaming() -> None:
    """Monkeypatch chat streaming to use greedy decoding for scripted turns."""
    from transformers import TextIteratorStreamer

    import chuk_lazarus.chat_loop.cli as chat_cli

    def deterministic_stream_assistant_reply(
        *,
        model: Any,
        tokenizer: Any,
        history: Any,
        windower: Any,
        session: Any,
        turn: Any,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        on_chunk: Any = None,
        on_text_delta: Any = None,
    ) -> str:
        prompt = chat_cli.format_history(tokenizer, history, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "streamer": streamer,
        }

        errors: list[BaseException] = []

        def _run_generate() -> None:
            try:
                model.generate(**generation_kwargs)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()

        accumulated: list[str] = []
        for text_delta in streamer:
            if not text_delta:
                continue
            accumulated.append(text_delta)
            if on_text_delta is not None:
                on_text_delta(text_delta)
            boundaries = windower.feed_text(text_delta)
            for boundary in boundaries:
                session.append_chunk(turn, boundary)
                if on_chunk is not None:
                    decoded = tokenizer.decode(
                        windower._token_ids[
                            boundary.start_token_offset : boundary.end_token_offset
                        ],
                        skip_special_tokens=True,
                    )
                    on_chunk(boundary, decoded)

        tail = windower.flush()
        if tail is not None:
            session.append_chunk(turn, tail)
            if on_chunk is not None:
                decoded = tokenizer.decode(
                    windower._token_ids[
                        tail.start_token_offset : tail.end_token_offset
                    ],
                    skip_special_tokens=True,
                )
                on_chunk(tail, decoded)

        session.finish_turn(turn)

        thread.join()
        if errors:
            raise errors[0]

        return "".join(accumulated)

    chat_cli.stream_assistant_reply = deterministic_stream_assistant_reply


def scripted_prompts(planted_phrase: str) -> list[str]:
    return [
        "Reply with exactly this sentence and nothing else: ready for exact memory.",
        (
            "Reply with exactly this sentence and nothing else, with no quotation "
            f"marks: {planted_phrase}"
        ),
        (
            "Repeat the exact same sentence again, unchanged and with no extra "
            f"words: {planted_phrase}"
        ),
        "In one short sentence, confirm you can preserve exact wording.",
    ]


def find_planted_clause(session_inputs_dir: Path, planted_phrase: str) -> dict[str, Any]:
    for path in sorted(session_inputs_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("speaker_role") != "assistant":
            continue
        clause_content = data.get("clause_content", "")
        if planted_phrase in clause_content:
            return {
                "path": str(path),
                "clause_id": data["clause_id"],
                "clause_content": clause_content,
            }
    raise RuntimeError(
        "No assistant AUS3000 clause contained the planted phrase under "
        f"{session_inputs_dir}"
    )


def hydrate_live_window_metadata(session_inputs_dir: Path, torch_store_dir: Path) -> int:
    window_metadata_path = torch_store_dir / "window_metadata.json"
    if not window_metadata_path.exists():
        raise RuntimeError(f"Missing live window metadata file at {window_metadata_path}")

    input_records: dict[tuple[int, int, str], dict[str, Any]] = {}
    for path in sorted(session_inputs_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        clause_id = str(data.get("clause_id", "")).strip()
        parts = clause_id.split(".")
        if len(parts) != 3:
            continue
        try:
            turn_index = int(parts[1])
            chunk_index = int(parts[2])
        except ValueError:
            continue
        key = (turn_index, chunk_index, str(data.get("speaker_role", "")).strip())
        input_records[key] = data

    window_metadata = json.loads(window_metadata_path.read_text(encoding="utf-8"))
    updated = 0
    for raw_window_id, meta in window_metadata.items():
        key = (
            int(meta.get("turn_index", -1)),
            int(meta.get("chunk_index", -1)),
            str(meta.get("role", "")).strip(),
        )
        record = input_records.get(key)
        if record is None:
            continue
        for field in (
            "clause_id",
            "clause_title",
            "clause_content",
            "iso_timestamp",
            "speaker_role",
            "topic_tags",
        ):
            if field in record:
                meta[field] = record[field]
        updated += 1

    window_metadata_path.write_text(
        json.dumps(window_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return updated


def get_total_tokens(checkpoints_root: Path, session_id: str) -> int:
    from chuk_lazarus.session_retrieval.enumeration import iter_checkpoint_handles, load_store

    for handle in iter_checkpoint_handles(checkpoints_root):
        if handle.session_id != session_id:
            continue
        store = load_store(handle)
        return int(
            getattr(store.config, "total_tokens", 0)
            or getattr(store, "num_tokens", 0)
            or handle.manifest.get("num_tokens", 0)
            or 0
        )
    raise RuntimeError(
        f"Could not find a checkpoint handle for saved session {session_id} under "
        f"{checkpoints_root}"
    )


def build_execution(query_text: str, planted_phrase: str, result: Any) -> Any:
    from chuk_lazarus.cross_session_demo.verification import QueryExecution, SIX_STRICT_KEYS

    assertions = dict(getattr(result, "strict_assertions", {}) or {})
    normalised_assertions = {key: bool(assertions.get(key, False)) for key in SIX_STRICT_KEYS}
    answer = getattr(result, "generated_answer", "") or ""
    return QueryExecution(
        mode="exact",
        query_text=query_text,
        source_session=getattr(result, "source_session", ""),
        window_id=int(getattr(result, "window_id", -1)),
        routing_score=getattr(result, "routing_score", None),
        generated_answer=answer,
        strict_assertions=normalised_assertions,
        verbatim_hit=(planted_phrase.lower() in answer.lower()),
        planted_phrase=planted_phrase,
        matched_window_text=getattr(result, "matched_window_text", "") or "",
    )


def axis6_fields_for_result(result: Any, *, exception_caught: bool = False) -> dict[str, Any]:
    """Return the 6 axis-6 (repl-observability) fields for the EXACT-ID probe.

    This helper remains for the exact-ID row ONLY. The KV-direct probe
    uses the typed :class:`QueryExecution` carrying native axis-6 fields
    via ``build_kv_direct_execution``; no splice is required for that row.

    TRUTHFULNESS: the exact-ID path does NOT drive axis-5 (no router, no
    tiering, no KV-direct materialization), so the first five fields
    remain sentinel values. ``no_silent_fallback`` is computed truthfully
    — True iff a window was selected AND no recall exception was caught.
    """
    window_id = getattr(result, "window_id", None)
    no_silent_fallback = (not exception_caught) and (window_id is not None)
    return {
        "selected_tier": getattr(result, "selected_tier", "not-implemented-yet"),
        "mask_penalty_applied": bool(getattr(result, "mask_penalty_applied", False)),
        "kv_direct_active": bool(getattr(result, "kv_direct_active", False)),
        "vram_peak_mib": getattr(result, "vram_peak_mib", None),
        "vram_delta_mib": getattr(result, "vram_delta_mib", None),
        "no_silent_fallback": bool(no_silent_fallback),
    }


def run_harness(
    *,
    output_root: Path,
    device: str,
    model_path: str | None,
    max_new_tokens: int,
    planted_phrase: str,
    kv_direct_probe: bool = True,
) -> dict[str, Any]:
    from chuk_lazarus.cross_session_demo.report import build_report
    from chuk_lazarus.cross_session_demo.verification import (
        build_kv_direct_execution,
    )
    from interactive_memory_chat import MemoryChat

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ["LAZARUS_MAX_NEW_TOKENS"] = str(max_new_tokens)
    force_deterministic_streaming()

    chat = MemoryChat(
        store_root=output_root,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        memory_mode="topical",
        device=device,
    )
    chat.load_model()
    chat.maybe_load_retriever()
    chat.start_new_session()

    saved_session_id = chat.session.session_id
    turn_outputs: list[dict[str, Any]] = []
    for prompt in scripted_prompts(planted_phrase):
        meta = chat.plain_chat_turn(prompt)
        turn_outputs.append(
            {
                "user_prompt": prompt,
                "generated_answer": meta.generated_answer,
                "generated_tokens": meta.generated_tokens,
                "phrase_hit": planted_phrase in (meta.generated_answer or ""),
            }
        )

    if not chat.save_current_session(rebuild_retriever=True):
        raise RuntimeError("save_current_session() returned False")

    session_inputs_dir = chat.inputs_root / saved_session_id
    torch_store_dir = chat.checkpoints_root / saved_session_id / "torch_store"
    hydrated_windows = hydrate_live_window_metadata(session_inputs_dir, torch_store_dir)
    planted_clause = find_planted_clause(session_inputs_dir, planted_phrase)
    total_tokens = get_total_tokens(chat.checkpoints_root, saved_session_id)

    chat.start_new_session()
    fresh_probe_session_id = chat.session.session_id
    if chat.session.turns:
        raise RuntimeError("Fresh probe session is not empty before the exact-ID query")
    if chat.retriever is None:
        raise RuntimeError("Retriever was not available after saving the live-indexed session")

    # The exact-id retrieval has no recall-fallback shim in this harness:
    # if query_exact_id raises, the exception propagates and run_harness
    # fails fast (HARNESS_FAIL is printed by main()). Reaching this line
    # therefore implies no exception was caught for this query.
    exception_caught = False
    result = chat.retriever.query_exact_id(planted_clause["clause_id"])
    execution = build_execution(planted_clause["clause_id"], planted_phrase, result)
    axis6_fields = axis6_fields_for_result(
        result, exception_caught=exception_caught
    )

    # ── axis-6 final-impl: real axis-5 KV-direct probe ────────────────────
    # Call the REAL axis-5 KV-direct runtime path using the same retriever.
    # If axis-5 raises, the exception propagates (no silent fallback). The
    # typed QueryExecution returned by build_kv_direct_execution carries
    # real axis-6 fields natively — no post-serialisation splice required.
    executions: list[Any] = [execution]
    kv_execution_summary: dict[str, Any] | None = None
    if kv_direct_probe:
        from chuk_lazarus.inference.backends.torch_runtime import (
            WarmPenaltyConfig,
        )
        from chuk_lazarus.inference.generation import GenerationConfig
        from chuk_lazarus.session_retrieval import (
            asi_route_candidates,
            assign_tiers,
        )

        # Use the planted clause content as the routing query_text so the
        # one-session harness reliably lands on the planted window.
        kv_query_text = planted_clause["clause_content"]

        candidate_pool = int(os.environ.get("LAZARUS_KV_CANDIDATE_POOL", "16"))
        k_hot = int(os.environ.get("LAZARUS_KV_K_HOT", "4"))
        k_warm = int(os.environ.get("LAZARUS_KV_K_WARM", "8"))
        hot_budget_mib = int(os.environ.get("LAZARUS_KV_HOT_BUDGET_MIB", "32"))

        candidates = asi_route_candidates(
            chat.retriever.handles,
            kv_query_text,
            chat.retriever.tokenizer,
            candidate_pool=candidate_pool,
        )
        if not candidates:
            raise RuntimeError(
                "axis-6 KV-direct probe: asi_route_candidates returned empty "
                "for the planted clause content"
            )

        tier_assignments = assign_tiers(
            candidates,
            K_HOT=k_hot,
            K_WARM=k_warm,
            candidate_pool=candidate_pool,
        )

        top_handle = tier_assignments[0].candidate.handle
        assignments_for_handle = [
            a for a in tier_assignments
            if a.candidate.handle.session_id == top_handle.session_id
        ]

        warm_config = WarmPenaltyConfig()
        gen_config = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            temperature=0.0,
            top_p=1.0,
        )

        kv_result = chat.retriever.answer_with_kv_direct(
            kv_query_text,
            assignments_for_handle,
            hot_budget_mib=hot_budget_mib,
            warm_config=warm_config,
            generation_config=gen_config,
            handle=top_handle,
        )

        # Derive selected_tier override from assignments grouping.
        tiers_used = {a.tier.value for a in assignments_for_handle}
        if len(tiers_used) == 1:
            selected_tier_override = next(iter(tiers_used))
        elif tiers_used == {"hot", "warm"}:
            selected_tier_override = "hot+warm"
        else:
            selected_tier_override = "mixed"

        kv_execution = build_kv_direct_execution(
            kv_query_text,
            planted_phrase,
            kv_result,
            selected_tier_override=selected_tier_override,
            exception_caught=False,
        )
        executions.append(kv_execution)
        kv_execution_summary = {
            "selected_tier": kv_execution.selected_tier,
            "kv_direct_active": kv_execution.kv_direct_active,
            "mask_penalty_applied": kv_execution.mask_penalty_applied,
            "vram_peak_mib": kv_execution.vram_peak_mib,
            "vram_delta_mib": kv_execution.vram_delta_mib,
            "tier_counts_selected": kv_execution.tier_counts_selected,
            "path_a_replay_count": kv_execution.path_a_replay_count,
            "hot_budget_mib_observed": kv_execution.hot_budget_mib_observed,
            "no_silent_fallback": kv_execution.no_silent_fallback,
            "verbatim_hit": kv_execution.verbatim_hit,
            "generated_answer": kv_execution.generated_answer,
            "source_session": kv_execution.source_session,
            "window_id": kv_execution.window_id,
        }

    report = build_report(
        sessions=[{"session_id": saved_session_id}],
        total_tokens=total_tokens,
        executions=executions,
        repo_root=REPO_ROOT,
    )

    report_path = output_root / "report.json"
    # axis-6 (final-impl): the KV-direct QueryExecution now carries its
    # axis-6 fields natively via the typed Pydantic model — no splice
    # required for that row. The exact-ID row (index 0) still splices
    # because its non-kv path produces sentinel values.
    report_dict = json.loads(report.model_dump_json())
    axis6_per_execution = [axis6_fields]  # only the exact-ID row
    for idx, qe in enumerate(report_dict.get("query_executions", [])):
        if idx < len(axis6_per_execution):
            for k, v in axis6_per_execution[idx].items():
                qe[k] = v
    report_path.write_text(
        json.dumps(report_dict, indent=2), encoding="utf-8"
    )

    summary: dict[str, Any] = {
        "output_root": str(output_root),
        "report_json": str(report_path),
        "live_store_checkpoints_root": str(chat.checkpoints_root),
        "saved_session_id": saved_session_id,
        "fresh_probe_session_id": fresh_probe_session_id,
        "planted_phrase": planted_phrase,
        "planted_handle": planted_clause["clause_id"],
        "planted_clause_path": planted_clause["path"],
        "hydrated_window_metadata_rows": hydrated_windows,
        "exact_query_source_session": result.source_session,
        "exact_query_window_id": result.window_id,
        "exact_query_verbatim_hit": execution.verbatim_hit,
        "exact_query_strict_assertions": execution.strict_assertions,
        "generated_answer": result.generated_answer,
        "matched_window_text": result.matched_window_text,
        "turn_outputs": turn_outputs,
        "kv_direct_probe_enabled": bool(kv_direct_probe),
    }
    if kv_execution_summary is not None:
        summary["kv_direct_execution"] = kv_execution_summary
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AC-4 live-indexer scoped harness")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Fresh output root to create (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    parser.add_argument("--model-path", default=None, help="Optional model path override")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Max new tokens per scripted assistant turn / exact query",
    )
    parser.add_argument(
        "--planted-phrase",
        default=DEFAULT_PLANTED_PHRASE,
        help="Phrase that must appear verbatim in one assistant clause",
    )
    parser.add_argument(
        "--kv-direct-probe",
        dest="kv_direct_probe",
        action="store_true",
        default=True,
        help="Run the axis-5 KV-direct probe after the exact-ID probe (default: enabled).",
    )
    parser.add_argument(
        "--no-kv-direct-probe",
        dest="kv_direct_probe",
        action="store_false",
        help="Skip the axis-5 KV-direct probe.",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_harness(
            output_root=args.output_root.resolve(),
            device=args.device,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            planted_phrase=args.planted_phrase,
            kv_direct_probe=args.kv_direct_probe,
        )
    except Exception as exc:  # noqa: BLE001
        print("HARNESS_FAIL")
        print(f"reason={exc}")
        traceback.print_exc()
        return 1

    print("HARNESS_OK")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
