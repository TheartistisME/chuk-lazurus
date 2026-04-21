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


def run_harness(
    *,
    output_root: Path,
    device: str,
    model_path: str | None,
    max_new_tokens: int,
    planted_phrase: str,
) -> dict[str, Any]:
    from chuk_lazarus.cross_session_demo.report import build_report
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

    result = chat.retriever.query_exact_id(planted_clause["clause_id"])
    execution = build_execution(planted_clause["clause_id"], planted_phrase, result)
    report = build_report(
        sessions=[{"session_id": saved_session_id}],
        total_tokens=total_tokens,
        executions=[execution],
        repo_root=REPO_ROOT,
    )

    report_path = output_root / "report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    return {
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
    }


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
    args = parser.parse_args(argv)

    try:
        summary = run_harness(
            output_root=args.output_root.resolve(),
            device=args.device,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            planted_phrase=args.planted_phrase,
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
