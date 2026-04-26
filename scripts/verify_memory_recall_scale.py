#!/usr/bin/env python3
"""Actual-use recall verifier for a completed infinite-memory harness run.

This is a post-harness test: it does not plant new sessions. It reuses a
successful ``scripts/auto_verify_memory_repl.py`` run, samples the planted
100x100 scale markers from ``events.jsonl``, sends real MemoryChat recall
turns, and asserts the generated answer contains the routed marker plus the
expected session/turn identity.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "prod" / "validation" / "repl-autoverify"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class RecallProbe:
    marker: str
    expected_session_id: str
    expected_session_idx: int
    expected_turn_idx: int


@dataclass
class RecallResult:
    marker: str
    expected_session_idx: int
    expected_turn_idx: int
    source_session: str | None
    window_id: int | None
    mode: str | None
    no_silent_fallback: bool
    matched_contains_marker: bool
    answer_contains_marker: bool
    answer_contains_session: bool
    answer_contains_turn: bool
    generated_answer: str
    matched_window_text: str
    elapsed_s: float


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def localize_path(path_text: str) -> Path:
    """Map WSL ``/mnt/c/...`` paths to Windows paths when needed."""
    path = Path(path_text)
    if path.exists():
        return path
    normalized = path_text.replace("\\", "/")
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", normalized)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2).replace("/", "\\")
        candidate = Path(f"{drive}:\\{rest}")
        if candidate.exists():
            return candidate
    return path


def latest_pass_run(output_root: Path, *, min_sessions: int) -> Path:
    summaries = sorted(output_root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for summary_path in summaries:
        summary = load_json(summary_path)
        if summary.get("status") != "PASS":
            continue
        if int(summary.get("sessions", 0) or 0) < min_sessions:
            continue
        return summary_path.parent
    raise RuntimeError(f"No PASS run with sessions >= {min_sessions} under {output_root}")


def load_run_summary(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = latest_pass_run(Path(args.output_root), min_sessions=args.min_sessions)
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"summary.json not found under {run_dir}")
    summary = load_json(summary_path)
    if summary.get("status") != "PASS" and not args.allow_non_pass:
        raise RuntimeError(f"{summary_path} is not a PASS run; use --allow-non-pass to override")
    return run_dir, summary


def parse_scale_probes(events_path: Path) -> list[RecallProbe]:
    probes: list[RecallProbe] = []
    seen: set[str] = set()
    pattern = re.compile(r"belongs to session\s+(\d+)\s+turn\s+(\d+)", re.IGNORECASE)
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"event": "routing.probe"' not in line:
                continue
            event = json.loads(line)
            marker = str(event.get("marker", ""))
            expected_session_id = str(event.get("expected_session", ""))
            if not marker or marker in seen:
                continue
            text = str(event.get("window_text_head", ""))
            match = pattern.search(text)
            if match is None:
                for candidate in event.get("top_candidates", []) or []:
                    candidate_text = str(candidate.get("window_text_head", ""))
                    if marker.lower() in candidate_text.lower():
                        match = pattern.search(candidate_text)
                        break
            if match is None:
                continue
            seen.add(marker)
            probes.append(
                RecallProbe(
                    marker=marker,
                    expected_session_id=expected_session_id,
                    expected_session_idx=int(match.group(1)),
                    expected_turn_idx=int(match.group(2)),
                )
            )
    return probes


def select_probes(probes: list[RecallProbe], sample_size: int) -> list[RecallProbe]:
    if sample_size <= 0 or sample_size >= len(probes):
        return list(probes)
    if sample_size == 1:
        return [probes[0]]
    last = len(probes) - 1
    selected: list[RecallProbe] = []
    seen_indices: set[int] = set()
    for i in range(sample_size):
        idx = round(i * last / (sample_size - 1))
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        selected.append(probes[idx])
    return selected


def load_interactive_memory_chat() -> Any:
    script_path = REPO_ROOT / "scripts" / "interactive_memory_chat.py"
    spec = importlib.util.spec_from_file_location("interactive_memory_chat", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load import spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["interactive_memory_chat"] = module
    spec.loader.exec_module(module)
    return module


def force_deterministic_streaming() -> None:
    """Reuse the production harness' deterministic greedy patch."""
    spec = importlib.util.spec_from_file_location(
        "auto_verify_memory_repl", REPO_ROOT / "scripts" / "auto_verify_memory_repl.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import auto_verify_memory_repl")
    module = importlib.util.module_from_spec(spec)
    sys.modules["auto_verify_memory_repl"] = module
    spec.loader.exec_module(module)

    class _Log:
        def event(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    module.force_deterministic_streaming(_Log())


def contains_session(answer: str, expected: int) -> bool:
    return bool(re.search(rf"\bsession\s*[:=]?\s*{expected}\b", answer, re.IGNORECASE))


def contains_turn(answer: str, expected: int) -> bool:
    return bool(re.search(rf"\bturn\s*[:=]?\s*{expected}\b", answer, re.IGNORECASE))


def make_query(probe: RecallProbe) -> str:
    return (
        f"From memory, use retrieval key {probe.marker}. "
        "What planted scale memory does this key identify? "
        f"Answer exactly as: marker={probe.marker}; session=<number>; "
        "turn=<number>. Do not guess from this prompt; use memory."
    )


def run_probe(chat: Any, probe: RecallProbe, *, mode: str) -> RecallResult:
    chat.memory_mode = mode
    chat.start_new_session()
    query = make_query(probe)
    started = time.time()
    if mode == "kv_direct":
        meta = chat.kv_query_turn(query)
    elif mode == "topical":
        meta = chat.recall_chat_turn(query)
    else:
        raise ValueError(f"unsupported mode {mode!r}")

    elapsed = time.time() - started
    answer = str(getattr(meta, "generated_answer", "") or "")
    matched = str(getattr(meta, "matched_window_text", "") or "")
    return RecallResult(
        marker=probe.marker,
        expected_session_idx=probe.expected_session_idx,
        expected_turn_idx=probe.expected_turn_idx,
        source_session=getattr(meta, "source_session", None),
        window_id=getattr(meta, "window_id", None),
        mode=getattr(meta, "mode", None),
        no_silent_fallback=bool(getattr(meta, "no_silent_fallback", False)),
        matched_contains_marker=probe.marker.lower() in matched.lower(),
        answer_contains_marker=probe.marker.lower() in answer.lower(),
        answer_contains_session=contains_session(answer, probe.expected_session_idx),
        answer_contains_turn=contains_turn(answer, probe.expected_turn_idx),
        generated_answer=answer,
        matched_window_text=matched[:700],
        elapsed_s=elapsed,
    )


def result_passed(result: RecallResult, *, mode: str) -> bool:
    if result.mode != mode:
        return False
    if not result.no_silent_fallback:
        return False
    return (
        result.matched_contains_marker
        and result.answer_contains_marker
        and result.answer_contains_session
        and result.answer_contains_turn
    )


def write_report(report_path: Path, results: list[RecallResult], summary: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Completed repl-autoverify run directory.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--allow-non-pass", action="store_true")
    parser.add_argument("--min-sessions", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=100, help="0 means all parsed probes.")
    parser.add_argument("--mode", choices=("kv_direct", "topical"), default="kv_direct")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--required-hit-rate", type=float, default=0.99)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--quiet-model-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Parse run artifacts without loading the model.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir, harness_summary = load_run_summary(args)
    events_path = localize_path(str(harness_summary.get("events_jsonl") or run_dir / "events.jsonl"))
    store_root = localize_path(str(harness_summary.get("store_root") or run_dir / "store"))
    probes = select_probes(parse_scale_probes(events_path), args.sample_size)
    if not probes:
        raise RuntimeError(f"No scale routing probes found in {events_path}")
    if args.dry_run:
        print(
            f"DRY_RUN SCALE_ACTUAL_RECALL: run_dir={run_dir} "
            f"store_root={store_root} events={events_path} probes={len(probes)} "
            f"mode={args.mode}",
            flush=True,
        )
        return 0

    imc = load_interactive_memory_chat()
    force_deterministic_streaming()
    chat = imc.MemoryChat(
        store_root=store_root,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        memory_mode=args.mode,
        device=args.device,
    )
    chat.load_model()
    chat.maybe_load_retriever()
    if chat.retriever is None:
        raise RuntimeError(f"No retriever could be loaded from {store_root}")

    results: list[RecallResult] = []
    passed = 0
    for idx, probe in enumerate(probes, start=1):
        try:
            if args.quiet_model_output:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_probe(chat, probe, mode=args.mode)
            else:
                result = run_probe(chat, probe, mode=args.mode)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL SCALE_ACTUAL_RECALL probe={idx}/{len(probes)} marker={probe.marker}: {exc!r}")
            print(traceback.format_exc())
            return 1
        results.append(result)
        ok = result_passed(result, mode=args.mode)
        passed += int(ok)
        verdict = "PASS" if ok else "FAIL"
        print(
            f"{verdict} SCALE_ACTUAL_RECALL probe={idx}/{len(probes)} "
            f"marker={probe.marker} expected=session {probe.expected_session_idx} "
            f"turn {probe.expected_turn_idx} elapsed_s={result.elapsed_s:.2f}",
            flush=True,
        )
        if not ok:
            print(
                json.dumps(asdict(result), indent=2, sort_keys=True)[:2400],
                flush=True,
            )

    hit_rate = passed / max(1, len(results))
    final_summary = {
        "run_dir": str(run_dir),
        "store_root": str(store_root),
        "events_path": str(events_path),
        "mode": args.mode,
        "sample_size": len(results),
        "passed": passed,
        "hit_rate": hit_rate,
        "required_hit_rate": args.required_hit_rate,
    }
    report_path = args.report_json or (run_dir / f"scale-actual-recall-{args.mode}.json")
    write_report(report_path, results, final_summary)
    if hit_rate < args.required_hit_rate:
        print(
            f"FAIL SCALE_ACTUAL_RECALL: hit_rate={hit_rate:.3f} "
            f"required={args.required_hit_rate:.3f} report={report_path}",
            flush=True,
        )
        return 1
    print(
        f"PASS SCALE_ACTUAL_RECALL: mode={args.mode} hit_rate={hit_rate:.3f} "
        f"passed={passed}/{len(results)} report={report_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
