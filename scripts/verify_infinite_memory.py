#!/usr/bin/env python3
"""Fail-closed verifier for infinite-memory + bounded-KV claims.

This is intentionally not a unit-test file. The puzzle class we care about is
composition bugs in the real chat stack, so this runner is wired around the
same entry point operators use: ``scripts/interactive_memory_chat.py`` and its
``MemoryChat`` class.

Design rule: a missing proof is a FAIL, never a skip. The target must exit 0
only after every invariant below has a runnable Linux/CUDA proof chain.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_REPORT = Path("/tmp/lazarus-infinite-memory-report.jsonl")


@dataclass(frozen=True)
class Invariant:
    code: str
    dimension: str
    invariant: str
    proof_requires: str


@dataclass(frozen=True)
class ProbeResult:
    code: str
    status: str
    evidence: str
    elapsed_s: float


class ProbeFailure(RuntimeError):
    """A named invariant failed with a root-cause label."""

    def __init__(self, code: str, root_cause: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.root_cause = root_cause
        self.detail = detail


INVARIANTS: tuple[Invariant, ...] = (
    Invariant(
        "I1",
        "infinite-memory",
        "Routing precision stays above threshold across many saved sessions.",
        "Build 10/100/1000 planted session stores and assert routing@1 hit-rate.",
    ),
    Invariant(
        "I2",
        "infinite-memory",
        "Single-session deep live chains produce coherent residuals.",
        "Compare offline-built boundaries with live-indexed boundaries for same text.",
    ),
    Invariant(
        "I3",
        "infinite-memory",
        "Disk growth is linear in content, with no quadratic overhead.",
        "Save many sessions and assert bytes/session slope remains flat.",
    ),
    Invariant(
        "I4",
        "infinite-memory",
        "Recent-fact recency bias is tunable and correct.",
        "Plant old and new versions of a fact and assert the configured winner.",
    ),
    Invariant(
        "I5",
        "infinite-memory",
        "Contradictions resolve deterministically.",
        "Plant contradictions and assert the emitted answer follows policy.",
    ),
    Invariant(
        "I6",
        "infinite-memory",
        "Cross-session entity binding works.",
        "Plant an entity in an old session and answer an anaphoric question later.",
    ),
    Invariant(
        "I7",
        "infinite-memory",
        "Consolidation and pruning preserve key facts.",
        "Run compaction/pruning and prove planted markers remain retrievable.",
    ),
    Invariant(
        "K1",
        "bounded-kv",
        "VRAM peak stays under ceiling regardless of candidate-pool size.",
        "Run candidate_pool=16/64/256 and assert vram_peak_mib plateaus.",
    ),
    Invariant(
        "K2",
        "bounded-kv",
        "_apply_token_budget is binding under stress.",
        "Overfill the fact list and assert the governor drops the lowest scores.",
    ),
    Invariant(
        "K3",
        "bounded-kv",
        "HOT bonus and WARM penalty steer attention as documented.",
        "Probe L29 attention scores with and without tier bias and assert shift.",
    ),
    Invariant(
        "K4",
        "bounded-kv",
        "COLD tier is true exclusion after softmax.",
        "Probe COLD slots and assert post-softmax attention is exactly zero.",
    ),
    Invariant(
        "K5",
        "bounded-kv",
        "Within-session prefill reuses prior turn KV correctly.",
        "Time consecutive turns and assert per-turn prefill cost stays flat.",
    ),
    Invariant(
        "K6",
        "bounded-kv",
        "Mid-turn crash leaves the store consistent.",
        "Kill a child REPL mid-turn and assert the next load is clean or absent.",
    ),
    Invariant(
        "K7",
        "bounded-kv",
        "Parallel REPLs do not corrupt each other's stores.",
        "Run two MemoryChat processes against one root and assert no cross-write.",
    ),
    Invariant(
        "K8",
        "bounded-kv",
        "The no_silent_fallback flag is honest.",
        "Force fallback and assert WARN plus no_silent_fallback=False.",
    ),
    Invariant(
        "K9",
        "bounded-kv",
        "Injection hook fires exactly once per call and does not latch.",
        "Run 100 recall turns and assert one hook fire and no orphan hooks each.",
    ),
    Invariant(
        "E1",
        "end-to-end",
        "A planted fact in session N is retrievable in session N+1.",
        "Save, start a new session, recall, and assert planted marker hit.",
    ),
    Invariant(
        "E2",
        "end-to-end",
        "A planted fact is retrievable at session-distance 100.",
        "Repeat E1 with 100 intervening sessions.",
    ),
    Invariant(
        "E3",
        "end-to-end",
        "topical, vec_inject, and kv_direct retrieve the same planted fact.",
        "Query the same marker in all three modes and assert all hit.",
    ),
    Invariant(
        "E4",
        "end-to-end",
        "Successful recalls keep no_silent_fallback=True across 100 turns.",
        "Count successful recalls and fallback warnings in a 100-turn run.",
    ),
    Invariant(
        "E5",
        "end-to-end",
        "Auto-promotion topical to kv_direct preserves recall.",
        "Trigger vec_inject.npz write and assert the next recall still works.",
    ),
    Invariant(
        "E6",
        "end-to-end",
        "/exact, /entity, and /query probe without mutating session state.",
        "Run probes mid-session and assert transcript/history are unchanged.",
    ),
    Invariant(
        "E7",
        "end-to-end",
        "Memory mode off disables all retrieval paths.",
        "Set mode=off and assert no retrieval, no injection, plain chat only.",
    ),
)


def parse_codes(raw: str | None) -> set[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return {part.strip().upper() for part in raw.split(",") if part.strip()}


def load_memory_chat_class() -> type:
    script_path = REPO_ROOT / "scripts" / "interactive_memory_chat.py"
    spec = importlib.util.spec_from_file_location("interactive_memory_chat", script_path)
    if spec is None or spec.loader is None:
        raise ProbeFailure(
            "PREFLIGHT",
            "repl_import_failed",
            f"Could not load import spec for {script_path}",
        )
    module = importlib.util.module_from_spec(spec)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    spec.loader.exec_module(module)
    memory_chat = getattr(module, "MemoryChat", None)
    if memory_chat is None:
        raise ProbeFailure(
            "PREFLIGHT",
            "memorychat_missing",
            "interactive_memory_chat.py did not expose MemoryChat",
        )
    return memory_chat


class InfiniteMemoryVerifier:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.selected_codes = parse_codes(args.only)
        self.report_path: Path | None = args.report_jsonl
        self.memory_chat_class: type | None = None
        self._tmpdir: tempfile.TemporaryDirectory | None = None

    def close(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def store_root(self) -> Path:
        if self.args.store_root is not None:
            path = self.args.store_root
            path.mkdir(parents=True, exist_ok=True)
            return path
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="lazarus-imkv-")
        return Path(self._tmpdir.name)

    def run(self) -> int:
        if self.args.list:
            for invariant in INVARIANTS:
                print(
                    f"{invariant.code}\t{invariant.dimension}\t"
                    f"{invariant.invariant}"
                )
            return 0

        failures = 0
        started = time.time()
        try:
            self.preflight()
            for invariant in INVARIANTS:
                if self.selected_codes and invariant.code not in self.selected_codes:
                    continue
                try:
                    result = self.run_one(invariant)
                except ProbeFailure as exc:
                    failures += 1
                    self.emit_failure(exc)
                    if not self.args.continue_on_fail:
                        return 1
                else:
                    self.emit_pass(result)
            if failures:
                return 1
            elapsed = time.time() - started
            print(f"PASS ALL: {self.selected_label()} in {elapsed:.2f}s")
            return 0
        finally:
            self.close()

    def selected_label(self) -> str:
        if self.selected_codes:
            return ",".join(sorted(self.selected_codes))
        return "I1-I7,K1-K9,E1-E7"

    def preflight(self) -> None:
        if not self.args.skip_env_check:
            if platform.system() != "Linux":
                raise ProbeFailure(
                    "PREFLIGHT",
                    "not_linux",
                    "This proof target must run on Linux/CUDA, not Windows/macOS.",
                )
            try:
                import torch
            except Exception as exc:  # noqa: BLE001
                raise ProbeFailure(
                    "PREFLIGHT",
                    "torch_import_failed",
                    f"Could not import torch: {exc}",
                ) from exc
            if not torch.cuda.is_available():
                raise ProbeFailure(
                    "PREFLIGHT",
                    "cuda_unavailable",
                    "torch.cuda.is_available() returned False.",
                )
            device_name = torch.cuda.get_device_name(0)
            total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
            if total_mib < self.args.min_vram_mib:
                raise ProbeFailure(
                    "PREFLIGHT",
                    "vram_below_floor",
                    f"GPU {device_name!r} has {total_mib:.0f} MiB, "
                    f"floor is {self.args.min_vram_mib} MiB.",
                )
            print(f"PASS PREFLIGHT: Linux/CUDA visible on {device_name}")

        self.memory_chat_class = load_memory_chat_class()
        print("PASS PREFLIGHT: imported scripts/interactive_memory_chat.py::MemoryChat")

    def run_one(self, invariant: Invariant) -> ProbeResult:
        probe = getattr(self, f"probe_{invariant.code.lower()}", None)
        if probe is None:
            raise ProbeFailure(
                invariant.code,
                "probe_missing",
                "No probe method exists for this invariant.",
            )
        started = time.time()
        evidence = probe(invariant)
        return ProbeResult(
            code=invariant.code,
            status="PASS",
            evidence=evidence,
            elapsed_s=time.time() - started,
        )

    def emit_pass(self, result: ProbeResult) -> None:
        print(
            f"PASS {result.code}: {result.evidence} "
            f"elapsed_s={result.elapsed_s:.2f}",
            flush=True,
        )
        self.write_report(result)

    def emit_failure(self, exc: ProbeFailure) -> None:
        result = ProbeResult(
            code=exc.code,
            status="FAIL",
            evidence=f"root_cause={exc.root_cause}; {exc.detail}",
            elapsed_s=0.0,
        )
        print(f"FAIL {exc.code}: {result.evidence}", flush=True)
        self.write_report(result)

    def write_report(self, result: ProbeResult) -> None:
        if self.report_path is None:
            return
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")

    def fail_unverified(self, invariant: Invariant) -> str:
        raise ProbeFailure(
            invariant.code,
            "proof_not_implemented",
            (
                "Runnable probe slot exists, but no Linux/CUDA proof chain has "
                f"been implemented yet. Required proof: {invariant.proof_requires}"
            ),
        )

    def probe_i1(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_i2(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_i3(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_i4(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_i5(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_i6(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_i7(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_k1(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_k2(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_k3(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_k4(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_k5(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_k6(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_k7(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_k8(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_k9(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_e1(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_e2(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_e3(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_e4(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_e5(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_e6(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)

    def probe_e7(self, invariant: Invariant) -> str:
        return self.fail_unverified(invariant)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed infinite-memory + bounded-KV verifier. Exits 0 only "
            "when every selected invariant has a real PASS."
        )
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List invariant codes and exit without running probes.",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated invariant codes to run, for example E1,K2,K8.",
    )
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Collect all failures instead of halting on the first one.",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Developer escape hatch for listing/import checks off Linux/CUDA.",
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        help="Store root for generated proof fixtures. Defaults to a temp dir.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional Gemma-4 model path/id for future live probes.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for future live probes. Default: cuda.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Generation cap for future live probes. Default: 24.",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=100,
        help="Default session count for scale probes. Default: 100.",
    )
    parser.add_argument(
        "--turns-per-session",
        type=int,
        default=100,
        help="Default per-session turn count for scale probes. Default: 100.",
    )
    parser.add_argument(
        "--candidate-pools",
        default="16,64,256",
        help="Candidate-pool sizes for bounded-KV stress probes.",
    )
    parser.add_argument(
        "--routing-hit-threshold",
        type=float,
        default=0.99,
        help="Minimum routing@1 hit rate for routing scale probes.",
    )
    parser.add_argument(
        "--vram-ceiling-mib",
        type=float,
        default=30_000.0,
        help="Maximum allowed peak VRAM for bounded-KV probes.",
    )
    parser.add_argument(
        "--min-vram-mib",
        type=float,
        default=30_000.0,
        help="Linux/CUDA preflight VRAM floor. Default targets RTX 5090 32GB.",
    )
    parser.add_argument(
        "--report-jsonl",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Append JSONL probe results here. Default: {DEFAULT_REPORT}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verifier = InfiniteMemoryVerifier(args)
    try:
        return verifier.run()
    except ProbeFailure as exc:
        verifier.emit_failure(exc)
        return 1
    except KeyboardInterrupt:
        print("FAIL INTERRUPTED: root_cause=keyboard_interrupt", flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
