#!/usr/bin/env python3
"""Bounded meta-eval supervisor for eval and tool outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from scripts.eval_meta.context import SIGNOFF_FIELDS, latest_signoff, tail_lines
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.eval_meta.context import SIGNOFF_FIELDS, latest_signoff, tail_lines


@dataclass
class Grade:
    source: str
    kind: str
    status: str
    score: float
    max_score: float
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def percent(self) -> float:
        if self.max_score <= 0:
            return 0.0
        return round((self.score / self.max_score) * 100.0, 2)


@dataclass
class ActivationDecision:
    policy: str
    run_count: int
    allowed_by_policy: bool
    should_spawn: bool
    reason: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _grade_tradeguru_rows(path: Path, rows: list[dict[str, Any]], pass_percent: float) -> Grade:
    total = 0.0
    max_total = 0.0
    by_condition: dict[str, dict[str, Any]] = {}
    weak_rows: list[str] = []

    for row in rows:
        score = row.get("score") if isinstance(row.get("score"), dict) else {}
        row_total = float(score.get("total", 0) or 0)
        row_max = float(score.get("max_total", 0) or 0)
        total += row_total
        max_total += row_max
        condition = str(row.get("condition") or "unknown")
        bucket = by_condition.setdefault(condition, {"scores": [], "max_scores": [], "questions": []})
        bucket["scores"].append(row_total)
        bucket["max_scores"].append(row_max)
        bucket["questions"].append(row.get("question_id") or row.get("question") or "unknown")
        if row_max and (row_total / row_max) * 100.0 < pass_percent:
            weak_rows.append(f"{condition}:{bucket['questions'][-1]}={row_total:g}/{row_max:g}")

    findings: list[str] = []
    recommendations: list[str] = []
    metrics: dict[str, Any] = {"rows": len(rows), "conditions": {}}
    for condition, data in sorted(by_condition.items()):
        scores = data["scores"]
        max_scores = data["max_scores"]
        denom = sum(max_scores)
        mean_percent = round((sum(scores) / denom) * 100.0, 2) if denom else 0.0
        metrics["conditions"][condition] = {
            "count": len(scores),
            "mean_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "mean_percent": mean_percent,
        }
        if mean_percent < pass_percent:
            findings.append(f"{condition} averaged {mean_percent}% below pass threshold {pass_percent}%.")

    off = metrics["conditions"].get("memory_off_empty_store")
    on = metrics["conditions"].get("memory_on_tradeguru_store")
    if off and on and on["mean_percent"] < off["mean_percent"]:
        delta = round(on["mean_percent"] - off["mean_percent"], 2)
        findings.append(f"memory_on_tradeguru_store underperformed memory_off_empty_store by {delta} points.")
        recommendations.append("Investigate retrieval quality, context ranking, and whether memory evidence is distracting generation.")

    if weak_rows:
        findings.append("Weak rows: " + "; ".join(weak_rows[:10]))
        recommendations.append("Start with the lowest-scoring question/condition pairs before changing global policy.")

    if not recommendations:
        recommendations.append("No critical TradeGuru regressions detected; keep monitoring future runs.")

    percent = (total / max_total) * 100.0 if max_total else 0.0
    status = "pass" if percent >= pass_percent and not findings else "needs_improvement"
    return Grade(str(path), "tradeguru_jsonl", status, total, max_total, findings, recommendations, metrics)


def _grade_summary(path: Path, data: dict[str, Any], pass_percent: float) -> Grade:
    aggregate = data.get("aggregate") if isinstance(data.get("aggregate"), dict) else {}
    findings: list[str] = []
    recommendations: list[str] = []
    score = 0.0
    max_score = 0.0
    metrics: dict[str, Any] = {"conditions": {}, "warnings": data.get("warnings") or []}

    for condition, stats in sorted(aggregate.items()):
        if not isinstance(stats, dict):
            continue
        scores = [float(value) for value in stats.get("scores", []) if isinstance(value, (int, float))]
        max_each = max(scores) if scores else float(stats.get("max_score", 0) or 0)
        mean_score = float(stats.get("mean_score", 0) or 0)
        condition_max = max(max_each, mean_score, 1.0)
        percent = round((mean_score / condition_max) * 100.0, 2)
        metrics["conditions"][condition] = {"mean_score": mean_score, "estimated_percent": percent}
        score += mean_score
        max_score += condition_max
        if percent < pass_percent:
            findings.append(f"{condition} estimated at {percent}% below pass threshold {pass_percent}%.")

    warnings = data.get("warnings") or []
    for warning in warnings[:5]:
        findings.append(f"Eval warning: {warning}")
    if findings:
        recommendations.append("Open the paired JSONL for row-level diagnosis and verify warning conditions are intentional.")
    else:
        recommendations.append("Summary looks healthy; inspect paired JSONL before spawning repair work.")
    status = "pass" if max_score and (score / max_score) * 100.0 >= pass_percent and not warnings else "needs_improvement"
    return Grade(str(path), "summary_json", status, score, max_score, findings, recommendations, metrics)


def _grade_text(path: Path, text: str) -> Grade:
    lower = text.lower()
    line_count = len(text.splitlines())
    error_count = sum(lower.count(token) for token in ("traceback", "exception", "error", "failed", "failure"))
    warning_count = lower.count("warning") + lower.count("warn:")
    score = max(0.0, 100.0 - error_count * 20.0 - warning_count * 5.0)
    findings: list[str] = []
    if error_count:
        findings.append(f"Detected {error_count} error/failure markers.")
    if warning_count:
        findings.append(f"Detected {warning_count} warning markers.")
    recommendations = (
        ["Inspect the first traceback/error block and rerun the failing tool with a narrower target."]
        if findings
        else ["No obvious failure markers found in generic tool output."]
    )
    status = "pass" if score >= 80.0 else "needs_improvement"
    return Grade(
        str(path),
        "generic_text",
        status,
        score,
        100.0,
        findings,
        recommendations,
        {"lines": line_count, "error_markers": error_count, "warning_markers": warning_count},
    )


def grade_file(path: Path, pass_percent: float = 80.0) -> Grade:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".jsonl"):
        rows = _read_jsonl(path)
        if rows and any("score" in row and "condition" in row for row in rows):
            return _grade_tradeguru_rows(path, rows, pass_percent)
        return _grade_text(path, path.read_text(encoding="utf-8", errors="replace"))
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            return _grade_text(path, path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, dict) and "aggregate" in data:
            return _grade_summary(path, data, pass_percent)
        return _grade_text(path, json.dumps(data, indent=2, sort_keys=True))
    return _grade_text(path, path.read_text(encoding="utf-8", errors="replace"))


def decide_activation(policy: str, run_count: int, grade: Grade, spawn_on_pass: bool = False) -> ActivationDecision:
    normalized = policy.strip().lower()
    allowed = False
    reason = ""
    if normalized == "always":
        allowed = True
        reason = "policy allows every run"
    elif normalized == "never":
        allowed = False
        reason = "policy disables spawning"
    elif normalized.startswith("every:"):
        interval_text = normalized.split(":", 1)[1]
        try:
            interval = int(interval_text)
        except ValueError as exc:
            raise ValueError(f"invalid activation policy interval: {policy}") from exc
        if interval <= 0:
            raise ValueError("activation interval must be positive")
        allowed = run_count > 0 and run_count % interval == 0
        reason = f"run_count {run_count} {'is' if allowed else 'is not'} divisible by {interval}"
    else:
        raise ValueError("activation policy must be 'always', 'never', or 'every:N'")

    needs_work = grade.status != "pass" or spawn_on_pass
    should_spawn = allowed and needs_work
    if allowed and not needs_work:
        reason += "; grade passed so spawn suppressed"
    return ActivationDecision(policy, run_count, allowed, should_spawn, reason)


def build_prompt(grade: Grade, changelog_tail: str, signoff: dict[str, Any]) -> str:
    signoff_text = signoff.get("raw_text") or json.dumps(signoff, indent=2, sort_keys=True)
    recommendations = "\n".join(f"- {item}" for item in grade.recommendations)
    findings = "\n".join(f"- {item}" for item in grade.findings) or "- No findings."
    return f"""You are a bounded improvement agent for this repository.

Objective:
Improve the system based only on the meta-eval grade and recommendations below.

Grade:
{json.dumps(asdict(grade) | {"percent": grade.percent}, indent=2, sort_keys=True)}

Findings:
{findings}

Recommendations:
{recommendations}

Changelog tail:
{changelog_tail}

Latest signoff:
{signoff_text}

Mandatory signoff format:
Return a signoff record with these exact fields:
- files_modified
- agent_objective
- tldr_of_change
- dependencies_mandatory_for_system_understanding

Bounds:
- Prefer the smallest useful change.
- Do not use network access.
- Do not modify unrelated files.
- Run focused tests and report commands/results.
"""


def run_vee_spawn(prompt: str, *, spawn: bool, vee_binary: str = "vee") -> dict[str, Any]:
    vee_path = shutil.which(vee_binary)
    if not spawn:
        return {"dry_run": True, "spawned": False, "vee_path": vee_path, "command": None}
    if vee_path is None:
        raise RuntimeError(f"--spawn requested but {vee_binary!r} was not found on PATH")
    completed = subprocess.run(
        [vee_path, "spawn"],
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "dry_run": False,
        "spawned": completed.returncode == 0,
        "vee_path": vee_path,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Eval JSONL/summary JSON or generic tool output file.")
    parser.add_argument("--activation-policy", default="never", help="'always', 'never', or 'every:N'.")
    parser.add_argument("--run-count", type=int, default=1, help="Current run count for every:N activation.")
    parser.add_argument("--pass-percent", type=float, default=80.0)
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--signoff-dir", type=Path, default=Path(".vee/signoffs"))
    parser.add_argument("--prompt-out", type=Path, help="Optional file to write the generated vee prompt.")
    parser.add_argument("--json-out", type=Path, help="Optional file to write supervisor JSON.")
    parser.add_argument("--spawn", action="store_true", help="Actually run 'vee spawn'. Default is dry-run.")
    parser.add_argument("--spawn-on-pass", action="store_true", help="Allow spawn even when grade passes.")
    parser.add_argument("--vee-binary", default="vee")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grade = grade_file(args.input, args.pass_percent)
    decision = decide_activation(args.activation_policy, args.run_count, grade, args.spawn_on_pass)
    changelog_tail = tail_lines(args.changelog, 100)
    signoff = latest_signoff(args.signoff_dir)
    prompt = build_prompt(grade, changelog_tail, signoff)
    spawn_result = run_vee_spawn(prompt, spawn=args.spawn and decision.should_spawn, vee_binary=args.vee_binary)

    payload = {
        "grade": asdict(grade) | {"percent": grade.percent},
        "activation": asdict(decision),
        "prompt": prompt,
        "spawn": spawn_result,
    }
    if args.prompt_out:
        args.prompt_out.write_text(prompt, encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

