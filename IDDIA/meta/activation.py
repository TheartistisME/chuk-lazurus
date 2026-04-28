"""Activation policies for deciding when to spawn an improvement agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .state import DEFAULT_META_ROOT, read_json, utc_now, write_json


@dataclass(frozen=True)
class ActivationPolicy:
    mode: str = "never"
    every_n: int = 3
    target_score: float = 4.0


@dataclass(frozen=True)
class ActivationDecision:
    should_spawn: bool
    reason: str
    state_path: Path
    state: dict[str, Any]


def load_policy(path: Path | None = None) -> ActivationPolicy:
    if path is None:
        return ActivationPolicy()
    payload = read_json(path, default={}) or {}
    return ActivationPolicy(
        mode=str(payload.get("mode", "never")).lower(),
        every_n=max(1, int(payload.get("every_n", payload.get("n", 3)))),
        target_score=float(payload.get("target_score", payload.get("threshold", 4.0))),
    )


def evaluate_activation(
    grade_record: dict[str, Any],
    *,
    policy: ActivationPolicy,
    state_root: Path = DEFAULT_META_ROOT,
) -> ActivationDecision:
    state_path = state_root / "activation_state.json"
    state = read_json(state_path, default={}) or {}
    state.setdefault("total_grades", 0)
    state.setdefault("grades_since_spawn", 0)
    state["total_grades"] += 1
    state["grades_since_spawn"] += 1

    mode = policy.mode
    score = float(grade_record.get("overall_score", 0.0))
    should_spawn = False
    reason = "policy mode never"

    if mode == "always":
        should_spawn = True
        reason = "policy mode always"
    elif mode == "never":
        should_spawn = False
    elif mode in {"every_n", "every-n", "periodic"}:
        should_spawn = state["grades_since_spawn"] >= policy.every_n
        reason = (
            f"{state['grades_since_spawn']} grade(s) since spawn; target cadence {policy.every_n}"
        )
    elif mode in {"threshold", "below_target", "below-target"}:
        should_spawn = score < policy.target_score
        reason = f"grade {score:.2f} compared with target {policy.target_score:.2f}"
    else:
        should_spawn = False
        reason = f"unknown policy mode {policy.mode!r}; defaulting to no spawn"

    if should_spawn:
        state["grades_since_spawn"] = 0
        state["last_spawn_triggered_at"] = utc_now()
        state["last_spawn_grade_id"] = grade_record.get("grade_id")
    state["last_evaluated_at"] = utc_now()
    state["last_decision"] = {"should_spawn": should_spawn, "reason": reason, "mode": mode}
    write_json(state_path, state)
    return ActivationDecision(
        should_spawn=should_spawn,
        reason=reason,
        state_path=state_path,
        state=state,
    )
