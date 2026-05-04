"""Model-free decoder steering policy decisions for David."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .config import AdapterSessionMetadata
from .routing import RoutePacket


STEERING_VERSION = "david-decoder-steering-v1"

LANGUAGE_MARKERS = {
    "javascript": (
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        "javascript",
        "typescript",
        "node",
        "npm",
        "const ",
        "let ",
        "=>",
        "console.log",
    ),
    "python": (
        ".py",
        "python",
        "pytest",
        "def ",
        "self",
        "None",
        "pip",
        "py_compile",
    ),
}

FORBIDDEN_TOKEN_FAMILIES = {
    "javascript": ("python_block_syntax", "python_dunder_names", "python_none_bool_literals"),
    "python": ("javascript_declarations", "javascript_arrow_functions", "javascript_console_globals"),
}

CODE_METHODS = {"repo_patch", "source_dependency", "verify"}


class IncompatibleSteeringScope(ValueError):
    """Raised when decoder steering metadata crosses incompatible scopes."""


@dataclass(frozen=True)
class DecoderSteeringPolicy:
    task_type: str
    target_language: str
    forbidden_token_families: tuple[str, ...]
    alpha_min: float
    alpha_max: float
    logit_lock: bool
    steering_version: str = STEERING_VERSION

    @classmethod
    def for_route(
        cls,
        *,
        route: RoutePacket,
        adapter: AdapterSessionMetadata,
        session_id: str,
        prompt: str = "",
    ) -> "DecoderSteeringPolicy":
        del adapter, session_id
        task_type = _task_type(route.method)
        language = _detect_language(prompt, route)
        is_code = route.method in CODE_METHODS or language in FORBIDDEN_TOKEN_FAMILIES
        alpha_max = 0.35 if is_code else 0.15
        forbidden = FORBIDDEN_TOKEN_FAMILIES.get(language, ())
        return cls(
            task_type=task_type,
            target_language=language,
            forbidden_token_families=forbidden,
            alpha_min=0.0,
            alpha_max=alpha_max,
            logit_lock=bool(forbidden),
        )

    def constraints(self) -> dict[str, Any]:
        return {
            "policy": self.steering_version,
            "task_type": self.task_type,
            "target_language": self.target_language,
            "logit_lock": self.logit_lock,
            "forbidden_token_families": list(self.forbidden_token_families),
            "alpha_bounds": {"min": self.alpha_min, "max": self.alpha_max},
        }

    def scope_metadata(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "steering_version": self.steering_version,
            "target_language": self.target_language,
            "forbidden_token_families": list(self.forbidden_token_families),
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "logit_lock": self.logit_lock,
        }

    def prior_compatible_fields(self, *, adapter: AdapterSessionMetadata) -> dict[str, Any]:
        layer = adapter.kv_target_layer if adapter.kv_target_layer is not None else adapter.boundary_layer
        return {
            "model_id": adapter.model_id,
            "tokenizer_id": adapter.tokenizer_id,
            "adapter_family": adapter.adapter_family,
            "model_revision": adapter.model_revision,
            "insertion_family": adapter.insertion_family,
            "layer": layer,
            "task_type": self.task_type,
            "steering_version": self.steering_version,
        }

    def assert_scope_compatible(self, scope: Mapping[str, Any], *, adapter: AdapterSessionMetadata) -> None:
        expected = self.prior_compatible_fields(adapter=adapter)
        mismatches = []
        for field_name, expected_value in expected.items():
            candidate_value = scope.get(field_name)
            if field_name == "layer":
                candidate_value = scope.get("layer", scope.get("kv_target_layer", scope.get("boundary_layer")))
            if candidate_value is not None and str(candidate_value) != str(expected_value):
                mismatches.append(field_name)
        if mismatches:
            raise IncompatibleSteeringScope(f"decoder steering scope mismatch: {', '.join(mismatches)}")


def _task_type(method: str) -> str:
    if method == "repo_patch":
        return "code_patch"
    if method == "source_dependency":
        return "source_dependency"
    if method == "verify":
        return "verification"
    return method


def _detect_language(prompt: str, route: RoutePacket) -> str:
    evidence_text = " ".join(
        [
            prompt,
            *route.selected_windows,
            *(str(item.get("text", "")) for item in route.evidence),
            str(route.provenance.get("path", "")),
            str(route.provenance.get("language", "")),
        ]
    ).lower()
    scores = {
        language: sum(1 for marker in markers if _contains_marker(evidence_text, marker))
        for language, markers in LANGUAGE_MARKERS.items()
    }
    language, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        return "code" if route.method in CODE_METHODS else "unknown"
    return language


def _contains_marker(text: str, marker: str) -> bool:
    if marker.startswith("."):
        return marker in text
    if marker.strip().isidentifier():
        return re.search(rf"\b{re.escape(marker.strip())}\b", text) is not None
    return marker.lower() in text
