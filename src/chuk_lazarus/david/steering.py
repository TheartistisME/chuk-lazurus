"""Model-free decoder steering policy decisions for David."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping

from .config import AdapterSessionMetadata
from .routing import RoutePacket


STEERING_VERSION = "david-decoder-steering-v1"
RUNTIME_HOOK_POLICY = "forbidden-token-family-logit-penalty"
RUNTIME_HOOK_VERSION = "runtime-hook-v1"

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

TOKEN_FAMILY_MARKERS = {
    "python_block_syntax": ("def ", "class ", "self", "import ", "from ", ":\n"),
    "python_dunder_names": ("__init__", "__main__", "__name__"),
    "python_none_bool_literals": ("None", "True", "False"),
    "javascript_declarations": ("const ", "let ", "var ", "function "),
    "javascript_arrow_functions": ("=>", "() =>", "async "),
    "javascript_console_globals": ("console", "console.log", "require("),
}

CODE_METHODS = {"repo_patch", "source_dependency", "verify"}


class IncompatibleSteeringScope(ValueError):
    """Raised when decoder steering metadata crosses incompatible scopes."""


class EmptyForbiddenTokenSet(ValueError):
    """Raised when a locked steering policy cannot resolve forbidden tokens."""


@dataclass(frozen=True)
class _LanguageDetection:
    language: str
    confidence: float
    scores: tuple[tuple[str, int], ...]
    evidence: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class DecoderSteeringPolicy:
    task_type: str
    target_language: str
    forbidden_token_families: tuple[str, ...]
    alpha_min: float
    alpha_max: float
    logit_lock: bool
    steering_version: str = STEERING_VERSION
    language_confidence: float = 0.0
    language_scores: tuple[tuple[str, int], ...] = ()
    language_evidence: tuple[tuple[str, tuple[str, ...]], ...] = ()
    no_lock_reason: str | None = None
    runtime_hook_policy: str = RUNTIME_HOOK_POLICY
    runtime_hook_version: str = RUNTIME_HOOK_VERSION

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
        language_detection = _detect_language_diagnostics(prompt, route)
        language = language_detection.language
        is_code = route.method in CODE_METHODS or language in FORBIDDEN_TOKEN_FAMILIES
        alpha_max = 0.35 if is_code else 0.15
        forbidden = FORBIDDEN_TOKEN_FAMILIES.get(language, ())
        no_lock_reason = None if forbidden else _no_lock_reason(language=language, route=route)
        return cls(
            task_type=task_type,
            target_language=language,
            forbidden_token_families=forbidden,
            alpha_min=0.0,
            alpha_max=alpha_max,
            logit_lock=bool(forbidden),
            language_confidence=language_detection.confidence,
            language_scores=language_detection.scores,
            language_evidence=language_detection.evidence,
            no_lock_reason=no_lock_reason,
        )

    def constraints(self) -> dict[str, Any]:
        return {
            "policy": self.steering_version,
            "task_type": self.task_type,
            "target_language": self.target_language,
            "logit_lock": self.logit_lock,
            "forbidden_token_families": list(self.forbidden_token_families),
            "alpha_bounds": {"min": self.alpha_min, "max": self.alpha_max},
            "no_lock_reason": self.no_lock_reason,
            "diagnostics": self.diagnostics(),
            "runtime_hook": self.runtime_hook_metadata(),
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
            "language_confidence": self.language_confidence,
            "no_lock_reason": self.no_lock_reason,
            "runtime_hook_policy": self.runtime_hook_policy,
            "runtime_hook_version": self.runtime_hook_version,
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "language": {
                "target": self.target_language,
                "confidence": self.language_confidence,
                "scores": dict(self.language_scores),
                "evidence": _tuple_map_to_dict(self.language_evidence),
            },
            "logit_lock": {
                "active": self.logit_lock,
                "reason": "forbidden_token_families_available" if self.logit_lock else "no_logit_lock",
                "no_lock_reason": self.no_lock_reason,
            },
            "forbidden_markers": _forbidden_marker_metadata(self.forbidden_token_families),
        }

    def runtime_hook_metadata(self, *, fail_closed: bool = True) -> dict[str, Any]:
        return {
            "policy": self.runtime_hook_policy,
            "version": self.runtime_hook_version,
            "steering_version": self.steering_version,
            "fail_closed_default": True,
            "fail_closed": fail_closed,
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


@dataclass(frozen=True)
class DecoderLogitHookSpec:
    """Runtime-safe configuration for a decoder logits hook."""

    policy: DecoderSteeringPolicy
    adapter_scope: dict[str, Any]
    forbidden_token_ids: tuple[int, ...]
    alpha: float
    logit_penalty: float
    fail_closed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_penalty(self) -> float:
        return self.alpha * self.logit_penalty

    def scope(self) -> dict[str, Any]:
        return self.adapter_scope | {
            "task_type": self.policy.task_type,
            "target_language": self.policy.target_language,
            "steering_version": self.policy.steering_version,
            "runtime_hook_policy": self.policy.runtime_hook_policy,
            "runtime_hook_version": self.policy.runtime_hook_version,
            "forbidden_token_count": len(self.forbidden_token_ids),
            "forbidden_marker_count": _forbidden_marker_count(self.policy.forbidden_token_families),
            "alpha": self.alpha,
            "logit_penalty": self.logit_penalty,
        }


class DecoderSteeringLogitsProcessor:
    """Transformers-compatible callable that penalizes forbidden token ids."""

    def __init__(self, spec: DecoderLogitHookSpec) -> None:
        self.spec = spec
        self.forbidden_token_ids = spec.forbidden_token_ids

    @classmethod
    def from_policy(
        cls,
        *,
        policy: DecoderSteeringPolicy,
        adapter: AdapterSessionMetadata,
        tokenizer: Any,
        alpha: float | None = None,
        logit_penalty: float = 50.0,
        forbidden_token_ids: Iterable[int] = (),
        scope: Mapping[str, Any] | None = None,
        fail_closed: bool = True,
    ) -> "DecoderSteeringLogitsProcessor":
        candidate_scope = dict(scope or policy.prior_compatible_fields(adapter=adapter))
        policy.assert_scope_compatible(candidate_scope, adapter=adapter)
        _assert_runtime_scope_compatible(adapter=adapter, tokenizer=tokenizer, scope=candidate_scope)

        bounded_alpha = _bounded_alpha(policy, alpha)
        ids = _resolve_forbidden_token_ids(
            tokenizer=tokenizer,
            families=policy.forbidden_token_families,
            explicit_ids=forbidden_token_ids,
        )
        if policy.logit_lock and fail_closed and not ids:
            raise EmptyForbiddenTokenSet("decoder steering requested logit lock but resolved no forbidden token ids")
        spec = DecoderLogitHookSpec(
            policy=policy,
            adapter_scope=adapter.scope(),
            forbidden_token_ids=ids,
            alpha=bounded_alpha,
            logit_penalty=logit_penalty,
            fail_closed=fail_closed,
            metadata={
                "families": list(policy.forbidden_token_families),
                "forbidden_token_count": len(ids),
                "forbidden_markers": _forbidden_marker_metadata(policy.forbidden_token_families),
                "language": policy.diagnostics()["language"],
                "logit_lock": policy.diagnostics()["logit_lock"],
                "runtime_hook": policy.runtime_hook_metadata(fail_closed=fail_closed),
            },
        )
        return cls(spec)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        del input_ids
        if not self.forbidden_token_ids or self.spec.effective_penalty <= 0:
            return scores
        return _penalize_scores(scores, self.forbidden_token_ids, self.spec.effective_penalty)


def build_decoder_logits_processor(
    *,
    policy: DecoderSteeringPolicy,
    adapter: AdapterSessionMetadata,
    tokenizer: Any,
    alpha: float | None = None,
    logit_penalty: float = 50.0,
    forbidden_token_ids: Iterable[int] = (),
    scope: Mapping[str, Any] | None = None,
    fail_closed: bool = True,
) -> DecoderSteeringLogitsProcessor:
    return DecoderSteeringLogitsProcessor.from_policy(
        policy=policy,
        adapter=adapter,
        tokenizer=tokenizer,
        alpha=alpha,
        logit_penalty=logit_penalty,
        forbidden_token_ids=forbidden_token_ids,
        scope=scope,
        fail_closed=fail_closed,
    )


def _task_type(method: str) -> str:
    if method == "repo_patch":
        return "code_patch"
    if method == "source_dependency":
        return "source_dependency"
    if method == "verify":
        return "verification"
    return method


def _detect_language(prompt: str, route: RoutePacket) -> str:
    return _detect_language_diagnostics(prompt, route).language


def _detect_language_diagnostics(prompt: str, route: RoutePacket) -> _LanguageDetection:
    evidence_text = " ".join(
        [
            prompt,
            *route.selected_windows,
            *(str(item.get("text", "")) for item in route.evidence),
            str(route.provenance.get("path", "")),
            str(route.provenance.get("language", "")),
        ]
    ).lower()
    evidence = {
        language: tuple(marker for marker in markers if _contains_marker(evidence_text, marker))
        for language, markers in LANGUAGE_MARKERS.items()
    }
    scores = {language: len(markers) for language, markers in evidence.items()}
    language, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        language = "code" if route.method in CODE_METHODS else "unknown"
    return _LanguageDetection(
        language=language,
        confidence=_language_confidence(language=language, score=score),
        scores=tuple(scores.items()),
        evidence=tuple(evidence.items()),
    )


def _language_confidence(*, language: str, score: int) -> float:
    if score <= 0 or language not in LANGUAGE_MARKERS:
        return 0.0
    return min(1.0, score / 3.0)


def _no_lock_reason(*, language: str, route: RoutePacket) -> str:
    if language == "code":
        return "code_task_without_language_evidence"
    if language == "unknown":
        return "non_code_task_without_language_evidence"
    if route.method not in CODE_METHODS:
        return "non_code_task_without_forbidden_language_family"
    return "target_language_has_no_forbidden_token_family"


def _tuple_map_to_dict(value: tuple[tuple[str, tuple[str, ...]], ...]) -> dict[str, list[str]]:
    return {key: list(items) for key, items in value}


def _forbidden_marker_count(families: Iterable[str]) -> int:
    return sum(len(TOKEN_FAMILY_MARKERS.get(family, ())) for family in families)


def _forbidden_marker_metadata(families: Iterable[str]) -> dict[str, Any]:
    details = [
        {
            "family": family,
            "markers": list(TOKEN_FAMILY_MARKERS.get(family, ())),
            "marker_count": len(TOKEN_FAMILY_MARKERS.get(family, ())),
        }
        for family in families
    ]
    return {
        "count": sum(item["marker_count"] for item in details),
        "families": details,
    }


def _contains_marker(text: str, marker: str) -> bool:
    if marker.startswith("."):
        return marker in text
    if marker.strip().isidentifier():
        return re.search(rf"\b{re.escape(marker.strip())}\b", text) is not None
    return marker.lower() in text


def _bounded_alpha(policy: DecoderSteeringPolicy, alpha: float | None) -> float:
    candidate = policy.alpha_max if alpha is None else float(alpha)
    return min(policy.alpha_max, max(policy.alpha_min, candidate))


def _assert_runtime_scope_compatible(
    *,
    adapter: AdapterSessionMetadata,
    tokenizer: Any,
    scope: Mapping[str, Any],
) -> None:
    tokenizer_id = _tokenizer_identity(tokenizer)
    scope_tokenizer = scope.get("tokenizer_id")
    if scope_tokenizer is not None and str(scope_tokenizer) != str(adapter.tokenizer_id):
        raise IncompatibleSteeringScope("decoder steering scope mismatch: tokenizer_id")
    if tokenizer_id is not None and str(tokenizer_id) != str(adapter.tokenizer_id):
        raise IncompatibleSteeringScope("decoder steering runtime mismatch: tokenizer_id")
    scope_model = scope.get("model_id")
    if scope_model is not None and str(scope_model) != str(adapter.model_id):
        raise IncompatibleSteeringScope("decoder steering scope mismatch: model_id")


def _tokenizer_identity(tokenizer: Any) -> str | None:
    for attr in ("tokenizer_id", "name_or_path"):
        value = getattr(tokenizer, attr, None)
        if value:
            return str(value)
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        value = init_kwargs.get("name_or_path") or init_kwargs.get("tokenizer_id")
        if value:
            return str(value)
    return None


def _resolve_forbidden_token_ids(
    *,
    tokenizer: Any,
    families: Iterable[str],
    explicit_ids: Iterable[int],
) -> tuple[int, ...]:
    ids = {int(token_id) for token_id in explicit_ids}
    for family in families:
        for marker in TOKEN_FAMILY_MARKERS.get(family, ()):
            ids.update(_encode_marker(tokenizer, marker))
    return tuple(sorted(token_id for token_id in ids if token_id >= 0))


def _encode_marker(tokenizer: Any, marker: str) -> tuple[int, ...]:
    if hasattr(tokenizer, "encode"):
        try:
            encoded = tokenizer.encode(marker, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(marker)
        if encoded is None:
            return ()
        return tuple(int(token_id) for token_id in _flatten_token_ids(encoded))
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if token_id is None:
            return ()
        return (int(token_id),)
    return ()


def _flatten_token_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)):
        flattened: list[int] = []
        for item in value:
            flattened.extend(_flatten_token_ids(item))
        return tuple(flattened)
    if hasattr(value, "tolist"):
        return _flatten_token_ids(value.tolist())
    return ()


def _penalize_scores(scores: Any, forbidden_token_ids: tuple[int, ...], penalty: float) -> Any:
    if hasattr(scores, "clone") and hasattr(scores, "index_fill_"):
        result = scores.clone()
        try:
            import torch  # type: ignore[import-not-found]

            ids = torch.tensor(list(forbidden_token_ids), device=result.device, dtype=torch.long)
            result[..., ids] = result[..., ids] - penalty
            return result
        except Exception:
            pass
    if isinstance(scores, list):
        result = [row[:] if isinstance(row, list) else row for row in scores]
        rows = result if result and isinstance(result[0], list) else [result]
        for row in rows:
            if not isinstance(row, list):
                continue
            for token_id in forbidden_token_ids:
                if 0 <= token_id < len(row):
                    row[token_id] = row[token_id] - penalty
        return result
    for token_id in forbidden_token_ids:
        try:
            scores[..., token_id] = scores[..., token_id] - penalty
        except Exception:
            continue
    return scores
