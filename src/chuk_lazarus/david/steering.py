"""Model-free decoder steering policy decisions for David."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping

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
            "forbidden_token_count": len(self.forbidden_token_ids),
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
            metadata={"families": list(policy.forbidden_token_families)},
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
