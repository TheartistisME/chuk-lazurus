"""Opt-in bridge from benchmark runners to the standalone David router."""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

NATIVE_ROUTER_BACKEND = "native"
CENTRAL_ROUTER_BACKEND = "central"
ROUTER_BACKEND_CHOICES = (NATIVE_ROUTER_BACKEND, CENTRAL_ROUTER_BACKEND)

TEMPORAL_ORDINAL = "temporal_ordinal"
SYMBOLIC_CHAIN = "symbolic_chain"
DEPENDENCY_SOURCE = "dependency_source"
PATCH_TARGET = "patch_target"
DURABLE_CHAT_MEMORY = "durable_chat_memory"
GENERAL_RECALL = "general_recall"

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTER_WRAPPER_PATH = REPO_ROOT / "David" / "router.py"
CENTRAL_ROUTER_PATH = REPO_ROOT / "David" / "central router.py"

BENCHMARK_ADAPTER_VERSION = "benchmark-harness-v1"
BENCHMARK_MODEL_FAMILY = "synthetic-benchmark"
BENCHMARK_MODEL_REVISION = "central-router-bridge-v1"
BENCHMARK_MODEL_HASH = "benchmark-model-hash-v1"
BENCHMARK_TOKENIZER_ID = "synthetic-tokenizer"
BENCHMARK_TOKENIZER_HASH = "benchmark-tokenizer-hash-v1"
BENCHMARK_ROUTE_LAYER = "benchmark-route"
BENCHMARK_BOUNDARY_LAYER = "benchmark-boundary"
BENCHMARK_INJECTION_LAYER = "benchmark-injection"
BENCHMARK_ROUTE_DIMENSION = 384
BENCHMARK_KV_LAYOUT = "synthetic-kv-v1"
BENCHMARK_INDEX_STATUS = "ready"
APOLLO_MANIFEST_READY_KIND = "benchmark_jit_apollo_sequential_residual"


def add_router_backend_argument(
    parser: Any,
    *,
    flag: str = "--routing-backend",
    dest: str = "routing_backend",
    help_text: str | None = None,
) -> None:
    """Add a native-vs-central routing switch without changing defaults."""

    parser.add_argument(
        flag,
        dest=dest,
        choices=ROUTER_BACKEND_CHOICES,
        default=NATIVE_ROUTER_BACKEND,
        help=help_text
        or "Routing backend to use. Defaults to native to preserve existing benchmark behavior.",
    )


def is_central_backend(value: Any) -> bool:
    return str(value or NATIVE_ROUTER_BACKEND).strip().lower() == CENTRAL_ROUTER_BACKEND


@lru_cache(maxsize=1)
def central_router_module() -> Any:
    module_path = ROUTER_WRAPPER_PATH if ROUTER_WRAPPER_PATH.exists() else CENTRAL_ROUTER_PATH
    if not module_path.exists():
        raise FileNotFoundError(f"central router not found: {module_path}")
    spec = importlib.util.spec_from_file_location(
        "david_central_router",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load central router from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_benchmark_adapter_metadata() -> dict[str, Any]:
    """Return adapter identity for synthetic benchmark/chat routing calls."""

    return {
        "model_family": BENCHMARK_MODEL_FAMILY,
        "model_revision": BENCHMARK_MODEL_REVISION,
        "model_hash": BENCHMARK_MODEL_HASH,
        "tokenizer_id": BENCHMARK_TOKENIZER_ID,
        "tokenizer_hash": BENCHMARK_TOKENIZER_HASH,
        "adapter_version": BENCHMARK_ADAPTER_VERSION,
        "route_layer": BENCHMARK_ROUTE_LAYER,
        "boundary_layer": BENCHMARK_BOUNDARY_LAYER,
        "injection_layer": BENCHMARK_INJECTION_LAYER,
        "route_dimension": BENCHMARK_ROUTE_DIMENSION,
        "kv_layout": BENCHMARK_KV_LAYOUT,
    }


def build_ready_index_metadata(adapter_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return ready index metadata matching the supplied adapter identity."""

    adapter = dict(adapter_metadata or build_benchmark_adapter_metadata())
    adapter_key = "|".join(
        str(adapter.get(key) or "")
        for key in (
            "model_family",
            "model_revision",
            "model_hash",
            "tokenizer_hash",
            "adapter_version",
            "route_layer",
        )
    )
    return {
        "index_id": f"synthetic-central-router:{adapter_key}",
        "adapter_key": adapter_key,
        "status": BENCHMARK_INDEX_STATUS,
        "model_family": adapter.get("model_family"),
        "model_revision": adapter.get("model_revision"),
        "model_hash": adapter.get("model_hash"),
        "tokenizer_id": adapter.get("tokenizer_id"),
        "tokenizer_hash": adapter.get("tokenizer_hash"),
        "adapter_version": adapter.get("adapter_version"),
        "route_layer": adapter.get("route_layer"),
        "boundary_layer": adapter.get("boundary_layer"),
        "injection_layer": adapter.get("injection_layer"),
        "route_dimension": adapter.get("route_dimension"),
        "dimension": adapter.get("route_dimension"),
        "kv_layout": adapter.get("kv_layout"),
    }


def _load_apollo_manifest(manifest_or_path: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], Path | None]:
    if isinstance(manifest_or_path, Mapping):
        return dict(manifest_or_path), None
    path = Path(manifest_or_path)
    return json.loads(path.read_text(encoding="utf-8")), path


def _manifest_id(manifest: Mapping[str, Any], manifest_path: Path | None = None) -> str:
    explicit = manifest.get("manifest_id") or manifest.get("id") or manifest.get("chain_id")
    if explicit:
        return str(explicit)
    if manifest_path is not None:
        return str(manifest_path.resolve())
    return str(manifest.get("kind") or APOLLO_MANIFEST_READY_KIND)


def _apollo_layer(manifest: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = manifest.get(key)
        if value not in (None, ""):
            return str(value)
    arch_config = manifest.get("arch_config")
    if isinstance(arch_config, Mapping):
        for key in keys:
            value = arch_config.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def build_apollo_adapter_metadata(manifest_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Return adapter identity derived from a real Apollo manifest."""

    manifest, manifest_path = _load_apollo_manifest(manifest_or_path)
    arch_config = manifest.get("arch_config") if isinstance(manifest.get("arch_config"), Mapping) else {}
    manifest_id = _manifest_id(manifest, manifest_path)
    source_layer = _apollo_layer(manifest, "source_layer", "crystal_layer", "layer")
    target_layer = _apollo_layer(manifest, "target_layer", "injection_layer")
    hidden_dim = manifest.get("hidden_dim") or arch_config.get("hidden_dim")
    return {
        "model_family": str(
            manifest.get("model_id")
            or manifest.get("model_family")
            or arch_config.get("model_type")
            or "apollo-manifest"
        ),
        "model_revision": str(manifest.get("model_revision") or f"manifest-version-{manifest.get('version', 1)}"),
        "model_hash": str(manifest.get("model_hash") or manifest_id),
        "tokenizer_id": str(manifest.get("tokenizer_id") or manifest.get("tokenizer_name") or "apollo-tokenizer"),
        "tokenizer_hash": str(manifest.get("tokenizer_hash") or manifest_id),
        "adapter_version": f"apollo-manifest:{manifest_id}",
        "route_layer": source_layer,
        "boundary_layer": source_layer,
        "injection_layer": target_layer,
        "route_dimension": int(hidden_dim) if hidden_dim not in (None, "") else None,
        "kv_layout": str(manifest.get("kv_layout") or "apollo-residual-v1"),
    }


def build_apollo_index_metadata(manifest_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Return fast route index metadata derived from a real Apollo manifest."""

    manifest, manifest_path = _load_apollo_manifest(manifest_or_path)
    adapter = build_apollo_adapter_metadata(manifest_or_path)
    manifest_id = _manifest_id(manifest, manifest_path)
    return {
        "index_id": str(manifest.get("index_id") or f"apollo-manifest:{manifest_id}"),
        "adapter_key": adapter.get("adapter_version"),
        "status": str(manifest.get("index_status") or manifest.get("status") or "missing"),
        "model_family": adapter.get("model_family"),
        "model_revision": adapter.get("model_revision"),
        "model_hash": adapter.get("model_hash"),
        "tokenizer_id": adapter.get("tokenizer_id"),
        "tokenizer_hash": adapter.get("tokenizer_hash"),
        "adapter_version": adapter.get("adapter_version"),
        "route_layer": adapter.get("route_layer"),
        "boundary_layer": adapter.get("boundary_layer"),
        "injection_layer": adapter.get("injection_layer"),
        "route_dimension": adapter.get("route_dimension"),
        "dimension": adapter.get("route_dimension"),
        "kv_layout": adapter.get("kv_layout"),
    }


def build_apollo_residual_metadata(
    manifest_or_path: Mapping[str, Any] | str | Path,
    selected_window_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return residual materialization metadata for a Batch 1 Apollo manifest."""

    manifest, manifest_path = _load_apollo_manifest(manifest_or_path)
    base_dir = manifest_path.parent if manifest_path is not None else None
    document_order = tuple(str(item) for item in manifest.get("document_order") or ())
    source_window_refs = tuple(str(item) for item in (selected_window_ids or document_order))
    residual_dir = manifest.get("residual_streams_dir")
    boundaries_dir = manifest.get("boundaries_dir")
    boundary_path = manifest.get("boundary_residual_path")
    manifest_id = _manifest_id(manifest, manifest_path)
    residual_count = len(source_window_refs) if residual_dir else 0
    boundary_count = 1 if boundary_path or boundaries_dir else 0
    return {
        "status": str(manifest.get("status") or "missing"),
        "readiness_source": "apollo_manifest",
        "manifest_path": str(manifest_path) if manifest_path is not None else str(manifest.get("manifest_path") or ""),
        "manifest_id": manifest_id,
        "torch_store_path": str(base_dir) if base_dir is not None else str(manifest.get("torch_store_path") or ""),
        "window_count": int(manifest.get("num_windows") or manifest.get("window_count") or len(document_order) or 0),
        "boundary_residual_count": boundary_count,
        "residual_stream_count": residual_count,
        "boundary_residual_dir": str(base_dir / str(boundaries_dir)) if base_dir is not None and boundaries_dir else str(boundaries_dir or ""),
        "residual_stream_dir": str(base_dir / str(residual_dir)) if base_dir is not None and residual_dir else str(residual_dir or ""),
        "has_residual_streams": bool(residual_dir and residual_count > 0),
        "kv_direct_ready": bool(manifest.get("kv_direct_ready", False)),
        "pending_reason": "" if manifest.get("status") == "ready" else str(manifest.get("pending_reason") or ""),
        "source_layer": _apollo_layer(manifest, "source_layer", "crystal_layer", "layer"),
        "target_layer": _apollo_layer(manifest, "target_layer", "injection_layer"),
        "chain_id": str(manifest.get("chain_id") or ""),
        "parent_manifest_ids": tuple(str(item) for item in manifest.get("parent_manifest_ids") or ()),
        "source_window_refs": source_window_refs,
    }


def _has_adapter_metadata(data: Mapping[str, Any]) -> bool:
    return any(
        key in data and data.get(key) not in (None, "")
        for key in (
            "adapter",
            "adapter_metadata",
            "model_adapter",
            "model_family",
            "model_revision",
            "model_rev",
            "model_hash",
            "tokenizer_id",
            "tokenizer_hash",
            "adapter_version",
            "route_layer",
            "boundary_layer",
            "injection_layer",
            "route_dimension",
            "dimension",
            "kv_layout",
        )
    )


def _has_index_metadata(data: Mapping[str, Any]) -> bool:
    return any(
        key in data and data.get(key) not in (None, "")
        for key in (
            "index",
            "index_metadata",
            "route_index",
            "index_id",
            "adapter_key",
            "adapter_id",
            "index_status",
            "status",
            "index_dimension",
            "index_tokenizer_hash",
            "index_model_hash",
            "index_kv_layout",
            "built_at",
            "index_built_at",
        )
    )


def _has_apollo_residual_metadata(data: Mapping[str, Any]) -> bool:
    return any(
        key in data and data.get(key) not in (None, "")
        for key in (
            "apollo_residual",
            "apollo_residual_metadata",
            "apollo_readiness",
            "apollo_manifest_path",
            "manifest_path",
            "manifest_id",
            "torch_store_path",
            "source_layer",
            "window_count",
            "num_windows",
            "residual_stream_count",
            "boundary_residual_count",
        )
    )


def _has_structured_apollo_residual_metadata(data: Mapping[str, Any]) -> bool:
    return any(
        key in data and data.get(key) not in (None, "")
        for key in (
            "apollo_residual",
            "apollo_residual_metadata",
            "apollo_readiness",
            "manifest_id",
            "torch_store_path",
            "source_layer",
            "window_count",
            "num_windows",
            "residual_stream_count",
            "boundary_residual_count",
        )
    )


def _requires_apollo_metadata(data: Mapping[str, Any]) -> bool:
    return any(
        bool(data.get(key))
        for key in (
            "require_apollo_residual_ready",
            "require_kv_direct",
            "require_residual_stream",
            "require_boundary_residual",
        )
    )


def _metadata_with_harness_boundary(
    metadata: Mapping[str, Any] | None,
    *,
    capability_mode: str,
) -> dict[str, Any]:
    enriched = dict(metadata or {})
    adapter_metadata = (
        dict(enriched.get("adapter_metadata") or enriched.get("model_adapter") or enriched.get("adapter") or {})
        if _has_adapter_metadata(enriched)
        else build_benchmark_adapter_metadata()
    )
    if not _has_adapter_metadata(enriched):
        enriched["adapter_metadata"] = adapter_metadata

    if not _has_index_metadata(enriched):
        enriched["index_metadata"] = build_ready_index_metadata(adapter_metadata)

    enriched.setdefault("capability_request", str(capability_mode or GENERAL_RECALL))
    enriched.setdefault("capability_mode", str(capability_mode or GENERAL_RECALL))
    enriched.setdefault("benchmark_adapter_boundary", "central_router_bridge")
    return enriched


def _window_with_harness_metadata(
    window: Mapping[str, Any],
    *,
    adapter_metadata: Mapping[str, Any],
    index_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    data = dict(window)
    metadata: MutableMapping[str, Any] = dict(data.get("metadata") or {})
    has_adapter = _has_adapter_metadata(data) or _has_adapter_metadata(metadata)
    has_index = _has_index_metadata(data) or _has_index_metadata(metadata)
    if not has_adapter:
        metadata["adapter_metadata"] = dict(adapter_metadata)
    if not has_index:
        metadata["index_metadata"] = dict(index_metadata)
    metadata.setdefault("benchmark_adapter_boundary", "central_router_bridge")
    data["metadata"] = dict(metadata)
    return data


def _windows_with_harness_metadata(
    windows: Sequence[Mapping[str, Any]],
    *,
    adapter_metadata: Mapping[str, Any],
    index_metadata: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    return [
        _window_with_harness_metadata(
            window,
            adapter_metadata=adapter_metadata,
            index_metadata=index_metadata,
        )
        if isinstance(window, Mapping)
        else window
        for window in windows
    ]


def _window_with_real_metadata(
    window: Mapping[str, Any],
    *,
    adapter_metadata: Mapping[str, Any],
    index_metadata: Mapping[str, Any],
    apollo_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    data = dict(window)
    metadata: MutableMapping[str, Any] = dict(data.get("metadata") or {})
    if adapter_metadata and not (_has_adapter_metadata(data) or _has_adapter_metadata(metadata)):
        metadata["adapter_metadata"] = dict(adapter_metadata)
    if index_metadata and not (_has_index_metadata(data) or _has_index_metadata(metadata)):
        metadata["index_metadata"] = dict(index_metadata)
    if apollo_metadata and not (_has_apollo_residual_metadata(data) or _has_apollo_residual_metadata(metadata)):
        metadata["apollo_residual_metadata"] = dict(apollo_metadata)
    data["metadata"] = dict(metadata)
    return data


def _windows_with_real_metadata(
    windows: Sequence[Mapping[str, Any]],
    *,
    adapter_metadata: Mapping[str, Any],
    index_metadata: Mapping[str, Any],
    apollo_metadata: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    return [
        _window_with_real_metadata(
            window,
            adapter_metadata=adapter_metadata,
            index_metadata=index_metadata,
            apollo_metadata=apollo_metadata,
        )
        if isinstance(window, Mapping)
        else window
        for window in windows
    ]


def central_route_plan(
    *,
    query: str,
    capability_mode: str,
    windows: Sequence[Mapping[str, Any]],
    ordinal: int | None = None,
    scope_key: str | None = None,
    canonical_request: str | None = None,
    path_hints: Iterable[str] = (),
    identifiers: Iterable[str] = (),
    entities: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
    hot_count: int = 2,
    warm_count: int = 3,
    readiness_source: str = "synthetic_benchmark",
    ranking_policy: str | None = None,
    activation_query_vector: Sequence[float] | None = None,
    dense_query_vector: Sequence[float] | None = None,
    activation_gate_threshold: float | None = None,
    rrf_k: float | None = None,
    candidate_pool: int | None = None,
) -> Any:
    """Route neutral windows through David's centralized router."""

    module = central_router_module()
    readiness_source_norm = str(readiness_source or "synthetic_benchmark").strip().lower()
    if readiness_source_norm in {"synthetic", "synthetic_benchmark", "benchmark"}:
        request_metadata = _metadata_with_harness_boundary(
            metadata,
            capability_mode=str(capability_mode or GENERAL_RECALL),
        )
        adapter_metadata = dict(
            request_metadata.get("adapter_metadata")
            or request_metadata.get("model_adapter")
            or request_metadata.get("adapter")
            or build_benchmark_adapter_metadata()
        )
        index_metadata = dict(
            request_metadata.get("index_metadata")
            or request_metadata.get("route_index")
            or request_metadata.get("index")
            or build_ready_index_metadata(adapter_metadata)
        )
        route_windows = _windows_with_harness_metadata(
            windows,
            adapter_metadata=adapter_metadata,
            index_metadata=index_metadata,
        )
    else:
        request_metadata = dict(metadata or {})
        request_metadata.setdefault("capability_request", str(capability_mode or GENERAL_RECALL))
        request_metadata.setdefault("capability_mode", str(capability_mode or GENERAL_RECALL))
        request_metadata.setdefault("readiness_source", readiness_source_norm)
        manifest_ref = (
            request_metadata.get("apollo_manifest")
            or request_metadata.get("apollo_manifest_path")
            or request_metadata.get("manifest_path")
        )
        if manifest_ref:
            if not _has_adapter_metadata(request_metadata):
                request_metadata["adapter_metadata"] = build_apollo_adapter_metadata(manifest_ref)
            if not _has_index_metadata(request_metadata):
                request_metadata["index_metadata"] = build_apollo_index_metadata(manifest_ref)
            if not _has_structured_apollo_residual_metadata(request_metadata):
                selected_window_ids = [
                    str(window.get("window_id") or window.get("id"))
                    for window in windows
                    if isinstance(window, Mapping) and (window.get("window_id") or window.get("id")) is not None
                ]
                request_metadata["apollo_residual_metadata"] = build_apollo_residual_metadata(
                    manifest_ref,
                    selected_window_ids=selected_window_ids or None,
                )
        elif _requires_apollo_metadata(request_metadata):
            request_metadata.setdefault(
                "apollo_residual_metadata",
                {
                    "status": "missing",
                    "readiness_source": readiness_source_norm,
                    "pending_reason": "missing Apollo manifest",
                },
            )
        adapter_metadata = dict(
            request_metadata.get("adapter_metadata")
            or request_metadata.get("model_adapter")
            or request_metadata.get("adapter")
            or {}
        )
        index_metadata = dict(
            request_metadata.get("index_metadata")
            or request_metadata.get("route_index")
            or request_metadata.get("index")
            or {}
        )
        apollo_metadata = dict(
            request_metadata.get("apollo_residual_metadata")
            or request_metadata.get("apollo_residual")
            or request_metadata.get("apollo_readiness")
            or {}
        )
        route_windows = _windows_with_real_metadata(
            windows,
            adapter_metadata=adapter_metadata,
            index_metadata=index_metadata,
            apollo_metadata=apollo_metadata,
        )
    if ranking_policy is not None:
        request_metadata["ranking_policy"] = ranking_policy
    if activation_gate_threshold is not None:
        request_metadata["activation_gate_threshold"] = activation_gate_threshold
    if rrf_k is not None:
        request_metadata["rrf_k"] = rrf_k
    if candidate_pool is not None:
        request_metadata["candidate_pool"] = candidate_pool
    request = module.RouteRequest(
        query=str(query or ""),
        capability_mode=str(capability_mode or GENERAL_RECALL),
        ordinal=ordinal,
        scope_key=scope_key,
        canonical_request=canonical_request,
        path_hints=tuple(path_hints or ()),
        identifiers=tuple(identifiers or ()),
        entities=tuple(entities or ()),
        metadata=request_metadata,
        ranking_policy=ranking_policy,
        activation_query_vector=activation_query_vector,
        dense_query_vector=dense_query_vector,
        activation_gate_threshold=activation_gate_threshold,
        rrf_k=rrf_k,
        candidate_pool=candidate_pool,
    )
    router = module.CentralRouter(hot_count=hot_count, warm_count=warm_count)
    return router.route(request, route_windows)


def tier_by_window_id(plan: Any) -> dict[str, str]:
    tiers: dict[str, str] = {}
    for assignment in getattr(plan, "tier_assignments", ()) or ():
        tier = str(getattr(assignment, "tier", ""))
        for window in getattr(assignment, "windows", ()) or ():
            tiers[str(getattr(window, "window_id", ""))] = tier
    return tiers


def ordered_candidate_ids(plan: Any) -> list[str]:
    return [
        str(getattr(candidate, "window_id", getattr(getattr(candidate, "window", None), "window_id", "")))
        for candidate in getattr(plan, "candidates", ()) or ()
    ]


def ordered_materialized_ids(plan: Any) -> list[str]:
    return [
        str(getattr(window, "window_id", ""))
        for window in getattr(getattr(plan, "materialization_plan", None), "windows", ()) or ()
    ]


def candidate_scores_by_id(plan: Any) -> dict[str, float]:
    return {
        str(getattr(candidate, "window_id", getattr(candidate.window, "window_id", ""))): float(
            getattr(candidate, "score", 0.0)
        )
        for candidate in getattr(plan, "candidates", ()) or ()
    }
