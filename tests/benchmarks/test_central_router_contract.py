from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DAVID_ROOT = REPO_ROOT / "David"
CENTRAL_ROUTER_PATH = DAVID_ROOT / "central router.py"
ROUTER_WRAPPER_PATH = DAVID_ROOT / "router.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _router_module():
    return _load_module(CENTRAL_ROUTER_PATH, "central_router_contract")


def _adapter(router, *, dimension: int = 1024):
    return router.ModelAdapterMetadata(
        model_family="gemma",
        model_revision="gemma-4-test",
        model_hash="model-hash-a",
        tokenizer_id="tok-a",
        tokenizer_hash="tok-hash-a",
        adapter_version="adapter-v1",
        route_layer="layer-12",
        boundary_layer="layer-11",
        injection_layer="layer-13",
        route_dimension=dimension,
        kv_layout="paged-kv-v1",
    )


def _ready_index(router, *, dimension: int = 1024, status: str = "ready"):
    return router.IndexReadinessMetadata(
        index_id="idx-ready",
        adapter_key="adapter-v1",
        status=status,
        dimension=dimension,
        model_family="gemma",
        model_revision="gemma-4-test",
        tokenizer_id="tok-a",
        tokenizer_hash="tok-hash-a",
        model_hash="model-hash-a",
        adapter_version="adapter-v1",
        route_layer="layer-12",
        boundary_layer="layer-11",
        injection_layer="layer-13",
        route_dimension=dimension,
        kv_layout="paged-kv-v1",
    )


def _ready_apollo(router, *, kv_direct_ready: bool = False, refs=("w-hot", "w-warm", "w-cold")):
    return router.ApolloResidualReadinessMetadata(
        status="ready",
        readiness_source="apollo_manifest",
        manifest_path="/tmp/apollo/manifest.json",
        manifest_id="apollo-test-manifest",
        torch_store_path="/tmp/apollo",
        window_count=len(refs),
        boundary_residual_count=1,
        residual_stream_count=len(refs),
        boundary_residual_dir="/tmp/apollo/boundaries",
        residual_stream_dir="/tmp/apollo/residual_streams",
        has_residual_streams=True,
        kv_direct_ready=kv_direct_ready,
        source_layer="12",
        target_layer="13",
        source_window_refs=tuple(refs),
    )


def _windows(router, *, adapter=None, index=None):
    return [
        router.RouteWindow(
            window_id="w-hot",
            text="alpha target route memory",
            session_turn_index=1,
            adapter=adapter,
            index=index,
            metadata={
                "boundary_residual_path": "boundary/w-hot.bin",
                "residual_stream_path": "residual/w-hot.bin",
                "kv_artifact_path": "kv/w-hot.bin",
            },
        ),
        router.RouteWindow(
            window_id="w-warm",
            text="alpha supporting memory",
            session_turn_index=2,
            adapter=adapter,
            index=index,
            metadata={"kv_artifact_path": "kv/w-warm.bin"},
        ),
        router.RouteWindow(
            window_id="w-cold",
            text="distant but eligible memory",
            session_turn_index=3,
            adapter=adapter,
            index=index,
        ),
    ]


def test_router_wrapper_reexports_central_router_api() -> None:
    assert ROUTER_WRAPPER_PATH.exists(), "David/router.py wrapper must be importable"

    wrapper = _load_module(ROUTER_WRAPPER_PATH, "david_router_wrapper_contract")
    central = _router_module()
    exported_names = (
        "CentralRouter",
        "RouteRequest",
        "RouteWindow",
        "RoutePlan",
        "RoutePacket",
        "ModelAdapterMetadata",
        "IndexReadinessMetadata",
        "ApolloResidualReadinessMetadata",
        "MaterializationPlan",
        "CompatibilityProof",
        "JITIndexingRequired",
        "DecodePolicy",
        "VerificationPlan",
        "WriteBackPolicy",
        "CAPABILITY_MODES",
        "GENERAL_RECALL",
        "DURABLE_CHAT_MEMORY",
    )

    for name in exported_names:
        assert hasattr(wrapper, name), f"wrapper missing {name}"
        wrapper_value = getattr(wrapper, name)
        central_value = getattr(central, name)
        if isinstance(central_value, tuple):
            assert wrapper_value == central_value
        else:
            assert getattr(wrapper_value, "__name__", None) == getattr(
                central_value,
                "__name__",
                None,
            )

    wrapper_plan = wrapper.CentralRouter().route(
        wrapper.RouteRequest(query="alpha"),
        [wrapper.RouteWindow(window_id="w-1", text="alpha")],
    )
    assert isinstance(wrapper_plan, wrapper.RoutePlan)


def test_unknown_mode_requires_explicit_metadata_fallback() -> None:
    router = _router_module()
    request = router.RouteRequest(query="alpha", capability_mode="not-a-mode")

    with pytest.raises(ValueError, match="unknown capability mode"):
        router.CentralRouter().route(request, _windows(router))

    fallback_request = router.RouteRequest(
        query="alpha",
        capability_mode="not-a-mode",
        metadata={"allow_mode_fallback": True},
    )
    plan = router.CentralRouter().route(fallback_request, _windows(router))

    assert plan.router_metadata["capability_mode"] == router.GENERAL_RECALL
    assert plan.router_metadata["mode_fallback"] == {
        "requested_mode": "not-a-mode",
        "fallback_mode": router.GENERAL_RECALL,
    }


def test_adapter_compatibility_proof_accepts_matching_windows_and_rejects_mismatch() -> None:
    router = _router_module()
    adapter = _adapter(router)
    ready_index = _ready_index(router)
    request = router.RouteRequest(
        query="alpha",
        adapter=adapter,
        index=ready_index,
    )

    plan = router.CentralRouter().route(
        request,
        _windows(router, adapter=adapter, index=ready_index),
    )

    proof = plan.route_packet.compatibility_proof
    assert proof.adapter_compatible is True
    assert proof.index_ready is True
    assert proof.checked_window_ids == ("w-hot", "w-warm", "w-cold")
    assert proof.adapter_identity["route_dimension"] == 1024
    assert proof.adapter_identity["kv_layout"] == "paged-kv-v1"

    incompatible = _adapter(router, dimension=2048)
    with pytest.raises(RuntimeError, match="adapter metadata incompatible"):
        router.CentralRouter().route(
            request,
            _windows(router, adapter=incompatible, index=ready_index),
        )


@pytest.mark.parametrize(
    ("request_index", "window_index", "message"),
    [
        (None, None, "request index is not ready"),
        ("in_flight", "ready", "request index is not ready"),
        ("ready", None, "window index is not ready"),
        ("ready", "in_flight", "window index is not ready"),
        ("ready", "ready_bad_dimension", "window index is not ready"),
    ],
)
def test_index_readiness_gate_rejects_unready_indexes(
    request_index,
    window_index,
    message,
) -> None:
    router = _router_module()
    adapter = _adapter(router)
    request_index_obj = (
        None
        if request_index is None
        else _ready_index(router, status=request_index)
    )
    if window_index == "ready_bad_dimension":
        window_index_obj = _ready_index(router, dimension=2048)
    elif window_index is None:
        window_index_obj = None
    else:
        window_index_obj = _ready_index(router, status=window_index)

    with pytest.raises(RuntimeError, match=message):
        router.CentralRouter().route(
            router.RouteRequest(query="alpha", adapter=adapter, index=request_index_obj),
            _windows(router, adapter=adapter, index=window_index_obj),
        )


def test_allow_jit_indexing_stops_before_route_plan_is_returned() -> None:
    router = _router_module()
    adapter = _adapter(router)
    request = router.RouteRequest(
        query="alpha",
        adapter=adapter,
        metadata={"allow_jit_indexing": True},
    )

    with pytest.raises(router.JITIndexingRequired) as exc_info:
        router.CentralRouter().route(
            request,
            _windows(router, adapter=adapter, index=None),
        )

    stop = exc_info.value
    assert stop.jit_indexing_actions == (
        "jit_index_request",
        "jit_index_window:w-hot",
        "jit_index_window:w-warm",
        "jit_index_window:w-cold",
    )
    assert stop.compatibility_proof.index_ready is False
    assert stop.compatibility_proof.jit_indexing_actions == stop.jit_indexing_actions
    assert stop.metadata["jit_indexing_stop"] == "pre_route"
    assert request.metadata["jit_indexing_required"] is True
    assert request.metadata["jit_indexing_actions"] == stop.jit_indexing_actions


def test_fast_route_ready_does_not_imply_apollo_residual_ready() -> None:
    router = _router_module()
    adapter = _adapter(router)
    ready_index = _ready_index(router)
    windows = [
        router.RouteWindow(window_id="w-a", text="alpha", adapter=adapter, index=ready_index),
        router.RouteWindow(window_id="w-b", text="alpha beta", adapter=adapter, index=ready_index),
    ]

    plan = router.CentralRouter().route(
        router.RouteRequest(query="alpha", adapter=adapter, index=ready_index),
        windows,
    )

    proof = plan.route_packet.compatibility_proof
    assert proof.index_ready is True
    assert proof.fast_route_ready is True
    assert proof.apollo_residual_ready is False
    assert plan.materialization_plan.fast_route_ready is True
    assert plan.materialization_plan.apollo_residual_ready is False


def test_sparse_apollo_ready_metadata_is_rejected_when_required() -> None:
    router = _router_module()
    adapter = _adapter(router)
    ready_index = _ready_index(router)
    sparse_apollo = router.ApolloResidualReadinessMetadata(status="ready")

    with pytest.raises(RuntimeError, match="sparse"):
        router.CentralRouter().route(
            router.RouteRequest(
                query="alpha",
                adapter=adapter,
                index=ready_index,
                apollo_residual=sparse_apollo,
                metadata={"require_apollo_residual_ready": True},
            ),
            _windows(router, adapter=adapter, index=ready_index),
        )


def test_ready_apollo_residual_metadata_satisfies_required_materialization_contract() -> None:
    router = _router_module()
    adapter = _adapter(router)
    ready_index = _ready_index(router)
    apollo = _ready_apollo(router)

    plan = router.CentralRouter().route(
        router.RouteRequest(
            query="alpha",
            adapter=adapter,
            index=ready_index,
            apollo_residual=apollo,
            metadata={
                "require_apollo_residual_ready": True,
                "require_residual_stream": True,
                "require_boundary_residual": True,
            },
        ),
        [
            router.RouteWindow(window_id="w-hot", text="alpha", adapter=adapter, index=ready_index),
            router.RouteWindow(window_id="w-warm", text="alpha support", adapter=adapter, index=ready_index),
            router.RouteWindow(window_id="w-cold", text="distant alpha", adapter=adapter, index=ready_index),
        ],
    )

    proof = plan.route_packet.compatibility_proof
    assert proof.fast_route_ready is True
    assert proof.apollo_residual_ready is True
    assert proof.apollo_residual_identity["manifest_id"] == "apollo-test-manifest"
    assert plan.materialization_plan.apollo_manifest_path == "/tmp/apollo/manifest.json"
    assert "apollo residual readiness" in plan.route_packet.verification_plan.assertions
    assert "apollo_residual_manifest" in plan.route_packet.verification_plan.required_artifacts


def test_allow_jit_indexing_missing_apollo_readiness_stops_pre_route() -> None:
    router = _router_module()
    adapter = _adapter(router)
    ready_index = _ready_index(router)
    request = router.RouteRequest(
        query="alpha",
        adapter=adapter,
        index=ready_index,
        metadata={
            "allow_jit_indexing": True,
            "require_apollo_residual_ready": True,
        },
    )

    with pytest.raises(router.JITIndexingRequired) as exc_info:
        router.CentralRouter().route(
            request,
            _windows(router, adapter=adapter, index=ready_index),
        )

    stop = exc_info.value
    assert stop.jit_indexing_actions == ("jit_apollo_residual_materialization",)
    assert stop.compatibility_proof.fast_route_ready is True
    assert stop.compatibility_proof.index_ready is True
    assert stop.compatibility_proof.apollo_residual_ready is False
    assert stop.metadata["jit_indexing_stop"] == "pre_route"


def test_kv_direct_ready_is_not_implied_by_fast_route_index_readiness() -> None:
    router = _router_module()
    adapter = _adapter(router)
    ready_index = _ready_index(router)
    plan = router.CentralRouter().route(
        router.RouteRequest(query="alpha", adapter=adapter, index=ready_index),
        [
            router.RouteWindow(window_id="w-a", text="alpha", adapter=adapter, index=ready_index),
            router.RouteWindow(window_id="w-b", text="beta", adapter=adapter, index=ready_index),
        ],
    )

    assert plan.route_packet.compatibility_proof.fast_route_ready is True
    assert plan.materialization_plan.kv_direct_ready is False


@pytest.mark.parametrize(
    "sparse_index",
    [
        "request",
        "window",
    ],
)
def test_active_adapter_rejects_sparse_ready_index_identity(sparse_index: str) -> None:
    router = _router_module()
    adapter = _adapter(router)
    ready_index = _ready_index(router)
    sparse_ready_index = router.IndexReadinessMetadata(
        index_id="idx-sparse",
        status="ready",
    )
    request_index = sparse_ready_index if sparse_index == "request" else ready_index
    window_index = sparse_ready_index if sparse_index == "window" else ready_index

    with pytest.raises(RuntimeError, match=f"{sparse_index} index is not ready"):
        router.CentralRouter().route(
            router.RouteRequest(query="alpha", adapter=adapter, index=request_index),
            _windows(router, adapter=adapter, index=window_index),
        )


def test_active_adapter_rejects_cross_model_ready_index_identity() -> None:
    router = _router_module()
    gemma_adapter = _adapter(router)
    qwen_adapter = router.ModelAdapterMetadata(
        model_family="qwen",
        model_revision="qwen-3-test",
        model_hash="model-hash-qwen",
        tokenizer_id="tok-qwen",
        tokenizer_hash="tok-hash-qwen",
        adapter_version="adapter-qwen",
        route_layer="layer-12",
        boundary_layer="layer-11",
        injection_layer="layer-13",
        route_dimension=1024,
        kv_layout="paged-kv-v1",
    )
    gemma_index = _ready_index(router)

    with pytest.raises(RuntimeError, match="request index is not ready"):
        router.CentralRouter().route(
            router.RouteRequest(query="alpha", adapter=qwen_adapter, index=gemma_index),
            _windows(router, adapter=qwen_adapter, index=gemma_index),
        )

    plan = router.CentralRouter().route(
        router.RouteRequest(query="alpha", adapter=gemma_adapter, index=gemma_index),
        _windows(router, adapter=gemma_adapter, index=gemma_index),
    )

    assert plan.route_packet.compatibility_proof.index_ready is True
    assert plan.route_packet.compatibility_proof.index_identity["model_family"] == "gemma"
    assert plan.route_packet.compatibility_proof.index_identity["tokenizer_id"] == "tok-a"


def test_materialization_plan_carries_payload_decode_verification_and_writeback_fields() -> None:
    router = _router_module()
    adapter = _adapter(router)
    ready_index = _ready_index(router)
    request = router.RouteRequest(
        query="alpha",
        adapter=adapter,
        index=ready_index,
        apollo_residual=_ready_apollo(router, kv_direct_ready=True),
        metadata={
            "token_budget": 512,
            "decode_constraints": {"temperature": 0, "max_new_tokens": 64},
            "verification_expectations": ("selected windows exist", "hashes match"),
            "write_back_targets": ("chat_memory", "code_task_memory"),
            "require_kv_direct": True,
            "require_residual_stream": True,
            "require_boundary_residual": True,
        },
    )

    windows = _windows(router, adapter=adapter, index=ready_index)
    for window in windows:
        window.metadata.setdefault("boundary_residual_path", f"boundary/{window.window_id}.bin")
        window.metadata.setdefault("residual_stream_path", f"residual/{window.window_id}.bin")
        window.metadata.setdefault("kv_artifact_path", f"kv/{window.window_id}.bin")

    plan = router.CentralRouter().route(
        request,
        windows,
    )
    materialization = plan.materialization_plan

    assert materialization.token_budget == 512
    assert materialization.payload_plan == materialization.tier_window_ids
    assert materialization.per_tier_counts.keys() == set(router.TIER_ORDER)
    assert materialization.boundary_residual_paths == (
        "boundary/w-hot.bin",
        "boundary/w-warm.bin",
        "boundary/w-cold.bin",
    )
    assert materialization.residual_stream_paths == (
        "residual/w-hot.bin",
        "residual/w-warm.bin",
        "residual/w-cold.bin",
    )
    assert materialization.kv_direct_ready is True
    assert materialization.insertion_layer == "layer-13"
    assert materialization.decode_constraints == {"temperature": 0, "max_new_tokens": 64}
    assert materialization.verification_expectations == (
        "selected windows exist",
        "hashes match",
    )
    assert materialization.write_back_targets == ("chat_memory", "code_task_memory")


def test_route_packet_exposes_harness_router_contract_surfaces() -> None:
    router = _router_module()
    adapter = _adapter(router)
    ready_index = _ready_index(router)
    request = router.RouteRequest(
        query="alpha",
        adapter=adapter,
        index=ready_index,
        metadata={
            "decode_constraints": {"temperature": 0},
            "verification_expectations": ("selected windows exist",),
            "write_back_targets": ("chat_memory",),
            "allow_user_memory_writeback": True,
        },
    )

    plan = router.CentralRouter().route(
        request,
        _windows(router, adapter=adapter, index=ready_index),
    )
    packet = plan.route_packet

    assert packet.selected_memory == tuple(candidate.window for candidate in plan.candidates)
    assert packet.evidence == plan.evidence_supports
    assert packet.tiering == plan.tier_assignments
    assert packet.compatibility_proof.checked_window_ids == ("w-hot", "w-warm", "w-cold")
    assert packet.materialization_plan is plan.materialization_plan
    assert packet.decode_policy.constraints == {"temperature": 0}
    assert packet.decode_policy.boundary_layer == "layer-11"
    assert packet.decode_policy.injection_layer == "layer-13"
    assert packet.decode_policy.kv_layout == "paged-kv-v1"
    assert "adapter compatibility" in packet.verification_plan.assertions
    assert packet.verification_plan.expectations == ("selected windows exist",)
    assert packet.write_back_policy.targets == ("chat_memory",)
    assert packet.write_back_policy.allow_user_memory_writeback is True


def test_stale_and_superseded_chat_memory_is_filtered_or_requires_explicit_allowance() -> None:
    router = _router_module()
    windows = [
        router.RouteWindow(
            window_id="active",
            text="router updates should be concise",
            memory_scope="chat",
            source_authority="user",
            session_turn_index=3,
            metadata={"user_id": "u-1", "memory_scope": "chat"},
        ),
        router.RouteWindow(
            window_id="stale",
            text="outdated router update preference",
            stale=True,
            memory_scope="chat",
            source_authority="user",
            session_turn_index=4,
            metadata={"user_id": "u-1", "memory_scope": "chat", "stale": True},
        ),
        router.RouteWindow(
            window_id="superseded",
            text="superseded router update preference",
            superseded_by="active",
            memory_scope="chat",
            source_authority="user",
            session_turn_index=5,
            metadata={
                "user_id": "u-1",
                "memory_scope": "chat",
                "superseded_by": "active",
            },
        ),
    ]
    filtered = router.CentralRouter().route(
        router.RouteRequest(
            query="router updates",
            capability_mode=router.DURABLE_CHAT_MEMORY,
            scope_key="chat",
        ),
        windows,
    )

    assert [candidate.window_id for candidate in filtered.candidates] == ["active"]
    assert filtered.router_metadata["filtered_stale_window_ids"] == ("stale",)
    assert filtered.router_metadata["filtered_superseded_window_ids"] == ("superseded",)

    with pytest.raises(AssertionError, match="stale"):
        router.CentralRouter().route(
            router.RouteRequest(query="router updates"),
            [windows[1]],
        )

    allowed = router.CentralRouter().route(
        router.RouteRequest(
            query="router updates",
            metadata={"allow_stale_user_memory": True},
        ),
        [windows[1]],
    )
    assert allowed.selected_candidate.window_id == "stale"


def test_code_task_and_user_temporal_metadata_carry_harness_fields() -> None:
    router = _router_module()

    code_task = router.CodeTaskMetadata(
        workspace_id="ws-1",
        repo_path="/repo/project",
        branch="main",
        base_commit="base-a",
        current_commit="head-a",
        dirty_state_hash="dirty-a",
        symbols=("CentralRouter",),
        imports=("pathlib.Path",),
        exports=("RoutePacket",),
        dependency_edges=("central->tests",),
        test_links=("tests/benchmarks/test_central_router_contract.py",),
        failure_links=("failure-123",),
        patch_target_role="implementation_source",
    )
    temporal = router.UserTemporalMetadata(
        user_id="u-1",
        local_timezone="Australia/Perth",
        local_datetime="2026-05-02T11:00:00+08:00",
        observed_at="2026-05-02T03:00:00Z",
        asserted_at="2026-05-02T03:01:00Z",
        effective_at="2026-05-02T03:02:00Z",
        deadline_at="2026-05-03T00:00:00Z",
        stale_after="2026-06-01T00:00:00Z",
        last_confirmed_at="2026-05-02T03:03:00Z",
        superseded_by=None,
        conflict_ids=("conflict-a",),
        confidence=0.9,
        sensitivity="private",
    )
    chat_artifact = router.ChatUserMemoryArtifact(
        artifact_id="chat-a",
        user_id="u-1",
        artifact_path="/tmp/chat-a.json",
        stale_after="2026-06-01T00:00:00Z",
        conflict_ids=("conflict-a",),
        confidence=0.9,
        sensitivity="private",
    )
    code_artifact = router.CodeTaskMemoryArtifact(
        artifact_id="code-a",
        workspace_id="ws-1",
        repo_path="/repo/project",
        branch="main",
        base_commit="base-a",
        current_commit="head-a",
        dirty_state_hash="dirty-a",
        symbols=("CentralRouter",),
        imports=("pathlib.Path",),
        exports=("RoutePacket",),
        dependency_edges=("central->tests",),
        test_links=("contract",),
        failure_links=("failure-123",),
        patch_target_role="implementation_source",
    )

    assert code_task.workspace_id == "ws-1"
    assert code_task.symbols == ("CentralRouter",)
    assert temporal.local_timezone == "Australia/Perth"
    assert temporal.conflict_ids == ("conflict-a",)
    assert chat_artifact.artifact_path == "/tmp/chat-a.json"
    assert chat_artifact.confidence == 0.9
    assert code_artifact.dependency_edges == ("central->tests",)
    assert code_artifact.patch_target_role == "implementation_source"


def test_code_task_workspace_state_mismatch_rejects_selected_window() -> None:
    router = _router_module()
    request = router.RouteRequest(
        query="central router",
        code_task=router.CodeTaskMetadata(
            workspace_id="ws-1",
            repo_path="/repo/project",
            branch="main",
            base_commit="base-a",
            current_commit="head-a",
            dirty_state_hash="dirty-a",
        ),
    )
    windows = [
        router.RouteWindow(
            window_id="mismatch",
            text="central router",
            code_task=router.CodeTaskMetadata(
                workspace_id="ws-1",
                repo_path="/repo/project",
                branch="feature",
                base_commit="base-a",
                current_commit="head-a",
                dirty_state_hash="dirty-a",
            ),
        )
    ]

    with pytest.raises(AssertionError, match="branch"):
        router.CentralRouter().route(request, windows)


def test_opt_in_artifact_file_checks_verify_disk_existence(tmp_path: Path) -> None:
    router = _router_module()
    kv_path = tmp_path / "w-hot.kv"
    residual_path = tmp_path / "w-hot.residual"
    boundary_path = tmp_path / "w-hot.boundary"
    source_path = tmp_path / "source.py"
    artifact_path = tmp_path / "artifact.json"
    for path in (kv_path, residual_path, boundary_path, source_path, artifact_path):
        path.write_text("ok")

    window = router.RouteWindow(
        window_id="w-hot",
        text="alpha target route memory",
        source_path=str(source_path),
        metadata={
            "kv_artifact_path": str(kv_path),
            "residual_stream_path": str(residual_path),
            "boundary_residual_path": str(boundary_path),
            "artifact_path": str(artifact_path),
        },
    )
    request = router.RouteRequest(
        query="alpha",
        apollo_residual=_ready_apollo(router, kv_direct_ready=True, refs=("w-hot",)),
        metadata={
            "require_kv_direct": True,
            "require_residual_stream": True,
            "require_boundary_residual": True,
            "require_window_files": True,
            "require_artifact_files": True,
        },
    )

    plan = router.CentralRouter().route(request, [window])
    assert plan.selected_candidate.window_id == "w-hot"

    boundary_path.unlink()
    with pytest.raises(AssertionError, match="boundary residual file does not exist"):
        router.CentralRouter().route(request, [window])

    presence_only = router.RouteRequest(
        query="alpha",
        apollo_residual=_ready_apollo(router, kv_direct_ready=True, refs=("w-hot",)),
        metadata={
            "require_kv_direct": True,
            "require_residual_stream": True,
            "require_boundary_residual": True,
        },
    )
    assert router.CentralRouter().route(presence_only, [window]).selected_candidate.window_id == "w-hot"


def test_stale_after_and_superseded_user_temporal_metadata_requires_allowance() -> None:
    router = _router_module()
    stale_window = router.RouteWindow(
        window_id="stale-temporal",
        text="router update preference",
        user_temporal=router.UserTemporalMetadata(
            user_id="u-1",
            stale_after="2026-05-01T00:00:00Z",
        ),
    )
    superseded_window = router.RouteWindow(
        window_id="superseded-temporal",
        text="router update preference",
        user_temporal=router.UserTemporalMetadata(
            user_id="u-1",
            superseded_by="newer-memory",
        ),
    )

    with pytest.raises(AssertionError, match="stale"):
        router.CentralRouter().route(
            router.RouteRequest(query="router", metadata={"current_time": "2026-05-02T00:00:00Z"}),
            [stale_window],
        )
    with pytest.raises(AssertionError, match="superseded"):
        router.CentralRouter().route(router.RouteRequest(query="router"), [superseded_window])

    allowed = router.CentralRouter().route(
        router.RouteRequest(
            query="router",
            metadata={
                "allow_stale_user_memory": True,
                "current_time": "2026-05-02T00:00:00Z",
            },
        ),
        [stale_window, superseded_window],
    )
    assert {candidate.window_id for candidate in allowed.candidates} == {
        "stale-temporal",
        "superseded-temporal",
    }


def test_default_ranking_policy_preserves_score_temporal_window_order() -> None:
    router = _router_module()
    windows = [
        router.RouteWindow(window_id="w-later", text="alpha memory", session_turn_index=2),
        router.RouteWindow(window_id="w-earlier", text="alpha memory", session_turn_index=1),
        router.RouteWindow(window_id="w-cold", text="unrelated", session_turn_index=0),
    ]

    plan = router.CentralRouter().route(router.RouteRequest(query="alpha"), windows)

    assert [candidate.window_id for candidate in plan.candidates] == [
        "w-earlier",
        "w-later",
        "w-cold",
    ]
    assert [candidate.score for candidate in plan.candidates[:2]] == [
        plan.candidates[0].score,
        plan.candidates[0].score,
    ]


def test_utility_v2_ranking_is_opt_in_and_preserves_base_scores() -> None:
    router = _router_module()
    windows = [
        router.RouteWindow(
            window_id="base-top",
            text="alpha memory",
            session_turn_index=1,
            relevance_score=0.20,
            freshness_score=0.0,
            estimated_cost=0.0,
        ),
        router.RouteWindow(
            window_id="utility-top",
            text="alpha memory",
            session_turn_index=2,
            relevance_score=0.10,
            freshness_score=1.0,
            learned_reward=0.20,
            estimated_cost=0.0,
        ),
    ]

    default_plan = router.CentralRouter().route(router.RouteRequest(query="alpha"), windows)
    utility_plan = router.CentralRouter().route(
        router.RouteRequest(query="alpha", ranking_policy="utility-v2"),
        windows,
    )

    assert [candidate.window_id for candidate in default_plan.candidates] == [
        "base-top",
        "utility-top",
    ]
    assert [candidate.window_id for candidate in utility_plan.candidates] == [
        "utility-top",
        "base-top",
    ]
    assert utility_plan.candidates[0].score == default_plan.candidates[1].score
    assert utility_plan.candidates[0].route_trace["ranking_policy"] == "utility-v2"


def test_optional_asi_score_merge_requires_explicit_metadata_switch() -> None:
    router = _router_module()
    windows = [
        router.RouteWindow(window_id="plain", text="alpha memory", session_turn_index=1),
        router.RouteWindow(
            window_id="fresh",
            text="alpha memory",
            session_turn_index=2,
            freshness_score=1.0,
        ),
    ]

    default_plan = router.CentralRouter().route(router.RouteRequest(query="alpha"), windows)
    merged_plan = router.CentralRouter().route(
        router.RouteRequest(query="alpha", metadata={"merge_asi_scores": True}),
        windows,
    )

    assert [candidate.window_id for candidate in default_plan.candidates] == ["plain", "fresh"]
    assert merged_plan.selected_candidate.window_id == "fresh"
    assert merged_plan.selected_candidate.score > default_plan.candidates[1].score


def test_stale_superseded_filtering_is_unchanged_with_activation_utility_ranking() -> None:
    router = _router_module()
    request = router.RouteRequest(
        query="router updates",
        capability_mode=router.DURABLE_CHAT_MEMORY,
        scope_key="chat",
        ranking_policy="utility-v2",
        activation_query_vector=(1.0, 0.0),
    )
    windows = [
        router.RouteWindow(
            window_id="active",
            text="router updates should be concise",
            memory_scope="chat",
            source_authority="user",
            activation_route_vector=(1.0, 0.0),
            metadata={"user_id": "u-1", "memory_scope": "chat"},
        ),
        router.RouteWindow(
            window_id="stale",
            text="router updates should be verbose",
            stale=True,
            memory_scope="chat",
            source_authority="user",
            activation_route_vector=(1.0, 0.0),
            metadata={"user_id": "u-1", "memory_scope": "chat", "stale": True},
        ),
        router.RouteWindow(
            window_id="superseded",
            text="router updates should include old notes",
            superseded_by="active",
            memory_scope="chat",
            source_authority="user",
            activation_route_vector=(1.0, 0.0),
            metadata={"user_id": "u-1", "memory_scope": "chat", "superseded_by": "active"},
        ),
    ]

    plan = router.CentralRouter().route(request, windows)

    assert [candidate.window_id for candidate in plan.candidates] == ["active"]
    assert plan.router_metadata["filtered_stale_window_ids"] == ("stale",)
    assert plan.router_metadata["filtered_superseded_window_ids"] == ("superseded",)
    assert plan.candidates[0].activation_score == 1.0


def test_activation_vector_requirements_preserve_readiness_and_fail_clear_mismatch() -> None:
    router = _router_module()
    adapter = _adapter(router, dimension=2)
    ready_index = _ready_index(router, dimension=2)
    request = router.RouteRequest(
        query="alpha",
        adapter=adapter,
        index=ready_index,
        activation_query_vector=(1.0, 0.0),
        metadata={"require_activation_routes": True},
    )
    windows = [
        router.RouteWindow(
            window_id="w-a",
            text="alpha",
            adapter=adapter,
            index=ready_index,
            activation_route_vector=(1.0, 0.0),
        ),
        router.RouteWindow(
            window_id="w-b",
            text="alpha beta",
            adapter=adapter,
            index=ready_index,
            activation_route_vector=(0.5, 0.5),
        ),
    ]

    plan = router.CentralRouter().route(request, windows)

    assert plan.route_packet.compatibility_proof.fast_route_ready is True
    assert plan.route_packet.compatibility_proof.apollo_residual_ready is False

    bad_window = router.RouteWindow(
        window_id="bad",
        text="alpha",
        adapter=adapter,
        index=ready_index,
        activation_route_vector=(1.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="activation query/window vector"):
        router.CentralRouter().route(request, [bad_window])


def test_materialization_preserves_activation_dense_route_metadata() -> None:
    router = _router_module()
    adapter = _adapter(router, dimension=2)
    ready_index = _ready_index(router, dimension=2)
    request = router.RouteRequest(
        query="alpha",
        adapter=adapter,
        index=ready_index,
        activation_query_vector=(1.0, 0.0),
        dense_query_vector=(0.0, 1.0),
        metadata={
            "require_activation_routes": True,
            "require_dense_vectors": True,
            "verification_expectations": ("selected windows exist",),
        },
    )
    windows = [
        router.RouteWindow(
            window_id="w-hot",
            text="alpha",
            adapter=adapter,
            index=ready_index,
            activation_route_vector=(1.0, 0.0),
            dense_vector=(0.0, 1.0),
            metadata={"kv_artifact_path": "kv/w-hot.bin"},
        ),
        router.RouteWindow(
            window_id="w-warm",
            text="alpha support",
            adapter=adapter,
            index=ready_index,
            activation_route_vector=(0.5, 0.5),
            dense_vector=(0.0, 0.5),
            metadata={"kv_artifact_path": "kv/w-warm.bin"},
        ),
        router.RouteWindow(
            window_id="w-cold",
            text="distant alpha",
            adapter=adapter,
            index=ready_index,
            activation_route_vector=(0.0, 1.0),
            dense_vector=(1.0, 0.0),
            metadata={"kv_artifact_path": "kv/w-cold.bin"},
        ),
    ]

    plan = router.CentralRouter().route(request, windows)

    assert "activation route metadata is preserved for selected windows" in plan.materialization_plan.notes
    assert "dense vector metadata is preserved for selected windows" in plan.materialization_plan.notes
    assert plan.materialization_plan.verification_expectations == (
        "selected windows exist",
        "activation route vectors present",
        "dense vectors present",
    )
    assert plan.materialization_plan.kv_direct_ready is True
    assert set(plan.route_packet.verification_plan.required_artifacts) >= {
        "activation_route_vector",
        "dense_vector",
    }
