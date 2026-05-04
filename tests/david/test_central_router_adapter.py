from __future__ import annotations

from dataclasses import dataclass, field
import sys
from typing import Any

from chuk_lazarus.david.central_router_adapter import (
    CentralRouterAdapter,
    load_david_router_wrapper,
)
from chuk_lazarus.david.product_router import ProductRouter


@dataclass
class FakeRouteRequest:
    query: str
    capability_mode: str
    scope_key: str | None = None
    ordinal: int | None = None
    path_hints: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeRouteWindow:
    window_id: str
    text: str = ""
    scope_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    lexical_score: float | None = None


@dataclass
class FakeCandidate:
    window: FakeRouteWindow
    score: float
    reasons: tuple[str, ...] = ()
    activation_score: float | None = None
    lexical_score: float | None = None
    freshness_score: float | None = None

    @property
    def window_id(self) -> str:
        return self.window.window_id


@dataclass
class FakeAssignment:
    tier: str
    windows: tuple[FakeRouteWindow, ...]


@dataclass
class FakeMaterializationPlan:
    windows: tuple[FakeRouteWindow, ...]
    tier_order: tuple[str, ...] = ("HOT", "WARM", "COLD")
    tier_window_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    per_tier_counts: dict[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    payload_plan: dict[str, tuple[str, ...]] = field(default_factory=dict)
    kv_direct_ready: bool = False
    apollo_residual_ready: bool = False
    boundary_residual_paths: tuple[str, ...] = ()
    residual_stream_paths: tuple[str, ...] = ()
    recapture_requirements: tuple[str, ...] = ()
    insertion_layer: str | None = None
    decode_constraints: dict[str, Any] = field(default_factory=dict)
    verification_expectations: tuple[str, ...] = ()
    write_back_targets: tuple[str, ...] = ()
    fast_route_ready: bool = True
    apollo_residual_identity: dict[str, Any] = field(default_factory=dict)
    apollo_manifest_path: str | None = None
    apollo_torch_store_path: str | None = None


@dataclass
class FakeCompatibilityProof:
    adapter_compatible: bool = True
    index_ready: bool = True
    fast_route_ready: bool = True
    apollo_residual_ready: bool = True
    checked_window_ids: tuple[str, ...] = ()
    adapter_identity: dict[str, Any] = field(default_factory=lambda: {"model_family": "gemma-e2b"})
    index_identity: dict[str, Any] = field(default_factory=lambda: {"index_id": "idx-1"})
    apollo_residual_identity: dict[str, Any] = field(default_factory=lambda: {"manifest_id": "apollo-1"})
    jit_indexing_actions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass
class FakeEvidenceSupport:
    window_id: str
    supports_claim: str
    confidence: float
    evidence: tuple[str, ...] = ()
    mode: str = "dependency_source"
    route_trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeDecodePolicy:
    constraints: dict[str, Any] = field(default_factory=dict)
    insertion_layer: str | None = None
    boundary_layer: str | None = None
    injection_layer: str | None = None
    kv_layout: str | None = None


@dataclass
class FakeVerificationPlan:
    assertions: tuple[str, ...] = ()
    expectations: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()


@dataclass
class FakeWriteBackPolicy:
    targets: tuple[str, ...] = ()
    allow_user_memory_writeback: bool = False
    allow_code_task_writeback: bool = False
    notes: tuple[str, ...] = ()


@dataclass
class FakeHarnessPacket:
    compatibility_proof: FakeCompatibilityProof = field(default_factory=FakeCompatibilityProof)
    decode_policy: FakeDecodePolicy = field(default_factory=FakeDecodePolicy)
    verification_plan: FakeVerificationPlan = field(default_factory=FakeVerificationPlan)
    write_back_policy: FakeWriteBackPolicy = field(default_factory=FakeWriteBackPolicy)


@dataclass
class FakePlan:
    candidates: tuple[FakeCandidate, ...]
    tier_assignments: tuple[FakeAssignment, ...]
    materialization_plan: FakeMaterializationPlan
    router_metadata: dict[str, Any]
    selected_candidate: FakeCandidate
    evidence_supports: tuple[FakeEvidenceSupport, ...] = ()
    route_packet: FakeHarnessPacket = field(default_factory=FakeHarnessPacket)

    def tiers_present(self) -> tuple[str, ...]:
        return tuple(assignment.tier for assignment in self.tier_assignments if assignment.windows)


class FakeCentralRouter:
    seen_requests: list[FakeRouteRequest] = []
    seen_windows: list[list[FakeRouteWindow]] = []

    def __init__(self, *, hot_count: int = 2, warm_count: int = 3) -> None:
        self.hot_count = hot_count
        self.warm_count = warm_count

    def route(self, request: FakeRouteRequest, windows: list[FakeRouteWindow]) -> FakePlan:
        self.seen_requests.append(request)
        self.seen_windows.append(windows)
        selected = windows[0]
        candidate = FakeCandidate(
            window=selected,
            score=9.0,
            reasons=("central plan chose source dependency evidence",),
            activation_score=0.75,
            lexical_score=0.5,
            freshness_score=0.25,
        )
        return FakePlan(
            candidates=(candidate,),
            tier_assignments=(FakeAssignment("HOT", (selected,)),),
            materialization_plan=FakeMaterializationPlan(
                windows=(selected,),
                tier_window_ids={"HOT": (selected.window_id,)},
                per_tier_counts={"HOT": 1},
                kv_direct_ready=True,
                apollo_residual_ready=True,
                boundary_residual_paths=("boundary.bin",),
                decode_constraints=request.metadata.get("decode_constraints", {}),
                verification_expectations=tuple(request.metadata.get("verification_expectations", ())),
                write_back_targets=tuple(request.metadata.get("write_back_targets", ())),
                apollo_manifest_path="apollo.json",
            ),
            router_metadata={"capability_mode": request.capability_mode, "selected_window_id": selected.window_id},
            selected_candidate=candidate,
            evidence_supports=(
                FakeEvidenceSupport(
                    window_id=selected.window_id,
                    supports_claim="dependency evidence supports selected path",
                    confidence=0.9,
                    evidence=("import edge",),
                    route_trace={"path_hint_count": len(request.path_hints)},
                ),
            ),
            route_packet=FakeHarnessPacket(
                compatibility_proof=FakeCompatibilityProof(checked_window_ids=(selected.window_id,)),
                decode_policy=FakeDecodePolicy(
                    constraints=request.metadata.get("decode_constraints", {}),
                    insertion_layer="layer.10",
                    boundary_layer="layer.9",
                    injection_layer="layer.10",
                    kv_layout="fake-kv",
                ),
                verification_plan=FakeVerificationPlan(
                    assertions=("adapter compatibility",),
                    expectations=tuple(request.metadata.get("verification_expectations", ())),
                    required_artifacts=("selected windows exist",),
                ),
                write_back_policy=FakeWriteBackPolicy(
                    targets=tuple(request.metadata.get("write_back_targets", ())),
                    allow_code_task_writeback=True,
                ),
            ),
        )


class FakeModule:
    RouteRequest = FakeRouteRequest
    RouteWindow = FakeRouteWindow
    CentralRouter = FakeCentralRouter


def test_product_router_uses_injected_full_central_router_adapter() -> None:
    FakeCentralRouter.seen_requests.clear()
    FakeCentralRouter.seen_windows.clear()
    adapter = CentralRouterAdapter(module=FakeModule)

    packet = ProductRouter(router=adapter, proof_router_available=True).route(
        "Trace the source dependency",
        methodology="dependency_source",
        session_id="session-a",
        evidence=[{"text": "src/app.py imports src/db.py", "score": 8, "path": "src/app.py"}],
        adapter_metadata={"model_family": "gemma-e2b", "route_dimension": 16},
        index_readiness_metadata={"index_id": "idx-1", "status": "ready"},
        apollo_residual_readiness={"status": "ready", "manifest_id": "apollo-1"},
        path_hints=["src/app.py"],
        identifiers=["AppService"],
        ordinal=2,
        decode_constraints={"valid_paths": ["src/app.py"]},
        verification_expectations=["selected windows exist"],
        writeback_targets=["code_task_memory"],
    )

    assert packet.method == "source_dependency"
    assert packet.methodology == "dependency_source"
    assert packet.selected_windows == ["src/app.py imports src/db.py"]
    assert packet.route_reason == "central plan chose source dependency evidence"
    assert packet.tier == "hot"
    assert packet.activation_score == 0.75
    assert packet.lexical_score == 0.5
    assert packet.recency_score == 0.25
    assert packet.residual_available is True
    assert packet.kv_ready is True
    assert packet.provenance["router"] == "david.central_router.full"
    assert packet.provenance["proof_router_available"] is True
    assert packet.provenance["route_plan_metadata"]["selected_window_id"] == "src/app.py"
    assert packet.adapter_metadata["model_family"] == "gemma-e2b"
    assert packet.index_readiness_metadata["index_id"] == "idx-1"
    assert packet.apollo_residual_readiness["manifest_id"] == "apollo-1"
    assert packet.path_hints == ["src/app.py"]
    assert packet.identifiers == ["AppService"]
    assert packet.ordinal == 2
    assert packet.decode_constraints == {"valid_paths": ["src/app.py"]}
    assert "selected windows exist" in packet.verification_expectations
    assert packet.writeback_targets == ["code_task_memory"]
    assert packet.provenance["decode_policy"]["insertion_layer"] == "layer.10"
    assert packet.provenance["verification_plan"]["assertions"] == ["adapter compatibility"]
    assert packet.provenance["write_back_policy"]["targets"] == ["code_task_memory"]
    assert packet.provenance["compatibility_proof"]["adapter_identity"]["model_family"] == "gemma-e2b"
    assert packet.provenance["evidence_supports"][0]["supports_claim"] == "dependency evidence supports selected path"
    assert packet.provenance["tier_details"][0]["tier"] == "HOT"
    assert FakeCentralRouter.seen_requests[0].capability_mode == "dependency_source"
    assert FakeCentralRouter.seen_requests[0].path_hints == ("src/app.py",)
    assert FakeCentralRouter.seen_requests[0].identifiers == ("AppService",)
    assert FakeCentralRouter.seen_windows[0][0].scope_key == "session-a"


def test_stable_wrapper_load_does_not_import_protected_benchmark_scripts() -> None:
    protected_modules = {
        "scripts.run_swebench_pro_parity",
        "scripts.run_locobench_benchmark",
        "scripts.run_ruler_benchmark",
        "scripts.run_mrcr_benchmark",
        "scripts.interactive_memory_chat",
        "scripts.benchmark_jit_indexing",
    }
    for name in protected_modules:
        sys.modules.pop(name, None)

    module = load_david_router_wrapper()

    assert hasattr(module, "CentralRouter")
    assert protected_modules.isdisjoint(sys.modules)


def test_full_adapter_falls_back_through_product_router_on_router_error() -> None:
    class BrokenAdapter:
        def route(self, **_: Any) -> Any:
            raise RuntimeError("central router unavailable")

    packet = ProductRouter(router=BrokenAdapter()).route(
        "Use dependency routing",
        evidence=[{"text": "src/app.py imports src/db.py", "score": 3}],
        methodology="dependency_source",
    )

    assert packet.provenance["router"] == "david.product_router.fail_safe"
    assert packet.route_reason == "fail-safe product routing used after router error"
    assert packet.selected_windows == ["src/app.py imports src/db.py"]


def test_full_adapter_converts_jit_required_signal_to_product_metadata() -> None:
    @dataclass
    class FakeJITRequired(RuntimeError):
        compatibility_proof: FakeCompatibilityProof
        jit_indexing_actions: tuple[str, ...]
        metadata: dict[str, Any]

    class JITCentralRouter:
        def __init__(self, *, hot_count: int = 2, warm_count: int = 3) -> None:
            self.hot_count = hot_count
            self.warm_count = warm_count

        def route(self, request: FakeRouteRequest, windows: list[FakeRouteWindow]) -> Any:
            raise FakeJITRequired(
                compatibility_proof=FakeCompatibilityProof(
                    index_ready=False,
                    fast_route_ready=False,
                    jit_indexing_actions=("jit_index_request",),
                ),
                jit_indexing_actions=("jit_index_request",),
                metadata={
                    "jit_indexing_required": True,
                    "jit_indexing_stop": "pre_route",
                    "checked_window_ids": tuple(window.window_id for window in windows),
                },
            )

    class JITModule(FakeModule):
        CentralRouter = JITCentralRouter

    packet = ProductRouter(router=CentralRouterAdapter(module=JITModule)).route(
        "Trace the source dependency",
        methodology="dependency_source",
        evidence=[{"text": "src/app.py imports src/db.py", "score": 8, "path": "src/app.py"}],
        adapter_metadata={"model_family": "gemma-e2b"},
        index_readiness_metadata={"status": "missing"},
    )

    assert packet.route_reason == "full David central router requires JIT before routing"
    assert packet.jit_required is True
    assert packet.jit_actions == ["jit_index_request"]
    assert packet.provenance["jit_required"] is True
    assert packet.provenance["compatibility_proof"]["index_ready"] is False
