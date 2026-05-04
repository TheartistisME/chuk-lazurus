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
    kv_direct_ready: bool = False
    apollo_residual_ready: bool = False
    boundary_residual_paths: tuple[str, ...] = ()
    residual_stream_paths: tuple[str, ...] = ()


@dataclass
class FakeCompatibilityProof:
    adapter_compatible: bool = True
    index_ready: bool = True


@dataclass
class FakeHarnessPacket:
    compatibility_proof: FakeCompatibilityProof = field(default_factory=FakeCompatibilityProof)


@dataclass
class FakePlan:
    candidates: tuple[FakeCandidate, ...]
    tier_assignments: tuple[FakeAssignment, ...]
    materialization_plan: FakeMaterializationPlan
    router_metadata: dict[str, Any]
    selected_candidate: FakeCandidate
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
                kv_direct_ready=True,
                apollo_residual_ready=True,
                boundary_residual_paths=("boundary.bin",),
            ),
            router_metadata={"capability_mode": request.capability_mode, "selected_window_id": selected.window_id},
            selected_candidate=candidate,
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
    assert FakeCentralRouter.seen_requests[0].capability_mode == "dependency_source"
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
