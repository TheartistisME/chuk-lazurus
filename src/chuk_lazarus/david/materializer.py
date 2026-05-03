"""Safe residual/KV materialization metadata for offline David."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AdapterSessionMetadata
from .routing import RoutePacket


@dataclass(frozen=True)
class MaterializedContext:
    strategy: str
    text_context: str
    compatibility: dict[str, Any]
    refused: bool = False
    reason: str = "ok"


class Materializer:
    def materialize(self, route: RoutePacket, adapter: AdapterSessionMetadata) -> MaterializedContext:
        compatibility = adapter.scope() | {
            "route_memory_family": route.memory_family,
            "route_tier": route.tier,
            "residual_available": route.residual_available,
            "kv_ready": route.kv_ready,
        }
        if route.kv_ready and adapter.kv_source_layer is None:
            return MaterializedContext(
                strategy="refuse",
                text_context="",
                compatibility=compatibility,
                refused=True,
                reason="kv route requested without adapter kv_source_layer",
            )
        strategy = "kv_direct" if route.kv_ready else "boundary_text"
        return MaterializedContext(
            strategy=strategy,
            text_context="\n".join(route.selected_windows),
            compatibility=compatibility,
        )

