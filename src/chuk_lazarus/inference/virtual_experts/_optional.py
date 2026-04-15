"""
Optional ``chuk_virtual_expert`` compatibility layer.

The virtual experts package is optional in Lazarus. When it is not installed,
we provide small in-tree stand-ins that preserve the subset of the API used by
the inference virtual-expert package and its focused tests.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar

try:
    from chuk_virtual_expert import (  # type: ignore[import-not-found]
        VirtualExpert,
        VirtualExpertAction,
        VirtualExpertResult,
    )

    HAS_CHUK_VIRTUAL_EXPERT = True
except ImportError:
    HAS_CHUK_VIRTUAL_EXPERT = False

    @dataclass(slots=True)
    class VirtualExpertAction:
        """Fallback action payload used when the optional package is absent."""

        expert: str
        operation: str
        parameters: dict[str, Any] = field(default_factory=dict)
        confidence: float = 1.0
        reasoning: str = ""

        @classmethod
        def none_action(cls, reasoning: str = "") -> "VirtualExpertAction":
            return cls(
                expert="none",
                operation="passthrough",
                parameters={},
                confidence=1.0,
                reasoning=reasoning,
            )

        def model_dump(self) -> dict[str, Any]:
            return {
                "expert": self.expert,
                "operation": self.operation,
                "parameters": self.parameters,
                "confidence": self.confidence,
                "reasoning": self.reasoning,
            }

        def model_dump_json(self) -> str:
            return json.dumps(self.model_dump())

    @dataclass(slots=True)
    class VirtualExpertResult:
        """Fallback execution result returned by the stub VirtualExpert."""

        success: bool
        data: Any = None
        error: str | None = None

    class VirtualExpert:
        """Fallback base class for optional virtual-expert plugins."""

        name: ClassVar[str] = "virtual_expert"
        description: ClassVar[str] = ""
        version: ClassVar[str] = "0.0.0"
        priority: ClassVar[int] = 0

        def can_handle(self, prompt: str) -> bool:
            return False

        def get_operations(self) -> list[str]:
            raise NotImplementedError("VirtualExpert subclasses must define get_operations()")

        def get_calibration_prompts(self) -> tuple[list[str], list[str]]:
            return [], []

        async def execute(self, action: VirtualExpertAction) -> VirtualExpertResult:
            if action.expert not in {self.name, "none"}:
                return VirtualExpertResult(
                    success=False,
                    error=f"Action expert {action.expert!r} does not match {self.name!r}",
                )

            if action.operation not in self.get_operations():
                return VirtualExpertResult(
                    success=False,
                    error=f"Unknown operation: {action.operation}",
                )

            try:
                result = self.execute_operation(action.operation, action.parameters)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:
                return VirtualExpertResult(success=False, error=str(exc))

            return VirtualExpertResult(success=True, data=result)

        def __repr__(self) -> str:
            return (
                f"{self.__class__.__name__}(name={self.name!r}, "
                f"priority={self.priority})"
            )

