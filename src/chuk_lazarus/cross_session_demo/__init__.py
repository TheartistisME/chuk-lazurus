"""Axis-5 cross-session demo: terminal acceptance gate for ``turn-aligned``.

Public surface:

- :func:`run_demo`
- :class:`DemoResult`
- :class:`SessionPlan`
- :class:`VerificationReport`
- :data:`SIX_STRICT_KEYS`

The package assembles axes 1-4 into a single end-to-end verification
pipeline. It does not modify any upstream module; heavy imports (torch,
transformers) are deferred to the CLI entry point.
"""

from __future__ import annotations

from chuk_lazarus.cross_session_demo.cli import DemoResult, run_demo
from chuk_lazarus.cross_session_demo.report import VerificationReport, build_report
from chuk_lazarus.cross_session_demo.session_generator import (
    DEFAULT_PLANS,
    SessionPlan,
    generate_session,
    plant_location,
)
from chuk_lazarus.cross_session_demo.verification import (
    SIX_STRICT_KEYS,
    QueryExecution,
    all_assertions_pass,
    run_cross_session_queries,
)

__all__ = [
    "DEFAULT_PLANS",
    "DemoResult",
    "QueryExecution",
    "SIX_STRICT_KEYS",
    "SessionPlan",
    "VerificationReport",
    "all_assertions_pass",
    "build_report",
    "generate_session",
    "plant_location",
    "run_cross_session_queries",
    "run_demo",
]
