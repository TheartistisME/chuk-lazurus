# David Router Architecture Index

## Core Files

| File | Role |
| --- | --- |
| `David/central router.py` | Standalone centralized router primitives, mode dispatch, scoring, tier assignment, materialization plan, and HOT/WARM/COLD invariant enforcement. |
| `David/smoke_test_central_router.py` | Local smoke tests for every supported routing mode. |
| `David/benchmark_row_validation.py` | One-row validation for MRCR, RULER, LoCo, SWE, and Chat fixtures. |
| `David/README.md` | Operator-facing overview, architecture, invariants, and WSL validation commands. |
| `David/CHANGELOG.md` | Dated implementation notes for the harness. |

## Benchmark To Capability Map

| Benchmark | Product Routing Capability | Router Mode | What It Validates | Core Files |
| --- | --- | --- | --- | --- |
| MRCR | Temporal duplicate request recall | `temporal_ordinal` | Selects the requested occurrence within a scoped repeated request sequence. | `central router.py`, `smoke_test_central_router.py`, `benchmark_row_validation.py` |
| RULER | Symbolic proof/assignment recall | `symbolic_chain` | Resolves variable-style chains and keeps chain evidence inspectable. | `central router.py`, `smoke_test_central_router.py`, `benchmark_row_validation.py` |
| LoCo | Source and dependency retrieval | `dependency_source` | Selects implementation/dependency source windows from path hints, identifiers, activation metadata, and route traces. | `central router.py`, `smoke_test_central_router.py`, `benchmark_row_validation.py` |
| SWE | Patch target planning | `patch_target` | Prioritizes implementation files ahead of tests, docs, assets, and padding while retaining selected test metadata. | `central router.py`, `smoke_test_central_router.py`, `benchmark_row_validation.py` |
| Chat | Durable chat memory routing | `durable_chat_memory` | Routes current user/task/tool memory and filters stale or superseded memories before active tiering. | `central router.py`, `smoke_test_central_router.py`, `benchmark_row_validation.py` |

## Tier Contract

All benchmark mappings rely on the same tier contract: if at least three
eligible route windows exist, the router must return HOT, WARM, and COLD in both
`tier_assignments` and `materialization_plan.tier_window_ids`.

This index describes the standalone David harness only. It does not imply that
the centralized router is wired into the product runtime.
