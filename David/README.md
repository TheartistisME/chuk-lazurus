# David Centralized Router Harness

## Overview

This directory contains a standalone centralized router harness for validating
how long-context and chat memories should be selected before materialization.
It is intentionally small and stdlib-only so benchmark and smoke workers can
exercise the routing contract without depending on the product runtime.

The harness is not wired into product runtime yet. Product integration still
needs caller-specific adapters, runtime tests, and ownership decisions before
these primitives should be treated as live routing behavior.

## Architecture

- `central router.py` defines the neutral router primitives and deterministic
  routing policy.
- `smoke_test_central_router.py` checks all supported routing modes with local
  fixtures and asserts full tier coverage.
- `benchmark_row_validation.py` runs one representative local row for MRCR,
  RULER, LoCo, SWE, and Chat-style durable memory validation.

Core data flow:

1. A `RouteRequest` names the capability mode, query, scope, path hints,
   identifiers, entities, and metadata.
2. Candidate `RouteWindow` objects carry text, source path, temporal scope,
   memory authority, stale/superseded state, and metadata.
3. `CentralRouter.route(request, windows)` dispatches to the mode-specific
   scorer, ranks `RouteCandidate` objects, assigns tiers, and builds a
   `MaterializationPlan`.
4. `RoutePlan` returns candidates, tier assignments, evidence supports,
   materialization metadata, and the selected candidate.
5. `RoutePlan.assert_tier_coverage()` enforces the tier invariant before a plan
   is returned.

## Key Concepts

- Capability mode: the routing behavior requested by a benchmark row, product
  workflow, or smoke fixture.
- Candidate: a scored route window with reasons, evidence, and trace metadata.
- Evidence support: a proof-like record explaining why a window participates in
  the route.
- Materialization plan: the concrete HOT, WARM, and COLD windows a downstream
  caller would materialize.
- Router metadata: inspectable route details such as selected window id,
  eligible window count, tiers present, filtered stale memory ids, and
  mode-specific traces.

## Routing Capabilities

- `temporal_ordinal`: resolves scoped duplicate requests by ordinal position,
  used by MRCR-style "third occurrence" rows.
- `symbolic_chain`: follows assignment-style chains and records evidence for
  each step, used by RULER-style variable/value rows.
- `dependency_source`: favors source and dependency windows from path hints,
  identifiers, activation metadata, and recursive route traces.
- `patch_target`: extends dependency routing for SWE-style patch planning,
  prioritizing implementation source over tests, docs, assets, and padding
  while preserving selected test metadata.
- `durable_chat_memory`: routes current durable memory, separates user memory
  from task/tool memory, and filters stale or superseded memories before active
  tiering.
- `general_recall`: provides fallback lexical, literal, and entity-style recall.

## HOT/WARM/COLD Tier Invariant

Every routing mode must return HOT, WARM, and COLD tier windows whenever at
least three eligible windows exist. This is a hard harness invariant, not a
best-effort preference:

- `tier_assignments` must contain non-empty HOT, WARM, and COLD assignments.
- `materialization_plan.tier_window_ids` must also contain HOT, WARM, and COLD.
- `RoutePlan.assert_tier_coverage()` raises if either surface omits a required
  tier.

Rows with fewer than three eligible windows may legitimately omit lower tiers,
but all current smoke and benchmark-row fixtures are designed to exercise the
full HOT/WARM/COLD guarantee.

## Running Validations From WSL

From Windows PowerShell, run the smoke test through the default WSL login
environment and the repo root:

```bash
wsl --cd /mnt/c/Users/jehma/Desktop/lazarus/chuk-lazurus -- bash -lc 'python3 David/smoke_test_central_router.py'
```

Run the benchmark-row validation:

```bash
wsl --cd /mnt/c/Users/jehma/Desktop/lazarus/chuk-lazurus -- bash -lc 'python3 David/benchmark_row_validation.py'
```

Run one benchmark row by name, benchmark, or capability:

```bash
wsl --cd /mnt/c/Users/jehma/Desktop/lazarus/chuk-lazurus -- bash -lc 'python3 David/benchmark_row_validation.py --row MRCR'
wsl --cd /mnt/c/Users/jehma/Desktop/lazarus/chuk-lazurus -- bash -lc 'python3 David/benchmark_row_validation.py --row temporal_ordinal'
```

From inside WSL after changing into the repo root, the equivalent commands are:

```bash
python3 David/smoke_test_central_router.py
python3 David/benchmark_row_validation.py
python3 David/benchmark_row_validation.py --row Chat
```

Expected results are JSON summaries with passing status, selected window ids,
tier counts or tier lists, route trace metadata, and explicit confirmation that
the HOT/WARM/COLD coverage checks passed.
