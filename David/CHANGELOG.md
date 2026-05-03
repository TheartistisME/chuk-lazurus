# Changelog

## 2026-05-03 - Model Config Validator Phase 2 Gate

- Added documentation for `David/validate_model_config.py` as the standalone
  load gate for `get_model_config.py` reports.
- Documented the end-to-end validation phases: report integrity, topology,
  model identity, tensor projection, Phase 2 prefix-cache behavioral layer
  ablation, and final harness load policy.
- Fixed and documented the Gemma4 logical-to-physical cache-layer behavior:
  candidate `kv_source_layer=28` and `kv_target_layer=29` validates against
  `behavior_cache_layer=23` because Gemma4 shares physical cache layers.
- Confirmed the Gemma E4B Phase 2 run selected
  `adapter_config_candidate+recommended+kv_candidate_0` and returned
  `validation_status="accepted"`, `confidence="high"`, and
  `auto_load_allowed=true`.
- Kept the validation scope explicit: Phase 2 proves prefix-cache behavioral
  impact with layer ablation, but it is not yet a full harness KV-direct
  injection replay test.

## 2026-05-02 - Production Model Config Getter Contract

- Updated the David docs for the production-grade `get_model_config.py` report:
  it now emits a versioned contract with `schema_name`, `schema_version`,
  provenance, structured warnings, recommendation review state, and
  `adapter_config_candidate`.
- Documented the intended harness consolidation path: callers supply a model,
  the harness can run config getter and JIT readiness checks, then load the
  adapter candidate automatically once confidence is ready.
- Clarified confidence semantics: legacy/KV disagreement is `review_required`,
  strong legacy matches with safe margin can be `ready` even when head scan is
  unavailable, and `--fail-on-low-confidence` can make low-confidence reports
  fail the process.
- Noted Gemma/Qwen/Llama-ish adapter-family and projection alias support, plus
  structured diagnostics when hooks or projection introspection are unavailable.
- Added structured failure-report behavior for model/dependency load failures
  so `--json-out` still receives a harness-blocking report instead of only a
  traceback.
- Added WSL examples for inspect-only use, measured narrow calibration, and
  writing JSON output into Windows temp.

## 2026-05-02 - Apollo Residual Harness Readiness

- Added Apollo sequential residual sidecars to the benchmark JIT path while
  preserving the fast route query files, `activation_routes.npy` and
  `window_tokens.npz`.
- The Apollo pass writes per-window `boundaries/window_*.npy`,
  `residual_streams/window_*.npy`, final `boundary_residual.npy`, and a
  manifest that records document order, layer/source identity, activation route
  row alignment, artifact paths, and readiness status.
- Updated LoCo Phase 3 to prefer ready Apollo residual stores for selected
  windows instead of independently recapturing them; the old Layer-13 recapture
  path remains a fallback unless strict Apollo store mode is requested.
- SWE-bench Pro continues to use the shared LoCo Phase-3 path and now carries
  the row-to-JIT-case mapping needed to find the corresponding Apollo store.
- Added `ApolloResidualReadinessMetadata` to the David router contract so fast
  route readiness and Apollo residual/KV materialization readiness are proven
  separately.
- Extended the central router bridge with `readiness_source="apollo_manifest"`
  and manifest-derived adapter, index, and Apollo residual metadata. Synthetic
  benchmark readiness remains the default for representative benchmark/chat
  fixtures.
- Added tests for Apollo sidecar writing, Phase-3 store loading/fallback, router
  Apollo readiness stops, and bridge manifest metadata derivation.

## 2026-05-02 - Harness Router Contract Promotion

- Promoted the docs from a central-router-only description to the harness-router
  contract: `David/router.py` is the importable wrapper, while
  `David/central router.py` remains a temporary compatibility file for older
  fixtures that still reference the spaced filename.
- Documented the proof boundary between central routing and harness routing:
  adapter validity, index compatibility/readiness, synthetic benchmark metadata
  injection, decode policy, verification plan, and write-back policy are exposed
  through `CompatibilityProof`, expanded `MaterializationPlan`, and
  `RoutePacket`.
- Documented adapter/index readiness metadata, including model/tokenizer hashes,
  route dimensions, KV layout, index status, and JIT-indexing actions, plus the
  strict unknown-mode behavior that only falls back when `allow_mode_fallback`
  is explicitly set.
- Added the `JITIndexingRequired` stop signal: `allow_jit_indexing` now records
  the required request/window JIT actions and raises before scoring or route
  materialization, so callers must build indexes and re-enter routing instead of
  using a route with `index_ready=False`.
- Hardened active-adapter index readiness so a ready index must carry matching
  model, tokenizer, adapter, layer, route-dimension, and KV-layout identity.
  Sparse `status=ready` metadata and Gemma/Qwen cross-model indexes are
  rejected.
- Added first-class memory artifact coverage for `ChatUserMemoryArtifact` and
  `CodeTaskMemoryArtifact`, plus richer user temporal and code/task metadata.
- Clarified that MRCR, RULER, LoCo, SWE, and Chat entries are benchmark
  adapters/fixtures issuing capability requests, not product ontology.
- Kept the validation caveat explicit: local representative rows and smoke tests
  verify the contract shape, but full real benchmark suites were not run.

## 2026-05-02 - Standalone Model Config Calibrator

- Added `David/get_model_config.py`, a standalone Hugging Face causal-LM
  calibrator that emits model identity, tokenizer hash metadata, layer topology,
  legacy vec-inject layer/head findings, KV source/target candidates, projection
  variant checks, and a recommended adapter-style config.
- Kept legacy `injection_layer` output separate from KV-direct
  `kv_source_layer` and `kv_target_layer` so old copy-head peaks can become
  candidate KV targets without being misread as source layers.
- Smoke-tested the script against the local Gemma-4 E2B snapshot and
  `HuggingFaceTB/SmolLM2-360M-Instruct` with narrow layer/head limits.
- Documented WSL invocation examples for topology-only inspection and narrow
  Gemma/SmolLM smoke runs.

## 2026-05-02 - Initial Central Router Harness

- Added the standalone `David/central router.py` centralized router harness with
  neutral request/window dataclasses, route candidates, evidence supports, tier
  assignments, materialization plans, and route metadata.
- Implemented routing modes for temporal ordinal lookup, symbolic chains,
  dependency source selection, SWE-style patch targets, durable chat memory, and
  general recall.
- Added smoke coverage in `David/smoke_test_central_router.py` for
  `temporal_ordinal`, `symbolic_chain`, `dependency_source`, `patch_target`,
  `durable_chat_memory`, and `general_recall`.
- Added `David/benchmark_row_validation.py` with one representative local row
  each for MRCR, RULER, LoCo, SWE, and Chat validation.
- Documented the harness in `David/README.md` and mapped benchmark capabilities
  in `David/index.md`.
- Guaranteed that all routing modes return HOT, WARM, and COLD tier windows
  whenever at least three eligible windows exist, across both tier assignments
  and materialization windows.
- Kept the harness standalone; it is not wired into product runtime yet.
