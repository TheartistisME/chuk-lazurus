# David Router Architecture Index

## Core Files

| File | Role |
| --- | --- |
| `David/router.py` | Importable wrapper for the harness router. Prefer this path for new callers. |
| `David/central router.py` | Temporary compatibility file with the historical spaced filename; defines router primitives, mode dispatch, scoring, tier assignment, compatibility proof, route packet, materialization plan, and HOT/WARM/COLD invariant enforcement. |
| `David/smoke_test_central_router.py` | Local smoke tests for every supported routing mode. |
| `David/benchmark_row_validation.py` | One representative local row for MRCR, RULER, LoCo, SWE, and Chat adapters/fixtures. These rows call capability requests and are not product ontology. |
| `David/get_model_config.py` | Standalone open-weight model config getter that emits a versioned report contract with identity, topology, legacy vec-inject findings, KV candidates, projection checks, provenance, warnings, review status, and `adapter_config_candidate`. |
| `David/validate_model_config.py` | Standalone model config validator that consumes config getter reports, runs topology/projection/behavioral KV gates, and emits `selected_config`, confidence, and harness auto-load policy. |
| `David/README.md` | Operator-facing overview, architecture, invariants, and WSL validation commands. |
| `David/CHANGELOG.md` | Dated implementation notes for the harness. |

## Apollo Residual Integration Points

| File | Role |
| --- | --- |
| `scripts/benchmark_jit_indexing.py` | Keeps the fast JIT route matrix and writes Apollo sequential residual sidecars: per-window boundaries, residual streams, final boundary residual, and `manifest.json`. |
| `scripts/run_locobench_benchmark.py` | Phase 3 loads ready Apollo residual stores for selected HOT windows before falling back to recapture. |
| `scripts/run_swebench_pro_parity.py` | Reuses the LoCo Phase-3 loader and supplies the row-to-JIT-case map for SWE rows. |
| `scripts/central_router_bridge.py` | Supplies synthetic benchmark readiness by default and real manifest-derived readiness when `readiness_source="apollo_manifest"`. |

## Benchmark Adapter Boundary

MRCR, RULER, LoCo, SWE, and Chat are benchmark adapters/fixtures that translate
external row shapes into neutral capability requests. They validate router
contracts at the adapter boundary; they do not define product memory ontology.

## Benchmark To Capability Map

| Benchmark | Product Routing Capability | Router Mode | What It Validates | Core Files |
| --- | --- | --- | --- | --- |
| MRCR | Temporal duplicate request recall | `temporal_ordinal` | Selects the requested occurrence within a scoped repeated request sequence. | `router.py`, `central router.py`, `smoke_test_central_router.py`, `benchmark_row_validation.py` |
| RULER | Symbolic proof/assignment recall | `symbolic_chain` | Resolves variable-style chains and keeps chain evidence inspectable. | `router.py`, `central router.py`, `smoke_test_central_router.py`, `benchmark_row_validation.py` |
| LoCo | Source and dependency retrieval | `dependency_source` | Selects implementation/dependency source windows from path hints, identifiers, activation metadata, and route traces. | `router.py`, `central router.py`, `smoke_test_central_router.py`, `benchmark_row_validation.py` |
| SWE | Patch target planning | `patch_target` | Prioritizes implementation files ahead of tests, docs, assets, and padding while retaining selected test metadata. | `router.py`, `central router.py`, `smoke_test_central_router.py`, `benchmark_row_validation.py` |
| Chat | Durable chat memory routing | `durable_chat_memory` | Routes current user/task/tool memory and filters stale or superseded memories before active tiering. | `router.py`, `central router.py`, `smoke_test_central_router.py`, `benchmark_row_validation.py` |

## Harness Router Contract

| Surface | Fields Or Types | Contract |
| --- | --- | --- |
| Import boundary | `David/router.py`, `David/central router.py` | New callers import `router.py`; the spaced central file remains for compatibility while fixtures migrate. |
| Mode behavior | `capability_mode`, `allow_mode_fallback` | Unknown modes raise by default. Fallback to `general_recall` is opt-in and recorded in route metadata. |
| Adapter readiness | `ModelAdapterMetadata` | Tracks model family/revision/hash, tokenizer identity, adapter version, route/boundary/injection layers, route dimension, and KV layout. |
| Index readiness | `IndexReadinessMetadata` | Tracks index id, adapter key/version, status, build time, model/tokenizer identity, route/boundary/injection layers, route dimension, and KV layout. With active adapter metadata, sparse ready-like status is not enough; the index must carry matching identity evidence. |
| Apollo readiness | `ApolloResidualReadinessMetadata` | Tracks manifest/store paths, window counts, boundary/residual stream counts, source/target layers, source window refs, and KV-direct readiness. It is separate from fast route index readiness. |
| JIT readiness stop | `JITIndexingRequired` | When `allow_jit_indexing` is set and required indexes are missing or incompatible, the router records JIT actions and raises before scoring/materialization. The production harness must run JIT externally and re-enter routing. |
| Compatibility proof | `CompatibilityProof` | Records adapter compatibility, fast route readiness, Apollo residual readiness, checked window ids, adapter/index/Apollo identity, JIT-indexing actions, and notes. On a JIT stop, the proof is carried by `JITIndexingRequired` instead of a returned route. |
| Memory artifacts | `ChatUserMemoryArtifact`, `CodeTaskMemoryArtifact` | Chat/user and code/task memories are first-class artifacts with stale/superseded state, artifact paths, temporal/workspace metadata, conflicts, tests, failures, dependencies, and patch-target roles. |
| Rich metadata | `UserTemporalMetadata`, `CodeTaskMetadata` | User time fields and code workspace fields travel on both requests and windows without becoming benchmark ontology. |
| Materialization | `MaterializationPlan` | Carries tier windows, payload plan, token budget, boundary/residual/KV paths, Apollo manifest/store identity, recapture requirements, insertion layer, decode constraints, verification expectations, and write-back targets. |
| Packet proof | `RoutePacket` | Returns selected memory, evidence, tiering, `CompatibilityProof`, `MaterializationPlan`, `DecodePolicy`, `VerificationPlan`, and `WriteBackPolicy`. |

## Readiness Sources

| Source | Behavior |
| --- | --- |
| `synthetic_benchmark` | Default bridge mode for MRCR/RULER/LoCo/SWE/Chat representative rows. It injects synthetic adapter/index metadata and does not imply product Apollo readiness. |
| `apollo_manifest` | Real bridge mode for Apollo/KV paths. It derives adapter, fast index, and Apollo residual metadata from a manifest and refuses to silently synthesize benchmark identity. |

Fast route readiness means the router can score/select compatible windows.
Apollo residual readiness means selected windows have compatible residual
artifacts that can be loaded for KV materialization. The two proofs are kept
separate so a route can be useful for scoring without pretending it is ready for
decode-time materialization.

## Tier Contract

All benchmark mappings rely on the same tier contract: if at least three
eligible route windows exist, the router must return HOT, WARM, and COLD in both
`tier_assignments` and `materialization_plan.tier_window_ids`.

This index describes the standalone David harness only. It does not imply that
the centralized router is wired into the product runtime.

Local smoke tests and representative benchmark rows check this contract shape,
including `RoutePacket`, `CompatibilityProof`, strict fallback, and benchmark
adapter boundaries. They are not full real MRCR, RULER, LoCo, SWE-bench, or chat
memory benchmark suites.

## Model Adapter Calibration Map

| Surface | Output Fields | Why It Exists |
| --- | --- | --- |
| Report contract | `schema_name`, `schema_version`, `provenance`, `warnings`, `diagnostics` | Gives future harness consolidation a stable JSON shape instead of scraping human text. |
| Failure report | `status`, `failure`, unavailable calibration sections, `adapter_config_candidate.status=review_required` | Lets harness startup stop cleanly when model/dependency loading fails, while still preserving machine-readable provenance. |
| Model identity | `model`, `wrapper_model_type`, `text_model_type`, `model_revision_or_hash`, `tokenizer_hash`, `hidden_size` | Readiness manifests need exact model/tokenizer compatibility, not just a friendly model name. |
| Layer topology | `full_attention_layers`, `sliding_attention_layers`, per-layer projection and KV-sharing flags | KV-direct injection must choose valid source/target geometry per model family. |
| Legacy vec-inject | `legacy_retrieval_layer`, `legacy_query_head`, `legacy_injection_layer`, `routing_score` | Preserves the old answer-copy calibration signal for comparison and migration. |
| KV candidates | `kv_source_layer`, `kv_target_layer`, `insertion_family`, `lineage`, `materialization_safe` | Separates boundary/source residual capture from the target attention layer that receives projected KV. |
| Projection variants | `raw_residual`, `input_layernorm_then_kv_proj`, norm ratios, adapter-family aliases | Measures whether raw or pre-normalized residual projection is better for Gemma/Qwen/Llama-ish adapters. |
| Recommendation review | `recommendation.review_status`, confidence, review reasons | Marks legacy/KV disagreement as review-required, allows strong safe-margin legacy matches to be ready even without head scan, and supports fail-fast low-confidence runs. |
| Adapter candidate | `adapter_config_candidate` | Gives the harness a measured adapter proposal it can load automatically after confidence and JIT readiness are proven, without mutating the product registry. |

## Model Config Validation Map

| Surface | Output Fields | Why It Exists |
| --- | --- | --- |
| Validator report | `schema_name`, `schema_version`, `provenance`, `source_report_summary`, `warnings` | Gives harness startup a separate proof artifact before loading a measured adapter config. |
| Report integrity gate | `report_integrity.status`, `candidate_count`, warnings | Rejects malformed or incomplete config reports before touching model weights. |
| Topology gate | `topology_gate.results`, source/target layer status | Ensures candidate source and target layers are in range and ordered correctly for the reported topology. |
| Model identity gate | `model_identity_gate`, model types, revision/hash, hidden size, heads, dtype, device | Confirms the loaded model matches the report identity closely enough for adapter loading decisions. |
| Projection gate | `projection_gate.candidates`, `ranked_candidates`, K/V tensor stats | Verifies candidate residuals can produce finite, non-zero K/V tensors through the resolved projection modules. |
| Phase 2 behavior gate | `behavior_gate.mode=prefix_cache_layer_ablation`, `full_cache_delta`, `ablation_drop`, `behavior_cache_layer` | Proves the selected cache layer actually affects recall, including shared physical cache-layer mappings such as Gemma4 logical target 29 -> cache layer 23. |
| Load decision | `validation_status`, `confidence`, `selected_config`, `harness_load_policy`, `auto_load_allowed` | Converts validation evidence into the harness policy: auto-load only when accepted, high confidence, and behavioral KV validation pass. |
