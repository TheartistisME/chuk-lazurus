# Epic 1: AUS3000 Accuracy Program Workstreams

## 1. Purpose

This document decomposes **Epic 1** of the AUS3000 accuracy program into
non-conflicting workstreams that can be executed safely against the **current
repository baseline**.

Epic 1 scope here is **workstream decomposition only**. No implementation is
performed by this document. The goal is to create a file-scoped execution plan
for the later AUS3000 accuracy work so parallel agents can move quickly without
overlapping write scopes.

Primary source inputs used for this decomposition:

- [`complex_task_filled.md`](./complex_task_filled.md)
- `tools/build_aus3000_clause_aligned_variant.py`
- `tools/evaluate_aus3000_variant.py`
- `src/chuk_lazarus/inference/context/knowledge/route.py`
- `src/chuk_lazarus/inference/context/knowledge/torch_store.py`
- `src/chuk_lazarus/inference/context/knowledge/torch_query.py`
- `src/chuk_lazarus/cli/commands/context/generate/_torch.py`
- `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_validation_report.txt`
- `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant/torch_store/window_metadata.json`

## 2. Current Baseline

The workstreams below assume the following baseline is already true and must be
preserved until explicitly changed by a later stream:

- Clause-aligned AUS3000 torch store already exists and loads successfully.
- Current clause-aligned store shape is `1203` windows over `199258` tokens.
- `window_metadata.json` shows `27` split clauses (`60` split windows total).
- Query-time routing currently centers on TF-IDF token overlap with optional
  keyword fallback in:
  `src/chuk_lazarus/inference/context/knowledge/route.py` and
  `src/chuk_lazarus/inference/context/knowledge/torch_store.py`.
- Answer generation currently centers on:
  `src/chuk_lazarus/inference/context/knowledge/torch_query.py`.
- `context generate` has a second torch-specific store path in:
  `src/chuk_lazarus/cli/commands/context/generate/_torch.py`.
- Clause-aligned builder logic currently lives primarily in:
  `tools/build_aus3000_clause_aligned_variant.py`, with persisted store layout
  primitives in `src/chuk_lazarus/inference/context/knowledge/torch_build.py`.
- The latest 30-minute report executed `423` cases with:
  `310 PASS`, `38 REVIEW`, `75 FAIL`.
- Stable fail cluster from the current report:
  `accessible_definition`, `accessible_vs_readily`,
  `switchboard_definition`, `insulation_resistance_results`.
- Stable review cluster from the current report:
  `insulated_definition`, `rcd_definition`.

Implications for decomposition:

- Routing is still brittle for clause IDs, definition lookups, and
  morphologically similar titles.
- Builder/index work must be isolated from query-time routing work because both
  touch retrieval quality but modify different layers of the stack.
- Answering/grounding must be isolated from routing because the current
  evaluator conflates wrong-route and wrong-answer outcomes.
- Validation/reporting must stay independent so it can verify every wave
  without sharing write ownership with the implementation streams.

## 3. Parallelization Rules

- **Can run in parallel after benchmark contract is frozen:**
  benchmark repair, exact-routing improvements, and validation/reporting
  scaffolding.
- **Cannot run in parallel with exact-routing improvements:**
  builder/index changes if they alter retrieval artifact contracts consumed by
  `torch_store.py` or `torch_query.py`.
- **Cannot run in parallel with exact-routing improvements or builder/index
  changes:**
  answering/grounding changes that depend on stable routed windows and stable
  metadata.
- **Must run last in each integration wave:**
  validation/reporting full-run work, because it is the quality gate that
  judges merged behavior rather than partial local behavior.

Dependency summary:

1. Benchmark contract must be repaired first.
2. Exact routing starts once benchmark categories and pass/fail rules are fixed.
3. Builder/index changes start only if routing work proves current store
   artifacts are insufficient.
4. Answering/grounding starts after routing outputs are stable and, if needed,
   after builder/index artifact changes are merged.
5. Validation/reporting runs after every merge wave and owns the final proof.

## 4. Workstreams

### WS-1: Benchmark Repair And Benchmark Contract

This stream turns the current evaluator into the frozen Epic 1 benchmark
contract. It must separate route failure from answer failure and define what
counts as PASS, REVIEW, and FAIL before any retrieval or answering code is
changed.

| Field | Value |
|---|---|
| Owner scope | `tools/evaluate_aus3000_variant.py`, **new** `docs/aus3000_accuracy_program/03-benchmark-definition.md` |
| Forbidden | `tools/build_aus3000_clause_aligned_variant.py`, `src/chuk_lazarus/inference/context/knowledge/route.py`, `src/chuk_lazarus/inference/context/knowledge/torch_store.py`, `src/chuk_lazarus/inference/context/knowledge/torch_query.py`, `src/chuk_lazarus/cli/commands/context/generate/_torch.py`, every file under `src/chuk_lazarus/inference/context/knowledge/` not explicitly owned |
| Prerequisites | None. This is the first implementation stream. |
| Responsibilities | Freeze the AUS3000 gold set, label exact failure categories, split route-vs-answer-vs-refusal scoring, encode hard-fail prompts for out-of-domain bleed, and make the report stable enough to compare wave-to-wave. |
| Quality gates | Local dry-run over the full prompt suite without changing production checkpoint contents; evaluator output must include routed-window evidence and distinct route/answer verdict signals; existing baseline summary must still be reproducible. |
| Test responsibilities | Add or extend evaluator-focused tests for case parsing, scoring thresholds, route-vs-answer attribution, and out-of-domain bleed detection; add a fixture-backed regression that preserves current baseline counts when run against the current checkpoint unless scoring rules intentionally change and are documented. |
| Integration points | Consumed by WS-2, WS-3, WS-4, and WS-5 as the single benchmark contract. |
| Stop conditions | Stop once benchmark inputs, verdict rules, and report schema are frozen and reviewed. Do not change routing, store artifacts, or answer prompting in this stream. |

### WS-2: Exact-Routing Improvements

This stream owns query-time exact addressability. It must improve clause-ID
matching, title alias handling, and deterministic top-k routing without
changing how the store is built.

| Field | Value |
|---|---|
| Owner scope | `src/chuk_lazarus/inference/context/knowledge/route.py`, `src/chuk_lazarus/inference/context/knowledge/torch_store.py`, `tests/inference/context/test_knowledge_store.py`, `tests/inference/context/test_torch_store.py` |
| Forbidden | `tools/build_aus3000_clause_aligned_variant.py`, `src/chuk_lazarus/inference/context/knowledge/torch_build.py`, `src/chuk_lazarus/inference/context/knowledge/torch_query.py`, `src/chuk_lazarus/cli/commands/context/generate/_torch.py`, `tools/evaluate_aus3000_variant.py` after WS-1 freeze except for read-only schema consumption |
| Prerequisites | WS-1 merged and benchmark contract frozen. |
| Responsibilities | Add exact clause-id pre-routing if warranted, title alias normalization, comparison-query multi-clause routing, deterministic tie-breaking, and route diagnostics that explain why windows were selected. |
| Quality gates | Stable pass on routing-only cases from the benchmark contract; routed windows for the stable fail cluster must include the expected clause before this stream can declare success; no changes to on-disk store format. |
| Test responsibilities | Unit tests for clause ID extraction/normalization, alias/title normalization, comparison prompt decomposition, deterministic top-k ordering, and route-top-k behavior when expansion terms are present. |
| Integration points | Supplies stable window selection to WS-4; produces evidence used by WS-3 to decide whether builder/index work is actually necessary. |
| Stop conditions | Stop once benchmark route failures attributable to query-time selection are resolved or clearly proven impossible without store-format changes. If store-format changes are needed, document the blocker and hand it to WS-3 instead of expanding scope here. |

### WS-3: Builder/Index Changes If Routing Alone Is Insufficient

This stream is conditional. It exists only if WS-2 proves the current
clause-aligned store artifacts are insufficient for exact routing or grounded
answering.

| Field | Value |
|---|---|
| Owner scope | `tools/build_aus3000_clause_aligned_variant.py`, `src/chuk_lazarus/inference/context/knowledge/torch_build.py`, `tests/inference/context/test_torch_build.py` |
| Forbidden | `src/chuk_lazarus/inference/context/knowledge/route.py`, `src/chuk_lazarus/inference/context/knowledge/torch_store.py`, `src/chuk_lazarus/inference/context/knowledge/torch_query.py`, `src/chuk_lazarus/cli/commands/context/generate/_torch.py`, `tools/evaluate_aus3000_variant.py` |
| Prerequisites | WS-1 merged; WS-2 complete enough to show that query-time routing alone cannot meet the benchmark contract. |
| Responsibilities | Add any missing exact-address metadata, alias indexes, or clause-aware retrieval artifacts required by the routing contract while preserving the base model and existing generic knowledge-store workflows. |
| Quality gates | New build must be reproducible, sidecar/manifest contents must remain loadable by current torch-store readers or include an explicitly coordinated backward-compatible change, and rebuilt checkpoint must preserve existing clause-aligned wins such as clause `5.6.2.5`. |
| Test responsibilities | Extend builder tests for new artifact files, manifest contract, metadata alias persistence, split-clause handling, and backward-compatible load behavior. |
| Integration points | Produces store artifacts consumed read-only by WS-2 and WS-4; must hand off any new manifest/store keys to WS-5 for reporting. |
| Stop conditions | Stop once the store exposes the exact metadata/index features required by the frozen routing contract. Do not change query scoring logic or answer prompting here. |

### WS-4: Answering And Grounding Changes

This stream owns answer generation only after routing is stable. It must reduce
false refusals, constrain answers to retrieved facts, and preserve out-of-domain
insufficiency behavior.

| Field | Value |
|---|---|
| Owner scope | `src/chuk_lazarus/inference/context/knowledge/torch_query.py`, `src/chuk_lazarus/cli/commands/context/generate/_torch.py`, `tests/inference/context/test_torch_query_helpers.py`, `tests/cli/commands/context/generate/test_cmd_torch.py`, **new** `tests/inference/context/test_torch_query_grounding.py` |
| Forbidden | `src/chuk_lazarus/inference/context/knowledge/route.py`, `src/chuk_lazarus/inference/context/knowledge/torch_store.py`, `tools/build_aus3000_clause_aligned_variant.py`, `src/chuk_lazarus/inference/context/knowledge/torch_build.py`, `tools/evaluate_aus3000_variant.py` except for read-only contract consumption |
| Prerequisites | WS-1 merged; WS-2 merged; WS-3 merged if WS-3 was activated. |
| Responsibilities | Tighten prompt rendering and grounding policy, make refusal behavior conditional on retrieved evidence rather than brittle misses, keep `knowledge query` and torch `context generate` behavior aligned, and prevent out-of-domain electrical bleedthrough. |
| Quality gates | Stable pass on previously routed-but-wrong benchmark cases; no regression on out-of-domain insufficiency prompts; `knowledge query` and `context generate` must agree on routed-window usage and answer mode for the same checkpoint. |
| Test responsibilities | Add targeted tests for grounded-answer rendering, insufficient-context phrasing, context-window merging, route-to-answer handoff, and parity between `torch_query.py` and `context generate/_torch.py` store-answer paths. |
| Integration points | Consumes routed windows from WS-2 and any new artifact metadata from WS-3; hands final answer behavior to WS-5 for full benchmark validation. |
| Stop conditions | Stop once answer failures are isolated to benchmark or routing defects outside this stream. Do not change route selection logic or store format in this stream. |

### WS-5: Validation, Reporting, And Reproduction Pack

This stream owns the proof layer. It must produce the reproducible evidence that
Epic 1 is ready to hand off into implementation waves or declare unfinished.

| Field | Value |
|---|---|
| Owner scope | **new** `docs/aus3000_accuracy_program/04-validation-matrix.md`, **new** `docs/aus3000_accuracy_program/05-reproduction.md` |
| Forbidden | `src/chuk_lazarus/inference/context/knowledge/route.py`, `src/chuk_lazarus/inference/context/knowledge/torch_store.py`, `tools/build_aus3000_clause_aligned_variant.py`, `src/chuk_lazarus/inference/context/knowledge/torch_build.py`, `src/chuk_lazarus/inference/context/knowledge/torch_query.py`, `src/chuk_lazarus/cli/commands/context/generate/_torch.py` |
| Prerequisites | WS-1 merged. Full final validation waits for WS-2, WS-3 if activated, and WS-4. |
| Responsibilities | Maintain smoke, dry-run, long-run, and regression report commands; publish final benchmark evidence; document exact reproduction commands; consume the frozen evaluator contract from WS-1 read-only. |
| Quality gates | Full benchmark run reproducible from documented commands; report pack includes baseline delta, stable failure/review trend, and exact checkpoint/store paths used; no hidden manual steps. |
| Test responsibilities | Add light tests for report serialization and reproduction-command formatting when those checks live outside the frozen evaluator; own any documentation spot checks tied to the benchmark/report schema. |
| Integration points | Reads outputs from every other stream; acts as the final evidence producer for Epic 1 readiness. |
| Stop conditions | Stop once all validation docs and command lines are up to date and the latest benchmark evidence is attached. Do not fix routing or answering defects inside this stream. |

## 5. What Can Run In Parallel

### Parallel Set A

These streams can run together after WS-1 is merged:

- WS-2 exact-routing improvements
- WS-5 validation/reporting scaffolding

Why this is safe:

- WS-2 owns query-time routing files only.
- WS-5 owns reporting docs only and consumes the WS-1 evaluator read-only.
- WS-5 must treat the WS-1 benchmark contract as read-only and may not alter
  scoring semantics once WS-2 is in flight.

### Parallel Set B

These streams **cannot** run together:

- WS-2 and WS-3
- WS-2 and WS-4
- WS-3 and WS-4

Why they conflict:

- WS-2 and WS-3 both change the retrieval contract, but at different layers.
  Running both at once would make routing regressions impossible to attribute.
- WS-4 depends on stable routed windows and, if activated, stable store
  metadata from WS-3.

### Always Serialized

- WS-1 must be first.
- WS-4 must wait for the final routing/store contract.
- WS-5 full benchmark certification must be last in every wave.

## 6. Recommended Wave Order

### Wave 0: Benchmark Contract

- WS-1 benchmark repair and benchmark-definition freeze

Exit gate:

- Benchmark categories, scoring, and report schema are fixed and reviewed.

### Wave 1: Retrieval Foundation

- WS-2 exact-routing improvements
- WS-5 validation/reporting scaffolding in parallel

Exit gate:

- Stable fail cluster is re-triaged with route-vs-answer attribution.
- Decision recorded: either routing fixes are sufficient, or WS-3 must open.

### Wave 2: Conditional Store Contract

- WS-3 builder/index changes, only if Wave 1 proves they are required

Exit gate:

- Rebuilt store artifacts are reproducible and loadable.
- Routing cases improve without regressing known clause-aligned wins.

### Wave 3: Answering And Grounding

- WS-4 answering/grounding changes

Exit gate:

- Routed-but-wrong/refusal cases are resolved or clearly reclassified as
  remaining routing issues.

### Wave 4: Certification

- WS-5 full validation/reporting pass

Exit gate:

- Smoke, dry-run, and sustained benchmark evidence are published.
- Remaining failures, if any, are filed as follow-up issues rather than hidden.

## 7. Reviewer Assignments

Each stream requires one **primary reviewer** and one **independent reviewer**
who did not author that stream. Because no team roster is committed in this
repository, the assignments are role-based and should be mapped to named people
or agents at execution time.

| Workstream | Primary reviewer | Independent reviewer |
|---|---|---|
| WS-1 Benchmark repair | Benchmark contract reviewer | Routing reviewer |
| WS-2 Exact routing | Retrieval/routing reviewer | Builder/index reviewer |
| WS-3 Builder/index | Knowledge-store builder reviewer | Benchmark contract reviewer |
| WS-4 Answering/grounding | Grounding/prompting reviewer | Retrieval/routing reviewer |
| WS-5 Validation/reporting | Validation/report reviewer | Epic lead reviewer |

Reviewer rules:

- Reviewers must verify owned-file boundaries were respected.
- Reviewers must verify forbidden files were not edited.
- Reviewers must verify tests named in the stream were actually run.
- Independent reviewers must confirm the stream stopped at its declared stop
  condition and did not absorb neighboring scope.

## 8. Epic 1 Completion Rule

Epic 1 is complete only when:

- WS-1 through WS-5 have either completed or been explicitly skipped according
  to the conditional rule on WS-3.
- The latest validation pack shows no hidden manual steps.
- Remaining gaps, if any, are attributed to a specific unresolved stream and
  filed as follow-up work rather than bundled into vague "next steps".
