# [Complex Task Template]

<Metadata>
Version: 1.0
Date: 2026-04-16
Author: Codex Coordinator
Workspace: /mnt/c/users/jehma/desktop/lazarus/chuk-lazurus
Vee Workspace: /mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/.vee
Primary Dataset: /mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018
Base Model: /home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf
Target Variant: /mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant
Current Clause-Aligned Store: /mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant/torch_store
Target Hardware: CUDA GPU under WSL
Python Runtime: uv-managed project environment
Core Dependencies: vee, uv, torch backend, transformers, Lazarus CLI, pytest, ripgrep
Known Good Inputs:
- Clause-aligned AUS3000 store exists and loads successfully.
- Clause 5.6.2.5 retrieval regression was fixed by clause-aligned chunking.
- 30-minute clause-aligned validation baseline exists at:
  /mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_validation_report.txt
Baseline Metrics:
- 423 executed
- 310 PASS
- 38 REVIEW
- 75 FAIL
Stable Failure Cluster:
- accessible_definition
- accessible_vs_readily
- switchboard_definition
- insulation_resistance_results
Stable Review Cluster:
- insulated_definition
- rcd_definition
Critical Learnings Already Established:
- Flat fixed-token windows caused boundary-straddle retrieval failures.
- Clause-aligned windows solved the 5.6.2.5 showers/bathrooms miss.
- Bigger windows are not the fix; exact retrieval units are.
- TF-IDF token overlap alone is brittle for clause IDs, casing, and title morphology.
- Apollo docs indicate high factual accuracy came from content-aware sampling, chat-template activation of the retrieval circuit, exact-address style routing, and inject-only-the-matched-fact discipline.
</Metadata>

# Task:
Achieve 100% accuracy for the AUS3000 Lazarus knowledge variant on a rigorous AUS3000 evaluation suite without breaking existing Lazarus behavior, existing user work, the base Gemma model, or the user's normal non-AUS3000 model path.

Success criteria:
- AUS3000 exact-clause lookup benchmark reaches 100% PASS on the agreed evaluation suite.
- No REVIEW and no FAIL remain on the production benchmark suite.
- Out-of-domain prompts do not bleed unnecessary electrical information.
- Clause-specific prompts retrieve the correct clause deterministically and reproducibly.
- The base Gemma model remains untouched.
- The existing normal model workflow and existing non-AUS3000 stores continue to work.
- The resulting AUS3000 build flow is documented, reproducible, and test-covered.

Constraints:
- Do not modify the base model weights.
- Do not overwrite or destroy existing stores or checkpoints.
- Do not regress current Lazarus `knowledge build/query` or `context prefill/generate` flows.
- Do not revert unrelated local user changes.
- Do not use beads/bd for workflow. Use vee for mission and agent lifecycle.
- Do not claim success based on one-off cherry-picked prompts; success must be benchmark-backed.
- Do not rely on Claude-only slash commands or Sonnet-only tooling that is unavailable in this Codex environment; use Codex-equivalent independent review waves instead.

# Project Deliverables:
- A full spec and workstream pack for the AUS3000 100% accuracy program.
- A production-grade AUS3000 retrieval and evaluation implementation that reaches 100% PASS on the agreed benchmark.
- A validated report pack including smoke, dry-run, real benchmark, regression checks, and exact reproduction commands.

## Workflow:
- Run `vee --help` to understand how we will use vee in our agentic lifecycle.
- Understand task, the why, the how, and the where.
- Begin document preparation phase.
- Continue through each phase and each epic until we have completed our task in all success criteria. I expect to have Lazarus AUS3000 Knowledge Variant working completely and holistically at every level when I run the application.

# Non-negotiables:
- Entire process completed end to end.
- All code dependencies installed so I can simply run the app and it works.
- Lazarus AUS3000 Knowledge Variant ready to use out of the box after changes.
- Industry standard code practices.
- All code inserted MUST have its equivalent test/s.
- All tests passing smoke/dry-run/real.

## Epics:
- <Epic-1> Create the following documents:

### Documents:
- Spec/specs with exact code snippets, line numbers, files touched, and code dependencies.
- Workstream/s documents that segment specs into non-conflicting workstreams that our agents will work in parallel, allowing spec/specs to be implemented safely and completely.
- Benchmark definition document with the exact AUS3000 gold set, scoring rules, red-line prompts, and regression gates.

### Document Preparation Workflow:

*Pre-Phase*
- Run `vee --help` to understand how we will use vee in our agentic lifecycle.
- Setup vee project for complete task completion.
- Review the current artifacts and reports before writing any new spec:
  - docs/aus3000_accuracy_program/complex_task_filled.md
  - tools/build_aus3000_clause_aligned_variant.py
  - tools/evaluate_aus3000_variant.py
  - /mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_validation_report.txt
  - /mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant/torch_store/window_metadata.json

*Phase 1* `vee mission create`, `vee mission update`, `vee agent spawn codex`, `vee agent message`, `vee agent check-in`
- Spawn Codex lead using vee spawn.
- Codex lead cannot jump straight into broad edits; it must first synthesize exact file paths, dependencies, likely regression surfaces, and required tests.
- Codex lead must delegate document writing and research to Codex team mates through coordinator-approved parallel workstreams. The coordinator remains the live `vee agent` operator.
- Once spec/specs and workstream documents are made, Codex lead will centralize them into one folder and begin phase 2.

*Phase 2* `vee agent spawn codex`, `vee agent message`, `vee agent check-in`
- Codex lead must request one team mate per document or per non-conflicting document cluster.
- Each team mate performs an explicit independent review wave for its assigned document:
  - verify assumptions against local code
  - verify exact file paths
  - verify test targets
  - verify no workstream overlaps
  - verify success criteria are measurable
- Because Claude-specific `/review-loop` and Sonnet agents are unavailable here, the mandatory equivalent is:
  - one Codex reviewer per file or file cluster
  - one independent review pass per reviewer
  - one final lead synthesis pass
- Once review waves have greenlit all necessary documents, Codex lead reports back; coordinator runs the final greenlight check as a quality gate and we begin Epic 2.

### Checklist:

Document Scope
[ ] Spec names the exact accuracy target and benchmark contract.
[ ] Spec defines 100% PASS in measurable terms.
[ ] Spec distinguishes retrieval failure from generation failure.
[ ] Spec distinguishes clause coverage, route quality, and answer grounding.
[ ] Spec lists all touched files with rationale.

Known Problems To Address
[ ] Clause ID routing brittleness.
[ ] Clause title casing and morphology brittleness.
[ ] Incomplete exact-address routing for clause lookups.
[ ] Remaining stable failures from the latest 30-minute report.
[ ] Remaining REVIEW cluster from the latest 30-minute report.
[ ] Out-of-domain bleedthrough regression prevention.

Architecture Decisions
[ ] Decide whether to add exact clause-id pre-routing.
[ ] Decide whether to add clause-title alias routing.
[ ] Decide whether to add clause-aware metadata augmentation.
[ ] Decide whether to add a `kvectors_full`-style exact factual path for AUS3000.
[ ] Decide how clause-aligned windows and any exact factual index interact.
[ ] Decide benchmark categories and hard fail conditions.

Testing Plan
[ ] Unit tests for routing helpers.
[ ] Unit tests for clause ID extraction and normalization.
[ ] Unit tests for title alias normalization.
[ ] Integration tests for exact clause retrieval.
[ ] Integration tests for out-of-domain non-bleed behavior.
[ ] Long-run harness revalidation.

- <Epic-2>

*E2 Phase1* `vee mission update`, `vee agent spawn codex`, `vee agent message`, `vee agent check-in`
Spawn a Codex lead using vee spawn and hand off documents and coding instructions to Codex lead. Ensure Codex lead follows mandatory orchestration protocols.

### Orchestration Protocols:
- Codex lead does not write the bulk of implementation first; it delegates specs with exact implementation changes needed to Codex code agents using coordinator-managed vee panes.
- Codex lead must delegate correctly across non-conflicting workstreams to ensure completion.
- Quality gates must be run by every agent after every change to ensure code quality.
- Codex lead must complete all stages start to finish and will not report success until we have achieved the success criteria of this task.

### Lead Rules:
- Lead must keep a single source of truth for benchmark status and remaining gaps.
- Lead must not accept "looks better" as success; only benchmark-green counts.
- Lead must request additional workers when parallel workstreams are safe and justified.

### Sub Agent Rules:
- Each sub agent owns only its assigned workstream and files.
- Each sub agent must run the narrowest useful tests before handing back.
- Each sub agent must report exact changes, exact files, and exact evidence.

## Batch Protocols:
- Parallel work only when write scopes do not overlap.
- Benchmark and regression runs must use stable, versioned prompt suites.
- Every wave ends with integration, rerun, and gap triage before the next wave begins.

## Workflow:
- Run `vee --help` to understand how we will use vee in our agentic lifecycle.
- Understand task, the why, the how, and the where.
- Begin document preparation phase.
- Continue through each phase and each epic until we have completed our task in all success criteria. I expect to have Lazarus AUS3000 Knowledge Variant working completely and holistically at every level when I run the application.

# Non-negotiables:
- Entire process completed end to end.
- All code dependencies installed so I can simply run the app and it works.
- Lazarus AUS3000 Knowledge Variant ready to use out of the box after changes.
- Industry standard code practices.
- All code inserted MUST have its equivalent test/s.
- All tests passing smoke/dry-run/real.
