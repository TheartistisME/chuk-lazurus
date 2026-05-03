# David Production Roadmap

David should become a runnable terminal coding agent, not a benchmark runner
with terminal clothing. The current David directory proves routing,
configuration, validation, and memory ideas through local rigs. The production
work is to wrap those proven methods in a stable operator surface:

```text
User
  -> david terminal agent
  -> open model harness
  -> model config getter
  -> model config validator
  -> model adapter
  -> memory readiness check
  -> task/methodology detector
  -> central router
  -> residual/KV materializer
  -> decoding controller
  -> verifier
  -> memory writer
```

Benchmarks remain proof rigs. Runtime code should select capabilities and
methodologies such as `patch_target`, `dependency_source`,
`symbolic_chain`, `temporal_ordinal`, and `durable_chat_memory`, not MRCR,
RULER, LoCoBench, or SWE-bench scripts.

## Current State

The useful pieces already exist, but they are not wired into one agent.

| Surface | Current role | Production gap |
| --- | --- | --- |
| `David/get_model_config.py` | Standalone measured model scanner. | Needs to be callable from `david doctor` and boot flows without mutating runtime state. |
| `David/validate_model_config.py` | Standalone startup gate for config reports. | Needs to feed the harness boot policy and explain fail-closed decisions to operators. |
| `David/router.py` and `David/central router.py` | Capability router proof rig and compatibility wrapper. | Needs a product adapter boundary, not direct benchmark invocation. |
| `David/benchmark_row_validation.py` | Representative local rows for proof-rig behavior. | Must remain validation-only and never become product ontology. |
| `src/chuk_lazarus/harness` | Metadata-only boot spine and readiness contracts. | Needs a terminal-agent runtime wrapper and user-facing status reporting. |
| `src/chuk_lazarus/repl_agent_tools.py` | Bounded local coding tools and append-only traces. | Needs to be mounted inside the David turn loop with policy, verification, and memory write-back. |
| `src/chuk_lazarus/chat_loop/cli.py` | Gemma chat loop with streaming windowing. | Needs to be generalized into a tool-using coding agent and avoid implicit model downloads in tests and dry runs. |
| `pyproject.toml` scripts | `lazarus` and related commands; parallel work may already expose `david`. | The script must target a complete CLI/runtime/TUI stack, not only a parser shell. |

## Operator Contract

The first production cut should support these flows from a repo root:

```bash
david doctor --workspace . --model ./models/gemma-e2b --validation-report ./validation.json --offline
david --workspace . --model ./models/gemma-e2b --validation-report ./validation.json --once "inspect README.md" --offline
david --workspace . --model ./models/gemma-e2b --validation-report ./validation.json
```

`doctor` should explain model validation, adapter auto-load policy, memory/index
readiness, planned JIT actions, and tool availability. `--once` should execute
one non-interactive agent turn and return a process status suitable for CI.
Without `--once`, David should enter an injected-stream-friendly terminal UI.

The runtime must never download a model unless the operator explicitly allows
online loading. Tests and dry runs must use injected model/decoder fakes.

## Architecture Principles

- Keep benchmark adapters outside product runtime. They validate methods; they
  do not define memory artifacts or task types.
- Fail closed before model loading when model config validation is missing,
  rejected, low confidence, or incompatible with memory artifacts.
- Treat memory readiness as a startup gate with explicit JIT re-entry actions.
  A missing index should produce a plan, not silent degraded routing.
- Keep routing and decoding separate. The router selects evidence; the decoder
  constrains generation.
- Keep the local tool runner bounded and trace-first. Tool calls should be
  recorded as append-only JSONL before any derived memory index is trusted.
- Make every production surface dependency-injectable: streams, boot function,
  model loader, decoder, router, tool runner, verifier, and memory writer.
- Prefer plain serializable contracts at boundaries so terminal output, tests,
  and future daemon modes can share the same runtime state.

## Target Components

### CLI

Module: `chuk_lazarus.david.cli`

Responsibilities:

- expose `main(argv=None, stdin=None, stdout=None, stderr=None, runtime_factory=None)`;
- expose `build_parser()` for parser tests and shell completion later;
- validate workspace and report paths before constructing the runtime;
- map CLI flags into a serializable runtime config;
- support `doctor`, `--once`, interactive TUI mode, `--offline`, `--dry-run`,
  `--no-shell`, and explicit memory root overrides;
- return integer exit codes instead of calling `sys.exit()` in testable paths;
- keep all benchmark commands out of the product CLI.

Acceptance criteria:

- `pyproject.toml` exposes `david = "chuk_lazarus.david.cli:main"`;
- `david --help` describes the terminal agent, workspace, model validation,
  memory readiness, and one-shot turn mode;
- `david doctor` can run with a fake validation report and no model weights;
- `david --once` can run with injected runtime and streams.

### Runtime

Module: `chuk_lazarus.david.runtime`

Responsibilities:

- define `DavidRuntimeConfig`, `DavidRuntime`, and serializable turn/result
  records;
- call `boot_harness()` before model adapter loading;
- refuse unsafe auto-load decisions and surface blocking warnings;
- mount `LocalCodingToolRunner` inside the turn loop;
- detect task methodology from user text and workspace context;
- ask the router for evidence by capability, not benchmark name;
- materialize residual/KV context only after compatibility checks;
- run decoding with constraints derived from router evidence and decoder priors;
- verify outputs and tool effects before final answer;
- write user/task/decoder memory after verification.

Acceptance criteria:

- dry-run initialization never calls the model loader;
- missing workspace indexes produce `jit_required` and planned actions;
- repo patch requests select `patch_target`/repo-patch methodology;
- source/dependency questions select dependency-source routing;
- temporal recall selects temporal-ordinal routing;
- tool-call loops execute local tools, write trace JSONL, and feed
  `TOOL_RESULT` back into decoding;
- benchmark scripts are not imported during normal task detection.

### TUI

Module: `chuk_lazarus.david.tui`

Responsibilities:

- expose `run_tui(runtime, stdin=None, stdout=None, stderr=None)`;
- work with ordinary file-like streams, not only a real TTY;
- display boot status, validation status, memory/index readiness, JIT actions,
  and tool availability;
- support `/status`, `/tools`, `/doctor`, `/quit`, and normal prompts;
- call `runtime.run_once(prompt)` for each user prompt;
- stream assistant text when the runtime supplies deltas, but fall back to
  final-answer printing for tests;
- never display benchmark names as primary product modes.

Acceptance criteria:

- injected `StringIO` input can drive `/status`, `/tools`, prompts, and
  `/quit`;
- TUI output includes readiness and verification summaries;
- TUI exits cleanly with status 0 on `/quit` or EOF.

### Model Config Getter And Validator

The getter remains measurement-only. It should emit identity, tokenizer,
revision/hash, topology, projection aliases, layer candidates, insertion
family, confidence, review status, and provenance.

The validator remains the startup gate. It should validate report integrity,
model identity, topology, layer ordering, projection availability, behavioral
KV evidence when required, selected adapter config, confidence, and auto-load
policy.

Production wiring:

- `david doctor` can run getter/validator or read existing reports;
- `DavidRuntime.initialize()` consumes validator output through `boot_harness`;
- low-confidence or review-required reports are blocking unless an explicit
  unsafe override is added later and recorded loudly.

### Model Adapter Layer

David needs an adapter contract that can start with Gemma E2B and grow to other
families:

- tokenizer;
- hidden size;
- route layer and dimension;
- route query head;
- boundary layer;
- residual capture layer;
- KV source and target layer;
- projection producer and behavior cache layer;
- KV layout;
- residual capture method;
- logits and decoding hooks.

Adapters must be selected from validated reports, not from hard-coded benchmark
assumptions.

### Memory Readiness And JIT

The production runtime should check three stores separately:

- user memory: person-in-time memory;
- task/code memory: workspace-in-codebase memory;
- decoder prior store: generation-control memory.

Readiness states should be visible in `doctor`, runtime events, and TUI status.
If the workspace index is missing or stale, David should plan a boot/workspace
JIT action and stop or run the JIT path explicitly. If HOT residual streams are
missing, the materializer should recapture only the required windows.

### Router And Methodology Detection

Task detection should return product capabilities:

| User task | Methodology |
| --- | --- |
| "Fix this failing test" | repo patch targeting, proven by SWE-style rows |
| "Which source depends on this symbol?" | dependency routing, proven by LoCo-style rows |
| "Follow this variable chain" | symbolic chain routing, proven by RULER-style rows |
| "What did I say the third time?" | temporal ordinal recall, proven by MRCR/chat rows |
| "Remember my preference" | durable user memory |

The router should return selected windows, memory family, session/workspace
identity, tier, route reason, evidence, token cost, activation/lexical/ordinal
scores, recency, residual availability, KV readiness, and provenance.

### Materializer

The materializer chooses the cheapest compatible context strategy:

- boundary-only for semantic anchoring;
- full residual stream for token-level replay;
- KV-direct when the adapter and memory artifacts prove compatibility;
- recapture HOT windows when required streams are missing.

It must refuse unsafe mixing across tokenizer, model, adapter config, layer,
insertion family, or memory family.

### Decoding Controller

The decoder should consume router evidence and decoder priors without owning
retrieval. For code tasks it should bias valid file paths, symbols, imports,
dependencies, known tests, and patch-compatible edits. For structured tasks it
should constrain keys, IDs, answer formats, and exact requested values. For
user memory it should respect recency, stale or superseded state, confirmed
preferences, deadlines, and sensitive boundaries.

Decoder prior memory belongs to the decoding controller, scoped by model,
tokenizer, adapter config, layers, steering version, task type, and provenance.

### Verification And Write-Back

Verification should prove the result before memory write-back:

- code tasks: touched files, symbols, spans, tests, patch targets, and failures;
- temporal memory: exact occurrence, not merely similar text;
- multi-hop memory: evidence for every hop;
- user memory: time and staleness rules respected;
- KV/residual memory: injection traceable and compatible;
- adapter loading: getter plus validator prove safety;
- decoder priors: scope matches model/tokenizer/adapter/layer.

Only verified task context should be written to durable task memory. User
memory write-back should separate confirmed user facts from speculative
assistant guesses.

## Milestones

### P0: Product Skeleton

- Add `chuk_lazarus.david` package.
- Add `david` console script.
- Implement CLI parser, config object, injected streams, and no-download dry
  run.
- Implement `doctor` with boot status from existing harness.
- Keep all model loading behind dependency injection.

Exit criteria:

- `david --help`, `david doctor`, and `david --once "hello"` run with fakes;
- tests in `tests/david/test_cli.py` pass.

### P1: Boot-Gated Runtime

- Wrap `boot_harness()` in `DavidRuntime.initialize()`.
- Surface validation status, auto-load policy, memory roots, index readiness,
  and JIT actions.
- Refuse unsafe model loading.
- Add structured runtime events.

Exit criteria:

- dry runs prove no model loader call;
- missing and stale indexes produce explicit planned actions;
- tests in `tests/david/test_runtime.py` boot cases pass.

### P2: Tool-Using Turn Loop

- Mount `LocalCodingToolRunner`.
- Parse model tool calls with existing helpers.
- Execute bounded tools with workspace path guards.
- Feed `TOOL_RESULT` back to the decoder.
- Return final answers with tool traces and verification placeholders.

Exit criteria:

- one-shot mode can read, search, and patch files in `tmp_path`;
- trace JSONL records are written for each tool call.

### P3: Methodology Detection And Routing

- Implement product task detection.
- Translate detected tasks into router capability requests.
- Keep benchmark adapters out of runtime imports.
- Return route evidence and tier summaries in runtime events.

Exit criteria:

- repo patch, dependency, symbolic, temporal, and user-memory prompts select
  the expected product methodology names.

### P4: Materialization And Decoding Control

- Add adapter-backed materialization choices.
- Add decoder prior store lookup and update.
- Add path/symbol/test/JSON/ID constraints by task type.
- Refuse incompatible residual/KV artifacts.

Exit criteria:

- materialization plans are traceable and scoped;
- decoder priors are not reused across incompatible model/tokenizer/adapter
  scopes.

### P5: Verification And Memory Writer

- Add code verifier hooks for tests, linters, and file diffs.
- Add evidence verifier for temporal and multi-hop tasks.
- Add user memory staleness and sensitivity guards.
- Write task/user/decoder memory only after verification.

Exit criteria:

- runtime results include verification summaries;
- write-back records have provenance and can be replayed.

### P6: Terminal UX

- Implement stream-friendly TUI.
- Add `/status`, `/tools`, `/doctor`, `/quit`.
- Show boot readiness, route evidence summaries, tool activity, verification,
  and memory write-back status.
- Keep output quiet enough for repeated coding work.

Exit criteria:

- tests in `tests/david/test_tui.py` pass with injected streams;
- interactive sessions are usable in a normal terminal.

### P7: Model Family Expansion

- Promote Gemma E2B adapter as the first production adapter.
- Add validator-backed adapter candidates for additional open model families.
- Add adapter compatibility tests with synthetic reports before real model
  smoke runs.

Exit criteria:

- adapter loading is report-driven and family-specific logic stays isolated.

### P8: Release Hardening

- Add focused QA command group for David.
- Add offline CI tests.
- Add optional real-model smoke tests gated by environment variables.
- Document installation, model prep, validation, workspace indexing, and
  troubleshooting.

Exit criteria:

- `david doctor` explains every fail-closed state;
- no ordinary test downloads a model or mutates benchmark fixtures.

## Test Strategy

The first tests should describe behavior before implementation:

- `tests/david/test_cli.py`: console script, parser/main injection, one-shot
  command, doctor command, workspace validation.
- `tests/david/test_runtime.py`: boot gating, no dry-run model load, tool-loop
  trace behavior, product methodology detection.
- `tests/david/test_tui.py`: injected stream TUI, status/tools commands,
  prompt dispatch, graceful exit.

All tests should use `tmp_path`, `monkeypatch`, injected streams, fake
validation reports, fake runtimes or decoders, and no benchmark mutations.
Expected failures are acceptable until the product package exists; failures
should be clear enough to drive implementation.

## Risks

| Risk | Mitigation |
| --- | --- |
| Product runtime imports benchmark scripts directly. | Test methodology names and keep benchmark terms out of TUI/CLI primary output. |
| Model loads during tests or dry runs. | Require injected model loader and offline flags; make accidental calls raise in tests. |
| JIT readiness silently degrades to lexical-only routing. | Treat missing/stale indexes as planned actions or fail-closed states. |
| Tool runner writes outside workspace. | Keep `LocalCodingToolRunner` path guards and add runtime-level policy flags. |
| Decoder priors leak across models or adapters. | Scope priors by model, tokenizer, adapter config, layers, steering version, and task type. |
| User memory stores speculative assistant guesses. | Write only confirmed user facts or verified task results with provenance. |
| TUI becomes hard to test. | Keep a stream-based interface and layer optional rich terminal rendering later. |

## Definition Of Done

David is product-ready for the first release when:

- `david` is an installed console command;
- `david doctor` runs offline and explains boot readiness;
- `david --once` can complete a local coding turn with tools and traces;
- interactive TUI works with normal terminal input;
- validated model configs gate adapter loading;
- missing memory indexes produce explicit JIT plans;
- task detection selects product methodologies;
- router evidence, materialization, decoding constraints, verification, and
  write-back are all traceable;
- no default path downloads models or mutates benchmark artifacts during tests;
- documentation distinguishes proof rigs from production runtime behavior.
