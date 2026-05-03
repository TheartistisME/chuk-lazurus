# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Windows / WSL Tool Access

If a project tool is not visible from PowerShell, invoke it through the default WSL login shell from the repo root:

```bash
wsl --cd /mnt/c/Users/jehma/Desktop/lazarus/chuk-lazurus -- bash -lc '<command>'
```

This is the expected path for tools installed in WSL, such as `bd` under `~/.local/bin`. Do not force a specific distro unless you have verified its PATH. For TinyTool/tinydex, use `/mnt/c/Users/jehma/Desktop/TinyTool/bin/tinydex`.

## Filedex / Tinydex Cadence

All agents must use tinydex/filedex as a tiny file-memory ritual:

1. Before reading or editing any file you plan to touch, run `tinydex scan <files...>` from the repo root. If using the WSL full path, run `/mnt/c/Users/jehma/Desktop/TinyTool/bin/tinydex scan <files...>`.
2. Before work, fetch relevant cards: `dependencies`, `tests`, and `risks`.
3. While working, set useful discoveries with `tinydex set <file> <category> "..." --agent <name>`.
4. At handoff, update `status`, `tests`, `next_steps`, and optionally `agent_notes`.
5. This applies to subagents too; coordinators must tell delegated agents to use this cadence.
6. Keep entries short and agent-readable.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

Absolutely. This is the cleaner version of the vision now.

The system should not be “a benchmark runner with clever tricks.” It should be a **terminal AI agent harness**: something like Codex or Claude Code, but powered by an open model and a memory/control runtime that gives the model pseudo-infinite working context.

Benchmarks are not the product. Benchmarks are proof rigs. They validate the methodologies the harness uses during real work.

```text
User
  -> Terminal Agent
  -> Open Model Harness
  -> Model Config Getter
  -> Model Config Validator
  -> Model Adapter
  -> Memory Readiness Check
  -> Task/Methodology Detection
  -> Route Memory + Evidence
  -> Materialize Residual/KV Context
  -> Decode With Constraints
  -> Verify
  -> Write Back User + Task Memory
```

**Core Vision**

The end goal is a terminal agent that can take an open-source model, load it, understand its internal geometry, and operate it through a harness that gives it benchmark-proven capabilities.

Gemma E2B is the first target model, not the final architecture.

```text
Open Model Runtime
  -> scan model config
  -> validate model config
  -> load model adapter
  -> check memory/index readiness
  -> detect task type
  -> select methodology
  -> route relevant memory
  -> materialize residual/KV context
  -> decode with constraints
  -> verify result
  -> write new memory back
```

**Benchmarks As Proof Rigs**

MRCR, RULER, LoCoBench, and SWE-bench are not product modules. They are baseline evaluations for different problem-solving methodologies.

```text
MRCR      -> proves temporal ordinal recall
RULER     -> proves symbolic multi-hop task memory
LoCoBench -> proves source/dependency routing
SWE-bench -> proves repo patch-target routing
Chat      -> proves durable user/task memory
```

The production harness should not think:

```text
Run SWE-bench logic.
```

It should think:

```text
This is a repo patch task.
Use the patch-targeting methodology proven by SWE-bench.
```

It should not think:

```text
Run LoCoBench logic.
```

It should think:

```text
This task requires source/dependency routing.
Use the dependency methodology proven by LoCoBench.
```

**Terminal Agent**

The user operates the model through the harness.

The terminal agent should be able to:

- enter a codebase;
- load an open model;
- scan and validate that model’s layer/KV/residual geometry;
- check whether the workspace has a valid memory index for that model;
- JIT-index the workspace if needed;
- understand the user’s task;
- select the right methodology;
- retrieve the right memories/files/symbols;
- materialize context through residual/KV mechanisms;
- constrain decoding so the model produces valid work;
- verify the output;
- remember the task, result, and user context.

**Model Config Getter**

The harness needs a standalone model scanner.

It should inspect the model and emit a measured report containing:

- model identity;
- tokenizer identity;
- model revision/hash;
- adapter family;
- hidden size;
- attention head count;
- KV head count;
- projection aliases;
- layer topology;
- route layer candidate;
- route query head candidate;
- boundary layer candidate;
- KV source layer candidate;
- KV target layer candidate;
- insertion family;
- confidence/review status;
- provenance.

This report is measurement, not mutation.

**Model Config Validator**

The validator is the startup gate.

Before the harness trusts a model config, it should validate:

- report integrity;
- model identity;
- topology;
- source/target layer ordering;
- K/V projection availability;
- behavioral KV evidence when required;
- selected adapter config;
- confidence;
- auto-load policy.

The harness should only auto-load an adapter config when the validator proves it is safe enough.

**Model Adapter Layer**

Each model family needs an adapter contract.

The adapter should expose:

- tokenizer;
- hidden size;
- route layer;
- route dimension;
- route query head;
- boundary layer;
- residual capture layer;
- KV source layer;
- KV target/injection layer;
- projection producer layer;
- behavior cache layer;
- KV layout;
- residual capture method;
- decoding/logits hooks.

Gemma E2B is the first implementation. Later the same harness should support other open-source model families without rewriting routing, memory, decoding, or verification.

**Memory Artifacts**

The harness should not store benchmark-specific artifacts as the product ontology.

It should store real operating memory:

```text
Chat/User Memory Artifact
  person-in-time memory

Code/Task Memory Artifact
  workspace-in-codebase memory

Decoder Prior Store
  durable generation-control memory
```

**Chat/User Memory Artifact**

This stores memories about the user as a real person operating in time:

- previous chats;
- preferences;
- recurring goals;
- deadlines;
- decisions;
- corrections;
- long-running context;
- relationship continuity;
- stale or superseded preferences;
- real-world time relevance.

**Code/Task Memory Artifact**

This stores memories about work being performed:

- source files;
- repo structure;
- symbols;
- dependency spans;
- patch targets;
- failing tests;
- prior attempts;
- retrieved evidence;
- multi-hop task state;
- verification results.

**Task/Methodology Detection**

The harness should automatically detect what kind of problem the user is asking.

Examples:

```text
Repo patch task
  -> SWE-proven patch targeting + constrained edit decoding

Source/dependency reasoning
  -> LoCo-proven dependency routing

Symbolic multi-hop task
  -> RULER-proven chain routing

Temporal recall task
  -> MRCR/chat-proven ordinal memory

Interactive user continuity
  -> durable chat/user memory
```

This is the key shift: the harness selects **methodology**, not benchmark script.

**Shared Indexer**

The harness should have one shared indexing contract, but multiple capture cadences.

- **Boot/workspace JIT path**
  - Used when entering a codebase.
  - Runs before work if no valid index exists for the active model adapter.

- **Fast JIT path**
  - Used for large static workloads.
  - Ideal for codebases, benchmark corpora, and static task contexts.

- **Live incremental path**
  - Used for chat and ongoing sessions.
  - Preserves session, turn, timestamp, role, and real-world time.

Default capture should include activation routes, boundary residuals, metadata, and provenance. Full token-level residual streams should be captured on demand for HOT windows and KV-direct materialization.

**Central Router**

The router should be capability-based, not benchmark-based.

Capabilities:

- topical search;
- exact ID recall;
- entity mention recall;
- temporal ordinal recall;
- symbolic chain routing;
- source/dependency routing;
- repo patch-target routing;
- multi-hop task routing;
- KV-direct memory recall.

The router should return:

- selected windows;
- memory family;
- session/workspace identity;
- tier: cold, warm, hot;
- route reason;
- evidence;
- token cost;
- activation score;
- lexical score;
- ordinal score;
- recency score;
- residual availability;
- KV readiness;
- provenance.

**KV/Residual Materializer**

The materializer chooses the cheapest valid context injection strategy.

- Boundary-only when the model needs semantic anchoring.
- Full residual stream when token-level KV replay is needed.
- Re-capture HOT windows if streams are missing.
- Use the model adapter to determine boundary/KV layers.
- Refuse unsafe mixing across tokenizer, model, layer, insertion family, or memory family.

**Decoding Controller**

The decoder should remain separate from the router.

The router decides what memory matters.
The decoder decides how generation should be constrained.

For code tasks, decoding should bias toward:

- valid file paths;
- existing symbols;
- known imports;
- dependency names;
- patch-compatible edits;
- known tests;
- known failure traces.

For structured tasks, it should constrain:

- JSON keys;
- answer format;
- IDs;
- variable names;
- ordinal targets;
- exact requested values.

For user memory, it should respect:

- recency;
- uncertainty;
- stale memories;
- confirmed preferences;
- deadlines;
- sensitive/private memory boundaries.

**Decoder Prior Store**

The harness should remember decoding priors across runs without training the model.

For code generation, this includes:

- target language/dialect;
- forbidden foreign-language token families;
- accepted case count;
- successful streak;
- temptation events;
- steering applications;
- bounded seed alpha;
- cross-case decay;
- model/tokenizer/layer scope;
- adapter config scope;
- steering version;
- task type;
- provenance.

This belongs to the decoding controller, not the router.

**Verification Layer**

Every capability needs proof.

- Code/task memory: right files, symbols, spans, tests, patch targets.
- Temporal memory: correct occurrence, not similar text.
- Multi-hop memory: complete chain with evidence for every hop.
- Chat/user memory: time/staleness respected.
- KV/residual memory: injected memory is traceable and compatible.
- Model adapter loading: getter + validator prove config safety.
- Decoder memory: prior scope matches adapter/model/tokenizer/layer.

**The Architecture Call**

Centralize the harness, not the benchmark scripts.

```text
Terminal Agent
  uses

Open Model Harness
  model config getter
  model config validator
  model adapter
  memory readiness check
  task/methodology detector
  central router
  KV/residual materializer
  decoding controller
  verifier
  memory writer
```

Benchmarks prove the methods.
The terminal agent uses the methods.
The harness is the product.

That gives us the actual end goal: a terminal AI agent, powered first by Gemma E2B but architected for open-source model families, with validated model loading, startup memory readiness, benchmark-proven routing methodologies, KV/residual context management, constrained decoding, durable user memory, durable task memory, and pseudo-infinite memory for real work.
