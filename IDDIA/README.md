# IDDIA

IDDIA is a standalone tool that turns a legally available DDIA PDF into
agent-stage context packages:

1. The PDF is stored as immutable source under `IDDIA/artifacts/ddia/source/`.
2. Each PDF page is converted to one Markdown file with Microsoft MarkItDown.
3. Page Markdown is chunked into JSONL records with stable ids and provenance.
4. zvec stores a rebuildable vector index over those chunk records.
5. `python -m IDDIA package` creates bounded context packages for an agent
   task, lifecycle stage, and next steps.

`IDDIA/artifacts/` is ignored by git. The repo carries the tool, schemas, and
command chain; book-derived source text and vector stores stay local and
reproducible.

## Agentic Operating Guide

Use IDDIA as an agent context and self-improvement loop. Your job is not just to
retrieve DDIA-shaped advice; your job is to make bounded, provable progress and
leave a durable handoff for the next agent.

### First Five Minutes

Run these before changing behavior:

```bash
python -m IDDIA stages
python -m IDDIA.meta helper-context
python -m pytest IDDIA/tests -q
```

If your task is about retrieval quality, also establish the frozen eval baseline:

```bash
python IDDIA/evals/run_brownfield_greenfield.py
```

Treat the eval output as the benchmark. Do not claim the tool improved unless
the same benchmark or same meta-grade command improves after your change.

### Build A Context Package For Your Task

Every context package needs three inputs:

- `task`: the concrete work the agent is doing.
- `stage`: one of `onboard`, `plan`, `build`, `verify`, `handoff`, or `exit`.
- `next_steps`: the known next command, constraint, or decision.

Example:

```bash
python -m IDDIA package \
  --stage plan \
  --task "Improve schema migration retrieval for brownfield projects" \
  --next-steps "Raise the frozen schema-migration eval above 4.0/5 without hurting other scenarios" \
  --format json \
  --output IDDIA/artifacts/ddia/packages/schema-migration-plan.json
```

Read the package as a bounded context, not as an oracle. Prefer hits that have:

- matching `concept_tags` for the task;
- a relevant `chapter_title` / `section_title`;
- a useful `why_this_hit.summary`;
- low or empty `noise_flags`;
- provenance fields such as page and chunk id.

Do not paste long DDIA source text into commits, reports, or prompts. Package
outputs and eval reports intentionally omit or cap source snippets.

### Agent Lifecycle

Work through this loop:

```text
onboard -> plan -> build -> verify -> handoff -> exit -> onboard
```

Use the stage to choose your behavior:

- `onboard`: read helper context, manifests, recent signoffs, and current evals.
- `plan`: state the claim, metric, intended files, and rollback path.
- `build`: make the smallest scoped change that can move the metric.
- `verify`: run focused tests, full IDDIA tests, lint, eval, and meta-grade.
- `handoff`: append changelog/signoff with proof fields.
- `exit`: supervise spawned agents, close panes, commit scoped changes, report blockers.

### Proof Contract

Before changing behavior, write down the claim you intend to prove:

```text
Claim: <what should improve>
Metric: <fixed command/rubric that decides the claim>
Baseline: <score/result before the change>
Pass condition: <score/result that counts as better>
```

After the change, rerun the same metric. Tests prove implementation contracts;
they do not prove retrieval quality by themselves. For retrieval work, use:

```bash
python IDDIA/evals/run_brownfield_greenfield.py
python -m IDDIA.meta grade <fresh-package.json> \
  --expected-concept <concept> \
  --preferred-chapter "<chapter>"
```

A valid proof signoff must include:

- files modified;
- agent objective;
- TLDR of the change;
- mandatory dependencies/context;
- proof claim;
- proof metric/rubric;
- proof evidence;
- proof verdict: `proven`, `partially_proven`, or `not_proven`.

Append it with:

```bash
python -m IDDIA.meta signoff append \
  --file IDDIA/core.py \
  --objective "Improve schema migration retrieval" \
  --tldr "Added concept diversification for migration/replay/manifest queries" \
  --dependency "python -m pytest IDDIA/tests -q" \
  --dependency "python IDDIA/evals/run_brownfield_greenfield.py" \
  --proof-claim "Schema migration retrieval rises above 4.0/5" \
  --proof-metric "brownfield-medium-schema-migration grade in the frozen eval" \
  --proof-evidence "before: 3.1/5; after: 4.4/5" \
  --proof-verdict proven
```

### Let IDDIA Spawn An Optimizer Agent

Grade a weak package:

```bash
python -m IDDIA.meta grade <package.json> \
  --expected-concept schema \
  --expected-concept replay \
  --preferred-chapter "Encoding and Evolution"
```

Spawn an optimizer from that grade:

```bash
python -m IDDIA.meta spawn IDDIA/artifacts/meta/grades/<grade>.json \
  --objective "Improve the weakest retrieval criteria and prove the result with the frozen eval" \
  --vee-agent claude \
  --tmux-session iddia-meta \
  --tmux-window agents \
  --worker-name iddia-improve-schema
```

Then supervise it:

```bash
python -m IDDIA.meta supervise IDDIA/artifacts/meta/spawn_requests/<request>.json \
  --wait-seconds 60
```

Supervisor statuses mean:

- `running`: the pane is still active and no fresh proof signoff exists yet.
- `completed`: a signoff created after the spawn request has complete proof fields.
- `failed_no_signoff`: the agent is gone and no matching fresh signoff exists.
- `failed_no_proof`: a fresh signoff exists but proof fields are incomplete.

Do not accept an optimizer result until `supervise` reports `completed` and you
have independently rerun the claimed metric.

### Improve IDDIA Itself

When editing IDDIA, stay inside `IDDIA/` unless the user explicitly asks
otherwise. Keep source data immutable, logs append-only, and indexes
rebuildable. Prefer adding or tuning:

- query-profile extraction;
- concept synonyms and affinity boosts;
- reranking or top-K diversification;
- eval scenarios;
- meta-grader criteria or score gates;
- supervisor validation.

Run this verification stack before signoff:

```bash
python -m pytest IDDIA/tests -q
uvx ruff check IDDIA
python IDDIA/evals/run_brownfield_greenfield.py
```

For the specific package you changed behavior for, also run a fresh meta-grade.
The result should show either no score gates or an intentional lower score with
a clear recommendation.

### Handoff Checklist

Before leaving:

- Confirm `git status --short -- IDDIA` only shows intentional IDDIA changes.
- Append a changelog entry with the measured result.
- Append a signoff with proof fields.
- Supervise any spawned agent until it is `completed` or an explicit failure.
- Commit scoped IDDIA changes.
- Report external blockers such as broken `bd`, dirty non-IDDIA files, or remote
  push permissions.

## Data Layout

```text
IDDIA/
  core.py
  __main__.py
  requirements.txt
  slash/agent-context/
  tests/
  artifacts/ddia/
    source/
      pdf/ddia.pdf
      manifest.json
    markdown/
      manifest.json
      pages/page_0001.md
      pages/page_0002.md
    chunks/
      chunks.jsonl
    vectors/
      manifest.json
      zvec/
    packages/
```

The layout follows the project principles the tool is meant to teach:

- Immutable source first: raw inputs have hashes and manifests.
- Logs before indexes: page Markdown and `chunks.jsonl` are the durable facts.
- Derived views are disposable: zvec can be rebuilt from `chunks.jsonl`.
- Context is bounded: package generation retrieves a small, stage-specific set.
- Handoffs carry provenance: every hit reports page, chunk id, score, tags,
  section labels when available, and why it matched.

Package generation opens zvec in read-only mode with a short retry loop so
parallel agents are less likely to trip over the local collection lock. Retrieval
asks zvec for a wider bounded candidate pool, then reranks locally with a hybrid
score:

- deterministic hash-vector similarity from zvec;
- lexical overlap, weighted toward the agent task and next steps over the
  generic stage lens;
- lexical fallback candidates from the chunk log, so a highly specific hit can
  still surface when the vector candidate set misses it;
- query-aware boosts for DDIA concepts such as event logs, snapshots,
  materialized views, manifests, schema evolution, atomicity, replay,
  checkpoints, deterministic rebuilds, batch processing, partitioning,
  consistency, durability, and source-of-truth records;
- narrow chapter affinities for common agent questions: schema migration maps
  toward Encoding and Evolution, crash recovery toward replication/stream
  recovery, deterministic replay toward batch/stream processing, and tenant
  isolation toward transaction isolation;
- penalties for low-value context such as tables of contents, front matter,
  chapter openers, index pages, and bibliography/reference-like chunks.

The package includes a `why_this_hit` explanation in JSON output and a
`Why this hit` line in Markdown output so agents can judge whether a retrieved
chunk is useful or merely nearby.

## Retrieval Evals

Brownfield and greenfield scenario tests live in `IDDIA/evals/`. They query the
same package builder agents use, suppress source snippets, grade each response
for expected concepts and chapter direction, and write a local improvement
report under ignored artifacts:

```bash
python IDDIA/evals/run_brownfield_greenfield.py
```

The latest report is written to
`IDDIA/artifacts/ddia/evals/brownfield-greenfield/latest_report.md`.

## Meta Grading And Improvement Agents

`IDDIA/meta/` is a self-contained meta system for grading IDDIA outputs and
future tool-output JSON. Runtime state is append-only and local under ignored
`IDDIA/artifacts/meta/`.

Grade a package or report JSON:

```bash
python -m IDDIA.meta grade IDDIA/artifacts/ddia/evals/brownfield-greenfield/<run>/packages/<scenario>.json \
  --expected-concept event-log \
  --expected-concept materialized-view \
  --preferred-chapter "Stream Processing"
```

Each grade writes a structured record under `IDDIA/artifacts/meta/grades/`.
Default criteria cover expected concept coverage, preferred chapter coverage,
top-hit relevance, noise flags, explanation coverage, improvement backlog
clarity, mandatory handoff/signoff fields, and handoff actionability (whether
`next_stage` is paired with concrete next-step content). Extra criteria can be
added with a JSON config containing `criteria` entries of kind `required_terms`
or `required_fields`.

Concept matching also accepts paraphrases: `replay` matches `rebuild`/`rerun`,
`manifest` is a first-class concept aliased to `provenance`/`manifests`, and the
`improvement_backlog_clarity` criterion now credits a `next_steps` field on the
package alongside an `improvement_backlog` list.

Activation policies are JSON files. Supported modes are:

```json
{"mode": "never"}
{"mode": "always"}
{"mode": "every_n", "every_n": 3}
{"mode": "threshold", "target_score": 4.0}
```

Run grading with an activation policy:

```bash
python -m IDDIA.meta grade package.json \
  --policy IDDIA/meta/activation_policy.example.json \
  --objective "Improve IDDIA retrieval quality"
```

When activation requests an improvement agent, IDDIA invokes vee through WSL
using the local vee checkout at `C:\Users\jehma\Desktop\vee` by default. The
spawner creates or reuses a tmux target first, then calls
`vee agent spawn <agent> --name <worker> --target <session:window>`, waits for
the pane to boot, and delivers the prompt with `vee agent start <worker> <prompt>`.
The agent defaults to `claude` or `IDDIA_VEE_AGENT`, and the tmux target defaults
to `iddia-meta:agents` or `IDDIA_VEE_TMUX_SESSION` / `IDDIA_VEE_TMUX_WINDOW`.
Spawned panes receive `VEE_BIN` so they can close themselves with
`node "$VEE_BIN" agent kill <worker>` when `vee` is not on PATH.
If WSL, vee, tmux, or the local checkout is unavailable, the spawner writes a pending request and prompt under
`IDDIA/artifacts/meta/spawn_requests/` instead of failing the grade.

Spawn directly from an existing grade:

```bash
python -m IDDIA.meta spawn IDDIA/artifacts/meta/grades/<grade>.json \
  --objective "Patch the weakest IDDIA retrieval criteria" \
  --vee-agent claude \
  --tmux-session iddia-meta \
  --tmux-window agents \
  --worker-name iddia-improve-schema
```

Agents can inspect durable meta context:

```bash
python -m IDDIA.meta helper-context
```

Append durable local meta notes and signoffs:

```bash
python -m IDDIA.meta changelog append "Tuned concept coverage grading."
python -m IDDIA.meta signoff append \
  --file IDDIA/meta/grader.py \
  --objective "Improve meta grading" \
  --tldr "Added a custom criterion" \
  --proof-claim "The fixed benchmark score improves under the same rubric" \
  --proof-metric "before/after score from the same grade command" \
  --proof-evidence "before: 3.41/5; after: 4.05/5" \
  --proof-verdict "partially_proven" \
  --dependency "stdlib only" \
  --dependency "vee spawning runs through WSL when available"
```

Each signoff is stored as its own file under
`IDDIA/artifacts/meta/signoffs/` and is also appended to
`IDDIA/artifacts/meta/signoffs.md` for a single-file trail.
Improvement agents are instructed to prove their point before signoff: they must
state the claim, use a fixed metric or frozen rubric, record evidence, and mark
the verdict as `proven`, `partially_proven`, or `not_proven`.

## Commands

Install the optional dependencies:

```bash
python -m pip install -r IDDIA/requirements.txt
```

Build the local knowledge artifacts:

```bash
python -m IDDIA ingest-ddia
```

Create a package for a specific agent:

```bash
python -m IDDIA package \
  --stage plan \
  --task "Add a zvec-backed agent context workflow" \
  --next-steps "Wire CLI, docs, slash commands, and verification"
```

Print the chain:

```bash
python -m IDDIA stages
```

On PowerShell you can also run:

```powershell
.\IDDIA\iddia.ps1 package --stage plan --task "..." --next-steps "..."
```

## Lifecycle

The chain loops in this order:

```text
onboard -> plan -> build -> verify -> handoff -> exit -> onboard
```

Each stage adds a different retrieval lens:

- `onboard`: source of truth, boundaries, manifests, provenance.
- `plan`: command/query separation, consistency contracts, replayable stages.
- `build`: idempotent writes, append-first facts, derived indexes.
- `verify`: invariants, recovery, source-to-index checks.
- `handoff`: decisions, durable state, next command.
- `exit`: issue status, commits, push state, replay notes.

Tracked slash-command templates live in `IDDIA/slash/agent-context/`. Local
Claude runtime copies can live in `.claude/commands/agent-context/`, which this
repo ignores. The nested directory keeps the slash namespace while avoiding
colon characters in Windows filenames.

Install local runtime copies with:

```bash
python IDDIA/install_slash_commands.py
```
