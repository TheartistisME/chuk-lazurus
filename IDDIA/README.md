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
clarity, and mandatory handoff/signoff fields. Extra criteria can be added with
a JSON config containing `criteria` entries of kind `required_terms` or
`required_fields`.

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
cadence is `vee agent spawn <agent> --name <worker> --then <prompt>`, with the
agent name defaulting to `claude` or `IDDIA_VEE_AGENT`. If WSL, vee, tmux, or
the local checkout is unavailable, the spawner writes a pending request and prompt under
`IDDIA/artifacts/meta/spawn_requests/` instead of failing the grade.

Spawn directly from an existing grade:

```bash
python -m IDDIA.meta spawn IDDIA/artifacts/meta/grades/<grade>.json \
  --objective "Patch the weakest IDDIA retrieval criteria" \
  --vee-agent claude
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
  --dependency "stdlib only" \
  --dependency "vee spawning runs through WSL when available"
```

Each signoff is stored as its own file under
`IDDIA/artifacts/meta/signoffs/` and is also appended to
`IDDIA/artifacts/meta/signoffs.md` for a single-file trail.

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
