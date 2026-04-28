# Agent Context From DDIA

This workflow turns a legally available DDIA PDF into an agent-stage context
system:

1. The PDF is stored as immutable source under `artifacts/agent_context/ddia/source/`.
2. Each PDF page is converted to one Markdown file with Microsoft MarkItDown.
3. Page Markdown is chunked into JSONL records with stable ids and provenance.
4. zvec stores a rebuildable vector index over those chunk records.
5. `lazarus agent-context package` creates bounded context packages for an
   agent task, lifecycle stage, and next steps.

`artifacts/` is ignored by git. The repo should carry the pipeline, schemas,
and command chain; book-derived source text and vector stores stay local and
reproducible.

## Data Layout

```text
artifacts/agent_context/ddia/
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
- Handoffs carry provenance: every hit reports page, chunk id, score, and tags.

## Commands

Install the optional dependencies:

```bash
python -m pip install "markitdown[pdf]>=0.1.3" "pypdf>=5.0.0" "zvec>=0.3.0"
```

Build the local knowledge artifacts:

```bash
lazarus agent-context ingest-ddia
```

Create a package for a specific agent:

```bash
lazarus agent-context package \
  --stage plan \
  --task "Add a zvec-backed agent context workflow" \
  --next-steps "Wire CLI, docs, slash commands, and verification"
```

Print the chain:

```bash
lazarus agent-context stages
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

Tracked slash-command templates live in `prompts/slash/agent-context/`. Local
Claude runtime copies can live in `.claude/commands/`, which this repo ignores.

Install local runtime copies with:

```bash
python scripts/install_agent_context_slash_commands.py
```
