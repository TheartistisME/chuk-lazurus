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
- Handoffs carry provenance: every hit reports page, chunk id, score, and tags.

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
