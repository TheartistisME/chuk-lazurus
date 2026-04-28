# IDDIA Retrieval Evals

This directory contains brownfield and greenfield scenarios for checking how
well IDDIA packages context for agent work at different lifecycle stages.

Run the suite from the repo root:

```bash
python IDDIA/evals/run_brownfield_greenfield.py
```

The runner:

- calls the same JSON package builder used by agents;
- suppresses source snippets in saved eval outputs;
- grades each scenario for expected concept coverage, preferred chapter
  direction, explanation coverage, noisy hits, and top-hit relevance;
- writes per-scenario packages and a report under
  `IDDIA/artifacts/ddia/evals/brownfield-greenfield/`.

## Scenario Shape

Each scenario defines:

- `mode`: `brownfield` or `greenfield`;
- `complexity`: `simple`, `medium`, or `complex`;
- `stage`: one lifecycle stage from `onboard`, `plan`, `build`, `verify`,
  `handoff`, or `exit`;
- `task` and `next_steps`: the query sent to the tool;
- `expected_concepts`: semantic signals the retrieved context should cover;
- `preferred_chapters`: DDIA chapter directions that should appear in top hits.

## Current Improvement Priorities

The first brownfield/greenfield run showed strong coverage overall, but the next
retrieval work should focus on:

- schema migration queries that mention compatibility, old readers, manifests,
  lineage, rollback, and replayable rebuilds;
- stronger replay and manifest recognition for document indexing and
  multi-tenant artifact-service prompts;
- checkpoint recognition for distributed training drift and restart validation;
- regression assertions that keep schema migration anchored in Encoding and
  Evolution while still surfacing replay and lineage context.

