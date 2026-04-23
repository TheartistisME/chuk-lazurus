# `prod/vindex-layer-maps/`

Per-model layer maps that tell the LARQL baking pipeline **at which decoder
layer each fact type should be injected** when a user memory is committed
into a vindex.

## Why this directory exists

When `chuk_lazarus` bakes a user-emitted fact (e.g. *"Banjo is a golden
retriever"*) into a model's weights via `larql build`, it must pick a target
layer. Different fact shapes — attributes, relational triples, numeric
values, proper-noun anchors, multi-hop compositions — live at different
depths in the transformer. Picking the wrong layer either:

- **too shallow**: the inserted feature is overwritten by later layers'
  computation and never reaches the decoder head, OR
- **too deep**: the feature lands after the layer that would have used it,
  so no downstream circuit can compose with it.

The right layer is the one **just before the layer at which the model
would naturally output the target token** (logit-lens criterion). That
layer varies by fact type and by model family. One file per model captures
the mapping.

## File naming

One Markdown file per model:

```
vindex-layer-maps/
├── README.md             ← you are here
└── gemma4-e2b.md         ← google/gemma-4-E2B-it (35 layers, hidden 1536)
```

Future models (e.g. `gemma4-e4b.md`, `qwen3-4b.md`) follow the same
schema.

## File schema

Every model file has three sections:

1. **Model / source reference** — HF id, layer count, hidden size, vindex
   layer bands from `index.json`.
2. **Interim defaults** — table of fact-shape → layer picks derived from
   published literature (LARQL band analysis, Geva et al., ROME). Ships
   before any empirical calibration.
3. **Calibrated rows** — **append-only** table rows produced by running
   the logit-lens probe harness against that model's vindex. Each row
   records: fact shape, layer, reasoning.

## Append-only discipline

- **Never overwrite** an existing row. If a later experiment supersedes
  an earlier layer pick, add a new row that references the prior row in
  its "Reasoning" column (e.g. *"supersedes numeric-value L18 after
  N=200 probe"*).
- **Exact table shape required** — every row uses the same
  `┌──────────┬──────┬──────────┐` Unicode box schema so the files stay
  grep-able and diffable. See `gemma4-e2b.md` for the canonical shape.
- **One row per finding** — no batching. If a calibration run produces
  10 new layer picks, emit 10 rows.
- **Timestamp-free** — the file is a layer map, not a timeline. Git history
  is the source of truth for chronology.

## Who reads this file

- **`chuk_lazarus.larql_backend.classify_fact(triple)`** — at `/save`
  time, looks up the fact shape → layer mapping for the active model.
- **Research harnesses** (`memory_layer_probe.py` and successors) —
  append new rows as they produce calibrated answers.
- **Humans auditing a baked vindex** — cross-reference an inserted
  feature's `@L<N>` assignment against the canonical reasoning here.

## Workflow: adding a calibrated row

1. Run the probe harness for a new fact shape (or a shape you want to
   refine): see `src/chuk_lazarus/memory_layer_probe.py` (forthcoming).
2. Harness emits `(fact_shape, layer, reasoning_summary)`.
3. Append the row to the correct model's file, under the "Calibrated rows"
   section, in the canonical box shape.
4. Commit with a message like `layer-maps: add L22 for "location-of"
   (Gemma-4-E2B, N=50 probe, P(target) margin 3.2x)`.
