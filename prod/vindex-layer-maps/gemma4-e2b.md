# Gemma-4-E2B Vindex Layer Map

Canonical reference for *which decoder layer* each fact type should be injected
at when baking a user memory into the `gemma4-e2b.vindex` via `larql build`.

All entries below follow the same shape:

- **Fact shape** — the semantic category of the (subject, relation, object) triple.
- **Layer** — the decoder layer index (0-34) where the fact is inserted.
- **Reasoning** — the empirical / literature basis for the pick.

New rows are **append-only**. When a calibration run produces a confirmed
answer, add a row; never overwrite.

---

## Model / source reference

- Model: `google/gemma-4-E2B-it` (35 decoder layers, hidden 1536, f16)
- Vindex: `gemma4-e2b.vindex` (extract_level=all, 4.3 GB)
- Layer bands (from `index.json`):
  - syntax: L0-L13
  - knowledge: L14-L27
  - output: L28-L34

---

## Interim defaults (published-literature priors)

*Source: LARQL layer band analysis + Geva et al. (FFN as key-value memories) + ROME (Meng et al.) + Gemma-family circuit findings. These ship before empirical calibration; they are replaced as calibrated rows arrive below.*

```
┌──────────────────────────┬───────┬─────────────────────────────────────────┐
│        Fact shape        │ Layer │                Reasoning                │
├──────────────────────────┼───────┼─────────────────────────────────────────┤
│ Attribute / "is-a"       │       │ Entity-attribute facts cluster in       │
│ (Banjo is a golden       │ 20    │ mid-knowledge band                      │
│ retriever)               │       │                                         │
├──────────────────────────┼───────┼─────────────────────────────────────────┤
│ Relational (X lives-in   │ 24    │ Subject-relation-object compositions    │
│ Y, X works-at Y)         │       │ consistently peak here in Gemma family  │
├──────────────────────────┼───────┼─────────────────────────────────────────┤
│ Preference / opinion (X  │ 22    │ Between attribute and relational        │
│ loves Y, X prefers Y)    │       │                                         │
├──────────────────────────┼───────┼─────────────────────────────────────────┤
│ Proper-noun anchor       │       │                                         │
│ (codeword =              │ 26    │ Late-knowledge band, near-output        │
│ 'heliotrope')            │       │                                         │
├──────────────────────────┼───────┼─────────────────────────────────────────┤
│ Numeric value (weighs 68 │ 18    │ Earlier — numeric facts land earlier    │
│  pounds)                 │       │ per ROME                                │
├──────────────────────────┼───────┼─────────────────────────────────────────┤
│ Multi-hop (X is Y and Y  │ 24 +  │ Insert at both, redundancy helps        │
│ implies Z)               │ 26    │                                         │
└──────────────────────────┴───────┴─────────────────────────────────────────┘
```

---

## Calibrated rows (append below)

*Each new empirical finding goes here in the same table shape. Do not modify
existing rows — supersede by appending a new row with updated reasoning
that references the prior row.*
