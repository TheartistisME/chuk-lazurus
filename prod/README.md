# `prod/` — production hub

Hub directory for everything that graduates from research/experiment status
into a shipped, documented, reproducible piece of the `chuk_lazurus` stack.
Each subdirectory owns one research line; this README routes you to the
right one.

## Subdirectories

| Directory | What it holds | Status |
|---|---|---|
| `canonical/` | Upstream `chrishayuk/chuk-lazurus` source excerpts (offline reference) | Zero-mod, never edit |
| `validation/` | Evidence trail from the turn-aligned-canonical-port run-1 | Frozen audit — run-1 complete 2026-04-21 |
| `vindex-layer-maps/` | Per-model layer maps for LARQL fact injection (fact-shape → layer N) | Active research — append-only findings |

## Pipeline summary (what ships from here)

1. **Turn-aligned canonical port** (run-1, ACHIEVED): cross-session residual
   retrieval gained architectural parity with the canonical MLX implementation
   on Gemma-4-E2B-it / PyTorch-HF. See the section below and
   `validation/ROUND3_VERDICT.md` / `CRITERION4_VERDICT.md`.
2. **LARQL layer-map calibration** (ongoing): systematic mapping of fact
   shape → decoder layer for the `larql build` baking step that commits
   chat-session memories into model weights. See `vindex-layer-maps/README.md`.

---

## Section 1 — turn-aligned-canonical-port (run-1)

**Status:** `scope-complete` · `ACHIEVED` (4/4 acceptance criteria PASS)
**Run:** run-1, lead-only topology, autonomous mode
**Completed:** 2026-04-21
**Root session:** `ve-ses-0mo89la170000320133`

This run gave `chuk_lazurus`'s turn-aligned conversational memory architectural parity with the canonical `chrishayuk/chuk-lazurus` knowledge-store implementation, adapted for Gemma-4-E2B-it on PyTorch-HF.

## What it does

Before this run: our cross-session retriever on conversation content produced **token-salad** because `generate_with_residual` replaced only the LAST position's hidden state at the crystal layer (29). Layers 30-47 then attended to a KV cache anchored on a high-magnitude donor vector with no semantic context.

After this run: a new method `generate_with_residual_prefill_seeded` mirrors the canonical MLX `prefill_to_layer(initial_residual=boundary) + prefill_from_layer` mechanism on Gemma-4's optimised `input_ids` path. Seed token at position 0, `forward_pre_hook` on `layers[0]` replaces its post-embedding hidden state with the donor residual, real prompt tokens flow through. Layers 30-47 attend to a coherent KV cache.

## Layout

```
prod/
├── README.md               ← you are here (hub index)
├── RUNBOOK.md              ← how to run / test / extend (run-1)
├── ARCHITECTURE.md         ← canonical-parity spec + Gemma-4 adaptation
├── VEE_RECORDS.md          ← curated trail of canonical records
├── CHANGELOG.md            ← run-1 delivery summary
├── canonical/              ← upstream chrishayuk/chuk-lazurus excerpts (offline reference)
│   ├── canon_inject.py
│   ├── canon_cli_query.py
│   ├── canon_build.py
│   ├── canon_config.py
│   ├── canon_store.py
│   ├── canon_route.py
│   └── …
├── validation/             ← evidence trail from run-1
│   ├── ROUND3_VERDICT.md         ← post-refactor verification (criteria 1-3)
│   ├── CRITERION4_VERDICT.md     ← end-to-end verification (criterion 4)
│   ├── VERDICT.md, VERIFICATION_REPORT.md
│   ├── direct_probe_results.json ← 3/3 verbatim hits
│   ├── direct_probe.py           ← UUID-real probe harness
│   ├── criterion4_report.json    ← VerificationReport from e2e
│   ├── 06-pytest-round3.log      ← pytest 45/45 pass
│   ├── 07-direct-probe-round3.log
│   ├── 08-apollo-q{1,2,3}-round3.log ← Apollo demo 3/3 coherent
│   ├── 09-criterion4-e2e.log
│   └── BLOCKING_ISSUES.txt
└── vindex-layer-maps/      ← per-model layer maps for LARQL fact injection
    ├── README.md                 ← schema + append-only discipline
    └── gemma4-e2b.md             ← google/gemma-4-E2B-it layer map
```

## Production code location

Code lives at its canonical package location under `src/chuk_lazarus/` so imports and tests work without path rewrites. **See `RUNBOOK.md` for exact paths and line numbers.** Quick pointers:

| File | Role |
|------|------|
| `src/chuk_lazarus/inference/backends/torch_runtime.py:307-472` | New method `generate_with_residual_prefill_seeded` |
| `src/chuk_lazarus/session_retrieval/retriever.py:394` | Rewire to new method |
| `src/chuk_lazarus/chat_loop/` | Conversational entry (axis-1) |
| `src/chuk_lazarus/session_close/` | Session → AUS3000 clause emission (axis-2) |
| `src/chuk_lazarus/session_store/` | Clause-aligned torch store build (axis-3) |
| `src/chuk_lazarus/session_retrieval/` | Cross-session retriever (axis-4) |
| `src/chuk_lazarus/cross_session_demo/` | End-to-end harness (axis-5) |
| `scripts/cross_session_demo.sh` | Shell wrapper for the full 5-axis flow |
| `scripts/criterion4_e2e_verify.py` | Reproducible criterion-4 harness |
| `tests/session_retrieval/` | 45 tests for the retriever path |

## Zero-mod primitives (inviolate)

These files are proven primitives from the upstream canonical implementation. **Never modify.**

- `tools/build_clause_aligned_store.py`
- `src/chuk_lazarus/inference/context/knowledge/inject.py`
- `src/chuk_lazarus/inference/context/knowledge/torch_query.py`
- `src/chuk_lazarus/inference/context/knowledge/torch_store.py`
- `src/chuk_lazarus/inference/context/knowledge/torch_capture.py`
- `src/chuk_lazarus/inference/context/knowledge/torch_build.py`

## Quick start

See `RUNBOOK.md`. TL;DR:

```bash
# Verify the port is installed and tests pass
pytest tests/session_retrieval/

# Run the criterion-4 end-to-end harness
python scripts/criterion4_e2e_verify.py --output-root /tmp/csd-smoke

# Run the Apollo demo against the new method
bash scripts/run_apollo_memory_demo.sh
```

## Canonical trail

All decisions, evidence, and interventions are recorded in vee. The curated reading list is in `VEE_RECORDS.md` — start there if you need to understand **why** the implementation looks the way it does.

---

## Section 2 — LARQL layer-map calibration (ongoing)

When a user memory is baked into a vindex via `larql build`, the pipeline
must pick a target decoder layer. The right layer depends on the fact's
semantic shape (attribute, relation, numeric, proper-noun anchor, multi-hop)
and on the model family. `vindex-layer-maps/` holds one file per model
recording the calibrated `fact-shape → layer` mapping.

Files are **append-only**. Interim defaults come from published literature
(LARQL band analysis, Geva et al., ROME). Calibrated rows come from the
logit-lens probe harness running against each model's vindex on the 5090.

Start here: [`vindex-layer-maps/README.md`](vindex-layer-maps/README.md).
Active map: [`vindex-layer-maps/gemma4-e2b.md`](vindex-layer-maps/gemma4-e2b.md).
