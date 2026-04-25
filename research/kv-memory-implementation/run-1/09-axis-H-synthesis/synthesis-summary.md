# run-1 synthesis — exec summary

**Charter:** kv-memory-implementation · Run 1 · Mission `chuk-lazurus-iad` (axis-H)
**Branch:** `impl/kv-direct-wire-prop-k5`
**Lead:** lead-axis-H · session `ve-ses-0moe2fsgr0000cf0504`

## Headline

PROP K.5 KV-direct wire-up ships on Gemma-4-E2B-it global-attention layers,
end-to-end proven on a real model. The load-bearing acceptance signal is
the **smoke E2E canary FAIL → PASS transition** driven by the
axis-runtime-fix patch.

## What landed

| Component | Path | Commit |
|---|---|---|
| axis-A — global-attention layer fixture | `src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py` | `cd250b9` |
| axis-BC — PROP K.5 adapter + PROP K.0 guard | `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py` | `b22c561` |
| axis-runtime-fix — Gemma-4 KV-sharing fix | `src/chuk_lazarus/inference/backends/_torch_runtime.py` (3 sites) | `fac1f36` |
| axis-E — smoke E2E test | `tests/inference/backends/test_axis_E_kv_direct_e2e_apollo.py` | (test additions) |
| axis-runtime-fix — consumer-layer test | `tests/inference/backends/test_axis_runtime_fix_kv_consumer_layers.py` | (test additions) |
| axis-D — PATH-2 logits-equivalence | (decision-only; no production code change) | n/a |
| axis-F — regression battery | (runner-only; existing parity test re-validated GREEN) | n/a |

## What's deferred

- **axis-G** — sliding-window window-offset bookkeeping (mission
  `chuk-lazurus-cr8`; supervisor Q3 ACK
  `ve-ins-0modv5m4z00001881b9`). Filed as
  `research/kv-memory-implementation/run-1/follow-up-mission-axis-G.md`.

## What's filed forward as follow-ups (NON-bead in this run)

| # | Workstream | Vee reference |
|---|---|---|
| 1 | axis-G sliding-window bookkeeping (DEFERRED) | `ve-ins-0moe2i06j0000a88437` |
| 2 | apollo-demo data product extension | `ve-ins-0moe2ijrd000095213a` |
| 3 | vee CLI patch (P1 + P2/P3) | `ve-ins-0moe2j2ym00006ece06` |
| 4 | axis-rope-phase-fix (`_prepare_archived_prefix` RoPE-identity) | `ve-ins-0moe2k6qp0000ca2643` |

## Load-bearing canary

The single most informative signal in run-1 is the smoke E2E canary
`test_kv_direct_synthetic_smoke_e2e_layer_29`:

- **Before axis-runtime-fix (HEAD `b22c561`):** FAIL with
  `AttributeError: 'Gemma4TextAttention' object has no attribute 'k_proj'`
  at `_torch_runtime.py:2165`. Gemma-4-E2B-it strips
  `k_proj`/`v_proj`/`k_norm`/`v_norm` from KV-consumer layers (29..34).
- **After axis-runtime-fix (HEAD `fac1f36`):** PASS — wire flows from
  synthetic K/V through the BC adapter, through `KVDirectMaterialization`,
  through `runtime.generate_with_kv_direct_materialization`, through
  `model.generate`. CUDA bf16; 70.39 s wall-clock.

Per-axis verdicts and AMD compliance live in `amd-compliance.md`. Validator
anti-patterns observed in run-1 are documented in
`validator-anti-pattern-recurrence.md`.

## Recordkeeping (one-screen)

- Recipe: `ve-ins-0modtwi7v0000ff6d88` `[OWNER_KV_RECIPE_V1]`
- End-state pointer: `ve-ins-0moduwf2i000085f08a` (axis-H)
- Manifest: `ve-ins-0moe28bjw0000809b2c`
- Run-debrief (full): `research/kv-memory-implementation/run-1/run-debrief.md`
- Wire-up narrative: `docs/kv-memory-prop-k5-wire-up.md`

## Next-run priority order

1. axis-rope-phase-fix (½–1 day; HIGH).
2. apollo-demo data product extension (1–2 days; HIGH).
3. vee CLI patch P1 + P2 (vee-maintainer team; HIGH).
4. axis-G window-offset bookkeeping (2–4 days; MEDIUM; depends on (2)).

After (1)–(4) land, `impl/kv-direct-wire-prop-k5` becomes a candidate for
supervisor-gated promotion to main.
