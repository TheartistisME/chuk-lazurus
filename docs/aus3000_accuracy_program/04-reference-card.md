# AUS3000 Reference Card

## Goal

Get the clause-aligned AUS3000 Lazarus/Gemma knowledge variant to a strict `single_pass_gate` score of `23/23 PASS` with no electrical bleed on out-of-domain prompts.

## Current Status

- Status: `complete`
- Latest clean live strict rerun: `23/23 PASS`
- Latest report: `/tmp/aus3000_single_pass_gate_rerun.txt`
- Stable no-regression bucket: `17/17 PASS`
- Stable out-of-domain bucket: `6/6 PASS`
- Route gate: `17/17 PASS`
- Grounding gate: `17/17 PASS`

## What Has Been Completed

- Built JSON to corpus tooling for TradeGuru and AUS3000 ingestion.
- Built the clause-aligned AUS3000 store and Lazarus checkpoint variant.
- Added AUS3000 benchmark fixture and strict gate evaluator flow.
- Added clause-aware exact routing using `window_metadata.json`.
- Hardened exact routing so explicit clause IDs dominate.
- Demoted ambiguous aliases into routing hints instead of exact matches.
- Added safe one-token exact-title handling for unique non-generic titles.
- Replaced the blanket `_single_token_in_longer_alias` guard with a
  2-token-conflict count. Single-token full titles now become exact
  matches when the token is not in the generic blocklist and appears in
  fewer than two 2-token aliases, which unblocks `accessible`,
  `insulated`, `switchboard` (and similar safe definitions) while
  keeping `cable`, `circuit`, `conductor`, `wiring` ambiguous.
- Added containment-based suppression in `_collect_pattern_matches`:
  a shorter pattern whose occurrence falls entirely inside the range of
  a longer already-matched pattern is skipped and the next non-contained
  occurrence is searched for instead. Prevents `(earthed,)` leaking in
  from `(multiple, earthed, neutral, men, system)`.
- Fixed the final three failing cases:
  - `accessible_vs_readily` — routes `['1.4.2', '1.4.3']` in order.
  - `insulated_definition` — routes `['1.4.72']`.
  - `switchboard_definition` — routes `['1.4.121']`.
- Earlier fixes that remain in place:
  - `showers_and_bathrooms`
  - `rcd_not_sole_basic_protection`
  - `rcd_live_conductor_faults`
  - `domestic_residential_rcds_au`
  - `insulation_resistance_results`
  - `efli_when_required`
  - `operation_of_rcds`
  - all out-of-domain and adversarial cases in the strict gate
- Hardened the evaluator so strict mode writes its output file before runtime/store load begins.

## Primary Artifacts

- Dataset root:
  - `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018`
- Corpus:
  - `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/as_nzs_3000_2018_lazarus_corpus.txt`
- Clause-aligned checkpoint:
  - `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant`
- Clause-aligned store:
  - `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant/torch_store`
- Benchmark fixture:
  - [tests/fixtures/aus3000/benchmark/epic1_v1.json](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/tests/fixtures/aus3000/benchmark/epic1_v1.json)
- Latest rerun report:
  - `/tmp/aus3000_single_pass_gate_rerun.txt`

## Key Code Paths

- Exact/title routing:
  - [route.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/src/chuk_lazarus/inference/context/knowledge/route.py)
- Store routing + exact/hint merge:
  - [torch_store.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/src/chuk_lazarus/inference/context/knowledge/torch_store.py)
- Query preparation and exact-route handling:
  - [torch_query.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/src/chuk_lazarus/inference/context/knowledge/torch_query.py)
- Evaluator and strict gate:
  - [evaluate_aus3000_variant.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/tools/evaluate_aus3000_variant.py)

## Key Test Coverage

- Route/store exact-routing tests:
  - [test_aus3000_clause_route.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/tests/inference/context/test_aus3000_clause_route.py)
  - [test_torch_store.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/tests/inference/context/test_torch_store.py)
- Query-path exact-routing tests:
  - [test_aus3000_torch_query.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/tests/inference/context/test_aus3000_torch_query.py)
- Evaluator strict-gate tests:
  - [test_evaluate_aus3000_variant.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/tests/tools/test_evaluate_aus3000_variant.py)

All four files: `25 passed, 1 skipped` after the fix.

## Live Rerun Command

```bash
uv run python tools/evaluate_aus3000_variant.py \
  --mode single_pass_gate \
  --max-cases 23 \
  --device cuda \
  --output /tmp/aus3000_single_pass_gate_rerun.txt
```

## Implementation Snapshot

- Current local branch state when this card was written: `main...origin/main [ahead 48, uncommitted]`
- Latest implementation commit (pre-fix): `a87eb70 Finalize AUS3000 exact-routing and evaluator hardening`
- Pending local changes for this pass:
  - `src/chuk_lazarus/inference/context/knowledge/route.py`
  - `tests/inference/context/test_aus3000_clause_route.py`
  - `tests/inference/context/test_aus3000_torch_query.py`
  - `tests/inference/context/test_torch_store.py`

## Known Environment Issue

- Heavy CUDA benchmark/test runs have intermittently hit WSL/kernel `D`-state I/O stalls.
- The evaluator writes a startup marker before runtime/store load so a cold-start stall is visible in the report file.
- Logic-level direct assertions for route/store/query/evaluator are passing alongside the heavyweight live runner.

## Next Exact Moves

- Commit the two-part routing fix and the aligned unit tests.
- Consider promoting the `soak_gate` run once the `single_pass_gate` result is locked to `23/23`.
