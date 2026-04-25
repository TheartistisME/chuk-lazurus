# axis-2 axis-WarmPenaltyConfig-fix notes (run-4)

## Lead session
ve-ses-0moed1ikk00008d3de0

## Decision
DEFAULT (per Q2 supervisor binding ve-ins-0moecncxb00006f970b).

The dataclass field was already present in WIP at run-4 entry; the
remaining work was the bonus arithmetic at the consumption site + JSON
envelope round-trip + cross-file contract test. A no-op extension would
have left observable behaviour identical to FALLBACK; DEFAULT was selected
to actually enable the documented contract `logit += hot_bonus_value`.

## Surgery sites (post-edit)
- `src/chuk_lazarus/inference/backends/torch_runtime.py:3030-3038` — HOT
  branch in `apply_tier_attention_mask`
- `src/chuk_lazarus/inference/backends/torch_runtime.py:3096-3100` —
  serializer warm_config_dict
- `src/chuk_lazarus/inference/backends/torch_runtime.py:3171, 3176` —
  deserializer

## Test surface
- `tests/inference/backends/test_axis_WarmPenaltyConfig_contract.py` (NEW;
  5 tests; 266 LoC)

## Attestation
- `prod/validation/diagnostic_axis_warmpenaltyconfig_20260425T133138Z-192e9bcb.jsonl`
  (4 records: header, contract_test, regression_battery, smoke_kv_query_no_crash)

## Vee records
- baseline: ve-ins-0moed6ehy0000f812c4
- surgery pattern: ve-ins-0moedbfbd0000363355
- validator GREEN: ve-ins-0moedp6wz0000951c2e

## Key insights
- Bug record ve-ins-0moe7elql0000afaa2b cited line 998 of chat script as
  construction site; current WIP construction site is line 1132 (line
  shift due to ~150-line vec_inject WIP). Functional content unchanged.
- The crash itself is no longer reproducible on
  `impl/kv-memory-finalize-run-4` because the WIP dataclass already grew
  the field. The residual semantic gap on this branch was the missing
  bonus arithmetic — DEFAULT path closes that gap.
- AST-based contract test is robust to module-level chat-script side
  effects (heavy argparse + model loading guards) — does NOT import the
  script; reads as text and walks the Call AST.
