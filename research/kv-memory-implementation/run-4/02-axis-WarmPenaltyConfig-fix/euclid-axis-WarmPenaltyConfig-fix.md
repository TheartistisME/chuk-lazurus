# /euclid proof chain — axis-WarmPenaltyConfig-fix (HOT bonus arithmetic + JSON round-trip)

> Authored as part of kv-memory-implementation run-4 axis-2.
> Lead session: ve-ses-0moed1ikk00008d3de0.
> Branch: impl/kv-memory-finalize-run-4 (off main @ f6129e2).
> Decision: DEFAULT (per Q2 supervisor binding ve-ins-0moecncxb00006f970b).

## CLAIM

The Q2 DEFAULT extension to `WarmPenaltyConfig` at
`src/chuk_lazarus/inference/backends/torch_runtime.py:2910` adds an optional
`hot_bonus_value: float | None = None` field (line 2926) and wires the
documented contract `logit += hot_bonus_value` for HOT-tier slices in
`apply_tier_attention_mask` (line 3030-3038), with full JSON envelope
round-trip support in `attention_tier_mask_inputs_to_dict` (lines 3096-3100)
and `attention_tier_mask_inputs_from_dict` (lines 3171, 3176). Backward
compatibility is preserved: `ATTENTION_TIER_MASK_SCHEMA_VERSION` is held at
1 and run-1/2/3 envelopes lacking the new key deserialize cleanly with
`hot_bonus_value=None`. The cross-file API contract between
`scripts/interactive_memory_chat.py:1132` and the dataclass is enforced by a
new AST-based contract test that catches the run-3 `/kv_query` REPL crash
class (ve-ins-0moe7elql0000afaa2b) by static inspection — without
importing the heavy chat-script module.

## PASS test

- pytest node-id: `tests/inference/backends/test_axis_WarmPenaltyConfig_contract.py`
  (whole-file invocation; 5 tests)
- run command: `uv run pytest tests/inference/backends/test_axis_WarmPenaltyConfig_contract.py -v --tb=short`
- run-4 jsonl: `prod/validation/diagnostic_axis_warmpenaltyconfig_20260425T133138Z-192e9bcb.jsonl`
- result: PASS (5/5); CUDA-gated test ran on RTX 5090 (not skipped), bf16
  baseline tensor allocation on `device='cuda'` per AMD 14.
- evidence file:line citations:
  - `src/chuk_lazarus/inference/backends/torch_runtime.py:2926` — dataclass
    field `hot_bonus_value: float | None = None` (already in WIP at run-4 entry)
  - `src/chuk_lazarus/inference/backends/torch_runtime.py:3030-3038` — HOT
    branch in `apply_tier_attention_mask` applies the bonus additively when
    set; preserves byte-identity when `None` (legacy contract)
  - `src/chuk_lazarus/inference/backends/torch_runtime.py:3096-3100` —
    serializer warm_config_dict carries `hot_bonus_value` (None → null)
  - `src/chuk_lazarus/inference/backends/torch_runtime.py:3171, 3176` —
    deserializer reads `.get("hot_bonus_value", None)` and forwards into
    `WarmPenaltyConfig(...)` constructor
  - `tests/inference/backends/test_axis_WarmPenaltyConfig_contract.py` —
    5 tests (4 pure-python + 1 CUDA-gated tensor); all PASS

## FAIL behavior

The 5 contract tests assert, in concert:

1. **`test_construct_with_hot_bonus_value`** — Full-shape and chat-script
   kwarg-only construction (`WarmPenaltyConfig(hot_bonus_value=2.0)`) must
   succeed without TypeError and the field must be readback-correct. Any
   regression that drops the field from the dataclass surfaces as
   TypeError → AssertionError here.

2. **`test_construct_without_hot_bonus_value`** — Backward-compat: legacy
   3-field construction shape (`WarmPenaltyConfig(penalty_value=...,
   per_warm_uniform=..., clamp_min=...)`) must continue to work and yield
   `hot_bonus_value is None`. Any regression that makes the field
   non-optional fires this assertion.

3. **`test_chat_script_construction_path_kwarg_alignment`** — STATIC AST
   parse of `scripts/interactive_memory_chat.py`; for every
   `WarmPenaltyConfig(...)` Call node, every kwarg name must be a valid
   field of `dataclasses.fields(WarmPenaltyConfig)`. THIS is the test that
   catches the run-3 `/kv_query` REPL crash class — a future regression
   adding a new kwarg at the call site without growing the dataclass field
   surfaces as `unknown = kwarg_names - valid_field_names` → AssertionError
   with a message that names the unknown kwarg(s) explicitly. The test
   skips with a clear reason if the call site is removed or imported
   lazily.

4. **`test_bonus_arithmetic_applied_when_hot`** (CUDA-gated) — Pure-tensor
   verification that `logit += hot_bonus_value` is observable. Two calls on
   the same baseline `attn_logits = torch.zeros(1, 1, 1, 4, device='cuda')`:
   Call A with `hot_bonus_value=None` (identity baseline; HOT slices remain
   zero) and Call B with `hot_bonus_value=2.0` (HOT slices become 2.0).
   WARM slices must be byte-equal across A and B (HOT bonus does not leak
   into WARM). Sanity check: WARM penalty fires (default `penalty_value=4.0`
   yields WARM logit = -4.0). A regression that fails to apply the bonus
   makes B's HOT slice equal A's (zero); the assertion
   `torch.equal(out_b[..., 0:2], torch.full_like(..., 2.0))` fires.

5. **`test_json_envelope_round_trip_preserves_hot_bonus_value`** —
   Round-trip with `hot_bonus_value=3.5`, with `None`, AND from a legacy
   envelope (no `hot_bonus_value` key in `warm_config` dict) must all
   produce the expected dataclass state. A regression that bumps schema
   version, drops the serializer field, or stops accepting legacy
   envelopes fires this test.

## UNKNOWN edges (out of scope of this proof chain)

- Live `/kv_query` REPL smoke run on the actual chat process is deferred to
  Axis-5 e2e per Q2 binding. The contract test
  (`test_chat_script_construction_path_kwarg_alignment` +
  `test_construct_with_hot_bonus_value`) is sufficient evidence here that
  the construction path no longer raises.
- Numerical equivalence of HOT bonus + WARM penalty composition under
  GQA/KV-sharing layer routing is not asserted in this chain; that's the
  axis-runtime-fix concern (see run-1 `euclid-axis-runtime-fix.md`).
- Bonus magnitude calibration (what `hot_bonus_value` value actually
  improves recall quality) is a downstream tuning concern, not a
  correctness concern; this proof chain only locks in the additive shape.
- Interaction with `clamp_min` on HOT (currently HOT has no clamp; WARM
  does) is intentional per the run-1 contract record
  ve-ins-0mo9p9ke3000060c0c7 — HOT is identity-or-bonus, no clamp.

## adaptation-status

- run-4 verdict: PASS (5/5 contract tests; CUDA-gated test ran)
- known bugs: none introduced by axis-2 surgery
- regression risk: low
  - Surgery is localized to 3 sites (HOT branch arithmetic + serializer
    field + deserializer field-with-default)
  - Schema version unchanged; legacy envelopes deserialize cleanly
  - Pure-python construction tests + AST contract test catch the run-3
    crash class statically
  - Tensor test verifies the additive shape with byte-equal assertions on
    bf16/fp32 tensors
  - Pre-existing failures in `tests/inference/backends/` (3/150 — axis-E
    schema-gap, storage format, frozen-dataclass) are unrelated to
    WarmPenaltyConfig surgery (verified by reading their test bodies)
- next-mission recommendations:
  - Axis-5 e2e: live `/kv_query` REPL smoke probe with `LAZARUS_KV_HOT_BONUS=2.0`
    to confirm end-to-end (this axis closes the contract; Axis-5 closes the
    user-facing REPL gate)
  - Axis-4 (session-route): the chat session-route construction site at
    `scripts/interactive_memory_chat.py:1132` is now contract-locked; any
    future kwarg additions there are gated by
    `test_chat_script_construction_path_kwarg_alignment`

## Cross-refs

- Bug authority: vee record `ve-ins-0moe7elql0000afaa2b` (run-3 `/kv_query`
  REPL crash; canonical two-option patch tree)
- Axis-2 end-state: `ve-ins-0moebmdps0000540c8e`
- Axis-2 mission proposal: `ve-ins-0moebpdpm00009181eb`
- Critical-learning record: `ve-ins-0moe7opep00003ef31f` (cross-file
  contract tests are mandatory; pure unit tests do NOT catch this class)
- Recipe authority: `ve-ins-0modtwi7v0000ff6d88` `[OWNER_KV_RECIPE_V1]`
- Run-3 axis chains (format precedent):
  `research/kv-memory-implementation/run-3/01-e2e-testing/euclid-axis-rope-phase-fix.md`
  and siblings
- Surgery pattern record: `ve-ins-0moedbfbd0000363355`
- Validator GREEN record: `ve-ins-0moedp6wz0000951c2e`
- Baseline reference record: `ve-ins-0moed6ehy0000f812c4`
