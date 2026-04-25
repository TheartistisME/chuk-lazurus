# AMD 1-15 Compliance Audit (run-1)

Per-AMD audit of the run-1 amendment table inherited verbatim by all leads.
Each AMD is graded `COMPLIED` / `DEVIATED` / `N/A` with citation evidence.

**Convention:** an AMD is `COMPLIED` if its prescription was honoured by the
shipped artefacts. `DEVIATED` is recorded only when a deliberate, supervisor-acknowledged
departure occurred. `N/A` covers AMDs that don't bind axis-H scope.

## AMD 1 — single-writer-per-axis

**Status:** COMPLIED

**Evidence:** Each axis (A, BC, D, E, F, runtime-fix, G, H) had exactly one
LEAD pane. No two LEADs held write authority over the same scope manifest
glob simultaneously. The axis-BC bundling of missions
`chuk-lazurus-vnw` + `chuk-lazurus-3y8` was supervisor-authorised in the
manifest `ve-ins-0modvo9qs0000f767db` and folded both into a single LEAD.

## AMD 2 — quote-in-full / Pattern A authority

**Status:** COMPLIED

**Evidence:** All LEAD invocations carried Pattern A (autonomous,
quote-in-full) per supervisor Q-ACK and ATTRIBUTION-PROTOCOL OVERRIDE
`ve-ins-0mody4xaa0000d8c5ab`. axis-H ran Pattern A end-to-end.

## AMD 3 — read-only canonical references

**Status:** COMPLIED

**Evidence:** `examples/inference/demo_c_apollo11_torch.py` was treated as
read-only by all axes. `[OWNER_KV_RECIPE_V1]` (`ve-ins-0modtwi7v0000ff6d88`)
was treated as read-only authority. axis-H referenced both as research
context only, no edits.

## AMD 4 — recipe-correction note required for synthesis

**Status:** COMPLIED

**Evidence:** Run-debrief `Recipe-correction note` section explicitly
documents that the canonical global-attention set is the axis-A
enumeration `{4, 9, 14, 19, 24, 29, 34}` (NOT the recipe's empirical
parity set). Section also appears in `docs/kv-memory-prop-k5-wire-up.md`
under "The problem" and CHANGELOG run-notes.

## AMD 5 — feature-branch hygiene (commits NOT on main)

**Status:** COMPLIED

**Evidence:** `git log --oneline -10` shows all run-1 commits
(`cd250b9`, `b22c561`, `fac1f36`, axis-H docs commit) are on
`impl/kv-direct-wire-prop-k5`. Merge to `main` is held pending supervisor
authorisation.

## AMD 6 — CUDA-only test execution

**Status:** COMPLIED

**Evidence:** All run-1 tests are CUDA-gated (e.g.
`test_axis_E_kv_direct_e2e_apollo.py` requires `cuda` device; axis-F
parity battery runs on `cuda` bf16 RTX 5090; axis-runtime-fix new test
`test_axis_runtime_fix_kv_consumer_layers.py` is CUDA-gated). No CPU
fallback variants were authored.

## AMD 7 — pinned snapshot for Gemma-4-E2B-it

**Status:** COMPLIED

**Evidence:** Snapshot `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf` is
referenced uniformly across axis-A enumeration, axis-BC adapter tests,
axis-E E2E, axis-F regression, and axis-runtime-fix tests.

## AMD 8 — `full_attention` aliased to `global_attention`

**Status:** COMPLIED

**Evidence:** axis-A fixture file
`src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py`
treats the literal label `full_attention` as `global_attention`; the
exported set name is `GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS`. The axis-A
README and `gemma4_e2b_it_layer_types_b4a60110.jsonl` document the
alias.

## AMD 9 — gap-is-the-star

**Status:** COMPLIED (this is the load-bearing AMD for run-1)

**Evidence:** PROP K.5 wire is **fully proven end-to-end** on
global-attention layers via:

- 22/22 axis-BC unit tests PASS
- axis-D PATH-2 logits-equivalence proof PASS
- axis-F regression GREEN (5 PASS + 1 SKIP_LINEAGE)
- **axis-E smoke E2E canary FAIL → PASS** (load-bearing transition; the
  star). The smoke test exercises the complete wire on a real
  Gemma-4-E2B-it model.
- axis-runtime-fix new test PASS (consumer-layer routing covered)

The "gap" — sliding-layer injection + apollo-demo full fact-recall — is
named, scoped, and filed as named follow-ups (axis-G + apollo data
extension).

## AMD 10 — research-dir mirroring of debrief

**Status:** COMPLIED

**Evidence:** Run-debrief mirrored at
`research/kv-memory-implementation/run-1/run-debrief.md`. axis-G defer
follow-up mirrored at `08-axis-G-defer/follow-up-mission-axis-G.md` AND
top-level `follow-up-mission-axis-G.md`. All three other follow-ups
have top-level mirrors.

## AMD 11 — sliding-window-hazard

**Status:** COMPLIED

**Evidence:** PROP K.0 guard in
`kv_direct_adapter.py` raises `SlidingWindowLayerRefusedError` on any
non-global `target_layer`. axis-G defer reference cites AMD 11
explicitly. CLAUDE.md includes operational rule.

## AMD 12 — supervisor Q-ACK authority for axis-G defer

**Status:** COMPLIED

**Evidence:** axis-G is DEFERRED per Q3 in supervisor-ack
`ve-ins-0modv5m4z00001881b9`. The follow-up reference
`ve-ins-0moe2i06j0000a88437` cites this authority.

## AMD 13 — feature-branch commit (no main commit, no merge)

**Status:** COMPLIED

**Evidence:** axis-H commit lands on `impl/kv-direct-wire-prop-k5`. ORCH
coordinates merge with supervisor at run-close. No push, no merge
performed by axis-H.

## AMD 14 — vee-record-creation as canonical artefact

**Status:** COMPLIED

**Evidence:** All four follow-up workstreams filed as
`record_type=reference` with `tag=follow-up-mission-requested`.
session-opened pattern record `ve-ins-0moe2gw0g00003e6f47` filed.
lead-report scope-complete will be filed at run-close per Step 9.

## AMD 15 — apollo-data SHA-verified mirror per Q5

**Status:** COMPLIED (within axis-E scope)

**Evidence:** `research/kv-memory-implementation/run-1/04-axis-E-apollo-data-copy/`
+ `04-axis-E-apollo-data-copy-manifest.json` (`all_match=true`). axis-H
read-only consumed; no edits to apollo data copy.

## Summary

15/15 AMDs COMPLIED. No deliberate deviations recorded. The only
ATTRIBUTION-PROTOCOL OVERRIDE (vee CLI session_id collision; Spec v9 §23
Fix A relaxation) was a **supervisor-authorised tactical relaxation for
run-1 only**, classified as vee-CLI tooling-bug remediation, and is not
itself an AMD deviation — see `ve-ins-0mody4xaa0000d8c5ab` and
`ve-ins-0mody99g200003a78ed`.
