# VEE_RECORDS — curated canonical trail

These are the **most relevant** vee records for understanding this port. They are the minimum set you need to reconstruct *why* the implementation looks the way it does. Full workspace state is under `.vee/` (gitignored; local only). To look up a record by id on a fresh clone, use the handoff record title as a starting query; it transitively references the rest.

## Reading order

1. **Handoff from prior series** → understand the problem
2. **Lead-report v2** → understand the solution
3. **Capstone learning-pattern** → understand the three re-usable patterns
4. **OOM learning-pattern** → understand the Gemma-4 trap
5. **Scope-expansion authorization + manifest v2** → understand criterion-4 extension
6. **Criterion-4 supplementary + e2e verification** → understand the end-to-end proof
7. **Root session** → thread everything together

---

## 1. Supervisor handoff (prior series → this run)

**ID:** `ve-ins-0mo883vnd000053084c`
**Title:** `SUPERVISOR HANDOFF: turn-aligned series (3 runs closed; canonical-port next)`
**Type:** reference, foundational, tagged `supervisor-handoff` + `turn-aligned-{,restore,apollo}` + `handoff` + `canonical-port`

The 3-run backstory. Run A built the architecture (4/5 axes pass, axis-5 token-salad). Run B forensic-restored + committed (wrong env-drift diagnosis). Run C localised root cause to the generator's last-position-only donor injection. This record prescribed the canonical-port next run and identified `chrishayuk/chuk-lazurus`'s two-stage prefill as the reference implementation.

**Why it matters:** the handoff also lists the §4 inventory of modified / new files and §5 path-A vs path-B implementation options. Read this first.

---

## 2. Lead-report v2 (ACHIEVED)

**ID:** `ve-ins-0mo8c3prp0000ff909a`
**Title:** `lead-report (v2, SUPERSEDES ve-ins-0mo8bf7tv00006b0a6f): turn-aligned-canonical-port:canonical-prefill-port run 1 ACHIEVED`
**Type:** reference, foundational, tagged `lead-report` + `scope-complete` + `achieved` + `supersede`

The canonical run-1 outcome. All 4 acceptance criteria PASS. Zero scope violations, zero hard-rule violations, zero silent overrides. Contains the full sub-agent trail (7 spawns across 4 rounds), the implementation summary, and the cross-reference to the verification artifacts.

**Predecessor:** `ve-ins-0mo8bf7tv00006b0a6f` (v1, tombstoned — supersede reason in the tombstone record).

**Why it matters:** this is the single record to read if you want to know what was delivered.

---

## 3. Capstone learning-pattern (three re-usable patterns)

**ID:** `ve-ins-0mo8bdqfg0000f54b22`
**Title:** `learning-pattern: canonical prefill port — 3 patterns from turn-aligned-canonical-port run 1`
**Type:** reference, foundational, tagged `learning-pattern` + `gemma-4` + `canonical-mirror`

The three patterns:

**a)** MLX `prefill_to_layer(initial_residual=boundary)` actually seeds at the **embedding layer output**, not at the crystal layer. The GOAL_BODY shorthand was misleading. The MLX source (`kv_generator.py:182-226`) is authoritative.

**b)** Gemma-4 `inputs_embeds` is a deterministic OOM trap: `get_per_layer_inputs` broadcasts `inputs_embeds × embed_weight` to shape (B, S, V, H) ≈ 253 GiB on a 600-token prompt.

**c)** Canonical-equivalent PyTorch workaround: seed token + `forward_pre_hook` on `layers[0]` preserves canonical semantics on Gemma-4's optimised `input_ids` path. Shape-gate the hook so decode calls are no-ops.

**Why it matters:** this is what a future agent needs to port the same mechanism to another model family.

---

## 4. Gemma-4 OOM learning-pattern (forensic)

**ID:** `ve-ins-0mo8azci200009633ee`
**Title:** `learning-pattern: Gemma-4 inputs_embeds OOM via get_per_layer_inputs broadcast`
**Type:** reference, tactical, tagged `learning-pattern` + `gemma-4` + `inputs-embeds-oom`

The forensic detail: exact line number in `modeling_gemma4.py`, exact broadcast shape, exact allocation size. Includes the mitigation pointer (see record 3).

**Why it matters:** read this if you hit OOM on any HF-transformers model and suspect a `get_per_layer_inputs`-style trap.

---

## 5. Scope-expansion authorization + manifest v2

**Authorization ID:** `ve-ins-0mo8bkkal0000f9c8f3`
**Manifest v2 ID:** `ve-ins-0mo8bki2i000023c97e`
**Tags:** `supervisor-intervention` + `scope-expansion-authorized` + `criterion-4`

The supervisor intervention that expanded run-1's scope mid-flight so criterion 4 could be exercised end-to-end instead of being deferred to a next-run mission. Manifest v2 superseded v1 (`ve-ins-0mo89m0950000858c69`, tombstoned) with read+execute access added for `chat_loop/`, `session_close/`, `session_store/`, and verification-only write access for `scripts/criterion4_*`, `tests/criterion4_e2e/**`, `/tmp/validation-logs/**`.

**Why it matters:** this is the example of how in-run scope expansion is done correctly — authorisation + new manifest + tombstone + LEAD ack + Pattern-A override relay with token-count discipline.

---

## 6. Criterion-4 supplementary lead-report + reproducible harness

**ID:** `ve-ins-0mo8c2eli00006d9b70`
**Title:** `lead-report-supplementary: criterion 4 e2e PASS (turn-aligned-canonical-port run 1)`
**Type:** reference, foundational, tagged `lead-report` + `criterion-4` + `scope-expansion`

The end-to-end proof. Uses the existing `scripts/cross_session_demo.sh` harness to exercise chat_loop → session_close → session_store → fresh `SessionRetriever` → exact-ID probe. Verbatim hit on handle `d32252f1a1394bad9c12285b48be10c3.11.0`:

> **"the coral reef shimmered cobalt at dawn over Fujairah"**

with coherent surrounding English. Reproducible via `scripts/criterion4_e2e_verify.py --output-root /tmp/csd-criterion4`.

**Why it matters:** this is the proof that the port works for conversational memory end-to-end, not just for the narrower knowledge-store retrieval case.

---

## 7. Root session

**ID:** `ve-ses-0mo89la170000320133`
**Role:** `hand-supervisor`
**Task:** `turn-aligned-canonical-port`

The supervisor session for run-1. Every supervisor-attributed record above is parented to this session id. (Caveat: due to a vee v0.1.0 CLI drift, the LEAD's own session was attributed to a stale pane-session `ve-ses-0mo85locf0000fda906` instead of a fresh child of this root — tag-based polling recovered all reports correctly. See tooling-drift bug-report `ve-ins-0mo89qjdp00009053f1`.)

**Why it matters:** this is the single id to filter on if you want the complete supervisor-side event log for run-1.

---

## Querying these on a fresh clone

The `.vee/` workspace is gitignored — a fresh clone won't have these records locally. To recover:

1. Install `vee` in the destination (`https://github.com/chrishayuk/vee`)
2. Re-fetch the workspace state from wherever it's archived (this varies by environment)
3. Or: re-read the offline copies in `prod/canonical/` and `prod/validation/`, which contain the code + evidence the records reference. The records themselves are metadata / decision logs; the artefacts are the source of truth.

## Next-run lineage

If a run-2 happens (e.g. residual-only recall test, `build_clause_aligned_store.py` fix), the supervisor-handoff for that run will supersede the record set above with a new `SUPERVISOR HANDOFF:` entry. Use the same search pattern (`vee query "SUPERVISOR HANDOFF:" --mode lexical --limit 1 --include-body`) to find it.
