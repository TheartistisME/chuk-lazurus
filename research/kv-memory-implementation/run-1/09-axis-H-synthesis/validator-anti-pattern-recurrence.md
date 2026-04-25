# Validator anti-pattern recurrence — run-1 observations

This note documents two **presence-vs-authorship** validator false-positives
caught during run-1 closures, plus one related stale-ground-truth attribution
event. Both contain prescriptive validator-prompt updates for future runs.

## Anti-pattern 1 — "presence-vs-authorship" on runner-only missions

### What happened

The axis-F closure validator (first pass) REJECTED the lead-axis-F
scope-complete report on the grounds that "no test additions were
detected." axis-F mission charter (`chuk-lazurus-1g3`) was explicitly
**runner-only**: re-execute the existing
`test_kv_materialisation_parity_layers_27_to_32` parity battery on the
post-change worktree and confirm GREEN. No new test code was authored,
no production code was touched. The validator's first-pass heuristic
was looking for *file additions* under the test directory and
mis-applied that heuristic to a runner-only mission.

### Recovery

Validator was re-prompted with the framing "this mission's deliverable
is the *attestation* (jsonl + summary), not new test code". On second
pass the validator accepted, citing the manifest globs:

- `prod/validation/diagnostic_axis_F_regression_*.jsonl` (created)
- `research/kv-memory-implementation/run-1/06-axis-F-regression/**` (created)

Both presence checks PASSED.

### Recommended validator-prompt update for future runs

> When auditing a scope-complete report, distinguish between **authorship
> missions** (must add or modify code matching certain globs) and
> **runner-only missions** (must produce attestation artefacts matching
> certain globs). Read the mission's end-state pointer and mission
> manifest before applying file-presence heuristics. A runner-only
> mission that produces no code additions but produces all required
> attestation artefacts is a PASS, not a REJECT.

## Anti-pattern 2 — "stale-ground-truth" on supervisor acceptance test (b)

### What happened

When supervisor-decision Framing α (`ve-ins-0moe06p5g00002de88a`) spawned
lead-axis-runtime-fix, the seven acceptance tests included:

- (a) `test_kv_direct_synthetic_smoke_e2e_layer_29` (FAIL → PASS canary)
- (b) `test_kv_direct_materialized_real_gemma4.py::test_kv_materialisation_parity_layers_27_to_32`
  (parity battery; expected GREEN)
- (c) … (e) … axis-BC + axis-D regression tests

Test (b) FAILED on the post-fix worktree. The momentary attribution
chain assumed runtime-fix's three-site patch had broken the parity
battery. Lead-axis-runtime-fix did the right thing: stash the patch,
re-run (b) on a clean checkout against `fac1f36`'s parent. Test (b)
**still failed** — the parity gap is **pre-existing** and is rooted in
`_prepare_archived_prefix`'s RoPE-identity slicing
(`cos[:, 0:1, :].expand(...)`). Supervisor accepted the
pre-existing-parity-failure note `ve-ins-0moe1s9qz0000a1e04a` and
re-classified test (b) as out-of-scope for runtime-fix's mission.

### Recovery

Lead-axis-runtime-fix's reproducible isolation procedure is the
canonical answer: **stash + re-run against the parent of HEAD** before
attributing a regression to the new patch. Where stashing isn't
practical (e.g. infrastructure tests), the equivalent is `git
checkout HEAD~1 -- <test_file>` + `pytest <test_file>` to confirm
pre-existence.

### Recommended validator-prompt update for future runs

> When a supervisor-defined acceptance test FAILS on a post-fix
> worktree, the validator MUST run the canonical isolation procedure
> before attributing the failure to the new patch:
>
> 1. `git stash` the new patch (or `git checkout <parent_sha> -- <files>`).
> 2. Re-run the test.
> 3. If it still FAILS, the failure is **pre-existing** and the new
>    patch is not the cause. File a `pre-existing-parity-failure`
>    record with the test name and parent SHA.
> 4. Restore the patch and continue with the remaining acceptance
>    tests; do NOT mark the new patch as a regression-introducer.
>
> A validator that skips this step risks rejecting valid patches and
> creating spurious churn. The cost of the isolation procedure is one
> additional test run; the cost of mis-attribution is hours of debug
> + supervisor escalation.

## Cross-cutting recommendations

1. **Validator prompts should embed mission-type classification.** The
   "authorship vs runner-only" distinction is high-leverage. A small
   field in the mission charter (`mission_type: authorship | runner_only |
   research`) would let validators apply the right heuristic
   automatically.
2. **Pre-existing-failure records should be a first-class vee tag.** The
   `pre-existing-parity-failure` tag used at
   `ve-ins-0moe1s9qz0000a1e04a` is a good template; lifting it to the
   validator's vocabulary would close a recurring failure mode.
3. **Hybrid attribution (lead-report-declared paths PRIMARY + git diff
   SECONDARY) is durable.** Run-1's ATTRIBUTION-PROTOCOL OVERRIDE
   `ve-ins-0mody4xaa0000d8c5ab` validated this hybrid for the vee CLI
   session_id collision class. Recommend baking the hybrid into spec
   v10 §23 once the vee CLI bug is fixed (`vee-cli-patch` follow-up).

## Related run-1 records

- supervisor-decision Framing α: `ve-ins-0moe06p5g00002de88a`
- runtime-fix lead-report: `ve-ins-0moe20pup00000fedf9`
- runtime-fix scope-complete closure: `ve-ins-0moe27dru000066e528`
- pre-existing-parity-failure: `ve-ins-0moe1s9qz0000a1e04a`
- ATTRIBUTION-PROTOCOL OVERRIDE: `ve-ins-0mody4xaa0000d8c5ab`
- ORCH override-ack: `ve-ins-0mody99g200003a78ed`
