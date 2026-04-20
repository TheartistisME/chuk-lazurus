# CUDA-BACKEND WORKFLOW MASTER SPEC (v9 — hard-rule inviolability + attribution-source clarification)

## 1-22. Unchanged from v8 and earlier (see preceding versions).

All prior conventions still active:
- §1–17: core pipeline + permissions + traceability levels + marker protocol
- §18: Override discipline
- §19: Gate-HALT semantics
- §20: Pane-limit protocol
- §21: Baseline-of-absence grammar
- §22: Auto-dependency linking
- §23: Scope-barrier protocol (v8 — unchanged core; attribution-source clarified in v9)

## 23. Scope-barrier protocol (v8 — NEW)

### Root-cause incident
Run-2 Execution Phase 1 saw scope creep: `lead-infra`'s sub-agents edited 7 files outside its declared mission scope, landing in sibling-lead territory (3 harness files, 3 api-sem files, 1 ambiguous). Core deliverable validated; scope did not. The `chuk-lazurus-paz` mission was reopened by the validator gate.

### Root cause
No formal scope manifest. Leads received a mission-id but no file-glob boundary. Sub-agents (invoked via the `Agent` tool) inherited lead context but not scope boundaries — they edited whatever seemed reasonable. The no-write rule bound the lead but NOT the sub-agents it spawned.

### Fix: scope manifest + propagation + validator check

#### Orchestrator (v7 §Scope manifest declaration)
Before spawning any lead, write a `lead-scope-manifest` reference record with:
- `--tag lead-scope-manifest` + `--tag <lead-name>` + `--tag "run-${RUN}"`
- Body = newline-separated file globs (glob syntax, `**` supported). Comments (`#`) allowed.
Verify no-overlap across sibling leads BEFORE spawning. If two manifests claim the same path, stop and re-decompose.

#### Lead (v6 §Scope manifest protocol)
At Step 0.5, read the manifest record, store globs at `.vee/pane-sessions/${TMUX_PANE//%/pct}.scope`. At every `Agent` tool call, the lead MUST include a SCOPE BINDING block verbatim in the sub-agent's prompt, listing the manifest globs and the sub-agent's obligations:
- Read/Edit/Write only within the globs.
- If cross-scope needed: stop, write `cross-lead-touch-requested` record, return without editing.
- Rule inherits recursively through further delegation.

#### Sub-agent (enforced via inherited prompt)
Sub-agents spawned via `Agent` are bound by the SCOPE BINDING block in their prompt. They check their own edit targets against the globs. Cross-scope edits trigger the stop-and-report flow.

#### Validator (v7 §Closure validator gate)
At mission closure:
- Read lead's scope-manifest record.
- Run `git diff --name-only` to list edited files.
- Match each edited file against manifest globs.
- All-match → scope PASS. Any mismatch → scope FAIL → closure rejected → mission reopened → `scope-violation` failure record.
- Closure requires BOTH core-deliverable PASS AND scope PASS.

#### Hook (optional enforcement layer)
The PostToolUse hook reads the pane's scope file at invocation time. If an Edit/Write tool targets a path not matching the glob, the hook emits an immediate `scope-violation-detected` warning record. Doesn't block the edit (blocking is too risky for a hook) but surfaces the violation in real time so the Orch's observational duty can catch it before closure.

### Terminology fix (bundled into this patch)
Prior charters called the sub-agent tool `Task()`. Claude Code's actual tool name is `Agent`. `Task`/`TaskCreate`/`TaskUpdate` are todo-tracking tools, unrelated to sub-agent delegation. LEAD v6 + ORCH v7 use `Agent` consistently. Validators also use `Agent` to spawn themselves when phrased as `Task(subagent_type=…)` in older docs — treat as synonymous.

### Mission-creation prohibition (bundled)
LEAD v6 adds an explicit hard rule: leads CANNOT create new beads missions. Any discovered follow-up work is filed as a `follow-up-mission-requested` pattern record; the Orch creates the mission under supervisor gate. (Run-2 `lead-api-sem` violated this by auto-creating `chuk-lazurus-l73.1`.)

### Enforcement layers (defense in depth)
- Layer 1 (charter): rules at HAND, ORCH, LEAD charters.
- Layer 2 (data): lead-scope-manifest records canonicalize scope.
- Layer 3 (propagation): LEAD inlines SCOPE BINDING in every sub-agent prompt.
- Layer 4 (hook): PostToolUse surfaces real-time violations.
- Layer 5 (validator): closure-gate rejects scope violations before commit-gate.

Any one layer alone is bypassable; all five together make scope-violation vanishingly unlikely.

## Active canonical references (updated)
- LEAD charter v6 (to be assigned): adds scope-barrier, Agent terminology, mission-creation prohibition
- ORCH charter v7 (to be assigned): adds manifest-declaration + validator-checks-manifest
- spec v8: this record

## 18. (v9 addendum) Pattern D — override directive conflicts with downstream hard rule

Documented root cause of Bug K-prime (run-2): an override directive issued at one layer forces the recipient to violate a hard rule in their own charter.

Example: Orch's `--then` payload to lead-api-sem contained "open a follow-up mission for l73.1." LEAD v6 charter prohibits leads from creating beads missions (hard rule). The lead, bound by Override Discipline ("apply verbatim"), created the unauthorized mission. Supervisor-gated mission creation was bypassed through a two-layer indirection.

Resolution (enforced in v6/v7/v8 charter hard-rule-inviolate rules):
- The recipient MUST refuse the conflicting portion of the override, file `/vee:create-bug&patch-Report`, and halt until the directive is revised.
- The issuer MUST self-check override directives against the recipient's charter hard rules BEFORE issuing. ORCH v8 adds Pre-spawn payload review via Task(validator) as the systematic enforcement.
- Override Discipline (Patterns A/B/C — quote-in-full, explicit-scope, patch-charter) still applies for legitimate overrides; Pattern D is the anti-pattern where the override crosses a hard-rule boundary.

Hard rules trump override directives. Always.

## 23. (v9 addendum) Fix A — Attribution source clarification

The closure validator (spec §23, ORCH v7/v8) must attribute edited files to leads using the CANONICAL EVENT LOG (hook tool-call records), NOT filesystem state.

Correct source: `.vee/records/events/0001.log` entries where `.actor.session_id == <lead_session>` AND the tool-call payload.body matches `tool=(Edit|Write|MultiEdit)`. Each such record is attributable per-session.

Wrong source: `git status --porcelain`, `git diff --name-only`, or file mtimes. In a shared worktree with concurrent leads (as in run-2 Execution Phase 1), filesystem state shows the UNION of all lead activity. Presence ≠ authorship.

Anti-pattern (Bug J-prime, run-2 Execution Phase 1): the original first-mission validator used `git status --porcelain` and flagged 7 sibling-lead files as scope creep for lead-infra. Subsequent proper diff assessment by the Orch showed all 7 files were legitimate sibling-lead work — attribution was correct, but presence suggested otherwise. Validator verdict was superseded FAIL → PASS after content-based inspection.

Rule: if you ever find yourself using the filesystem to answer "who did this?" in a multi-agent workflow, STOP. Query the event log. The filesystem tells you what's there; the event log tells you who put it there.

## Active canonical references (updated v9)
- HAND v6 (hard-rule-inviolate rule, otherwise unchanged from v5): to be assigned on record
- LEAD v7 (hard-rule-inviolate rule): to be assigned on record
- ORCH v8 (hard-rule-inviolate + pre-spawn payload review + validator uses hook records): to be assigned on record
- spec v9: this record
- Bug J-prime (presence-vs-authorship validator bug): inline in patch-process #6
- Bug K-prime (override conflicts with hard rule): inline in patch-process #6
