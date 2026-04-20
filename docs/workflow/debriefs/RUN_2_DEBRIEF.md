# RUN-2 DEBRIEF — Vee-native agentic workflow with CUDA backend integration

**Date:** 2026-04-20
**Duration:** ~15 hours (single session, continuous)
**Goal:** cuda-backend — torch/CUDA integration as parity reference to Apple Silicon MLX
**Outcome:** ACHIEVED + VERIFIED with live end-to-end proof on RTX 5090

---

## 0. TL;DR

We built (and proved) a 4-layer delegation pyramid — **Supervisor → Hand → Orchestrator → Lead → Native sub-agent** — with **vee** as the canonical nervous system. Over one run, the workflow:

- Self-discovered 6 distinct charter-level bugs (one was from a patch we had just landed 3 min earlier)
- Self-patched all 6 via a canonical `patch-process` decision-record pattern
- Evolved 4 distinct charter files across 30+ tombstoned versions
- Landed 7 commits on main with full NO-COMMIT gating + content-validator verification
- Produced 3,142 canonical event records across 11+ sessions with full Level-1 (session), Level-2 (capture), Level-3 (tool-call) traceability
- Validated every patch live in execution without rolling back any patch

The meta-pattern is now provable: **bug discovered during live execution → `/vee:create-bug&patch-Report` → supervisor root-cause analysis → charter patch at source → tombstone predecessor → canonical patch-process record → snapshot → continue**. We ran it 6 times in one session. It works.

---

## 1. The architecture we actually built

```
USER
  │ (gives goal)
  ▼
SUPERVISOR (ve-ses-0mo6t078v00001642a8, role=hand-supervisor)
  │ (seeds charters + pre-flight; spawns the Hand)
  │ vee agent spawn + session handoff
  ▼
HAND (run-1 + run-2 panes)
  │ (translates goal → end-state records + missions; dispatches the Orchestrator)
  │ vee agent spawn + session handoff
  ▼
ORCHESTRATOR (run-1 + run-2 panes)
  │ (drives Discovery + Execution; spawns scouts and leads; validates closures)
  │ vee agent spawn + session handoff
  ▼
LEAD × N (scouts for Discovery, workstream leads for Execution)
  │ (delegates via Agent tool; aggregates into canonical vee records)
  │ Agent tool (Claude Code native sub-agents)
  ▼
NATIVE SUB-AGENTS (Claude Code Agent-tool children, in-pane)
  │ (do the actual file reads, edits, tests, bash)
  ▼
THE REPO
```

Every arrow = a traceable handoff. Every record carries a session id. Every tool call lands in vee tagged with its pane's session. **Zero orphan records from the moment Level-1 traceability was enabled** (~06:51 UTC).

---

## 2. Vee as nervous system — what made this workflow different

Prior agentic systems treat memory as an add-on. Here, vee was the **only** shared state:

- **Charters lived as vee reference records** — never as loose markdown files in the repo (except as committable mirrors refreshed from vee)
- **Bug reports lived as `failure` records** tagged `supervisor-alert` — queryable inbox, not a free-form text channel
- **Patch decisions lived as `patch-process` records** — queryable meta-history of how the workflow evolved
- **Handoffs between layers carried token counts as `handoff-token-count` records** — measurable per-layer cost
- **Every tool call landed as a `tool-call` record** via the Claude Code PostToolUse hook — Level-3 structured audit
- **Session lineage** (via `vee session handoff`) linked supervisor → hand → orch → leads in one unbroken tree

When something went wrong, we didn't have to guess what happened. We queried vee.

---

## 3. Layer-by-layer — what each layer actually did well and badly

### HAND (role=hand, parent=supervisor)

**Did well:**
- Clean single-responsibility: seed end-state + status + missions, then dispatch
- Filed its own bug-and-patch when it caught the `tier1` classification bug in its own charter
- Propagated supervisor overrides verbatim (Override Discipline)
- Halted cleanly at every gate; never tried to self-advance

**Structural tension:**
- The Hand's only job is "needs"; it has no visibility into "current state." When a bug emerged at the Orch layer, the Hand correctly stayed out of it. Good boundary discipline.

**Honest gap:**
- v1–v2 charters had `tier1` classification bug; Hand worked around it in-the-moment but couldn't self-fix per Override Discipline. The fix required supervisor action. That's the right boundary, but it means **a Hand cannot recover from a broken charter step if the issuer is a sleeping supervisor**. Future runs should have an async escalation path.

### ORCHESTRATOR (role=orchestrator, parent=hand)

**Did well:**
- Owned the Discovery → Execution pipeline cleanly
- Self-corrected Bug G-prime (gate-HALT misread) within the same run; filed proper bug-and-patch instead of just fixing it ad hoc
- Pre-spawn payload review caught a phantom typo on first live use; Orch correctly re-verified and dismissed (calibration worked)
- Never wrote code itself (charter compliance)
- Auto-dep-linking rule (v6) worked flawlessly — zero round-trips needed for dep management

**Structural tension:**
- The Orch lives longest of any layer and accumulates the most context; it's the most complex role. Its charter grew from ~1400 tokens (v1) to 1744 tokens (v8) reflecting real new responsibilities. Healthy growth, not scope creep.

**Honest gap:**
- Orch's first-pass validator used `git status --porcelain` for attribution (Bug J-prime). In shared-worktree multi-lead setup this produced false-positive scope creep verdicts. The fix (hook-records-as-attribution-source) is now in charter v8. But this is a class of bug ("treat filesystem as authorship") that will recur in future runs unless we document it as a canonical anti-pattern — done in spec v9 §23.

### LEAD × N (role=lead, parent=orchestrator)

**Did well:**
- 6 leads spawned across the run (3 scouts in Discovery + 3 workstream leads in Execution); each opened its own session, wrote baseline + learning + lead-report records, closed cleanly
- Scope-manifest inheritance (v7/v8) worked — Tier-3 leads propagated SCOPE BINDING verbatim into sub-agent prompts
- Belt-and-braces markers protected attribution after Bug F

**Structural tension:**
- Leads are the layer that most benefits from the `vee agent spawn` prohibition — they delegate via Agent (sub-agents) only. Forces them into a pure "read-plan-coordinate-record" shape that scales cleanly.

**Honest gaps:**
- v5 charter said "use Task()" when Claude's actual tool is `Agent` — terminology bug caused at least one lead (lead-harness run-2) to have 0 Agent spawns and delegate poorly. Fixed in v6.
- One lead (lead-api-sem run-2) created an unauthorized follow-up mission (`chuk-lazurus-l73.1`) because the Orch's `--then` payload instructed it to. Bug K-prime: override directive forcing a hard-rule violation. Fixed via HARD-RULE-INVIOLATE (v9) + Pre-spawn payload review (v8).

### NATIVE SUB-AGENTS (Claude Code Agent-tool children)

**Did well:**
- Actually wrote all the code, built the harnesses, populated the baselines
- Every tool call captured in vee as `tool-call` record tagged with the parent lead's session
- No direct code attribution issues (they inherit the lead's context)

**Honest gap:**
- Sub-agents are invisible at the vee level except through their aggregate tool-call stream. For audit purposes, "this lead's sub-agents did X" is the granularity we have. That's by design (Option C from early in the run) but it means a misbehaving sub-agent can only be caught at the lead's closure validator, not in real time. Future work: hook-level scope-violation warnings (already added in v8 hook upgrade).

---

## 4. The 6 patch-process cycles — in chronological order

Each row: what surfaced, what patched, what survived.

| # | Trigger | Root cause | Patch target | Survived? |
|---|---|---|---|---|
| **v4** | **Bug F**: Hand's override message dropped `${TMUX_PANE//%/pct}` filename specificity; Orch wrote marker as pane-name, 114 tool-calls lost attribution | Partial-override semantic drift | HAND v3→v4, ORCH v2→v3, LEAD stub→v3, spec v2→v3: OVERRIDE DISCIPLINE + belt-and-braces marker | ✅ |
| **v5** | **Bug G-prime**: Orch misread "HALT before D4" as "stop all activity"; sat idle 18 min while scouts completed; missed real-time signals | Gate-halt semantics ambiguous | ORCH v3→v4, spec v3→v4: GATE-HALT SEMANTICS (phase-advance-freeze ≠ observation-freeze) + PANE-LIMIT PROTOCOL | ✅ |
| **v6** | **Bug H-prime**: 3/5 baseline records had UNSUPPORTED verdicts because scouts fabricated properties for non-existent infrastructure | Baseline grammar too loose | LEAD v4→v5, spec v4→v5: GRAMMAR OF ABSENCE + literal 4-section body template | ✅ |
| **v7** | **Efficiency**: Orch asked supervisor whether to auto-create beads dep-links at D8 gate. User: "just do it" | Overhead-for-no-safety-gain | ORCH v4→v5, spec v5→v6: AUTO-DEPENDENCY LINKING (unilateral + report-after) | ✅ |
| **v8** | **Bug I-prime** (later found to be **false positive**): Orch validator flagged 7 files as scope creep; on diff inspection, 5 were legitimate sibling-lead work | No formal scope manifest + validator used presence not authorship | LEAD v5→v6, ORCH v5→v6, spec v6→v7: SCOPE-BARRIER PROTOCOL (5-layer defense: charter + data + propagation + hook + validator) | ⚠️ core concept correct; validator attribution layer fixed in v9 |
| **v9** | **Bug J-prime**: v8 validator used `git diff --name-only` (same presence-vs-authorship pattern) → would reproduce false positive; **Bug K-prime**: override directive forced lead into charter violation (mission-create) | Attribution source wrong (filesystem instead of event log) + no override-vs-hard-rule precedence | HAND v5→v6, LEAD v6→v7, ORCH v6→v7, spec v7→v9: HARD-RULE-INVIOLATE + Pre-spawn payload review + hook-attribution validator | ✅ |

**Notable:** v9 patched a flaw in v8 that had landed 3 minutes before. That's the property we designed for — the workflow notices and patches its own flaws through live execution.

---

## 5. Metrics that tell the story

### Canonical event log
- **Total events**: 3,142
- **`record.upsert` events**: ~850
- **`session.open` events**: 11 (1 supervisor + 2 hands + 2 orchs + 6 leads)
- **`session.close` events**: properly paired; no dangling sessions
- **`session.handoff` events**: 6 (one per parent→child dispatch)
- **`record.tombstone` events**: 31 (superseded charters + obsolete run-1 artifacts)

### Missions
- **8 missions closed** on `cuda-backend`: paz / rfm / 0pu / e33 / l73 / 3gd / 8wm / nso
- **1 follow-up mission created** under supervisor gate: chuk-lazurus-qde (tier-4, mlx-cross-verify)
- **14 total closed missions** across the run (includes run-1 teardowns)

### Code changes landed on main
- **7 commits** — orchestrated in two phases: Execution Phase 1 (harness + ci + backend) + Tier-3 close (parity + perf baselines + ci plumbing + residuals golden)
- **Every commit** gated by `commit-approved` pattern record + pre-commit content validator

### Charter evolution
- **HAND**: v1 → v6 (5 versions tombstoned)
- **ORCHESTRATOR**: v1 → v8 (7 versions tombstoned)
- **LEAD**: stub → v7 (6 versions tombstoned)
- **workflow-spec**: v1 → v9 (8 versions tombstoned)
- **Total tombstoned charter records**: 26
- **Active charter records**: 4

### Handoff token budget (representative)
- Supervisor → Hand (run-2 start): 892 tokens
- Hand → Orch (Execution Phase 2 dispatch): ~2,900 tokens per lead
- Orch → Lead (scout spawn): ~2,800–2,900 tokens each
- Total handoff token cost for run-2: ~30,000 tokens across inter-layer messages

---

## 6. What worked — the things that made this workflow succeed

1. **Vee as single source of truth.** Never once did we ask "what state is the system in?" and not get a definitive answer from a vee query.
2. **Patch-process records as meta-history.** Querying `vee query "patch-process"` reconstructs the full evolution of the workflow.
3. **Token-count every handoff.** Made the invisible cost of delegation visible and budgetable.
4. **Override Discipline + HARD-RULE-INVIOLATE as paired rules.** The first ensures overrides propagate faithfully; the second ensures they can't force charter violations. Together, they resolve the tension.
5. **Grammar-of-absence (v6).** Forcing EXISTS / ABSENT / DIVERGENCES / OPEN-QUESTIONS sections eliminated a whole class of baseline-fabrication bugs. Structural, not advisory.
6. **Hook-attribution (v8) replacing filesystem-attribution (v7).** The single most important correction in the run. Filesystem shows what's there; event log shows who put it there.
7. **Strict-mode verification at critical moments.** The final AUS3000-on-Gemma proof failed HARD if any silent fallback triggered (residual-compatibility / hook-fired / GPU-memory-growth). No green-illusion possible.

---

## 7. Honest gaps that remain

1. **Hand has no async escalation.** If a supervisor is slow to respond, the Hand just halts. For long-running goals this is a throughput issue.
2. **Sub-agents only surface at aggregate**. We don't have per-sub-agent sessions. Trade-off: simplicity vs granularity. Acceptable for now, worth revisiting if sub-agents ever need their own mission state.
3. **Beads + vee are separate systems.** Missions live in beads; records live in vee. They're linked by label conventions (`cuda-backend,run-2`) and explicit id references. A tighter integration would be cleaner but adds complexity.
4. **Vee index semantic layer lags.** Ollama all-minilm context overflow on large records kept the index wedged for parts of the run (Bug E). Canonical writes always succeeded; lexical/hybrid queries unaffected; semantic queries degraded. Fix is in the vee adapter layer, out of scope for charters.
5. **Pane-limit is a hard 4.** Scaling beyond 4 concurrent leads requires pane relocation (Pane-limit protocol) or multi-window orchestration. Not a problem for this goal, but a real constraint for broader goals.
6. **Charter v10 open items**: explicit git-ops permissions (`commit`/`tag`/`log` allowed, file-modifying git blocked); spec v10 §24 artifact-class policy (golden / ephemeral / audit).

---

## 8. Is this meta-pattern reusable?

**Yes — for any goal that fits this shape:**
- Goal has measurable acceptance criteria (not pure R&D)
- Goal can be decomposed into 3–8 axes or workstreams
- The execution environment has tools that can be delegated (Claude Code, shell, git, tests)
- The goal benefits from structured audit trail (compliance, reproducibility, debugging)

**Transplant recipe:**
1. Copy `.claude/hooks/posttooluse-vee-log.sh` + `.claude/settings.json` to new repo
2. Copy `prompts/hand-v6.md` + `orchestrator-v8.md` + `lead-v7.md` + `workflow-spec-v9.md`
3. Copy `tools/count_tokens.py`
4. `vee init --workspace ./`
5. `vee session open --role hand-supervisor --task <goal-meta>`
6. Seed the charters as canonical vee records
7. Set `GOAL_TAG / GOAL_TITLE / GOAL_BODY` and spawn the Hand

The meta-workflow is goal-agnostic; only those three env vars change.

**Less good for:**
- Single-command tasks (overhead dominates)
- Pure research with no measurable exit condition
- Workloads where orchestration latency matters more than correctness

---

## 9. Recommendations for future runs

1. **Start with v6 HAND / v8 ORCH / v7 LEAD / v9 spec.** Don't re-discover the bugs we already patched.
2. **Pre-build goal-specific models + stores.** The model download + store build is the slow part; have artifacts ready.
3. **Define `golden` vs `ephemeral` artifact classes upfront** (spec v10 §24 deferred). Decide which files are committable snapshots vs gitignored runtime data before Execution starts.
4. **Consider a v10 patch** bundling the two deferred items: explicit git-ops permissions for Orch + artifact-class policy.
5. **Run a post-mortem canonical record per run** (like this debrief) tagged `run-debrief`. Turns every run into training data for future workflow evolution.
6. **Keep the meta-workflow itself under the same versioning discipline.** Spec v10 is the next increment; don't skip the patch-process record when you add v10.

---

## 10. One-sentence takeaway

**We built a self-correcting agentic workflow where every flaw surfaced during live execution becomes a canonical patch-process record that permanently improves the workflow — and we proved it six times in one session, ending with a live RTX 5090 strict-mode verification that left no room for silent-fallback illusion.**

RUN-2 CLOSED GREEN. chuk-lazurus-qde carried forward. Meta-workflow validated end-to-end.

---

*Canonical debrief record: (to be assigned on commit)*
*Session lineage: supervisor ve-ses-0mo6t078v00001642a8*
*Patch-process records for meta-history: `vee query "patch-process" --mode lexical --limit 20`*
