# MLP Router v2 — Hand → Orchestrator Brief

**Role**: You are the **Orchestrator** (Claude code, Opus). You report to the Hand.
**Your tools**: `vee agent spawn`, `vee agent message`, `vee agent check-in`, `vee agent kill`, `vee mission`, `vee record`, `vee query`, `vee prime`, `vee session`, plus the Task tool only if needed for quick scouting. **You do not write code directly.** You spawn one Claude Lead who spawns Claude sub-agents to do the work.
**Workspace**: `/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus`
**Hand's active session**: `ve-ses-0mo2nvjtf0000ff02d2` (role=hand). You open your own session and hand off from the Hand's session on spawn.

---

## #Task

Fix three diagnosed bugs in the v2 MLP window router, retrain, and re-evaluate against `tests/fixtures/aus3000/benchmark/epic1_hard.json`. **Success = top-1 >= 1.00 on the full hard fixture.** Pre-diagnosis is complete — no rediagnosis needed. Ship only the fixes described below.

**Must-not-break** (non-regressable):
- `src/chuk_lazarus/inference/context/knowledge/route.py` (frozen — 23/23 single_pass_gate)
- `src/chuk_lazarus/inference/context/knowledge/torch_store.py`, `torch_query.py`, `torch_runtime.py`
- Any file that imports `mlx`
- `tests/fixtures/aus3000/benchmark/epic1_v1.json` (frozen regression reference)
- `src/chuk_lazarus/models_v2/core/backend/registry.py` (torch backend already registered)

**Skip doc creation up-front.** Docs happen AFTER results:
- If top-1 >= 1.00: write success docs + skill + commit.
- If top-1 < 1.00: write next-Hand handoff documenting what was tried and measured, then stop.

---

## #Root Cause (already diagnosed — do not re-diagnose)

Full finding in vee record `ve-ins-0mo2ne2l6000068adee`. Summary:

### Bug 1 (DATA, full-run blocker) — 27,760 / 103,609 rows (26.8%) are label collisions
The store has 324 / 1203 windows (27%) sharing a `clause_title` with at least one other window. 7,128 distinct texts in `artifacts/router/aus3000_v2_ds.jsonl` are labeled with 2+ different `window_ids` each.
- 81 windows → `clause_title = "General"`
- 51 windows → `"GENERAL"`
- 12 → `"Scope"`, 10 → `"Identification"`, 8 → `"Switchboards"`, 7 → `"Application"`, 7 → `"Wiring Systems"`, etc.

Every one of the 83 augmented/paraphrased templates rendered on a "General" window produces identical text labeled 81 ways. Cross-entropy cannot converge on contradictory labels.

**Fix**: in `tools/_window_router/dataset.py` and `tools/_window_router/augmentation.py`, detect store-level clause_title duplicates and force every emitted sample for a collision-group window to **always include the clause_id as a disambiguator**. Concretely:

- Compute `collision_titles = {title | count(title) >= 2 in store.window_metadata}` inside `build_router_dataset` once.
- When `title in collision_titles and clause_id`, skip the plain `title` emit and the title-only augmentations (all 73 templates in `DEFAULT_TITLE_AUGMENTATION_TEMPLATES` that have only `{title}` as a field). Only emit templates that contain `{clause_id}` in the rendered output.
- For paraphrase templates (`DEFAULT_PARAPHRASE_TEMPLATES`), apply the same rule: skip title-only renderings for collision-title windows.
- For multi-clause pair templates, keep as-is (they already include both clause_ids).
- Excerpt emission (first N tokens via tokenizer) stays as-is — each window's excerpt is unique.

**Validation gate**: `paste <(jq -c .text artifacts/router/aus3000_v2_ds.jsonl) <(jq -c .window_id artifacts/router/aus3000_v2_ds.jsonl) | sort -u | cut -f1 | sort | uniq -d | wc -l == 0`

### Bug 2 (ENCODER, ceiling is TF-IDF) — `embedding-layer-v1` is not a real encoder
At `tools/_window_router/encoder.py:381` the encoder does `self._embeddings(input_ids)` — only the input embedding layer, no transformer forward pass. Mean-pooling raw token embeddings = bag-of-gemma-embeddings. TF-IDF (0.696) beats it (0.268 full BoW v2, 0.2 pilot).

**Fix**: switch to full forward pass + last-hidden-state mean pool.
- In `GemmaEmbedEncoder._ensure_embeddings`, keep loading `AutoModel.from_pretrained(self.model_id)` but **keep the full model**, not just its embedding layer. Store `self._model = model`.
- In `_pooled_batch_embeddings`, call `self._model(input_ids=input_ids, attention_mask=mask, output_hidden_states=False)` inside `torch.no_grad()` and use `.last_hidden_state`. Mean-pool with attention mask.
- Update `encoder_version` in `tools/_window_router/cache.py` from `embedding-layer-v1::...` to `last-hidden-state-v1::...` so old caches are invalidated by SHA256.
- **Do not** break the BoW encoder path, the clause-hierarchy channel, or the test suite.
- Keep `max_length=512`, keep fp32 pool, keep attention-mask pooling.

**Validation gate**: run an encode of `"What does clause 1.4.2 mean by accessible, in plain language?"` vs `"Explain Basic protection"` and `"Accessible"` — the first two should have cos-sim < 0.92 and the first-vs-third should have cos-sim > 0.85.

### Bug 3 (PILOT, gate is meaningless) — 1K pilot trains on only 11 anchor classes
`artifacts/router/aus3000_v2_pilot_1k.jsonl` is anchor-filtered to 11 window_ids matching the 5 pilot cases' ground truths. Cannot distinguish "encoder works on 11 classes" from "encoder works on 1202 classes".

**Fix**: replace anchor-pilot with a **stratified 5% random subsample** across all cleaned labels.
- After dataset rebuild (Bug 1 fix), write a helper that samples `max(2, round(0.05 * per_label_count))` rows from each label using `random.Random(seed=42)`.
- Rebuild the pilot feature cache (SHA256 will change due to encoder_version bump in Bug 2).
- Pilot gate: train for 5 epochs on the stratified pilot, eval on `artifacts/router/epic1_hard_pilot5.json` (the existing 5-case pilot fixture). **Pilot pass = top-1 >= 0.4, top-3 >= 0.8.** If pilot fails, stop and escalate — do not burn full-train compute.

---

## #Project Deliverables (ordered)

1. `tools/_window_router/dataset.py` + `augmentation.py` — collision-aware emission (Bug 1).
2. `tools/_window_router/encoder.py` + `cache.py` — last-hidden-state encoder + version bump (Bug 2).
3. `tools/_window_router/pilot.py` (new) — stratified pilot sampler (Bug 3).
4. Rebuilt `artifacts/router/aus3000_v2_ds.jsonl` (no collisions).
5. Rebuilt feature cache with new encoder_version.
6. Stratified pilot dataset + feature cache + trained pilot checkpoint + pilot eval report.
7. Full-dataset feature cache + trained full checkpoint + full eval report on `epic1_hard.json`.
8. All existing tests green (39/39 window_router tests + 14/14 torch training tests + frozen aus3000 regression tests 23/23).
9. Conditional: success docs + skill OR failure handoff doc.

---

## #Non-negotiables

- **No MLX edits.** No edits to frozen files listed above.
- **Lead does not write code.** Lead only synthesises specs and delegates to Claude sub-agents via the Task tool.
- **Every sub-agent writes vee records** (`vee record pattern|failure|decision|reference|guide|convention`) before marking work complete.
- **Every sub-agent runs quality gates** (tests + linter) after every change. Green or revert.
- **Run `vee prime` before spawning every sub-agent** so they get synchronised canonical context.
- **Pilot gate is mandatory** before full train. top-1 >= 0.4 or stop.
- **AUS3000 regression `single_pass_gate` must stay 23/23** at every gate.
- **Industry-standard code practices.** Type hints, docstrings, no TODOs, no stubs.
- **Every new module ships with tests.** No untested code merged.

---

## #Epics

### EPIC 1 — Bug fixes (parallel workstreams, 3 workers)

**Owner**: Claude Lead. Delegates to 3 parallel sub-agents (`code-surgeon` or `general-purpose` with Write tool), one per workstream.

**Pre-Phase**:
- Lead runs `vee --help`, `vee prime --recent-hours 4`, `vee query "MLP router root cause collision encoder"`.
- Lead reads `docs/learned_router/HANDOFF-to-next-hand.md` and `docs/learned_router/specs/00-hand-to-orchestrator-brief.md` (this file).
- Lead creates vee mission items: `vee mission create "MLP v2 bug fixes" --type epic --priority 1` and sub-missions for each WS.

**Phase 1.1** [`vee mission create` per WS, `vee record` on completion]:
- **WS-A (data hygiene)**: `tools/_window_router/dataset.py` + `augmentation.py` + `tests/tools/window_router/test_dataset.py` + `test_augmentation.py`. Collision-title detection + clause-id-prefixed emission for collision windows. Zero regressions in existing tests. Validator: re-build dataset, run the `paste | sort | uniq -d | wc -l` gate.
- **WS-B (encoder)**: `tools/_window_router/encoder.py` + `tools/_window_router/cache.py` + `tests/tools/window_router/test_encoder.py`. Full-forward `last_hidden_state` mean-pool + encoder_version bump. Preserve BoW path + clause-hierarchy channel. Tests include a mock `_model` that returns a known `last_hidden_state` tensor to verify mean-pool math.
- **WS-C (pilot sampler)**: new module `tools/_window_router/pilot.py` + `tests/tools/window_router/test_pilot.py`. Function `stratified_sample(records, *, fraction=0.05, min_per_label=2, seed=42) -> list[dict]`. Deterministic output given seed. Test: round-trip labels, verify per-label count bounds.

**Phase 1.2** (merge gate):
- Lead spawns a sub-agent `validator` (native subagent_type) to run: all window_router tests + all torch training tests + aus3000 regression `tests/inference/context/` + `tests/tools/test_evaluate_aus3000_variant.py`.
- Lead reports back to Orchestrator via `vee agent message <coord>` once green. Orchestrator verifies with its own `vee agent check-in`.

### EPIC 2 — Retrain pipeline (sequential, 1 worker at a time)

**Owner**: Claude Lead.

**Phase 2.1** — Rebuild dataset:
- Sub-agent runs: `uv run python tools/train_window_router.py build-dataset --store-path <AUS3000 store> --out-jsonl artifacts/router/aus3000_v2_ds.jsonl --benchmark-fixture tests/fixtures/aus3000/benchmark/epic1_hard.json --feature-encoder gemma-embed --model-id <gemma cache path> --device auto`.
- Validate collision gate = 0.

**Phase 2.2** — Pilot:
- Sub-agent builds stratified pilot JSONL via new `pilot.py` helper.
- Sub-agent precomputes pilot feature cache (CUDA, ~1-3 min).
- Sub-agent runs `train` with `--epochs 5`, then `eval` on `artifacts/router/epic1_hard_pilot5.json`.
- **Pilot gate**: top-1 >= 0.4 AND top-3 >= 0.8. Report to Orchestrator.

**Phase 2.3** — Full train (only if pilot passed):
- Sub-agent precomputes full feature cache (CUDA, 45-90 min).
- Sub-agent runs `train` with `--epochs 10 --hidden 256 --batch-size 32 --lr 1e-3 --device auto`.
- Saves checkpoint to `artifacts/router/aus3000_v2_final.pt`.

### EPIC 3 — Re-evaluate

**Phase 3.1**:
- Sub-agent runs `eval` against `tests/fixtures/aus3000/benchmark/epic1_hard.json`, output to `artifacts/router/aus3000_v2_final_eval/`.
- Lead summarises metrics to Orchestrator: top-1, top-3, MRR, single-clause top-1, multi-clause recall@3.
- Orchestrator decides: success path or handoff path.

### EPIC 4 — Conditional docs (only after EPIC 3)

**Success branch (top-1 >= 1.00)**:
- Sub-agents write: `docs/learned_router/60-mlp-v2-workflow.md` (<= 200 lines), update `50-reference-card.md` with v2 numbers, update `30-aus3000-results.md`, build global skill at `~/.claude/skills/train-mlp-router/` (SKILL.md + README.md + `assets/template_brief.md` + `assets/example_cli_session.md`).
- Sub-agent runs `git add` + `git commit` with detailed message. Do not push.
- Lead writes vee success record.

**Failure branch (top-1 < 1.00)**:
- Sub-agents write: `docs/learned_router/HANDOFF-to-next-hand-v3.md` with exact state, what was tried, what failed, iteration ladder for next Hand.
- Sub-agent preserves all artifacts for next Hand to inspect.
- Lead writes vee failure record with concrete diagnostic data.

---

## #Orchestration Protocols

- **Orchestrator**: spawns the Lead once via `vee agent spawn claude --name mlpv3-lead --target <tmux-session> --cwd /mnt/c/users/jehma/desktop/lazarus/chuk-lazurus --then "$LEAD_BRIEF"`. Checks in every 3-5 min via `vee agent check-in mlpv3-lead`. Messages the Lead with `vee agent message mlpv3-lead "..."` for nudges. Does NOT write code, does NOT spawn workstream sub-agents directly.
- **Lead**: uses the Task tool with `subagent_type="code-surgeon"` for edits, `subagent_type="test-writer"` for new tests, `subagent_type="validator"` for quality gates, `subagent_type="general-purpose"` for research. Lead synthesises specs and hands off a single-agent-sized chunk per delegation. Lead never edits files itself.
- **Every agent** runs `vee prime` at start, `vee record` at end, and `vee mission update` on state transitions.

## #Session Discipline (mandatory at every layer)

Every agent — Orchestrator, Lead, every sub-agent — MUST own a `vee session` for its run. Sessions make lineage queryable and fuel the handoff index.

**Open on start** (capture the `session_id` returned in JSON — you'll need it for close/handoff):
- Orchestrator: `vee session open --role orchestrator --json` immediately after reading this brief.
- Lead: `vee session open --role lead --json` as first action after spawn.
- Every sub-agent: `vee session open --role worker --json` as the first tool call in its run.

**Handoff on spawn** (preserves the lineage chain):
- Orchestrator → Lead: after spawning `mlpv3-lead` via `vee agent spawn`, run `vee session handoff --session <orchestrator_session_id> --to <lead_session_id> --scope "<scope description>"` (lead's session_id comes back in its first check-in reply, or via `vee session status` when the Lead reports back).
- Lead → sub-agent: the Lead passes its own session_id to each sub-agent in the Task prompt; sub-agent opens its session then handoffs from Lead's session to its own.

**Status checkpoints**:
- Orchestrator runs `vee session status --session <lead_session_id>` on every check-in cycle.
- Lead runs `vee session status` on itself and on each live sub-agent session_id before dispatching the next batch.

**Close on finish**:
- Sub-agent runs `vee session close --session <own> --outcome success|failure|partial` with a one-line summary immediately before returning its final message.
- Lead closes its session after all phases are green (or after escalation to Orchestrator on failure).
- Orchestrator closes its session only after reporting the final top-1 number back to the Hand.

**Records attach to sessions**: when an agent runs `vee record`, the active session is automatically associated. Do not close sessions prematurely — a closed session stops attracting records.

---

## #Lead Profile

- Model: Claude Opus (default). Can self-downgrade to Sonnet for mechanical delegation if it wants.
- Tools allowed: Task (for sub-agent spawning), Read, Grep, Glob, Bash (for vee only, never code modification), and the vee CLI.
- Tools forbidden: Write, Edit, NotebookEdit, anything that touches source.

## #Lead Rules

1. **Open** `vee session open --role lead --json` first. Record session_id for close/handoff. Accept handoff from Orchestrator.
2. Record learnings in vee before reporting completion of any phase (`vee record pattern|failure|decision`).
3. Spawn one sub-agent per well-bounded work unit. No sub-agent gets two files of unrelated scope.
4. Before every batch, run `vee prime --recent-hours 2 --max-bytes 50000` and pass the output + your Lead session_id to each sub-agent via the Task prompt.
5. On any phase gate failure: stop, record failure to vee, message Orchestrator via `vee agent message <orchestrator-pane>`.
6. Keep the Lead context clean — delegate file reads to sub-agents when possible.
7. **Close** `vee session close --session <lead> --outcome ...` only after the final phase gate passes or after escalation to Orchestrator.

## #Sub-Agent Rules

1. **Open** `vee session open --role worker --json` as your first tool call. Accept handoff from Lead's session.
2. Record learnings in vee (`vee record`) before claiming completion. Records attach to your open session.
3. Run quality gates (tests + lint) after every change. Green or revert. No partial commits.
4. Update vee mission item (`vee mission update <id> --status in_progress|completed`).
5. If a change affects frozen files, abort and escalate to Lead.
6. Every new code file ships with tests.
7. **Close** `vee session close --session <own> --outcome success|failure|partial` with a one-line summary as your penultimate action (before returning your final reply).

---

## ##Batch Protocols

- `vee prime --recent-hours 2 --max-bytes 50000` before every sub-agent batch.
- `vee query "<batch topic>"` to pull any prior-session learnings relevant to the batch.
- Sub-agents never spawn further sub-agents.
- Lead aggregates vee records from each sub-agent's session-id and rolls up a phase-level record.

---

## #Starting commands for the Orchestrator (execute in order)

```bash
cd /mnt/c/users/jehma/desktop/lazarus/chuk-lazurus

# 1. Open your own session + accept handoff from Hand
ORCH_SID=$(vee session open --role orchestrator --json | jq -r '.data.session_id')
echo "Orchestrator session: $ORCH_SID"
vee session handoff --session ve-ses-0mo2nvjtf0000ff02d2 --to "$ORCH_SID" --scope "Hand -> Orchestrator for MLP v2 rescue"

# 2. Verify context
vee query "MLP router root cause collision encoder" | head -60
cat docs/learned_router/HANDOFF-to-next-hand.md | head -80
cat docs/learned_router/specs/00-hand-to-orchestrator-brief.md   # this file

# 3. Create mission
vee mission create "MLP v2 Router Rescue: bug fix → retrain → eval" --type epic --priority 1

# 4. Spawn the Lead. Pass this file and your session_id as its primary references.
# The Lead starts in a fresh tmux pane split from your pane.
vee agent spawn claude --name mlpv3-lead \
  --cwd /mnt/c/users/jehma/desktop/lazarus/chuk-lazurus \
  --then "Your first command: vee session open --role lead --json  (capture your session_id). Then vee session handoff --from $ORCH_SID --to <your-lead-sid>. Then read docs/learned_router/specs/00-hand-to-orchestrator-brief.md in full. You are the Lead. Do not write code — delegate everything via the Task tool. Pass your Lead session_id to every sub-agent in their Task prompt so they can handoff from you. Start EPIC 1 only after briefing me on your delegation plan via: vee agent message mlpv3-orch 'EPIC1-PLAN: <your plan>'. Orchestrator session_id: $ORCH_SID."

# 5. Nudge cadence
vee agent check-in mlpv3-lead --tail 60
vee session status --session <lead-sid>   # once Lead reports its sid back

# 6. On green EPIC 3 result: report top-1/top-3/MRR back to Hand, then close your session.
vee session close --session "$ORCH_SID" --outcome <success|failure|partial>
```

---

**End of brief. Spawn now.**
