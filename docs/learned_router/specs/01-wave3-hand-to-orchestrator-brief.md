# MLP Router v2 — Wave 3 Hand → Orchestrator Brief

**Wave**: 3 (after wave 2 failure on quantized-Gemma encoder load)
**Hand session**: `ve-ses-0mo37hi6c000079ddaf` (role=hand, open)
**Mission**: `chuk-lazurus-4tw`
**Wave-2 handoff**: `docs/learned_router/HANDOFF-to-next-hand-v3.md` (READ BEFORE STARTING)
**Encoder pre-verified by Hand**: `ve-ins-0mo37ljfe0000ab254f`
**Wave-2 Hand protocol gap record**: `ve-ins-0mo2tak5k000070d549`

---

## #Task

Swap the encoder from the broken quantized Gemma-4-E2B-it to `sentence-transformers/all-MiniLM-L6-v2` (pre-verified by Hand), retrain, re-evaluate. **Primary success gate: top-1 = 1.000 on `tests/fixtures/aus3000/benchmark/epic1_hard.json` (56 cases).**

**Wave-2 preserved state** (do NOT redo): EPIC 1 WS-A/B/C code changes + 72/72 tests + 23/23 regression gate + dataset hygiene (74,721 rows, collision=0). Verify preservation in Phase 0.0 then build on it.

---

## #Root Cause From Wave 2 (DO NOT rediagnose)

Gemma-4-E2B-it has quantized multimodal safetensors. `AutoModel.from_pretrained` silently random-initialised every transformer layer. `last_hidden_state` was noise. MLP trained on noise → top-1 0.000 vs TF-IDF 0.786. The cos-sim sanity gate defined in wave-2 brief was not treated as a mandatory numbered phase; that's the protocol gap wave 3 closes.

---

## #Hand's Pre-Verified Encoder Choice

Encoder: **`sentence-transformers/all-MiniLM-L6-v2`**
- Loadable via plain `transformers.AutoModel` (sentence-transformers package NOT required).
- Hidden size: **384**. Max length: 512.
- First load ~27s (download + init). Needs network on first invocation.
- Hand tested load report: only `embeddings.position_ids` UNEXPECTED (benign buffer). **Zero MISSING on attention/FFN layers.**
- Hand tested cos-sim ranking: correct relative ordering on 5 reference prompts. See `ve-ins-0mo37ljfe0000ab254f`.

**Integration**: extend existing `tools/_window_router/encoder.py` with a generic sentence-transformer encoder class (or generalize `GemmaEmbedEncoder` to a `HFSentenceEncoder` that runs full forward pass, masked mean-pool, optional L2 normalize). The 26-dim clause-hierarchy channel MUST remain concatenated AFTER the 384-dim sentence vector. Final feature dim = 384 + clause_feature_dim.

**Encoder version**: bump `encoder_version` in `tools/_window_router/cache.py` to `sentence-transformer-v1::<model_id>` so SHA256 cache key invalidates.

---

## #Mandatory Gates (enforced as numbered phases, not bullets)

These gates exist because wave 2 skipped them. Each gate has an owner sub-agent, a pass/stop line, and a kill-switch if violated.

### Gate 0.0 — Preservation check
Owner: `validator` sub-agent.
Run: `pytest tests/tools/window_router tests/training/torch tests/models_v2 tests/inference/context/knowledge/test_route.py tests/tools/test_evaluate_aus3000_variant.py` + `ls artifacts/router/aus3000_v2_ds.jsonl` + collision gate `paste <(jq -c .text artifacts/router/aus3000_v2_ds.jsonl) <(jq -c .window_id artifacts/router/aus3000_v2_ds.jsonl) | sort -u | cut -f1 | sort | uniq -d | wc -l` must be `0`.
**Pass**: all green + collision=0. **Fail**: escalate to Orchestrator.

### Gate 0.1 — Encoder load report
Owner: `code-surgeon` sub-agent (after landing encoder code change) → `validator` sub-agent.
After landing the new encoder code, run a precompute on a 200-sample subset with logs captured.
- `grep -c "MISSING" <log>` must be `<= 3` (benign buffer entries only).
- `grep -c "UNEXPECTED" <log>` must be `<= 3`.
- If `MISSING` count is in the 10s or 100s on transformer layers, this is wave-2's failure mode. **STOP IMMEDIATELY.**

### Gate 0.2 — Cos-sim ranking sanity
Owner: `validator` sub-agent. Script lives at `tools/_window_router/_encoder_sanity.py` (new file).
Encode these 5 prompts and verify RELATIVE ORDERING (not absolute thresholds):
- `Q1 = "What does clause 1.4.2 mean by accessible, in plain language?"`
- `T1 = "Accessible"`
- `T2 = "Basic protection"`
- `Q2 = "What mix of training, qualifications, knowledge, and experience makes someone a competent person under clause 1.4.34?"`
- `T3 = "Competent person"`

Pass conditions:
- `cos(Q1, T1) > cos(Q1, T2)` (accessible query ranks Accessible above Basic protection)
- `cos(Q2, T3) > cos(Q2, T2)` (competent query ranks Competent above Basic protection)
- `cos(Q1, T1) > cos(Q2, T1)` (accessible query is closer to Accessible than competent query is)
- `cos(Q2, T3) > cos(Q1, T3)` (competent query is closer to Competent than accessible query is)

**If any pair fails, STOP.** This gate takes <5 seconds and catches silent-random-init.

### Gate 0.3 — Fast-completion pause rule
Owner: Orchestrator (monitoring sub-agent reports).
If any phase completes in **< 30%** of its brief-stated budget:
- STOP dispatching downstream phases.
- Record why it was fast via `vee record`.
- Have validator sub-agent sanity-check artifacts (e.g. file non-empty, expected row counts, correct dimensions).
- Only resume after validator confirms no silent-failure. Wave-2 precompute finished in ~4 min vs 45-90 min budget — that was the smoke signal ignored.

### Budget expectations for wave 3
- Feature cache encode of 74,721 rows with MiniLM (384-dim, ~15ms/batch@32 on CUDA): **~35 seconds**. If <10s or >120s, pause.
- Training 10 epochs, MLP 384+26 → 256 → 1203, batch 32, lr 1e-3: **~5-10 minutes**.
- Eval on 56 cases: **<30 seconds**.

### Pilot-gate override rule (wave 2 violation fix)
A pilot-gate fail can only be overridden if:
1. Gate 0.0/0.1/0.2 all green, AND
2. The override decision states a FALSIFIABLE claim about what full-train will produce differently, AND
3. The Orchestrator (not the Lead) issues the override after confirming (1) and (2).

---

## #Project Deliverables (ordered)

1. New/generalized encoder class in `tools/_window_router/encoder.py` + tests in `tests/tools/window_router/test_encoder.py`.
2. `tools/_window_router/_encoder_sanity.py` + tests in `tests/tools/window_router/test_encoder_sanity.py`.
3. `encoder_version` bumped to `sentence-transformer-v1::<model_id>` in `cache.py`. Existing tests updated.
4. New feature cache built with MiniLM.
5. Trained `aus3000_v3_final.pt` checkpoint (preserve wave-2 `aus3000_v2_final.pt` — do NOT overwrite).
6. `artifacts/router/aus3000_v3_final_eval/report.{json,md}`.
7. Conditional: success docs OR wave-4 handoff.

---

## #Epics

### EPIC 1 — Encoder swap (single workstream)

Owner: Claude Lead. Delegates to one Claude sub-agent for code changes, then a validator sub-agent for Gate 0.1 and Gate 0.2.

**Phase 1.1**: `code-surgeon` extends `encoder.py` with `HFSentenceEncoder` (or renames/generalizes `GemmaEmbedEncoder`). Preserves BoW path + clause-hierarchy channel. Updates `cache.py::encoder_version`. Updates `train_window_router.py` to accept `--encoder sentence-transformer --model-id <id>` as a new option. Writes tests.

**Phase 1.2**: `validator` runs all window_router tests + new encoder tests. Green or revert.

**Phase 1.3**: `validator` runs Gate 0.1 (load report check on 200-sample precompute). Pass or STOP.

**Phase 1.4**: `validator` runs Gate 0.2 (cos-sim ranking sanity). Pass or STOP.

### EPIC 2 — Full precompute + train

Owner: Claude Lead. One sub-agent, sequential.

**Phase 2.1**: precompute feature cache for full 74,721-row dataset. Log to `artifacts/router/logs/precompute_wave3.log`. Time the run. **If <10s or >120s, invoke Gate 0.3 (pause + verify)**.

**Phase 2.2** (SKIP PILOT — per handoff-v3 step 4; pilot gate was the wave-2 mistake that enabled the false-negative override): go straight to full train on the cleaned 74,721-row dataset. Log to `train_wave3.log`. Save to `artifacts/router/aus3000_v3_final.pt`.

**Phase 2.3**: eval against `tests/fixtures/aus3000/benchmark/epic1_hard.json`. Output to `artifacts/router/aus3000_v3_final_eval/`.

### EPIC 3 — Result analysis + conditional path

Owner: Claude Lead.

**If top-1 >= 1.000**:
- Sub-agent writes `docs/learned_router/70-wave3-success-report.md` (numbers, diff from TF-IDF, what changed from wave 2, compute cost).
- Sub-agent updates `50-reference-card.md` with wave-3 final numbers.
- Sub-agent updates `30-aus3000-results.md`.
- Sub-agent builds `~/.claude/skills/train-mlp-router/` global skill (SKILL.md + README.md + assets).
- `git add` + `git commit` with detailed message. Do not push.

**If top-1 in [0.95, 1.000)**: produce `docs/learned_router/70-wave3-gap-analysis.md` identifying exact miss cases by category (paraphrase / multi-clause / explicit-clause-id / semantic). Recommend wave-4 direction.

**If top-1 < 0.95**: produce `docs/learned_router/HANDOFF-to-next-hand-v4.md` with full iteration ladder. Preserve all artifacts.

---

## #Non-negotiables

- **No MLX edits.** No edits to `route.py`, `torch_store.py`, `torch_query.py`, `torch_runtime.py`, or `epic1_v1.json`.
- **Lead does not write code.** All edits via Claude sub-agents spawned through the Task tool.
- **Every agent opens + closes its own `vee session`.** Handoff chain: Hand → Orchestrator → Lead → each sub-agent.
- **Every agent records at least one `vee record`** (pattern|failure|decision|reference|guide|convention) before claiming completion.
- **Gate 0.0 → 0.1 → 0.2 must pass before Phase 2.1.** No exceptions.
- **Fast-completion pause rule** is enforced by the Orchestrator.
- **AUS3000 regression gate (23/23 single_pass_gate)** held at every phase gate.
- `pre-commit` / `ruff` / `pytest` clean before any git commit.

---

## #Orchestration Protocols

- **Orchestrator**: spawns ONE Claude Lead via `vee agent spawn claude --name mlpv3-lead --cwd /mnt/c/users/jehma/desktop/lazarus/chuk-lazurus --then "<lead brief>"`. Messages + check-ins only. Does not write code.
- **Lead**: uses Task tool with `subagent_type="code-surgeon"` for edits, `subagent_type="test-writer"` for new tests, `subagent_type="validator"` for quality gates, `subagent_type="general-purpose"` for research. Passes its own vee session_id to every sub-agent in the Task prompt.
- **Sub-agents**: never spawn further sub-agents. Must `vee session open --role worker --json`, complete work, `vee record ...`, then `vee session close --session <own> --outcome ... --summary ...`.
- **Every agent** runs `vee prime` at start and `vee record` at end.

---

## #Session Discipline (mandatory)

- `vee session open --role <hand|orchestrator|lead|worker> --json` → capture `session_id`.
- `vee session handoff --session <from-sid> --to <to-sid> --scope "<text>"` on every spawn.
- `vee session status --session <sid>` on every check-in cycle.
- `vee session close --session <sid> --outcome <success|failure|partial> --summary "<one-line>"` before returning final reply.

---

## #Lead Rules

1. First action: `vee session open --role lead --json`, record session_id, accept handoff from Orchestrator.
2. Second action: read `docs/learned_router/specs/01-wave3-hand-to-orchestrator-brief.md` in full + `HANDOFF-to-next-hand-v3.md` full iteration ladder.
3. `vee prime --recent-hours 12 --max-bytes 60000` (wave 2 records are recent).
4. Delegate via Task tool. No code writing by the Lead.
5. Enforce every numbered gate before dispatching the next phase.
6. Record phase-level learning via `vee record` after each gate passes.
7. Report results to Orchestrator pane via `vee agent message mlpv3-orch "..."`.
8. Close Lead session only after EPIC 3 completes (success or handoff written).

---

## #Sub-Agent Rules

1. `vee session open --role worker --json` as your very first tool call.
2. Accept handoff from Lead's session via `vee session handoff --session <lead-sid> --to <own-sid> --scope "<task>"`.
3. Never write code for more than one workstream in a single sub-agent run.
4. Quality gate after every change (tests + ruff). Green or revert.
5. Frozen-file edit triggers abort + escalate.
6. Every new code file ships with at least one test.
7. `vee record` before close. `vee session close --session <own> --outcome ... --summary "..."` last.

---

## #Orchestrator Starting Commands

```bash
cd /mnt/c/users/jehma/desktop/lazarus/chuk-lazurus

# 1. Open session + handoff from Hand
ORCH_SID=$(vee session open --role orchestrator --json | jq -r '.data.session_id')
echo "ORCH_SID=$ORCH_SID"
vee session handoff --session ve-ses-0mo37hi6c000079ddaf --to "$ORCH_SID" --scope "Hand -> Orchestrator for MLP v2 wave 3"

# 2. Prime + read
vee prime --recent-hours 12 --max-bytes 60000
cat docs/learned_router/HANDOFF-to-next-hand-v3.md
cat docs/learned_router/specs/01-wave3-hand-to-orchestrator-brief.md

# 3. Spawn Lead
vee agent spawn claude --name mlpv3-lead \
  --cwd /mnt/c/users/jehma/desktop/lazarus/chuk-lazurus \
  --then "Your first 3 actions: (1) vee session open --role lead --json; (2) vee session handoff --session $ORCH_SID --to <your-sid> --scope 'Orchestrator -> Lead for wave 3'; (3) Read docs/learned_router/specs/01-wave3-hand-to-orchestrator-brief.md in full. Then read HANDOFF-to-next-hand-v3.md iteration ladder. You are the Lead. Do not write code — delegate via Task tool. Pass your Lead session_id to every sub-agent. Enforce every numbered gate. Start with Gate 0.0 (preservation check) via validator sub-agent, then EPIC 1 Phase 1.1. Report gate results back to me via: vee agent message mlpv3-orch 'GATE-X.X: ...'. Orchestrator session_id: $ORCH_SID."

# 4. Check-in cadence
vee agent check-in mlpv3-lead --tail 80

# 5. On EPIC 3 final result: message Hand's pane + close session.
vee session close --session "$ORCH_SID" --outcome <success|failure|partial> --summary "<top-1 number + branch taken>"
```

---

**End of brief. Encoder pre-verified. Protocol patches numbered. Spawn when ready.**
