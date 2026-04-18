# MLP Router v2 — Handoff to Next Hand

**Written**: 2026-04-17 (end of a long session, context at 90%+)
**Target reader**: the next Hand chat + Lead it spawns
**Mission**: finish v2 — get a learned MLP window router matching v1's hardcoded routing quality on AUS3000

---

## Read this in 5 min and you have the full context

### The one-line mission

**Train a learned MLP that hits 100% top-1 on `tests/fixtures/aus3000/benchmark/epic1_hard.json`, matching the reliability of v1's hardcoded `route.py` exact-match routing on the easier fixture.**

v1 (hardcoded route.py) does 23/23 on the easy benchmark and 579/579 on the 30-min soak. v2 (learned MLP) must reach parity on a HARDER fixture (paraphrases + multi-clause + edge cases).

### Where we are right now

| metric | value | target |
|---|---|---|
| v1 BoW (easy fixture `epic1_v1.json`) | 0.765 top-1 | was acceptable for v1 |
| v2 BoW on hard fixture | **0.268 top-1** | embarrassing — worse than TF-IDF 0.696 |
| v2 gemma-embed pilot (5 cases) | **0.2 top-1 / 0.8 top-3** | pilot gate 0.8 top-1 — **FAILED** |
| v2 gemma-embed full run | not attempted | 1.0 top-1 on `epic1_hard.json` |

### Why the pilot failed (the KEY insight)

**Top-3 at 0.8 with top-1 at 0.2 means the encoder is giving the right semantic neighborhood but the MLP can't rank within it.**

The smoking gun: **window 160 (clause 1.5.6.1 "Basic protection") appeared in 3 of 5 miss predictions** — it's acting as an attractor. The model predicts it even for unrelated queries.

Three causes to investigate **in order** (record: `ve-ins-0mo2ms5f20000e92c23`):
1. **Class imbalance** in the 103,609-row augmented dataset — cheap diagnostic
2. **Template content bleed** — augmentation templates generate text similar to window 160's content
3. **Multi-clause cross-references** — the Deliverable 6 multi-clause samples may reference window 160 too often

### Exact first action for the next Lead

**Before any code change, before any retrain, run this:**

```bash
cd /mnt/c/users/jehma/desktop/lazarus/chuk-lazurus
jq -r '.label' artifacts/router/aus3000_v2_ds.jsonl | sort | uniq -c | sort -rn | head -20
```

- If window 160 (label `160`) is in the top-3 heaviest: **class imbalance confirmed**. Fix = per-label sample cap (cap max 200/label), rebuild dataset, re-pilot.
- If distribution is uniform: **template content bleed**. Fix = rewrite augmentation templates for clause-specific phrasing.
- If neither: escalate to Hand.

**DO NOT add clause-ID feature channel until class balance is verified.** Adding feature channels to an imbalanced dataset makes the model learn the bias harder.

---

## What we've done (building up to now)

### 1. AUS3000 routing fix (v1, shipped on main)

- Commit `30e5f04` on `main` of chuk-lazarus
- `route.py` two-part fix: count-based gate for single-token titles + containment suppression in `_collect_pattern_matches`
- Result: `single_pass_gate` went 20/23 → **23/23**, `soak_gate` **579/579** across 25 full cycles
- Reference card: `docs/aus3000_accuracy_program/04-reference-card.md`

### 2. Generic torch-native training infrastructure (shipped, uncommitted)

All under `src/chuk_lazarus/`:
- `models_v2/core/backend/torch_backend.py` (completed)
- `models_v2/core/backend/registry.py` (torch registered)
- `models_v2/models/classifiers/torch_linear.py`
- `models_v2/models/classifiers/torch_mlp.py`
- `models_v2/models/classifiers/torch_token_embedding.py`
- `training/torch/torch_base_trainer.py`
- `training/torch/torch_classification_trainer.py`

Plus matching tests under `tests/` — all green (14/14 torch training, 34/34 window router).

**MLX path preserved** — every file is a torch-native parity file sitting alongside an untouched MLX twin.

### 3. Generic window-router CLI (shipped)

`tools/train_window_router.py` with three subcommands:
- `build-dataset` — consumes a `TorchKnowledgeStore`, writes training JSONL
- `train` — trains an MLP classifier, writes checkpoint `.pt`
- `eval` — runs checkpoint against a benchmark fixture, produces `report.md` + `report.json`

Modular helpers under `tools/_window_router/`:
- `dataset.py` — dataset builder, augmentation wiring
- `encoder.py` — `BowCharacterEncoder` + `GemmaEmbedEncoder`
- `augmentation.py` — paraphrase templates + explicit multi-clause samples
- `cache.py` — feature cache (SHA256 keyed on `encoder_type | encoder_version | dataset_hash`)
- `eval.py` — single-clause top-k and multi-clause recall scoring

Tests under `tests/tools/window_router/` — 39/39 pass after v2 work.

### 4. v1 learned router (shipped)

- Trained artifact: `artifacts/router/aus3000_bow.pt` (19.8 MB)
- Training dataset: `artifacts/router/aus3000_ds.jsonl` (458 KB, 7218 samples)
- Eval report: `docs/learned_router/eval/aus3000_eval.md`
- Numbers on `epic1_v1.json`: **top-1 0.765, top-3 0.824, MRR 0.794**
- Acceptable for v1 but this is the fixture where the misses clustered on umbrella-clause patterns

### 5. Epic-2 docs (5 files, shipped, ≤150 lines each)

Under `docs/learned_router/`:
- `10-architecture.md`
- `20-training-guide.md`
- `30-aus3000-results.md`
- `40-extending-to-new-stores.md`
- `50-reference-card.md`

### 6. v2 infrastructure (shipped, pilot failed)

Dataset: `artifacts/router/aus3000_v2_ds.jsonl` (8.85 MB, **103,609 rows**)
Harder fixture: `tests/fixtures/aus3000/benchmark/epic1_hard.json`
Pilot cache: `artifacts/router/aus3000_v2_pilot_1k.gemma-embed.13e18c16ec26d996.features.pt`
Pilot eval: `artifacts/router/aus3000_v2_pilot_eval/report.json`
TF-IDF baseline on hard fixture: top-1 **0.696** (stored in `artifacts/router/_baseline_hard_eval/`)

---

## Where we've failed (important — do not retry these blindly)

### Fail 1 — BoW v2 on 96K augmented rows: 0.268 top-1

Bag-of-words + uniform-weight training + template-heavy augmentation → noise gets weighted like signal. Trained MLP UNDERPERFORMS raw TF-IDF (0.696). Recorded as `ve-ins-0mo2l917l0000606b23`.

**Lesson**: don't use BoW with augmented data. Use gemma-embed.

### Fail 2 — v2 gemma-embed pilot: attractor on window 160

Even with gemma-embed semantically correct inputs, the MLP predicted window 160 in 3 of 5 miss cases. Top-3 was 0.8 (semantics right), top-1 was 0.2 (ranking wrong).

**Lesson**: the dataset has a class-balance or content-bleed issue that must be fixed before gemma-embed can shine. See diagnostic recipe above.

### Fail 3 — CPU training took "ages" before cancellation

Three compounding causes: CLI default `--device cpu` (even though CUDA available), per-sample encoder calls in the training hot loop (no precompute cache), and a second call site hardcoding `device="cpu"` in the eval path. All fixed in v2. Recorded as `ve-ins-0mo2jplio0000087a38` and `ve-ins-0mo2jregc000091e5b0`.

**Lesson (baked into v2)**: `--device auto` default, precompute cache mandatory, one device propagation path end-to-end.

### Fail 4 — LARQL CUDA Epic-2 Batch 1 crashed mid-bench

Separate parallel mission (not part of MLP v2). Codex build-lead died while writing `docs/cuda/baseline.md`. Bench file `crates/larql-compute/benches/cuda_decode_tok_s.rs` survived. Recorded as `ve-ins-0mo2dzdky0000e6df56` (in the LARQL vee workspace). Not this handoff's concern unless you're also resuming LARQL.

---

## Where we want to be (success criteria)

1. **100% top-1** on `tests/fixtures/aus3000/benchmark/epic1_hard.json` with the v2 checkpoint.
2. **Complete v2 workflow cataloged** as `docs/learned_router/60-mlp-v2-workflow.md` (≤200 lines). Must cover: pre-encode gate, pilot gate, full-run gate, class-balance diagnostic recipe, iteration ladder.
3. **Global skill at `~/.claude/skills/train-mlp-router/`** (SKILL.md + README.md + `assets/{template_brief.md, example_cli_session.md}`) — future AIs can invoke the skill to train an MLP router on any new store + fixture pair without rediscovering this.

---

## Must-read vee records (in this order)

| rank | id | record_type | why |
|---|---|---|---|
| 1 | `ve-ins-0mo2ms5f20000e92c23` | pattern | **Pilot-gate diagnostic + attractor pattern** — the most important record for the next Lead |
| 2 | `ve-ins-0mo2jplio0000087a38` | failure | **CPU training + precompute cache failure** — the precompute architecture that's now baked in |
| 3 | `ve-ins-0mo2l917l0000606b23` | pattern | **BoW + uniform augmentation anti-pattern** — why BoW + templates underperforms TF-IDF |
| 4 | `ve-ins-0mo2g0ihj0000bfe1eb` | guide | **Gemma-embed encoder rationale + serialization pattern** — the 19 MB vs 8 GB checkpoint gotcha |
| 5 | `ve-ins-0mo2jregc000091e5b0` | failure | **Eval-path CPU hardcode** — the second bug caught by the prior Lead |
| 6 | `ve-ins-0mo2dxhni0000917217` | reference | **AUS3000 final state snapshot** — what's on disk, what's uncommitted |
| 7 | `ve-ins-0mo2cuh7r00007a8141` | reference | **v1 WS-5 results** — the 0.765 baseline everyone keeps comparing to |

## Good supporting records (read when you hit the specific question)

| id | record_type | when to read |
|---|---|---|
| `ve-ins-0mo2dxh0y0000a85a55` | decision | When you need to know why route.py was fixed a specific way |
| `ve-ins-0mo2g0j3g000075c895` | reference | When someone asks how big the v2 checkpoint is on mobile |
| `ve-ins-0mo2dvraa00002e666d` | pattern | If spawning another 4-tier orchestration hierarchy |
| `ve-ins-0mo2dvrvz0000d4faae` | pattern | When deciding Claude vs Codex for an agent role |
| `ve-ins-0mo2dwjb10000633081` | failure | When an agent hits context-limit |
| `ve-ins-0mo2dwjwh000038021a` | failure | When user attaches to a live agent pane |
| `ve-ins-0mo2dwki200003942a4` | failure | When Lead-Orchestrator handoff stalls |
| `ve-ins-0mo2dygha0000077e73` | guide | Template for every future handoff brief |
| `ve-ins-0mo2dyh2l00009838cb` | reference | Tmux + vee command cheat sheet |
| `ve-ins-0mo2dxi920000360596` | reference | vee record syntax gotcha (record_type is positional) |
| `ve-ins-0mo2dyhon00003bc6e2` | convention | MLX/torch parity conventions |
| `ve-ins-0mo2lsoo20000359702` | failure | Codex input-buffer stuck recovery |
| `ve-ins-0mo28m77t0000f1f2b5` | guide | Lazarus/Markov architecture for anyone new to the codebase |
| `ve-ins-0mo28mlsi0000c0d901` | reference | What a window is |
| `ve-ins-0mo28n9dt000006533e` | decision | Mobile Path C runtime decision |
| `ve-ins-0mo28nokt00003e5b05` | reference | Mobile app bill of materials |
| `ve-ins-0mo28o6dq0000898000` | reference | LiteRT-LM rejected |

Retrieval commands:

```bash
vee query "pilot gate attractor distribution shift"
vee query "gemma-embed CPU training precompute"
vee query "BoW augmentation underperforms TF-IDF"
vee query "gemma-embed serialization checkpoint"
vee query "MLP router v1 results"
```

---

## The exact state of the codebase right now

### Uncommitted on chuk-lazarus main

All the v2 infrastructure, docs, and artifacts:

```
tools/_window_router/{cache.py, encoder.py, dataset.py, augmentation.py, eval.py}
tools/train_window_router.py
tests/tools/window_router/*.py  (6 test files, 39/39 pass)
tests/tools/window_router/test_feature_cache.py  (new)
tests/fixtures/aus3000/benchmark/epic1_hard.json  (harder benchmark)
artifacts/router/aus3000_v2_ds.jsonl  (103,609 rows, augmented + multi-clause)
artifacts/router/aus3000_v2.pt  (48 MB — the BoW-v2 failed checkpoint, probably delete)
artifacts/router/aus3000_v2_pilot_1k.*
artifacts/router/aus3000_v2_pilot_eval/report.json
docs/learned_router/10-50-*.md  (5 Epic-2 docs)
docs/learned_router/eval/aus3000_eval.md  (v1 report)
docs/learned_router/HANDOFF-to-next-hand.md  (this file)
```

No commit has been made. The next Hand should decide whether to commit this state as "v2 infrastructure + pilot failure state" or wait until v2 hits 100%.

### Frozen (do not touch)

- `src/chuk_lazarus/inference/context/knowledge/route.py` — the 23/23 fix lives here
- `src/chuk_lazarus/inference/context/knowledge/torch_store.py`
- `src/chuk_lazarus/inference/context/knowledge/torch_query.py`
- Any file that `import mlx`
- `docs/aus3000_accuracy_program/04-reference-card.md`
- `tests/fixtures/aus3000/benchmark/epic1_v1.json` (keep as regression reference)

### AUS3000 knowledge store

- Path: `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant/torch_store`
- 1203 windows, 1170 unique clauses, 27 multi-part clauses
- Gemma 4 E2B HF cache at `/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/` (partial — may need completion on first run)

---

## Agent hierarchy for next Hand chat

Same as this session:

```
HAND (you, Claude top-orchestrator, Opus 4.7 1M)
 └─▶ Codex ORCHESTRATOR       — vee agent spawn: ALLOWED
      └─▶ Codex LEAD          — vee agent spawn: ALLOWED
           └─▶ Codex WORKSTREAMS — vee agent spawn: BANNED, native codex sub-agents OK
```

Spawn pattern:

```bash
tmux new-session -d -s mlpv3orch -c /mnt/c/users/jehma/desktop/lazarus/chuk-lazurus
BRIEF=$(cat /tmp/orch_brief.txt)
vee agent spawn codex --name mlpv3-orch --target mlpv3orch \
  --cwd /mnt/c/users/jehma/desktop/lazarus/chuk-lazurus --then "$BRIEF"
```

Codex auto-submits — no `tmux send-keys Enter` needed after spawn.

---

## First actions for next Lead (the exact recipe)

1. `vee query "pilot gate attractor distribution shift"` — pull the pattern record
2. `jq -r '.label' artifacts/router/aus3000_v2_ds.jsonl | sort | uniq -c | sort -rn | head -20` — run class-balance diagnostic
3. Report top-20 label histogram to Orchestrator before proposing any fix
4. Orchestrator reviews + approves one of: **label cap**, **template rewrite**, or **clause-ID channel (only if first two are ruled out)**
5. Lead applies fix, rebuilds dataset, rebuilds pilot cache (the SHA256 key will change, forcing fresh encode)
6. Re-run pilot gate. Must hit top-1 ≥ 0.8 or stop.
7. If pilot passes, run full dataset encode (wall-clock 45-90 min on CUDA, one-time), full train (~10-20 min), full eval
8. Full eval on `epic1_hard.json` must hit **1.000 top-1**. If not, iteration ladder (Step A, Step B) in the Lead's authority.
9. Only when 100% hits: ship the v2 checkpoint, write the workflow doc, build the skill.

---

## Non-negotiables

- 100% top-1 on `epic1_hard.json`. Not 90%, not 97%.
- AUS3000 `single_pass_gate` stays **23/23** at every batch gate — non-regressable.
- No MLX file edits.
- No edits to `route.py`, `torch_store.py`, `torch_query.py`, `torch_runtime.py`.
- Checkpoint serialization: MLP state_dict + encoder config only. **Never** save full Gemma into `.pt`.
- Precompute cache is mandatory — encoder must never be called in the training hot loop.
- Pilot gate is mandatory. Top-1 ≥ 0.8 or STOP.
- `/compact` at 70% context. Don't wait until it breaks.

---

## Key files to read if you want more detail

- `docs/learned_router/50-reference-card.md` — current state summary
- `docs/learned_router/20-training-guide.md` — CLI workflow
- `docs/learned_router/eval/aus3000_eval.md` — v1 eval report (the 0.765 baseline)
- `artifacts/router/aus3000_v2_pilot_eval/report.json` — the v2 pilot failure numbers
- `tools/_window_router/cache.py` — the precompute cache module
- `tools/_window_router/encoder.py` — gemma-embed + BoW encoders

---

## Vee workspaces

- chuk-lazarus vee workspace: `/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/.vee/` — all MLP router records live here
- LARQL vee workspace: `/mnt/c/Users/jehma/Desktop/larql/.vee/` — LARQL CUDA records (separate mission, not this handoff)

Both use `vee` CLI via `/home/jehmal/.cargo/bin/larql` → actually `vee` itself is at whatever binary your PATH points to. Run `which vee` to confirm.

---

**End of handoff. Good luck. The pilot gate already caught the hard bug — now it's a diagnostic problem, not an architectural one.**
