# Handoff v3 — MLP v2 Router Rescue (Failed Iteration)

The MLP v2 router rescue attempt (mission chuk-lazurus-z3c, workspace chuk-lazurus-04f) completed the full pipeline end-to-end on 2026-04-17 and produced a trained classifier that scores top1=0.000, top3=0.018, mrr=0.009 on the 56-case epic1_hard benchmark versus a TF-IDF baseline at top1=0.786, top3=0.857, mrr=0.826 — a hard failure against the top1 >= 1.00 success gate. The single-sentence root cause: the cached Gemma-4-E2B-it checkpoint at `/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/` is a quantized multimodal checkpoint whose transformer weights do not map onto the `Gemma3ForConditionalGeneration` attribute namespace, so every `language_model.*` tensor was reported missing and silently randomly initialised, making the encoder emit noise. The single-sentence recommendation: before any further compute is dispatched, the next Hand must run the Bug 2 cos-sim sanity gate end-to-end on the real encoder and switch to a checkpoint that loads cleanly (Gemma-3-4B-it at `/home/jehmal/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767/` is the first candidate to try).

Vee failure record: `ve-ins-0mo2t4uxr0000757ed3`. Orchestrator SID: `ve-ses-0mo2nzckf0000ce5c25`. Lead SID: `ve-ses-0mo27zk2u000051cae8`.

## Final Eval Numbers

Evaluation set: `tests/fixtures/aus3000/benchmark/epic1_hard.json`, n=56.

### MLP v2 (this iteration)

- top1: 0.000
- top3: 0.018
- mrr: 0.009
- single_clause_top1: 0.000
- multi_recall@3: 0.000

### TF-IDF baseline

- top1: 0.786
- top3: 0.857
- mrr: 0.826
- single_clause_top1: 0.854
- multi_recall@3: 0.484

### Success gate

- Target: top1 >= 1.00
- Outcome: FAILED

## Root Cause

Bug 2 (encoder fix) from the prior Hand's brief is incompatible with the cached Gemma-4-E2B-it checkpoint.

Facts:

- Gemma-4-E2B-it at `/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf` is a quantized multimodal checkpoint.
- Weights are stored under paths like `language_model.embed_tokens.linear.weight`, with accompanying scale tensors `.input_min` and `.output_max`.
- `transformers.AutoModel.from_pretrained` resolves this architecture to `Gemma3ForConditionalGeneration`, which expects `language_model.embed_tokens.weight` (no `.linear.` segment, no scale tensors).
- Every `language_model.*` weight was reported MISSING on load and silently randomly initialised.
- Log evidence: `Loading weights: 0it [00:00, ?it/s]` — 0 shards actually loaded. See `artifacts/router/logs/precompute_full.log` and `artifacts/router/logs/eval_full.log`.
- `self._model(input_ids, attention_mask).last_hidden_state` therefore outputs random noise. The MLP trained on random features → 0.000 top-1 is exactly what a well-trained classifier over noise produces against a 1203-class target.
- The root-level `embed_tokens` table itself DID load correctly in the previous encoder (embedding-layer-v1), which scored 0.2 top-1. The input embedding matrix loads cleanly; only the transformer layers on top of it fail to load.

### Why it was not caught before full-train dispatch

- WS-B `test_encoder.py` mocked `_model` to return a known `last_hidden_state` tensor. Mean-pool math was verified. Encoder loading on the real checkpoint was not verified end-to-end.
- Bug 2's cos-sim sanity gate (per brief: unrelated < 0.92, related > 0.85) was defined but never executed against the real Gemma before full compute dispatch.
- Feature cache completion was suspiciously fast — approximately 4 minutes for 74,721 rows, against an expected 45-90 minute window. This anomaly should have triggered a pause-and-verify. It did not.
- The 5% pilot gate FAILED (top1=0.000) but was ruled a false-negative on the rationale that 5% x 1203 classes is roughly 3 samples per class, below any plausible learning threshold. The orchestrator overrode the gate and dispatched full train on that reasoning. The rationale was plausible but wrong: the encoder is the real problem, and the pilot gate was genuinely diagnostic rather than undersized.

## What Worked (Preserve)

Do not redo this work. It is correct and tests pass.

### EPIC 1 WS-A/B/C code changes

- 72/72 window_router tests green
- 14/14 torch training tests green
- 23/23 single_pass_gate tests green
- 12/12 aus3000 regression tests green
- 9/9 variant evaluator tests green
- 56/56 models_v2 tests green
- ruff clean across touched packages

### EPIC 2.1 data hygiene

- Collision gate: 0 collisions on 75,924 rows x 1203 labels with benchmark coverage complete.
- This was a real fix. The original Hand brief's Bug 1 diagnosis missed 679 split-clause collisions (27 clause_ids shared across sibling windows via `part_index`) and 44 benchmark-pair emissions (intentional dual-labels).
- WS-A-EXT scope expansion decision is recorded as `ve-ins-0mo2qrtor0000e9c3b5`.

### Pipeline plumbing

- End-to-end train, eval, and CLI behaviour is deterministic and correct.
- The broken component is the encoder content, not the pipeline's control flow.

## What Did Not Work

- Bug 2 encoder fix on the Gemma-4-E2B-it quantized multimodal checkpoint. The weight-name mismatch between the stored quantized tensors and the `Gemma3ForConditionalGeneration` attribute namespace silently defeats `from_pretrained`.
- 5% stratified pilot as a gate for a 1203-class problem. At this label cardinality the pilot is fundamentally undersized and produces ambiguous signal that invites false-negative overrides.

## Artifacts Preserved

The next Hand can inspect these without re-running any compute.

- `artifacts/router/aus3000_v2_ds.jsonl` — 75,924 rows, collision-free
- `artifacts/router/aus3000_v2_ds.jsonl.meta.json`
- `artifacts/router/full_features/aus3000_v2_full.gemma.features.pt` — 829 MB, broken feature cache
- `artifacts/router/aus3000_v2_final.pt` — 19 MB, trained MLP over broken features
- `artifacts/router/aus3000_v2_final_eval/report.json`
- `artifacts/router/aus3000_v2_final_eval/report.md`
- `artifacts/router/pilot_5pct.pt`
- `artifacts/router/pilot_5pct_eval/`
- `artifacts/router/logs/eval_full.log`
- `artifacts/router/logs/train_full.log`
- `artifacts/router/logs/precompute_full.log`

## Iteration Ladder for Next Hand

Each step gates the next. Do not skip ahead.

### 1. Verify the Bug 2 sanity gate end-to-end on the real Gemma BEFORE dispatching any compute

Concrete check:

```
from tools._window_router.encoder import GemmaEmbedEncoder
enc = GemmaEmbedEncoder(model_id='<path>', device='cuda')
v1 = enc.encode(['What does clause 1.4.2 mean by accessible, in plain language?'])[0]
v2 = enc.encode(['Explain Basic protection'])[0]
v3 = enc.encode(['Accessible'])[0]
assert cos(v1, v2) < 0.92
assert cos(v1, v3) > 0.85
```

If this fails, stop and fix the encoder first. Do not spend cache or train compute on a broken encoder.

### 2. Diagnose the quantization load mismatch

Candidate approaches, ordered by expected cost:

- a. Use a different cached model. Gemma-3-4B-it exists at `/home/jehmal/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767/`. Verify `model.safetensors` is present and that `from_pretrained` loads a non-zero number of shards cleanly.
- b. Use `sentence-transformers/all-MiniLM-L6-v2` or a similar purpose-built sentence encoder. 384-dim, fast, no quantization surprises.
- c. Load Gemma-4-E2B's quantized weights manually with a loader that maps `.linear.weight` plus scale tensors (`.input_min`, `.output_max`) to the HF attribute namespace. Likely requires custom code or a specific `transformers`-supported Gemma-4 fork.
- d. Build a hand-rolled encoder over just the `embed_tokens` table (what embedding-layer-v1 did, which scored 0.2 top-1) but improve downstream: compose `embed_tokens` with a small trainable attention pooling module trained jointly with the MLP.

### 3. Re-run the pipeline with a working encoder

- EPIC 2.1 data hygiene and the WS-C stratified pilot sampler are preserved and correct. Do not redo WS-A/B/C.
- The feature cache version must bump again if the encoder class changes, so that stale features are not reused.

### 4. Redefine the pilot gate

5% x 1203 classes is fundamentally undersized for this label cardinality. Either:

- a. Raise pilot fraction to at least 25-30%, or
- b. Replace the pilot gate entirely with the encoder sanity gate plus a direct sanity check on the first epoch of full train (loss decreasing monotonically, top-1 strictly greater than random at epoch 1).

### 5. Success-gate realism

top1 >= 1.00 on 56 cases is an extremely high bar. Even a well-tuned encoder plus MLP may land at 0.85-0.95. Consider alternative bars:

- top1 >= TF-IDF baseline plus N points
- recall@3 >= 0.9

Escalate the bar choice before dispatching the next full-train cycle.

## Protocol Learnings

- Running the Bug 2 cos-sim sanity gate end-to-end on the real encoder before full-train compute would have caught this failure in roughly 30 seconds. Budget that check as a non-negotiable pre-compute step.
- Suspicious fast-completion of long-running steps — here, a 4-minute feature cache against a 45-90 minute expectation — should trigger a pause-and-verify, not an acceleration. A step that finishes "too fast" is evidence, not momentum.
- Per-component tests with mocked underlying models verify the glue but not the content. At least one integration test must run the real model, even on 10 samples, to catch load-time failures. A test that cannot catch a silently randomly initialised model is not protecting the pipeline.
- When a gate FAILS and is ruled a false-negative, the override rationale should be falsifiable by a cheap additional check (here: the encoder sanity gate). Plausible-but-unchecked rationales are how broken encoders reach full-train.

## Cross-References

- Vee failure record for this iteration: `ve-ins-0mo2t4uxr0000757ed3`
- WS-A-EXT scope expansion decision: `ve-ins-0mo2qrtor0000e9c3b5`
- Orchestrator session: `ve-ses-0mo2nzckf0000ce5c25`
- Lead session: `ve-ses-0mo27zk2u000051cae8`
