# Handoff v4 — MLP v2 Router (Wave 3 below success gate)

Wave 3 (mission `chuk-lazurus-4tw`) executed the full pipeline cleanly on 2026-04-18 with the encoder swapped from the broken quantized Gemma-4-E2B-it to `sentence-transformers/all-MiniLM-L6-v2`. All mandatory gates (0.0 preservation, 0.1 load report, 0.2 cos-sim sanity) passed. The trained `aus3000_v3_final.pt` scores top1=0.6964 on the 56-case `epic1_hard.json` benchmark versus a top1 target of 1.000 — below not only the target but also the 0.95 gap-analysis floor, which triggers this wave-4 handoff. Single-sentence root cause: learned features tie or beat the MiniLM-tokenised TF-IDF baseline on top3, MRR, and multi-clause recall@3, but REGRESS on single-clause top1 (model 0.5854 vs baseline 0.7805), and the binding constraint is a train/eval distribution mismatch on the 1238-dim clause-hierarchy one-hot channel — training rows frequently carry explicit clause IDs, natural-language eval queries rarely do, so the channel collapses to zeros at inference and the MLP loses the signal it learned to rely on.

Vee session IDs: LEAD_SID `ve-ses-0mo27zk2u000051cae8`. Decision records written this wave: `ve-ins-0mo38951m0000e3f6f5` (Gate 0.0), `ve-ins-0mo38qthi00007746c3` (Phase 1.1 + 1.2), `ve-ins-0mo3936cf000027bce6` (Gate 0.1 + 0.2), `ve-ins-0mo397sw50000808bc9` (EPIC 2.1 precompute), `ve-ins-0mo39dwhc0000c46a45` (EPIC 2.2 train), `ve-ins-0mo39hkw00000f6bf2b` (EPIC 2.3 eval).

## Final Eval Numbers

Evaluation set: `tests/fixtures/aus3000/benchmark/epic1_hard.json`, n=56, single_clause_n=41, multi_clause_n=15, multi_clause_clause_count=31.

### MLP v3 (this iteration)

- top1: 0.6964
- top3: 0.7679
- mrr: 0.7292
- single_clause_top_1: 0.5854 (41 cases)
- multi_clause_recall@3: 0.6452 (15 cases, 31 target clauses)

### TF-IDF baseline (MiniLM tokenizer)

- top1: 0.6964
- top3: 0.7143
- mrr: 0.7054
- single_clause_top_1: 0.7805
- multi_clause_recall@3: 0.3548

### Comparison vs wave-2 baseline

Wave-2 reported TF-IDF top1 0.786 with the Gemma-4 tokenizer. Wave-3 baseline at the MiniLM tokenizer is 0.6964 — the TF-IDF floor SHIFTED under the tokenizer swap. Apples-to-apples the MLP v3 matches MiniLM-TFIDF on top1 and beats it on top3 (+0.054), MRR (+0.024), and multi-clause recall@3 (+0.290). It still underperforms Gemma-tokenizer TF-IDF on top1 (0.6964 < 0.786). Future benchmarks should report TF-IDF at BOTH tokenizers so cross-wave comparisons remain valid.

### Success gate outcome

- Target: top1 >= 1.000
- Outcome: FAILED (0.6964)
- Wave-4 handoff path (top1 < 0.95) selected.

## Root Cause (single-clause regression)

Wave-2's root cause (quantized-Gemma silent-random-init) is RESOLVED. Gate 0.1 confirms a clean load against `sentence-transformers/all-MiniLM-L6-v2`: MISSING=0, UNEXPECTED=1 (benign `embeddings.position_ids` buffer). Gate 0.2 passed all four cos-sim ordering pairs. The encoder emits real semantic vectors (SENT_STD=0.051031, SENT_NORM_MEAN=1.000 per precompute log). Wave 3 therefore does not repeat wave-2's diagnosis; it diagnoses the NEW gap.

### Case-level diff evidence

From `report.json`, the model loses to the baseline on these single-clause cases where the prompt does NOT contain an explicit clause ID:

- `para_accessible_service_access` — "Under AS/NZS 3000, when is equipment considered accessible for inspection, maintenance, or repair?" expected window 5. Model top3: [384, 692, 212]. Baseline top3: [5, 104, 415, 499, 821].
- `para_competent_person_site_role` — "Who counts as a competent person for AS/NZS 3000 electrical work?" expected 37. Model top3: [217, 53, 423]. Baseline top3: [37, 415, 315, 499].
- `para_switchboard_equipment_assembly` — "In the standard, what equipment assembly is classed as a switchboard?" expected 124. Model top3: [982, 356, 325]. Baseline top3: [124, 151, 635, 636].
- `para_efli_clause_8391` — "Clause 8.3.9.1: when is EFLI testing required for socket-outlets?" expected 1195. Model top3: [790, 495, 1194]. Baseline top3: [1195, 690, 217]. Baseline wins even though the prompt does mention "8.3.9.1" — the clause-ID feature did not save the model here.

Where clause ID is IN the prompt and short/lexically dominant (e.g. `para_accessible_plain_language` "What does clause 1.4.2 mean by accessible, in plain language?"), the model routes correctly — window 5 is top1. Where it isn't, the model's predictions are NOT near-neighbours of the correct window (384, 692, 982, 217 are scattered across the index space), suggesting the model is not falling back gracefully to sentence-vector similarity alone.

### Three candidate hypotheses (not conclusions)

1. **Feature distribution shift on the clause-hierarchy channel.** Training rows (74,721) were synthesised from paraphrase templates that often contain explicit clause references, so the 1238-dim clause-hierarchy channel is frequently non-zero during training. Natural-language eval queries rarely include a clause ID, so those 1238 dims are almost always all-zero at inference. The MLP appears to have learned to rely on this channel to disambiguate windows, and at eval time the channel collapses to zeros and the model falls back to weaker signal from the 384-dim sentence vector. This is a classic train-eval distribution mismatch.

2. **Sentence / clause feature scale mismatch.** Sentence vectors are L2-normalised (unit sphere; `SENT_NORM_MEAN=1.000` in precompute log). Clause-hierarchy features are sparse binary indicator counts with L2 norms that vary with how many clause IDs appear in a training row. An MLP without explicit per-channel normalisation may over-weight the high-variance channel during training, making it brittle when that channel is absent at inference.

3. **MiniLM is a general-purpose semantic encoder, not domain-adapted.** The embedding geometry may not sufficiently separate electrical-standards terminology. Concepts tightly coupled in AS/NZS 3000 ("assembly" vs "switchboard", "accessible" vs "readily accessible") may embed at distances that do not reflect domain semantics, because MiniLM was trained on general web text.

## What Worked (Preserve)

Do not redo. Tests pass and artifacts are clean.

- EPIC 1 WS-A/B/C code changes from wave 2 — 72/72 window_router tests still green; wave 3 added 12 new tests for the encoder swap + sanity gate — 84/84 total.
- Collision-free 74,721-row dataset `artifacts/router/aus3000_v2_ds.jsonl` + sidecar `aus3000_v2_ds.jsonl.meta.json`.
- New `HFSentenceEncoder` class in `tools/_window_router/encoder.py` and `encoder_version` bump in `tools/_window_router/cache.py` to `sentence-transformer-v1::sentence-transformers/all-MiniLM-L6-v2`. MiniLM checkpoint loads cleanly (MISSING=0, UNEXPECTED=1 benign).
- `tools/_window_router/_encoder_sanity.py` + `tests/tools/window_router/test_encoder.py` coverage — a deterministic pre-train sanity gate that would have caught wave-2's silent-random-init in under 5 seconds once the model is loaded. Execution cost of the gate end-to-end this wave: 33.77s (model load) + ~1s (encode + 4 ordering checks).
- Feature cache `artifacts/router/aus3000_v2_ds.sentence-transformer.ad1d23c2b93349cb.features.pt` — shape (74721, 1622), float32, ~463 MB. Encoder version: `sentence-transformer-v1::sentence-transformers/all-MiniLM-L6-v2`. Dataset hash: `afba5649f774db3ee36ff47f17e38d3ec1a90dcb828647701c1cae691de5f831`.
- Trained checkpoint `artifacts/router/aus3000_v3_final.pt` (~11 MB, num_labels=1203, dim=1622). Wave-2 `aus3000_v2_final.pt` preserved unmodified.
- Eval artifacts `artifacts/router/aus3000_v3_final_eval/report.{json,md}`.
- Pipeline plumbing (precompute → train → eval CLI, cache keying, tokenizer-configurable TF-IDF baseline) is deterministic and correct.

## What Did Not Work

- `sentence-transformers/all-MiniLM-L6-v2` (384-dim) + 1238-dim sparse clause-hierarchy channel concat + residual MLP (hidden=256, labels=1203) + 10 epochs at lr=1e-3 batch=32. The pipeline ran cleanly but did not produce a model that generalises to natural-language queries without explicit clause-ID lexical hints.
- Treating the clause-hierarchy one-hot channel as a free-form concat addition. At the fitted vocab size for 74,721 rows, the channel is 1238-dim — more than triple the sentence-embedding dim — and sparsely populated in a way that does not match eval-time inputs.

## Artifacts Preserved

Inspect without recomputing.

- `artifacts/router/aus3000_v2_ds.jsonl` — 74,721 rows, collision-free
- `artifacts/router/aus3000_v2_ds.jsonl.meta.json` — dataset sidecar
- `artifacts/router/aus3000_v2_ds.sentence-transformer.ad1d23c2b93349cb.features.pt` — ~463 MB, (74721, 1622) float32 feature cache (encoder_version `sentence-transformer-v1::sentence-transformers/all-MiniLM-L6-v2`)
- `artifacts/router/aus3000_v3_final.pt` — ~11 MB trained MLP over real features (num_labels=1203, dim=1622)
- `artifacts/router/aus3000_v2_final.pt` — ~19 MB wave-2 checkpoint (preserved, NOT overwritten)
- `artifacts/router/aus3000_v3_final_eval/report.json` — full case-level detail
- `artifacts/router/aus3000_v3_final_eval/report.md` — summary
- `artifacts/router/aus3000_v2_final_eval/report.{json,md}` — wave-2 eval (preserved)
- `artifacts/router/logs/gate01_load_report.log` — Gate 0.1 PASS, MISSING=0, UNEXPECTED=1
- `artifacts/router/logs/gate02_sanity.log` — Gate 0.2 PASS (4/4 ordering pairs, 49.06s wall)
- `artifacts/router/logs/precompute_wave3.log` — encode elapsed 10.42s on 74,721 rows, wall 60.10s
- `artifacts/router/logs/train_wave3.log` — checkpoint save line only, wall 88.30s (see Protocol Learnings re: missing loss lines)
- `artifacts/router/logs/eval_wave3.log` — model vs TF-IDF summary, wall 52.00s

## Iteration Ladder for Next Hand

Each step gates the next. Do not skip ahead. Steps 1-2 are cheap and should be run BEFORE any new encoder or compute decisions.

### 1. Before any new compute: measure the train-eval distribution gap on the clause channel

Cheap probe. For each of the 56 eval-case prompts, compute the clause-hierarchy feature vector using the same clause-feature fitter used in precompute (the 1238-dim one-hot). Compare against the per-row distribution across the 74,721 training rows.

Report:

- What fraction of eval cases have an all-zero clause-feature vector?
- What fraction of training rows have an all-zero clause-feature vector?
- Cosine distance distribution between eval clause-vectors and the training-set mean clause-vector.

**Decision rule**: if >80% of eval cases have an all-zero clause-feature vector while <30% of training rows are all-zero, hypothesis #1 is confirmed → proceed to step 2. If the distributions are comparable, skip to step 3.

### 2. Drop or remap the clause-hierarchy channel

Candidate approaches, ordered by expected cost:

- a. **Zero-out the clause channel at TRAIN time** for a configurable fraction (e.g. 50%) of rows — feature dropout that matches the eval distribution. Cheapest. Re-uses existing encoder + cache.
- b. **Remove the clause channel entirely** — train on the 384-dim sentence vector only. Expect the ceiling to be TF-IDF on single-clause but a higher ceiling on multi-clause and MRR. Requires a new cache key (bump `encoder_version` to a variant that excludes the clause channel).
- c. **Project the clause channel through a small learned dense embedding** (e.g. 1238 → 32) concatenated with the 384-dim sentence vector — smooths the distribution shift without discarding the signal.

### 3. Stronger or domain-adapted encoder

- a. `sentence-transformers/all-mpnet-base-v2` — 768-dim, better general STS than MiniLM. Drop-in replacement via the existing `HFSentenceEncoder` path.
- b. `BAAI/bge-large-en-v1.5` — larger, retrieval-tuned.
- c. Contrastive fine-tune of MiniLM on the AUS3000 data itself — positive pairs = paraphrases of the same window, negatives = different windows. Requires new training-data wiring but reuses the 74,721-row dataset.

### 4. Architecture and regularisation

- a. Linear probe on frozen features (remove the MLP hidden layer) — establishes a linear-separability floor.
- b. Add per-channel L2 normalisation to the clause channel, or batch-norm the concatenated input.
- c. Explicit per-channel dropout during training — drop the clause channel with p=0.5 for half the rows. Complements (2a).

### 5. Revisit the benchmark and success gate

Wave 3 confirmed the encoder was not wave-2's bottleneck and that the pipeline is sound. Consider whether top1 = 1.000 on 56 cases is achievable even with perfect semantic features, or whether a more nuanced gate better reflects real utility. Proposals:

- `top1 >= MiniLM-TFIDF + 0.10` (i.e. 0.796).
- `multi_clause_recall@3 >= 0.80` (current: 0.6452; baseline: 0.3548 — model already beats baseline by ~0.29 here).
- Composite: top1 >= 0.85 AND multi_clause_recall@3 >= 0.70 AND top3 >= 0.90.

The current model BEATS MiniLM-TFIDF on top3, MRR, and multi-clause; that is real utility even at 0.6964 top1. Escalate the gate choice to the Hand before dispatching the next full-train cycle.

## Protocol Learnings

- The cos-sim sanity script (`tools/_window_router/_encoder_sanity.py`) is now permanent infrastructure. Run it before any precompute whenever the encoder changes. It catches wave-2-style silent-random-init in under 5 seconds once the model is loaded. Total wall on this wave: 49.06s including load.
- Feature-cache timing discrepancies were benign this wave. Encode elapsed 10.42s on 74,721 rows vs the brief-stated 35-second budget. MiniLM-L6-v2 is simply faster than the Gemma reference the budget assumed. Use Gate 0.3 (fast-completion pause) as a prompt to sanity-check artifacts (shape, norms, non-zero std), not a hard stop.
- Training logs are silent. `tools/train_window_router.py::main` never calls `logging.basicConfig`, so the root logger emits nothing and `train_wave3.log` contains only the final save line and wall time (88.30s). Loss trajectory can only be inferred indirectly (train-set accuracy probe). Future hands should add `logging.basicConfig(level=logging.INFO)` at the CLI entrypoint.
- Train-time speed on cached features was ~264 steps/sec for a 3-layer MLP on 1622-dim features — 88.3s for 10 epochs on 74,721 rows at batch 32. Budget the next wave's training at ~100s on cached features, not 5-10 min.
- `--tokenizer-model-id` affects the TF-IDF baseline, not the model. Cross-wave baseline comparisons require holding the tokenizer constant. Future handoffs should report TF-IDF at BOTH the previous wave's tokenizer AND the current one. This wave's wave-2 comparison was initially confusing for exactly this reason (baseline shifted from 0.786 to 0.6964 under the tokenizer swap).
- The mandatory-numbered-gate protocol (0.0 → 0.1 → 0.2 → 2.x) worked. No false-negative overrides, no skipped sanity checks, no silent encoder failures. Keep the same discipline for wave 4.

## Cross-References

- Gate 0.0 PASS (preservation check): `ve-ins-0mo38951m0000e3f6f5`
- Phase 1.1 + 1.2 PASS (encoder swap + tests): `ve-ins-0mo38qthi00007746c3`
- Gate 0.1 + 0.2 PASS (load report + cos-sim sanity): `ve-ins-0mo3936cf000027bce6`
- EPIC 2.1 (precompute): `ve-ins-0mo397sw50000808bc9`
- EPIC 2.2 (train): `ve-ins-0mo39dwhc0000c46a45`
- EPIC 2.3 (eval): `ve-ins-0mo39hkw00000f6bf2b`
- Wave-3 orchestrator session: `ve-ses-0mo37p0dh000038bbac`
- Wave-3 lead session: `ve-ses-0mo27zk2u000051cae8`
- Wave-3 brief: `docs/learned_router/specs/01-wave3-hand-to-orchestrator-brief.md`
- Wave-2 handoff (template + prior context): `docs/learned_router/HANDOFF-to-next-hand-v3.md`

(Lead may append further record IDs in this section post-facto.)
