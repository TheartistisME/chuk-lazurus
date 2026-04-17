# AUS3000 Learned Router Results

date: 2026-04-17  
fixture: `epic1_v1`  
cases: 23 total, 17 evaluated, 6 skipped by the CLI

## Snapshot

The learned bag-of-words MLP reached top-1 0.765, top-3 0.824, and MRR 0.794 on the 17 evaluated AUS3000 cases. That is 13/17 exact top-1 matches.

The TF-IDF baseline in the shipped CLI reported 0.000 for top-1, top-3, and MRR. That result is not a model-quality comparison: the CLI returned empty `baseline_top3` lists for every case, so the baseline path was broken. This was out of scope for WS-5.

The infrastructure still lands regardless. The learned router, dataset builder, trainer, and eval reporting path are now in place and usable for future store/benchmark pairs.

## Metrics

| metric | learned MLP (bow) | TF-IDF baseline |
|--------|-------------------|-----------------|
| top-1 accuracy | 0.765 | 0.000 |
| top-3 accuracy | 0.824 | 0.000 |
| MRR | 0.794 | 0.000 |
| top-1 matches | 13/17 | 0/17 |

Run metadata from the eval artifact: encoder=`bow`, hidden=256, epochs=20, device=`cpu`, num_labels=1203, bow_dim=2881, dataset=7218 samples, train time about 160 s.

## Top-1 Misses

The four top-1 misses are:

| case_name | primary_clause_id | predicted | expected |
|-----------|-------------------|-----------|----------|
| switchboard_definition | 1.4.121 | 547 | 124 |
| rcd_not_sole_basic_protection | 1.5.6.1 | 645 | 160 |
| rcd_live_conductor_faults | 2.6.1 | 418 | 304 |
| insulation_resistance_results | 8.3.6.3 | 1187 | 1190 |

These misses are consistent with the bag-of-words failure mode: the model lands on an adjacent or umbrella clause window instead of the exact target clause. `insulation_resistance_results` is a near miss because the expected window is rank 2, so it still counts inside top-3.

## Honest Comparison

The learned model beat the broken TF-IDF CLI baseline numerically, but that baseline number is not trustworthy as a routing benchmark. The useful result here is that the learned-router infrastructure is now working end to end and the AUS3000 eval path is reproducible on the real store/fixture pair.

The comparison that matters next is a repaired TF-IDF path or a stronger embedding encoder, then a rerun on the same 17 evaluated cases.

## Source Check

The figures above were cross-checked against `docs/learned_router/eval/aus3000_eval.md` and `docs/learned_router/eval/aus3000_report.json`.
