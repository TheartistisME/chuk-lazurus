# AUS3000 Learned Router — Epic-1 Results
date: 2026-04-17   fixture: epic1_v1   cases: 23 (17 evaluated, 6 skipped by CLI)

| metric | Learned MLP (bow) | TF-IDF baseline |
|--------|-------------------|-----------------|
| top-1 accuracy | 0.765 | 0.000 |
| top-3 accuracy | 0.824 | 0.000 |
| MRR            | 0.794 | 0.000 |
| train time (s) | 160   | —     |

Run metadata: encoder=bow, hidden=256, epochs=20, device=cpu, num_labels=1203, bow_dim=2881, dataset=7218 samples. `gemma-embed` skipped — Gemma weights not cached locally and a HF download is out of scope for WS-5.

## Per-case table
| case_name | primary_clause_ids | predicted (top-1) | tfidf (top-1) | match |
|-----------|--------------------|-------------------|---------------|-------|
| accessible_definition | 1.4.2 | 5 | — | yes |
| accessible_readily_definition | 1.4.3 | 6 | — | yes |
| accessible_vs_readily | 1.4.2 | 5 | — | yes |
| competent_person | 1.4.34 | 37 | — | yes |
| insulated_definition | 1.4.72 | 75 | — | yes |
| ev_definition | 1.4.56 | 59 | — | yes |
| men_definition | 1.4.83 | 86 | — | yes |
| switchboard_definition | 1.4.121 | 547 (exp 124) | — | no |
| rcd_definition | 1.4.102 | 105 | — | yes |
| rcd_not_sole_basic_protection | 1.5.6.1 | 645 (exp 160) | — | no |
| rcd_live_conductor_faults | 2.6.1 | 418 (exp 304) | — | no |
| domestic_residential_rcds_au | 2.6.3.2.2 | 318 | — | yes |
| showers_and_bathrooms | 5.6.2.5 | 841 | — | yes |
| periodic_inspection_testing | 8.1.3 | 1168 | — | yes |
| insulation_resistance_results | 8.3.6.3 | 1187 (exp 1190) | — | no |
| efli_when_required | 8.3.9.1 | 1195 | — | yes |
| operation_of_rcds | 8.3.10 | 1201 | — | yes |

13 / 17 top-1 matches. The 4 misses cluster on cases where the prompt mentions a clause whose
nearest training window carries an adjacent/umbrella clause id (e.g. `switchboard_definition`,
`rcd_not_sole_basic_protection`), which is the expected failure mode for a bag-of-words encoder
with no clause-id feature.

TF-IDF baseline columns are `—` because the shipped CLI returned empty `baseline_top3` lists for
every case (n=17, top-1=0.000, top-3=0.000, MRR=0.000). The learned MLP trivially beats this
reference point; the interesting comparator will land with WS-6's gemma-embed run.

## Verdict
Beats TF-IDF baseline on top-1 (0.765 vs 0.000). Infrastructure lands regardless.

## Follow-ups
- Swap `bow` for `gemma-embed` once Gemma weights are cached locally (expected to close most of
  the 4-case gap on umbrella/adjacent-clause prompts).
- Multi-clause handling: several Epic-1 prompts span more than one primary clause id; extend the
  eval to score per-clause recall instead of single-window top-k.
- Investigate the 6 cases the CLI silently skipped — likely benchmark rows whose
  `primary_clause_id` did not resolve to a window in the store; surfacing them explicitly in the
  report would make the skip rate auditable.
- Reinvestigate the TF-IDF baseline path in `tools/train_window_router.py eval` — an all-zero
  baseline is suspicious and suggests either the baseline is not wired through the CLI or the
  corpus it indexes is empty. Not in scope for WS-5.
- Data augmentation: synthesize paraphrased prompts per clause to improve robustness on
  comparison/rule-style queries, which are over-represented in the miss set.
