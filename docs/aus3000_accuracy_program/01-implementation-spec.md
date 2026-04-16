# Epic 1: AUS3000 Accuracy Program - Implementation Spec

Status: Draft
Owner: `chuk-lazurus-312.1`
Scope: Epic 1 analysis/spec only. No product code changes are made by this document.
Repo root: `/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus`

---

## 1. Purpose & Scope

This spec defines the current AUS3000 clause-aligned baseline, identifies the exact
failure clusters that block a 100% PASS benchmark, maps those failures to the live
routing/query surfaces, and recommends the implementation order for Epic 2.

In scope:

- Exact benchmark contract and baseline metrics.
- Exact failure clusters and their likely root-cause classes.
- Exact code surfaces that later implementation work should touch.
- Apollo-derived lessons that transfer to AUS3000.
- Architecture options, regression surfaces, and required tests.

Out of scope:

- Editing runtime or tool code.
- Rebuilding checkpoints.
- Changing the benchmark suite itself beyond documenting its current behavior and
  noted gaps.

### 1.1 Source Artifacts

Primary sources for this spec:

- `docs/aus3000_accuracy_program/complex_task_filled.md:1-190`
- `tools/build_aus3000_clause_aligned_variant.py:96-181, 287-440, 474-640, 645-727`
- `tools/evaluate_aus3000_variant.py:119-137, 139-193, 269-540, 543-577, 580-709`
- `src/chuk_lazarus/cli/commands/context/generate/_torch.py:206-345`
- `src/chuk_lazarus/inference/context/knowledge/route.py:23-45, 94-176`
- `src/chuk_lazarus/inference/context/knowledge/torch_query.py:17-20, 159-199, 213-239, 255-379`
- `src/chuk_lazarus/inference/context/knowledge/torch_store.py:89-145, 146-195, 197-252`

External local artifacts used as factual baseline:

- `VR` = `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_validation_report.txt`
- `WM` = `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant/torch_store/window_metadata.json`
- `TP` = `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant/torch_prefill.json`
- `BM` = `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant/clause_aligned_build_manifest.json`

---

## 2. Baseline Contract

### 2.1 Current Benchmark Definition

The live evaluator defines a 23-case prompt suite in
`tools/evaluate_aus3000_variant.py:269-432`. It covers:

- lookup
- comparison
- definition
- rule
- requirement
- special_location
- verification
- testing
- out_of_domain
- adversarial

Scoring currently works as follows:

- Out-of-domain/adversarial cases PASS when the answer clearly states insufficiency
  and avoids electrical bleedthrough:
  `tools/evaluate_aus3000_variant.py:435-443, 496-518`.
- In-domain cases PASS when keyword-hit ratio is at least `0.5` and either the
  expected clause appears in routed text or answer/source overlap is at least
  `0.30`: `tools/evaluate_aus3000_variant.py:522-540`.
- In-domain cases REVIEW when partial grounding is detected:
  `tools/evaluate_aus3000_variant.py:535-537`.

### 2.2 Exact Baseline Run

The latest clause-aligned 30-minute run is the report at `VR`.
The report header and summary establish the exact run configuration and outcome:

- `VR:4-19`: started `2026-04-16T13:45:35.121881+00:00`, `top-k=1`,
  `temperature=0.0`, chat template enabled, `1170` corpus records, `23` prompt
  cases, `1203` checkpoint windows, `199258` checkpoint tokens.
- `VR:13009-13015`: completed `2026-04-16T14:15:00.790752+00:00`,
  `423` cases executed, `310 PASS`, `38 REVIEW`, `75 FAIL`, `73.29%` pass rate.

Exact store/build stats are corroborated by `TP` and `BM`:

- `TP`: `1170` clause records, `1203` windows, `27` split clauses,
  `64` overlap tokens, `max_keywords=12`, `topic_expansion_tokens=0`.
- `BM`: same `1203` windows and `11191` entries.

### 2.3 Important Baseline Interpretation

`423` executions does not mean `423` unique prompts. It is repeated execution of the
same 23-case suite:

- `18` full rounds = `414` cases.
- The run then executed cases `1-9` from round `19`, producing the final `423`.

That matters because the six non-PASS clusters below are not sporadic. They repeat
deterministically every time those prompts are encountered.

### 2.4 Baseline by Category

Derived from the report and the evaluator's case suite:

| Category | Executions | PASS | REVIEW | FAIL | Notes |
|---|---:|---:|---:|---:|---|
| `lookup` | 38 | 19 | 0 | 19 | One clause lookup passes, one fails every round. |
| `comparison` | 19 | 0 | 0 | 19 | Current config cannot satisfy multi-clause retrieval. |
| `definition` | 114 | 57 | 38 | 19 | All REVIEWs and one FAIL cluster are here. |
| `testing` | 54 | 36 | 0 | 18 | One stable fail cluster. |
| `special_location` | 18 | 18 | 0 | 0 | `5.6.2.5` fix is holding. |
| `rule` | 36 | 36 | 0 | 0 | Stable green. |
| `requirement` | 18 | 18 | 0 | 0 | Stable green. |
| `verification` | 18 | 18 | 0 | 0 | Stable green. |
| `out_of_domain` | 72 | 72 | 0 | 0 | Refusal path is already green. |
| `adversarial` | 36 | 36 | 0 | 0 | Ignore-the-store and SQL probes are green. |

### 2.5 Normative Benchmark Constants Imported From `03-benchmark-definition.md`

Epic 2 implementation and signoff must use the same frozen constants as the
benchmark-definition document:

- Canonical checkpoint path:
  `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant`
- Canonical store path:
  `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant/torch_store`
- Benchmark mode ordering:
  `store_evidence_gate -> single_pass_gate -> soak_gate`
- Authoritative greenlight gate:
  `single_pass_gate` is the counted `23/23 PASS` requirement. `soak_gate` is
  secondary stability proof after single-pass is green.
- Exact-address routing scope:
  runtime-first, using existing `window_metadata.json` to do exact clause-id and
  normalized clause-title routing before TF-IDF, with TF-IDF retained as the
  backstop. No `kvectors_full` or Apollo-style exact factual path is required
  unless the metadata-first path fails to close the benchmark.
- Named gates imported from the benchmark definition:
  `store_evidence_gate`, `route_gate`, `grounding_gate`, `ood_gate`,
  `no_regression_gate`, `single_pass_gate`, `soak_gate`

The 30-minute clause-aligned report remains the authoritative baseline evidence for
today's defect shape, but it is not the authoritative counted mode for Epic 2
greenlight.

---

## 3. Exact Failure Clusters

All `113` non-PASS outcomes (`38 REVIEW + 75 FAIL`) are concentrated in exactly six
stable cases. Aggregating the report shows each one routes to the same wrong window on
every observed repetition.

### 3.1 Stable Non-PASS Clusters

| Cluster | Verdict count | Expected clause window(s) | Actual routed window(s) | Evidence | Root-cause class |
|---|---:|---|---|---|---|
| `accessible_definition` | `19 FAIL` | `1.4.2 -> window 5` (`WM:47-54`) | `window 76 -> 1.4.73 Insulation system` | `VR:28-57` | Exact clause-id/title lookup miss. |
| `accessible_vs_readily` | `19 FAIL` | `1.4.2 -> 5`, `1.4.3 -> 6` (`WM:47-63`) | `window 46 -> 1.4.43 Current, short-circuit` | `VR:91-122` | Multi-clause query unsupported under `top-k=1`; wrong lexical address. |
| `switchboard_definition` | `19 FAIL` | `1.4.121 -> window 124` (`WM:1118-1125`) | `window 492 -> 3.9.7.3 MIMS cable` | `VR:253-282` | Title/keyword collision; no exact-address route. |
| `insulation_resistance_results` | `18 FAIL` | `8.3.6.3 -> window 1190` (`WM:10712-10719`) | `window 1091 -> 7.5.6 Arrangement of PELV Circuits` | `VR:480-509` | Clause-id lookup brittle; numeric clause references are not deterministically resolved. |
| `insulated_definition` | `19 REVIEW` | `1.4.72 -> window 75` (`WM:677-684`) | `window 786 -> 5.4.1.1 Exposed Conductive Parts` | `VR:157-186` | Semantic-neighbor contamination; target term mentioned, exact definition not routed. |
| `rcd_definition` | `19 REVIEW` | `1.4.102 -> window 105` (`WM:947-954`) | `window 1162 -> 7.9.3.3 Facilities for Mode 3 and 4 Charging` | `VR:285-314` | Semantic-neighbor contamination; acronym mention retrieves a rule, not the definition. |

### 3.2 What These Clusters Prove

1. The checkpoint already contains the right clauses as single-part windows.
   Evidence: `WM` maps all six target clauses to a single exact window
   (`5`, `6`, `75`, `105`, `124`, `1190`), not to split multi-part ranges.

2. The dominant defect is routing/addressing, not boundary compatibility.
   Representative failing blocks show:
   `Response Mode: residual`, `Routing Mode: tfidf`, and
   `Residual Status: boundary and model shapes are compatible`
   (`VR:34-39`, `VR:98-103`, `VR:163-168`, `VR:291-296`, `VR:486-491`).

3. The clause-aligned chunking fix worked for the known boundary-straddle issue but
   did not solve exact-address lookup.
   `showers_and_bathrooms` routes to exact `window 841` and stays green:
   `VR:416-446`, `WM:7571-7578`.

4. Out-of-domain and adversarial refusal behavior is already green and must not be
   traded away for in-domain accuracy:
   `VR:576-730`.

### 3.3 Retrieval Failure vs Generation Failure

This distinction must stay explicit:

- Retrieval failure:
  the routed windows do not include the exact target clause window(s).
  That explains all four hard FAIL clusters.
- Mixed retrieval/grounding failure:
  the routed window contains related terminology but not the exact definitional
  clause. That explains both REVIEW clusters.
- Pure generation failure:
  exact target window is routed and the answer still fails to ground.
  Current evidence does not show a stable pure-generation cluster.

Inference from the report:
the current AUS3000 gap is primarily address resolution, not a broken residual
injection path.

---

## 4. Current Implementation Inventory

### 4.1 Clause-Aligned Builder

The builder is already doing most of the right pre-processing work:

| File | Lines | Current behavior |
|---|---|---|
| `tools/build_aus3000_clause_aligned_variant.py` | `195-246` | Normalizes bad controls, spacing, quotes, headings, bullets. |
| `tools/build_aus3000_clause_aligned_variant.py` | `287-297` | Serializes each clause with explicit `clause_title`, `standard_id`, `standard_title`, `clause_id`, `clause_content` headers. |
| `tools/build_aus3000_clause_aligned_variant.py` | `398-440` | Makes each clause its own retrieval unit when it fits the window budget; only long clauses are split. |
| `tools/build_aus3000_clause_aligned_variant.py` | `344-395` | Builds metadata keywords and alias variants from clause ID and clause title. |
| `tools/build_aus3000_clause_aligned_variant.py` | `474-493` | Injects alias tokens and keyword variants into each window's routing token set. |
| `tools/build_aus3000_clause_aligned_variant.py` | `614-625` | Persists `window_metadata.json` containing `clause_id`, `clause_title`, source file, and part metadata. |
| `tools/build_aus3000_clause_aligned_variant.py` | `627-640` | Persists manifest with `clause_aligned: true` and `window_metadata` pointer. |
| `tools/build_aus3000_clause_aligned_variant.py` | `645-727` | Persists `torch_prefill.json` with record counts, split clause counts, overlap, and artifact paths. |

Important conclusion:
the builder already emits the metadata needed for deterministic clause routing. The
runtime simply does not use it.

### 4.2 Live Router / Store / Query Path

| File | Lines | Current behavior | AUS3000 implication |
|---|---|---|---|
| `src/chuk_lazarus/inference/context/knowledge/route.py` | `94-139` | `TFIDFRouter` scores bag-of-token overlap using IDF weights. No phrase order, no exact metadata path, no clause-awareness. | Alias tokens help recall but cannot guarantee exact clause address. |
| `src/chuk_lazarus/inference/context/knowledge/torch_store.py` | `104-145` | `load()` reads window tokens, token lists, IDF, keywords, manifest. It does not load `window_metadata.json` or any clause index. | Builder metadata is currently dead weight at query time. |
| `src/chuk_lazarus/inference/context/knowledge/torch_store.py` | `146-159` | `route(method="auto")` tries TF-IDF first and falls back to keyword only when TF-IDF returns `None`. | A wrong TF-IDF hit prevents rescue by keyword routing. |
| `src/chuk_lazarus/inference/context/knowledge/torch_store.py` | `161-195` | `route_top_k()` uses TF-IDF and only merges expansion-based results if base results are fewer than `k`. | With report config `top-k=1`, expansion cannot replace a wrong first hit. |
| `src/chuk_lazarus/inference/context/knowledge/torch_query.py` | `213-239` | `_expand_query()` generates "distinctive words and phrases" and tokenizes them into expansion IDs. | In the current report, expansion never surfaces real AUS3000 terms. |
| `src/chuk_lazarus/inference/context/knowledge/torch_query.py` | `255-379` | `_prepare_store_response()` expands, routes, renders prompt with chat template, then uses routed windows for replay/residual. | The first routing decision dominates the answer path. |
| `src/chuk_lazarus/inference/context/knowledge/torch_query.py` | `159-199` | `_render_prompt()` uses the tokenizer chat template when available. | This behavior must be preserved. |
| `src/chuk_lazarus/cli/commands/context/generate/_torch.py` | `206-345` | `_run_torch_store_generate()` can either delegate to `_prepare_store_response()` or do its own `route_top_k()` / `route(..., method="auto")` selection in `--no-chat-template` mode. | Exact-address routing must cover both the knowledge-query path and the torch generate path or the behaviors will diverge. |

### 4.3 Evaluator Surfaces That Matter

| File | Lines | Current behavior | Why it matters |
|---|---|---|---|
| `tools/evaluate_aus3000_variant.py` | `269-432` | Defines the exact 23-case suite. | This is the benchmark contract until the benchmark doc supersedes it. |
| `tools/evaluate_aus3000_variant.py` | `484-493` | `routed_match()` checks clause ID or title substring presence in routed text. | This can over-credit semantically related wrong windows. |
| `tools/evaluate_aus3000_variant.py` | `496-540` | PASS / REVIEW / FAIL thresholds. | REVIEW currently mixes exact hits with semantic-neighbor mentions. |
| `tools/evaluate_aus3000_variant.py` | `620-709` | Uses routed windows to score and writes the summary. | Epic 2 changes must stay benchmark-verifiable here. |

### 4.4 Query Expansion Is Currently Not Helping

The report shows every `Expansion Terms:` line contains only prompt-template boilerplate
such as `the`, `specific`, `distinctive`, `words`, `phrases`, `from`, `this`,
`event`, `are`, `particular`, `is` (`VR:37`, `VR:69`, `VR:101`, `VR:166`, and the
same pattern throughout the report).

Derived from the full report:

- `423/423` expansions contain only boilerplate prompt words.
- `0/423` contain any AUS3000-specific terms or clause IDs.

That means:

- query expansion is currently inert for AUS3000 recovery work; and
- because the report ran with `top-k=1`, even a useful expansion would not rescue
  a wrong first TF-IDF hit due to `torch_store.py:177-195`.

---

## 5. Apollo-Derived Lessons That Transfer

The AUS3000 work should borrow Apollo lessons selectively, not mechanically.

| Apollo source | Local evidence | Transfer to AUS3000 |
|---|---|---|
| `docs/SPEC_V7.md:23-24, 100-139` | Apollo accuracy improved when sampling captured content-bearing units rather than arbitrary intervals. | Already reflected in clause-aligned windowing. Keep exact content-aware units; do not go back to larger blind windows. |
| `docs/SPEC_V7.md:145-149` | Chat template activates the retrieval circuit. | Preserve chat-template use in `torch_query.py:184-190`. Do not "optimize" it away during AUS3000 fixes. |
| `docs/context/routing/kv_route.md:8-15` and `docs/SPEC.md:420, 625-636` | Apollo fast factual path works when the same circuit that retrieves a fact also addresses it. | AUS3000 needs an exact address layer. Clause ID and title are natural address keys; use them before generic overlap routing. |
| `docs/context/prefill/vec_inject.md:109-126` | Inject-matched-only is critical; injecting many facts is unstable. | If AUS3000 later adds a `kvectors_full` or `vec_inject`-style exact factual path, it must still select exact clause(s) first and replay/inject only those windows. |

### 5.1 What Does Not Transfer Directly

Apollo's H4/Q.K exact-factual path is not the first AUS3000 move.

Reason:

- AUS3000 already has explicit clause IDs and exact titles.
- The current failures are explainable by address resolution before they are
  explainable by deep model-routing limitations.
- When the right clause window is hit, the current residual path already answers
  correctly on many cases (`VR:62-89`, `VR:191-218`, `VR:221-250`, `VR:416-446`,
  `VR:447-478`, `VR:512-542`).

Therefore the Apollo lesson is not "jump straight to H4 routing." The transferable
lesson is "build an exact address path first, and only escalate to heavier factual
indexing if the simple address path cannot close the benchmark."

---

## 6. Architecture Options

### 6.1 Option A - Deterministic Clause Pre-Router

Summary:
load `window_metadata.json` into the runtime, build exact lookup indexes for clause ID
and normalized clause-title aliases, and use that layer before TF-IDF.

Target behavior:

```python
def route_aus3000_query(question: str, top_k: int) -> list[int]:
    clause_ids = extract_clause_ids(question)
    if clause_ids:
        return exact_windows_for_clause_ids(clause_ids)

    title_hits = exact_windows_for_title_aliases(question)
    if title_hits:
        return title_hits[:top_k]

    return tfidf_backstop(question, top_k=top_k)
```

Pros:

- Smallest change set.
- Directly addresses all four hard FAIL clusters.
- Makes multi-clause comparisons deterministic by returning both exact clause windows.
- Keeps non-AUS3000 path almost unchanged if gated on presence of clause metadata.

Cons:

- AUS3000-specific logic has to live somewhere explicit.
- Title-only prompts still need careful alias normalization to avoid false positives.

Expected file surfaces:

- `src/chuk_lazarus/inference/context/knowledge/torch_store.py`
- `src/chuk_lazarus/inference/context/knowledge/route.py`
- likely a new helper module such as
  `src/chuk_lazarus/inference/context/knowledge/aus3000_clause_route.py`

### 6.2 Option B - Hybrid Metadata-Aware Router

Summary:
keep TF-IDF, but add clause metadata as a first-class score source rather than a hard
bypass. Exact clause-id hits win immediately; title alias hits become strong priors;
TF-IDF remains the backstop.

Possible scoring sketch:

```python
score = (
    exact_clause_id_hit * 1_000_000
    + exact_title_alias_hit * 100_000
    + tfidf_overlap_score
)
```

Pros:

- More general than a strict bypass.
- Can still leverage content overlap for paraphrased prompts.
- May reduce maintenance burden if later generalized to other structured standards.

Cons:

- More tuning surface.
- Easier to accidentally introduce regressions in non-AUS3000 stores.
- Less transparent than a deterministic pre-router.

Expected file surfaces:

- same as Option A, plus possible evaluator changes if confidence tiers are exposed

### 6.3 Option C - Apollo-Style Exact Factual Path

Summary:
add an exact factual fast path for AUS3000 definitions/lookups, likely backed by
`kvectors_full`, `vec_inject`, or a clause-scoped exact-address index, then replay or
inject only the matched clause window(s).

Pros:

- Strong long-term architecture if clause-title/address routing alone is insufficient.
- Aligns with Apollo's "exact address, matched-only" model.

Cons:

- Highest implementation and validation cost.
- Not yet justified by the present evidence.
- Risks expanding scope before deterministic metadata routing has been exhausted.

Expected file surfaces:

- `tools/build_aus3000_clause_aligned_variant.py`
- `src/chuk_lazarus/inference/context/knowledge/torch_store.py`
- `src/chuk_lazarus/inference/context/knowledge/torch_query.py`
- additional exact-factual index loaders/builders

### 6.4 Recommendation

Recommended order:

1. Option A first.
2. Option B only if A needs extra paraphrase tolerance.
3. Option C only if A+B still fail to reach benchmark-green.

---

## 7. Recommended Target Architecture

### 7.1 Routing Contract

For AUS3000 clause-addressable prompts, routing must become deterministic.

Required precedence:

1. Exact clause ID extraction from the user prompt.
2. Exact clause-title alias matching against `window_metadata.json`.
3. TF-IDF backstop for non-addressable paraphrase prompts.
4. Keyword fallback only when TF-IDF has no useful hit.

This exact-address metadata path is the recommended first move. Apollo-style
`kvectors_full` or other heavier factual indexing is out of scope unless the
metadata-first route still fails the benchmark.

### 7.2 Multi-Clause Queries

`accessible_vs_readily` proves the current `top-k=1` configuration is incompatible
with comparison prompts (`VR:12`, `VR:91-122`).

Required behavior:

- If the query names two or more clause IDs, the route result must include all exact
  matched windows in clause order.
- The benchmark contract should evaluate comparison prompts with
  `effective_top_k = len(primary_clause_ids)`, even if the generic caller default is
  `top-k=1`.
- The query path must not truncate exact matches to one window just because generic
  routing defaulted to `top-k=1`.

### 7.3 Metadata Index Shape

Minimum metadata index fields:

- `clause_id -> [window_ids]`
- normalized `clause_title -> [window_ids]`
- alias variants such as:
  - `clause 1.4.2`
  - `1.4.2`
  - `accessible`
  - `accessible readily`
  - `residual current device`
  - `rcd`

The builder already emits enough raw information to construct this at load time
(`tools/build_aus3000_clause_aligned_variant.py:614-625`). No new external package is
required for the initial implementation.

### 7.4 Grounding Discipline

Apollo's inject-matched-only lesson transfers directly here:

- once exact clause windows are selected, only those windows should be replayed or
  used for residual injection;
- do not widen context to semantically related windows unless the benchmark category
  explicitly requires it.

This is especially important for:

- `insulated_definition`
- `rcd_definition`
- any future acronym-heavy definition prompts

### 7.5 Benchmark Gates Imported From `03-benchmark-definition.md`

The benchmark-definition document is authoritative for counted validation. Epic 2
implementation should treat the current evaluator and 30-minute report as baseline
evidence, then prove the named gates in this order:

1. `store_evidence_gate`
2. `single_pass_gate`
3. `soak_gate`

Within `single_pass_gate`, the implementation must satisfy:

- `route_gate`
- `grounding_gate`
- `ood_gate`
- `no_regression_gate`

This closes the current gap where `routed_match()`
(`tools/evaluate_aus3000_variant.py:484-493`) can over-credit semantically related
wrong windows. The baseline report remains useful for comparison, but it is not the
final counted contract.

---

## 8. Projected Implementation File Surfaces

These are the files later coding work is expected to touch, with rationale.

| File | Why it is in scope |
|---|---|
| `src/chuk_lazarus/inference/context/knowledge/torch_store.py` | Load and expose `window_metadata.json`; add exact clause metadata indexes; adjust `route_top_k()` behavior for exact hits. |
| `src/chuk_lazarus/inference/context/knowledge/route.py` | Add clause ID extraction, normalization helpers, alias normalization, and possibly a metadata-aware router class. |
| `src/chuk_lazarus/inference/context/knowledge/torch_query.py` | Honor exact route results, preserve chat-template behavior, and avoid expanding/replaying unrelated windows. |
| `src/chuk_lazarus/cli/commands/context/generate/_torch.py` | Keep torch checkpoint `context generate` aligned with the same exact-address routing contract, including the `--no-chat-template` auto-routing branch. |
| `tools/build_aus3000_clause_aligned_variant.py` | Only if runtime needs an explicit precomputed alias index rather than reconstructing from `window_metadata.json`. |
| `tools/evaluate_aus3000_variant.py` | `AUS-WS-1` must harden the scorer and command surface so counted runs use the canonical clause-aligned checkpoint/store and the named gates from the benchmark definition. |
| `docs/aus3000_accuracy_program/02-workstreams.md` | Parallel work decomposition for implementation. |
| `docs/aus3000_accuracy_program/03-benchmark-definition.md` | Explicit gold set, hard-fail rules, and regression gates. |

No new third-party dependency is required for Option A or B. Existing dependencies
(`torch`, `transformers`, `numpy`, local Lazarus runtime code) are sufficient.

---

## 9. Regression Surfaces

Any Epic 2 implementation must preserve the named regression gates imported from the
benchmark definition:

1. `route_gate`
   every in-domain gold case must route to the exact primary clause window(s) defined
   by the benchmark contract.
2. `grounding_gate`
   every in-domain gold case must answer from the routed clause window(s) without
   drifting into semantically related but wrong material.
3. `ood_gate`
   `capital_of_france`, `ocean_haiku`, `simple_math`, `recipe_request`,
   `ignore_store_fifa`, and `sql_definition` must remain PASS with explicit
   insufficiency and zero electrical bleed.
4. `no_regression_gate`
   the current stable PASS set must remain PASS:
   `accessible_readily_definition`, `competent_person`, `ev_definition`,
   `men_definition`, `rcd_not_sole_basic_protection`, `rcd_live_conductor_faults`,
   `domestic_residential_rcds_au`, `showers_and_bathrooms`,
   `periodic_inspection_testing`, `efli_when_required`, `operation_of_rcds`,
   `capital_of_france`, `ocean_haiku`, `simple_math`, `recipe_request`,
   `ignore_store_fifa`, `sql_definition`
5. Generic knowledge-store path
   non-AUS3000 stores that do not have clause metadata must continue to use the
   current TF-IDF / keyword behavior.
6. Plain model path
   the no-store torch query path in `torch_query.py:397-422` must remain unchanged.
7. Base model integrity
   no base Gemma model weight changes and no overwrite of existing checkpoints.

---

## 10. Required Tests

### 10.1 Existing Test Seams To Extend

The current tree already provides these grounded test surfaces:

- `tests/inference/context/test_torch_store.py`
  - extend for `window_metadata.json` load behavior, exact route precedence over
    TF-IDF, exact multi-window returns for comparison prompts, and safe fallback when
    metadata is absent
- `tests/inference/context/test_torch_query_helpers.py`
  - extend for query normalization, helper-level clause-id extraction, and
    comparison-prompt handling where helper coverage fits
- `tests/cli/commands/context/generate/test_cmd_torch.py`
  - extend for torch checkpoint `context generate` behavior under exact-route hits and
    `--no-chat-template` store-routing branches
- `tests/cli/commands/knowledge/test__query_backend.py`
  - extend for knowledge-query grounding behavior against exact clause routes
- `tests/cli/commands/knowledge/test__chat_backend.py`
  - extend for knowledge-chat behavior and refusal preservation

### 10.2 New AUS3000-Specific Tests

New tests should be explicit new files rather than implied existing seams:

- new `tests/inference/context/test_aus3000_clause_route.py`
  - clause-id extraction from prompts such as `1.4.2` and `clause 8.3.6.3`
  - normalization of punctuation/casing for `Accessible, readily`
  - acronym/title alias handling for `RCD`
  - multi-clause extraction from comparison prompts
- new `tests/inference/context/test_aus3000_torch_query.py`
  - exact route results are replayed without unrelated window widening
  - comparison prompts use all exact matched clause windows
  - out-of-domain refusal remains explicit with zero electrical bleed
- new `tests/tools/test_evaluate_aus3000_variant.py`
  - evaluator continues to parse and score the 23-case suite
  - exact metadata-backed route assertions behave as intended
  - strict PASS/FAIL-only gate math remains correct
- new `tests/tools/test_build_aus3000_clause_aligned_variant.py`
  - builder continues to emit `window_metadata.json`
  - alias metadata is present and deterministic if additive builder work is needed
  - split-clause metadata remains correct

### 10.3 Integration Expectations

Minimum integration coverage for Epic 2:

- exact clause retrieval for `accessible_definition`, `switchboard_definition`, and
  `insulation_resistance_results`
- exact multi-clause retrieval for `accessible_vs_readily`
- semantic-neighbor disambiguation for `insulated_definition` and `rcd_definition`
- preservation checks for `showers_and_bathrooms`, the out-of-domain refusal cases,
  and the adversarial ignore-store cases

### 10.4 Validation Order

Required validation order for the implementation wave:

1. Targeted narrow tests for the routing and grounding helpers.
2. Existing-seam and AUS3000-specific integration tests.
3. `store_evidence_gate` against the canonical clause-aligned store.
4. `single_pass_gate`, proving `route_gate`, `grounding_gate`, `ood_gate`, and
   `no_regression_gate` together on the canonical checkpoint/store.
5. `soak_gate` after single-pass is green, using the same canonical checkpoint/store
   and strict PASS/FAIL-only scoring.

---

## 11. Open Questions

1. Should AUS3000 exact clause routing live as a generic structured-store feature in
   `torch_store.py`, or as an AUS3000-specific helper gated by manifest metadata such
   as `clause_aligned: true`?
2. Should the runtime build the clause alias index at load time from
   `window_metadata.json`, or should the builder emit a dedicated additive alias-index
   artifact for faster startup and a tighter contract?
3. If deterministic metadata-first clause routing resolves all six stable misses, do
   we stop there, or still invest in an Apollo-style exact factual path for future
   scale/performance headroom?

---

## 12. Ranked Implementation Plan

1. Add deterministic clause-address routing using `window_metadata.json`, with clause ID
   extraction, title alias normalization, and exact-window returns ahead of TF-IDF.
2. Add explicit support for multi-clause queries so prompts like
   `accessible_vs_readily` can return both exact clause windows without being truncated
   by the generic `top-k=1` path.
3. Keep TF-IDF as the backstop for non-addressable paraphrase prompts, but never let a
   weak TF-IDF hit override an exact clause-id or exact-title hit.
4. Implement the named benchmark gates so `store_evidence_gate` passes first, then
   `single_pass_gate` proves `route_gate`, `grounding_gate`, `ood_gate`, and
   `no_regression_gate` on the canonical checkpoint/store.
5. Run `soak_gate` only after single-pass is green, and only consider heavier
   Apollo-style exact factual indexing if the deterministic metadata-first route still
   misses benchmark-green.
