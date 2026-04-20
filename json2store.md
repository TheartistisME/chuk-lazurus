# JSON to Store

This is the practical workflow for turning a directory of JSON files into a Lazarus torch knowledge store.

Two pipelines live in this repo. The **clause-aligned** pipeline is the primary path: each JSON file becomes its own retrieval window with stable ID-based routing. The **flat-corpus** pipeline is the fallback for heterogeneous or ID-less JSON, producing one concatenated text corpus. Read the next section before choosing.

## When to use which

### Clause-aligned (PRIMARY)

Use when:

- Every JSON file represents a single addressable unit with a stable ID (a clause, a chapter, a FAQ entry, a product SKU, a regulation paragraph, etc.).
- Your queries will often ask about a specific ID (e.g. "what does clause 1.4.72 define?", "summarise section 3.2", "what is SKU ABC-1234?").

What you get:

- Each record becomes its own retrieval window.
- Best retrieval quality for ID-based queries — exact clause-ID routing dominates over TF-IDF/learned routing.
- Per-window metadata lookup so the strict demo can route a query straight to the matching window without fuzzy search.

Requires JSON schema (all five required, per-file):

- `standard_id`
- `standard_title`
- `clause_id`
- `clause_title`
- `clause_content`

### Flat-corpus (FALLBACK)

Use when:

- The JSON is loose prose, the fields are heterogeneous, or you have no stable per-record ID.
- Queries will be topical rather than ID-exact ("summarise the safety procedure", "what does the training material say about PPE?").

What you get:

- All text is concatenated into one corpus file and chunked into 512-token windows.
- Retrieval is topical. There is no exact-ID fast path.

If this is you, skip to [Fallback: flat-corpus pipeline (heterogeneous JSON)](#fallback-flat-corpus-pipeline-heterogeneous-json).

## Primary pipeline: clause-aligned

The clause-aligned path skips the text-corpus intermediate entirely. It reads per-clause JSON, builds one retrieval window per clause (splitting only when a clause exceeds the window token budget), attaches exact-ID metadata tokens, captures residuals directly, and writes a self-contained checkpoint.

### Core files

- [tools/build_clause_aligned_store.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/tools/build_clause_aligned_store.py) — generic clause-aware builder. Primary tool.
- [examples/inference/demo_clause_aligned_strict.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/examples/inference/demo_clause_aligned_strict.py) — generic strict-mode retrieval demo. Apollo-11 residual injection, six strict runtime assertions (CUDA, model device, residual compatibility, injection-hook fired, GPU memory growth, non-empty routed window).
- [scripts/run_clause_aligned_demo.sh](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/scripts/run_clause_aligned_demo.sh) — bash wrapper that pins the Linux-native HF cache and forwards all args verbatim to the demo.

### Builder flags

Required:

- `--input-dir` — directory of per-clause JSON files.
- `--checkpoint` — output checkpoint directory; will be created. Contains `torch_prefill.json`, `clause_aligned_build_manifest.json`, and a `torch_store/` subdirectory.

Optional:

- `--model` — base Gemma model path (defaults to the locally cached `google/gemma-4-E2B-it`).
- `--window-size` — max tokens per clause-aligned window (default 512).
- `--overlap-tokens` — overlap between subchunks when a clause exceeds the window budget (default 64).
- `--entries-per-window` — minimum injection entries per window (default 8).
- `--max-keywords` — keywords stored per window (default 12).
- `--topic-expansion-tokens` — optional model-driven topic-expansion token budget added to retrieval metadata (default 0 = disabled).
- `--device` — torch device (default `cuda`).
- `--force` — delete the output checkpoint if it already exists before rebuilding.

### Demo flags

- `--store` — path to the `torch_store/` subdirectory inside a clause-aligned checkpoint.
- `--model` — HF id or local path (default `google/gemma-4-E2B-it`).
- `--device` — torch device (default `cuda`). CUDA is REQUIRED; strict mode raises if `torch.cuda.is_available()` is False.
- `--question` — the question to route + answer.
- `--max-new-tokens` — generation cap (default 120).
- `--system-prompt` — override the system prompt. If omitted, the demo derives one from `torch_prefill.json` (`source.name` / `source.standard_title`) or falls back to a generic clause-aware prompt.

### Minimal invocation

Build a brand-new clause-aligned checkpoint from a custom JSON dir:

```bash
python tools/build_clause_aligned_store.py \
  --input-dir /path/to/clause_json \
  --checkpoint /path/to/checkpoint
```

Query it via the generic demo (CUDA required):

```bash
bash scripts/run_clause_aligned_demo.sh \
  --store /path/to/checkpoint/torch_store \
  --question "What does clause 1.4.72 define?"
```

The wrapper forwards every flag verbatim, so you can pass `--model`, `--device`, `--max-new-tokens`, `--system-prompt` alongside `--store` and `--question`.

### Tuned invocation

```bash
python tools/build_clause_aligned_store.py \
  --input-dir /path/to/clause_json \
  --checkpoint /path/to/checkpoint \
  --window-size 512 \
  --overlap-tokens 64 \
  --entries-per-window 8 \
  --max-keywords 16 \
  --topic-expansion-tokens 32 \
  --device cuda \
  --force
```

## Artifacts the clause-aligned builder produces

Everything lands under `<checkpoint>/` and its `torch_store/` subdirectory. A complete build MUST contain all of these — if any are missing the build is incomplete.

### `<checkpoint>/torch_prefill.json`

Prefill sidecar metadata. Fields:

- `version` — sidecar schema version.
- `kind` — always `"torch-prefill"`.
- `status` — `"complete"` on a successful build.
- `backend` — always `"torch"`.
- `created_at` — ISO-8601 UTC timestamp.
- `model` — `{id, device}`.
- `source` — `{name, standard_id, standard_title, input_file, max_tokens}`. `name` is `"<standard_id> Clause Aligned"`; `input_file` is the JSON input dir.
- `windowing` — `{window_size, num_tokens, num_windows}`.
- `arch_config` — the serialised ArchitectureConfig (retrieval_layer, query_head, injection_layer, crystal_layer, window_size, entries_per_window, etc.).
- `artifacts` — table of every file path written into `torch_store/`, including `manifest`, `entries`, `window_tokens`, `window_token_lists`, `idf`, `keywords`, `boundaries_dir`, `boundary_residual`, `window_metadata`.
- `clause_aligned` — `{enabled, record_count, split_clause_count, overlap_tokens, max_keywords, topic_expansion_tokens}`. `enabled=true` marks this as a clause-aligned checkpoint.

### `<checkpoint>/clause_aligned_build_manifest.json`

Build-time manifest. Fields:

- `input_dir`, `checkpoint`, `store_path` — absolute paths.
- `model`, `device` — resolved model path and actual device (e.g. `cuda:0`).
- `window_size`, `overlap_tokens`, `entries_per_window`, `max_keywords`, `topic_expansion_tokens` — the exact knobs this build used.
- `record_count` — number of clause JSON files ingested.
- `stub_clause_count` — clauses whose `clause_content` was empty and fell back to `clause_title`.
- `window_count` — total windows produced (>= record_count when any clause was split).
- `split_clause_count` — clauses that exceeded the window budget and were split into subchunks.
- `split_clause_ids` — sorted list of the `clause_id` values that were split.
- `num_tokens`, `num_windows`, `num_entries` — store-level totals.

### `<checkpoint>/torch_store/manifest.json`

Store manifest loaded by `TorchKnowledgeStore.load`. Fields:

- `version` — store layout version (apollo-v12).
- `num_entries`, `num_windows`, `num_tokens` — counts.
- `entries_per_window` — resolved from ArchitectureConfig.
- `crystal_layer` — the layer residuals were captured at.
- `window_size` — per-window token budget.
- `arch_config` — full serialised architecture config.
- `has_residuals` — `false` (residuals live in the separate `.npy` artifacts).
- `window_metadata` — name of the per-window metadata file (`window_metadata.json`).
- `clause_aligned` — `true`.

### `<checkpoint>/torch_store/window_metadata.json`

Per-window lookup, keyed by string `window_id`. Each entry records:

- `clause_id` — exact ID for this window.
- `clause_title` — human-readable title.
- `source_file` — original JSON filename.
- `part_index`, `part_count` — 1-indexed position within a split clause (both `1` if unsplit).
- `token_count` — tokens in this window including the metadata header.
- `content_was_empty` — true if the original `clause_content` was blank and fell back to `clause_title`.

This file is how the strict demo routes an exact-clause-ID query to a specific window. It is what makes the clause-aligned path qualitatively better than flat-corpus for ID queries.

### `<checkpoint>/torch_store/entries.npz`

Injection entries as a structured numpy array. Columns:

- `token_id` — the target token to inject.
- `coefficient` — per-token injection coefficient.
- `window_id` — the window this entry belongs to.
- `position_in_window` — position within the window token list.
- `fact_id` — globally unique id across the store.

### Retrieval indexes

- `<checkpoint>/torch_store/window_tokens.npz` — per-window unique token sets (uint32, sorted).
- `<checkpoint>/torch_store/window_token_lists.npz` — per-window ordered token lists (uint32, insertion order).
- `<checkpoint>/torch_store/idf.json` — inverse-document-frequency scores across windows.
- `<checkpoint>/torch_store/keywords.json` — per-window keyword strings including clause-ID and title aliases.

### Captured residuals

- `<checkpoint>/torch_store/boundaries/window_000.npy`, `window_001.npy`, ... — one residual tensor per window captured from the crystal_layer.
- `<checkpoint>/torch_store/boundary_residual.npy` — the final boundary residual tensor (`shape = (1, 1, hidden)`, float32).

These are what the demo loads and injects at inference time via the forward-pre-hook on `crystal_layer`.

## Clause JSON schema

One JSON object per file. All five required fields are strings. Extra fields are currently ignored by the builder.

```json
{
  "standard_id": "AS/NZS 3000",
  "standard_title": "Electrical Installations (Wiring Rules)",
  "clause_id": "1.4.72",
  "clause_title": "Safety service",
  "clause_content": "A service intended to operate in the event of a hazard to persons, and includes emergency lighting, fire detection and alarm systems, smoke extraction fans, CO extraction fans, fire service lifts and the like.",
  "source_page": 42
}
```

Notes:

- `source_page` above is an optional extra field. The builder ignores anything beyond the five required fields for now.
- If `clause_content` is blank, the builder falls back to `clause_title` and flags the window with `content_was_empty: true` in `window_metadata.json` and counts it in `stub_clause_count` in the build manifest.
- Files are ingested in natural sort order of their filenames; files ending in `_metadata.json` are skipped.
- JSON is read as `utf-8-sig` so leading BOMs are tolerated.

## End-to-end example (clause-aligned)

Placeholder `/path/to/clause_json` holds one JSON per clause, matching the schema above. Build + query in two commands:

```bash
# Build the clause-aligned checkpoint
python tools/build_clause_aligned_store.py \
  --input-dir /path/to/clause_json \
  --checkpoint /path/to/checkpoint

# Query it with residual injection, strict mode
bash scripts/run_clause_aligned_demo.sh \
  --store /path/to/checkpoint/torch_store \
  --question "Summarise what clause 1.4.72 defines."
```

Expected checkpoint layout after build:

```text
/path/to/checkpoint/
  torch_prefill.json
  clause_aligned_build_manifest.json
  torch_store/
    manifest.json
    window_metadata.json
    entries.npz
    window_tokens.npz
    window_token_lists.npz
    idf.json
    keywords.json
    boundary_residual.npy
    boundaries/
      window_000.npy
      window_001.npy
      ...
```

Sanity checks after build:

```bash
ls -lh /path/to/checkpoint/
ls -lh /path/to/checkpoint/torch_store/
jq '.clause_aligned' /path/to/checkpoint/torch_prefill.json
jq '.record_count, .window_count, .split_clause_count' \
  /path/to/checkpoint/clause_aligned_build_manifest.json
jq 'to_entries[0]' /path/to/checkpoint/torch_store/window_metadata.json
```

Inspect the strict demo's routing line in stdout — it must log `routed via exact` for any question containing a clause-ID token like `1.4.72`. If it logs `tfidf` or `auto` for such a query, the clause-ID was not registered correctly in the metadata.

## Fallback: flat-corpus pipeline (heterogeneous JSON)

If your JSON is prose without a stable per-record ID, use this older flow. It is preserved verbatim for cases where clause-aligned is not applicable.

### What Lazarus actually wants

Lazarus `knowledge build` does **not** ingest a JSON directory.

It wants:

1. A single plain-text UTF-8 corpus file
2. A model path or model id
3. An output directory for the built store

The correct flat-corpus pipeline is:

```text
JSON files
-> deterministic cleaning + extraction
-> one plain-text corpus file
-> token count / budget check
-> lazarus knowledge build
-> knowledge store directory
-> knowledge query + corpus validation
```

### Read these files first

- [tools/build_tradeguru_corpus.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/tools/build_tradeguru_corpus.py) — generic JSON -> corpus converter.
- [src/chuk_lazarus/cli/_parsers/_knowledge.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/src/chuk_lazarus/cli/_parsers/_knowledge.py) — exact supported CLI flags for `knowledge build`, `knowledge query`, `knowledge chat`.
- [src/chuk_lazarus/cli/commands/knowledge/_build.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/src/chuk_lazarus/cli/commands/knowledge/_build.py) — CLI reads a text file, tokenizes, builds windows.
- [src/chuk_lazarus/inference/context/knowledge/torch_build.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/src/chuk_lazarus/inference/context/knowledge/torch_build.py) — store layout truth.
- [src/chuk_lazarus/inference/context/knowledge/torch_store.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/src/chuk_lazarus/inference/context/knowledge/torch_store.py) — loader/routing contract.
- [examples/inference/build_knowledge_store_torch.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/examples/inference/build_knowledge_store_torch.py) — narrow direct torch-native build example.
- [docs/SPEC_V7.md](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/docs/SPEC_V7.md:236) — short CLI examples and expected command shape.

### Methodology

1. Inspect a sample of JSON files and confirm the real schema.
2. Decide which fields matter to the final corpus.
3. Normalize text noise without destroying meaning.
4. Emit one deterministic corpus text file in natural filename order.
5. Count tokens with the target model tokenizer.
6. Build the store.
7. Query the store.
8. Validate the answer against the corpus text with `rg`.

Do **not** spend time hand-editing JSON files unless the schema is completely broken. Write a deterministic converter script and rerun it.

### What to keep from JSON

Keep only fields that improve retrieval quality. For TradeGuru that was `Title` and `Info`. If your JSON has `body`, `content`, `summary`, `notes`, or `steps`, treat those like `Info`. Do not include IDs, timestamps, UUIDs, or database metadata unless they are query-relevant.

### Cleaning rules

- Read JSON as `utf-8-sig`.
- Preserve semantic content. Normalize line endings to `\n`.
- Remove BOM, zero-width chars, invalid control characters.
- Replace NBSP with a normal space.
- Normalize smart quotes to ASCII quotes.
- Normalize em/en dashes to plain separators; ellipsis `…` to `...`.
- Convert `☐` to `[ ]`, `☑`/`☒` to `[x]`.
- Strip markdown heading markers like `##`.
- Normalize bullets (`•`, `○`, `▪`) to `- `.
- Flatten simple pipe tables to plain rows.
- Collapse repeated blank lines.
- Preserve real technical symbols that carry meaning.

Do **not** over-clean. Keep electrical symbols in diagram explanations, numbered steps, useful headings, and short structural spacing.

### Converter script

Use [tools/build_tradeguru_corpus.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/tools/build_tradeguru_corpus.py). Flags:

- `--input-dir`, `--output`, `--manifest`
- `--record-format compact|labeled|tagged`
- `--model`, `--max-tokens`
- `--quarantine-dir`, `--allow-failures`

```bash
python tools/build_tradeguru_corpus.py \
  --input-dir /path/to/json \
  --output /path/to/corpus.txt \
  --manifest /path/to/corpus_manifest.json \
  --model /path/to/local/model \
  --max-tokens 30000
```

### Validate the corpus before building

```bash
ls -lh /path/to/corpus.txt
rg -n "☐|☑|☒|##|<br>|\\ufeff|\\u200b" /path/to/corpus.txt
rg -n -C 3 "Safety Checks|lockout|tagout|PPE|Verify the Absence of Voltage" /path/to/corpus.txt
python -m py_compile tools/build_tradeguru_corpus.py
```

### Exact build command

Supported flags for `knowledge build`: `--model`, `--input`, `--output`, `--window-size`, `--entries-per-window`, `--max-tokens`, `--backend`, `--device`.

```bash
CHUK_BACKEND=torch uv run chuk-lazarus knowledge build \
  --model /path/to/local/model \
  --input /path/to/corpus.txt \
  --output /path/to/store \
  --backend torch \
  --device cuda
```

Full TradeGuru reference:

```bash
CHUK_BACKEND=torch uv run chuk-lazarus knowledge build \
  --model /home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf \
  --input /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_lazarus_corpus.txt \
  --output /tmp/tradeguru_store \
  --backend torch \
  --device cuda
```

### What the flat-corpus store looks like

The torch build writes a v12 store with:

- `manifest.json`
- `entries.npz`
- `window_tokens.npz`
- `window_token_lists.npz`
- `idf.json`
- `keywords.json`
- `boundary_residual.npy`
- `boundaries/window_000.npy`, etc.

Note: flat-corpus stores do **not** contain `window_metadata.json`, because there are no per-record IDs to route to.

### Persist the store somewhere real

```bash
rm -rf /path/to/persistent_store
cp -a /tmp/tradeguru_store /path/to/persistent_store
```

### Exact query command

Supported flags for `knowledge query`: `--model`, `--store`, `--prompt`, `--max-tokens`, `--temperature`, `--top-k`, `--backend`, `--device`.

```bash
CHUK_BACKEND=torch uv run chuk-lazarus knowledge query \
  --model /path/to/local/model \
  --store /path/to/store \
  --prompt "Summarize the key safety checks in the training material." \
  --backend torch \
  --device cuda \
  --max-tokens 80
```

### Validate the store against the corpus

Do **not** trust the model answer blindly. After querying:

1. Inspect routing/debug output.
2. Search the corpus with `rg`.
3. Compare the answer to the exact source text.

```bash
rg -n -C 3 "Safety Checks|lockout|tagout|PPE|Confirm Isolation Again|Test After Repair" \
  /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_lazarus_corpus.txt
```

If the answer mentions content you cannot find in the corpus, treat it as suspect.

### A complete flat-corpus example

```bash
python tools/build_tradeguru_corpus.py \
  --input-dir /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/json \
  --output /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_lazarus_corpus.txt \
  --manifest /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_lazarus_manifest.json \
  --model /home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf

CHUK_BACKEND=torch uv run chuk-lazarus knowledge build \
  --model /home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf \
  --input /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_lazarus_corpus.txt \
  --output /tmp/tradeguru_store \
  --backend torch \
  --device cuda

rm -rf /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_store
cp -a /tmp/tradeguru_store /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_store

CHUK_BACKEND=torch uv run chuk-lazarus knowledge query \
  --model /home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf \
  --store /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_store \
  --prompt "Summarize the key safety checks in the training material." \
  --backend torch \
  --device cuda \
  --max-tokens 80
```

## Deprecation notice

Two shims exist so older invocation paths still work. Prefer the generic scripts in new work.

- [tools/build_aus3000_clause_aligned_variant.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/tools/build_aus3000_clause_aligned_variant.py) — deprecation shim. Prints a `[DEPRECATED]` notice then delegates to `tools/build_clause_aligned_store.py`, filling in `--input-dir` and `--checkpoint` with the aus3000 defaults if not passed.
- [examples/inference/demo_c_aus3000_torch_strict.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/examples/inference/demo_c_aus3000_torch_strict.py) — deprecation shim. Prints a `[DEPRECATED]` notice then delegates to `examples/inference/demo_clause_aligned_strict.py`. The generic demo already defaults `--store` to the aus3000 clause-aligned variant, so calling the shim with no args still works.

Bash wrappers:

- [scripts/run_clause_aligned_demo.sh](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/scripts/run_clause_aligned_demo.sh) — generic wrapper. Pins `HF_HOME` to the Linux-native HF cache, pre-flights the gemma-4-E2B-it blob, then `exec`s the generic demo with all args forwarded.
- [scripts/run_aus3000_demo_fast.sh](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/scripts/run_aus3000_demo_fast.sh) — convenience wrapper. Same HF cache pinning as above; runs the generic demo with aus3000-clause-aligned defaults. A single bare positional arg is treated as `--question` for backwards compatibility with the old usage.

New work should prefer `tools/build_clause_aligned_store.py` and `scripts/run_clause_aligned_demo.sh` directly.

## Do

- Do use the clause-aligned pipeline when your JSON has a stable per-record ID.
- Do include all five required schema fields in every clause JSON file.
- Do inspect `window_metadata.json` after a clause-aligned build to confirm clause-IDs are present.
- Do verify the strict demo logs `routed via exact` for clause-ID queries.
- Do fall back to flat-corpus only when clause-aligned does not fit the data shape.
- Do build one deterministic corpus file for the flat-corpus path.
- Do keep only retrieval-relevant fields.
- Do count tokens with the target model tokenizer.
- Do preserve semantics while removing noise.
- Do write a manifest and validate both the input and the built store.
- Do persist the final store outside `/tmp`.

## Do not

- Do not run the flat-corpus pipeline when every record has a stable ID — you will lose exact-ID routing quality.
- Do not point `knowledge build` at a JSON directory; it needs a text file.
- Do not hand-clean hundreds of files one by one; write a deterministic converter.
- Do not invent clause IDs or fabricate `clause_content` stubs silently — the builder already records empty content via `content_was_empty`, and you should fix the upstream JSON instead.
- Do not include junk metadata unless it helps retrieval.
- Do not trust a model answer without source-text validation.
- Do not leave the only successful build in a temporary directory.
- Do not call the deprecation shims in new code; use the generic scripts.

## Troubleshooting

### Clause-aligned: builder says a required field is missing

Every JSON file must have non-empty `standard_id`, `standard_title`, `clause_id`, `clause_title`. Check the file flagged in the error — most commonly a stringified null or an accidentally-empty title. `clause_content` may be empty; the builder will fall back to `clause_title` and log the window with `content_was_empty: true`.

### Clause-aligned: checkpoint already exists

Pass `--force` to rebuild, or delete the checkpoint directory manually.

### Clause-aligned: strict demo fails with `routing_mode != 'exact'`

The question contains a clause-ID-like token but the routing fell through to TF-IDF. Usually means the clause ID is not in `window_metadata.json` under that exact spelling, or `_collect_exact_matches` does not recognise the format. Inspect `window_metadata.json` and verify the clause_id string matches what the question used.

### Clause-aligned: strict demo fails with `residual_is_compatible returned False`

The boundary residual hidden-size does not match the loaded model. The strict demo refuses the prompt-context fallback. Rebuild the checkpoint with the same model you are querying.

### Clause-aligned: strict demo fails with `GPU peak memory did not exceed post-load memory`

Strict mode suspects a silent CPU fallback. Confirm `--device cuda` and that `torch.cuda.is_available()` is True.

### Flat-corpus: build says input file not found

`--input` must point to the plain-text corpus file, not the JSON directory.

### Flat-corpus: build runs but the store is empty or tiny

Check the corpus file is non-empty, `manifest.json` exists, `window_token_lists.npz` exists, and `boundaries/` contains per-window files.

### Query takes a long time

Normal with a large local model because it reloads weights. Check:

```bash
ps -fp $(pgrep -f "chuk-lazarus knowledge query" | tr '\n' ' ')
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

### Query output looks wrong

For flat-corpus, search the corpus directly with `rg`. For clause-aligned, inspect the routed window text logged by the strict demo (`[3/6]` line) and cross-check against the original JSON file for that `clause_id`.

## Final rule

When moving from JSON to store, optimize for:

- ID-exact routing quality (clause-aligned) or topical retrieval (flat-corpus), not both at once
- semantic density
- deterministic output
- token efficiency
- reproducibility
- source-text validation

Pick the pipeline that matches your data shape. Clause-aligned is the default; flat-corpus is the fallback. That is the difference between a store that merely builds and a store that actually routes the question a user asked.
