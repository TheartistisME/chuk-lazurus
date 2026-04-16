# JSON to Store

This is the practical workflow for turning a directory of JSON files into a Lazarus torch knowledge store.

It is based on the TradeGuru -> Gemma 4 -> TorchKnowledgeStore flow that was built and validated in this repo.

## What Lazarus Actually Wants

Lazarus `knowledge build` does **not** ingest a JSON directory.

It wants:

1. A single plain-text UTF-8 corpus file
2. A model path or model id
3. An output directory for the built store

The correct pipeline is:

```text
JSON files
-> deterministic cleaning + extraction
-> one plain-text corpus file
-> token count / budget check
-> lazarus knowledge build
-> knowledge store directory
-> knowledge query + corpus validation
```

## Read These Files First

These are the files that define the real behavior, not guesses:

- [tools/build_tradeguru_corpus.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/tools/build_tradeguru_corpus.py)
  Generic JSON -> corpus converter we created from this workflow.
- [src/chuk_lazarus/cli/_parsers/_knowledge.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/src/chuk_lazarus/cli/_parsers/_knowledge.py)
  Exact supported CLI flags for `knowledge build`, `knowledge query`, and `knowledge chat`.
- [src/chuk_lazarus/cli/commands/knowledge/_build.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/src/chuk_lazarus/cli/commands/knowledge/_build.py)
  Shows that the CLI reads a text file, tokenizes it, and builds windows from raw text.
- [src/chuk_lazarus/inference/context/knowledge/torch_build.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/src/chuk_lazarus/inference/context/knowledge/torch_build.py)
  The torch store builder. This is the store layout truth.
- [src/chuk_lazarus/inference/context/knowledge/torch_store.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/src/chuk_lazarus/inference/context/knowledge/torch_store.py)
  The loader/routing contract for the built store.
- [examples/inference/build_knowledge_store_torch.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/examples/inference/build_knowledge_store_torch.py)
  Narrow example of a direct torch-native build.
- [examples/inference/demo_c_apollo11_torch.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/examples/inference/demo_c_apollo11_torch.py)
  Apollo torch query reference. Useful for understanding how the store is used at inference time.
- [docs/SPEC_V7.md](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/docs/SPEC_V7.md:236)
  Short CLI examples and expected command shape.

## Core Rule

Do **not** spend time hand-editing JSON files unless the schema is completely broken.

Write a deterministic converter script and rerun it.

## Minimal Methodology

The method that worked well here was:

1. Inspect a sample of JSON files and confirm the real schema.
2. Decide which fields matter to the final corpus.
3. Normalize text noise without destroying meaning.
4. Emit one deterministic corpus text file in natural filename order.
5. Count tokens with the target model tokenizer.
6. Build the store.
7. Query the store.
8. Validate the answer against the corpus text with `rg`.

## What to Keep from JSON

Keep only fields that improve retrieval quality.

For TradeGuru, that was:

- `Title`
- `Info`

That was enough. Keeping every key would have wasted tokens and weakened the signal.

If your JSON has fields like `body`, `content`, `summary`, `notes`, or `steps`, treat those like `Info`.

If your JSON has IDs, timestamps, UUIDs, or database metadata, do **not** include them unless they are query-relevant.

## Cleaning Rules

The cleaning rules that worked well:

- read JSON as `utf-8-sig`
- preserve semantic content
- normalize line endings to `\n`
- remove BOM, zero-width chars, and invalid control characters
- replace NBSP with a normal space
- normalize smart quotes to ASCII quotes
- normalize `—` and `–` to plain text separators
- normalize `…` to `...`
- convert `☐` to `[ ]`
- convert `☑` and `☒` to `[x]`
- strip markdown heading markers like `##`
- normalize bullets like `•`, `○`, `▪` to `- `
- flatten simple pipe tables into plain text rows
- collapse repeated blank lines
- preserve real technical symbols when they carry meaning

Do **not** over-clean.

Examples:

- Keep electrical symbols if they are part of a diagram explanation.
- Keep numbered steps.
- Keep headings if they improve retrieval.
- Keep short structural spacing.

## Recommended Corpus Shape

Use compact plain text unless you have a strong reason not to.

This worked well:

```text
Title
Normalized body text

Next title
Normalized body text
```

That shape is cheap in tokens and still gives the store enough structure for routing.

Do **not** wrap every record in verbose JSON-like markers unless you actually need them.

## The Converter Script

Use [tools/build_tradeguru_corpus.py](/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/tools/build_tradeguru_corpus.py).

It supports:

- `--input-dir`
- `--output`
- `--manifest`
- `--record-format compact|labeled|tagged`
- `--model`
- `--max-tokens`
- `--quarantine-dir`
- `--allow-failures`

### Example: generic run

```bash
python tools/build_tradeguru_corpus.py \
  --input-dir /path/to/json \
  --output /path/to/corpus.txt \
  --manifest /path/to/corpus_manifest.json
```

### Example: with tokenizer counting against the target model

```bash
python tools/build_tradeguru_corpus.py \
  --input-dir /path/to/json \
  --output /path/to/corpus.txt \
  --manifest /path/to/corpus_manifest.json \
  --model /path/to/local/model
```

### Example: enforce a token ceiling

```bash
python tools/build_tradeguru_corpus.py \
  --input-dir /path/to/json \
  --output /path/to/corpus.txt \
  --manifest /path/to/corpus_manifest.json \
  --model /path/to/local/model \
  --max-tokens 30000
```

### Example: quarantine bad files

```bash
python tools/build_tradeguru_corpus.py \
  --input-dir /path/to/json \
  --output /path/to/corpus.txt \
  --manifest /path/to/corpus_manifest.json \
  --quarantine-dir /path/to/json_quarantine
```

## Validate the Corpus Before Building

### Check that the file exists and is non-empty

```bash
ls -lh /path/to/corpus.txt
```

### Search for obvious noise

```bash
rg -n "☐|☑|☒|##|<br>|\\ufeff|\\u200b" /path/to/corpus.txt
```

### Spot-check important sections

```bash
rg -n -C 3 "Safety Checks|lockout|tagout|PPE|Verify the Absence of Voltage" /path/to/corpus.txt
```

### Compile the script if you changed it

```bash
python -m py_compile tools/build_tradeguru_corpus.py
```

## Exact Build Command

The real supported flags for `knowledge build` are:

- `--model`
- `--input`
- `--output`
- `--window-size`
- `--entries-per-window`
- `--max-tokens`
- `--backend`
- `--device`

### Generic torch build

```bash
CHUK_BACKEND=torch uv run chuk-lazarus knowledge build \
  --model /path/to/local/model \
  --input /path/to/corpus.txt \
  --output /path/to/store \
  --backend torch \
  --device cuda
```

### The exact TradeGuru build used here

```bash
CHUK_BACKEND=torch uv run chuk-lazarus knowledge build \
  --model /home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf \
  --input /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_lazarus_corpus.txt \
  --output /tmp/tradeguru_store \
  --backend torch \
  --device cuda
```

### Optional tuning flags

```bash
CHUK_BACKEND=torch uv run chuk-lazarus knowledge build \
  --model /path/to/local/model \
  --input /path/to/corpus.txt \
  --output /path/to/store \
  --window-size 512 \
  --entries-per-window 8 \
  --max-tokens 30000 \
  --backend torch \
  --device cuda
```

## What the Store Looks Like

The torch build writes a v12 store with files like:

- `manifest.json`
- `entries.npz`
- `window_tokens.npz`
- `window_token_lists.npz`
- `idf.json`
- `keywords.json`
- `boundary_residual.npy`
- `boundaries/window_000.npy`, etc.

If those files are missing, the build is not complete.

## Persist the Store Somewhere Real

Do not leave the only copy in `/tmp` if you need it later.

Copy it to a stable location:

```bash
rm -rf /path/to/persistent_store
cp -a /tmp/tradeguru_store /path/to/persistent_store
```

Example:

```bash
rm -rf /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_store
cp -a /tmp/tradeguru_store /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_store
```

## Exact Query Command

The real supported flags for `knowledge query` are:

- `--model`
- `--store`
- `--prompt`
- `--max-tokens`
- `--temperature`
- `--top-k`
- `--backend`
- `--device`

### Generic query

```bash
CHUK_BACKEND=torch uv run chuk-lazarus knowledge query \
  --model /path/to/local/model \
  --store /path/to/store \
  --prompt "Summarize the key safety checks in the training material." \
  --backend torch \
  --device cuda \
  --max-tokens 80
```

### TradeGuru example

```bash
CHUK_BACKEND=torch uv run chuk-lazarus knowledge query \
  --model /home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf \
  --store /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_store \
  --prompt "Summarize the key safety checks in the training material." \
  --backend torch \
  --device cuda \
  --max-tokens 80
```

## Validate the Store Against the Corpus

Do **not** trust the model answer blindly.

After querying:

1. inspect routing/debug output
2. search the corpus with `rg`
3. compare the answer to the exact source text

Example:

```bash
rg -n -C 3 "Safety Checks|lockout|tagout|PPE|Confirm Isolation Again|Test After Repair" \
  /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_lazarus_corpus.txt
```

If the answer mentions content you cannot find in the corpus, treat it as suspect.

## Methodology for Unknown JSON Schemas

If your JSON does not look like TradeGuru, use this approach:

1. Sample 5 to 10 files.
2. Identify the real content-bearing keys.
3. Ignore transport metadata.
4. Normalize all textual values before concatenation.
5. If the schema varies, map variants into one common `title + body` shape.
6. Quarantine parse failures instead of silently dropping them.
7. Emit a manifest with counts and failure reasons.

### Good patterns

- `{"title": "...", "content": "..."}`
- `{"name": "...", "description": "..."}`
- `{"heading": "...", "steps": ["...", "..."]}`
- `{"page_title": "...", "body_markdown": "..."}`

### Bad patterns

- dumping the entire raw JSON object into the corpus
- including IDs, timestamps, hashes, audit fields, and null-heavy metadata
- mixing different record orders between runs
- silently skipping invalid files

## Do

- do build one deterministic corpus file
- do keep only retrieval-relevant fields
- do count tokens with the target model tokenizer
- do preserve semantics while removing noise
- do keep originals untouched
- do write a manifest
- do validate both the corpus and the built store
- do persist the final store outside `/tmp`

## Do Not

- do not point `knowledge build` at a JSON directory
- do not hand-clean hundreds of files one by one
- do not flatten everything into one giant paragraph
- do not overwrite the original JSON files
- do not include junk metadata unless it helps retrieval
- do not assume a model answer is correct without corpus validation
- do not leave the only successful store build in a temporary directory

## Troubleshooting

### Build says input file not found

Your `--input` must point to the plain-text corpus file, not the JSON directory.

### Build runs but the store is empty or tiny

Check:

- corpus file is non-empty
- `manifest.json` exists
- `window_token_lists.npz` exists
- `boundaries/` contains per-window files

### Query takes a long time

That can be normal with a large local model because it reloads the weights.

Check:

```bash
ps -fp $(pgrep -f "chuk-lazarus knowledge query" | tr '\n' ' ')
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

### Query output looks wrong

Search the corpus directly:

```bash
rg -n -C 2 "your topic words here" /path/to/corpus.txt
```

If the source text does not support the answer, the answer is not validated.

## A Complete Example

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

rg -n -C 3 "Safety Checks|lockout|tagout|PPE|Confirm Isolation Again|Test After Repair" \
  /mnt/c/Users/jehma/Desktop/TradeGuru/tradeguru-agent-training/tradeguru_lazarus_corpus.txt
```

## Final Rule

When moving from JSON to store, optimize for:

- semantic density
- deterministic output
- token efficiency
- reproducibility
- source-text validation

That is the difference between a corpus that merely builds and a corpus that actually routes well.
