# ARCHITECTURE — canonical parity, Gemma-4 adapted

## The parity claim

This port gives our turn-aligned conversational memory **architectural parity** with the canonical `chrishayuk/chuk-lazurus` knowledge-store implementation. "Parity" here means: at the residual-stream / KV-cache level, our runtime produces the same anchoring effect the canonical MLX implementation does, so layers 30-47 of Gemma-4 attend to a coherent, donor-anchored KV cache during generation.

**Parity does NOT mean implementation identity.** The canonical implementation targets MLX and uses `inputs_embeds`-style prepending. Gemma-4 on PyTorch-HF has a deterministic OOM trap on that code path (explained below), so we preserve the *semantic* of canonical prefill via a different *mechanism* on the optimised `input_ids` path. The property we preserve:

> The donor residual seeds **position 0** of the prefill. Real prompt tokens flow through all layers and build a KV cache anchored on the donor. Every real-prompt position attends to position 0 through every layer via the causal mask.

That property is what makes the generated output coherent rather than token-salad.

## Canonical mechanism (what we're mirroring)

Source: `chrishayuk/chuk-lazurus` — `src/chuk_lazarus/inference/context/kv_generator.py:182-226`, `inject.py`, `cli/commands/knowledge/_query.py`. Offline copies in `prod/canonical/`.

1. Decode matched window tokens back to text.
2. Chat-template `[window_text + question]` → long prompt (~600+ tokens).
3. **Prefill the entire prompt** with boundary applied as `initial_residual` at the **embedding layer output** (not at the crystal layer) via `prefill_to_layer` + `prefill_from_layer`.
4. Boundary residual seeds **position 0**; real prompt tokens flow through and build a coherent KV cache anchored on the donor.
5. Generation proceeds with optional **agreement gate** (`inject.py:267-269`) that stops replacement injection within ~10 steps when injected and natural logits agree.

**Key detail corrected during run-1:** the prior handoff's shorthand said "inject at crystal layer position 0". The canonical source shows injection is at the embedding layer output, seq-length extended by 1, flowing through ALL layers up to target_layer. The causal mask makes every real-prompt position attend to position 0 through every layer — that's the load-bearing property. The "crystal layer" wording referred to which *target* layer's final residual is the donor, not where it's injected. See `ve-ins-0mo8bdqfg0000f54b22` (canonical-port capstone learning-pattern).

## Gemma-4 adaptation

### The trap: `get_per_layer_inputs` broadcast

`transformers/models/gemma4/modeling_gemma4.py:1635` contains:

```python
inputs_embeds[:, :, None, :] == self.embed_tokens.weight[None, None, :, :] * self.config.hidden_size**0.5
```

This broadcasts to shape `(B, S, V, H)` where V=~256000, H=8192. On a 600-token prompt that's **~253 GiB**. Blows any 32 GiB GPU deterministically. Any `model.generate(inputs_embeds=...)` call on Gemma-4 OOMs.

This is a "fallback for debuggers" path meant to recover per-layer input projections via full-vocab broadcast comparison instead of a cosine-match. The `input_ids` path bypasses it entirely.

**Evidence:** `prod/validation/VERDICT.md`, `prod/validation/04-direct-probe.log` (3 probes, 253 / 255 / 257 GiB OOM-killed).

### The workaround: seed-token + `forward_pre_hook`

Stay on Gemma-4's optimised `input_ids` path. Inject the donor residual at the post-embedding hidden state of position 0 via a hook:

1. Tokenise prompt → `input_ids` shape `(1, S)`
2. Prepend a **seed token** (BOS / pad / 0 — id is irrelevant, hook overwrites it) → `seeded_input_ids` shape `(1, S+1)`
3. `register_forward_pre_hook` on `layers[0]`:
   - On **prefill** (`hidden_states.shape[1] > 1`): overwrite `hidden_states[:, 0, :]` with the boundary residual
   - On **decode** (`hidden_states.shape[1] == 1`): **no-op and do not latch** (the hook must remain re-entry-safe across KV-cached decode steps)
4. `model.generate(input_ids=seeded_input_ids, attention_mask=ones(1, S+1), use_cache=True, ...)` — standard fast path
5. Slice generated output past position `S+1`

**Result:** layers 30-47 attend to a KV cache built from real prompt tokens with the donor seeded at position 0. Coherent English on convo content, no OOM, no regression on AUS3000 strict assertions.

**Implementation:** `src/chuk_lazarus/inference/backends/torch_runtime.py:307-472` (`generate_with_residual_prefill_seeded`). The old `generate_with_residual` method is **preserved unchanged** at `torch_runtime.py:254` for any other callers.

## Why parity (not identity)

We could have stayed on the `inputs_embeds` path with a patched `get_per_layer_inputs` (e.g. `_gemma_patches.py`'s `patch_clippable_linear` + friends), but that creates a forked modeling file that must be maintained against upstream Gemma-4 updates. The hook approach:

- Works on vanilla `transformers` (any Gemma-4 release that keeps the `input_ids` path)
- Has no OOM hazard by construction (no vocab-wide broadcast)
- Preserves the canonical architectural property (donor at position 0, attend via causal mask)
- Keeps the `inputs_embeds` code path usable for other models that don't have the broadcast trap

If a future Gemma release replaces `get_per_layer_inputs` with a cosine-match or `torch.isclose()` implementation, the `inputs_embeds` path becomes viable again — at that point the hook approach becomes optional rather than required. The learning-pattern record (`ve-ins-0mo8bdqfg0000f54b22` §b) documents this re-evaluation trigger.

## Decode-step invariant

The `forward_pre_hook` MUST be **shape-gated**: prefill (shape[1] > 1) triggers the replacement; decode (shape[1] == 1) is a no-op. Latching on decode would:

- corrupt the KV cache's first-token projection mid-generation
- cause drift from prompt-conditioning back toward the donor content
- regress coherence at long-context generation

There is no `injected_once` latch in the final implementation — the shape gate is sufficient because generation-time calls into `layers[0]` always have shape[1] == 1 after prefill. See `torch_runtime.py:307-472` for the exact implementation.

## Strict assertions (6)

Every retriever query reports:

| Assertion | Meaning |
|-----------|---------|
| `cuda_available` | `torch.cuda.is_available()` |
| `model_on_cuda` | model params moved to CUDA |
| `residual_compatible` | donor-residual dtype/shape matches model residual stream |
| `hook_fired` | spy hook on `layers[crystal_layer]` observed the forward pass |
| `gpu_memory_grew` | allocated memory increased during generate (sanity: generation actually ran) |
| `store_window_nonempty` | the routed window has non-zero token count |

All six must be `True` for the retriever's result to be trusted. In run-1 verification they were 6/6 `True` on every probe; see `prod/validation/direct_probe_results.json` and `prod/validation/criterion4_report.json`.

## Verification summary (run-1)

| Criterion | Evidence | Status |
|-----------|----------|--------|
| 1. Apollo demo ≥1/3 coherent | `08-apollo-q{1,2,3}-round3.log` | **3/3 PASS** |
| 2. AUS3000 strict-asserts no regression | `06-pytest-round3.log` | **45/45 PASS, 6/6 strict** |
| 3. Multi-probe verbatim + coherent | `07-direct-probe-round3.log`, `direct_probe_results.json` | **3/3 verbatim** |
| 4. Chat_loop → session_close → fresh retrieval | `09-criterion4-e2e.log`, `CRITERION4_VERDICT.md`, `criterion4_report.json` | **PASS** |

See `prod/CHANGELOG.md` for the run-1 delivery summary.

## What's NOT in this port

(Explicitly out-of-scope for run-1; see `prod/CHANGELOG.md` §Follow-ups)

- BM25 routing
- Vee workspace embed-queue improvements
- Generalisation beyond Gemma-4-E2B-it
- Architectural rewrite of `capture_window_boundaries`
- Residual-only recall test (tighter criterion-4 without window echo)
- Fix for sporadic 1-of-5 `build_clause_aligned_store.py` failure (zero-mod primitive)
