# Apollo Gemma-4 Zero Injection Token Salad Fix Lesson

## TL;DR

The important clue was that **zero injection broke generation**.

That means the failure was not the injected vector's content. A zero residual
cannot add bad information. It can only break the run if the injection mechanism
itself changes the model's execution state.

The broken path was `generate_with_residual`: it injected by replacing
`hidden_states[:, -1, :]` at `crystal_layer`. That last position is the live
prompt token used to compute next-token logits. So "inject zero" really meant
"erase the final prompt token's residual at L29." Random, Apollo, AUS3000, and
zero all failed because every one of them corrupted the decode anchor.

The fix is to give the boundary its own carrier slot:

1. Prepend one seed token.
2. Run the normal prompt after it.
3. At the requested layer, replace only `hidden_states[:, 0, :]`.
4. Leave every real prompt token, especially the final token, intact.

That is the invariant:

> Boundary residuals need a dedicated carrier token. They must not overwrite the
> final real prompt token that supplies next-token logits.

## The Symptom

Run D looked like a data, boundary, or Gemma-4 architecture problem:

- Gemma-4 + AUS3000 + boundary injection worked.
- Gemma-3 + Apollo + boundary injection worked.
- Gemma-4 + Apollo + prompt context without injection worked.
- Gemma-4 + Apollo + **any** injection produced token salad.

The isolation was decisive:

- No injection: coherent answer.
- Apollo boundary: garbage.
- Known-working AUS3000 boundary: garbage.
- Zero vector: garbage.
- Random vector: garbage.

That last pair ruled out semantic content. The vector was not the poison. The
act of entering the injection path was the poison.

## What I Did First

I initially treated this like an architecture/data interaction:

- checked whether L29 was the wrong layer;
- considered L14, the KV producer, because Gemma-4 shares KV across later
  layers;
- looked at PRE/POST hook symmetry;
- considered prompt format and OCR damage;
- considered sliding-window length;
- considered chain depth;
- compared Apollo and AUS3000 boundary tensor health.

Those were reasonable probes, but they kept assuming the injected residual was
being added into a neutral place. It was not.

## Why Those Attempts Did Not Work

Every failed probe kept the same bad carrier slot.

Changing the vector did not help because the carrier was still the final real
prompt token. Changing the layer did not fully explain it because the mechanism
still replaced a live token state. Cleaning or wrapping the Apollo prompt did not
help because the mechanism still damaged the position used for the first
generated token.

That also explains the strange Apollo/AUS3000 split:

- AUS3000 happened to survive the old path in some cases because the structured
  clause prompt had a more forgiving final-position state.
- Apollo's OCR-damaged transcript prompt was much less forgiving.
- The real bug was universal: replacing the suffix token at generation time is
  not a valid boundary-injection contract.

## The Violated Invariant

Autoregressive decoding treats the final prompt position as the launch point for
the next token. In a standard causal LM forward pass, the logits used for the
first generated token are read from the final sequence position.

So this old code path was fatal:

```python
injected_hidden[:, -1, :] = residual
```

It was not "injecting into memory." It was replacing the current question's last
token representation. With a zero vector, the model sees a blank final residual.
With a random vector, it sees an impossible final residual. With an unrelated
boundary, it sees a donor state in the slot where the prompt's decode anchor
should be.

That is why zero injection was the cleanest proof.

## Why Gemma-4 Was Less Tolerant

Gemma-4-E2B-it makes this bug easier to trigger because its architecture has
less slack around the injection site:

- 35 layers with many sliding-window layers.
- Full/global layers only at `{4, 9, 14, 19, 24, 29, 34}`.
- KV-sharing/consumer layers around the later stack.
- Smaller hidden size than Gemma-3-4B.

The codebase records this in:

- `src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py`
- `tests/inference/backends/test_axis_runtime_fix_kv_consumer_layers.py`

L29 itself is global/full attention, but the surrounding stack still expects a
coherent sequence of token states. Replacing the final prompt state at L29 gives
downstream layers a malformed current-token state. Gemma-3's all-full-attention,
no-KV-sharing stack tolerated the Apollo prompt better, but that was an
accidental tolerance, not a safe invariant.

## What Made It Work

The fix was to stop using the final prompt token as the carrier.

I added a dedicated seeded carrier path:

- `src/chuk_lazarus/inference/backends/torch_runtime.py:1652`
  `generate_with_residual_seeded_at_layer`

The new path:

1. Tokenizes the real prompt.
2. Prepends a BOS/pad/eos seed token.
3. Builds an attention mask for the seeded sequence.
4. Registers a `forward_pre_hook` on `residual_state.layer_index`.
5. Replaces only `hidden_states[:, 0, :]` with the boundary.
6. Leaves `hidden_states[:, -1, :]` alone.
7. Uses normal generation from the seeded prompt.

The store/query runtime was then rewired to use the seeded carrier:

- `src/chuk_lazarus/inference/context/knowledge/torch_query.py:457`
- `src/chuk_lazarus/cli/commands/context/generate/_torch.py:156`

The older low-level method remains present:

- `src/chuk_lazarus/inference/backends/torch_runtime.py:1599`
  `generate_with_residual`

That old method is still useful as a steering/ablation primitive, but it should
not be the knowledge-store boundary path for Gemma-4 Apollo queries.

## Why This Is Different From The Existing Layer-0 Seeded Path

The repo already had:

- `src/chuk_lazarus/inference/backends/torch_runtime.py:1807`
  `generate_with_residual_prefill_seeded`

That method injects at `layers[0]`, position 0. It is the canonical Apollo demo
recipe already documented in the research notes.

The new method keeps the same carrier-slot lesson but preserves the requested
layer:

- old canonical demo: carrier slot position 0, injection layer 0;
- new fix: carrier slot position 0, injection layer `crystal_layer` such as L29.

So it answers this puzzle's constraint: use Gemma-4, use Apollo, use boundary
injection, and still inject at the requested crystal layer without erasing the
final prompt token.

## Tests To Keep Around

The important regression tests are:

- `tests/inference/backends/test_torch_runtime.py:348`
  `TestResidualSeededAtLayer`
- `tests/inference/backends/test_torch_runtime.py:351`
  `test_seeded_layer_injection_preserves_last_prompt_token`
- `tests/inference/context/test_aus3000_torch_query.py:375`
  `test_auto_routed_single_window_uses_seeded_residual_carrier`

What they prove:

- the seeded carrier slot receives the residual;
- the last prompt token remains intact;
- the query path calls the seeded residual carrier, not the old suffix-token
  replacement method.

The Linux-side live test should also include the four-condition Apollo probe:

1. no injection;
2. Apollo boundary;
3. known-working AUS3000 boundary;
4. zero vector;
5. random vector.

After the fix, condition 4 should no longer produce token salad. A zero vector in
a dedicated carrier slot may still influence the prompt by adding an attended
seed position, but it should not erase the final prompt token or collapse into
garbage.

## Code Reading Trail

Primary runtime code:

- `src/chuk_lazarus/inference/backends/torch_runtime.py:1599`
  old `generate_with_residual`, suffix-token replacement.
- `src/chuk_lazarus/inference/backends/torch_runtime.py:1652`
  new `generate_with_residual_seeded_at_layer`, crystal-layer carrier fix.
- `src/chuk_lazarus/inference/backends/torch_runtime.py:1807`
  existing `generate_with_residual_prefill_seeded`, canonical layer-0 seeded path.

Call sites:

- `src/chuk_lazarus/inference/context/knowledge/torch_query.py:457`
  store query path now uses the seeded carrier.
- `src/chuk_lazarus/cli/commands/context/generate/_torch.py:156`
  explicit torch checkpoint generate path now uses the seeded carrier.
- `examples/inference/demo_c_apollo11_torch.py`
  existing Apollo demo path uses `generate_with_residual_prefill_seeded`.

Architecture evidence:

- `src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py`
  frozen Gemma-4-E2B-it layer type map.
- `tests/inference/backends/test_axis_runtime_fix_kv_consumer_layers.py`
  KV-consumer/shared-KV runtime coverage.

Research notes already in this repo:

- `research/kv-memory-recipe-research/run-1/10-cross-axis-syntheses/06-axis-A-two-runtime-recipes-coexist-LAYER-0-VS-LAYER-30.md`
  explains the two runtime recipes.
- `research/kv-memory-recipe-research/run-1/11-final-deliverables/[OWNER_KV_RECIPE_V1].md`
  the larger extraction recipe and PROP K.A3 / PROP K.0 background.

## External Reading

- [The Residual Stream Is All You Need: On the Redundancy of the KV Cache in Transformer Inference](https://arxiv.org/abs/2603.19664)
- [PDF for arXiv 2603.19664](https://arxiv.org/pdf/2603.19664.pdf)
- [KV-Direct reference code](https://github.com/Kaleemullahqasim/KV-Direct)
- [alphaXiv overview for 2603.19664v1](https://www.alphaxiv.org/overview/2603.19664v1)

The key idea to carry forward from that paper is that KV state can be derived
from residual state only when the residual stream remains position-coherent. The
fix here is a small version of that principle: preserve the live prompt
positions, put memory in its own carrier slot, and let the transformer process
that carrier through the normal attention machinery.

## Rule For Future Boundary-Injection Work

Never treat "zero injection is harmless" as automatically true.

It is harmless only if the injection operation is an additive no-op or a
replacement into a slot that is already reserved for external memory. If zero
injection breaks generation, the first question should be:

> What live model state did the injection path replace?

For this bug, the answer was:

> It replaced the final prompt token's residual at L29.

That was the whole puzzle.
