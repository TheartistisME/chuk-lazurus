# axis-rope-phase-fix run-2 — run notes

**Lead pane:** kv-memory-implementation-lead-axis-rope-phase-fix
**LEAD session:** ve-ses-0modz7rr60000505fe6 (parent ORCH-v2 ve-ses-0modvfyrd00005b7a03)
**Beads mission:** chuk-lazurus-8nl (supervisor-pre-approved; created by ORCH-v2)
**Branch:** impl/kv-rope-phase-fix-run-2 (base main@90fd8f3)
**Mode:** autonomous
**Final verdict:** PASS

---

## 1. Bug surface (one-line)

`src/chuk_lazarus/inference/backends/torch_runtime.py:1889-1892`
inside the closure `_prepare_archived_prefix` defined within
`_kv_direct_patched_forward_factory(self, fire_counter, propagation_state)`:

    cos[:, 0:1, :].expand(...)  → all archived slots get position-0 RoPE phase = identity

The `cos`/`sin` passed in were the model's outer-call `position_embeddings`
computed for the **fresh-input current-generation positions**. Slicing position 0
and broadcasting over the N archived slots yields cos=1, sin=0 at every slot,
so archived K is unrotated. The parity test's hard assertion
`assert k_delta_prerope_max > 1e-3` therefore failed.

---

## 2. Fix mechanism (verbatim from `torch_runtime.py:1889-1922`)

Re-invoke the model's rotary embedding at the archived absolute positions
{0..N-1} with the prefix layer's `layer_type`, then apply that cos/sin to the
archived K via `apply_rotary_pos_emb(ak, cos_archived, sin_archived, unsqueeze_dim=2)`.

Key surface:

- `self._model.model.language_model.rotary_emb` is the
  `Gemma4TextRotaryEmbedding`. Its `forward(x, position_ids, layer_type=...)`
  reads `{layer_type}_inv_freq` and returns `(B, N, 2*inv_freq.shape[0])`.
- `prefix_attn_module.layer_type` is `'sliding_attention'` for the sliding
  master and `'full_attention'` for the global master. **Strict raise** on
  missing — fail-fast catches new model integrations.
- `propagation_state['archived_position_ids']` is an OPTIONAL hook for future
  axes (axis-G window-offset bookkeeping). Default is `arange(N)` per the
  bug-spec recommendation; matches the parity test's expectation
  (positions 0..N-1).

---

## 3. Acceptance — verbatim observed numerics (group A)

From `prod/validation/diagnostic_axis_rope_phase_fix_20260425T092651Z-90fd8f36.jsonl`:

    L=27 PASS k_delta_prerope_max=0.2421875
    L=28 PASS k_delta_prerope_max=0.255859375
    L=29 SKIP_LINEAGE  (full_attention not in sliding lineage — informational PASS)
    L=30 PASS k_delta_prerope_max=0.21484375
    L=31 PASS k_delta_prerope_max=0.26953125
    L=32 PASS k_delta_prerope_max=0.19140625

All 5 sliding-lineage layers cleared the `> 1e-3` hard threshold by ~2 orders of
magnitude. `kv_direct_active=True`, `prefix_forwards>=1`,
`path_a_replay_count==0` for all PASS layers.

---

## 4. Group B (new unit test) — 16/16 PASS

`tests/inference/backends/test_axis_rope_phase_fix_unit.py`:

- `test_rotary_forward_returns_3d_shape` × {sliding_attention, full_attention} — 2 cases
- `test_rotary_cos_per_position_dynamism` × {sliding_attention, full_attention} × n_offset∈{1,2,3} — 6 cases
- `test_rotary_sin_per_position_dynamism` × ditto — 6 cases
- `test_rotary_position_zero_is_identity` × {sliding_attention, full_attention} — 2 cases

**Iteration:** the first authored revision (sha8 `fac120f6`) asserted a
shape of `(1, N, head_dim)` for both layer types. That failed for
`full_attention` because `Gemma4TextRotaryEmbedding` uses a layer-type-specific
`inv_freq`:

    sliding_attention_inv_freq.shape == (128,)  → cos shape (B, N, 256)  [matches head_dim]
    full_attention_inv_freq.shape    == (256,)  → cos shape (B, N, 512)  [does NOT match head_dim]

The fix (sha8 `4882601a`) derives `expected_dim = 2 * inv_freq.shape[0]` from
the rotary module attribute per `layer_type`. Dynamism + position-zero-identity
assertions were not modified.

---

## 5. Group C (regression battery) — 6/6 GREEN

| Suite | Verdict | Count |
|---|---|---|
| `tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py` (PROP K.5 adapter) | GREEN | 9/9 |
| `tests/inference/backends/test_axis_BC_global_attention_guard.py` (PROP K.0 guard) | GREEN | 13/13 |
| `tests/inference/backends/test_axis_D_logits_equivalence.py` | GREEN | 1/1 |
| `tests/inference/backends/test_axis_E_kv_direct_e2e_apollo.py::test_kv_direct_synthetic_smoke_e2e_layer_29` | GREEN | 1/1 |
| `tests/inference/backends/test_axis_runtime_fix_kv_consumer_layers.py` | GREEN | 3/3 |
| `tests/inference/context/research/vec_inject/providers/test_local_file_torch.py` | GREEN | 6/6 |

---

## 6. Bug-spec path correction surfaced

Bug-spec acceptance §3 listed:

    tests/inference/backends/test_axis_BC_kv_direct_adapter.py (9/9 GREEN)

Actual location:

    tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py

The file exists and was committed at `b22c56142f0a905265030d0a62b75b5dcaf773b4`
(axis-BC PROP K.5 KV-direct adapter + PROP K.0 sliding-window guard) on main.
Validator-1 and validator-2 both reported NOT_FOUND because they searched only
under `tests/inference/backends/`. Lead resolved by running directly at the
correct path. Filed observational record `ve-ins-0moe5i7ml000032e36b` with
`follow-up-mission-requested` tag for bug-spec patch.

---

## 7. Decision log

| Decision | Choice | Rationale |
|---|---|---|
| Fix path | bug-spec implementation note (a): rotary_emb re-invoke | Mirrors the parity test's gold-reference rerope at lines 639-665; no fresh-input position-arithmetic edge cases |
| `layer_type` default on missing attribute | strict raise | ORCH-v2 OPEN-Q1 ACK: fail-fast catches new model integrations |
| `archived_position_ids` hook | optional `propagation_state['archived_position_ids']` with `arange(N)` default | ORCH-v2 OPEN-Q2 APPROVED for forward-compat with axis-G window-offset bookkeeping |
| Function signature | preserved unchanged | caller compatibility (`patched_forward` line 2026) |
| `cos`/`sin` parameters | retained but unused | signature stability for future re-use; minimal surgery |
| Unit-test shape assertion | derive from `inv_freq` per layer_type | Gemma-4 layer-type-asymmetric rotary; head_dim alone was wrong |
| Parity-test edits | NONE | production-fix-only path satisfied bug-spec acceptance (1); /euclid CLAIM gate not triggered |

---

## 8. Cross-references

- **Bug-spec:** `ve-ins-0moe2k6qp0000ca2643`
- **Supervisor-directive:** `ve-ins-0moe3k5xc0000590251`
- **Lead-scope-manifest:** `ve-ins-0moe3yftt0000f10053`
- **Baseline reference:** `ve-ins-0moe4ei8i0000653fc1`
- **Path-correction observation:** `ve-ins-0moe5i7ml000032e36b`
- **Run-1 status-supersede (parity test pre-existing WIP):** `ve-ins-0moe3gell0000d806d8`
- **Prior un-landed RoPE-phase fix attempt (design-pattern only, not bound):** `ve-ins-0mod7odhc000067b0a4`
- **Run-1 axis-A fixture (read-only):** `src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py`
- **Run-1 axis-runtime-fix (read-only):** `src/chuk_lazarus/inference/backends/_torch_runtime.py` (Gemma-4 KV-sharing handling — separate file from this fix's `torch_runtime.py`)
