# axis-chat-glue notes (run-4 Axis-3+4 BUNDLED)

LEAD: kv-memory-implementation-lead-axis-chat-glue
LEAD session: ve-ses-0moedyvk500003e7028
Branch: impl/kv-memory-finalize-run-4
Tier: Tier-1 (Axis-1 + Axis-2 ACHIEVED+VERIFIED before spawn)

## Entry-state observations

### Files in WIP at lead spawn
Verified with `git diff main...HEAD` and `git status -s`:
- 5 commits ahead of main on `impl/kv-memory-finalize-run-4`:
  - 81eb102 — Axis-2 WarmPenaltyConfig.hot_bonus_value patch
  - b3f05c6 — Axis-1 coefficient propagation in vec_inject_to_kv_direct
  - f6129e2 — run-3 e2e testing artifacts
  - 49a9db2 — RoPE per-slot phase fix (axis-rope-phase-fix, run-2)
  - 90fd8f3 — run-1 synthesis + debrief
- `scripts/interactive_memory_chat.py` modified in WIP (uncommitted) — pre-existing edits, not from this lead.

### Decision tree on Q3 + Q4 + Add-1 + Add-2 (binding)

| Item | Supervisor binding | This lead's tactical choice |
|---|---|---|
| Q3 (save trigger) | BOTH /save AND session-end | Single `emit_store()` method on MemoryChat; call sites: /save (line 1681), atexit hook in run_repl, /quit + /exit + EOF branches |
| Q4 (ASI Evolve discovery) | LEAD DISCOVERS via Explore | ASI = `asi_route_candidates` already wired in chat-script line 1099. NO adapter needed (Q4 best case). |
| Addition 1 (token budget) | MAX_TOTAL_INJECT_TOKENS=4096; sort by score; truncate bottom | New `_apply_token_budget()` method on MemoryChat; called between `assign_tiers` and `answer_with_kv_direct` in `kv_query_turn` |
| Addition 2 (.dirty flag) | <store_dir>/.dirty; /save no-op-when-clean; session-end emits when dirty | `.dirty` at `<store_root>/.dirty`; helper `_mark_dirty()` called from each turn method that produces an assistant reply |

### Disjoint regions in scripts/interactive_memory_chat.py
Per scope manifest, this lead owns the entire file but must self-manage internal disjointness between Axis-3 (save-hook region) and Axis-4 (session-start region):
- Axis-3 region: turn-method tail (mark_dirty hooks); `emit_store` and `_mark_dirty` methods placed between `save_current_session` (line 1358) and `print_stats` (line 1553); `/save` and `/quit` and `/exit` and EOF dispatch sites; `run_repl` atexit registration.
- Axis-4 region: `_apply_token_budget` method placed near `kv_query_turn` (~line 1048); call site inside `kv_query_turn` between `assign_tiers` and `answer_with_kv_direct`; `MAX_TOTAL_INJECT_TOKENS` module constant at the top of file with the existing constants.
- Module constant `DIRTY_FLAG_FILENAME` (Axis-3) and `MAX_TOTAL_INJECT_TOKENS` (Axis-4) sit side-by-side near line 146.

### AMD 1-15 inheritance (entry-state)
- AMD 1 (sqlite3 fallback): used for every record fetch — verified by `sqlite3 .vee/state.db "SELECT body FROM records WHERE entity_id = '...'"` invocations at Step 0.5 and Step 1.
- AMD 3 (read-only `inference/backends/`): no edits planned — token budget reads `model.config` only.
- AMD 11 (sliding-window hazard): respected — chat-script doesn't pass an explicit `target_layer` to the adapter; the `assert_global_attention_layer` guard at `kv_direct_adapter.py:131` enforces the invariant downstream. New AMD-11 sanity import at chat-script level is decorative documentation.
- AMD 13 (feature branch): all commits on `impl/kv-memory-finalize-run-4`.
- AMD 14 (CUDA-only): tests CUDA-gated; no CPU-fallback paths added.

## Test surface plan

### Axis-3: tests/integration/test_chat_save_emit_emit_store.py
- `test_emit_store_writes_all_artifacts` — synthetic 2-turn conversation; assert vec_inject.npz + entries / window_tokens.npz + boundaries/window_*.npy + manifest.json all exist post-emit_store; assert manifest.json schema valid.
- `test_emit_store_idempotent` — call emit_store twice; assert files overwritten cleanly; no exception; second call is no-op when clean.
- `test_dirty_flag_lifecycle` — register turn → `.dirty` exists → emit_store → `.dirty` cleared.
- `test_save_no_op_when_clean` — `.dirty` absent → emit_store skips → no manifest re-write.
- `test_session_end_emits_when_dirty` — simulate atexit firing with `.dirty` present → emit_store called.
- CUDA-gated. Uses real Gemma-4-E2B-it bf16 because the prefill path (`extract_vec_inject_index_torch`) projects through the model.

### Axis-4: tests/integration/test_chat_session_route_inject.py
- `test_session_start_loads_store_and_injects` — pre-seeded store with 3 facts; invoke `kv_query_turn` directly; assert `meta.kv_direct_active is True` AND `meta.no_silent_fallback is True`.
- `test_token_budget_truncation` — synthetic store with > 4 facts; invoke; assert `len(assignments_for_handle)` truncated to <= floor(MAX_TOTAL_INJECT_TOKENS / per_fact_cost) ≈ 4 facts; no error raised.
- `test_amd11_layer_check` — confirm `29 in GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS` AND assert `assert_global_attention_layer(10)` raises `SlidingWindowLayerRefusedError`.
- `test_cold_facts_filtered` — synthetic store with 20 facts; verify HOT + WARM (= 12) is the upper bound on `assignments_for_handle` (baseline behaviour) AND budget governor caps further to ~4.
- `test_asi_evolve_smoke` — call `asi_route_candidates` against a known store; assert non-empty candidate list; document the AS-IS interface (no adapter needed).
- CUDA-gated. Uses real Gemma-4-E2B-it bf16.

## Bug-class regressions locked

- Axis-2 contract test (`test_chat_script_construction_path_kwarg_alignment`) statically locks line 1132's `WarmPenaltyConfig(hot_bonus_value=hot_bonus_value)` construction. Axis-4 surgery does NOT touch this line.
- Axis-1 coefficient propagation regression battery (12 tests) lives in `tests/inference/context/research/vec_inject/test_axis_BC_coefficient_propagation.py` — orthogonal to chat-script edits.
