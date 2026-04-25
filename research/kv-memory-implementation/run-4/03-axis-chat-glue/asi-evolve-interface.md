# ASI Evolve interface — chat-glue (run-4 Axis-3+4)

LEAD pane: kv-memory-implementation-lead-axis-chat-glue
LEAD session: ve-ses-0moedyvk500003e7028
Q4 supervisor decision: LEAD DISCOVERS.
Discovery vehicle: Explore sub-agent (read-only, very thorough).

## A. ASI Evolve interface

Module path: `src/chuk_lazarus/session_retrieval/asi_router.py`

Public dataclasses:
- `AsiRouterCandidate` — line 79
- `AsiRouterState` — line 92

Public entry-point function — `asi_router.py:302`:
```python
def asi_route_candidates(
    handles: Sequence[CheckpointHandle],
    query_text: str,
    tokenizer: Any,
    *,
    ucb1_c: float = 1.414,
    num_islands: int = 5,
    migration_interval: int = 10,
    migration_rate: float = 0.1,
    exploration_ratio: float = 0.2,
    exploitation_ratio: float = 0.3,
    candidate_pool: int = 64,
    archive_root: Path | None = None,
) -> list[AsiRouterCandidate]:
```

`AsiRouterCandidate` carries: `handle` (CheckpointHandle), `window_id` (int), `ucb1_score` (float), `raw_router_score` (float), `island_id` (int), `visit_count` (int), `mean_reward` (float).

## B. Decision: NO adapter required

Return type is `list[AsiRouterCandidate]`, NOT `list[VecInjectMatch]`. However, the chat-script does not pass ASI output directly to `vec_inject_to_kv_direct`. Instead:

1. ASI candidates → `assign_tiers(candidates, K_HOT, K_WARM, candidate_pool)` → `list[TierAssignment]` (`session_retrieval/tier_policy.py:63`)
2. TierAssignments → `retriever.answer_with_kv_direct(query_text, assignments_for_handle, ...)` (`session_retrieval/retriever.py`)
3. The retriever internally translates TierAssignments into `VecInjectMatch` records and calls `vec_inject_to_kv_direct` (post-Axis-1 patched at `inference/context/research/vec_inject/kv_direct_adapter.py:151`).

Therefore the adapter `asi_to_vec_inject_match()` mentioned in the Q4 contingency is NOT needed: the wiring is via `assign_tiers` → `answer_with_kv_direct`, and the VecInjectMatch construction lives inside the retriever (read-only for chat-glue).

Q4 verdict: ASI Evolve is wired AS-IS into the chat-script via the existing tier-assignment plumbing.

## C. Existing chat-script wiring (entry-state)

Call site is `scripts/interactive_memory_chat.py:1099` inside `kv_query_turn()`:

```python
candidates = asi_route_candidates(
    self.retriever.handles,
    query_text,
    self.retriever.tokenizer,
    candidate_pool=candidate_pool,
)
...
tier_assignments = assign_tiers(
    candidates,
    K_HOT=k_hot,
    K_WARM=k_warm,
    candidate_pool=candidate_pool,
)
...
result = self.retriever.answer_with_kv_direct(
    query_text,
    assignments_for_handle,
    hot_budget_mib=hot_budget_mib,
    warm_config=warm_config,
    generation_config=gen_config,
    handle=top_handle,
    **selector_kwargs,
)
```

`kv_query_turn` is invoked in two places:
- Direct `/kv_query <text>` slash command — `_handle_command` at line 1741.
- Auto-route through `recall_chat_turn` at line 950 when `memory_mode == "kv_direct"`.

The auto-promotion `topical → kv_direct` fires at line 1527 inside `save_current_session()` once `vec_inject.npz` is successfully written (`vec_inject_available == True`).

## D. Save-emit infrastructure (Axis-3 entry-state)

- `_emit_vec_inject_npz` — line 548; writes `<session_root>/vec_inject.npz` from `torch_store/window_tokens.npz` via `extract_vec_inject_index_torch`. Best-effort; never raises.
- `save_current_session` — line 1358; idempotent. Writes:
  - transcript JSON at `transcripts_root/<sid>.json`
  - AUS3000 clauses at `inputs_root/<sid>/`
  - manifest.json at `<sid>/torch_store/manifest.json` (via `indexer.flush_and_close()`)
  - boundaries at `<sid>/torch_store/boundaries/window_*.npy`
  - selection_ready descriptors at `<sid>/torch_store/selection_ready/`
  - save-state.json at `<sid>/save-state.json`
  - vec_inject.npz at `<sid>/vec_inject.npz`
- `/save` slash-command handler — line 1681, calls `self.save_current_session()`.
- `/quit` and `/exit` handlers — line 1668, interactive `[Y/n]` prompt before `save_current_session()`.
- EOF/Ctrl-C handler — line 1636, same interactive `[Y/n]` prompt.

GAPS for Axis-3:
- No unified `emit_store(...)` function (Q3 supervisor decision wraps both `/save` and session-end through a single entry).
- No `.dirty` flag (Addition 2). Every assistant turn currently extends conversation tokens but does not mark the store dirty.
- No non-interactive `atexit` registration. The interactive prompts are the only session-end flushers.

## E. Session-route infrastructure (Axis-4 entry-state)

- `kv_query_turn` — line 1050. Already invokes ASI Evolve and threads through `answer_with_kv_direct`.
- WarmPenaltyConfig construction — line 1132 (post-Axis-2 patched).
- Token-budget knobs in env: `LAZARUS_KV_CANDIDATE_POOL` (default 16), `LAZARUS_KV_K_HOT` (4), `LAZARUS_KV_K_WARM` (8), `LAZARUS_KV_HOT_BUDGET_MIB` (32), `LAZARUS_KV_HOT_BONUS` (0.0).
- `recall_chat_turn` dispatches `memory_mode == "kv_direct"` → `kv_query_turn` (line 950).

GAPS for Axis-4:
- No `MAX_TOTAL_INJECT_TOKENS = 4096` budget governor (Addition 1). Current behaviour is governed only by `K_HOT + K_WARM = 12` and the runtime's `hot_budget_mib`.
- AMD 11 sanity check is implicit (lives in `kv_direct_adapter.py:131` `assert_global_attention_layer`); not surfaced as a chat-script-level invariant.

## F. AMD 11 layer fixture (read-only confirmation)

`src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py:73`:
```python
GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS: Final[frozenset[int]] = frozenset(
    {4, 9, 14, 19, 24, 29, 34}
)
```

`vec_inject_to_kv_direct` invokes `assert_global_attention_layer(target_layer)` at line 186 before any K/V materialisation. Sliding layers raise `SlidingWindowLayerRefusedError` (line 117–121).

The chat-script's `_derive_arch_config` returns `injection_layer = 13` (Gemma-4 producer layer that owns `k_proj` / `v_proj`). The actual KV-share-aware target layer (29 per recipe canonical) is selected downstream inside the retriever / runtime, where K/V is inherited via shared_kv_states across {14, 19, 24, 29, 34}.

## G. Token-budget governor inputs (Addition 1)

From `model.config` (HuggingFace Gemma-4 native):
- `self.model.config.head_dim` — int (= 256 for Gemma-4-E2B-it)
- `self.model.config.num_key_value_heads` — int (= 4 for Gemma-4-E2B-it)

Per-fact KV-tokens-equivalent cost:
```
per_fact_cost = head_dim * num_kv_heads
              = 256 * 4 = 1024 tokens / fact
```

At `MAX_TOTAL_INJECT_TOKENS = 4096`, the budget admits 4 facts maximum (truncate the rest from the bottom of the score-sorted list). This is consistent with the existing `K_HOT = 4` env default, so the governor binds tightest exactly when the user has tuned K_WARM up.

## H. Cold filter

The "cold/warm/hot" tiers are assigned by `assign_tiers` (`tier_policy.py:63`):
- rank < K_HOT (default 4) → HOT
- rank < K_HOT + K_WARM (default 12) → WARM
- rank ≥ 12 → COLD

The retriever consumes `assignments_for_handle` directly; cold tiers participate in the page layout but their coefficient is depressed via Axis-1 / WarmPenaltyConfig HOT bonus contrast, NOT silently dropped. For Axis-4, "filter cold" = drop by tier (HOT + WARM only) when constructing `assignments_for_handle`, and bound the retained set by the token-budget governor.

## I. Existing tests in tests/integration/

`tests/integration/` does NOT exist yet. The closest neighbours are `tests/chat_loop/test_live_indexer_integration.py` and `tests/session_retrieval/test_asi_router.py`. New tests under `tests/integration/` will be created fresh by Axis-3 and Axis-4 sub-agents.

## Cross-refs

- Axis-3 end-state: ve-ins-0moebmfex00005de840
- Axis-4 end-state: ve-ins-0moebmh4c00008fd6d6
- Mission proposals: ve-ins-0moebpfew00000db813 (Axis-3) + ve-ins-0moebph460000234bb0 (Axis-4)
- Scope manifest: ve-ins-0moecnemd0000340b9c
- Recipe: ve-ins-0modtwi7v0000ff6d88 [OWNER_KV_RECIPE_V1]
- Axis-1 lead-report: ve-ins-0moedtikd0000fb0bfb (commit b3f05c6)
- Axis-2 lead-report: ve-ins-0moedugx70000803ac3 (DEFAULT)
