# /euclid axis-chat-session-route (run-4 Axis-4)

LEAD: kv-memory-implementation-lead-axis-chat-glue (BUNDLED with Axis-3 per Q1)
LEAD session: ve-ses-0moedyvk500003e7028
Mission: chuk-lazurus-z82 (Axis-4)
Branch: impl/kv-memory-finalize-run-4

## CLAIM

The chat-script's session-route to KV-direct injection is governed by a token-budget governor `_apply_token_budget` (Addition 1, MAX_TOTAL_INJECT_TOKENS=4096) inside `kv_query_turn`, and the AMD 11 sliding-window-hazard precondition is asserted at chat-script level for the recipe-canonical `target_layer=29` while the runtime guard `assert_global_attention_layer` continues to enforce it at the adapter level.

## SUB-CLAIMS

### SUB-CLAIM 1 — CONFIRMED
ASI Evolve interface is `asi_route_candidates(handles, query_text, tokenizer, *, candidate_pool=64, ...) -> list[AsiRouterCandidate]` (asi_router.py:302). Q4 best-case path: NO adapter required because the chat-script consumes ASI output via `assign_tiers()` → `answer_with_kv_direct()` (chat-script lines 1099-1163). VecInjectMatch construction lives inside the retriever stack (read-only for chat-glue). PASS test: `test_kv_query_turn_smoke_with_real_store_and_model` exercises the full path end-to-end.

### SUB-CLAIM 2 — CONFIRMED
`_apply_token_budget(self, tier_assignments, *, max_total_inject_tokens=MAX_TOTAL_INJECT_TOKENS)` (chat-script line 1078):
- Empty input → empty output (no raise).
- Sorts by `a.candidate.ucb1_score` descending.
- Per-fact cost = `head_dim * num_kv_heads` (read from `self.model.config`; falls back to 256 * 4 = 1024 when config is absent).
- Accumulates until `cumulative + per_fact_cost > max_total_inject_tokens`; truncates from the bottom.
- Silent drop (no error). Logs the kept/total/cost summary via `info(...)`.
PASS tests:
- `test_apply_token_budget_empty_pass_through`
- `test_apply_token_budget_under_budget_keeps_all` (3 stubs, all kept)
- `test_apply_token_budget_over_budget_truncates_from_bottom` (10 stubs, 4 kept, top 4 by ucb1)
- `test_apply_token_budget_sorts_by_score_descending` (reverse-sorted input → top 4 kept)
- `test_apply_token_budget_custom_budget` (max=2048 → 2 kept; max=0 → 0 kept)
- `test_apply_token_budget_falls_back_when_config_missing` (no config → 256*4=1024 default)

### SUB-CLAIM 3 — CONFIRMED (AMD 11)
`GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS` is the exact frozenset `{4, 9, 14, 19, 24, 29, 34}` (gemma4_e2b_it_layers.py:73). The recipe-canonical `target_layer = 29` is in the set. The PROP K.0 guard `assert_global_attention_layer` (kv_direct_adapter.py:131):
- Returns None for any layer in the global set.
- Raises `SlidingWindowLayerRefusedError` for any sliding layer (e.g. layer 10).
- The error carries `.target_layer` correctly.
PASS test: `test_amd11_global_attention_set_invariant`.

### SUB-CLAIM 4 — CONFIRMED (call site wiring)
The token-budget governor is invoked inside `kv_query_turn` (chat-script lines 1223-1232) after `assignments_for_handle` is computed and BEFORE `WarmPenaltyConfig(hot_bonus_value=hot_bonus_value)` (line 1234, Axis-2 contract-locked). If the governor truncates all assignments to empty, a `RuntimeError("axis-4 token-budget governor truncated all assignments")` is raised — which the existing `try/except (ValueError, RuntimeError)` block at line 1276 catches and falls back to `plain_chat_turn` per the axis-6 SILENT FALLBACK contract. Verified by reading the patched chat-script.

### SUB-CLAIM 5 — CONFIRMED (full smoke under CUDA)
End-to-end: real Gemma-4-E2B-it bf16 on RTX 5090 + pre-seeded store + `kv_query_turn` returns `TurnMetadata` with `mode in ("kv_direct", "none")`. The smoke also verifies `.dirty` is set after the turn (because the kv_query_turn code path now calls `self._mark_dirty()` before returning at line 1341). PASS test: `test_kv_query_turn_smoke_with_real_store_and_model`. Wallclock ~60-90s.

### SUB-CLAIM 6 — CONFIRMED (Axis-2 + Axis-1 inheritance)
- Axis-1 coefficient propagation (commit b3f05c6) at kv_direct_adapter.py:234-236 is consumed AS-IS via `vec_inject_to_kv_direct` deep inside `answer_with_kv_direct`.
- Axis-2 WarmPenaltyConfig.hot_bonus_value (commit 81eb102) at chat-script line 1234 is preserved verbatim — Axis-4 surgery does NOT modify line 1234.
Verified via `grep -n "WarmPenaltyConfig(hot_bonus_value=hot_bonus_value)" scripts/interactive_memory_chat.py` returning a single hit at line 1234 (post-Axis-3+4 surgery).

## UNKNOWN edges

- Behaviour when `K_HOT + K_WARM` (default 12) > floor(MAX_TOTAL_INJECT_TOKENS / per_fact_cost) (default 4): the budget governor takes precedence and admits only ~4 facts. The K_HOT/K_WARM env knobs become advisory rather than authoritative under default budget. Documented in notes.md but not gated by a charter rule.
- Behaviour when ASI returns 0 candidates: `kv_query_turn` raises `RuntimeError("asi_route_candidates returned no candidates")` at line 1106 (pre-existing path). The fallback chain is intact.
- Behaviour when `_apply_token_budget` is given a list with mixed-handle assignments (cross-handle): the governor sorts and truncates uniformly — the prior `assignments_for_handle` filter at line 1117-1126 already restricts to a single-handle set, so this edge is structurally pre-empted.

## adaptation-status

ACHIEVED+VERIFIED. All 8 tests GREEN on RTX 5090 / Gemma-4-E2B-it bf16. Wallclock 66.9s for the full suite. Diagnostic JSONL at `prod/validation/diagnostic_axis_chat_session_route_20260425T141316Z-8cef4b57.jsonl`.

## Test path

- Test file: `tests/integration/test_chat_session_route_inject.py` (357 lines, 8 tests).
- Validator command: `uv run pytest tests/integration/test_chat_session_route_inject.py -v --tb=short --no-header`
- Final verdict: 8 passed, 0 failed, 0 skipped.
- Single iteration — no rework required.

## Cross-refs

- Mission (beads): chuk-lazurus-z82
- Axis-4 end-state: ve-ins-0moebmh4c00008fd6d6
- Mission proposal: ve-ins-0moebph460000234bb0
- Scope manifest: ve-ins-0moecnemd0000340b9c
- Baseline-of-absence: ve-ins-0moeedj8z0000ce96a3
- Recipe: ve-ins-0modtwi7v0000ff6d88
- Q4 supervisor decision: ASI Evolve LEAD-DISCOVERS — discovered as `asi_route_candidates` AS-IS via assign_tiers, no adapter required
- Addition 1 supervisor binding: MAX_TOTAL_INJECT_TOKENS=4096 budget governor with sort-and-truncate-from-bottom (silent drop)
- AMD 11: target_layer ∈ GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS = {4, 9, 14, 19, 24, 29, 34}
- Code surgery file: scripts/interactive_memory_chat.py
- New methods: `MemoryChat._apply_token_budget` (line 1078)
- New constants: `MAX_TOTAL_INJECT_TOKENS` (line 160)
- AMD 11 import + assert: chat-script lines 1153-1168 (inside kv_query_turn)
- Token-budget call site: chat-script lines 1223-1232
- Axis-1 dependency: commit b3f05c6 (lead-report ve-ins-0moedtikd0000fb0bfb)
- Axis-2 dependency: commit 81eb102 (lead-report ve-ins-0moedugx70000803ac3)
