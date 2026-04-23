# ASI-Evolve Router for Offline Window Selection

## Source Anchors

- `GAIR-NLP/ASI-Evolve/database/algorithms/ucb1.py` — canonical UCB1 arm
  selection used by ASI-Evolve to pick the next program to evolve.
- `GAIR-NLP/ASI-Evolve/database/algorithms/island.py` — island / migration
  bookkeeping for the MAP-Elites-style population.
- `GAIR-NLP/ASI-Evolve/database/database.py` — persistence layer for
  per-program visit counts and mean rewards.
- `GAIR-NLP/ASI-Evolve/config.yaml` — defaults (`ucb1_c=1.414`,
  `num_islands=5`, `migration_interval=10`, `migration_rate=0.1`,
  `exploration_ratio=0.2`, `exploitation_ratio=0.3`).

## Canonical UCB1 Formula

```
ucb1(w) = q_w + c * sqrt( ln(N) / n_w )    when n_w > 0
ucb1(w) = +inf                              when n_w == 0
```

with `c = ucb1_c = 1.414` (from ASI-Evolve `config.yaml` default).

## Default Hyperparameters

| Name | Default | Source |
|------|---------|--------|
| `ucb1_c` | `1.414` | ASI-Evolve config.yaml |
| `num_islands` | `5` | ASI-Evolve config.yaml |
| `migration_interval` | `10` | ASI-Evolve config.yaml |
| `migration_rate` | `0.1` | ASI-Evolve config.yaml |
| `exploration_ratio` | `0.2` | ASI-Evolve config.yaml |
| `exploitation_ratio` | `0.3` | ASI-Evolve config.yaml |
| `candidate_pool` | `64` | axis-2 contract |

## Adaptations from ASI-Evolve

(i) Reward signal adaptation: ASI-Evolve scores program candidates against an
    executable evaluator returning reward in [0,1]. For offline query-time
    window selection no evaluator exists. This runtime defines q_w as the
    per-candidate-pool min-max-normalized TFIDFRouter.score_window output
    from src/chuk_lazarus/inference/context/knowledge/route.py. If the pool
    has a single unique score the normalization degenerates to 1.0 and the
    adaptation surfaces reward_signal_degenerate=true in the per-query
    metadata.

(ii) Visit-count adaptation: ASI-Evolve maintains n_w across multi-iteration
    evolutionary search. For one-shot query selection all n_w start at 0 and
    the UCB1 exploration term is degenerate (+inf). This runtime operates in
    exploration-disabled mode at query time by default: when all candidates
    have n_w == 0 the tie-break falls through to q_w descending with a
    deterministic secondary key (session_id, window_id). When prior telemetry
    is present (visit_counts / mean_rewards loaded from asi_router_state.json)
    the full UCB1 formula is evaluated.

(iii) Island assignment adaptation: ASI-Evolve assigns programs to islands via
    a feature-map. For windows this runtime defines the feature-map over a
    stable tuple (session_id, recency_bucket, keyword_count_bucket) with
    recency_bucket = floor(log2(1 + session_age_seconds)) clamped to
    [0, num_islands-1] and keyword_count_bucket = min(keyword_count // 5,
    num_islands-1). The final island_id assigned to a window is the pair-sum
    modulo num_islands.

(iv) Multi-candidate emission adaptation: ASI-Evolve returns one selected
    program per island cycle. Window selection needs a ranked SET of size
    candidate_pool so axis-3 can tier. This runtime emits the full ranked
    list of AsiRouterCandidate, sorted by ucb1_score descending, with
    deterministic secondary keys.

## State File Schema

The router state is persisted as `asi_router_state.json` (filename exported
as `ASI_ROUTER_STATE_FILENAME`) under a caller-provided `archive_root`:

```json
{
  "schema_version": 1,
  "current_island": 0,
  "total_selections": 0,
  "num_islands": 5,
  "migration_interval": 10,
  "migration_rate": 0.1,
  "visit_counts":   { "<session_id>:<window_id>": <int> },
  "mean_rewards":   { "<session_id>:<window_id>": <float> },
  "islands":        [ { ... } ],
  "feature_map":    { "<session_id>:<window_id>": <int island_id> }
}
```

Rules:

- Composite key: `f"{session_id}:{window_id}"`.
- Missing file OR missing `schema_version` sentinel → fresh defaults.
- `schema_version` present and not equal to `1` → `RuntimeError`; no silent
  fallback.
- Writes go through `save_asi_router_state` which is atomic (temp file +
  `os.replace`).

## Public API

Module: `chuk_lazarus.session_retrieval.asi_router` (re-exported from
`chuk_lazarus.session_retrieval`).

- `AsiRouterCandidate` — frozen dataclass
  `(handle, window_id, ucb1_score, raw_router_score, island_id, visit_count,
  mean_reward)`.
- `AsiRouterState` — mutable dataclass holding the UCB1 / island bookkeeping.
- `ASI_ROUTER_STATE_FILENAME` — constant `"asi_router_state.json"`.
- `asi_route_candidates(handles, query_text, tokenizer, *, ucb1_c=1.414,
  num_islands=5, migration_interval=10, migration_rate=0.1,
  exploration_ratio=0.2, exploitation_ratio=0.3, candidate_pool=64,
  archive_root=None)` — return the full ranked list.
- `load_asi_router_state(archive_root, *, num_islands=5,
  migration_interval=10, migration_rate=0.1)` — read state (or fresh).
- `save_asi_router_state(archive_root, state)` — atomic persist.
- `compute_ucb1(q_w, n_w, total_visits, *, ucb1_c=1.414)` — canonical math.
- `assign_island(session_id, window_id, *, keyword_count,
  session_age_seconds, num_islands=5)` — deterministic bin.
- `advance_island(state)` — tick island cursor and migrate feature-map.

## Integration

Axis-3 will consume the returned `list[AsiRouterCandidate]` to tier windows
into exploration / exploitation pools (using `exploration_ratio` and
`exploitation_ratio`). Axis-4 / axis-5 are responsible for updating
`visit_counts` and `mean_rewards` after downstream evaluation and calling
`save_asi_router_state` to persist.

The router does NOT persist state itself — callers are expected to call
`save_asi_router_state` explicitly after any downstream telemetry update.
This preserves purity of `asi_route_candidates` except for the optional
load-from-archive step.

## Provenance

LEAD frozen contract record: `ve-ins-0mo9p7q1a000047eddd`
(goal `asi-kv-direct-chat`, axis-2 implementation, run-1).
Landed under manifest `ve-ins-0mo9okfv10000abec98`.
