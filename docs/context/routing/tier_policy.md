# Tier Policy

Deterministic tier policy for ASI-router candidates. Given a list already
ranked descending by `ucb1_score` (as produced by `asi_route_candidates`),
the policy assigns each candidate a `TierLabel` (`HOT` / `WARM` / `COLD`)
purely as a function of its zero-based rank.

The compatibility policy remains `rank-v1`. The interactive memory/KV-direct
path now defaults to `utility-v2`, which selects windows by expected utility
under budget and then assigns tiers dynamically.

## Provenance

- LEAD frozen contract: `ve-ins-0mo9p8kou0000d20e0d` (axis-3 of
  `asi-kv-direct-chat`, run 1).
- Upstream axis-2 closure: `ve-ins-0mo9vtofg00005ae032`.
- Baseline-of-absence: `ve-ins-0mo9p63sh0000f78047`.

## Canonical Policy — rank-v1

Input is assumed already ranked descending by `ucb1_score`; `assign_tiers`
does NOT re-sort. Let `kept = candidates[:candidate_pool]`. For each
zero-based rank `i` in `kept`:

- `i < K_HOT` → `HOT`
- `K_HOT <= i < K_HOT + K_WARM` → `WARM`
- `i >= K_HOT + K_WARM` → `COLD`

Edge case: when `len(candidates) < K_HOT`, every kept candidate is `HOT`;
zero `WARM`; zero `COLD`. More generally, when fewer candidates are supplied
than `candidate_pool`, tiers fill in order `HOT → WARM → COLD` and later
tiers may be empty.

## Default Hyperparameters

| Name | Default | Source |
|------|---------|--------|
| `K_HOT` | `4` | axis-3 contract |
| `K_WARM` | `12` | axis-3 contract |
| `candidate_pool` | `64` | axis-2 contract |
| `policy_version` | `"rank-v1"` | axis-3 contract |

## Utility Policy - utility-v2

`utility-v2` keeps up to `candidate_pool` candidates for telemetry, but only
HOT/WARM assignments are active for KV-direct generation. It scores:

```
U(w) = alpha * Rel(w) + beta * RRF(w) + gamma * Freshness(w)
     + delta * LearnedReward(w) - eta * CostNorm(w)
```

Defaults: `alpha=1.0`, `beta=1.0`, `gamma=0.10`, `delta=0.50`,
`eta=0.10`, `mmr_lambda=0.75`, `rrf_k=60`.

Selection is budget-aware: candidates are greedily admitted under `budget`
using MMR pressure:

```
MMR(w) = lambda * Rel(w, q) - (1 - lambda) * max Sim(w, selected)
```

The similarity term uses dense vectors when present and falls back to content
fingerprints. This suppresses duplicate or near-duplicate windows before they
consume HOT/WARM budget.

Dynamic tiers:

- `HOT`: active, high-utility candidate that survives budget pressure.
- `WARM`: active, useful but lower-confidence candidate.
- `COLD`: retained for evidence/telemetry but not active KV in the REPL's
  `utility-v2` path.

The old fixed slicing remains available with
`LAZARUS_ASI_SELECTOR_POLICY=rank-v1` or `policy_version="rank-v1"`.

## Public API

Module: `chuk_lazarus.session_retrieval.tier_policy`.

- `TierLabel` — `str` enum; members `HOT="hot"`, `WARM="warm"`, `COLD="cold"`.
- `TierAssignment` — frozen dataclass
  `(candidate, tier, rank, policy_version, policy_params)`.
- `assign_tiers(candidates, *, K_HOT=4, K_WARM=12, candidate_pool=64,
  policy_version="rank-v1", budget=None, mmr_lambda=0.75, rrf_k=60, ...)`
  — deterministic rank-v1 or dynamic utility-v2 tier assignment.
- `tier_assignment_to_dict(ta)` — one-shot dict encode for a single
  assignment.
- `tier_assignment_from_dict(data)` — inverse; `ValueError` on drift.
- `tier_assignments_to_json(ts)` — byte-deterministic JSON envelope encode
  (`sort_keys=True`, compact separators).
- `tier_assignments_from_json(raw)` — inverse; raises `ValueError` on
  schema mismatch, unknown tier label, missing required keys, or a mixed
  envelope (`policy_version` disagreeing with the first assignment).
- `POLICY_VERSION_RANK_V1 = "rank-v1"`.
- `POLICY_VERSION_UTILITY_V2 = "utility-v2"`.
- `TIER_POLICY_SCHEMA_VERSION = 1`.

## JSON Schema v1

```json
{
  "schema_version": 1,
  "policy_version": "rank-v1",
  "assignments": [
    {
      "tier": "hot",
      "rank": 0,
      "policy_version": "rank-v1",
      "policy_params": {"K_HOT": 4, "K_WARM": 12, "candidate_pool": 64},
      "candidate": {
        "window_id": 17,
        "ucb1_score": 1.2345,
        "raw_router_score": 0.8765,
        "island_id": 2,
        "visit_count": 0,
        "mean_reward": 0.0,
        "handle": {
          "session_id": "abc...",
          "checkpoint_dir": "/abs/path/to/session",
          "torch_store_dir": "/abs/path/to/session/torch_store",
          "manifest": { "clause_aligned": true, "num_windows": 32, "num_entries": 128 },
          "original_input_dir": "/abs/path/to/original/session"
        }
      }
    }
  ]
}
```

Rules:

- Path fields serialize via `str(path)`; deserialize via `Path(value)`.
- `original_input_dir=None` serializes as JSON `null`.
- `tier` serializes as its `.value` string; parsed back via `TierLabel(value)`.
- `policy_params` values are preserved as JSON scalar values.
- Envelope `policy_version` MUST match the first assignment's
  `policy_version` — mixed-policy blobs are rejected.

## Invariants

- `len(returned) == min(len(candidates), candidate_pool)`.
- Tier counts are exactly `(K_HOT, K_WARM, candidate_pool - K_HOT - K_WARM)`
  when `len(candidates) >= candidate_pool`; otherwise degrade gracefully
  (fewer `COLD` first, then fewer `WARM`, then fewer `HOT`).
- Assignment is stable: same input → same output, byte-identical.

## Integration

Axis-4 (mute / compress / mask): consumes the returned
`list[TierAssignment]` and applies per-tier attention treatments. `HOT`
windows are preserved at full fidelity; `WARM` windows receive
compress/mask; `COLD` windows are muted or dropped. The `policy_params`
captured on each assignment let axis-4 recover the exact rank thresholds
without re-reading router config.

Axis-5 (kv-direct-expansion): reads the same assignments to decide which
windows expand into the direct KV path. The JSON envelope round-trip
guarantees axis-5 can replay a previously chosen tier layout exactly,
enabling deterministic offline replay of the KV-direct chat surface.

## Runtime Flags and Eval

The REPL resolves selector controls through `MemoryRecallConfig` and env vars:

- `LAZARUS_ASI_SELECTOR_POLICY`: `utility-v2` default in the REPL; set
  `rank-v1` for compatibility.
- `LAZARUS_ASI_DENSE_SCORING`: `deterministic`, `auto`, `provided`, or `off`.
- `LAZARUS_ASI_RRF_K`: RRF denominator constant, default `60`.
- `LAZARUS_ASI_MMR_LAMBDA`: MMR relevance/diversity tradeoff, default `0.75`.
- `LAZARUS_ASI_COST_MODE`: `windows` default, or `tokens` for token-count cost.

Run:

```
python scripts/evaluate_memory_selector.py --output artifacts/memory_selector_eval.json
```

The report is machine-readable and includes baseline/new score, delta,
per-query selected windows and tiers, and pass/fail criteria.

## Future Policies

`policy_version` is a free-form string so future schemes (for example
`score-v1` using a percentile cut over `ucb1_score`, or `hybrid-v1` mixing
rank and score) can coexist with `rank-v1` without a schema bump. Consumers
MUST log the observed `policy_version` before acting on the assignments so
offline analysis can correlate retrieval behaviour with the exact policy
that produced each tiering decision.
