# Follow-up Mission Reference — vee CLI patch (P1 + P2/P3)

**Vee record:** ve-ins-0moe2j2ym00006ece06
**Status:** OPEN — owner team is `vee-maintainer`, NON-bead in chuk-lazurus

## Summary

Two leads spawning within ~3 s of each other received the SAME `session_id` from `vee session open`. Concrete collisions in run-1:

- `ve-ses-0modwqdrc0000bfaf68` — axis-BC + axis-D pair
- `ve-ses-0mody0x9f000054f70b` — axis-E + axis-F pair

HAND filed canonical bug-report ve-ins-0mody35e50000cdf8bb. Supervisor adjudicated as **vee CLI tooling bug (NOT charter bug)** — see ATTRIBUTION-PROTOCOL OVERRIDE record ve-ins-0mody4xaa0000d8c5ab.

## Hypothesised root cause (H2 accepted as more likely than alternatives)

Silent `--task` / `--parent` flag drop in the session-open path → empty-tuple / null-payload idempotent collapse. When two near-simultaneous opens both reduce to the same identity tuple, the deterministic ID generator returns the same id and the second caller silently *joins* the first session.

## Patches needed

| ID | Priority | Description |
|---|---|---|
| **P1** | high | Verify `--task` / `--parent` flag plumbing in `vee session open`. Ensure both flags are persisted to the sessions row (currently they appear silently dropped in some code path). |
| **P2** | high | Ensure concurrent `vee session open` invocations produce distinct `session_id` rows even when `--task` / `--parent` are not provided (e.g. seed deterministic ID with monotonic time + workspace + parent + a sub-millisecond nonce). |
| **P3** | medium | Add a `vee session list` filter for `parent_session` that surfaces collisions early (helps charter authors spot regressions). |

## Charter cleanup (low priority, post-fix)

Once the collision class is closed, HAND v6 + ORCH v8 Event-wait can use sessions-table primary lookup keyed on `session_id`. Spec v9 §23 Fix A was tactically RELAXED for run-1 only via the supervisor override.

## References

- Vee record (this follow-up): ve-ins-0moe2j2ym00006ece06
- HAND canonical bug-report: ve-ins-0mody35e50000cdf8bb
- supervisor-ack + ATTRIBUTION-PROTOCOL OVERRIDE: ve-ins-0mody4xaa0000d8c5ab
- ORCH override-ack: ve-ins-0mody99g200003a78ed
- Actual collisions: ve-ses-0modwqdrc0000bfaf68, ve-ses-0mody0x9f000054f70b

## In-run-1 reproduction note

This lead-axis-H session (ve-ses-0moe2fsgr0000cf0504) was opened ~hours after the collision pairs and received a distinct id, consistent with H2: temporal isolation suffices to avoid the collapse.
