# Real-World Multi-Fact Evidence Bundle

Commit under evidence:

```text
39860eb71d1c341250c2ad89cdd0986c5bffcf29
Add real-world multi-fact memory invariant
Commit date: 2026-04-27T05:42:26+08:00
```

Primary full run:

```text
prod/validation/repl-autoverify/20260426T205135Z-20260426t205135
status: PASS
sessions: 100
turns_per_session: 100
checks: 15
```

Bundled artifacts:

```text
summary.json
events.jsonl
transcript.log
scale-actual-recall-multi_fact.json
scale-actual-recall-real_world_multi_fact.json
commands.txt
SHA256SUMS.txt
```

Key full-run evidence:

```text
MULTI_FACT_RECALL:
multi-fact recall HOT=4/4 WARM=4/4 COLD=0/4 selected_tier=hot

REAL_WORLD_MULTI_FACT_RECALL:
natural multi-fact recall HOT=4/4 WARM=4/4 COLD=0/4
conflict_preserved=True final_decision_present=True selected_tier=hot

VRAM_BOUNDED:
vram peaks=[20112.0, 20112.0, 20112.0] delta=0.0 MiB

INFINITE_TURN_LATENCY:
50 measured turns flat: first10_mean=1.976s last10_mean=1.982s
median=1.913s max=2.331s ceiling=2.371s

HARNESS_PASS:
15/15 checks passed
```

Scale reports captured:

```text
scale-actual-recall-multi_fact.json
mode=multi_fact passed=50/50 hit_rate=1.000 required_hit_rate=1.000

scale-actual-recall-real_world_multi_fact.json
mode=real_world_multi_fact passed=25/25 hit_rate=1.000 required_hit_rate=0.900
```

The real-world scale report was produced by the optional natural-language scale
mode against a disposable store and copied here so the proof does not depend on
oral memory of the run directory where it was first written.
