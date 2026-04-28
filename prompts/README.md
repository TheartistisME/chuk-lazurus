# prompts/ — active charter + spec snapshots

On-disk snapshots of the canonical vee records, committed so the workflow travels with the repo. Vee records remain the **authoritative source of truth**; files here are pinned copies of the currently-active versions.

## Active files

| File | Role | Vee record id | Tokens (codex) |
|---|---|---|---|
| `hand-v6.md`            | HAND charter v6            | `ve-ins-0mo727b0n0000425b37` | 1660 |
| `orchestrator-v8.md`    | ORCHESTRATOR charter v8    | `ve-ins-0mo727cr60000761d65` | 1744 |
| `lead-v7.md`            | LEAD charter v7            | `ve-ins-0mo727bvl00008fbfb0` | 1556 |
| `workflow-spec-v9.md`   | Master workflow spec v9    | `ve-ins-0mo727dn20000429012` | 1824 |

Related artifacts in the repo:
- Human-readable mirror doc: `docs/workflow/CUDA_BACKEND_RUN.md`
- Tool-call hook: `.claude/hooks/posttooluse-vee-log.sh` + `.claude/settings.json`
- Token counter: `tools/count_tokens.py`
- Bug-reporting slash command: `.claude/commands/vee:create-bug&patch-Report.md` (authored by run-1 Hand)
- Agent context slash command templates: `prompts/slash/agent-context/` (install local
  runtime copies with `python scripts/install_agent_context_slash_commands.py`)

## Version history

- **v6 HAND / v8 ORCH / v7 LEAD / v9 spec** (active, 2026-04-20): adds HARD-RULE-INVIOLATE hard rule to all 3 charters (charter hard rules trump override directives; refuse + bug-report on conflict). ORCH v8 adds PRE-SPAWN PAYLOAD REVIEW (Task(validator) self-check against recipient's hard rules before every `vee agent spawn`). ORCH v8 §Closure validator gate rewritten — attribution source is now the canonical event log (hook tool-call records), NOT `git status` / `git diff`. Spec v9 adds §18 Pattern-D anti-pattern + §23 attribution-source clarification. Root-cause fix for **Bug J-prime** (presence ≠ authorship in shared worktrees) and **Bug K-prime** (override directive conflicts with hard rule). Patch-process: `ve-ins-0mo727hz000000ab422`.
- **v5 HAND / v7 ORCH / v6 LEAD / v8 spec** (tombstoned, 2026-04-20): SCOPE-BARRIER PROTOCOL — manifest + SCOPE BINDING + hook warnings. Root-cause fix for Bug I-prime. Patch-process: `ve-ins-0mo71tuon0000b3e117`.
- **v5 HAND / v6 ORCH / v5 LEAD / v7 spec** (tombstoned, 2026-04-20): AUTO-DEPENDENCY LINKING. Efficiency-driven patch. Patch-process: `ve-ins-0mo70h5pg0000e1436a`.
- **v5 HAND / v5 ORCH / v5 LEAD / v6 spec** (tombstoned, 2026-04-20): GRAMMAR OF ABSENCE + 4-section baseline body template. Root-cause fix for Bug H-prime. Patch-process: `ve-ins-0mo6zq7u100007131d6`.
- **v5 HAND / v5 ORCH / v4 LEAD / v5 spec** (tombstoned, 2026-04-20): GATE-HALT SEMANTICS + PANE-LIMIT PROTOCOL. Root-cause fix for Bug G-prime. Patch-process: `ve-ins-0mo6z2352000039d0c3`.
- **v5 HAND / v4 ORCH / v4 LEAD / v4 spec** (tombstoned, 2026-04-20): OVERRIDE DISCIPLINE hard rule + belt-and-braces pane-session marker. Root-cause fix for Bug F. Patch-process: `ve-ins-0mo6xjqmo00004c6361`.
- **v4 HAND / v3 ORCH / v3 LEAD / v3 spec** (tombstoned): introduced event-wait protocol (read-first, tail-second), session-id retrieval from event log (fix for null JSON), `run-<N>` markers, bug-reporting via `/vee:create-bug&patch-Report`.
- **v3 HAND / v2 ORCH / v2 LEAD / v2 spec** (tombstoned): traceability hardening — Step 0 session open, `--session` on every vee call, pane-session markers, session tree via `--parent` + `vee session handoff`.
- **v2 HAND / v1 ORCH / stub LEAD / v1 spec** (tombstoned): traceability-flagged but without the Step-0 session open pattern.
- **v1 HAND** (tombstoned): initial charter from pre-flight; classification-example bug (`tier1` → `foundational`).

## Regeneration protocol

1. Patches land in vee first (new `reference` record via `vee record`).
2. Predecessor is tombstoned (`vee delete --mode tombstone`).
3. A `patch-process` decision record documents trigger → RCA → edits → lesson.
4. THEN the file snapshot here is regenerated from `/tmp/charter-<x>-vN.txt`.

Do NOT edit these files directly — they will be overwritten next regeneration. Changes go in vee; files follow.

## Quick audit helpers

```bash
# Current active charters (vee is truth)
vee query "HAND charter"          --mode lexical --limit 3 --json | jq '.data.results[] | .id'
vee query "ORCHESTRATOR charter"  --mode lexical --limit 3 --json | jq '.data.results[] | .id'
vee query "LEAD charter"          --mode lexical --limit 3 --json | jq '.data.results[] | .id'
vee query "workflow master spec"  --mode lexical --limit 3 --json | jq '.data.results[] | .id'

# Patch history (meta-workflow)
vee query "patch-process" --mode lexical --limit 20 --json | jq '.data.results[] | .id'

# Bug reports (supervisor inbox)
vee query "supervisor-alert" --mode lexical --limit 20 --json | jq '.data.results[] | .id'
```

## Session / run context

- Root supervisor session: `ve-ses-0mo6t078v00001642a8` (task: cuda-backend-meta; continuous across runs)
- Current run: **run-2** (run-transition record `ve-ins-0mo6v08fd0000032d85`)
- Workspace: `.vee/state.db` (gitignored; records live locally)
