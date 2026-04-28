# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Filedex / Tinydex Cadence

All agents must use tinydex/filedex as a tiny file-memory ritual:

1. Before reading or editing any file you plan to touch, run `tinydex scan <files...>` from the repo root. If using the WSL full path, run `/mnt/c/Users/jehma/Desktop/TinyTool/bin/tinydex scan <files...>`.
2. Before work, fetch relevant cards: `dependencies`, `tests`, and `risks`.
3. While working, set useful discoveries with `tinydex set <file> <category> "..." --agent <name>`.
4. At handoff, update `status`, `tests`, `next_steps`, and optionally `agent_notes`.
5. This applies to subagents too; coordinators must tell delegated agents to use this cadence.
6. Keep entries short and agent-readable.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
