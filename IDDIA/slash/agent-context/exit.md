---
description: Build an exit context package from the DDIA zvec index.
argument-hint: <agent task>
allowed-tools: Bash(python -m IDDIA*)
---

# /agent-context:exit

Use the agent task in `$ARGUMENTS`.

Run:

```bash
python -m IDDIA package --stage exit --task "$ARGUMENTS" --next-steps "File follow-ups, close issues, commit, sync, push, and leave replay notes." --output IDDIA/artifacts/ddia/packages/latest-exit.md
```

Read the package, complete the repo closeout workflow, then loop to `/agent-context:onboard` for the next task.
