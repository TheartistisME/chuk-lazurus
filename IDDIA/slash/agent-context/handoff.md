---
description: Build a handoff context package from the DDIA zvec index.
argument-hint: <agent task>
allowed-tools: Bash(python -m IDDIA*)
---

# /agent-context:handoff

Use the agent task in `$ARGUMENTS`.

Run:

```bash
python -m IDDIA package --stage handoff --task "$ARGUMENTS" --next-steps "Summarize durable state, verification, remaining work, and the next command." --output IDDIA/artifacts/ddia/packages/latest-handoff.md
```

Read the package, write the handoff, then continue with `/agent-context:exit`.
