---
description: Build an onboarding context package from the DDIA zvec index.
argument-hint: <agent task>
allowed-tools: Bash(lazarus agent-context*)
---

# /agent-context:onboard

Use the agent task in `$ARGUMENTS`.

Run:

```bash
lazarus agent-context package --stage onboard --task "$ARGUMENTS" --next-steps "Read manifests, identify source of truth, then continue to planning." --output artifacts/agent_context/ddia/packages/latest-onboard.md
```

Read the package, apply the stage lens, then continue with `/agent-context:plan`.
