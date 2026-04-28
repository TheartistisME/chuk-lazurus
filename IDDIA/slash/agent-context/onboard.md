---
description: Build an onboarding context package from the DDIA zvec index.
argument-hint: <agent task>
allowed-tools: Bash(python -m IDDIA*)
---

# /agent-context:onboard

Use the agent task in `$ARGUMENTS`.

Run:

```bash
python -m IDDIA package --stage onboard --task "$ARGUMENTS" --next-steps "Read manifests, identify source of truth, then continue to planning." --output IDDIA/artifacts/ddia/packages/latest-onboard.md
```

Read the package, apply the stage lens, then continue with `/agent-context:plan`.
