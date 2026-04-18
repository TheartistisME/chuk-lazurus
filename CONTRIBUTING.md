# Contributing to <PROJECT>

Thanks for your interest in improving <PROJECT>. This document describes how
to get set up, how we expect changes to be shaped, and how to get them
reviewed.

## Quick start

```bash
git clone <repo-url>
cd <PROJECT>
# install deps (adapt to your toolchain)
# npm install   # or: bun install / pip install -e . / cargo build
```

## Project layout

See the "Project layout" section in the README for a tree of the top-level
directories.

## Development loop

1. Create a branch from `main`.
2. Make a small, focused change.
3. Run the full local check: lint, build, test.
4. Commit using Conventional Commits (see below).
5. Open a pull request against `main`.

## Testing

Run the test suite before opening a PR. New features and bug fixes should
come with tests that would have failed without the change.

## Commit style

We follow [Conventional Commits](https://www.conventionalcommits.org):

- `feat: …` — a new user-visible feature
- `fix: …` — a bug fix
- `docs: …` — documentation only
- `refactor: …` — code change that neither fixes a bug nor adds a feature
- `test: …` — adding or updating tests
- `chore: …` — tooling, deps, or housekeeping
- `ci: …` — CI configuration

Keep commits atomic — one logical change per commit. Write the body in the
imperative mood ("add X", not "added X").

## Branching

We use [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow):

- `main` is always releasable.
- Feature work happens on short-lived branches off `main`.
- Merges into `main` go through a pull request and passing CI.

## Pull requests

- Fill in the PR template.
- Link related issues.
- Keep the diff reviewable — split large changes into a stack of PRs if
  necessary.
- Expect review comments; iterate in-place rather than force-pushing a
  rewritten history.

## Reporting bugs

Open a GitHub issue using the "Bug report" template. Include a minimal
reproduction, expected vs. actual behaviour, and your environment.

## Security

Please do not report security issues in public issues. See
[SECURITY.md](SECURITY.md) for the private disclosure process.
