## What and why

<!-- One behaviour change per pull request. Link the issue. -->

## Checklist

- [ ] `ruff check .`, `ruff format --check .`, `mypy` and `pytest -q` pass locally
- [ ] New or changed behaviour has a test (rules: one positive and one negative)
- [ ] `CHANGELOG.md` updated under `[Unreleased]` for anything a user can see
- [ ] Report text, exit codes or spec format changed? Say so here, and refresh `spec.schema.json` if the spec changed
- [ ] No number suggested by the library; no new upward import between layers
