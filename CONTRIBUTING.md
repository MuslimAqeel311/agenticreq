# Contributing to agenticreq

Thanks for helping. agenticreq compares an agent's declared requirements (`spec.yml`) with what its code does. It
never calls an LLM and never suggests numbers, and contributions keep it that way.

## Set up

```bash
git clone https://github.com/MuslimAqeel311/agenticreq && cd agenticreq
python -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install            # optional: runs ruff on each commit
```

Python 3.10 or newer. Some tests skip when an optional tool is missing (`opa` on `PATH`, a framework package).

## Before you open a pull request

```bash
ruff check .
ruff format --check .
mypy                          # strict
pytest -q                     # whole suite
```

CI runs the same on Linux and macOS across the supported Python versions, then builds the sdist and wheel, installs
the wheel in a clean virtual environment and runs `agenticreq --help` and a smoke `agenticreq check`.

## Ground rules

- **One behaviour change per pull request**, with a test. A new rule needs one positive and one negative test
  (see `tests/test_rules_matrix.py`).
- **Report text, exit codes and the spec format are public.** Changing them needs a CHANGELOG entry and, for the
  spec format, the `spec.schema.json` refresh (`python scripts/export_schema.py`).
- **Layers only point down:** `loader` and `models`, then analysis (`ast`, `detectors`, `recognizers`, `capture`,
  `traces`, `mcp`), then `rules`, `engine`, `report`, `cli`. `tests/test_layering.py` fails on a new upward import.
  See [docs/architecture.md](docs/architecture.md).
- **No numbers from the library.** Defaults never contain a limit; users write them.
- **Golden files** in `tests/golden/` change only when the output is meant to change. Say so in the pull request.

## Commit and review

Small, focused commits with a clear message. Update `CHANGELOG.md` under `[Unreleased]` for anything a user can see.
By contributing you agree your work is licensed under the Apache License 2.0, like the rest of the project.

## Reporting bugs and ideas

Use the issue templates. For a security problem, do not open an issue: see [SECURITY.md](SECURITY.md).
