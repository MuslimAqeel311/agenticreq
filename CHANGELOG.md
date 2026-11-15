# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/).

## [Unreleased]

## [0.1.0a3] - 2026-11-15

Checked end to end on a multi-agent project in each of 15 frameworks ([docs/verification.md](docs/verification.md)).

### Fixed
- Approval gates of LangChain (HumanInTheLoopMiddleware), LlamaIndex (wait_for_event), Strands (hook events),
  Claude Agent SDK (PreToolUse hooks) and Haystack (ConfirmationHook), and project helpers that ask a person, are
  recognised; a gate decided at run time is "Check with a mock run", never a gap or a pass (data/approval_gates.yml).
- A mock run older than the code (content or time) is never evidence: its items return to "Check with a mock run"
  and the result is not a pass.
- Mock runs: platform lookups start no program; every redirected per-user folder exists; `env:` wins; a blocked
  program and an import failure each have their own message; each handoff is taken once per run.
- Limits read from class attributes, helper call sites and per-agent factories; several values or a run-time value go
  to sign-off. Gemini / ADK upper-case schema types. `$` inside spec text is kept.

- A framework that was only installed or loaded next to the agent (seen in the mock run, not imported by the code) is
  no longer taken as the project's own: a LangGraph project is not credited with the OpenAI Agents SDK's default
  tracing, so its `audit_log` gap is reported whichever environment ran the mock run.

### Added
- `--entry` on `audit` / `check`, and `include:` / `exclude:` in `spec.yml`; the report header names what was scanned.
- `--spec` resolves against the folder being checked.
- `mocks` stubs tools that reach a database or HTTP service and names connections the run will block.
- Windows classifier; `httpx>=0.28,<1`; docs linked from `--help`.

## [0.1.0a1] - 2026-11-02

First alpha release. Packaging, repository and CI work only; the checks, rules, report text, exit codes and spec
format are those of `0.1.0.dev0` (below).

### Added
- `pyproject.toml` metadata (classifiers, keywords, project URLs, `test` and `dev` extras), `py.typed`, and the data
  tables, profiles and `spec.schema.json` shipped inside the wheel.
- `spec.schema.json` in the package and the repository root, with `scripts/export_schema.py` to refresh or check it.
- `docs/architecture.md` and `tests/test_layering.py` (a lower layer must not import a higher one).
- GitHub Actions: `ci.yml` (Linux and macOS on Python 3.10 to 3.14: ruff, mypy, pytest with coverage, build, `twine
  check`, wheel smoke test, `pip-audit`), `publish-testpypi.yml` and `publish.yml` (Trusted Publishing, no tokens).
- Dependabot for pip and GitHub Actions; pre-commit configuration; `.editorconfig`.
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue and pull request templates.

### Changed
- Version `0.1.0.dev0` to `0.1.0a1`, read from `agenticreq.__version__` (single source).
- Lower bounds are the oldest versions the suite passes on (Python 3.10): `pydantic>=2.11`, `typer>=0.15.4`.
- `cedarpy` moved from the runtime-adjacent `dev` list to the `test` extra (only a test imports it).
- Code formatted with `ruff format`; unused imports and stray whitespace removed; `mypy --strict` clean on current mypy.
- sdist no longer ships `requirements.txt`, `ac.py` or `inst.txt`.
- README: install from PyPI, exit codes, editor schema header and the Limits section (dynamic agents and tools,
  CodeAgent, MCP only through `--mcp`, numbers never suggested). The stale note that the name was taken on PyPI is gone.

## 0.1.0.dev0

Internal development builds before the first release. Their changes are part of `0.1.0a1`.
