# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/).

## [Unreleased]

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
