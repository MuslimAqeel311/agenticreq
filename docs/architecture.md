# Architecture

```
 spec.yml ──► loader/ ──► models/ (typed spec, defaults/) ─┐
 project code ──► ast/ ──► detectors/ + recognizers/ ──────┼──► rules/ ──► engine.py ──► report/ ──► CLI (cli.py)
 mocks.yml ──► capture/ (mock-run) + traces/ ──► observe.py ┘            │
 MCP tools (mcp/) ─────────────────────────────────────────┘      derived.py (trace_coverage, enforceability)
 data/ (YAML tables)  export/ (schema)  fix/ (validate --fix)  graph/ (delegation)  scaffold.py (init)
```

Data flows left to right: three sources of evidence (the spec, the code, and what a run or a trace showed) meet in
`rules/`, `engine.py` runs the rules and groups the results, `report/` prints them, and `cli.py` turns them into an
exit code.

## Layers

Imports point down or sideways, never up:

| layer | modules | role |
|---|---|---|
| 0 foundation | `models/`, `defaults/`, `data/`, `names`, `pathutil`, `versions`, `jsonschema_util`, `shapes`, `signoff`, `subfields`, `not_needed`, `params`, `thresholds`, `pyenv` | typed spec and findings, YAML tables, small helpers |
| 1 loader | `loader/` | `spec.yml` → typed spec (extends, short forms, wrong-field-name errors) |
| 2 analysis | `ast/`, `detectors/`, `recognizers/`, `capture/`, `traces/`, `mcp/`, `graph/`, `observe.py`, `export/`, `fix/` | evidence from code, runs, traces and MCP tool lists; delegation graph; exports; `validate --fix` |
| 3 rules | `rules/` | one function per rule id, each `(CheckContext) -> findings` |
| 4 engine | `engine.py`, `derived.py`, `coverage_table.py`, `progress.py` | run the rules, group results, compute coverage and enforceability |
| 5 report | `report/` | console, JSON and SARIF output |
| 6 cli | `cli.py`, `scaffold.py` (`init`), `mockgen.py` (`mocks`), `schema.py` | commands and exit codes |

`loader → models` in the diagram is the data flow: the loader *produces* models, so in code the loader imports
`models/`, not the other way round.

### Enforced by a test

`tests/test_layering.py` parses every module and fails when a lower layer imports a higher one. Eight upward imports
exist today, all inside functions (they break a cycle at import time); they are listed in `_KNOWN_UPWARD` in that
test, and the list can only shrink:

- `data → ast`, `pyenv → ast`, `models → ast`, `models → rules`, `thresholds → rules`
- `loader → capture`, `loader → rules`
- `traces → rules`

Fixing them changes where code lives, not what it does, and is tracked separately from the release work.

## Extension points

- **Extractor**: anything that fills an `ObservedAgent` (`ast/`, `capture/`, `mcp/`, `traces/`).
- **Rule**: `@rule("conformance", "AS-2xx")` on a function `(CheckContext) -> Iterable[Finding]`, with metadata in
  `rules/registry.py`.
- **Recognizer**: an optional per-framework plugin for graph reachability, registered under the entry-point group
  `agenticreq.recognizers` (`RECOGNIZER_API_VERSION = 1`).
- **Detector**: evidence that a prohibited action happened, from `prohibited[].detector: {type, params}`.

More detail on each, and on static extraction, is in the README ("Architecture").
