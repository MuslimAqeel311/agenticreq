# Verification

What was run to check agenticreq, and what it showed. Every result here was run on this repository's own code, not
on a package from PyPI. Anyone can repeat it with the steps at the end.

## Test suite

- **2731 tests pass** (37 skipped: optional tools that are not installed).
- `ruff`, `mypy --strict` and the spec-schema check are clean.
- Every change that removes a finding has a test proving that the real defect is still reported.

## End-to-end runs: a multi-agent project in each of 15 frameworks

For each framework, a small multi-agent project was written: one agent finds an order, another refunds it, and the
refund is a write that the spec says needs approval. Then the whole journey ran on it, with no manual edits to what
agenticreq generated:

1. `agenticreq audit` (no spec)
2. `agenticreq init` (draws `spec.yml` from the code), then every `TODO` filled in as a user would
3. `agenticreq validate`
4. `agenticreq check`
5. `agenticreq mocks` (writes `mocks.yml`)
6. `agenticreq mock-run` (the agent runs with a canned model: no API key, no network)
7. `agenticreq check` with the run evidence
8. the `confirmed:` sign-off for every item code cannot prove
9. `agenticreq check` again

| framework | every step ran | model calls captured in the mock run | sign-offs left |
|---|---|---|---|
| LangGraph | yes | all | 0 |
| LangChain | yes | all | 0 |
| CrewAI | yes | all | 0 |
| OpenAI Agents | yes | all | 0 |
| Claude Agent SDK | yes | all | 0 |
| Google ADK | yes | all | 0 |
| smolagents | yes | all | 0 |
| AutoGen | yes | all | 0 |
| LlamaIndex | yes | all | 0 |
| PydanticAI | yes | all | 0 |
| Haystack | yes | all | 0 |
| Agno | yes | all | 0 |
| Semantic Kernel | yes | all | 0 |
| DSPy | yes | all | 0 |
| Strands | yes | all | 0 |

In every framework the mock run reached the second agent and its tool, and nothing was left in "Check with a mock
run". The final report lists only real gaps: the test projects deliberately have no approval gate or limit on the
refund, and agenticreq reported exactly that.

## What the runs found, and what was fixed

The end-to-end runs found places where agenticreq was wrong or incomplete. Each was fixed, with a test:

- approval gates of LangChain, LlamaIndex, Strands, Haystack, Claude Agent SDK and project helpers are recognised
- a mock run older than the code is never read as evidence and never a pass
- platform lookups, per-user folders and `env:` work the same on Windows, Linux and macOS
- limits are read from constants, imports, class attributes, helper call sites and per-agent factories
- `mocks` finds the function that runs the agent, stubs database and HTTP clients, and never picks a tool or a graph
  node as the entry
- the canned model reaches every agent of a team (handoffs, delegation, managed agents) and takes each one once
- framework-made tools (Agno team tools, DSPy `submit`, Google ADK `set_model_response`, CLI built-ins) are not
  reported as undeclared, while a built-in the code itself allows still counts as the code's
- Google ADK, DSPy and the Claude Agent SDK mock runs are captured

## What is not claimed here

- The runs were made on macOS. Windows and Linux path handling is covered by unit tests; a CI run on those systems
  is the next check.
- This page reports what was run and seen. It does not state precision or recall numbers: those come from the
  external benchmark, which is reported separately when it is run on this code.
- A mock run uses a canned model. It shows which tools are offered and in what order, never how a real model
  behaves (see the README's Limits).

## Repeat it

From a clone of the repository:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

For an end-to-end run, write a small multi-agent project with the framework installed in its own environment, then
run the steps above from its folder, with `agenticreq mock-run --python <that environment's python>`.
