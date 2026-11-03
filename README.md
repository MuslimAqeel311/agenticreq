# agenticreq

**Does the agent's code actually enforce what its spec declares?** You write the requirements in `spec.yml`;
agenticreq reads your agent's code (without running it) and reports which requirements the code enforces, which it
does not, and which need a person or a test run to confirm. No LLM is called, and agenticreq never suggests numbers:
you write them, it checks the code matches.

It is not a runtime guardrail or an eval tool. It finds where enforcement is missing, and can export the spec as a
policy for OPA or Cedar.

## Install

```bash
pip install agenticreq
agenticreq version
```

Python 3.10 or newer. Installing writes nothing into your project.

## Quick start

Run these from the agent's folder:

```bash
pip install agenticreq     # 1. writes nothing into your project
agenticreq audit           # 2. findings immediately: no spec, no setup
agenticreq init            # 3. writes spec.yml; replace every TODO
agenticreq mocks           # 4. writes mocks.yml for a test run with mock tools
agenticreq mock-run        #    runs the agent with a canned model
agenticreq check           # 5. compares spec.yml and the latest mock run with the code
agenticreq guide           #    where this folder stands, and the next command
```

## How it works

Run the commands from the agent's folder. `agenticreq guide` always shows where the folder stands and the next command.

| step | what it does | command |
|---|---|---|
| 1. audit | quick scan, no spec needed | `agenticreq audit` |
| 2. spec | writes `spec.yml`; you replace every `TODO` | `agenticreq init` |
| 3. check | compares `spec.yml` with the code | `agenticreq check` |
| 4. mock-run | runs the agent with mock tools to see what the model is offered | `agenticreq mocks`, then `agenticreq mock-run` |
| 5. sign-off | the owner confirms what code cannot show, under `confirmed:` in `spec.yml` | edit `spec.yml` |

**New project (no code yet).** Run `agenticreq init` in the empty folder, fill `spec.yml` first (what the agent must
and must not do), build the agent, then run `agenticreq check` as you go.

**Existing project.** Run `agenticreq audit` for findings with no setup. Then `agenticreq init` reads your code and
writes `spec.yml` with the facts it found filled in (`_derived`) and a `TODO` for each decision only you can make.
Fill those, then `agenticreq check`. Add steps 4 and 5 when you want run evidence and sign-offs.

## The spec: four levels

| level | who writes it | what it holds |
|---|---|---|
| 1. user (12) | you; `check` fails on `TODO` | `agent_id`, `risk_level`, `goal`, `out_of_scope`, `role_boundary`, `prohibited`, `approval_required`, `limits`, `escalation`, `untrusted_inputs`, `owner` and `technical_owner`, `data_rules` |
| 2. agent (41) | agenticreq, from `risk_level` (yes/no and choice settings only; numbers are always yours) | goal and termination, tool discipline, state, reliability, security, human oversight, observability |
| 3. multi-agent (14) | you write `topology`, `agent_registry`, `declared_delegation`, `conflict_resolution`; the rest have defaults | delegation depth, authority, cycles, cross-agent trust, budgets |
| 4. derived (2) | computed, never written | `trace_coverage`, `enforceability` |

`agenticreq params` lists every field. `agenticreq rules` lists every rule.

## Result groups

There is no overall score. Each result lands in one of four groups:

| group | meaning |
|---|---|
| Spec vs code gaps | the spec says X and the code does not do it, or a required field is missing or `TODO` |
| Human sign-off | only a person can confirm it; answered under `confirmed:`, never counted as passed |
| Check with a mock run | a run of the agent could show it; each line ends with the next command |
| Passed | read from the code, or seen in every captured run |

## Exit codes

| code | meaning |
|---|---|
| `0` | the `--fail-on` condition passed (`never` always exits 0) |
| `1` | it failed: `gaps` (default) a spec vs code gap; `all` also an open sign-off or a check waiting for a mock run |
| `2` | the spec is invalid; nothing was checked |
| `3` | no agent or readable code was found; nothing was checked |

## Editor support

`agenticreq init` writes `spec.schema.json` next to `spec.yml` and starts the file with this line, which VS Code and
other `yaml-language-server` editors read for completion and wrong-field warnings:

```yaml
# yaml-language-server: $schema=./spec.schema.json
```

For an existing spec: `agenticreq export-schema > spec.schema.json`.

## Limits

- **Dynamic agents and tools.** Agents and tool lists built at run time (from config, a registry, a loop) cannot be
  read from the code. A mock run is the evidence for them; what no run reaches is never reported as a pass.
- **CodeAgent.** An agent that writes and runs Python as its action (smolagents `CodeAgent`) has no tool call to read;
  its result waits for a mock run and is never a pass.
- **MCP only through `--mcp`.** MCP server tools are seen only with `--mcp "<server command>"` or `--mcp-tools <file>`.
- **Numbers are never suggested.** The library compares the numbers you write with your code; it proposes none
  ([docs/numbers.md](docs/numbers.md)).
- **Fields only a person can confirm** (for example `role_boundary.capabilities`, `untrusted_inputs[].kind`) become
  sign-off questions, never a pass.
- **Mock runs** follow one path each and use a canned model, so they show what the agent offers and in which order,
  never how a real model behaves. Docker images and remote interpreters are out of scope.
- **Safety.** `mock-run` runs your agent with mock tools. Use it only on projects you trust.

## Supported framework versions (`mock-run`)

`audit` and `check` never run the agent. `mock-run` captures at the HTTP-client boundary (httpx / httpx2); a version
outside the validated range still runs, with a note, and a run that captures nothing exits 1 with the reason.

| framework | supported by `mock-run` | validated on |
|---|---|---|
| CrewAI | crewai >=1.9.3,<2 | 1.9.3, 1.15.28 (0.86 hangs) |
| LangGraph | langgraph >=1.2,<2 | 1.2.15, 1.2.17 |
| OpenAI Agents | openai-agents >=0.23,<1 | 0.23.1, 0.24.0 |

Frameworks not in the table are not gated. Recognised frameworks include LangGraph, CrewAI, OpenAI Agents, smolagents,
PydanticAI, LangChain, AutoGen, LlamaIndex, Google ADK, Agno, DSPy, Haystack, Semantic Kernel and Strands.

## More

[Architecture](docs/architecture.md) · [GitHub Action](docs/github-action.md) · [Numbers](docs/numbers.md) ·
[Tools and mock runs](docs/tools-and-mock-runs.md) · [Rules that need a mock run](docs/code-pattern-rules.md) ·
[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Changelog](CHANGELOG.md)

## License

Apache-2.0.
