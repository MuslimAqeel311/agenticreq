# agenticreq

**AgenticReq supports both Specification-Driven Development and Existing-Code Auditing for Agentic AI applications.**

You write what your agent must and must not do in `spec.yml`. agenticreq reads the agent's code, without running it,
and tells you which requirements the code enforces, which it does not, and which need a person or a test run to
confirm. No LLM is called.

**agenticreq never suggests numbers. You write them; it checks your code matches.**

## What it is, and what it is not

| it is | it is not |
|---|---|
| a check that the code matches the spec you wrote | a runtime guardrail: it does not stop the agent while it runs |
| a list of where enforcement is missing, with file and line | a security certificate: a clean report is not a guarantee |
| a way to see which tools a model is offered, with no API key | a prompt-injection detector |

It can also export the spec as a policy for OPA or Cedar.

## Install

```bash
pip install agenticreq
agenticreq version
```

Python 3.10 or newer, on Windows, Linux and macOS. Installing writes nothing into your project.

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

## Two ways to start

| NEW project: write the spec first | EXISTING project: audit the code you have |
|---|---|
| 1. `agenticreq init` in the empty folder | 1. `agenticreq audit`: findings with no setup |
| 2. Fill `spec.yml`: what the agent must and must not do | 2. `agenticreq init`: drafts `spec.yml` from your code |
| 3. Build the agent to match it | 3. Review the draft and replace every `TODO` |
| 4. `agenticreq check` as you go | 4. `agenticreq check` |

A short spec looks like this:

```yaml
agent_id: support-agent
risk_level: medium
approval_required:
  - {tool: issue_refund, approver: lead@example.com, timeout: 1h, on_timeout: deny}
prohibited:
  - {tool: delete_customer, reason: never from a chat}
bounds: {max_steps: 6}
```

`agenticreq params` lists every field and its shape.

## The 5 steps, the 4 report groups and the exit codes

| step | command | what it does |
|---|---|---|
| 1. audit | `agenticreq audit` | quick scan, no spec needed |
| 2. spec | `agenticreq init` | writes `spec.yml`; you replace every `TODO` |
| 3. check | `agenticreq check` | compares `spec.yml` with the code |
| 4. mock-run | `agenticreq mocks`, then `agenticreq mock-run` | runs the agent with mock tools |
| 5. sign-off | edit `confirmed:` in `spec.yml` | a person confirms what code cannot show |

There is no overall score. Each result lands in one of four groups:

| group | meaning |
|---|---|
| Spec vs code gaps | the spec says X and the code does not do it, or a required field is missing or `TODO` |
| Human sign-off | only a person can confirm it; answered under `confirmed:`, never counted as passed |
| Check with a mock run | a run of the agent could show it; each line ends with the next command |
| Passed | read from the code, or seen in every captured run |

| exit code | meaning |
|---|---|
| `0` | the `--fail-on` condition passed (`never` always exits 0) |
| `1` | it failed: `gaps` (default) a spec vs code gap; `all` also an open sign-off or a check waiting for a mock run; also when the mock run is older than the code |
| `2` | the spec is invalid or missing; nothing was checked |
| `3` | no agent or readable code was found; nothing was checked |

## Sign-off is a statement, not proof

Some things only a person can confirm, for example that a kill switch exists outside the code. You answer them under
`confirmed:` in `spec.yml`:

```yaml
confirmed:
  - clause: kill_switch
    answer: yes
    by: lead@example.com
    date: 2026-11-01
    expires: 2027-05-01
    evidence: "agent.py:42 checks AGENT_KILL_SWITCH every step"
```

This is a person's statement on the record. agenticreq checks that the entry is complete, not expired, and that a
`file:line` in it points at a real line. It does not check what that line says, and it cannot check who wrote `by:`.
A wrong "yes" closes the item. Treat sign-offs like code review: someone accountable reads them.

`reason:` and `evidence:` are interchangeable; write at least one. A value with a comma or a colon must be in quotes.

## The mock run

`agenticreq mock-run` runs your agent with a canned model: no API key, no real LLM, and the network blocked. It
shows which tools the model is offered and in what order. It never shows how a real model would behave.

- Tools that write, pay, send or delete, and tools that reach a database or the network, are replaced in `mocks.yml`
  (`patch:` + `returns:`), so the real function never runs.
- Writes outside `allow_write:` and starting other programs are blocked; the report lists them.
- If you change the code after a run, the run's results go back to "Check with a mock run" until you run it again.
- Use it only on projects you trust: it runs the agent's own code.

## Limit the scan

By default the whole folder is scanned. When one folder holds several agents or scripts, scope it:

```bash
agenticreq check --entry agent.py        # start from agent.py; the files it imports are read too
```

or in `spec.yml`:

```yaml
include: [agent.py]
exclude: [scripts/, "**/batch_*.py"]
```

The report header says what was scanned.

## Limits

- **Dynamic agents and factories.** Agents and tool lists built at run time (from config, a registry, a loop) cannot
  always be read from the code. A mock run is the evidence for them; what no run reaches is never a pass.
- **Prompt-only code agents.** An agent that writes and runs Python as its action (smolagents `CodeAgent`) has no
  tool call to read; its result waits for a mock run and is never a pass.
- **Transports that cannot be intercepted.** The mock run captures model calls made through httpx, requests and
  aiohttp. A provider reached another way is named ("mock run is not supported for ...") and its items go to sign-off.
- **Frameworks whose tool schema cannot carry a numeric cap.** A limit on a tool argument is then checked in the
  tool body; the report says the schema cannot carry it.
- **Values only known at run time.** A limit read from the command line or an unset environment variable goes to
  sign-off. A name with several different values in the code is never guessed: it goes to sign-off too.
- **MCP only through `--mcp`.** MCP server tools are seen only with `--mcp "<server command>"` or `--mcp-tools <file>`.
- **Numbers are never suggested** ([docs/numbers.md](docs/numbers.md)).
- **Environment variables in the spec are never expanded.** A value that is only `$NAME` or `${NAME}` is reported as a
  placeholder; text that contains a `$` is kept exactly as written.
- **Docker images and remote interpreters** are out of scope for the mock run.

## Supported frameworks

LangGraph, LangChain, CrewAI, OpenAI Agents, Claude Agent SDK, Google ADK, smolagents, AutoGen, LlamaIndex,
PydanticAI, Haystack, Agno, Semantic Kernel, DSPy and Strands, plus plain provider SDK code (anthropic, openai,
google-genai, groq and other OpenAI-compatible clients). How each one pauses for a person is listed in
`src/agenticreq/data/approval_gates.yml`.

`audit` and `check` never run the agent. For `mock-run`, the framework versions it was validated on live in
`src/agenticreq/data/supported_versions.yml`; a version outside the range still runs, with a note:

| framework | supported by `mock-run` | validated on |
|---|---|---|
| CrewAI | crewai >=1.9.3,<2 | 1.9.3, 1.15.28 (0.86 hangs) |
| LangGraph | langgraph >=1.2,<2 | 1.2.15, 1.2.17 |
| OpenAI Agents | openai-agents >=0.23,<1 | 0.23.1, 0.24.0 |

## Platforms

Windows, Linux and macOS. OS-specific behaviour is limited to the mock run:

- Paths are compared case-insensitively on Windows, with long-path (`\\?\`) and UNC prefixes understood.
- The run gives the agent its own scratch home: `APPDATA` / `LOCALAPPDATA` on Windows, `~/Library/...` on macOS,
  `XDG_*` on Linux. All are created before the run starts.
- The platform lookups SDKs make for an OS header (`cmd /c ver`, `uname`, `sw_vers`) never start a program.

## Editor support

`agenticreq init` writes `spec.schema.json` next to `spec.yml` and starts the file with this line, which VS Code and
other `yaml-language-server` editors read for completion and wrong-field warnings:

```yaml
# yaml-language-server: $schema=./spec.schema.json
```

For an existing spec: `agenticreq export-schema > spec.schema.json`.

## More

[Architecture](docs/architecture.md) · [GitHub Action](docs/github-action.md) · [Numbers](docs/numbers.md) ·
[Tools and mock runs](docs/tools-and-mock-runs.md) · [Rules that need a mock run](docs/code-pattern-rules.md) ·
[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Changelog](CHANGELOG.md)

## License

Apache-2.0.
