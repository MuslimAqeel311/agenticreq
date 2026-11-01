# agenticreq

**Does the agent's code actually enforce what its spec declares?** agenticreq answers with a
measured confidence for every claim, and without calling any LLM.

**agenticreq never suggests numbers. You write them; it checks your code matches.**

agenticreq is **not** a runtime guardrail, firewall, eval framework, prompt-injection defence, or
observability tool. It finds where enforcement is missing, and exports policy to where
enforcement actually happens (OPA / Cedar / tool schemas).

Existing tools filter at runtime (NeMo, Guardrails AI), enforce policy (OPA, Cedar, MCP
gateways), or measure capability (evals). None of them compares the *declared requirement* with
the *implementation*. agenticreq does.

## Install

To try it on your own agents and judge every finding, see [TESTING.md](TESTING.md).

```bash
pip install agenticreq              # runtime: pydantic, pyyaml, rich, typer; writes nothing into your project
pip install -e ".[dev]"             # from a checkout: + pytest, ruff, mypy, build tools
agenticreq version
```

Python ≥ 3.10. Capture hooks `httpx.Client/AsyncClient.send` **and**
`httpx2.Client/AsyncClient.send`:

| SDK | HTTP client | tested here |
|---|---|---|
| `openai < 3` | httpx | 1.83.0 (via CrewAI) |
| `openai >= 3` | httpx2 | 3.31.0 |
| `anthropic < 1` | httpx | — |
| `anthropic >= 1` | httpx2 | 1.14.0 |
| supported ranges | `httpx >= 0.25`, `httpx2 >= 2.0` | httpx 0.28.1, httpx2 2.13.1 |

## Quick start

Value first, configuration second. Install agenticreq into the agent's own `.venv`, then run these from the agent's
folder. No paths, no JSON on the command line:

```bash
pip install agenticreq     # 1. creates nothing in your project
agenticreq audit           # 2. findings immediately: no spec, no setup, AST only
agenticreq init            # 3. scaffolds spec.yml: _derived filled, TODO for the 12 user fields
agenticreq mocks           # 4. writes mocks.yml: the entry that runs the agent, its inputs, one entry per tool
agenticreq mock-run        # 5. runs the agent with a canned model and the tools in mocks.yml
agenticreq check           # 6. spec.yml + the latest mock run: Level 1 + 2 + 3
agenticreq guide           #    where this folder stands, and the exact next command
```

### The five steps and your progress

| step | what it means | command |
|---|---|---|
| 1. audit | quick scan, no setup | `agenticreq audit` |
| 2. spec | write what the agent must and must not do (spec.yml, no TODO left) | `agenticreq init` |
| 3. check | compare spec.yml with the project: no "Spec vs code gaps" left | `agenticreq check` |
| 4. mock-run | test run with mock tools, newer than the code, nothing left in "Check with a mock run" | `agenticreq mocks`, then `agenticreq mock-run` |
| 5. sign-off | owner confirms what the scan cannot see: nothing left in "Human sign-off" | entries under `confirmed:` in spec.yml |

`audit`, `init`, `mocks`, `mock-run` and `check` end with the same block (on stderr; `--quiet` hides it, and JSON /
SARIF never show it):

```
Your progress
  ✓ 1. audit      done
  ✓ 2. spec       done
  ✓ 3. check      spec matches
  ✓ 4. mock-run   test run with mock tools
  ✗ 5. sign-off   owner confirms 21 items

Next: add this under confirmed: in spec.yml, with your own details:
  - clause: owner
    answer: yes
    by: you@company.com
    date: 2026-10-28
    expires: 2027-04-28
    evidence: "Name of the person who owns this agent"

Help: agenticreq guide
```

`✓` is a step done, `✗` the step you are at; when every step is done it says `All 5 steps done`. `Next:` is one
sentence ending in the exact command (for step 5, the entry to paste, built from the first open item). The state is
read from the folder only (spec.yml, mocks.yml, `.agenticreq/captures.json` and a small `.agenticreq/state.json`
that records that audit ran and what the latest check found), never the network. `agenticreq guide` prints the
same state with a table of the five steps; `agenticreq` with no command prints a short overview.

`check` shows what matters at your step: while spec.yml still has TODO fields it shows the gaps and the TODO list,
and one line in place of the sign-offs ("Human sign-off: 26 items, shown after the spec is complete"); "Check with
a mock run" shows once the gaps are fixed. `check --all` shows every group now.

1. **`pip install`** writes no file into your project: no `spec.yml`, no `.agenticreq/`, no config,
   no cache. `import agenticreq` has no side effects either. Both are enforced by tests
   (`tests/test_onboarding.py`).
2. **`audit`** needs no spec and does not import your agent, so the agent's own dependencies need
   not be installed. It checks the **Level 2** engineering defaults (for `risk_level: medium`):

   | checked without a spec | rule |
   |---|---|
   | model version pinned | AS-210 |
   | write tools without an approval gate | AS-212 |
   | missing idempotency key on writes | AS-211 |
   | hard-coded secrets | AS-214 |
   | step / cost / duration bounds missing | AS-208 / AS-221 / AS-222 |
   | unbounded tools present (http / sql / shell) | AS-213 / AS-215 / AS-216 |
   | fail-open error handling (`except: pass`) | AS-219 |
   | self-modification tool present | AS-220 |
   | environment separation (config from env, no hard-coded prod URL) | AS-223 |
   | structured output | AS-224 |
   | audit logging / tracing hooks | AS-225 |

   It **cannot** check Level 1 or Level 3 without a spec: `prohibited`, `limits`,
   `approval_required`, `escalation`, `untrusted_inputs`, `role_boundary`, `declared_delegation`.
   agenticreq can't know what your business forbids, what the money threshold is, or which
   sub-agents you intended. audit says they need a spec (AS-907) and tells you to run
   `init` and then `check`. It never implies they passed.
   Approval gates are scoped: `needs_approval` / `requires_approval` on a tool, or `interrupt()` /
   `request_approval()` inside a tool's body, gate that tool only. CrewAI `Task(human_input=True)`
   reviews the final answer after the tools ran, so it is not counted as an approval gate.
   `audit --spec spec.yml` applies **only** that file's unexpired `waivers:`, so a deliberate
   exception doesn't show up on every audit. Everything else in the spec is ignored by `audit`.
3. **`init`** is the only command that creates `spec.yml`. Under each TODO it writes suggestions read
   from the code, as comments (risk level from the write tools, prohibited / approval candidates, numeric
   limits, untrusted sources, sensitive fields, and for multi-agent code the registry and the delegation
   as wired). The values themselves stay TODO: what is forbidden, and who approves, is your decision.
   **agenticreq never suggests numbers. You write them; it checks your code matches.** `init` writes no number
   (steps, cost, duration, retries, tool calls, progress guard, output size, confidence, delegation depth, timeouts):
   it lists their names in a comment, and a number left out is reported as "not set: write a value, or not_needed
   with a reason" ([docs/numbers.md](docs/numbers.md)). `init` refuses to overwrite an existing
   file unless you pass `--force`. Business intent is never auto-filled, and every TODO is a spec vs code gap
   (AS-101) until it is replaced.
4. **`mocks`** reads the code (the same extraction as `audit` / `init`) and writes `mocks.yml`: `entry:` the function
   that runs the agent (for example `finance_agents.agents:run_agent`; `TODO` with a comment when none is found),
   `args:` its inputs, `python:` empty (detected), `order:` every tool found, and one `tools:` entry per tool.
   Read-only tools get sample `args`; tools that write, pay, send or delete get `patch:` + `returns:` so the real
   function never runs; tools under `prohibited:` in spec.yml get `call: false`. It never overwrites `mocks.yml`
   unless `--force`.
5. **`mock-run`** runs the agent with a canned model (no real LLM, no API key) and the tools from `mocks.yml`, and
   saves what it sent to the model in `.agenticreq/captures.json`. It reads `entry`, `args` and `python` from
   `mocks.yml` (`--entry`, `--args`, `--python`, `--runs`, `--timeout`, `--mocks` override them). With `runs:` in
   `mocks.yml` it makes one mock run per entry and saves all of them; `--run <name>` runs only that one (repeat the
   flag for several) and `--keep` keeps the runs already saved that you did not run now. It finds the Python that has
   your agent's dependencies by itself ([table below](#which-python-mock-run-uses)) and prints
   `python: <path> (from <source>)`. Before it runs anything it checks that interpreter starts and can import your
   agent's framework; if not it stops and names the missing packages. agenticreq need not be installed there. It prints the calls captured, the tools offered
   (structured-output schemas such as `Plan` on their own line), the tools skipped by `call: false`, the tools never
   offered to the model with the reason (which agent no run reached, and the run to add under `runs:` in `mocks.yml`),
   the credentials withheld from the agent, and where it saved the run; it exits 1 when the run crashed, captured no
   LLM call, or changed the project's files. Without `mocks.yml` it stops: `mock-run needs a mocks file: run
   ``agenticreq mocks`` to create mocks.yml` (exit 2).
6. **`check`** compares the spec with the code and the latest mock run (read automatically; `--captures` reads
   another one), and reports Level 4 **enforceability**, for example
   *"you declared 9 rules; 2 enforced in code, 1 only via export (OPA/Cedar), 6 rely on the model or a human alone"*.

More:

```bash
# a mock run RUNS the agent's code. Real credentials are withheld and network / other programs / file writes are
# blocked at the Python level (see Safety below). It refuses to start without mocks.yml.
agenticreq mock-run                                          # -> .agenticreq/captures.json
agenticreq check --traces traces/ -f sarif -o agenticreq.sarif

# hand enforcement to the systems that do it
agenticreq export . --spec spec.yml -f opa   > policy.rego
agenticreq export . --spec spec.yml -f cedar > policy.cedar

# structural autofix: step cap, idempotency key, model pinning (dry-run by default)
agenticreq fix . --spec spec.yml            # prints a diff
agenticreq fix . --spec spec.yml --apply    # writes it

agenticreq stats . --spec spec.yml          # agent registry, aggregates, Level-4 derived values
agenticreq validate spec.yml                # AS-1xx only
agenticreq validate spec.yml --fix          # rename wrong field names (one clear match only), shows a diff, asks
agenticreq export-schema > spec.schema.json # JSON Schema: your editor completes and checks field names
agenticreq params                           # every parameter (Levels 1-4) and how it is verified
agenticreq rules                            # every rule id
agenticreq export-spec my_spec.py:SPEC      # Python builder -> spec.yml
```

## Exit codes

| code | meaning |
|---|---|
| `0` | the `--fail-on` condition passed (`never` always exits 0) |
| `1` | the condition failed: `gaps` (default) a spec vs code gap; `all` also an open sign-off or a check still waiting for a mock run |
| `2` | the spec is invalid (a wrong field name or a bad value); nothing was checked |
| `3` | no agent or readable code was found; nothing was checked |

## Editor support: `spec.schema.json`

The spec has a JSON Schema, shipped inside the package and in the repository root (`spec.schema.json`). `agenticreq init`
writes it next to `spec.yml` and starts the file with the line the YAML extension of VS Code (and other editors using
`yaml-language-server`) reads:

```yaml
# yaml-language-server: $schema=./spec.schema.json
agent_id: support-agent
```

You get completion for field names, a squiggle on a wrong one, and the note for each field on hover. Write it yourself
for an existing spec with `agenticreq export-schema > spec.schema.json`. A `TODO` or `not_needed` is accepted in any
field; `check` reports a `TODO` as a gap.

## A wrong field name in spec.yml

A field agenticreq does not know would never be checked, so `check` stops (exit 2) and names the closest field:

```
Unknown field `max_iterations` in spec.yml (line 12). Did you mean `bounds.max_steps`? Fix it, or run: agenticreq validate --fix
```

Names are compared after normalising them (lowercase, camelCase and kebab-case to snake_case, a trailing plural
`s` and filler words dropped), including nested fields (`max_steps` -> `bounds.max_steps`), then against the few
"also called" words each parameter has in the params catalog (`agenticreq params` lists them: `prohibited` is also
called `forbidden`, `banned`, `blocked`). Two equally close fields are both listed and never picked for you; with
no close field there is no guess. `validate --fix` renames only a key with exactly one clear match in the same
mapping, shows the diff, and asks (`--yes` writes without asking); comments stay. `init` writes
`spec.schema.json` and makes the first line of spec.yml
`# yaml-language-server: $schema=./spec.schema.json`, so VS Code completes field names and flags a wrong one.

## When a field does not apply: `not_needed`

Write `not_needed` as the value of any field, at any level and of any type (text, list, mapping, number, yes/no,
choice), or of a key inside one:

```yaml
conflict_resolution: not_needed
bounds: {max_cost_usd: not_needed, max_steps: 20}
confirmed:   # the reason; optional for validity, needed for the item to count as done
  - {clause: conflict_resolution, answer: not_needed, by: me@x.com, date: 2026-10-30, reason: "pipeline, no agent-vs-agent conflicts"}
```

The field counts as answered: no TODO, required-field or shape error. The report shows `NOT NEEDED  <parameter>
<reason>` under Human sign-off, never as a gap or a pass; without a `confirmed:` reason it shows `(no reason given)`
and stays in the "left" count. Not allowed for `agent_id`, `risk_level`, `goal`, `owner`, nor, at high or critical
risk, for `kill_switch`, `fail_closed`, `audit_log`, `version_pinning`, `self_modification`, `log_immutable`,
`transitive_prohibition` and `dry_run_required` (a spec error names the field and why). `[]` and `{}` still mean
"answered: nothing".

## Tools, limits and mock runs

How a tool's limits are found, how each tool is classified as READ, WRITE or UNCLEAR from its body (and
`tool_effects:` in spec.yml), how `allow_write:` works in a mock run, and every mock-run failure message:
[docs/tools-and-mock-runs.md](docs/tools-and-mock-runs.md). When agenticreq is not sure, the result goes to
"Check with a mock run" with the reason: never a gap, never a pass.

## Limits

- **Numbers.** agenticreq never suggests numbers. It compares the numbers you write in spec.yml with your code, and
  shows a number it found in your code as evidence ([docs/numbers.md](docs/numbers.md)).
- **Fields only a person can confirm.** The code cannot show these, so each becomes a plain sign-off question under
  `confirmed:` (never a PASS): `role_boundary.capabilities` ("Does the agent only do what is listed in
  role_boundary.capabilities?"), `role_boundary.data_access` ("Does the agent only read the data listed in
  role_boundary.data_access?"), `untrusted_inputs[].kind`, `output_size_limits.max_items`,
  `write_safety.compensating_action` and `agent_registry[].role`.
- **Dynamic agents and tools.** Agents and tool lists built at run time (from a config file, a registry, a loop over
  names, `getattr`) cannot be read from the code. They stay absent from the static result; a mock run is the evidence
  for them, and what no run reaches is reported under "Check with a mock run", never as a pass.
- **CodeAgent.** An agent that writes and runs Python as its action (smolagents `CodeAgent`) has no tool call to read:
  its result waits for a mock run and is never a PASS.
- **MCP only through `--mcp`.** Tools served by an MCP server are seen only when you pass `--mcp "<server command>"`
  (agenticreq runs `tools/list` against it) or `--mcp-tools <file>` (a saved `tools/list` result). Otherwise they are
  not part of the check.
- **Numbers are never suggested.** agenticreq compares the numbers you write in spec.yml with your code; it never
  proposes a limit, a threshold or a timeout.
- **Mock runs** follow one path each; agents the inputs never reach need a run you write
  ([Limits of mock runs](#limits-of-mock-runs)).
- **Environments**: Docker images and remote interpreters are out of scope
  ([Limits of environment detection](#limits-of-environment-detection)).

## Safety

mock-run runs your agent with mock tools. Run it on projects you trust; for a project from someone else, use a separate machine.

## The spec: four levels

| level | who writes it | parameters |
|---|---|---|
| **1. user** (12) | you. These are business decisions; `check` fails on TODO. | `agent_id` · `risk_level` · `goal` · `out_of_scope` · `role_boundary` · `prohibited` · `approval_required` · `limits` · `escalation` · `untrusted_inputs` · `owner` + `technical_owner` · `data_rules` |
| **2. agent** (41) | the library, from `risk_level` (override when needed; yes/no safeguards weaker than the risk floor are AS-111; yes/no and choices only: every number is yours, [docs/numbers.md](docs/numbers.md)) | goal & termination · tool discipline · state & context · reliability · security & trust · human oversight · observability & change safety |
| **3. multi-agent** (14) | you write `topology`, `agent_registry`, `declared_delegation`, `conflict_resolution`; the rest have defaults | delegation depth, authority inheritance, transitive prohibition, cycles, cross-agent trust, aggregate budget, ... |
| **4. derived** (2) | computed, never written | `trace_coverage` · `enforceability` |

`agenticreq params` prints every parameter and what verifies it: a rule that reads the code
(`code`), validation of the spec itself (`spec`), real runs (`runtime`: traces or real-model captures
check `stop_conditions`, `progress_guard`, `bounds` and `rate_limits`, AS-228..231), or a human
sign-off (`sign-off`, AS-501) for clauses that live in infrastructure or prompts: a person confirms them under
`confirmed:` in spec.yml. Runtime clauses fall back to a sign-off when there are no real runs. A mock run is answered
by a canned model, so it shows which tools the agent offers and in which order, never how a real model behaves.

**Frameworks recognised (23):** langgraph, crewai, openai-agents, smolagents, pydantic-ai, langchain, autogen,
microsoft-agent-framework, llamaindex, google-adk, agno, camel, claude-agent-sdk, dspy, haystack, semantic-kernel,
strands, swarm, browser-use, needle, mcp-server, anthropic-sdk, openai-sdk. Code using none of them is still scanned
for plain tool functions; the framework is named from what the code imports (`from google import adk`,
`importlib.import_module("crewai")` and aliases included; a project's own `agents/` package is not the OpenAI SDK).
A framework that is only declared in requirements / pyproject / setup / Pipfile / conda files is reported as
`declared, not imported` (a guess: its built-in defaults are not assumed), and a mock run that
loads one names it as observed. `AS-904` still says so when nothing is recognised.
`data/supported_versions.yml` lists the versions `mock-run` was validated on.

**Level 3 across frameworks.** Delegation is read from each framework's own idiom: OpenAI Agents
`handoffs=` and `agent.as_tool()`, Google ADK `sub_agents=` and `AgentTool(agent=)`, AutoGen teams
(`RoundRobinGroupChat`, `SelectorGroupChat`, `Swarm` with `handoffs=["name"]`) and 0.2
`initiate_chat`, LlamaIndex `AgentWorkflow` with `can_handoff_to=`, LangGraph `create_supervisor` /
`create_handoff_tool`, smolagents `managed_agents=`, a tool whose body runs another agent (PydanticAI
agent delegation, any framework), and CrewAI `allow_delegation` and hierarchical crews. On the wire,
`transfer_to_*`, ADK `transfer_to_agent(agent_name=)` and LlamaIndex `handoff(to_agent=)` give observed
edges. A team is a coordinator node: it routes work but grants no authority, and a
teammate's tools count as reachable for transitive prohibition (AS-304).

**CI.** `action.yml` is a GitHub Action: it runs `check` (or `audit`), writes SARIF with repo-relative
paths, uploads it to code scanning so findings annotate the PR, and fails per `--fail-on`
(`gaps` | `all` | `never`). See [docs/github-action.md](docs/github-action.md).

**Exit codes.** `--fail-on gaps` (default): `1` when there is a spec vs code gap. `--fail-on all`: `1` also when a
sign-off is still open or a check waits for a mock run. `--fail-on never`: always `0`. `2` the spec is invalid; `3`
nothing was checked (no agent or readable code found). A result that waits for a mock run or a sign-off is never a
pass. A test asserts that every spec field appears in that catalog, so no parameter goes silently unchecked.

### Why YAML is canonical

`spec.yml` is the source of truth. The Python builder API is a way to *produce* it, and it always
writes YAML (`SpecBuilder(...).write("spec.yml")` or `agenticreq export-spec my_spec.py:SPEC`):

- **It is readable where the agent's code can't run**: CI without credentials, a different Python
  environment, a compliance or security review. Reading a Python spec means importing its module,
  with all its side effects and dependencies. YAML has neither.
- **A clean git diff** shows who changed a limit, and when.
- **Non-engineers can read and edit it.** A procurement lead or a QC manager owns these
  requirements, so they need to be able to review them.

`mocks.yml` (`agenticreq mocks` writes it; edit it freely):

```yaml
entry: myapp.agent:run                         # the function that runs the agent
args: {question: "What can you help me with?"} # its inputs
python:                                        # empty: found for you (see "Which Python mock-run uses")
order: [get_inventory, submit_order]           # the order the canned model asks for tools
tools:
  get_inventory: {args: {sku: A-100}}
  issue_refund: {call: false}                  # never ask for this one
  pay_invoice: {patch: "myapp.tools:pay_invoice", returns: {status: ok}}  # the real function never runs
final_text: "Done."
human_input: "no"                              # what an input() approval prompt is answered
env: {AGENT_MODEL: gpt-4o-2024-08-06}          # non-secret settings (live-looking keys are refused)
allow_write: [data]                            # folders the agent may write to during the run
```

A `patch:` on a tool object (a LangChain `@tool`, a CrewAI tool) replaces the function inside it, so lists the agent
built at import time get the stub too.

#### Several mock runs: one per task

One mock run follows one path. In a router (`Triage` handing off to `Accounts`, `Fraud` or `Loans`) a run reaches
one specialist, so the tools of the others are "never offered to the model". List one run per path under `runs:`.
The top level is the default for every run; a run changes only what it names (`name`, `entry`, `args`, `order`,
`tools` with `args:` / `call:`, `structured`, `final_text`, `human_input`, `max_llm_calls`):

```yaml
entry: bank.agent:run
args: {task: "What is my balance?"}
order: [transfer_to_accounts, transfer_to_fraud, transfer_to_loans, get_balances, flag_transaction, loan_estimate]
tools:
  get_balances: {args: {customer_id: c1}}
  loan_estimate: {args: {amount: 1000}}
runs:
  - name: accounts
    order: [transfer_to_accounts, get_balances]      # the canned model picks the handoff from order:, not from the task
  - name: fraud
    args: {task: "I do not recognise a charge"}
    order: [transfer_to_fraud, flag_transaction]
  - name: loans
    args: {task: "Can I get a loan?"}
    order: [transfer_to_loans, loan_estimate]
```

```bash
agenticreq mock-run                           # every run; saves them all in .agenticreq/captures.json
agenticreq mock-run --run loans               # only loans; the file now holds just that run
agenticreq mock-run --run loans --keep        # run loans again, keep accounts and fraud as saved
```

The output has one `mock run <name>: <n> LLM calls: <tools>` line per run, then the combined `captured` and `tools
offered to the model` lines. `check` reads every saved run: a tool is offered when any run offered it. A file without
`runs:` is one run, exactly as before. `call: false` only means "do not ask the model to call it": a `call: false`
tool that no run reached is still listed under "never offered to the model", with both facts in the line.

#### Limits of mock runs

- The canned model is deterministic: it requests each offered tool once, in `order:`. It does not read the task, so a
  handoff the model chooses at run time (a router, a `RouteDecision` schema, `TOOLSETS[name]`) is reached only by the
  `order:`, `structured:` or `args:` you write for it. agenticreq names the agent no run reached; it cannot write that
  run for you.
- An agent nothing hands off to, a branch behind a condition the inputs never meet, or an agent built from data the
  code does not show stays under "Check with a mock run" until you add a run whose task reaches it.
- `--keep` merges by run name: a run run again replaces its earlier capture, the others stay as saved (a file
  from an older agenticreq is kept by the run ids in it).

### Which Python mock-run uses

`mock-run` runs your agent with the interpreter that has the agent's own dependencies, whatever tool made it. The
first match wins, and the output always says which one it used: `python: <path> (from <source>)`.

| # | Source | What is read |
|---|---|---|
| 1 | `--python <interpreter>` | the flag |
| 2 | `python:` in mocks.yml | the file |
| 3 | the active environment | `$VIRTUAL_ENV`, then `$CONDA_PREFIX` (an active conda `base` is skipped: pass `--python` to use it) |
| 4a | a folder of the project | any top-level folder holding `pyvenv.cfg`: `.venv`, `venv`, `env`, uv, virtualenv, Poetry in-project, any name |
| 4b | Poetry | `poetry env info --executable`, when `poetry` is installed and the project uses it |
| 4c | Pipenv | `pipenv --venv`, when `pipenv` is installed and there is a `Pipfile` |
| 4d | Conda | `name:` in `environment.yml`, matched against `conda env list`, when `conda` is installed |
| 4e | pyenv / asdf | `.python-version` (`~/.pyenv/versions/<v>`), `.tool-versions` (`~/.asdf/installs/python/<v>`) |
| 5 | agenticreq's own interpreter | with a warning: the agent's dependencies must be installed there |

An interpreter is `bin/python`, or `Scripts/python.exe` on Windows; a folder without a working one is skipped.
poetry, pipenv and conda are called only when present, read-only, with a short timeout, and any failure falls through
to the next source. agenticreq never installs or creates anything.

Before the run it starts the interpreter and checks it can import the agent's main framework (from the entry file's
imports and `requirements.txt` / `pyproject.toml` / `Pipfile`). If not, it stops before the run and says which Python
it used and why, the missing packages, and `<python> -m pip install -r requirements.txt`. A Python version outside
`requires-python` (or `.python-version`) is a warning, not a failure. agenticreq does not have to be installed in the
agent's environment; the agent's environment needs `pydantic` (almost every framework installs it).

#### Limits of environment detection

- Docker images, dev environments in the cloud and remote interpreters are out of scope: mock-run runs a local
  Python. Install agenticreq where the project runs, or pass `--python` for a local interpreter.
- Only the environment folders listed above are found. An environment kept elsewhere (a shared `~/envs/x`, a tox
  folder) needs `--python` or `python:` in mocks.yml.

### mock-run: supported framework versions

`mock-run` captures at the HTTP-client boundary (httpx / httpx2), not inside the framework, so it runs on any
framework version. The table (`src/agenticreq/data/supported_versions.yml`) lists the ranges it was validated on:
a version outside a range still runs, with a note; only a version known to break a mock run (`known_bad`) is
refused before anything runs (exit 2). A run that captures nothing, crashes or misses LLM calls exits 1 with
the reason, so an untested version can never look like an all-clear. A run that does not finish within
`--timeout` (default 120 s) is stopped with a clear message, never a traceback.

| framework | supported by `mock-run` | validated on |
|---|---|---|
| CrewAI | crewai >=1.9.3,<2 | 1.9.3, 1.15.28 (0.86 hangs) |
| LangGraph | langgraph >=1.2,<2 | 1.2.15, 1.2.17 |
| OpenAI Agents | openai-agents >=0.23,<1 | 0.23.1, 0.24.0 |

Also captured cleanly (not gated): smolagents 1.26.0, pydantic-ai 2.61.0, AutoGen AgentChat 0.7.5
(`python validation/val_frameworks.py <framework>=<python of that env> ...` re-runs this on any version).

`audit` and `check` never run the agent. The framework version they compare against the table is the agent's:
a virtual or conda environment inside the project, then a lock file (`uv.lock`, `poetry.lock`, `pdm.lock`,
`Pipfile.lock`), then an exact pin. agenticreq's own Python counts only when it matches what the project
declares. A range such as `crewai>=0.100` is not a version: AS-908 fires only when the range allows no
validated version.

Frameworks not in the table are not gated.

## Result groups (the foundation)

There is no overall score. Every result lands in exactly one of four groups, one line per parameter:

```
agenticreq check · finance-multi-agent-system · spec: spec.yml · mock run: 2026-10-27 11:56 (6/6 calls)

Result: FAIL    10 gaps · 11 sign-offs left (10 signed) · 9 to check · 6 passed

SPEC VS CODE GAPS (10)
  ✗ bounds.max_steps        Code allows 60 steps; spec.yml says 25.

HUMAN SIGN-OFF (11 left · 10 signed)
  SIGN-OFF  escalation      Who handles hard cases? Confirm under confirmed: in spec.yml.
  10 signed, use --verbose to list

CHECK WITH A MOCK RUN (9)
  ? suggest_rebalance       Never reached the model. Add it to order in mocks.yml, then run: agenticreq mock-run

PASSED (6)
  ✓ prohibited              Held in 1 of 1 runs: execute_real_trade (never offered to the model).
```

| group | status | what goes here |
|---|---|---|
| Spec vs code gaps | `✗` GAP | the spec says X and the code does not do it, or a required spec field is missing or TODO |
| Human sign-off | SIGN-OFF / SIGNED / NOT NEEDED | things code cannot show (owner, escalation, kill_switch, data_rules, ...): a person answers them under `confirmed:` in spec.yml (see below). A signed item stays here as SIGNED and is never counted as passed; it opens again when its `expires` date passes ("sign-off expired") or a file named in its `evidence` (`file:line`) changed after its `date` ("file changed since sign-off") |
| Check with a mock run | `?` CHECK | things a run of the agent could show: a code-pattern result, a tool that never reached the model, a number agenticreq cannot read from code. Each line ends with the next command |
| Passed | `✓` PASS | read from the code, or seen in every captured run |

Each entry under `confirmed:` answers one item with `answer:` (missing means `yes`):

| answer | needs | shown as |
|---|---|---|
| `yes` | `by`, `date`, `expires`, `evidence` | SIGNED |
| `no` | `by`, `date`, `reason` | a gap: `✗ kill_switch  Owner says this is not in place yet: <reason>` (exit 1) |
| `not_needed` | `by`, `date`, `reason` | `NOT NEEDED  kill_switch  <reason>` in Human sign-off; counts as done |

An entry missing a field its answer needs stays an open SIGN-OFF ("evidence is missing"). At `high` or `critical`
risk, `kill_switch`, `fail_closed`, `self_modification`, `log_immutable`, `version_pinning`, `audit_log`,
`dry_run_required` and `transitive_prohibition` cannot be `not_needed`: that is a spec error ("kill_switch cannot be
not_needed at high risk", exit 2).

1. A result that waits for a mock run or a sign-off never becomes a PASS. The model layer raises if you try.
2. Counts cover only the extracted portion, and the report says how much that was.
3. The header names the command, the agent, the spec and the mock run; the Result line is always second; empty
   groups are hidden. A long line is cut at the terminal width with `…`, never wrapped. `NO_COLOR` and a non-TTY
   give plain text; a terminal that cannot print `✓ ✗ · …` (a Windows cp1252 console) gets `+ x - ...`.
   `--verbose` adds rule ids, file:line, every signed item and the mock run file.
4. Traces prove violations, not safety: "violation found" vs "held in N of N runs".

Several rules reach only a code pattern from code, and some have no route to PASS at all: see
[docs/code-pattern-rules.md](docs/code-pattern-rules.md).

Findings are grouped under 10 quality attributes: Goal Integrity, Boundedness, Least Privilege,
Autonomy Safety, Human Oversight, Recoverability, Trust Boundaries, Delegation Integrity,
Accountability, Change Safety.

## Rules

| range | what | examples |
|---|---|---|
| AS-1xx | spec validation | TODO left (101), tool prohibited *and* allowed (103), approver missing (104), limit contradiction (105), expired waiver (106), prose escalation (108), safeguard override below its risk floor (111; numbers have no floor) |
| AS-2xx | spec ⟷ code | never reached the model → waits for a mock run (201), undeclared tool on wire (202), prohibited tool reachable (203) / invoked (204), approval gate missing (205), limit not in schema (207), unpinned model (210; for providers with no dated snapshots, such as Groq, it reports "can change without a code change" as a mock-run check instead of impossible pinning advice; see `data/provider_pinning.yml`), write w/o idempotency (211), hard-coded secret (214), fail-open tool (219), self-modification tool (220), no cost / time cap (221 / 222), no environment separation (223), no structured output (224), no audit hook (225), no checkpointer (226), write without dry-run (227) |
| AS-3xx | delegation | undeclared edge (301), depth (302), cycle (303), **transitive prohibition `A → B → C can reach X`** (304), authority escalation (305); each `agent_registry` entry is enforced on that agent itself: its own prohibited tool (203) or a tool its entry does not list (206); a cycle written into `declared_delegation` (303) |
| AS-4xx | taint | untrusted source → LLM → un-gated write (401 static, 402 observed) |
| AS-5xx | human sign-off | clause code can't show (501), signed off (502) |
| AS-6xx | schema fidelity drift | description dropped (602), optional sent as required (603), **contradictory contract: required + default on the wire (604, blocker)**, default dropped (605), constraint dropped (609), required sent as optional (610) |
| AS-9xx | meta | prompt-only / CodeAgent → waits for a mock run (901), extraction failed (902), no runtime evidence (903), audit without a spec: Level 1/3 need a spec (907) |

Every schema-fidelity finding starts its evidence with the framework version that produced the
wire, e.g. `crewai 1.9.3 (runtime) | ...`. Rewriting changes between framework versions (see
SCOPE.md L20), so fidelity is measured on every run and never compared with a stored table.

Example AS-603: *"code says warehouse is optional (default 'KHI') with description 'Warehouse
code.'; the model is told it is required with no description"*.

## Framework × check matrix

Verified on fixture agents with the canned responder, macOS, Python 3.11, **and** on the
boundary-capture spike's own 7 agents with the spike's recorded S2 captures
(`tests/test_spike_acceptance.py`; numbers in [SCOPE.md](SCOPE.md)).

| framework (tested version) | tool extraction (AST) | boundary capture | declared ⟷ observed | schema fidelity | execution model | delegation graph | approval gates |
|---|---|---|---|---|---|---|---|
| LangGraph 1.2.15 + langchain-openai 1.7.2 | proven (`@tool`, `ToolNode`, `bind_tools`) | 4/4 (httpx2) | proven / mock-run check for never-reached; static graph reachability | proven, found: `Args:` descriptions dropped | schema-based | not exercised | code pattern (`interrupt`, `interrupt_before`) |
| OpenAI Agents 0.23.1 | proven (`@function_tool`) | 5/5, 6/6 (httpx2, Responses API) | proven; `transfer_to_*` allowlisted | proven, found: optional→required while keeping the default (AS-604) | schema-based | code `handoffs=` proven; wire edges code pattern | code pattern (`needs_approval`, guardrails) |
| CrewAI 1.9.3 | proven (`@tool`, `BaseTool`) | 6/6 (httpx, openai 1.83) | proven; delegate tools allowlisted | proven, found: 9/9 param descriptions dropped, required + default (AS-604) | schema-based | code pattern (`delegate_work_to_coworker`) | code pattern (`human_input`) |
| smolagents 1.26 ToolCallingAgent | proven (`@tool`, `Tool`) | 3/3 (httpx2) | proven; `final_answer` allowlisted | proven, found: default dropped | schema-based | not exercised | — |
| smolagents 1.26 CodeAgent | proven | 1/1 (httpx2) | **waits for a mock run, never PASS** | n/a (`tools: []`) | **prompt-only → AS-901** | — | — |
| PydanticAI 2.56 (Anthropic) | proven (`@agent.tool`, `tool_plain`) | 3/3 (httpx2) | proven | proven, faithful | schema-based | not exercised | — |
| raw anthropic 1.14 `beta.messages` | proven (dict schema) | 3/3 (httpx2) | proven | n/a (hand-written schema) | schema-based | — | — |
| raw openai 3.31 async + streaming | proven (dict schema) | 2/2 (httpx2, SSE) | proven | n/a | schema-based | — | — |
| AutoGen, LlamaIndex, Google ADK, Agno, Haystack, Semantic Kernel | code pattern: tool idioms covered by AST unit tests only (no real framework installed) | if they use httpx/httpx2 | untested | untested | code pattern | untested | untested |

## Architecture

```
spec.yml + profiles/          agent code  (+ local modules it imports)
      │                            │
      ▼                            ▼
   loader  ──► AgentSpec     ast/symbols ─► ast/extractor ──┐   (one tool resolver)
 (extends merge)   │          capture/ (S2, importers) ─────┤──► ObservedAgent
                   │          mcp/ ─────────────────────────┤
                   │          traces/ ──────────────────────┘
                   └──────► rules/ ◄────────────┘
                              ├── validate     AS-1xx      ├── taint    AS-4xx
                              ├── conformance  AS-2xx      ├── attest   AS-5xx
                              ├── hygiene      AS-219..227 ├── meta     AS-9xx
                              ├── fidelity     AS-6xx      └── params.py: Level 1-4 catalog
                              └── graph        AS-3xx
                                    ▼
                         report/ (console | json | sarif) ──► export/ (opa | cedar | schema)
```

`models/` depends only on Pydantic, and everything else depends on it.

**Static extraction** has two layers. `ast/symbols.py` parses the scanned files plus every local
module they import (transitively; relative imports, aliases, re-exports, `import pkg.mod`), and
resolves a name to its definition. Framework packages are never followed unless their source lies
inside the scanned root. `ast/extractor.py` then sends every tool expression through one resolver
(`_tools_from_expr`), so each idiom is recognised wherever it appears:

| idiom | frameworks |
|---|---|
| `@tool`, `@function_tool`, `@agent.tool`, `@kernel_function`, `@register_for_llm` | LangChain, OpenAI Agents, PydanticAI, smolagents, Strands, Semantic Kernel, AutoGen |
| `tool(f)`, curried `tool(parse_docstring=True)(f)`, `tool("Name")(f)` | LangChain, CrewAI, smolagents |
| `StructuredTool.from_function(func=f)`, `FunctionTool.from_defaults(fn=f)`, `Tool(func=f)`, `register_function(f, ...)` | LangChain, LlamaIndex, Haystack, AutoGen |
| bare functions in `tools=[...]` / `bind_tools([...])` | PydanticAI, Google ADK, Agno, AutoGen, LangGraph |
| tool classes, including subclasses of project / `*Tool` bases (`RagTool`) | CrewAI, LangChain, smolagents |
| pydantic models / TypedDicts in a tool list | LangChain `bind_tools([Model])` |
| agents as tools: `agent.as_tool(tool_name=...)`, `AgentTool(agent=...)` | OpenAI Agents, Google ADK |
| comprehensions `[tool(f) for f in TOOLS]` (TOOLS may live in another module), `+`, `*starred`, local variables | all |
| late wiring: `agent.handoffs = [...]`, `agent.tools.append(t)` | OpenAI Agents and others |

Dynamic tool lists (built from config or a registry at runtime) are not resolvable statically and
stay absent; the wire capture is the evidence for those.

**Capture importers** (`capture/importers.py`) accept foreign recordings that kept raw request
bodies (the boundary-capture spike's JSON, or a plain list of `{url, request, response}`), re-parsed
with the same wire parser the mock run uses. `--captures` detects the format.

Four extension points:

- **Extractor**: anything that fills an `ObservedAgent` (`ast/`, `capture/`, `mcp/`, `traces/`).
- **Rule**: `@rule("conformance", "AS-2xx")` on a function `(CheckContext) -> Iterable[Finding]`,
  with metadata in `rules/registry.py`.
- **Recognizer**: an optional per-framework plugin for graph reachability. It is a versioned
  contract (`RECOGNIZER_API_VERSION = 1`) registered under the entry-point group
  `agenticreq.recognizers`.
- **Detector**: evidence that a prohibited *action* happened, from `prohibited[].detector:
  {type, params}`. Ships `tool_arg_regex`; register your own with `agenticreq.detectors.register()`.
  Detectors read only real evidence (traces, non-synthetic captures), so they can find a
  violation but never prove safety.

A new framework needs a new recognizer (optional), an entry in `data/known_injected_tools.yml`,
and, if it bounds its loop by default, one in `data/framework_defaults.yml`. Its tool idioms usually
need no change (see the table above); a new wrapper name goes in `TOOL_WRAPPERS` /
`TOOL_DECORATORS`. No rules are rewritten.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy              # --strict, clean
pytest            # one positive + one negative per rule, onboarding contract (test_onboarding.py), phase acceptance tests,
                  # spike acceptance (tests/test_spike_acceptance.py), tool-idiom matrix (test_ast_generalization.py),
                  # layering (test_layering.py)
```

Integration tests skip when a framework isn't installed. CrewAI conflicts with `openai>=3`, so
its capture is checked in as `tests/fixtures/captures/crewai_po.json` (to re-record, use a
separate venv with `mock-run --python`). The OPA tests use `opa` on `PATH` or in `.tools/opa`.
The Cedar tests use `cedarpy`. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md),
[CHANGELOG.md](CHANGELOG.md) and [docs/architecture.md](docs/architecture.md).

## License

Apache-2.0 (includes a patent grant).
