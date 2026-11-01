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

## 0.1.0.dev0 (development history)

### No numbers from the library, three sign-off fields, which Python mock-run uses
- **agenticreq never suggests numbers.** The per-risk numbers (`bounds.*`, `max_delegation_depth`, `progress_guard`,
  `output_size_limits`, `confidence_threshold`) are gone from `defaults/`; `init` writes none (it lists the names and
  `approval_required` gets `timeout: TODO`); `params` and examples show `<number>`; results say "Found in code: N" or
  "No <name> set: write a value, or not_needed with a reason." `fix` writes a step cap only from your `bounds.max_steps`.
  `docs/defaults-rationale.md` became `docs/numbers.md` (no numbers).
- **Sign-off, not "NOT CHECKED (library gap)":** `role_boundary.capabilities`, `role_boundary.data_access`,
  `untrusted_inputs[].kind` (and `output_size_limits.max_items`, `write_safety.compensating_action`,
  `agent_registry[].role`) are plain questions under Human sign-off.
- **`mock-run` finds the agent's Python** (`--python`, `python:`, `$VIRTUAL_ENV` / `$CONDA_PREFIX`, any folder with
  `pyvenv.cfg`, Poetry, Pipenv, Conda, pyenv / asdf) and prints `python: <path> (from <source>)`. Before the run it
  checks the interpreter starts and can import the agent's framework, and stops with the missing packages. The run no
  longer needs agenticreq or pyyaml in the agent's environment.

### Several mock runs in one mocks.yml
- **`runs:`** in mocks.yml: one named mock run per task (`name`, `entry`, `args`, `order`, `tools`, `structured`, ...);
  the top level is the default. A file without `runs:` is one run, as before.
- **`mock-run --run <name>`** runs some of them (default: all); **`--keep`** merges into the saved
  `.agenticreq/captures.json` (a run run again replaces only itself). `check` already counted a tool as offered when
  any saved run offered it, so every specialist of a router is covered by one run each.
- **`call: false`** no longer hides a tool no run reached: it is listed under "never offered to the model" with its
  reason and the `call: false` fact. The reason now names the agent and says to add a run under `runs:`.

### not_needed, names, runtime agents, grouping after a mock run, sub-fields, output polish

- **`not_needed`** on any field at any level and any type (or a key inside one) answers it: NOT NEEDED under Human
  sign-off; the reason comes from `confirmed:` (without one: "(no reason given)", still left). Blocked for agent_id,
  risk_level, goal, owner, and safety items at high/critical risk. Every shape error ends with how to fix it.
- **Names are the user's requirement**: `init` writes TODO for role_boundary and agent_registry and lists what the code
  has in `_derived` (declared_agents, declared_tools, delegation_in_code); `--from-code` copies them, with a warning.
  `check` reports both directions: "spec lists X; the code has no such tool/agent" (AS-217, AS-233) and "the code has
  X; it is not in your spec" (AS-206, AS-308).
- **One name normalization** (`agenticreq/names.py`) everywhere a name is compared; "Web Search" and web_search are
  one tool, never both offered and never offered. Two code names that normalize alike are a gap (AS-234).
- **Runtime agents**: a call the framework makes itself goes to `_derived.injected_by_framework` (never a gap); an
  unnamed project agent is a sign-off (AS-910, clause runtime_agents); a generated id is never asked to be registered.
  Tool-list results never pass while a tool built at runtime is unknown. The count is printed at the top.
- **After a mock run** newer than the code, "Check with a mock run" keeps only what editing mocks.yml settles; a spec
  number with no limit in code becomes a gap, what agenticreq cannot read a sign-off, a mocked approval gate a
  sign-off. A `confirmed:` entry settles any sign-off line by its parameter name.
- **Every sub-field** is checked by a rule or shown as "NOT CHECKED (library gap): <field>" (AS-911,
  `agenticreq/subfields.py`). A declared approval timeout with an approval call that waits forever is a gap.
- **Output**: nothing is cut with "…" (long text wraps to the text column); waivers accept a parameter name; a waiver
  or `confirmed:` entry that matches nothing is an error (AS-119); audit's first group is "GAPS FOUND IN CODE"; every
  command defaults to "."; `check --coverage` lists every parameter; a clear wrong-environment mock-run message.
- **Cross-framework matrix** (`tests/fixtures/matrix`, `tests/test_framework_matrix.py`): CrewAI, LangGraph, OpenAI
  Agents, PydanticAI, smolagents, AutoGen, LlamaIndex, Google ADK, each through the whole 5-step path.

### Limits in any form, tools read by their body, allow_write, clear mock-run failures
- **Limits** are found in the forms code writes them: slicing, `islice`, a counter that stops the loop, `min()` /
  `max()`, a size check that refuses, SQL `LIMIT` (also f-strings and query parameters), `.limit()` / `.head()` /
  `.nlargest()`, capping keywords (`limit=`, `max_results=`, `top_k=` ...), `Field(le=..., max_length=...)`, and
  project helpers (`truncate(text, 6000)`), through constants, class attributes and imports. The report says the
  cap and where: `list_files capped at 200 (counter, tools.py:193)`. A cap read from env or config goes to "Check
  with a mock run: limit set from <ENV>".
- **Tools are READ, WRITE or UNCLEAR by their body**, following project calls: file / HTTP / SQL / ORM / mail /
  cloud writes, programs started; reads; a pure function is READ ("no side effects found"). The name is only a
  weak hint when the body cannot be read. UNCLEAR names the call it cannot follow and goes to a mock run, never a
  gap. Every write rule (approval, write safety, idempotency, dry run, trust boundary) uses it. New `tool_effects:`
  in spec.yml; a body that writes against a declared read is a gap (new rule AS-232) with the line.
- **allow_write:** paths are normalised (`/` and `\`, `..`, symlinks, case on Windows); writes inside are listed
  ("wrote workspace/final_answer.md, 2 KB") and are not a change of the project's files; writes outside are
  blocked and named. Without it only `.agenticreq/` and the temp folder are writable. `agenticreq mocks` writes
  `allow_write:` and pre-fills folders the code writes to.
- **mock-run never fails silently**: entry not found, a missing dependency, a crash before the model call, the SDK
  rejecting the request, the mock model unable to answer, an HTTP client that cannot be captured, a timeout (with
  the step reached), no model call, a retry hiding the first error, and a partial run each get one sentence with
  file:line and the next step, and exit 1. Docs: docs/tools-and-mock-runs.md.

### Five steps, progress after every command, smarter field names
- **mock-run no longer checks where it runs**: the environment check, its warning and `--require-...` option are
  gone. Every protection stays as it was: real credentials withheld (and listed: "credentials withheld from the
  agent: [...]"), the network block, the process guard, and the check that the run did not change the project's
  files. README "Safety": mock-run runs your agent with mock tools; run it on projects you trust, and use a separate
  machine for a project from someone else.
- **Your progress**: `audit`, `init`, `mocks`, `mock-run` and `check` end with a five-step block (1 audit, 2 spec,
  3 check, 4 mock-run, 5 sign-off) and one `Next:` sentence ending in the exact command; for step 5 the entry to
  paste under `confirmed:`. State comes from the folder only (spec.yml, mocks.yml, `.agenticreq/captures.json`,
  `.agenticreq/state.json`). `--quiet` hides it; JSON and SARIF never show it. `+ x` replace `✓ ✗` where the
  terminal cannot print them.
- **check shows what matters at your step**: with TODO fields left, the gaps and the TODO list only (sign-offs are
  one line: "Human sign-off: N items, shown after the spec is complete"); mock-run items after the gaps are fixed.
  `check --all` shows everything.
- **Sign-off answers**: each entry under `confirmed:` has `answer: yes | no | not_needed` (missing = yes). `yes`
  needs by, date, expires, evidence (SIGNED); `no` needs by, date, reason and is a gap ("Owner says this is not in
  place yet: <reason>", exit 1); `not_needed` needs by, date, reason and shows as NOT NEEDED (done). An entry missing
  a field stays open ("evidence is missing"); expired ("sign-off expired") or evidence changed ("file changed since
  sign-off") opens it again. At high / critical risk the safety items cannot be `not_needed` (spec error, exit 2).
  JSON `counts` gains `not_needed`; new rules AS-503 (owner says no) and AS-504 (owner says not needed).
- **`agenticreq guide`** prints where the folder stands, the exact next command and a table of the five steps (and
  which of spec.yml / mocks.yml is missing). `agenticreq` with no command prints a short overview. Every command's
  `--help` starts with the step it belongs to; `--help` groups them under "The 5 steps" and "More".
- **A wrong field name in spec.yml stops check (exit 2)** with the closest field: "Unknown field `max_iterations` in
  spec.yml (line 12). Did you mean `bounds.max_steps`? Fix it, or run: agenticreq validate --fix". Names are
  normalised (case, camelCase, kebab-case, a plural, filler words) and matched against every field, nested ones
  included, then against a few "also called" words stored in each params catalog entry (new `aka`, shown by
  `agenticreq params`). Equally close fields are all listed; no close field means no guess. `validate --fix`
  renames a key only with one clear match in the same mapping, shows the diff and asks (`--yes` to skip); comments
  stay. New `agenticreq export-schema` prints a JSON Schema for spec.yml; `init` writes it as `spec.schema.json` and
  makes the first line of spec.yml `# yaml-language-server: $schema=./spec.schema.json`.
- **Docs**: docs/defaults-rationale.md maps the framework names read from code (`recursion_limit`, `max_turns`,
  `max_iter`, `request_limit`, ...) to the library parameters. `validation/e2e_steps.py` walks a LangGraph team
  through the five steps with the built wheel.

### Four result groups, mock runs, plain words
- **Every result is in exactly one of four groups**, in this order: **Spec vs code gaps** (`✗` GAP: the spec says X
  and the code does not do it, or a required spec field is missing or TODO; before: BLOCKER, WARNING, `missing:`,
  AS-101, AS-102), **Human sign-off** (SIGN-OFF / SIGNED: things code cannot show, confirmed by a person; before:
  ATTESTATION, "needs your confirmation"), **Check with a mock run** (`?` CHECK: a code-pattern result, a tool that
  never reached the model, a number agenticreq cannot read from code; before: UNKNOWN, insufficient evidence,
  heuristic, `not proven:`) and **Passed** (`✓` PASS). The rules detect exactly what they did before; only the grouping
  changed. A signed sign-off stays in its group as SIGNED and is never counted as passed. It opens again when its
  `expires` date passes, or when a file named in its `evidence` (`file:line`) changed after its `date`. When the code
  shows something about a sign-off item, the line says so ("Found: ...; confirm it works.").
- **`--fail-on` has new values** (the old ones are removed): `gaps` (default: exit 1 on a spec vs code gap), `all`
  (also on an open sign-off or a check waiting for a mock run), `never` (always 0). `2` is still an invalid spec and
  `3` nothing checked. The old `any` and `missing` are a usage error. The Result line is PASS only when the selected
  condition is met.
- **JSON / SARIF:** every line, finding and passed check has `group` (`gap` | `sign_off` | `mock_run` | `passed`),
  `status` (`GAP` | `SIGN-OFF` | `SIGNED` | `CHECK` | `PASS`) and `reason`; the top level has `result`, `fail_on` and
  `counts` `{gaps, sign_off_left, signed, mock_run, passed}`. `kind`, `buckets` and `bucket_counts` are gone (findings
  are one `findings` list); the internal confidence is given in plain words as `how_sure`. SARIF: a gap is `error`, an
  open sign-off or a mock-run check is `note`, a signed sign-off is `none`.
- **`discover` is now `mock-run`**, with no alias (nothing was released). `agenticreq mock-run` reads `entry`, `args`
  and `python` from `mocks.yml`; `--entry`, `--args`, `--python`, `--runs`, `--timeout` and `--mocks` override them.
  `python:` empty uses `./.venv` (`.venv\Scripts\python.exe` on Windows, `.venv/bin/python` elsewhere) when it exists,
  else the interpreter running agenticreq, and says which. It prints the calls captured, the tools offered, the tools
  that never reached the model and where the run was saved, and exits 1 when the run crashed or captured no call.
  `init --entry` runs a mock run with `--mocks` (default `mocks.yml`).
- **New `agenticreq mocks`** writes `mocks.yml` from the code: the entry that runs the agent (or `TODO` with a comment),
  its `args`, `order` with every tool, and one entry per tool (sample `args` for read-only tools, `patch:` + `returns:`
  for tools that write, pay, send or delete, `call: false` for tools under `prohibited:` in spec.yml). It never
  overwrites `mocks.yml` without `--force`. A `patch:` on a tool object (LangChain `@tool`, CrewAI) now stubs the
  function inside it, so tool lists built at import time get the stub too.
- **The sandbox concept is removed**: no `--sandbox`, no `sandbox.yml`, no `SandboxRefused`. A mock run needs
  `mocks.yml` (without it: "mock-run needs a mocks file: run `agenticreq mocks` to create mocks.yml", exit 2).
  The protections stay as they were: real credentials are withheld, live-looking values under `env:` are refused,
  network connects / other programs / writes outside a scratch area are blocked at the Python level, and a run outside
  a container prints a warning. `allow_write:` moved to `mocks.yml`; the network allowlist is gone (a mock run has no
  network). The guard module is `capture/guard.py`.
- **`check` reads the latest mock run** (`.agenticreq/captures.json`) automatically; `--captures` still reads another
  file. The header shows when it ran and how many calls it captured, or "no mock run yet", and says "Code changed
  after the last mock run; run `agenticreq mock-run` again." when a scanned file is newer. The file path shows only
  with `--verbose`. `audit` and `check` default to the current folder, so from the agent's folder no argument is
  needed.
- **`attestations:` is renamed `confirmed:`.** A spec that still uses `attestations:` stops with "`attestations:` was
  renamed to `confirmed:`; rename the key in spec.yml" (exit 2).
- **New terminal layout**: a header (command · agent · spec · mock run), the Result line second, then the four groups
  (empty ones hidden), one line per parameter cut at the terminal width with `…` and never wrapped, `(+N more)` for
  repeats, `(N agents)` when several agents share a result and `[agent]` when one of several has it. Every "Check with
  a mock run" line ends with the next command. Red / yellow / cyan / green; plain with `NO_COLOR` or a non-TTY;
  `+ x - ...` instead of `✓ ✗ · …` when the terminal cannot print them. `--verbose` adds rule ids, file:line, the
  tools of each agent and every signed item. The old bucket view (BLOCKERS / WARNINGS / UNKNOWNS / ATTESTATIONS) is
  gone.
- **Plain words everywhere the user looks**: terminal, JSON values, SARIF messages, `--help`, README and docs no longer
  say attestation, unknown, not proven, heuristic, insufficient evidence, sandbox or discover. `params` lists
  `sign-off` where it said `attestation`; `audit_scope.level1_unknown` is `needs_spec`; `docs/heuristic-rules.md` is
  `docs/code-pattern-rules.md`. Internal names (`Bucket`, `ConfidenceLevel`, `capture/discover.py`) stay and are
  never printed.

### Numeric thresholds are recommendations
- Numeric thresholds are now recommendations. A number fails only when it is missing or when code and spec.yml disagree.
- The risk_level numbers (`bounds.*`, `max_delegation_depth`, `progress_guard`, `output_size_limits`, `confidence_threshold`) are shown as `recommended for <risk>: <value>` next to a result and written into the `agenticreq init` scaffold; any positive value is valid at any risk level (`high` and `critical` too). The numeric floors of AS-111 are gone; the yes/no floors (`kill_switch`, `fail_closed`, ...) stay. A value of 0 or below, a wrong type, or a `confidence_threshold` outside 0..1 is a spec shape error.
- One decision table for every number (`agenticreq/thresholds.py`, `docs/defaults-rationale.md`), per agent. New: `bounds.max_retries`, `bounds.max_tool_calls`, `aggregate_budget.*` and `approval_required[].timeout` are compared with the code (or confirmed under `attestations:`) when spec.yml states them. Each number is its own report line (`bounds.max_steps`, `bounds.max_cost_usd`, ...).
- A step bound passes when code sets one, at least as strict as the spec's `max_steps` when it has one; `AS-229` / `AS-230` / `AS-228` measure runs against the number in spec.yml, not against a risk_level value.
- A step bound is attributed to the agent whose constructor carries it; a bound on a run or graph call applies to every agent.

### Per-agent tools, runtime proof, any framework version
- **Each agent gets the tools the code gives it** for the common per-agent table shapes: if / elif chains and
  `match` on the agent key, tools kept on `self` in `__init__`, tool names in agents.yaml looked up in a registry
  (`REGISTRY[name]()`, `self.tools[name]`). Checked on CrewAI, LangGraph, AutoGen and OpenAI Agents. Before, these
  gave every agent no tools, or the wrong ones (an if-chain gave each agent the fallback branch's tools). A branch
  the scan cannot decide keeps both sides' tools, marked conditional. The console prints `tools:` under each agent.
- **discover exits 1 when the run proves nothing**: a crash, an import error, no LLM call captured, missed calls
  or changed files, each with its reason. An import error names the agent's environment and how to point
  `--python` at a clean one. `audit` / `check --captures` warn when the captures file holds no call.
- **`prohibited` and `approval_required` can PASS on runtime evidence.** A tool never offered to the model in any
  captured run: `held in N of N runs`. discover records approval gates (`input()`, LangGraph `interrupt()`,
  OpenAI Agents `needs_approval` when it requires approval) after the model asks for a tool; an approval before
  the tool's result in every run is a PASS, a result with no approval is `FAIL not proven: ran without a recorded
  approval`. `mocks.yml` `human_input:` sets what an `input()` prompt is answered (default `yes`); an `input()`
  prompt no longer blocks or crashes the run.
- **The framework version is the agent's**, not agenticreq's: a project venv / conda env (macOS, Linux and
  Windows layouts), then `uv.lock` / `poetry.lock` / `pdm.lock` / `Pipfile.lock`, then an exact pin. agenticreq's
  own Python counts only when it matches the project. A range (`crewai>=0.100.0`) is no longer read as version
  0.100.0: AS-908 fires only when the range allows no validated version. New `agenticreq.versions` (PEP 440
  specifiers, no new dependency).
- **discover runs the agent in ITS environment only.** Installed in a busy site-packages (conda base), agenticreq
  used to put that whole folder on the agent's path, so old packages there broke the run (`cannot import name
  'Sequence' from 'collections'`); now only the agenticreq package is added. A crash shows the exception line,
  not a note a framework appended after it. `mocks.yml` `env:` passes non-secret settings (a model name) to the
  agent; live-looking credentials are still refused.
- **discover runs any framework version**: outside the validated range it runs with a note; only `known_bad`
  versions are refused. Validated again on crewai 1.9.3, langgraph 1.2.17, openai-agents 0.24.0, smolagents 1.26.0,
  pydantic-ai 2.61.0 and autogen-agentchat 0.7.5 (new fixture `agents/autogen_po`; `validation/val_frameworks.py`).

### PASS / FAIL report (output and exit code only; no rule detects anything differently)
- Every result is **PASS** or **FAIL**, one line per parameter (named from the params catalog), grouped per
  agent with FAIL lines first, then `Result: PASS|FAIL (N missing, N not proven, N passed)`.
- PASS needs evidence: `proven` ("found in code"), `observed` ("seen when the agent ran"), or an attestation
  confirmed in spec.yml ("confirmed by you"). A `heuristic` result (a code pattern only) is always
  `FAIL not proven:`, never a PASS.
- Every FAIL reason starts with `missing:` (BLOCKER / WARNING: the code has to change) or `not proven:`
  (UNKNOWN, insufficient evidence, unconfirmed attestation, run-level evidence notes AS-9xx, heuristic passes:
  evidence has to be supplied).
- The internal confidence levels and the BLOCKERS / WARNINGS / UNKNOWNS / ATTESTATIONS / PASSED buckets are
  unchanged; the bucket view, rule ids and file:line are shown with `--verbose`. `--show-passed` is kept but
  hidden: passed lines always show.
- JSON: each finding and passed check gets `status`, `kind` (`missing` | `not_proven` | null) and `reason`;
  top level gets `result`, `counts` (`passed`, `missing`, `not_proven`) and `results` (the per-agent lines).
  The old bucket counts moved to `bucket_counts`.
- SARIF: `missing` -> level `error`, `not proven` -> `warning`, a confirmed attestation -> `note`; results carry
  `status` / `kind` / `reason`. GitHub shows `warning` as non-failing; the exit code fails the job.
- More PASS / FAIL mappings: a spec field still TODO is `FAIL not proven: spec field <name> is still TODO`
  (one line per field); an observed rule passes only when it held in every run (`held in N of N runs`),
  otherwise `FAIL not proven: held in only n of N runs`; an unexpired waiver is `PASS waived by you`.
  `audit` names no line after a Level-1 spec field, and keeps its "audit without a spec" header.
  "Nothing was checked" prints `NOTHING CHECKED` and no table (exit 3).
- Rules that reach only `heuristic` from code, and which of them have any route to PASS:
  [docs/code-pattern-rules.md](docs/code-pattern-rules.md).
- `validate` keeps its exit rule (1 only for a proven spec blocker), so a fresh `init` scaffold still validates;
  `check` fails on every TODO.
- `mypy --strict` is clean again: type-only fixes in `ast/` and `rules/taint.py` (no detection change; findings
  are identical on all fixture agents).
- **Exit code: from 0.1.0, any FAIL exits 1 by default (`--fail-on any`).** `--fail-on missing` fails only on
  `missing` results; `--fail-on never` reports only. `2` (invalid spec) and `3` (nothing checked) are unchanged.
  The old values `blocker` / `warning` / `unknown` are gone (0.1.0 is the first release, so no user depends on
  them). If this default is ever changed after a release, `missing` becomes the default and `any` returns only
  in a major version (1.0).

### Second audit round (report quality, env values, audit trail)
- A bound or model the code reads from the environment (`int(os.getenv("MAX", "10"))`, `os.environ[...]` with a
  `.env` / `.env.example` entry) is read and used as an **assumed** value, always labelled "assumed: default in
  code of env MAX" and never `proven`. No default and no `.env` entry stays UNKNOWN. Pinned default model:
  AS-210 passes as an assumption; alias default: a heuristic finding. `.env` secrets are never copied into reports.
- A bound keeps the framework's own name (`max_messages`, `max_round`, `max_agent_steps`) in messages; the same
  bound reached through two spellings on one line counts once, and repeated bounds print as `max_iter=5 x5`.
- AS-225 recognises an audit trail by what the code does: a project function that saves a transcript / the
  messages and writes or logs (through helpers and other files), or `json.dump` / `logger.info` handed the
  messages, is reported as UNKNOWN ("stored; per-call audit not verified") instead of "no audit". Logging
  inside a helper a tool calls counts for that tool. AutoGen `TimeoutTermination` is a run-time limit.
- Console report: starts with a plain coverage line ("6 of 7 checks could run; 1 could not ...") and a
  "Fix first" list (at most 3); the "How to read" legend moved to the end.

### Framework detection
- **Microsoft Agent Framework** (`agent_framework`) is recognised (23 frameworks); `autogen_core` / `autogen_ext` /
  `ag2`, `crewai_tools`, `langgraph_supervisor` / `langgraph_swarm` and `haystack_integrations` now name their
  framework, and so does `from google import adk`.
- Literal dynamic imports (`importlib.import_module("crewai")`, `__import__(...)`) count as imports.
- A project's own `agents/`, `swarm/`, `camel/`, `strands/` or `needle/` package is no longer taken for the
  framework of that name; a bare `agents` import is the OpenAI SDK only when it uses that SDK's names or submodules.
- A framework that is only declared (requirements*.txt, pyproject incl. Poetry, setup.py/cfg, Pipfile, conda
  files) is reported with `framework_source: dependency` ("declared, not imported"): it never outranks an imported
  framework and its built-in defaults (step bound, tracing) are not assumed. Lock files are ignored.
- `discover` records the frameworks the run loaded from site-packages (`module:<name>` in `versions`); when the
  static scan found nothing, the run names the framework (`framework_source: runtime`).
- Each framework's loop bound maps to the same step bound: AutoGen `MaxMessageTermination(n)` / `max_round`,
  Haystack `max_agent_steps`, DSPy `max_iters`, Semantic Kernel `maximum_iterations`, Agno `tool_call_limit`.
- A tool's body facts now include what the project helpers it calls do (`save_report -> _write -> open(..., "w")`
  is a file write), through functions, other modules and `self.` methods, to depth 4.

### Fixed (external audit, F1–F8)
- **F1** A prohibited tool that reaches the agent through an MCP server, a toolset or a registry
  (OpenAI Agents `mcp_servers=`, PydanticAI `toolsets=`, LangChain MCP client, Google ADK `MCPToolset`, CrewAI
  `mcps=`, smolagents `ToolCollection.from_mcp`, AutoGen `workbench=`, Agno / Strands / Haystack /
  Semantic Kernel MCP helpers, hosted `{"type": "mcp"}` tools) is now **UNKNOWN** (blocker severity), never a
  false PASS. An `--mcp` tool list or a capture resolves it.
- **F2** Every agent in the scanned code is checked, not only the first one. A prohibited tool on a second agent or
  on `agent.clone(...)` is a BLOCKER.
- **F3** New `--fail-on unknown` (CLI and the GitHub Action `fail-on` input): exit 1 also when a blocker-severity
  rule ended UNKNOWN. Exit codes 2 and 3 are unchanged. A home-made `interrupt()` is no longer taken for
  LangGraph's approval gate, and an action that runs *before* the gate is `partial`, not `all_paths`.
- **F4** A model chosen at run time (`os.environ[...]`, `os.getenv`, a config attribute, a helper call) is reported
  as AS-210 UNKNOWN and keeps its clause in the coverage denominator.
- **F5** Level-4 `enforceability` is derived from the rule results (PASS per clause); UNKNOWN / BLOCKER clauses
  and looser-than-declared schema constraints no longer count as enforced; "could be exported to OPA/Cedar" is
  reported separately from "enforced in code".
- **F6** Factory functions are read once per function node (memoised, recursion-safe): smolagents' source went
  from >25 minutes (never finished) to about 7 seconds with identical findings.
- **F7** UTF-8 BOM files (`.py`, JSON, YAML, JSONL) are read correctly instead of being dropped as parse errors.
- **F8** `discover` no longer claims to be a sandbox. Docs and `--help` say: run it only inside a container with
  no network and no secrets. New `--require-container` refuses to start anywhere else; without a container a
  warning is printed.
- `stats --spec` / `fix --spec` print a clean error for a missing file instead of a traceback.

### Added
- Frameworks recognised: agno, camel, claude-agent-sdk, dspy, haystack, semantic-kernel, strands, swarm
  (22 in total).
- `ProcessGuard` for `discover`: blocks starting programs and file writes outside a scratch area (Python-level,
  best-effort).
- AS-908: a framework version outside the validated range is reported as UNKNOWN.
- "Nothing was checked" is reported as such (exit code 3) instead of looking like a pass.
- Readable console report with a legend, per-finding "What we saw / What to do".

### Changed
- **Renamed** the package, CLI, imports, env vars and cache folder from `agentconf` to `agenticreq` (`agentconf` is
  taken on PyPI). Re-install with `pip install -e .`; `AGENTCONF_*` variables are now `AGENTICREQ_*` and
  `.agentconf/` is `.agenticreq/`.
- Added `.github/workflows/ci.yml`: Ubuntu / macOS / Windows x Python 3.10-3.13, `mypy --strict`, tests, and a
  same-output-for-different-hash-seeds check.
- `check` uses only the limits the spec declares (AS-221 / AS-222); defaults apply to `audit` only.
- `reliability_budget` removed from the derived values (it was not a measurement).
- Large internal modules split (`ast/extractor.py`: 2317 → ~360 lines); `mypy --strict` clean.
- Packaging: the sdist no longer ships scratch files; `cedarpy` is in the `dev` extra.
