# Validation on real repositories — run 1 (2026-10-15)

This is the first measurement of agenticreq on code we did not write. Nothing was tuned before,
during, or after this run. Where a rule produced noise, the noise is recorded below as a finding
about the rule. Decisions on changes come later.

**Setup:** macOS x86_64, Python 3.11/3.12, agenticreq `0.1.0.dev0` (147 tests passing at the time
of the run). Repos were shallow-cloned at the commits listed. `audit` ran with **no spec file**.
Raw results, drivers and the sandbox declaration are in [`validation/`](validation/).

## Targets and how each was run

| # | repo @ commit | target | run mode | why |
|---|---|---|---|---|
| 1 | langchain-ai/react-agent @613fdaa | whole repo | **discover** + AST | tools: Tavily search only (network → blocked) |
| 2 | langchain-ai/memory-agent @adc8f52 | whole repo | **discover** + AST | tool writes to a LangGraph store; driver passes an `InMemoryStore` |
| 3 | langchain-ai/data-enrichment @5ddc778 | whole repo | **discover** + AST | tools: search + scrape (network → blocked) |
| 4 | langchain-ai/open_deep_research @1b7d2e8 (archived) | whole repo | AST only | **install failed** (below) |
| 5 | assafelovic/gpt-researcher @3582a68 | `multi_agents/` | AST only | **unsafe to run**: orchestrator writes report files (`os.makedirs`, file writes) |
| 6 | openai/openai-cs-agents-demo @bd7bfca | `python-backend/` | **discover** + AST | tools mutate in-memory demo state only |
| 7 | openai/openai-agents-python @24ebf1c | `examples/financial_research_agent/` | **discover** + AST | hosted web search (network → blocked), agents-as-tools |
| 8 | crewAIInc/crewAI-examples @da94a91 (archived) | `crews/stock_analysis/` | AST only | **import failed** (below) |
| 9 | pydantic/pydantic-ai @541ba02 | `examples/pydantic_ai_examples/bank_support.py` | **discover** + AST | SQLite `:memory:` only |
| 10 | huggingface/smolagents @96f33fa | `examples/open_deep_research/` | AST only | **unsafe to run**: 22 local write sites, a `download_file` tool, a CodeAgent that executes generated code |

discover passed the gate with a **declared sandbox** (`validation/drivers/sandbox.yml`), not with
mocks. The tools ran for real with every network connection blocked and only fake credentials. Each
repo was read by hand for local writes, subprocesses and databases before it was allowed to run.
Drivers live outside the repos and only import and invoke the repo's existing graph or agent.

**Install / import failures (recorded, not forced):**
- open_deep_research: the declared dependencies fail with `pyarrow 23.0.1 … error: command 'cmake'
  failed with exit code 1` (pulled in by `langchain-google-vertexai 3.2.4`). One retry from the
  repo's own `uv.lock` (exported read-only) failed identically. No pyarrow wheel exists for macOS x86_64.
- stock_analysis: the repo's declared dependencies (`crewai[tools]>=0.152.0`, which resolved to crewai 1.6.1) install,
  but the code does not import: `ImportError: cannot import name 'BaseTool' from 'crewai_tools'`.
  The code has drifted from its own dependency range.

**Zero-touch:** 7,521 files across the 10 repos were hashed before any step and again after all
steps: **0 changed**, and `git status` is clean in every repo.

## Results

Counts are findings by confidence (`unknown` includes `insufficient_evidence`) and findings by
bucket. There were no blockers anywhere.

| repo | framework + version | execution model | declared tools (AST) | observed tools (wire) | proven | heuristic | unknown | blockers | warnings |
|---|---|---|---|---|---|---|---|---|---|
| react-agent | langgraph 1.2.15 (runtime) | schema-based | **0** (truth: 1) | 1 | 3 | 0 | 0 | 0 | 3 |
| memory-agent | langgraph 1.2.15 (runtime) | schema-based | **0** (truth: 1) | 1 | 1 | 0 | 0 | 0 | 1 |
| data-enrichment | langgraph 1.2.15 (runtime) | schema-based | **0** (truth: 3) | 3 | 3 | 0 | 0 | 0 | 3 |
| open_deep_research | langgraph 1.2.9 (repo `uv.lock`)¹ | schema-based | 4 (2 current + 2 legacy; truth: 4 current) | — not run | 1 | 1 | 1 | 0 | 2 |
| gpt-researcher/multi_agents | langgraph (unpinned in requirements.txt)¹ | schema-based² | 0 (truth: 0) | — not run | 0 | 0 | 1 | 0 | 0 |
| openai-cs-agents-demo | openai-agents 0.23.1 (runtime) | schema-based | 11 (truth: 11) | 6 (1 user tool + 5 handoffs)³ | 10 | 9 | 10 | 0 | 19 |
| financial_research_agent | openai-agents 0.23.1 (runtime) | schema-based | **0** (truth: 2 agent-tools + 1 hosted) | 0³ | 4 | 0 | 0 | 0 | 4 |
| stock_analysis | crewai 1.6.1 (runtime, import failed) | schema-based | 1 (truth: 3 own + library tools) | — import failed | 1 | 1 | 1 | 0 | 2 |
| bank_support | pydantic-ai 2.56.0 (runtime) | schema-based | 1 (truth: 1) | 2 (`customer_balance` + injected `final_result`) | 2 | 1 | 0 | 0 | 3 |
| smolagents/open_deep_research | smolagents 1.27.0.dev0 (repo)¹ | schema-based + CodeAgent → AS-901 | 10 (truth: 10) | — not run | 0 | 0 | 2 | 0 | 0 |

¹ For repos that never ran, agenticreq reported the version installed in **its own** environment
(`langgraph 1.2.15 (installed)`, `smolagents 1.26.0 (installed)`), not the repo's. The column shows
the repo's own pin, read by hand. See finding V8.
² gpt-researcher/multi_agents makes plain LLM calls with no tool binding, so "schema-based" here
only reflects the framework import.
³ Runs cut short by the canned responder (finding H1).

"Truth" is a hand-built inventory from reading each repo's code, listed in `validation/results/` and below.

## The four numbers

**1. Extraction coverage.** 9 of the 10 targets have tools (gpt-researcher has none, and agenticreq
correctly reports 0).

| measure | result |
|---|---|
| repos with tools where AST produced any inventory | **5/9 (56%)** |
| repos with tools where AST produced the complete inventory | **3/9 (33%)** (cs-agents, bank_support, smolagents) |
| repos with tools where AST + wire together produced any inventory | **8/9 (89%)**. financial_research produced nothing. |
| tool-level recall of AST (tools found / tools that exist) | **25/36 (69%)** |

**2. UNKNOWN rate: 15/68 = 22%.** That is unknown findings divided by all evaluated check results
(findings + passes). 10 of the 15 come from one repo: cs-agents' run was cut short, leaving 10
declared tools never reached (AS-201). Excluding that run it is 5/34 (15%). Most of those 5 are
AS-903 "no runtime evidence" on the four AST-only repos.

**3. False-positive rate: 18 of 36 warnings (50%) are not real problems.** There were 0 blockers.
The one AS-901 sits in the UNKNOWNS bucket; it is correct and is not counted here.
Standard used: a finding is a true positive only if its claim is factually correct about the code
**and** the concern applies to the code as written.

| rule | fired | false positives | why they fired |
|---|---|---|---|
| AS-202 undeclared tool | 5 | **5** | react/memory/data-enrichment declare tools as **plain `async def` functions passed to `bind_tools()` / `ToolNode()`**; data-enrichment builds `Info` as a dict schema at runtime. AST does not count either as a declaration, so every wire tool looks undeclared. |
| AS-208 no step bound | 4 | **4** | Framework defaults bound the loop: OpenAI Agents `DEFAULT_MAX_TURNS=10`, PydanticAI `request_limit=50`, CrewAI `max_iter=25`. open_deep_research enforces its own counters (`max_researcher_iterations`, `max_react_tool_calls`). The rule only looks for explicit kwargs. (LangGraph's default `recursion_limit` is 10007 in 1.2.15, effectively unbounded, so the rule would be right on a LangGraph app with no bound of its own.) |
| AS-210 unpinned model | 19 | **4** | 3 model strings come from **test files** (`tests/unit_tests/test_configuration.py`, `tests/evaluators.py`). 1 is a **duplicate**: bank_support's `openai:gpt-5.2` is reported once from code and once from the wire (prefix stripped). The other 15 are correct, but they are noisy: 8 findings for 2 distinct model strings in cs-agents, one per call site. |
| AS-211 write without idempotency | 4 | **4** | All four cs-agents tools are idempotent in their bodies. `issue_compensation` explicitly reuses an existing case id, and the others set state fields. The rule judges by tool name only, so the claim "a retry can repeat the side effect" is false here. |
| AS-212 write without approval | 4 | 0 (judged) | Factually correct: `cancel_flight`, `issue_compensation`, `book_new_flight` and `update_seat` run with no confirmation step. In this demo they mutate in-memory state only. If that is judged out of scope, the false-positive rate becomes 22/36 (61%). |
| AS-601 tool description changed | 1 | **1** | `@function_tool(description_override=...)` is not read by the extractor; it compares the docstring with the override. |

**4. Mean observed ÷ declared: 0.68 over the 6 runs, measured against the hand-made inventory.**

| repo | observed / true tools |
|---|---|
| react-agent | 1/1 |
| memory-agent | 1/1 |
| data-enrichment | 3/3 |
| cs-agents | 1/11 |
| financial_research | 0/2 |
| bank_support | 1/1 |

Against agenticreq's own AST inventory the ratio is undefined for 4 of the 6 runs (0 declared).
This is **not comparable with the spike's 0.80.** The two low ratios come from runs the canned
responder ended early (H1), not from a limit of boundary capture. The 4 runs that completed
showed every tool (1.0). As the spike's own caveat says, the ratio depends on the harness.

## Findings about the library (run 1; status after the fix: see Run 2 below)

**Extraction**
- **V1** Plain functions passed to `bind_tools()` / `ToolNode()` are not counted as declared tools.
  This is the dominant idiom in LangChain's own LangGraph templates, and it causes all 5 AS-202
  false positives.
- **V2** Pydantic-model tools (`bind_tools([ConductResearch, ResearchComplete])`) and
  `tool(Model)` are not extracted.
- **V3** Agents-as-tools (`agent.as_tool(tool_name=...)`) are not extracted.
- **V4** Tool classes that subclass something other than `BaseTool`/`Tool`/`StructuredTool` (e.g.
  `RagTool`) are not extracted.
- **V5** `@function_tool(description_override=...)` is ignored (AS-601 false positive).
- **V6** Model ids in config `Field(default="openai:gpt-4.1")` under non-standard names
  (`research_model`) are not found. This is a **missed** AS-210 in open_deep_research.
- **V7** Test files are scanned like agent code (3 AS-210 false positives).
- **V8** Framework version: when there is no run, the analyzer's own installed version wins over
  the repo's pin, labelled `(installed)`. That is honest about the source but answers the wrong
  question.
- **V9** Handoffs assigned after construction (`triage_agent.handoffs = [...]`) are not extracted,
  so the code-declared delegation graph of cs-agents is empty. The runtime edges are still observed.

**Rules**
- **R1** AS-208 ignores framework default bounds and app-level counters (4/4 false positives).
- **R2** AS-211 decides by tool name only (4/4 false positives in the one repo where it fired).
- **R3** AS-210 reports one finding per call site and can double-report the same model from code
  and wire.

**Harness**
- **H1** The canned responder cannot satisfy **structured-output** requests (JSON-schema
  `response_format` / `output_type`). OpenAI Agents guardrail agents (cs-agents) and the planner
  (financial_research) received prose, raised `ModelBehaviorError: Invalid JSON`, and ended the run.
  As a result, 10 cs-agents tools show as never reached.
- **H2** discover could not run 4 of 10 repos: 2 were unsafe (local writes / code execution),
  1 failed to install, and 1 failed to import with its own declared dependencies.

## What this run does not show

- No spec files were used, so `check` (Tier 1+3, delegation rules, taint with declared untrusted
  inputs) is **unmeasured** on real code. 0 blockers is a property of no-spec `audit`, not evidence
  that blockers are rare.
- The true/false-positive calls were made by one reviewer (the same agent that wrote the
  library). They should be re-checked independently.
- n = 10 targets, 6 runs, 36 warnings: too small for stable rates.

---

# Run 2 (2026-10-15, after the extractor rewrite and the run-1 fixes)

Same 10 repos, same commits, same ground truth (`validation/results/ground_truth.json`), same
sandbox and drivers. Venvs were rebuilt with uv on Python 3.11, so framework versions moved
(openai-agents 0.24.0, langgraph 1.2.15, pydantic-ai-slim 2.56.0, openai 3.31.0). Raw results:
`validation/results/discover_runs_run2.json`, `audit_results_run2.json`. Judging used the run-1
standard: a finding is a true positive only if its claim is factually correct about the code **and**
the concern applies to the code as written. One reviewer judged every finding by reading the code,
and that reviewer also wrote the library. **Re-check independently.**

**What changed in the library before the numbers below:** the run-1 findings V1–V7, V9, R1, R3 and
H1 were fixed. Running run 2 also exposed new defects, and those were fixed too (each has a
regression test in `tests/test_validation_run2_fixes.py`):

| defect found in run 2 | effect | fix |
|---|---|---|
| `context: str` dropped as "injected" by name | AS-607 FP (memory-agent) | drop by annotation; by name only for non-plain types |
| `agent.as_tool()` had no `input` param | AS-607 ×2 FP (financial) | agent-as-tool gets the framework's string param |
| `search(...)` capped inside the body (`max_results=`) | AS-213 ×2 FP | `bounded_output` body feature |
| OpenAI Agents `max_turns` / built-in tracing ignored | AS-306, AS-307 FP | `framework_defaults.yml`: run-level bound, built-in tracing |
| Gemini stable ids treated as aliases | AS-210 FP | pin pattern for `gemini-N.N-*` (not `-latest`) |
| one hub's 19 simple cycles reported one by one | AS-303 ×19 | one finding per strongly connected group |
| first pass followed `from smolagents import` into the repo's own `src/` | framework built-ins inventoried | framework packages never followed outside the scanned root |
| a helper-heavy `utils.py` became a "tool module" | 13 bogus tools | tool module needs ≥ half its functions to be tools |

## Run-2 numbers (with the run-1 numbers for comparison)

| measure | run 1 | run 2 |
|---|---|---|
| tool-level recall of AST | 25/36 (69%) | **35/36 (97%)**. Missing: data-enrichment `Info`, built at runtime (L7) |
| repos with tools where AST produced the complete inventory | 3/9 | **8/9** |
| repos where `discover` ran to completion | 4/6 (2 cut short by H1) | **6/6** (H1 fixed: schema-valid structured output) |
| LLM calls captured | 13/13 | **35/35** (cs-agents 3 → 19, financial 1 → 7) |
| mean observed ÷ declared (6 runs) | 0.68 | **0.98**. cs-agents 10/11: `baggage_tool` is defined but bound to no agent; AS-201 says so |
| UNKNOWN rate (unknown ÷ findings + passes) | 15/68 (22%) | **6/86 (7%)** |
| blockers (audit, no spec) | 0 | 3, **3 true** (AS-604 ×2, AS-302) |
| warnings: false-positive rate | 18/36 (50%) | 13/38 (34%), then **9/34 (26%)** after AS-211 needed body evidence (below) |

**AS-211 now needs evidence (R2 fixed).** It fires only when the spec explicitly sets
`write_safety.idempotency_key: true`, or when the tool body repeats a side effect on a retry: a fresh
id (`uuid4`, `random.choices`) that is not just the fallback of reused state (`state.id or new_id()`),
a record appended to something outside the function (`store.aput`, `self.rows.append`, `INSERT`), or
a remote write (HTTP POST/PUT/PATCH/DELETE). Name-only writes are not reported, and the clause counts
as not extracted. On the 10 repos this removed the 4 cs-agents false positives. The memory-agent
true positive stays (`memory_id or uuid4()`: the model omits `memory_id` on a retry).

**The 9 remaining false positives (not fixed):**

| rule | FPs | why |
|---|---|---|
| AS-213 unbounded output | 5 | name heuristic (`search*`, `find*`): smolagents browser tools return one viewport, CrewAI `RagTool`s return top-k chunks. The cap is not visible in the tool's own body |
| AS-210 unpinned model | 2 | strings in open_deep_research `src/legacy/`, code the project no longer runs |
| AS-208 no step bound | 1 | open_deep_research bounds itself with app-level counters (L5b) |
| AS-202 undeclared tool | 1 | data-enrichment's `Info` schema is built from state at runtime (L7) |

True positives worth naming: AS-216 `scrape_website` fetches a model-chosen URL; AS-211 on
memory-agent's `upsert_memory` (a retry without `memory_id` writes a duplicate under a fresh uuid);
cs-agents' wire schema facts (AS-603/605: optional `origin`/`destination` sent as required with the
default dropped; AS-604: `reason` / `seat_request` required **and** defaulted).

## `check` (spec vs code) on real repos

The repos ship no spec. Three **assumed** operator policies were written from each repo's own
prompts before running `check` (`validation/specs/`). They test the spec-vs-code path, not the
repos' real intent.

- The validator rejected the first cs-agents spec twice, correctly: `topology: hub-and-spoke` is not
  a valid value (AS-107), and `max_delegation_depth: 3` is weaker than the high-risk floor of 2
  (AS-111). Both were mistakes in the spec.
- **The first `check` on cs-agents exposed 3 library defects:** registry `match` was tested against
  prompts only, never agent names; an agent whose prompt mentions "Triage Agent" was mapped to
  triage; one agent with runtime-built instructions became several unnamed nodes; and
  `handoff(agent=x)` was not read. Together they produced 25 blockers, about 15 of them false
  (phantom `triage → triage` edges, AS-206 on other agents' tools, AS-301 on declared edges).
  The fixes: name-first registry resolution, tool-set node naming with merging per agent, the
  keyword handoff form, and graph roots for hubs. Each has a regression test.

After the fixes:

| repo | blockers | all true? | notes |
|---|---|---|---|
| cs-agents | 10 | **yes** | AS-205 ×2 (no gate on `issue_compensation`/`cancel_flight`), AS-604 ×2, AS-301 ×5 (handoffs in code the spec omits: seat→refunds, flight-info→booking, booking→seat, booking→refunds, refunds→faq, each confirmed in `airline/agents.py`), AS-302 (a 5-hop chain vs depth 2) |
| bank_support | 0 | — | AS-209: relies on PydanticAI's default `request_limit=50`, above the high-risk step bound. AS-210. Escalation, data rules and owner become attestations (not verifiable from code) |
| memory-agent | 0 | — | AS-207 (spec's `max_length: 2000` is not in the tool schema), AS-208, AS-211 |

Warnings on `check`: 35, of which **0** are false after the AS-211 change (it was 4/39 before). Noise to note:
**AS-305 fires 14 times on cs-agents.** Each is correct under the high-risk default
`authority_inheritance: subset`, but a hub where every specialist holds tools the hub lacks will
always violate "subset". The spec author should set the policy explicitly. Grouping per delegate
would make it readable. Not changed.

## Still not shown
- open_deep_research and crewAI stock_analysis did not install on macOS x86_64 (no `pyarrow` wheel),
  so they are AST-only. gpt-researcher and smolagents' example are AST-only by choice (unsafe to run).
- All runs use the canned responder: one deterministic path per agent. A real model would take others.
- n = 10 repos, 41 audit findings, 3 hand-written specs: too small for stable rates.

## browser-use/web-ui (repo supplied by the user, static only, 2026-10-15)

`/Users/mac/Documents/web-ui` (github.com/browser-use/web-ui @7eb62d4). It drives a real browser
and writes files, so it was **not run**: AST only. Judged sheet: `validation/results/webui/judge.csv`.

**First pass: 0 tools, 0 agents** (useless). It exposed 6 extractor gaps, all fixed with a regression
test: `@registry.action("description")` (browser-use), a factory that `return`s
`StructuredTool.from_function(coroutine=partial(f, ...))`, library tool classes (`WriteFileTool()`),
`class BrowserUseAgent(Agent)` subclasses, test files feeding bounds (AS-209 from `tests/`), and
browser-use's injected action parameters. A second pass then reported AS-401 paths between tools
of different agents (the research LLM vs the browser sub-agent), so tools bound together (one list
plus its `+=`, one `bind_tools` call, one action registry) are now one taint context, and project
factory functions are followed to the tool they return.

**Result: 6 tools, 5 warnings, 0 blockers. 4 sahi, 1 shor (20%).**
- sahi: `write_file` has no `root_dir` and no approval (AS-212); web search output reaches the same
  LLM that can write any path (AS-401); `read_file` → `write_file` (weak); `list_directory` unbounded (minor).
- shor: AS-212 on `upload_file`, which is restricted to an allowlist the user chooses.
- correct unknowns: `max_steps` comes from the UI at runtime (AS-209 ×3); no runtime evidence (AS-903).

**Not seen (false negatives):** browser-use's built-in actions (`go_to_url`, `click_element`,
`input_text`, ...) live in the framework's registry and are not inventoried. They are the browser
agent's real power, so a static run under-reports its risk. MCP tools load at runtime. Models are
chosen in the UI, so AS-210 has nothing to check. Only a `discover` run would show these.

## Cactus-Compute/needle (repo supplied by the user, static only, 2026-10-15)

`/Users/mac/Downloads/needle-main`: an on-device tool-calling model with its own `@needle.tool`
decorator and 6 example environments (smart home, kitchen, media, productivity, wearable, data
capture). The model runs **locally**, so there is no HTTP boundary: `discover` cannot see it, and only
static rules apply. Judged sheet: `validation/results/needle/judge.csv`.

**First pass: 30 tools found, 14 warnings, 12 of them noise (86%).** That pass exposed 4 rule defects:
- AS-212 asked for approval on every write, including a timer, a note and the volume (10 findings).
- AS-208 did not know Needle bounds its loop (`Needle.run(max_steps=8)`).
- AS-213 called `find_my_phone()` an unbounded search.
- Write detection missed actuator and messaging verbs, so `reply_to_notification` (which sends a
  message as the user) was not even a write.

**Fixes (each with a regression test):** actuator/messaging verbs count as writes (taint sinks);
zero-config `audit` asks for a gate only on high-impact writes: irreversible or external by name
(pay, delete, send, reply, book, cancel, ...) or by body (file write, exec, remote write). `check`
with a high/critical spec still gates every write. A `find`/`search` tool with no parameters is an
action, not a search. Needle was added to framework detection and to `framework_defaults.yml`.

**Result: 2 warnings, 2 sahi (0% noise).** AS-212 `reply_to_notification` sends a message with no
confirmation. AS-210 `deepseek/deepseek-flash-latest` is the training-data generator: weak, not
agent runtime.
**Possible false negative:** `set_oven` (a physical heating device) is now a low-impact write by
name, so a zero-config audit no longer asks for its confirmation. Impact tiering is name-based;
a spec with `approval_required` covers it.

**Effect on the 10 public repos:** warnings 34 → 31, blockers 3 → 3, and one AS-213 FP removed.
Two findings judged true in run 2 are no longer reported in `audit`, by design: AS-212 on
cs-agents' `update_seat` and on memory-agent's `upsert_memory` (low-impact writes). `check` at high
risk still reports them. Warning noise is **8/31 (26%)**.

## klavis-ai/klavis — first BLIND test (repo supplied by the user, static only, 2026-10-15)

`/Users/mac/Downloads/klavis-main`: an MCP platform. It holds dozens of MCP servers (636 `.py`
files, 154k lines) plus examples for 13 agent frameworks. **The library was run unchanged first,
and that result is recorded as-is** (`validation/results/klavis/blind_audit.json`). Fixes came after.

**Blind result: FAIL.**
- 62 s for the repo (3 s on everything before).
- **3 blockers, all false**: AS-214 "hard-coded secret" on `token_url = "https://..."`.
- **549 warnings, ~97% noise**:
  - 510 AS-401 taint paths inside a single MCP server, which has no LLM;
  - 22 AS-212 "no approval" on server tools, where approval is the MCP client's job;
  - 13 AS-213.
- The repo was labelled `crewai`, because one example uses it.
- The 13 examples show 0 tools. That part is correct: their tools come from Klavis's servers at runtime.

**Root causes (all general, each fixed with a regression test):**
1. No notion of an **MCP server** (a tool provider, not an agent). Server tools are now marked
   (`@mcp.tool`, a `@tool` alias in a FastMCP file, `types.Tool(...)`) and their `inputSchema` is read.
   Agent rules (AS-212 approval, AS-401 taint, AS-208 step bound) skip tools that no agent in the code binds.
2. **MCP `ToolAnnotations`** were ignored: 771 in this repo, including `**{...}` dict unpacking.
   `readOnlyHint` / `destructiveHint` / `idempotentHint` now decide read vs write, impact and
   idempotency before any name heuristic. New rule **AS-218** flags a destructive-looking tool that
   *claims* to be safe. A first version also flagged *unannotated* tools (162 findings). That was
   wrong: MCP's defaults already treat them as destructive. It was corrected before the numbers below.
3. Tool-name verbs: the first word is often a service (`reddit_create_post`, `close_create_lead`), so
   the first *known verb* is used. In `search_and_replace` the write after "and" wins. `export_*` is a write.
4. AS-214: a value that is a URL, a path or a constant name is where a secret goes, not a secret.
5. AS-213 needs body evidence. A schema-only tool (handler elsewhere) is not reported. A slice
   anywhere in the return (`objects[:20]`), or on an input parameter, counts as a bound.
6. Speed: `_used_as_tool_list` re-walked the whole project once per list constant. It is now one
   pass. **62 s → 10 s** on 154k lines.
7. Server transport functions (`run_streamable_http`) are no longer "hidden tools". That heuristic now
   needs a docstring and does not apply inside MCP server modules.

**After the fixes:** 982 tools (953 MCP-server tools, 426 annotated read-only), **0 blockers,
16 warnings: 14 sahi, 2 shor (12%)**. Judged sheet: `validation/results/klavis/judge.csv`.
- sahi: AS-218 `freshdesk_delete_thread` is declared `readOnlyHint: True` (a real bug: clients may
  delete without asking). 9 list/get tools return whole collections with no limit. 4 unpinned models.
- shor: `list_sheets` (one spreadsheet's sheet names) and `list_slide_templates` (a fixed set). The
  name-based AS-213 still cannot tell a small collection from a large one.

**What a blind test says about the library:** on a new kind of codebase (tool servers instead of
agents) the zero-config audit was unusable until fixed. The fixes are general, and they did not
change any other target's true findings (10 repos: 31 → 30 warnings; the one removed was a run-2 FP;
web-ui, needle and spike acceptance are unchanged). But "fixed after looking" is not "works blind".
The next blind repo is the real measure.


# Run 3 — the Level-2 rules on the same 10 repositories (2026-10-16)

`audit` only (AST, nothing installed or executed), same commits as run 1, `git status` clean afterwards.
An independent session judged every finding by reading the code (`validation-repos/RUN3_REPORT.md`):
55 findings, 40 TP, 6 FP, 9 debatable (strict precision 73%, 89% counting debatable as correct), plus
general false negatives. The general causes were then fixed; one-offs were left alone.

| finding (run 3) | cause | fix | after |
|---|---|---|---|
| AS-225 FP gpt-researcher | Monocle tracing (`setup_monocle_telemetry`) not known | tracing packages / calls, `os.environ[...TRACING...]`, `.env*` tracing switches | gone |
| AS-225 FP financial_research | Agents SDK `trace()` not known | SDK spans and default tracing count, unless tracing is disabled or the model is not OpenAI's | gone |
| AS-208 FP data-enrichment, open_deep_research | own loop counter (`loop_step < max_loops`) | counter-vs-limit comparisons are a step bound | gone |
| AS-223 FP data-enrichment | config injected per run (`configurable`, langgraph.json `env`) | counts as environment separation | gone |
| AS-222 false pass odr, gpt-researcher | a scraper's / sub-call's `timeout=` counted | only a timeout around an agent / graph / team run counts | now reported |
| AS-219 FN (6 sites) | only tool bodies were read; log/print counted as "reported" | helpers of tools, graph nodes, orchestrators; an error is reported only when it reaches the caller (raise, return, record) | 6 found, each checked by hand |
| AS-216 FN smolagents | URL reached `requests.get` through a helper method | parameters followed into project functions | found (visit_page, visualizer) |
| AS-210 FN smolagents | model id from a function default / argparse / `**params` | resolved | found (`o1`) |
| AS-215 FN smolagents | `CodeAgent(additional_authorized_imports=["*"])` | agents that execute model-written code: local + unrestricted is a finding, sandboxed executors pass | found ×2 |

Left as is: AS-213 on `find_on_page_ctrl_f` (one-off: output bounded by a fixed viewport), AS-213 on
CrewAI `RagTool` and AS-210 on local Ollama (debatable), AS-302 on a hub-and-spoke handoff demo
(debatable). One judgement differs from the run-3 reviewer: AS-225 on openai-cs-agents-demo is no longer
reported, because the OpenAI Agents SDK traces every run to the OpenAI platform by default.

By the run-3 judgements plus a hand check of each new finding: 62 findings, 52 TP, 1 FP, 9 debatable
(strict 84%, 98% counting debatable as correct). The new judgements are the maintainer's; an independent
re-review is the next step.


# Run 4 — 17 agents in 8 frameworks, single and multi-agent (2026-10-16)

An independent session built 17 agents (LangGraph, CrewAI, OpenAI Agents, PydanticAI, smolagents,
LlamaIndex, AutoGen RoundRobin + Swarm, Google ADK; single and multi-agent each) with full L1/L2/L3 specs,
ran them end to end against a scripted OpenAI-compatible stand-in (the Groq key had expired), captured
every LLM call with `discover`, and ran a 45-row mutation matrix per agent (`agenticreq-l3/RUN4_REPORT.md`).

| | before the fixes | after |
|---|---|---|
| matrix rows MISSED (17 agents) | 54 | 7, all by design (below) |
| FALSE-POS (negative controls) | 0 | 0 |
| ADK baseline blockers | 3 (S), 6 (M) | 0 |
| proven gaps (minimal snippets) | 15 | 13 fixed, G3 kept by design, G10 kept but printed correctly |

Fixed: Google ADK recognised (`google.adk` namespace import, `FunctionTool(require_confirmation=True)`,
`RunConfig.max_llm_calls` default, `run_async`); delegation from `create_supervisor`, `managed_agents`, and
a tool that runs another agent; budget passed through a config variable; chained `1 <= x <= 10` body
bounds; agent-as-tool never a write tool; AutoGen `output_content_type` / `TokenUsageTermination`;
`output_type=[str, ...]` is free text; a token cap is not a loop counter; `discover` reads child output as
UTF-8 on Windows; a `limit` parameter the body ignores bounds nothing; zero-trust through `as_tool`;
capture-mode: structured-output schemas and agents-as-tools are framework tools, and a request is
attributed by the tools it offers before prompt text (managers embed their sub-agents' descriptions).

The 7 remaining rows are design decisions: AS-306 on LangGraph / OpenAI Agents (their step bound counts
across the whole run), AS-208 on OpenAI Agents (framework default `max_turns`), and AS-301 for a removed
`team -> member` edge (a team's coordinator edges are the team itself, not delegation to declare).
Kept by design (G3): `future.result(timeout=)` / `wait_for(to_thread(...))` return control but the worker
thread keeps running and spending, so they are not a run-wide time limit — run 4's own smolagents run took
200 s under a 90 s timeout.

Not done in run 4: live runs on a real model (expired key), OpenTelemetry export for 4 of 8 frameworks.
