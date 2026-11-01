# SCOPE — what agenticreq does, what it does not, and where it is blind

This is the credibility document. If a claim is not supported here, agenticreq does not make it.

## What it does

agenticreq compares an agentic AI system's **declared requirements** (a human-written
`spec.yml`) with its **implementation**. It does this statically from the source (AST) and
dynamically from the LLM HTTP boundary (capture, MCP `tools/list`, OTel traces). Every claim
carries a confidence level:

| level | meaning | can it PASS? |
|---|---|---|
| `proven` | read directly from config / wire / AST | yes |
| `heuristic` | inferred from structure (names, patterns, request ordering) | yes, labelled |
| `observed (n/N)` | seen in n of N real traces/runs; never promoted to `proven` | "no violation observed" only |
| `insufficient_evidence` | traces exist but did not cover the scenario | **never** |
| `unknown` | could not be extracted | **never** |
| `attestation` | not verifiable from code; a human declares it | recorded, never verified |

Reports have four buckets (BLOCKERS / WARNINGS / UNKNOWNS / ATTESTATIONS) and **no overall
score**. Counts are over the extracted portion only, and the report says how much that was.

Traces **prove violations, not safety.** "violation found" (AS-204, AS-402) and "no violation
observed (0/N)" are deliberately different words. For safety claims such as `prohibited`, static
reachability is the primary evidence.

## What it does not do

- **No enforcement.** No runtime guardrail, firewall, proxy, or `bind()`. `export` hands
  policies to OPA / Cedar / tool schemas; those systems enforce.
- **No LLM anywhere.** Not in the library and not in the tests. `discover` answers the agent's
  LLM calls with a deterministic canned responder.
- **No prompt parsing.** The rules come from the human spec, never from the system prompt.
- No LLM-as-judge, no evals, no accuracy / latency / hallucination metrics (that is
  observability), no compliance or regulatory mapping, no web UI, no hosted service.
- It never modifies user code, except through `agenticreq fix --apply`, which you must request
  explicitly. The default is a dry-run diff.
- It never writes into your project on its own. Installing and importing create nothing. Only
  `init` creates `spec.yml`, and it refuses to overwrite one without `--force`. Other files appear
  only when you name them (`-o`, `discover --out`, `export-spec --out`).
- Without a spec, `audit` checks only Level-2 engineering defaults. It reports the Level-1/3 business
  rules (`prohibited`, `limits`, `approval_required`, `escalation`, `untrusted_inputs`,
  `role_boundary`, `declared_delegation`) as UNKNOWN (AS-907), never as passed.

## Spike results this build relies on

See "Spike gaps" and "Spike headline numbers" below. Those are copied verbatim from the spike's
`gaps.md`.

## Acceptance against the spike's own agents (added 2026-10-15)

The spike folder (`Spike/`: REPORT.md, gaps.md, 7 agents, captures) arrived after the first build.
Its 7 agents and S2 captures are now checked in under `tests/fixtures/spike/` and run by
`tests/test_spike_acceptance.py`. The captures are the spike's recorded wire bodies, imported
(`capture/importers.py`), not re-run.

**First run on them FAILED.** The AST found **0 declared tools in all 7 agents**: the spike defines
its tools in `shared_tools.py` and wraps them in each agent file (`tool(parse_docstring=True)(f)`,
`[tool(f) for f in (...)]`, `function_tool(f)`, bare functions in `tools=`). The extractor did not
follow imports or recognise those wrappers, so every wire tool became a false AS-202 (27 in total,
each labelled `proven`), and the hidden tools of `langgraph_phased` were missed. The old fixtures
had defined every tool in the same file with a decorator, so the tests never exercised this.

**After the fix (symbol table + one tool resolver, see README "Architecture"):**

| agent | spike verdict (S2) | spike: visible ÷ defined | agenticreq: declared (AST) | wire ∩ declared | spike: fields lost | agenticreq fidelity findings |
|---|---|---|---|---|---|---|
| langgraph_react | VERIFIABLE | 4/4 | 4 | 4 | 0 | none ✓ |
| langgraph_phased | PARTIAL | 3/5 | 5 | 3; `issue_refund` (unexecuted branch) and `search_policy` (direct Python call) → AS-201 `unknown` with those reasons ✓ | 0 | none ✓ |
| crewai | PARTIAL | 4/4 | 4 | 4 | 13 (desc 9, required 2, default 2) | AS-602×9, AS-603×2, AS-605×2 ✓ |
| pydanticai | VERIFIABLE | 4/4 | 4 | 4 | 0 | none ✓ |
| openai_agents | PARTIAL | 4/4 | 4 | 4 | 2 (required) | AS-604×2 ✓ (defaults kept → contradictory) |
| smolagents | PARTIAL | 4/4 | 4 | 4 | 2 (default) | AS-605×2 ✓ |
| smolagents_code | NOT VERIFIABLE → UNKNOWN | 0/4 | 4 | 0 → AS-901 + 4× AS-201, all `unknown`, no PASS ✓ | n/a | n/a ✓ |

Also matched: S2 call coverage 24/24 imported; AS-202 false positives 0; the `no_huge_orders`
input guardrail found statically (F6); CrewAI delegation and OpenAI Agents handoff edges built.

**`check` with a spec.** `tests/fixtures/spike/spec.yml` writes the spike's prose `SYSTEM_RULES`
as a spec (refund prohibited, `pay_invoice` needs approval, amount ≤ 5000, quantity 1–100,
`search_policy` untrusted). Blockers per agent were compared by hand with the agent sources:
AS-205 (no approval gate on `pay_invoice`) and AS-207 ×3 (limits only in docstrings) on all 7;
AS-203 on `langgraph_phased` (`issue_refund` bound on the escalate branch); AS-401 taint on every
agent where untrusted content reaches a write tool. That includes two routes found only after
generalising the taint rule: the `langgraph_phased` prefetch node (Python calls `search_policy`
and puts the result in the prompt, never visible at the wire), and CrewAI delegation (the
analyst's answer returns to the manager, who holds the write tools). OpenAI Agents is correctly
**not** flagged: its handoff moves the conversation forward to the analyst, who has no write tool.

**Not verified:** the spike's numbers were produced by its own canned model on Windows with other
library versions (crewai 1.15.28, openai-agents 0.20.0). agenticreq did not re-run those agents, it
re-analysed the recorded bodies. Field counting still differs from the spike's (see below); the
fidelity *findings* match one-to-one, the ratio definitions do not.

## What this build re-verified (macOS x86_64, Python 3.11)

The numbers in this section come from substitute fixture agents in `tests/fixtures/`, run live on
real framework versions with the canned responder (the spike's own agents are covered above).

| fixture | framework (version) | client lib | captured | zero-touch |
|---|---|---|---|---|
| langgraph_po | LangGraph 1.2.15 + langchain-openai 1.7.2 | httpx2 | 4/4 | yes |
| oai_agents_po | openai-agents 0.23.1 (Responses API) | httpx2 | 5/5 | yes |
| oai_agents_chain | openai-agents 0.23.1, 3 agents | httpx2 | 6/6 | yes |
| crew_po | CrewAI 1.9.3 (openai 1.83) | httpx | 6/6 | yes |
| crew_spike_shape | CrewAI 1.9.3 (openai 1.83), spike's tool signatures | httpx | 5/5 | yes |
| smol_tools | smolagents 1.26.0 ToolCallingAgent | httpx2 | 3/3 | yes |
| smol_code | smolagents 1.26.0 CodeAgent | httpx2 | 1/1 | yes |
| pydantic_ai_po | pydantic-ai-slim 2.56.0 (Anthropic) | httpx2 | 3/3 | yes |
| anthropic_beta | anthropic 1.14 `beta.messages.create` | httpx2 | 3/3 | yes |
| httpx2_async_stream | openai 3.31 AsyncOpenAI, `stream=True` | httpx2 | 2/2 | yes |
| httpx_legacy | raw httpx 0.28 | httpx | 2/2 | yes |
| urllib_bypass | urllib (not httpx) | — | **0/1** (miss counted, not hidden) | yes |

Schema drift found on these fixtures:
- **CrewAI 1.9.3:** all 9 parameter descriptions dropped, and 4 optional params sent as
  required (21/34 fields verbatim). Unlike the spike, CrewAI 1.9.3 **kept** the `'KHI'` default.
- **OpenAI Agents 0.23.1:** 2 optional params sent as required, matching the spike.
- **smolagents 1.26:** default dropped, matching the spike.
- **LangGraph / langchain `@tool`:** all Google-style `Args:` descriptions dropped. The spike
  did not report this.
- **PydanticAI 2.56:** faithful.

**Field counting is not the spike's.** The spike's definition of a "field" (44 per agent) was not
available. agenticreq counts: tool description + per param {type, required, description if
declared, default if declared, each constraint}. Absolute ratios are not comparable with the spike.

## Spike gaps (verbatim from the spike's gaps.md)

> Source: `spike.py` console output, section `V12 gaps:`
> 7 test agents × 3 interception strategies × 24 LLM calls.

1. **[S2a blind spot]** langgraph_react: httpx-only hook captured 0/3; S2 shows the calls went via ['httpx2'].

2. **[incomplete inventory]** langgraph_phased: `issue_refund` is registered on a branch the run never reached → never sent to the LLM → invisible.

3. **[client-side tool use]** langgraph_phased: `search_policy` is executed directly from Python; only its output appears, as message text, never as a tool.

4. **[client-side logic]** langgraph_phased: route1() branch condition (ESCALATE) runs locally and leaves no trace in any LLM request.

5. **[client-side logic]** langgraph_phased: phase-dependent bind_tools() runs locally and leaves no trace in any LLM request.

6. **[framework-injected tools]** crewai: ['ask_question_to_coworker', 'delegate_work_to_coworker'] are on the wire but not in user code (renamed: 'Delegate work to coworker' → delegate_work_to_coworker).

7. **[schema rewritten by framework]** crewai: 31/44 fields verbatim; lost: ['submit_order.sku: param description dropped', 'submit_order.quantity: param description dropped', 'submit_order.priority: param description dropped', 'pay_invoice.invoice_id: param description dropped', 'pay_invoice.amount: param description dropped', 'get_inventory.sku: param description dropped', 'get_inventory.warehouse: required=True but code says False', 'get_inventory.warehouse: param description dropped', "get_inventory.warehouse: default 'KHI' -> '<absent>'", 'search_policy.query: param description dropped', 'search_policy.max_results: required=True but code says False', 'search_policy.max_results: param description dropped', "search_policy.max_results: default 3 -> '<absent>'"]

8. **[S2a blind spot]** crewai: httpx-only hook captured 0/5; S2 shows the calls went via ['httpx2'].

9. **[fake-response cost]** pydanticai: framework streams (stream=True even from run_sync); the canned responder had to emit SSE. Responses-API streaming is NOT faked.

10. **[S1 blind spot]** pydanticai: 3 LLM calls bypassed the spec's S1 targets via ['httpx2: POST api.anthropic.com/v1/messages'].

11. **[S2a blind spot]** pydanticai: httpx-only hook captured 0/3; S2 shows the calls went via ['httpx2'].

12. **[client-side logic]** openai_agents: input guardrail no_huge_orders() runs locally and leaves no trace in any LLM request.

13. **[framework-injected tools]** openai_agents: ['transfer_to_policy_analyst'] are on the wire but not in user code.

14. **[schema rewritten by framework]** openai_agents: 42/44 fields verbatim; lost: ['get_inventory.warehouse: required=True but code says False', 'search_policy.max_results: required=True but code says False']

15. **[S1 blind spot]** openai_agents: 4 LLM calls bypassed the spec's S1 targets via ['httpx: POST api.openai.com/v1/responses'].

16. **[framework-injected tools]** smolagents: ['final_answer'] are on the wire but not in user code.

17. **[schema rewritten by framework]** smolagents: 42/44 fields verbatim; lost: ["get_inventory.warehouse: default 'KHI' -> '<absent>'", "search_policy.max_results: default 3 -> '<absent>'"]

18. **[framework prompt wrapping]** smolagents: system prompt 4806 chars vs 283 chars of user rules.

19. **[tool results as prose]** smolagents: 2/2 results only in plain text

20. **[no schema at all]** smolagents_code: `tools` is empty on every request; ['get_inventory', 'pay_invoice', 'search_policy', 'submit_order'] exist only as Python signatures inside a 9826-char system prompt; results come back as unattributed execution logs. Boundary capture must report UNKNOWN for this execution model.

21. **[framework prompt wrapping]** smolagents_code: system prompt 9826 chars vs 283 chars of user rules.

22. **[tool results as prose]** smolagents_code: 2/2 results only in plain text, 2 not attributable to a tool

23. **[delegation edges]** no request carries a parent/child id; the delegation graph must be inferred from ordering and the target name in handoff args. Parallel/async delegation breaks ordering.

24. **[path coverage]** runtime capture only sees the executed path. It is a LOWER BOUND on the tool inventory, not the declaration.

25. **[not tested]** MCP tools, provider-hosted tools, parallel tool calls, Bedrock/Vertex/Azure wire formats, non-httpx clients (aiohttp-native, requests, gRPC), local models, 13 other frameworks.

### Spike headline numbers

```
call coverage   S1=71% (17/24)   S2a=54% (13/24)   S2=100% (24/24)
frameworks with captures: 5 [crewai, langgraph, openai_agents, pydantic_ai, smolagents]
HYPOTHESIS: CONFIRMED-WITH-CAVEATS (not verifiable at the boundary: smolagents_code)
```

| agent | verdict (S2) | inventory | schema fields verbatim |
|---|---|---|---|
| langgraph_react | VERIFIABLE | 4/4 | 100% |
| langgraph_phased | PARTIALLY VERIFIABLE | 3/5 | 100% |
| crewai | PARTIALLY VERIFIABLE | 4/4 | 70% |
| pydanticai | VERIFIABLE | 4/4 | 100% |
| openai_agents | PARTIALLY VERIFIABLE | 4/4 | 95% |
| smolagents | PARTIALLY VERIFIABLE | 4/4 | 95% |
| smolagents_code | NOT VERIFIABLE → must report UNKNOWN | 0/4 | none |

**Important caveat:** all spike numbers come from a canned
(fake) model driving a single execution path. A real model would choose different
tools and arguments, so the 0.80 mean visible/defined ratio is specific to that
harness, not a general claim about boundary capture.

## New gaps found in this build (not in the spike)
- **L1** `discover` coverage means *LLM calls captured / LLM calls attempted*. It is **not** tool
  or branch coverage. The responder walks one deterministic path.
- **L2** Mock patching (`patch: module:attr`) works only for importable module attributes. Tools
  defined inside the entry script cannot be patched. `discover` executes the agent's code and its guards
  (HTTP hook, socket guard, process / file-write guard) are **best-effort, not a security sandbox**: they stop
  Python-level actions only, so `_socket.socket.connect`, ctypes / native code, reading `~/.ssh` and writes
  through a symlink get through. Run `discover` only inside a container with no network and no secrets
  (`--require-container` refuses to start anywhere else). Point it at side-effect-free tools.
- **L3** The socket guard blocks all INET connects, which may break asyncio on Windows (loopback
  socketpair). It was tested only on macOS.
- **L4** Streaming: canned SSE exists for Chat Completions, Responses and Anthropic. Only Chat
  Completions streaming is exercised by a test (OpenAI Agents and PydanticAI fixtures run
  non-streaming).
- **L5** Write-tool, untrusted-source, and unbounded-tool classification is name-based
  (`heuristic`). A tool named `process()` that deletes data is not flagged as a write, and AS-211
  still judges idempotency by name (validation R2: 4/4 false positives in cs-agents, not fixed).
- **L5a** (validation H1, fixed) The canned responder answers structured-output requests
  (`response_format` / Responses `text.format` json_schema, Anthropic `output_format`) with JSON that
  validates against the schema. Booleans default to `true`, so `is_relevant` / `is_safe` guardrails
  pass; mocks.yml `structured: {field: value}` overrides any field. A guardrail phrased the other way
  (`is_jailbreak`) would trip and needs that override.
- **L5b** AS-208 accepts a framework's finite default loop bound (`data/framework_defaults.yml`:
  OpenAI Agents `max_turns=10`, PydanticAI `request_limit=50`, CrewAI `max_iter=25`, smolagents
  `max_steps=20`) as a `heuristic` pass. These values are version-dependent and were taken from the
  frameworks' sources, not re-read per version. App-level counters (open_deep_research's
  `max_researcher_iterations`) are still not recognised. The same bound turns a delegation cycle
  (AS-303) into a warning ("bounded by ..."); with no bound it stays a blocker.
- **L5d** Agent nodes from the wire are named by the declared prompt, then by an exact match of the
  offered tools with one declared agent's tools, then (weakest) by an agent name mentioned in the
  prompt. Agents with identical tool sets and runtime-built prompts can still be mis-named. AS-305
  reports each delegator/delegate pair separately (14 findings on one hub in validation).
- **L5c** Static taint (AS-401) follows three routes: the agent holds the source; Python calls the
  source in a file that binds the agent's write tools; or an edge connects the agents
  (delegation results flow back to the delegator, handoffs flow forward to the target). The
  "same file" test for direct calls is coarse: it does not prove the result reaches the prompt.
- **L6** Approval-gate detection is pattern-based (`interrupt`, `interrupt_before`,
  `needs_approval`, `human_input`, guardrail decorators). Custom approval code is missed, so
  AS-205 can be a false positive. A global gate's scope is not proven.
- **L7** Static extraction cannot resolve dynamic tool lists (registries filled at runtime, plugin
  loading, tools built from config or state, e.g. data-enrichment's `Info` schema). Such tools are
  absent from `declared_tools`. Comprehensions over literal lists / constants, `+`, `*starred`,
  local variables and cross-module constants **are** resolved since the extractor rewrite.
- **L7b** Local imports are followed only to `.py` files under the scanned root, the entry's
  directory, or the nearest project root (`pyproject.toml` / `setup.py` / `.git`, at most 3 levels up,
  plus its `src/`). Framework packages (`FRAMEWORK_PACKAGES` in `ast/symbols.py`) are never followed
  unless they sit inside the scanned root. A local package that shadows one of those names
  (e.g. a project package called `agents`) outside the scanned root is therefore not followed.
- **L7c** A function from a "tool module" (≥ half of its public functions are tools) that the agent
  calls directly from Python is reported as a hidden tool with kind
  `tool-module function (heuristic)`. In a utils-style module it is not.
- **L8** The LangGraph recognizer is static. A router function in `add_conditional_edges` without
  a path map makes every node reachable (conservative).
- **L9** `data/model_pins.yml` is a static alias→snapshot table. Verify it against provider docs
  before `fix --apply`.
- **L10** `fix` adds `idempotency_key` to the signature only. Wiring it to the side effect is left
  as a TODO comment.
- **L11** (removed) the Tier-4 `reliability_budget` was dropped: it assumed independent steps and was a
  planning number, not a measurement. Tier 4 now has two derived values, `trace_coverage` and
  `enforceability` (the latter read from the rule results).
- **L12** trace_coverage relies on OTel GenAI attribute names (`gen_ai.*`) and simple event names
  (`approval.*`, `escalat*`), or explicit `agenticreq.scenario` tags.
- **L13** Python only. JS/TS agents (LangGraph.js, Mastra, Vercel AI SDK) are not supported.
- **L14** Exported Rego was checked with OPA 1.4.2 (`opa check` + `opa eval`) and Cedar with
  cedarpy 4.12 on one spec. Other policy shapes are generated by the same code but were not
  evaluated individually.
- **L15** (relates to spike #25) MCP: this build adds a `tools/list` stdio client, tested only
  against a fake server. There is no SSE / streamable-HTTP transport and no `tools/call`
  observation.
- **L16** (relates to spike #25) Provider-hosted tools are recorded as `hosted:*` with `unknown`
  confidence (AS-906). Their calls and results are not verified.
- **L17** (relates to spike #5) Phase-dependent `bind_tools()`: when a tool's wire schema changes
  between phases, only the first schema seen is used for fidelity; later variants are not diffed.
- **L18** (relates to spike #9) The canned responder now emits Responses-API SSE as well, but no
  fixture exercises it (L4). The spike's "Responses-API streaming is NOT faked" is unproven here.
- **L19** (relates to spike #8) Version drift: the spike saw CrewAI on httpx2. CrewAI 1.9.3 pins
  `openai<3`, so in this build it runs on httpx, in a separate venv. For schema differences see L20.
- **L20 — version-drift observation (relates to spike #7).** The spike's CrewAI tool signatures
  were rebuilt from gaps.md #7 in `tests/fixtures/agents_crewai/crew_spike_shape`: `submit_order(sku, quantity, priority)`,
  `pay_invoice(invoice_id, amount)`, `get_inventory(sku, warehouse='KHI')`, `search_policy(query, max_results=3)`.
  It was run on CrewAI 1.9.3 (capture: `tests/fixtures/captures/crewai_spike_shape.json`).
  Assumption: every parameter's description is written as a Google-style `Args:` docstring;
  the spike's source was not available.

  | field (spike gaps.md #7) | loss | spike (CrewAI version used by the spike) | CrewAI 1.9.3 |
  |---|---|---|---|
  | submit_order.sku | description dropped (AS-602) | lost | lost |
  | submit_order.quantity | description dropped (AS-602) | lost | lost |
  | submit_order.priority | description dropped (AS-602) | lost | lost |
  | pay_invoice.invoice_id | description dropped (AS-602) | lost | lost |
  | pay_invoice.amount | description dropped (AS-602) | lost | lost |
  | get_inventory.sku | description dropped (AS-602) | lost | lost |
  | get_inventory.warehouse | optional sent as required (AS-603) | lost | **became AS-604** |
  | get_inventory.warehouse | contradictory contract: required + default on the wire (AS-604) | — (default absent, so not applicable) | **lost** (`required: true`, `default: 'KHI'`) |
  | get_inventory.warehouse | description dropped (AS-602) | lost | lost |
  | get_inventory.warehouse | default 'KHI' dropped (AS-605) | lost | **kept** (`default: 'KHI'` on the wire) |
  | search_policy.query | description dropped (AS-602) | lost | lost |
  | search_policy.max_results | optional sent as required (AS-603) | lost | **became AS-604** |
  | search_policy.max_results | contradictory contract (AS-604) | — (default absent) | **lost** (`required: true`, `default: 3`) |
  | search_policy.max_results | description dropped (AS-602) | lost | lost |
  | search_policy.max_results | default 3 dropped (AS-605) | lost | **kept** (`default: 3` on the wire) |
  | *(any field not in the spike's list)* | — | — | nothing additional lost |

  Result: 11 of the spike's 13 losses still occur on 1.9.3. Both default drops do not. For the
  two optional→required fields, 1.9.3 now sends the `default` while still listing the field in
  `required`. That schema contradicts itself, and it is worse than either drift alone: the model
  must supply, on every call, a value the code says it does not need. agenticreq reports it as
  **AS-604 (blocker)** and suppresses AS-603 on that field, so one problem produces one finding.
  The same shape appears in OpenAI Agents 0.23.1 strict mode (2 fields). No real capture in this
  build shows plain AS-603 (required without a default, the spike's shape); only the rule's
  synthetic test exercises it. Counted the agenticreq way, 22/33 fields reach the model
  verbatim. That is not comparable with the spike's 31/44 (different field definition, see above).

  **Conclusion:** framework schema rewriting is version-dependent. agenticreq therefore measures
  schema fidelity empirically on every run, holds no expected-drift table, and puts the resolved
  framework version and its source at the start of the evidence of every fidelity finding
  (`crewai 1.9.3 (runtime) | ...`). The same tag goes on the other version-sensitive rules:
  AS-202 (injected-tool naming), AS-901 (execution model), and AS-301–305/308 (delegation
  inferred from handoff conventions). The version recorded inside the captured run takes
  precedence over the one installed where agenticreq runs. When no version resolves, the evidence
  says `(version unresolved)` instead of leaving it out.
