# Rules that reach only a code pattern from code alone

A PASS needs evidence: read from the code (config, wire or AST), seen in every captured run, or a waiver. A
code-pattern result (the code suggests the rule holds, but nothing shows it) lands in **Check with a mock run**, and
its line ends with the next command. A sign-off under `confirmed:` lands in **Human sign-off** as SIGNED: it is never
counted as passed. The line for a parameter is PASS only when **every** result on it passes.

These rules can record a pass only as a code pattern when agenticreq reads the code. The last column says what, if
anything, can still turn the line into a PASS. "No route" means that no mock run, trace or sign-off changes that
rule's result. The line stays in Check with a mock run unless the check is switched off in spec.yml, which the
risk-level floors allow only where the column says so.

| rule | what the pass would claim | parameter (line) | route to PASS |
|---|---|---|---|
| AS-203 | a prohibited tool is absent from the code | `prohibited` | Runtime evidence. When every captured run (`agenticreq mock-run` or real captures) shows the model's tool list and the tool is never in it or called, AS-203 is seen in every run: `PASS Held in N of N runs`. From code alone it stays a code pattern (or waits for a mock run, with tools agenticreq cannot resolve). |
| AS-205 | an `approval_required` tool has a human-approval gate on every path | `approval_required` | Runtime evidence. A mock run asks for the tool, and records approval gates: `input()`, LangGraph `interrupt()`, OpenAI Agents `needs_approval` (only when it requires approval). An approval fired before the tool's result in every run that asked for it: `PASS Held in N of N runs`. The result came back with no approval recorded: `CHECK <tool> ran without a recorded approval` (a gate of another kind: sign it off under `confirmed:`). |
| AS-208 | a step bound exists (the framework's built-in default) | `bounds.max_steps` | Write an explicit bound in code (AS-209, read from the code). A run-time AS-229 pass (mock run / traces) does not remove the AS-208 result. |
| AS-209 | the code's bound is within the spec's (a number in code passes on its own; `docs/numbers.md`) | `bounds.max_steps` / `bounds.max_retries` / `bounds.max_tool_calls` | Read from the code when the bound is a literal. A code pattern when it is the default of an env read (`int(os.getenv("MAX", "10"))`): write the literal. |
| AS-210 | the model id is pinned | `version_pinning` | Read from the code for a literal model id. A code pattern when it is an env default: write the dated model id in code. |
| AS-215 | a code executor runs isolated | `sandbox_policy` | No route (deployment property). |
| AS-216 | a URL-fetching tool has an egress guard | `egress_policy` | No route. |
| AS-219 | tools do not swallow errors (fail-closed) | `fail_closed` | Switch off with `fail_closed: false` (off by default at `low`; allowed at `medium`; floor at `high` / `critical`). |
| AS-221 | a run-wide cost / token budget exists | `bounds.max_cost_usd` | Only checked when the spec declares `bounds.max_cost_usd` (audit: when the code has none). The value is rarely readable from code: sign it off under `confirmed:` (`bounds.max_cost_usd`). No runtime check. |
| AS-222 | a run-wide time limit exists | `bounds.max_duration_s` | Only checked when the spec declares `bounds.max_duration_s` (audit: when the code has none). Read from `max_execution_time=`; otherwise sign it off under `confirmed:`. No runtime check. |
| AS-223 | configuration is separated by environment | `environment_separation` | Switch off with `environment_separation: false` (no floor at any risk level). A sign-off records the clause but does not remove the AS-223 result. |
| AS-224 | output is a validated structure | `structured_output` | Off by default at `low` / `medium`; switch off with `structured_output: false` at `high` / `critical` (no floor). |
| AS-225 | an audit / trace hook records tool calls | `audit_log` | Switch off with `audit_log: false` and `trace_policy: ""` (floor on `audit_log` only at `critical`). |
| AS-226 | a graph that must checkpoint has a checkpointer | `checkpointing` | Off by default at `low` / `medium`; switch off with `checkpointing: false` (no floor). |
| AS-227 | a write tool has a dry-run / preview parameter | `dry_run_required` | Only checked when `dry_run_required` is true (default at `high` / `critical`; floor at `critical`). |
| AS-202 | a wire tool not declared in code is the framework's own | `tool_inventory_report` | Read from the code for allowlisted framework tools (`known_injected_tools.yml`). A code pattern when inferred from a structured-output schema or an agent called as a tool. |
| AS-301 / AS-302 | a delegation edge is declared / delegation depth is within the limit | `declared_delegation` / `max_delegation_depth` | Read from the code for edges written in code. A code pattern for edges inferred from captured tool calls. Seen in runs for edges in OTel traces (`parent_span_id`). |
| AS-306 | a multi-agent run has an aggregate budget | `aggregate_budget` | No route for the budget itself. Its numbers (`aggregate_budget.max_steps` / `max_cost_usd` / `max_tool_calls`) are compared with the run-level bound in code or signed off under `confirmed:`. |
| AS-307 | one trace id spans all agents | `end_to_end_trace_id` | No route. |

Rules that **can** reach PASS through runtime evidence (the latest `agenticreq mock-run`, `--captures` or `--traces`):
AS-228 (`progress_guard.max_no_progress_steps`), AS-229 (`bounds.max_steps` / `bounds.max_tool_calls`, measured
against the number in spec.yml), AS-230 (`stop_conditions`), AS-231 (`rate_limits`). Each passes only when the rule
held in every run ("Held in N of N runs"). Fewer is `CHECK Held in only n of N runs`.

Clauses code cannot show at all (AS-501: owner, escalation, kill switch, ...) are **Human sign-off** items. A person
confirms them under `confirmed:` in spec.yml (AS-502); the line then reads `SIGNED Signed by <who> on <date>.` and
stays in the sign-off group. It opens again when its `expires` date passes, or when a file named in its `evidence`
(`file:line`) changed after its `date`. The entry's `answer:` decides the line: `yes` (by, date, expires, evidence)
is SIGNED; `no` (by, date, reason) is a gap, `Owner says this is not in place yet: <reason>` (AS-503); `not_needed`
(by, date, reason) is `NOT NEEDED <reason>` (AS-504), never allowed at high or critical risk for the safety items.
