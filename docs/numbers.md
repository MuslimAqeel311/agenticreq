# Numbers (`numbers`)

**agenticreq never suggests numbers.** Every number in a spec (`bounds.*`, `max_delegation_depth`,
`progress_guard`, `output_size_limits`, `confidence_threshold`, `rate_limits`, `aggregate_budget`,
`approval_required[].timeout`, `limits[].limit`, `escalation[].threshold`) comes from you. agenticreq does two
things with a number:

1. it compares the number in `spec.yml` with the number in your code, and
2. it shows a number it FOUND in your code as plain evidence (`Found in code: ...`).

It does not propose, default, recommend, range or floor a number. `init` writes no number, `params` and the JSON
schema show types and meaning only ("a positive number" is validation, not advice), and examples show `<number>`
where a number goes. What is still decided by `risk_level` are yes/no and choice settings (`fail_closed`,
`kill_switch`, `on_approval_timeout`, ...).

## When a number fails

A number fails only when it is **not defined**, or when **code and `spec.yml` disagree**. The same table applies to
every number, at every risk level, and to each agent separately when there are several:

| situation | group (status) | reason |
|---|---|---|
| value in spec and in code, code at least as strict | Passed (PASS) | `Found in code: <value>.` |
| value only in code, not in spec | Passed (PASS) | `Found in code: <value>.` |
| value in spec, code is looser (spec 40, code 80) | Spec vs code gaps (GAP) | `Code allows 80 steps; spec.yml says 40.` |
| value in spec, code sets no limit | Spec vs code gaps (GAP) | `spec.yml says 40; the code sets no limit.` |
| value nowhere (also a framework default that means "no limit", such as LangGraph's) | Spec vs code gaps (GAP) | `No <name> set: write a value, or not_needed with a reason.` |
| value in spec is `TODO` | Spec vs code gaps (GAP) | `<name> is still TODO in spec.yml.` |
| value in spec, but the library cannot read this limit from code | Check with a mock run (CHECK) | `spec.yml says <value>; not readable in code (or sign it off under confirmed:). Run: agenticreq mock-run` |
| same, signed off under `confirmed:` | Human sign-off (SIGNED) | `Signed by <who> on <date>.` |
| same, seen within the limit in every captured run | Passed (PASS) | `Held in N of N runs: ...` |

* "At least as strict" means lower for the `max_*` numbers and higher for `confidence_threshold`.
* A number that is zero or below, or the wrong type, is a spec shape error (exit 2); `confidence_threshold` must be
  between zero and one. Nothing else about the number is judged.
* Cost and duration cannot be read from code in most frameworks (`UsageLimits`, a budget callback and a `wait_for`
  prove that a limit exists, not its value); progress guard, rate limit, output size, confidence threshold and the
  approval timeout live in a runtime, a gateway or a process. For those the result waits for a mock run until you
  sign the number off under `confirmed:` (the clause is the key, e.g. `bounds.max_cost_usd`, its name `max_cost_usd`,
  or its parent field `bounds`) or a captured run shows it held.
* A number nobody asked for is not newly required (for example `max_cost_usd` when the spec does not state one).
  The exceptions are `progress_guard.max_no_progress_steps` and, at `critical`, `confidence_threshold`; the message
  names no value.

## What stays a floor

Yes/no safeguards keep their floor per risk level (a weaker value is AS-111): `self_modification`, `fail_closed`,
`on_approval_timeout`, `kill_switch`, `log_immutable`, `version_pinning`, `audit_log`, `dry_run_required` and
`transitive_prohibition`. There is no floor on a number.

## Framework names in code -> library parameters

Code extraction reads each framework's own spelling of a bound and maps it to one library parameter. Nothing
here is a guess about meaning: each name is a keyword argument (or constructor) the framework documents for that
bound. The tables live in `src/agenticreq/ast/extract_support.py` (`BOUND_KWARGS`, `BOUND_CALLS`,
`_DURATION_KWARGS`, `_COST_KWARGS`) and `src/agenticreq/rules/conformance.py` (`_STEP_KINDS`).

| Library parameter | Read from these names in code | Frameworks that use them |
|---|---|---|
| `bounds.max_steps` | `recursion_limit`, `max_iter`, `max_iterations`, `max_iters`, `maximum_iterations`, `max_tool_iterations`, `max_turns`, `max_round`, `max_messages`, `max_consecutive_auto_reply`, `MaxMessageTermination(n)`, `max_steps`, `max_agent_steps`, `max_llm_calls`, `request_limit`, `tool_calls_limit`, `tool_call_limit`, `maximum_auto_invoke_attempts`, `max_auto_invoke_attempts`; also a counted `while` / `range()` loop around the model call | LangGraph, CrewAI, OpenAI Agents SDK, AutoGen, smolagents, Google ADK, PydanticAI, Haystack, DSPy, Semantic Kernel, Agno |
| `bounds.max_duration_s` | `max_execution_time`, `max_time`, `time_limit`, `deadline`, `max_duration`, `max_duration_s`, `execution_timeout`, `run_timeout`, `max_wall_time`; `timeout=` only on an agent or a run call | CrewAI, generic run wrappers |
| `bounds.max_cost_usd` | `max_cost`, `max_cost_usd`, `max_budget`, `max_budget_usd`, `budget`, `budget_usd`, `cost_limit`, `max_price`, `max_spend`, token budgets (`total_tokens_limit`, `max_total_tokens`, ...), `UsageLimits`, `BudgetManager`, `TokenBudget`, `CostLimit`, `TokenUsageTermination` | PydanticAI, AutoGen, LiteLLM, generic |
| `bounds.max_retries` | `max_retries` | most HTTP / model clients |
| `bounds.max_tool_calls` | `max_tool_calls`, `tool_calls_limit` | PydanticAI, generic |

When the code sets no bound, the user's own framework may apply one (`src/agenticreq/data/framework_defaults.yml`).
It is shown as a labelled fact about that framework ("no explicit bound; openai-agents: max_turns=..."), compared with
the number in spec.yml, and never offered as advice for what to write.

These framework names are for **code**. In **spec.yml** the library's own names are used (`bounds.max_steps`, ...).
A framework name written in spec.yml stops `check` with the closest library name: `max_iterations` gives
"Unknown field `max_iterations` in spec.yml (line 12). Did you mean `bounds.max_steps`?". The few "also called"
words behind such suggestions are part of each parameter's entry in the params catalog (`agenticreq params`).
