# Numbers

**agenticreq never suggests numbers.** Every number in a spec (`bounds.*`, `max_delegation_depth`, `progress_guard`,
`output_size_limits`, `confidence_threshold`, `rate_limits`, `aggregate_budget`, `approval_required[].timeout`,
`limits[].limit`, `escalation[].threshold`) comes from you. agenticreq does two things with a number: it compares
the number in `spec.yml` with the one in your code, and it shows a number it found in your code as evidence
(`Found in code: ...`). `init` writes no number; `params` and the JSON schema show types and meaning only.

## When a number fails

A number fails only when it is **not defined**, or when **code and `spec.yml` disagree**. The same table applies to
every number, at every risk level, and to each agent separately.

| situation | group | reason |
|---|---|---|
| in spec and code, code at least as strict | Passed | `Found in code: <value>.` |
| only in code | Passed | `Found in code: <value>.` |
| in spec, code looser (spec 40, code 80) | Spec vs code gaps | `Code allows 80 steps; spec.yml says 40.` |
| in spec, code sets no limit | Spec vs code gaps | `spec.yml says 40; the code sets no limit.` |
| nowhere | Spec vs code gaps | `No <name> set: write a value, or not_needed with a reason.` |
| `TODO` in spec | Spec vs code gaps | `<name> is still TODO in spec.yml.` |
| in spec, not readable from code | Check with a mock run | `... not readable in code (or sign it off under confirmed:).` |
| same, signed off | Human sign-off | `Signed by <who> on <date>.` |
| same, held in every captured run | Passed | `Held in N of N runs: ...` |

- "At least as strict" means lower for `max_*` numbers and higher for `confidence_threshold`.
- A number that is zero or below, or the wrong type, is a spec error (exit 2); `confidence_threshold` must be
  between zero and one.
- Cost, duration, progress guard, rate limit, output size and approval timeout often live in a runtime or gateway, so
  they wait for a mock run until you sign the number off under `confirmed:` or a captured run shows it held.
- A number nobody asked for is not newly required. The exceptions are `progress_guard.max_no_progress_steps` and, at
  `critical`, `confidence_threshold`.

## What stays a floor

Yes/no safeguards keep a floor per risk level (a weaker value is AS-111): `self_modification`, `fail_closed`,
`on_approval_timeout`, `kill_switch`, `log_immutable`, `version_pinning`, `audit_log`, `dry_run_required` and
`transitive_prohibition`. Numbers have no floor.

## Framework names in code

Code extraction maps each framework's own spelling of a bound to one parameter (`recursion_limit`, `max_iter`,
`max_turns` and similar to `bounds.max_steps`; `max_execution_time` to `bounds.max_duration_s`; `UsageLimits` and
budget settings to `bounds.max_cost_usd`; `max_retries`; `max_tool_calls`). The tables are in
`src/agenticreq/ast/extract_support.py`. When the code sets no bound, the framework's own default is shown as a
labelled fact (`data/framework_defaults.yml`), compared with spec.yml, and never offered as advice.

In `spec.yml` use the library's names. A framework name written there stops `check` with the closest library name
(for example `max_iterations`: did you mean `bounds.max_steps`?).
