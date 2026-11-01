"""Where a mis-placed `name: number` entry belongs, decided from its key name only.

A lookup table, not interpretation: a key that matches no row is reported as "could not determine",
never guessed. Rows are tried in order; the first match wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Route:
    pattern: str  # regex over the key name (full match, case-insensitive)
    field: str  # destination, as `agenticreq params` names it
    write: str  # what to write, with {v} for the value
    note: str = ""


#: key name -> destination. Shown by `agenticreq params`-style messages and tested row by row.
ROUTES: tuple[Route, ...] = (
    Route(
        r"max_?iter(ation)?s?|max_steps|max_turns|recursion_limit|max_loops",
        "bounds.max_steps",
        "bounds: {{max_steps: {v}}}",
    ),
    Route(r"max_tool_calls(_per_\w+)?", "bounds.max_tool_calls", "bounds: {{max_tool_calls: {v}}}"),
    Route(
        r"max_cost(_usd)?|cost_limit|budget(_usd)?|max_budget(_usd)?",
        "bounds.max_cost_usd",
        "bounds: {{max_cost_usd: {v}}}",
    ),
    Route(
        r"max_duration(_s)?|timeout(_s)?|max_time|time_limit|max_execution_time",
        "bounds.max_duration_s",
        "bounds: {{max_duration_s: {v}}}",
        "in seconds",
    ),
    Route(r"max_retries|retries", "bounds.max_retries", "bounds: {{max_retries: {v}}}"),
    Route(
        r"\w*rate_?limit\w*|\w+_per_(second|minute|sec|min)|\w*_?(rpm|rps)",
        "rate_limits",
        "rate_limits: {{requests_per_minute: {rpm}}}",
        "rate_limits counts requests per minute",
    ),
    Route(
        r"temperature|top_p|top_k|seed|frequency_penalty|presence_penalty",
        "(no spec field)",
        "keep it in the agent's model config",
        "a model sampling / determinism setting: not a bound, not a rate limit, and agenticreq has no field for it",
    ),
    Route(
        r"max_tokens|max_context_tokens|context_window",
        "context_policy.max_tokens",
        "context_policy: {{max_tokens: {v}}}",
    ),
    Route(r"max_(items|results)|top_n", "output_size_limits.max_items", "output_size_limits: {{max_items: {v}}}"),
    Route(
        r"max_output_chars|max_response_chars", "output_size_limits.max_chars", "output_size_limits: {{max_chars: {v}}}"
    ),
    Route(
        r"\w+_(chars|length|len)",
        "limits (a tool argument)",
        "limits: [{{tool: <which tool>, arg: <which argument>, limit: {{max_length: {v}}}}}]",
        "name the tool and the argument it bounds",
    ),
)

UNKNOWN = "could not tell from the name - see `agenticreq params`"


def route(key: str) -> Route | None:
    for r in ROUTES:
        if re.fullmatch(r.pattern, key.strip(), flags=re.IGNORECASE):
            return r
    return None


def describe(key: str, value: Any) -> str:
    """`max_iterations: 8 -> bounds.max_steps: write bounds: {max_steps: 8}`"""
    r = route(key)
    if r is None:
        return f"{key}: {value}  ->  {UNKNOWN}"
    what = r.write.format(v=value, rpm=_per_minute(key, value))
    return f"{key}: {value}  ->  {r.field}: {what}" + (f"  ({r.note})" if r.note else "")


def _per_minute(key: str, value: Any) -> str:
    """10 per second -> 600 per minute; a per-minute figure stays; anything else is left for the user."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "<requests per minute>"
    k = key.lower()
    if re.search(r"(per_(second|sec)|rps)$", k):
        return f"{value * 60:g}"
    if re.search(r"(per_(minute|min)|rpm)$", k):
        return f"{value:g}"
    return "<requests per minute>"
