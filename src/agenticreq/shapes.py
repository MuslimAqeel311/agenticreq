"""The exact shape of every spec field, and a filled example of it.

One source of truth for: schema-error messages (every error shows what to write), `agenticreq params`,
`agenticreq validate --fix` and the TODO comments `init` writes. A test asserts every ``AgentSpec`` field
has an example and that each example validates against the model.
"""

from __future__ import annotations

import re
import types
import typing
from typing import Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel

from agenticreq.models.spec import AgentSpec

#: filled examples; Level-2 scalars not listed here use the risk_level=high default (see ``example``)
EXAMPLES: dict[str, Any] = {
    # ---- meta
    "spec_version": 1,
    "extends": "profile:baseline",
    "pinned_models": ["azure/my-prod-deployment"],
    "_derived": {"framework": "langgraph 1.2.15", "declared_tools": ["lookup_order"]},
    "confirmed": [
        {
            "clause": "kill_switch",
            "by": "dev@company.com",
            "date": "2026-10-17",
            "expires": "2027-04-17",
            "evidence": "agent.py:42 checks AGENT_KILL_SWITCH every step",
        }
    ],
    "tool_effects": {"read_file": "read", "save_report": "write"},
    "waivers": [
        {
            "rule_id": "AS-213",
            "subject": "search_kb",
            "reason": "results are capped by the KB API",
            "approved_by": "lead@company.com",
            "expires": "2027-01-01",
        }
    ],
    # ---- Level 1
    "agent_id": "support-agent",
    "risk_level": "high",
    "goal": {
        "statement": "Answer order questions; refund only within policy",
        "done_criteria": ["customer answered", "or ticket escalated to a human"],
    },
    "out_of_scope": ["closing accounts", "changing prices"],
    "role_boundary": {
        "allowed_tools": ["lookup_order", "search_kb", "issue_refund"],
        "capabilities": ["read:orders", "write:refunds"],
        "data_access": ["orders", "kb"],
    },
    "prohibited": [
        {"tool": "delete_customer", "reason": "support never deletes data"},
        {"action": "send_email_external", "reason": "no outbound email"},
    ],
    "approval_required": [
        {"tool": "issue_refund", "approver": "support-lead", "timeout": "1h", "on_timeout": "safe_stop"}
    ],
    "limits": [{"tool": "issue_refund", "arg": "amount", "limit": {"min": 0, "max": 100}}],
    "escalation": [{"metric": "refund_amount", "op": ">", "threshold": 100, "to": "support-lead"}],
    "untrusted_inputs": [
        {"source": "customer_message", "kind": "user_content"},
        {"source": "search_kb", "kind": "retrieved_document"},
    ],
    "owner": "owner@company.com",
    "technical_owner": "dev@company.com",
    "data_rules": {"never_store": ["card_number", "password"], "retention": {"chat_transcript": "30d"}},
    # ---- Level 2 (structured ones; scalars fall back to the high-risk default)
    "bounds": {"max_steps": 25, "max_cost_usd": 2.0, "max_duration_s": 300, "max_retries": 2, "max_tool_calls": 25},
    "stop_conditions": ["goal_done", "budget_exhausted", "unrecoverable_error"],
    "progress_guard": {"max_no_progress_steps": 5},
    "context_policy": {"must_never_drop": ["goal", "constraints", "decisions"], "max_tokens": 32000},
    "memory_policy": {"enabled": False, "write_validation": True},
    "output_size_limits": {"max_chars": 20000, "max_items": 20},
    "write_safety": {"idempotency_key": True, "compensating_action": True},
    "rate_limits": {"requests_per_minute": 30},
    "egress_policy": {"default": "deny", "allow": ["api.openai.com"]},
    "dry_run_required": True,
    "regression_triggers": ["model_change", "prompt_change", "tool_schema_change"],
    "rollback_path": "redeploy the previous tagged release",
    "drift_monitors": ["tool_schema_hash", "model_version"],
    "handoff_contract": "typed",
    # ---- Level 3
    "topology": "supervisor",
    "agent_registry": [
        {"id": "supervisor", "tools": []},
        {"id": "billing", "tools": ["issue_refund"], "prohibited": ["delete_customer"]},
    ],
    "declared_delegation": [{"from": "supervisor", "to": "billing"}],
    "max_delegation_depth": 2,
    "authority_inheritance": "subset",
    "transitive_prohibition": True,
    "no_cycles": True,
    "cross_agent_trust": "verified",
    "context_isolation": True,
    "conflict_resolution": "supervisor decides; ties go to a human",
    "aggregate_budget": {"max_cost_usd": 5.0, "max_steps": 60, "max_tool_calls": 60},
    "concurrency_policy": "one write per order at a time",
    "failure_isolation": True,
    "end_to_end_trace_id": True,
}


def spec_fields() -> list[str]:
    """Every key a spec may contain, by its YAML name."""
    return [f.alias or name for name, f in AgentSpec.model_fields.items()]


def _field(key: str) -> Any:
    for name, f in AgentSpec.model_fields.items():
        if key in (name, f.alias):
            return f
    raise KeyError(key)


NUMBER = "<number>"  # what an example shows in place of a number: every number in a spec comes from the user
DURATION = "<duration>"  # ... and in place of a length of time such as 30m or 4h
_DURATION_TEXT = re.compile(r"^\d+\s*[smhdw]$", re.IGNORECASE)


def mask_numbers(v: Any, key: str = "") -> Any:
    """``v`` with every number replaced by ``<number>`` and every duration by ``<duration>``. Examples show the shape
    of a field, never a value the library picked: the numbers are the user's to write."""
    if isinstance(v, bool) or key == "spec_version":
        return v
    if isinstance(v, (int, float)):
        return NUMBER
    if isinstance(v, str) and _DURATION_TEXT.match(v):
        return DURATION
    if isinstance(v, dict):
        return {k: mask_numbers(x, str(k)) for k, x in v.items()}
    if isinstance(v, list):
        return [mask_numbers(x) for x in v]
    return v


def example(key: str) -> Any:
    """A filled example of the field, with <number> where a number goes."""
    if key in EXAMPLES:
        return mask_numbers(EXAMPLES[key], key)
    from agenticreq.defaults import tier2_defaults

    d = tier2_defaults("high")
    if key in d and d[key] is not None:
        return mask_numbers(d[key], key)
    ann = _field(key).annotation
    return mask_numbers(_scalar_example(ann), key)


def _scalar_example(ann: Any) -> Any:
    for a in _non_none(ann):
        if get_origin(a) is Literal:
            return get_args(a)[0]
        if a is bool:
            return True
        if a is int:
            return 1
        if a is float:
            return 0.8
        if a is str:
            return "TODO"
    return "TODO"


def _non_none(ann: Any) -> list[Any]:
    if get_origin(ann) in (Union, types.UnionType):
        return [a for a in get_args(ann) if a is not type(None)]
    return [ann]


def shape_of(ann: Any, depth: int = 0) -> str:
    """A compact, exact description of what a value must look like."""
    alts = _non_none(ann)
    if len(alts) > 1:
        return " | ".join(shape_of(a, depth) for a in alts)
    a = alts[0]
    origin = get_origin(a)
    if origin is Literal:
        return " | ".join(str(x) for x in get_args(a))
    if origin in (list, typing.List):  # noqa: UP006
        (inner,) = get_args(a) or (Any,)
        return f"[{shape_of(inner, depth + 1)}, ...]"
    if origin in (dict, typing.Dict):  # noqa: UP006
        k, v = get_args(a) or (str, Any)
        return f"{{<{shape_of(k, depth + 1)}>: {shape_of(v, depth + 1)}}}"
    if isinstance(a, type) and issubclass(a, BaseModel):
        if depth > 2:
            return "{...}"
        parts = []
        for name, f in a.model_fields.items():
            key = f.alias or name
            mark = "" if f.is_required() else "?"
            parts.append(f"{key}{mark}: {shape_of(f.annotation, depth + 1)}")
        return "{" + ", ".join(parts) + "}"
    if a is Any:
        return "any"
    return {str: "string", int: "integer", float: "number", bool: "true | false"}.get(a, getattr(a, "__name__", str(a)))


def shape(key: str) -> str:
    return shape_of(_field(key).annotation)


def example_yaml(key: str) -> str:
    """The example as YAML, flow style, ready to paste: ``limits: [{tool: pay, arg: amount, ...}]``."""
    value = example(key)
    if not isinstance(value, (dict, list)):
        scalar = yaml.safe_dump(value, allow_unicode=True).strip()
        return f"{key}: {scalar.removesuffix(chr(10) + '...').strip()}"
    text = yaml.safe_dump({key: value}, default_flow_style=None, sort_keys=False, width=100, allow_unicode=True)
    return text.strip()
