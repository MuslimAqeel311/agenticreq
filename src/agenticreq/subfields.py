"""Every sub-field a person writes in spec.yml reaches a rule of its own, or a plain question for a person.

``SUBFIELDS`` says, for each sub-field of the spec's shapes, what checks it: the rule ids that read it, ``sign-off``
(its parent is a sign-off item, confirmed as a whole), or ``""`` (the code cannot show it, so a person answers a
question: ``QUESTIONS``). ``all_subfields`` walks the spec model, so a sub-field added to the model and missing here
gets a question, never a silent PASS. A sub-field the user wrote with no rule is its own line under Human sign-off
(AS-911): the question and where to confirm it. It is never folded into its parent's PASS.
"""

from __future__ import annotations

import types
import typing
from typing import Any, Union

from pydantic import BaseModel

#: sub-field -> what checks it ("AS-xxx ..." rules, "sign-off", or "" = no rule)
SUBFIELDS: dict[str, str] = {
    # Level 1
    "goal.statement": "AS-102",
    "goal.done_criteria": "AS-112",
    "role_boundary.allowed_tools": "AS-206 AS-217",
    "role_boundary.capabilities": "",
    "role_boundary.data_access": "",
    "prohibited[].tool": "AS-203 AS-204 AS-217",
    "prohibited[].action": "AS-204",
    "prohibited[].reason": "AS-109",
    "prohibited[].detector": "AS-204",
    "prohibited[].detector.type": "AS-204",
    "prohibited[].detector.params": "AS-204",
    "approval_required[].tool": "AS-205 AS-212",
    "approval_required[].approver": "AS-104",
    "approval_required[].timeout": "AS-205 AS-107",
    "approval_required[].on_timeout": "AS-110",
    "limits[].tool": "AS-217",
    "limits[].arg": "AS-207",
    "limits[].limit": "AS-207",
    "limits[].limit.min": "AS-207",
    "limits[].limit.max": "AS-207",
    "limits[].limit.max_length": "AS-207",
    "limits[].limit.enum": "AS-207",
    "limits[].limit.pattern": "AS-207",
    "escalation[].metric": "sign-off",
    "escalation[].op": "sign-off",
    "escalation[].threshold": "sign-off",
    "escalation[].to": "sign-off",
    "untrusted_inputs[].source": "AS-401 AS-402",
    "untrusted_inputs[].kind": "",
    "data_rules.retention": "sign-off AS-116",
    "data_rules.never_store": "sign-off AS-116",
    # Level 2
    "bounds.max_steps": "AS-208 AS-209",
    "bounds.max_cost_usd": "AS-221",
    "bounds.max_duration_s": "AS-222",
    "bounds.max_retries": "AS-209",
    "bounds.max_tool_calls": "AS-209",
    "progress_guard.max_no_progress_steps": "AS-228",
    "context_policy.must_never_drop": "sign-off",
    "context_policy.max_tokens": "sign-off",
    "memory_policy.enabled": "sign-off",
    "memory_policy.write_validation": "sign-off",
    "output_size_limits.max_chars": "AS-213",
    "output_size_limits.max_items": "",
    "write_safety.idempotency_key": "AS-211",
    "write_safety.compensating_action": "",
    "egress_policy.allow": "AS-216",
    "egress_policy.default": "AS-216",
    # Level 3
    "agent_registry[].id": "AS-308 AS-233",
    "agent_registry[].role": "",
    "agent_registry[].tools": "AS-206",
    "agent_registry[].prohibited": "AS-203",
    "agent_registry[].system_prompt_sha256": "AS-308",
    "agent_registry[].match": "AS-308",
    "declared_delegation[].from": "AS-301 AS-114",
    "declared_delegation[].to": "AS-301 AS-114",
    "aggregate_budget.max_cost_usd": "AS-306",
    "aggregate_budget.max_steps": "AS-306",
    "aggregate_budget.max_tool_calls": "AS-306",
}
#: fields whose keys are the user's own names (a tool, a field): checked as the field itself
_OPEN_MAPPINGS = frozenset({"rate_limits", "handoff_contract", "tool_effects", "data_rules.retention"})
_META = frozenset({"confirmed", "waivers", "_derived", "derived", "extends", "spec_version"})
#: the sub-fields the code cannot show: the question a person answers under confirmed: (the sub-field's own name)
QUESTIONS: dict[str, str] = {
    "role_boundary.capabilities": "Does the agent only do what is listed in role_boundary.capabilities?",
    "role_boundary.data_access": "Does the agent only read the data listed in role_boundary.data_access?",
    "untrusted_inputs[].kind": "Is every source under untrusted_inputs given the right kind?",
    "output_size_limits.max_items": "Does the agent never return more items than output_size_limits.max_items?",
    "write_safety.compensating_action": "Can every write the agent makes be undone by a compensating action?",
    "agent_registry[].role": "Does each agent in agent_registry do the role written for it?",
}


def question(path: str) -> str:
    """The plain question for a sub-field only a person can answer."""
    return QUESTIONS.get(path, f"Is {path} in place as written in spec.yml?")


def _models_in(ann: Any) -> list[tuple[type[BaseModel], bool]]:
    """(model, inside a list) for each pydantic model in an annotation."""
    origin = typing.get_origin(ann)
    if origin in (Union, types.UnionType):
        return [m for a in typing.get_args(ann) for m in _models_in(a)]
    if origin in (list, tuple):
        return [(m, True) for a in typing.get_args(ann) for m, _ in _models_in(a)]
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return [(ann, False)]
    return []


def all_subfields() -> list[str]:
    """Every sub-field of the spec's shapes (approval_required[].timeout, bounds.max_cost_usd ...), in order."""
    from agenticreq.models.spec import AgentSpec

    out: list[str] = []

    def walk(model: type[BaseModel], path: str) -> None:
        for name, f in model.model_fields.items():
            key = f.alias or name
            full = f"{path}.{key}"
            out.append(full)
            for m, in_list in _models_in(f.annotation):
                walk(m, full + ("[]" if in_list else ""))

    for name, f in AgentSpec.model_fields.items():
        key = f.alias or name
        if key in _META or name in _META:
            continue
        for m, in_list in _models_in(f.annotation):
            walk(m, key + ("[]" if in_list else ""))
    return list(dict.fromkeys(out))


def rule_of(path: str) -> str:
    """What checks ``path`` ("AS-207", "sign-off"), or "" when nothing does."""
    if path in SUBFIELDS:
        return SUBFIELDS[path]
    parent = path.rsplit(".", 1)[0] if "." in path else ""
    return SUBFIELDS.get(parent, "") if parent in _OPEN_MAPPINGS else ""


def sign_off_only() -> list[str]:
    """Sub-fields the code cannot show, so a person answers a question, for the docs and `check --coverage`."""
    return [p for p in all_subfields() if not rule_of(p)]


def written(spec: BaseModel) -> list[str]:
    """The sub-fields the user wrote (set in spec.yml, not defaults), as catalog paths."""
    out: list[str] = []

    def walk(v: Any, path: str) -> None:
        if isinstance(v, BaseModel):
            for name in v.model_fields_set:
                f = type(v).model_fields[name]
                key = f.alias or name
                full = f"{path}.{key}" if path else key
                if path:
                    out.append(full)
                if not path and (key in _META or name in _META or key in _OPEN_MAPPINGS):
                    continue
                walk(getattr(v, name), full)
        elif isinstance(v, list):
            for x in v:
                walk(x, path + "[]")

    walk(spec, "")
    order = {p: i for i, p in enumerate(all_subfields())}
    return sorted(dict.fromkeys(out), key=lambda p: (order.get(p, len(order)), p))  # never a set's order
