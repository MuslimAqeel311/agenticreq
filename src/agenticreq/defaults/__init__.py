"""risk_level → Tier 2 defaults, plus the yes/no *floor* an override may not go below.

This module holds yes/no and enum settings only. It holds NO numbers: every number in a spec (steps, cost, duration,
retries, tool calls, delegation depth, output size, confidence, rate limits) comes from the user, and a number fails
only when it is missing or when code and spec.yml disagree (agenticreq/thresholds.py). Floors are kept for safeguards only
(self_modification, fail_closed, on_approval_timeout, kill_switch, log_immutable, version_pinning, audit_log,
dry_run_required, transitive_prohibition)."""

from __future__ import annotations

import copy
from typing import Any, Callable

_BASE: dict[str, Any] = {
    "fail_closed": True,
    "stop_conditions": ["goal_done", "budget_exhausted", "unrecoverable_error"],
    "completion_verification": True,
    "termination_authority": "agent",
    "goal_drift_guard": False,
    "tool_schema_strictness": "strict",
    "tool_result_validation": True,
    "tool_error_contract": "structured",
    "read_write_separation": False,
    "tool_inventory_report": True,
    "unbounded_tool_policy": "warn",
    "dry_run_required": False,
    "context_policy": {"must_never_drop": ["goal", "constraints", "decisions"]},
    "memory_policy": {"enabled": False, "write_validation": True},
    "checkpointing": False,
    "structured_output": False,
    "write_safety": {"idempotency_key": True, "compensating_action": False},
    "fallback_policy": "safe_stop",
    "rate_limits": {},
    "trust_boundary": "tool_outputs_untrusted",
    "identity_propagation": False,
    "egress_policy": {"allow": [], "default": "deny"},
    "secrets_policy": "env_or_vault_only",
    "self_modification": False,
    "sandbox_policy": "required_for_code_exec",
    "clarification_policy": "ask_when_ambiguous",
    "on_approval_timeout": "safe_stop",
    "kill_switch": True,
    "handoff_contract": "typed",
    "trace_policy": "all_tool_calls",
    "audit_log": True,
    "log_immutable": False,
    "version_pinning": True,
    "regression_triggers": ["model_change", "prompt_change", "tool_schema_change"],
    "rollback_path": None,
    "drift_monitors": [],
    "environment_separation": True,
    # Tier 3 defaults (only consulted for multi-agent systems)
    "authority_inheritance": "subset",
    "transitive_prohibition": True,
    "no_cycles": True,
    "cross_agent_trust": "verified",
    "context_isolation": True,
    "end_to_end_trace_id": True,
    "failure_isolation": True,
}

_OVERRIDES: dict[str, dict[str, Any]] = {
    "low": {
        "fail_closed": False,
        "completion_verification": False,
        "write_safety": {"idempotency_key": False, "compensating_action": False},
        "unbounded_tool_policy": "allow",
    },
    "medium": {},
    "high": {
        "goal_drift_guard": True,
        "read_write_separation": True,
        "unbounded_tool_policy": "forbid",
        "dry_run_required": True,
        "checkpointing": True,
        "structured_output": True,
        "identity_propagation": True,
        "write_safety": {"idempotency_key": True, "compensating_action": True},
        "termination_authority": "human_or_policy",
        "log_immutable": True,
    },
    "critical": {
        "goal_drift_guard": True,
        "read_write_separation": True,
        "unbounded_tool_policy": "forbid",
        "dry_run_required": True,
        "checkpointing": True,
        "structured_output": True,
        "identity_propagation": True,
        "write_safety": {"idempotency_key": True, "compensating_action": True},
        "termination_authority": "human",
        "log_immutable": True,
        "cross_agent_trust": "none",
    },
}

RISK_LEVELS = ("low", "medium", "high", "critical")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def tier2_defaults(risk_level: str | None) -> dict[str, Any]:
    level = risk_level if risk_level in RISK_LEVELS else "medium"
    return deep_merge(_BASE, _OVERRIDES[level])


# ---------------------------------------------------------------- floors
# A floor (safeguards only, never a number): (dotted key, predicate(user_value, default_value) -> True if weaker)
_Weaker = Callable[[Any, Any], bool]


def _bool_true_required(v: Any, _d: Any) -> bool:
    return v is False


def _bool_false_required(v: Any, _d: Any) -> bool:
    return v is True


def _unsafe_timeout(v: Any, _d: Any) -> bool:
    """safe_stop and deny both keep the action from running without approval; anything else is weaker."""
    return v is not None and v not in ("safe_stop", "deny")


_FLOORS: dict[str, list[tuple[str, _Weaker]]] = {
    "low": [("self_modification", _bool_false_required)],
    "medium": [
        ("self_modification", _bool_false_required),
        ("on_approval_timeout", _unsafe_timeout),
    ],
    "high": [
        ("self_modification", _bool_false_required),
        ("fail_closed", _bool_true_required),
        ("on_approval_timeout", _unsafe_timeout),
        ("kill_switch", _bool_true_required),
        ("log_immutable", _bool_true_required),
        ("version_pinning", _bool_true_required),
    ],
    "critical": [
        ("self_modification", _bool_false_required),
        ("fail_closed", _bool_true_required),
        ("on_approval_timeout", _unsafe_timeout),
        ("kill_switch", _bool_true_required),
        ("log_immutable", _bool_true_required),
        ("version_pinning", _bool_true_required),
        ("audit_log", _bool_true_required),
        ("dry_run_required", _bool_true_required),
        ("transitive_prohibition", _bool_true_required),
    ],
}


def _dig(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def floor_violations(risk_level: str | None, user_values: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Return (dotted_key, user_value, floor_value) for overrides weaker than the floor."""
    if risk_level not in RISK_LEVELS:
        return []
    defaults = tier2_defaults(risk_level)
    out: list[tuple[str, Any, Any]] = []
    for key, weaker in _FLOORS[risk_level]:
        v = _dig(user_values, key)
        if v is None:
            continue
        d = _dig(defaults, key)
        if weaker(v, d):
            out.append((key, v, d))
    return out


def effective_tier2(risk_level: str | None, user_values: dict[str, Any]) -> dict[str, Any]:
    """Defaults for the risk level, overridden by whatever the user wrote (non-None)."""
    overrides = {k: v for k, v in user_values.items() if v is not None}
    return deep_merge(tier2_defaults(risk_level), overrides)
