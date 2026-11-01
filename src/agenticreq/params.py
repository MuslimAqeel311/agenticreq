"""The parameter catalog: every spec parameter, its level, and how agenticreq verifies it.

This is the single source of truth for "what does this library check, and with what". A test asserts
that every field of ``AgentSpec`` appears here, that every rule id named here is registered, and that
every parameter verified by attestation has a reason in ``rules/attest.py`` — so no parameter can be
silently ignored.

Levels
  1  user-written business intent — never auto-filled; `check` fails on TODO
  2  engineering defaults by risk_level — the library fills them; the user may override
  3  multi-agent / orchestrator — topology + defaults
  4  derived — computed by the library, never written
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

How = Literal["code", "spec", "sign-off", "derived", "runtime"]


@dataclass(frozen=True)
class Param:
    name: str
    level: int
    group: str
    how: How  # code: a rule reads the code/wire · spec: validated in the spec · attestation: a human vouches
    #           runtime: real runs (traces / real-model captures) prove violations; attestation without them
    rules: tuple[str, ...] = ()
    note: str = ""
    #: True when `agenticreq audit` (no spec) can check it from code, with the risk_level default
    audit: bool = False
    #: a few other names people use for it (3 to 5 at most), so a wrong name in spec.yml gets a suggestion:
    #: ("", words) for the field itself, ("max_steps", words) for a field inside it (bounds.max_steps)
    aka: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def also_called(self) -> list[tuple[str, tuple[str, ...]]]:
        """(full field name, its other names)."""
        return [(f"{self.name}.{sub}" if sub else self.name, words) for sub, words in self.aka]


def _p(
    name: str,
    level: int,
    group: str,
    how: How,
    rules: str = "",
    note: str = "",
    audit: bool = False,
    aka: str | dict[str, str] = "",
) -> Param:
    """``aka``: "forbidden banned blocked" for the field itself, or {"": ..., "max_steps": "max_iterations ..."}."""
    pairs = {"": aka} if isinstance(aka, str) else aka
    return Param(
        name,
        level,
        group,
        how,
        tuple(r for r in rules.split() if r),
        note,
        audit,
        tuple((sub, tuple(words.split())) for sub, words in pairs.items() if words),
    )


PARAMS: tuple[Param, ...] = (
    # ------------------------------------------------------------------ Level 1 (user writes)
    _p("agent_id", 1, "Identity", "spec", "AS-101 AS-102", aka="name agent_name id"),
    _p(
        "risk_level",
        1,
        "Identity",
        "spec",
        "AS-102 AS-107 AS-111",
        "selects every Level-2/3 default and floor",
        aka="risk risk_tier criticality",
    ),
    _p(
        "goal",
        1,
        "Intent",
        "spec",
        "AS-102 AS-112 AS-905",
        "done_criteria must be measurable",
        aka="objective purpose mission",
    ),
    _p("out_of_scope", 1, "Intent", "sign-off", "AS-102 AS-501", aka="off_topic excluded_topics not_allowed_topics"),
    _p("role_boundary", 1, "Authority", "code", "AS-206 AS-103", aka="permitted_tools tool_allowlist tool_permissions"),
    _p(
        "prohibited",
        1,
        "Authority",
        "code",
        "AS-203 AS-204 AS-304 AS-109",
        "presence of a tool in code is not permission; only the spec says what is forbidden",
        aka="forbidden banned blocked denied",
    ),
    _p(
        "approval_required",
        1,
        "Authority",
        "code",
        "AS-205 AS-104 AS-110",
        aka="human_approval needs_approval requires_approval hitl",
    ),
    _p(
        "limits",
        1,
        "Authority",
        "code",
        "AS-207 AS-105 AS-217",
        "the money / quantity thresholds",
        aka="thresholds caps max_values",
    ),
    _p(
        "escalation",
        1,
        "Oversight",
        "sign-off",
        "AS-108 AS-501",
        "must be {metric, op, threshold, to}",
        aka="escalate escalation_rules handoff_to_human",
    ),
    _p(
        "untrusted_inputs",
        1,
        "Trust",
        "code",
        "AS-401 AS-402 AS-403",
        aka="untrusted_sources tainted_inputs external_inputs",
    ),
    _p("owner", 1, "Accountability", "sign-off", "AS-102 AS-501", aka="business_owner product_owner accountable"),
    _p(
        "technical_owner",
        1,
        "Accountability",
        "spec",
        "AS-113",
        "written together with owner",
        aka="tech_owner engineering_owner maintainer",
    ),
    _p("data_rules", 1, "Trust", "sign-off", "AS-116 AS-501"),
    # ------------------------------------------------------------------ Level 2 (library defaults)
    # Goal & Termination
    _p("stop_conditions", 2, "Goal & Termination", "runtime", "AS-230 AS-501", "needs traces / real captures"),
    _p("progress_guard", 2, "Goal & Termination", "runtime", "AS-228 AS-501", "needs traces / real captures"),
    _p("completion_verification", 2, "Goal & Termination", "sign-off", "AS-501"),
    _p("termination_authority", 2, "Goal & Termination", "sign-off", "AS-501"),
    _p(
        "bounds",
        2,
        "Goal & Termination",
        "code",
        "AS-208 AS-209 AS-221 AS-222 AS-229",
        "max_steps, max_cost_usd, max_duration_s",
        audit=True,
        aka={
            "": "budgets run_limits",
            "max_steps": "max_iterations max_turns recursion_limit step_limit",
            "max_cost_usd": "max_cost cost_limit budget_usd",
            "max_duration_s": "timeout max_duration time_limit",
            "max_tool_calls": "tool_call_limit max_tools",
            "max_retries": "retries retry_limit",
        },
    ),
    _p("goal_drift_guard", 2, "Goal & Termination", "sign-off", "AS-501"),
    # Tool Discipline
    _p("fail_closed", 2, "Tool Discipline", "code", "AS-219", audit=True),
    _p(
        "tool_schema_strictness",
        2,
        "Tool Discipline",
        "code",
        "AS-601 AS-602 AS-603 AS-604 AS-605 AS-606 AS-607 AS-609 AS-610",
        "needs a mock run",
    ),
    _p("tool_result_validation", 2, "Tool Discipline", "sign-off", "AS-501"),
    _p("tool_error_contract", 2, "Tool Discipline", "sign-off", "AS-501"),
    _p("read_write_separation", 2, "Tool Discipline", "sign-off", "AS-501"),
    _p(
        "tool_inventory_report",
        2,
        "Tool Discipline",
        "code",
        "AS-201 AS-202 AS-234",
        "declared vs observed inventory; also `agenticreq stats`",
        audit=True,
    ),
    _p(
        "unbounded_tool_policy",
        2,
        "Tool Discipline",
        "code",
        "AS-213 AS-215 AS-216",
        "http / sql / shell / search tools",
        audit=True,
    ),
    _p("dry_run_required", 2, "Tool Discipline", "code", "AS-227"),
    # State & Context
    _p("context_policy", 2, "State & Context", "sign-off", "AS-501"),
    _p("memory_policy", 2, "State & Context", "sign-off", "AS-501"),
    _p(
        "checkpointing",
        2,
        "State & Context",
        "code",
        "AS-226",
        "verified for LangGraph; other frameworks wait for a mock run",
    ),
    _p("structured_output", 2, "State & Context", "code", "AS-224", audit=True),
    # Reliability & Recovery
    _p("write_safety", 2, "Reliability & Recovery", "code", "AS-211 AS-212", "idempotency key on writes", audit=True),
    _p("fallback_policy", 2, "Reliability & Recovery", "sign-off", "AS-501"),
    _p(
        "rate_limits",
        2,
        "Reliability & Recovery",
        "runtime",
        "AS-231 AS-501",
        "needs timestamps (a mock run / OTLP spans) and a requests_per_minute key",
    ),
    _p("output_size_limits", 2, "Reliability & Recovery", "code", "AS-213", audit=True),
    # Security & Trust
    _p("trust_boundary", 2, "Security & Trust", "code", "AS-401 AS-402", audit=True),
    _p("identity_propagation", 2, "Security & Trust", "sign-off", "AS-501"),
    _p("egress_policy", 2, "Security & Trust", "code", "AS-216", audit=True),
    _p("secrets_policy", 2, "Security & Trust", "code", "AS-214 AS-501", "hard-coded secrets", audit=True),
    _p("self_modification", 2, "Security & Trust", "code", "AS-220", audit=True),
    _p("sandbox_policy", 2, "Security & Trust", "code", "AS-215", audit=True),
    # Human Oversight
    _p("clarification_policy", 2, "Human Oversight", "sign-off", "AS-501"),
    _p("confidence_threshold", 2, "Human Oversight", "sign-off", "AS-501"),
    _p("on_approval_timeout", 2, "Human Oversight", "spec", "AS-110 AS-111"),
    _p("kill_switch", 2, "Human Oversight", "sign-off", "AS-501", aka="emergency_stop off_switch stop_switch"),
    _p("handoff_contract", 2, "Human Oversight", "sign-off", "AS-501"),
    # Observability & Change Safety
    _p("trace_policy", 2, "Observability & Change Safety", "code", "AS-225", audit=True),
    _p(
        "audit_log",
        2,
        "Observability & Change Safety",
        "code",
        "AS-225 AS-501",
        "audit logging hooks",
        audit=True,
        aka="audit_trail audit_logging tool_call_log",
    ),
    _p("log_immutable", 2, "Observability & Change Safety", "sign-off", "AS-501"),
    _p(
        "version_pinning",
        2,
        "Observability & Change Safety",
        "code",
        "AS-210",
        "model version pinned",
        audit=True,
        aka="model_pinning pin_model pinned_model",
    ),
    _p("regression_triggers", 2, "Observability & Change Safety", "sign-off", "AS-501"),
    _p("rollback_path", 2, "Observability & Change Safety", "sign-off", "AS-501"),
    _p("drift_monitors", 2, "Observability & Change Safety", "sign-off", "AS-501"),
    _p("environment_separation", 2, "Observability & Change Safety", "code", "AS-223 AS-501", audit=True),
    # ------------------------------------------------------------------ Level 3 (multi-agent)
    _p("topology", 3, "Multi-Agent", "spec", "", "single | pipeline | supervisor | hierarchical | swarm | peer"),
    _p("agent_registry", 3, "Multi-Agent", "code", "AS-308 AS-233 AS-114"),
    _p("declared_delegation", 3, "Multi-Agent", "code", "AS-301 AS-114"),
    _p("max_delegation_depth", 3, "Multi-Agent", "code", "AS-302"),
    _p("authority_inheritance", 3, "Multi-Agent", "code", "AS-305", "a delegate holds a subset of its delegator"),
    _p("transitive_prohibition", 3, "Multi-Agent", "code", "AS-304", "A → B → C cannot reach a prohibited tool"),
    _p("no_cycles", 3, "Multi-Agent", "code", "AS-303"),
    _p(
        "cross_agent_trust",
        3,
        "Multi-Agent",
        "code",
        "AS-401 AS-501",
        "`none`: a sibling agent's output is untrusted input",
    ),
    _p("context_isolation", 3, "Multi-Agent", "sign-off", "AS-501"),
    _p("conflict_resolution", 3, "Multi-Agent", "sign-off", "AS-501"),
    _p("aggregate_budget", 3, "Multi-Agent", "code", "AS-306"),
    _p("concurrency_policy", 3, "Multi-Agent", "sign-off", "AS-501"),
    _p("failure_isolation", 3, "Multi-Agent", "sign-off", "AS-501"),
    _p("end_to_end_trace_id", 3, "Multi-Agent", "code", "AS-307"),
    # ------------------------------------------------------------------ Level 4 (derived)
    _p("trace_coverage", 4, "Derived", "derived", "AS-905", "scenario matrix built from the spec vs traces"),
    _p(
        "enforceability",
        4,
        "Derived",
        "derived",
        "",
        "declared clauses that have an enforcement point in code / OPA / Cedar",
    ),
)

BY_NAME: dict[str, Param] = {p.name: p for p in PARAMS}

#: Level-1 and Level-3 parameters only a human can state. `audit` (no spec) never checks them and says so.
NEEDS_SPEC: tuple[str, ...] = (
    "prohibited",
    "limits",
    "approval_required",
    "escalation",
    "untrusted_inputs",
    "declared_delegation",
    "role_boundary",
)


def level(n: int) -> list[Param]:
    return [p for p in PARAMS if p.level == n]


def audit_checkable() -> list[Param]:
    return [p for p in PARAMS if p.audit]


#: checked by `audit` from the code alone, whatever (or whether any) framework is used
FRAMEWORK_AGNOSTIC: tuple[str, ...] = (
    "secrets_policy",
    "version_pinning",
    "environment_separation",
    "unbounded_tool_policy",
    "sandbox_policy",
    "egress_policy",
)
#: need a recognised framework: the constructs that enforce them are framework idioms
FRAMEWORK_SPECIFIC: tuple[str, ...] = (
    "bounds",
    "approval gates",
    "write_safety",
    "fail_closed",
    "structured_output",
    "audit_log / trace_policy",
    "checkpointing",
    "self_modification",
    "tool_inventory_report",
    "taint (untrusted -> write)",
)


def audit_scope(framework: str | None) -> dict[str, object]:
    """What `audit` (no spec) could and could not check: printed in the report header, never implied."""
    audited = [p.name for p in PARAMS if p.audit]
    if framework:
        checked = audited
        not_checked: list[str] = []
        note = None
    else:
        checked = [n for n in audited if n in FRAMEWORK_AGNOSTIC]
        not_checked = list(FRAMEWORK_SPECIFIC)
        note = (
            "no agent framework recognised: framework-specific checks did not run (they wait for a mock run, see AS-904); "
            "framework-agnostic ones did"
        )
    return {
        "framework": framework or "not recognised",
        "checked": checked,
        "not_checked": not_checked + ["sign-off items (kill switch, log immutability and the rest a person confirms)"],
        "needs_spec": list(NEEDS_SPEC),
        "framework_note": note,
    }
