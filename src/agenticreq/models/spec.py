"""AgentSpec — what the humans declare.

Tier 1 (12 fields) is written by the user and never auto-filled.
Tier 2 is optional; when omitted, values come from ``defaults/`` keyed by risk_level.
Tier 3 covers multi-agent systems.
Tier 4 (trace_coverage, enforceability) is derived, never written.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RiskLevel = Literal["low", "medium", "high", "critical"]
TimeoutAction = Literal["safe_stop", "deny", "escalate", "proceed"]
CompareOp = Literal[">", ">=", "<", "<=", "==", "!="]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------- Tier 1 parts
class Goal(_Strict):
    statement: str
    done_criteria: list[str] = Field(default_factory=list)


class RoleBoundary(_Strict):
    """Structured capability set — not prose."""

    allowed_tools: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)  # e.g. "read:inventory", "write:orders"
    data_access: list[str] = Field(default_factory=list)


class DetectorSpec(_Strict):
    type: str  # registered Detector name, e.g. "tool_arg_regex"
    params: dict[str, Any] = Field(default_factory=dict)


class Prohibition(_Strict):
    tool: str | None = None
    action: str | None = None
    reason: str | None = None
    detector: DetectorSpec | None = None


class ApprovalRule(_Strict):
    tool: str
    approver: str | None = None
    timeout: str | None = None  # "4h", "30m"
    on_timeout: TimeoutAction | None = None


class LimitSpec(_Strict):
    min: float | None = None
    max: float | None = None
    max_length: int | None = None
    enum: list[Any] | None = None
    pattern: str | None = None


class LimitEntry(_Strict):
    """Normalised limit: one tool argument and its bound."""

    tool: str
    arg: str
    limit: LimitSpec


class EscalationRule(_Strict):
    metric: str
    op: CompareOp
    threshold: float | int | str
    to: str


class UntrustedInput(_Strict):
    source: str  # tool name or channel ("email", "web")
    kind: str | None = None


class DataRules(_Strict):
    retention: dict[str, str] = Field(default_factory=dict)  # field -> "30d"
    never_store: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- Tier 2 parts
class Bounds(_Strict):
    # numbers are the user's choice at any risk level; only a value that cannot be a limit is a shape error
    max_steps: int | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_duration_s: float | None = Field(default=None, gt=0)
    max_retries: int | None = Field(default=None, gt=0)
    max_tool_calls: int | None = Field(default=None, gt=0)


class ProgressGuard(_Strict):
    max_no_progress_steps: int | None = Field(default=None, gt=0)


class ContextPolicy(_Strict):
    must_never_drop: list[str] = Field(default_factory=list)
    max_tokens: int | None = Field(default=None, gt=0)


class MemoryPolicy(_Strict):
    enabled: bool = False
    write_validation: bool = True


class OutputSizeLimits(_Strict):
    max_chars: int | None = Field(default=None, gt=0)
    max_items: int | None = Field(default=None, gt=0)


class WriteSafety(_Strict):
    idempotency_key: bool = False
    compensating_action: bool = False


class EgressPolicy(_Strict):
    allow: list[str] = Field(default_factory=list)
    default: Literal["deny", "allow"] = "deny"


# ---------------------------------------------------------------- Tier 3 parts
class AgentEntry(_Strict):
    id: str
    role: str | None = None
    tools: list[str] = Field(default_factory=list)
    prohibited: list[str] = Field(default_factory=list)
    system_prompt_sha256: str | None = None
    match: str | None = None  # substring identifying this agent's system prompt / name


class DelegationEdge(_Strict):
    from_: str = Field(alias="from")
    to: str


class AggregateBudget(_Strict):
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    max_tool_calls: int | None = Field(default=None, gt=0)


# ---------------------------------------------------------------- meta parts
SignOffAnswer = Literal["yes", "no", "not_needed"]
#: safety items that cannot be answered not_needed at high or critical risk
NOT_NEEDED_FORBIDDEN: tuple[str, ...] = (
    "kill_switch",
    "fail_closed",
    "self_modification",
    "log_immutable",
    "version_pinning",
    "audit_log",
    "dry_run_required",
    "transitive_prohibition",
)


class Attestation(_Strict):
    """One entry under ``confirmed:``: a person's answer for a sign-off item. Which fields it needs depends on the
    answer (``signoff.assess`` reports a missing one): yes = by, date, expires, evidence; no and not_needed = by,
    date, reason."""

    clause: str
    answer: SignOffAnswer = "yes"
    by: str | None = None
    date: dt.date | None = None
    expires: dt.date | None = None
    evidence: str | None = None
    reason: str | None = None

    @field_validator("answer", mode="before")
    @classmethod
    def _yaml_bool(cls, v: Any) -> Any:
        """YAML reads a bare yes / no as true / false."""
        if v is True:
            return "yes"
        if v is False:
            return "no"
        return "yes" if v is None else v


class Waiver(_Strict):
    rule_id: str
    subject: str | None = None
    reason: str | None = None
    evidence: str | None = None
    approved_by: str | None = None
    expires: dt.date | None = None

    @model_validator(mode="after")
    def _says_why(self) -> Waiver:
        if not (self.reason or self.evidence):
            raise ValueError("a waiver needs reason: or evidence:")
        return self

    @property
    def why(self) -> str:
        both = [x for x in (self.reason, self.evidence) if x]
        return " (evidence: ".join(both) + ")" if len(both) == 2 else "".join(both)


TIER1_FIELDS: tuple[str, ...] = (
    "agent_id",
    "risk_level",
    "goal",
    "out_of_scope",
    "role_boundary",
    "prohibited",
    "approval_required",
    "limits",
    "escalation",
    "untrusted_inputs",
    "owner",
    "data_rules",
)
#: Tier 1 fields that `check` requires (owner implies technical_owner as a warning only).
TIER1_REQUIRED: tuple[str, ...] = (
    "agent_id",
    "risk_level",
    "goal",
    "out_of_scope",
    "prohibited",
    "approval_required",
    "limits",
    "escalation",
    "untrusted_inputs",
    "owner",
)
TIER2_FIELDS: tuple[str, ...] = (
    "fail_closed",
    "bounds",
    "stop_conditions",
    "progress_guard",
    "completion_verification",
    "termination_authority",
    "goal_drift_guard",
    "tool_schema_strictness",
    "tool_result_validation",
    "tool_error_contract",
    "read_write_separation",
    "tool_inventory_report",
    "unbounded_tool_policy",
    "output_size_limits",
    "dry_run_required",
    "context_policy",
    "memory_policy",
    "checkpointing",
    "structured_output",
    "write_safety",
    "fallback_policy",
    "rate_limits",
    "trust_boundary",
    "identity_propagation",
    "egress_policy",
    "secrets_policy",
    "self_modification",
    "sandbox_policy",
    "clarification_policy",
    "confidence_threshold",
    "on_approval_timeout",
    "kill_switch",
    "handoff_contract",
    "trace_policy",
    "audit_log",
    "log_immutable",
    "version_pinning",
    "pinned_models",
    "regression_triggers",
    "rollback_path",
    "drift_monitors",
    "environment_separation",
)
TIER3_FIELDS: tuple[str, ...] = (
    "topology",
    "agent_registry",
    "declared_delegation",
    "max_delegation_depth",
    "authority_inheritance",
    "transitive_prohibition",
    "no_cycles",
    "cross_agent_trust",
    "context_isolation",
    "conflict_resolution",
    "aggregate_budget",
    "concurrency_policy",
    "failure_isolation",
    "end_to_end_trace_id",
)
META_FIELDS: tuple[str, ...] = ("extends", "confirmed", "waivers", "_derived", "spec_version", "technical_owner")


class AgentSpec(_Strict):
    # ---- meta
    spec_version: int = 1
    extends: str | list[str] | None = None
    derived: dict[str, Any] | None = Field(default=None, alias="_derived")
    confirmed: list[Attestation] = Field(default_factory=list)
    #: what each tool does, declared by the person: {read_file: read, save_report: write}. Decides a tool whose body
    #: cannot be followed; a body that writes while declared read is a gap (AS-232)
    tool_effects: dict[str, Literal["read", "write"]] | None = None
    waivers: list[Waiver] = Field(default_factory=list)

    # ---- Tier 1
    agent_id: str | None = None
    risk_level: RiskLevel | None = None
    goal: Goal | None = None
    out_of_scope: list[str] | None = None
    role_boundary: RoleBoundary | None = None
    prohibited: list[Prohibition] | None = None
    approval_required: list[ApprovalRule] | None = None
    limits: list[LimitEntry] | None = None
    escalation: list[EscalationRule] | None = None
    untrusted_inputs: list[UntrustedInput] | None = None
    owner: str | None = None
    technical_owner: str | None = None
    data_rules: DataRules | None = None

    # ---- Tier 2 (None = take the risk_level default)
    fail_closed: bool | None = None
    bounds: Bounds | None = None
    stop_conditions: list[str] | None = None
    progress_guard: ProgressGuard | None = None
    completion_verification: bool | None = None
    termination_authority: str | None = None
    goal_drift_guard: bool | None = None
    tool_schema_strictness: Literal["strict", "lenient"] | None = None
    tool_result_validation: bool | None = None
    tool_error_contract: str | None = None
    read_write_separation: bool | None = None
    tool_inventory_report: bool | None = None
    unbounded_tool_policy: Literal["forbid", "warn", "allow"] | None = None
    output_size_limits: OutputSizeLimits | None = None
    dry_run_required: bool | list[str] | None = None
    context_policy: ContextPolicy | None = None
    memory_policy: MemoryPolicy | None = None
    checkpointing: bool | None = None
    structured_output: bool | None = None
    write_safety: WriteSafety | None = None
    fallback_policy: str | None = None
    rate_limits: dict[str, Any] | None = None
    trust_boundary: str | None = None
    identity_propagation: bool | None = None
    egress_policy: EgressPolicy | None = None
    secrets_policy: str | None = None
    self_modification: bool | None = None
    sandbox_policy: str | None = None
    clarification_policy: str | None = None
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    on_approval_timeout: TimeoutAction | None = None
    kill_switch: bool | None = None
    handoff_contract: str | dict[str, Any] | None = None
    trace_policy: str | None = None
    audit_log: bool | None = None
    log_immutable: bool | None = None
    version_pinning: bool | None = None
    #: model ids the owner states are fixed versions (an Azure deployment, a local tag): the code must use these
    pinned_models: list[str] | None = None
    regression_triggers: list[str] | None = None
    rollback_path: str | None = None
    drift_monitors: list[str] | None = None
    environment_separation: bool | None = None

    # ---- Tier 3
    topology: Literal["single", "pipeline", "supervisor", "hierarchical", "swarm", "peer"] | None = None
    agent_registry: list[AgentEntry] | None = None
    declared_delegation: list[DelegationEdge] | None = None
    max_delegation_depth: int | None = Field(default=None, gt=0)
    authority_inheritance: Literal["subset", "independent"] | None = None
    transitive_prohibition: bool | None = None
    no_cycles: bool | None = None
    cross_agent_trust: Literal["none", "verified", "full"] | None = None
    context_isolation: bool | None = None
    conflict_resolution: str | None = None
    aggregate_budget: AggregateBudget | None = None
    concurrency_policy: str | None = None
    failure_isolation: bool | None = None
    end_to_end_trace_id: bool | None = None

    # ---- convenience views
    def prohibited_tools(self) -> set[str]:
        return {p.tool for p in self.prohibited or [] if p.tool}

    def approval_tools(self) -> dict[str, ApprovalRule]:
        return {a.tool: a for a in self.approval_required or []}

    def untrusted_sources(self) -> set[str]:
        return {u.source for u in self.untrusted_inputs or []}

    def spec_tools(self) -> set[str]:
        """Every tool name the spec mentions."""
        names = self.prohibited_tools() | set(self.approval_tools())
        names |= {lim.tool for lim in self.limits or []}
        if self.role_boundary:
            names |= set(self.role_boundary.allowed_tools)
        return names

    @property
    def is_multi_agent(self) -> bool:
        return bool(self.agent_registry and len(self.agent_registry) > 1) or self.topology not in (None, "single")
