"""Rule registry: rule ID → title → severity → quality attribute → remediation.

The registry holds *metadata*; rule implementations live in sibling modules and
create findings through :func:`make_finding`, so every finding carries consistent
severity / remediation / fixable values for its rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import RULE_ID_RE, Finding, QualityAttribute, Severity, SourceRef

QA = QualityAttribute


@dataclass(frozen=True)
class RuleMeta:
    rule_id: str
    name: str  # short kebab-case slug, used by SARIF
    title: str
    severity: Severity
    attribute: QualityAttribute
    remediation: str
    fixable: bool = False
    help: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


class RuleRegistry:
    def __init__(self) -> None:
        self._rules: dict[str, RuleMeta] = {}

    def register(self, meta: RuleMeta) -> RuleMeta:
        if not RULE_ID_RE.match(meta.rule_id):
            raise ValueError(f"bad rule id {meta.rule_id!r}")
        if meta.rule_id in self._rules:
            raise ValueError(f"duplicate rule id {meta.rule_id}")
        self._rules[meta.rule_id] = meta
        return meta

    def get(self, rule_id: str) -> RuleMeta:
        try:
            return self._rules[rule_id]
        except KeyError:
            raise KeyError(f"rule {rule_id} is not registered") from None

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._rules

    def all(self) -> list[RuleMeta]:
        return [self._rules[k] for k in sorted(self._rules)]


REGISTRY = RuleRegistry()


def make_finding(
    rule_id: str,
    *,
    confidence: Confidence,
    detail: str,
    evidence: str,
    title: str | None = None,
    severity: Severity | None = None,
    spec_ref: SourceRef | None = None,
    code_ref: SourceRef | None = None,
    path: list[str] | None = None,
    remediation: str | None = None,
    fixable: bool | None = None,
    reason: str | None = None,
    param: str | None = None,
    agent: str | None = None,
    status_hint: str | None = None,
) -> Finding:
    meta = REGISTRY.get(rule_id)
    return Finding(
        rule_id=rule_id,
        severity=severity or meta.severity,
        confidence=confidence,
        title=title or meta.title,
        detail=detail,
        evidence=evidence,
        spec_ref=spec_ref,
        code_ref=code_ref,
        path=path,
        remediation=remediation if remediation is not None else meta.remediation,
        fixable=meta.fixable if fixable is None else fixable,
        reason=reason,
        param=param,
        agent=agent,
        status_hint=status_hint,
    )


def _r(
    rule_id: str,
    name: str,
    title: str,
    severity: Severity,
    attribute: QualityAttribute,
    remediation: str,
    fixable: bool = False,
) -> None:
    REGISTRY.register(RuleMeta(rule_id, name, title, severity, attribute, remediation, fixable))


# ---------------------------------------------------------------------------
# AS-1xx  spec validation
# ---------------------------------------------------------------------------
_r(
    "AS-100",
    "spec-invalid",
    "Spec file has the wrong shape; nothing was checked",
    "blocker",
    QA.ACCOUNTABILITY,
    "Rewrite each listed field in the shape shown (`agenticreq validate --fix` does the unambiguous ones).",
)
_r(
    "AS-101",
    "todo-placeholder",
    "Spec still contains a TODO placeholder",
    "blocker",
    QA.ACCOUNTABILITY,
    "Replace every TODO with a real value; an unfinished spec cannot be checked.",
)
_r(
    "AS-102",
    "required-field-missing",
    "Required Tier-1 field is missing",
    "blocker",
    QA.ACCOUNTABILITY,
    "Add the missing field. Tier-1 fields express business intent and are never auto-filled.",
)
_r(
    "AS-103",
    "tool-in-conflicting-tiers",
    "Tool is both prohibited and permitted",
    "blocker",
    QA.LEAST_PRIVILEGE,
    "A tool cannot be prohibited and also allowed / approval-gated. Pick one.",
)
_r(
    "AS-104",
    "approver-missing",
    "approval_required entry has no approver",
    "blocker",
    QA.HUMAN_OVERSIGHT,
    "Name the approver role for every approval-gated tool.",
)
_r(
    "AS-105",
    "limit-contradiction",
    "Limits overlap or contradict",
    "blocker",
    QA.BOUNDEDNESS,
    "Make min <= max and declare each limit once per tool argument.",
)
_r(
    "AS-106",
    "waiver-expired",
    "Waiver or sign-off has expired",
    "blocker",
    QA.ACCOUNTABILITY,
    "Renew the waiver with a new expiry date or fix the underlying finding.",
)
_r(
    "AS-107",
    "spec-schema-error",
    "Spec value has the wrong type or an invalid value",
    "blocker",
    QA.ACCOUNTABILITY,
    "Fix the value so it matches the spec schema.",
)
_r(
    "AS-108",
    "escalation-unstructured",
    "Escalation rule is prose, not metric/op/threshold",
    "blocker",
    QA.HUMAN_OVERSIGHT,
    "Write escalation as {metric, op, threshold, to}. Prose cannot be verified.",
)
_r(
    "AS-119",
    "unmatched-reference",
    "A waiver or confirmed: entry matches nothing",
    "blocker",
    QA.ACCOUNTABILITY,
    "Write a rule id or the parameter name the report shows; `agenticreq params` lists them.",
)
_r(
    "AS-118",
    "prose-instead-of-structure",
    "Field written as prose; it cannot be verified",
    "blocker",
    QA.ACCOUNTABILITY,
    "Write the field in the structure shown; keep the sentence under `note:` if you want it.",
)
_r(
    "AS-109",
    "prohibition-without-reason",
    "Prohibited entry has no reason",
    "warning",
    QA.ACCOUNTABILITY,
    "Add a reason so reviewers know why the tool/action is forbidden.",
)
_r(
    "AS-110",
    "unsafe-approval-timeout",
    "Approval timeout does not stop safely",
    "blocker",
    QA.HUMAN_OVERSIGHT,
    "Set on_timeout: safe_stop (required for high/critical risk).",
)
_r(
    "AS-111",
    "override-below-floor",
    "Tier-2 override is weaker than the risk_level floor",
    "blocker",
    QA.AUTONOMY_SAFETY,
    "Remove the override or lower risk_level deliberately; the floor exists for this risk tier.",
)
_r(
    "AS-112",
    "goal-not-measurable",
    "Goal has no measurable done criteria",
    "warning",
    QA.GOAL_INTEGRITY,
    "Add goal.done_criteria with checkable statements.",
)
_r(
    "AS-113",
    "technical-owner-missing",
    "No technical_owner declared",
    "warning",
    QA.ACCOUNTABILITY,
    "Name an engineering owner who can act on findings.",
)
_r(
    "AS-114",
    "delegation-target-unregistered",
    "Declared delegation targets an unregistered agent",
    "blocker",
    QA.DELEGATION_INTEGRITY,
    "Add the agent to agent_registry or remove the delegation.",
)
_r(
    "AS-115",
    "extends-unresolved",
    "extends target missing or cyclic",
    "blocker",
    QA.CHANGE_SAFETY,
    "Point extends at an existing profile and break any cycle.",
)
_r(
    "AS-116",
    "data-rule-contradiction",
    "data_rules contradict each other",
    "blocker",
    QA.TRUST_BOUNDARIES,
    "A field listed in never_store cannot also have a retention period.",
)
_r(
    "AS-117",
    "unknown-spec-field",
    "Unknown field in spec.yml (stops check: it would never be checked)",
    "blocker",
    QA.ACCOUNTABILITY,
    "Rename it to the field the message names, or run `agenticreq validate --fix`; remove it if it has no match.",
)

# ---------------------------------------------------------------------------
# AS-2xx  conformance (spec vs code)
# ---------------------------------------------------------------------------
_r(
    "AS-201",
    "tool-never-reached",
    "Declared tool was never reached at the boundary",
    "info",
    QA.ACCOUNTABILITY,
    "Add it to order in mocks.yml and run `agenticreq mock-run` (or pass --traces); until then its behaviour is unchecked.",
)
_r(
    "AS-202",
    "undeclared-observed-tool",
    "Model was offered a tool not declared in code",
    "warning",
    QA.LEAST_PRIVILEGE,
    "Find where the tool is injected; declare it, or add it to known_injected_tools.yml if framework-owned.",
)
_r(
    "AS-203",
    "prohibited-tool-reachable",
    "Prohibited tool is reachable in code",
    "blocker",
    QA.LEAST_PRIVILEGE,
    "Remove the tool from the agent's tool list or move it behind a separate agent that is not delegated to.",
)
_r(
    "AS-204",
    "prohibited-tool-invoked",
    "Violation found: prohibited tool was invoked",
    "blocker",
    QA.LEAST_PRIVILEGE,
    "Remove the tool binding; the model already called it.",
)
_r(
    "AS-205",
    "approval-gate-missing",
    "approval_required tool has no human-approval gate in code",
    "blocker",
    QA.HUMAN_OVERSIGHT,
    "Gate the tool with interrupt()/HITL/needs_approval before execution, or export the policy to OPA/Cedar.",
)
_r(
    "AS-206",
    "tool-outside-role-boundary",
    "Tool is outside the declared role_boundary",
    "blocker",
    QA.LEAST_PRIVILEGE,
    "Remove the tool or extend role_boundary deliberately.",
)
_r(
    "AS-207",
    "limit-not-in-schema",
    "Declared limit is not enforced in the tool schema",
    "warning",
    QA.BOUNDEDNESS,
    "Add the bound to the tool schema (e.g. Field(le=...)) so the model and validator both see it.",
)
_r(
    "AS-208",
    "no-step-bound",
    "No step / iteration bound found",
    "warning",
    QA.BOUNDEDNESS,
    "Set recursion_limit / max_iter / max_turns / max_steps explicitly.",
    fixable=True,
)
_r(
    "AS-209",
    "bound-exceeds-spec",
    "Code bound is looser than the spec bound",
    "warning",
    QA.BOUNDEDNESS,
    "Lower the bound in code to match the spec.",
    fixable=True,
)
_r(
    "AS-210",
    "model-not-pinned",
    "Model identifier is not pinned to a snapshot",
    "warning",
    QA.CHANGE_SAFETY,
    "Use a dated snapshot ID so a silent model update cannot change behaviour.",
    fixable=True,
)
_r(
    "AS-211",
    "write-without-idempotency",
    "Write tool has no idempotency key",
    "warning",
    QA.RECOVERABILITY,
    "Add an idempotency_key parameter so retries cannot double-execute side effects.",
    fixable=True,
)
_r(
    "AS-212",
    "write-without-approval",
    "Write tool has no approval gate",
    "warning",
    QA.HUMAN_OVERSIGHT,
    "Declare it in approval_required and gate it in code, or justify autonomy in the spec.",
)
_r(
    "AS-213",
    "unbounded-tool",
    "Tool can return unbounded output",
    "warning",
    QA.BOUNDEDNESS,
    "Add a limit / max_results / page_size parameter with an upper bound.",
)
_r(
    "AS-214",
    "hardcoded-secret",
    "Hard-coded secret in agent code",
    "blocker",
    QA.TRUST_BOUNDARIES,
    "Move the secret to a secret manager / environment variable and rotate it.",
)
_r(
    "AS-215",
    "dynamic-exec-in-tool",
    "Tool executes dynamic code or shell commands",
    "warning",
    QA.TRUST_BOUNDARIES,
    "Run it in an isolated executor (declare it under sandbox_policy) or remove dynamic execution.",
)
_r(
    "AS-216",
    "unrestricted-egress",
    "Tool fetches a model-controlled URL",
    "warning",
    QA.TRUST_BOUNDARIES,
    "Restrict egress to an allowlist (egress_policy).",
)
_r(
    "AS-218",
    "mcp-destructive-unannotated",
    "MCP server tool looks destructive but is annotated as safe",
    "warning",
    QA.HUMAN_OVERSIGHT,
    "Remove readOnlyHint=True / destructiveHint=False (MCP defaults make clients ask), or rename the tool "
    "if it really is safe.",
)
_r(
    "AS-217",
    "spec-tool-not-in-code",
    "Spec references a tool that does not exist in code",
    "warning",
    QA.CHANGE_SAFETY,
    "Fix the tool name in the spec, or the spec is stale.",
)
_r(
    "AS-219",
    "tool-fails-open",
    "Tool swallows errors and continues (fail-open)",
    "warning",
    QA.RECOVERABILITY,
    "Re-raise, or return a structured error the agent must handle (fail_closed). A silent `except: pass` "
    "lets the agent act on a step that never happened.",
)
_r(
    "AS-220",
    "self-modification-tool",
    "Tool lets the agent modify its own instructions, tools or code",
    "blocker",
    QA.AUTONOMY_SAFETY,
    "Remove the tool. self_modification is false at every risk level; prompt/tool changes go through review.",
)
_r(
    "AS-221",
    "no-cost-bound",
    "No run-wide cost / token budget found",
    "warning",
    QA.BOUNDEDNESS,
    "Cap spend per run (UsageLimits(total_tokens_limit=...), max_budget, a cost callback) to match bounds.max_cost_usd.",
)
_r(
    "AS-222",
    "no-duration-bound",
    "No run-wide time limit found",
    "warning",
    QA.BOUNDEDNESS,
    "Cap wall-clock time per run (max_execution_time, asyncio.wait_for/timeout, timeout= on the run call) "
    "to match bounds.max_duration_s.",
)
_r(
    "AS-223",
    "no-environment-separation",
    "Configuration is not separated by environment",
    "warning",
    QA.CHANGE_SAFETY,
    "Read endpoints and credentials from the environment / a settings object so dev, staging and prod differ "
    "without a code change; never hard-code a production URL.",
)
_r(
    "AS-224",
    "no-structured-output",
    "Agent output is free text, not a validated structure",
    "warning",
    QA.GOAL_INTEGRITY,
    "Use the framework's typed output (response_format / output_type / with_structured_output / output_pydantic).",
)
_r(
    "AS-225",
    "no-audit-hook",
    "No audit / trace hook for tool calls found",
    "warning",
    QA.ACCOUNTABILITY,
    "Attach tracing (OpenTelemetry, LangSmith, Langfuse, callbacks=...) or log every tool call, so actions can be "
    "reconstructed after the fact.",
)
_r(
    "AS-226",
    "no-checkpointer",
    "No checkpointer found for a graph that must checkpoint",
    "warning",
    QA.RECOVERABILITY,
    "Compile the graph with checkpointer=... so a failed run resumes instead of repeating side effects.",
)
_r(
    "AS-227",
    "write-without-dry-run",
    "Write tool has no dry-run / preview mode",
    "warning",
    QA.RECOVERABILITY,
    "Add a dry_run (or preview) parameter so the effect can be shown before it is executed.",
)

_r(
    "AS-228",
    "no-progress-loop",
    "Agent repeated the same tool call without progress",
    "warning",
    QA.BOUNDEDNESS,
    "Add a progress guard: stop or change strategy after max_no_progress_steps identical calls.",
)
_r(
    "AS-229",
    "bound-exceeded-observed",
    "Violation found: a run exceeded its declared bound",
    "blocker",
    QA.BOUNDEDNESS,
    "Enforce bounds.max_steps / max_tool_calls in code (recursion_limit, max_turns, max_iter, UsageLimits).",
)
_r(
    "AS-232",
    "declared-read-tool-writes",
    "Tool declared read in spec.yml writes in its body",
    "blocker",
    QA.LEAST_PRIVILEGE,
    "Change tool_effects: to write (and gate it), or remove the write from the tool.",
)
_r(
    "AS-230",
    "stopped-without-stop-condition",
    "Run ended without reaching a declared stop condition",
    "warning",
    QA.GOAL_INTEGRITY,
    "Make every exit explicit: an answer (goal_done), the budget (budget_exhausted), or a surfaced error.",
)
_r(
    "AS-231",
    "rate-limit-exceeded-observed",
    "Violation found: request rate exceeded rate_limits",
    "warning",
    QA.BOUNDEDNESS,
    "Throttle model calls (max_rpm, a rate limiter, provider-side limits) to the declared rate.",
)

# ---------------------------------------------------------------------------
# AS-3xx  delegation / multi-agent
# ---------------------------------------------------------------------------
_r(
    "AS-301",
    "undeclared-delegation",
    "Delegation edge is not declared",
    "blocker",
    QA.DELEGATION_INTEGRITY,
    "Add the edge to declared_delegation or remove the handoff tool.",
)
_r(
    "AS-302",
    "delegation-too-deep",
    "Delegation depth exceeds max_delegation_depth",
    "blocker",
    QA.DELEGATION_INTEGRITY,
    "Flatten the delegation chain or raise the limit deliberately.",
)
_r(
    "AS-303",
    "delegation-cycle",
    "Delegation graph contains a cycle",
    "blocker",
    QA.DELEGATION_INTEGRITY,
    "Break the cycle; cycles allow unbounded delegation loops.",
)
_r(
    "AS-304",
    "transitive-prohibition",
    "Prohibited tool reachable through delegation",
    "blocker",
    QA.LEAST_PRIVILEGE,
    "Prohibitions are transitive: remove the tool from the downstream agent or the delegation edge.",
)
_r(
    "AS-305",
    "authority-escalation",
    "Delegate holds tools its delegator does not",
    "warning",
    QA.DELEGATION_INTEGRITY,
    "With authority_inheritance: subset, a delegate may only hold a subset of its delegator's tools.",
)
_r(
    "AS-306",
    "no-aggregate-budget",
    "Multi-agent system has no aggregate budget",
    "warning",
    QA.BOUNDEDNESS,
    "Declare aggregate_budget (cost/steps across all agents).",
)
_r(
    "AS-307",
    "no-end-to-end-trace-id",
    "No end-to-end trace id across agents",
    "warning",
    QA.ACCOUNTABILITY,
    "Propagate a trace id across handoffs so delegation can be proven, not inferred.",
)
_r(
    "AS-308",
    "unregistered-agent",
    "Observed agent node is not in agent_registry",
    "warning",
    QA.DELEGATION_INTEGRITY,
    "Register the agent so its authority can be checked.",
)
_r(
    "AS-911",
    "sub-field-sign-off",
    "A sub-field the spec sets is confirmed by a person, not by code",
    "attestation",
    QA.ACCOUNTABILITY,
    "Confirm it under confirmed: with the sub-field's name, e.g. agent_registry[].role.",
)
_r(
    "AS-910",
    "runtime-agent-unidentified",
    "An agent created at runtime could not be identified",
    "attestation",
    QA.ACCOUNTABILITY,
    "Say what it is under confirmed: (clause runtime_agents), or give it a name in the code.",
)
_r(
    "AS-234",
    "same-normalized-name",
    "Two names in the code normalize to the same name",
    "warning",
    QA.CHANGE_SAFETY,
    "Rename one of them: the spec, the wire and the report compare names after normalizing them.",
)
_r(
    "AS-233",
    "registry-agent-not-in-code",
    "agent_registry lists an agent the code does not have",
    "warning",
    QA.DELEGATION_INTEGRITY,
    "Fix the agent's name in agent_registry, or add the agent to the code.",
)

# ---------------------------------------------------------------------------
# AS-4xx  taint flow
# ---------------------------------------------------------------------------
_r(
    "AS-401",
    "untrusted-to-write",
    "Untrusted input can reach a write tool without approval",
    "blocker",
    QA.TRUST_BOUNDARIES,
    "Put an approval gate or validator between the untrusted source and the write sink.",
)
_r(
    "AS-402",
    "untrusted-to-write-observed",
    "Violation found: untrusted input flowed into a write tool",
    "blocker",
    QA.TRUST_BOUNDARIES,
    "Gate the write tool; this flow already happened in a captured run.",
)
_r(
    "AS-403",
    "untrusted-inputs-undeclared",
    "Tool looks like an untrusted source but untrusted_inputs omits it",
    "warning",
    QA.TRUST_BOUNDARIES,
    "List it under untrusted_inputs so taint checks cover it.",
)

# ---------------------------------------------------------------------------
# AS-5xx  human sign-off
# ---------------------------------------------------------------------------
_r(
    "AS-501",
    "sign-off-required",
    "Code cannot show this; a person signs it off",
    "attestation",
    QA.ACCOUNTABILITY,
    "Sign it off under confirmed: in spec.yml (by, date, evidence, expires).",
)
_r(
    "AS-502",
    "signed-off",
    "Signed off by a person; agenticreq does not check it",
    "attestation",
    QA.ACCOUNTABILITY,
    "Keep the sign-off current; agenticreq records it and does not check it.",
)
_r(
    "AS-503",
    "owner-says-not-in-place",
    "Owner says this is not in place yet",
    "blocker",
    QA.ACCOUNTABILITY,
    "Put it in place, then change the entry under confirmed: to answer: yes (by, date, expires, evidence).",
)
_r(
    "AS-504",
    "owner-says-not-needed",
    "Owner says this item is not needed here",
    "attestation",
    QA.ACCOUNTABILITY,
    "Nothing to do; the reason under confirmed: is kept with the report.",
)

# ---------------------------------------------------------------------------
# AS-6xx  schema fidelity drift
# ---------------------------------------------------------------------------
_r(
    "AS-601",
    "tool-description-drift",
    "Tool description changed on the wire",
    "warning",
    QA.GOAL_INTEGRITY,
    "Check the framework's tool conversion; the model sees a different description than the code declares.",
)
_r(
    "AS-602",
    "param-description-dropped",
    "Parameter description dropped on the wire",
    "warning",
    QA.GOAL_INTEGRITY,
    "Use the framework's supported description mechanism (Annotated/Field/docstring style) so it is transmitted.",
)
_r(
    "AS-603",
    "optional-made-required",
    "Optional parameter is sent to the model as required",
    "warning",
    QA.GOAL_INTEGRITY,
    "The model will invent a value. Make the framework emit required=false or remove the parameter.",
)
_r(
    "AS-604",
    "contradictory-field-contract",
    "Field carries a default and is marked required on the wire",
    "blocker",
    QA.GOAL_INTEGRITY,
    "The model is forced to supply a value on every call that the schema itself says has a default. Make the "
    "framework emit required=false for defaulted parameters (or drop the default).",
)
_r(
    "AS-605",
    "default-dropped",
    "Parameter default dropped on the wire",
    "warning",
    QA.GOAL_INTEGRITY,
    "Expose the default in the schema so the model does not guess.",
)
_r(
    "AS-606",
    "type-drift",
    "Parameter type changed on the wire",
    "warning",
    QA.GOAL_INTEGRITY,
    "Align the declared type and the transmitted JSON schema type.",
)
_r(
    "AS-607",
    "param-set-drift",
    "Parameter missing from or added to the wire schema",
    "warning",
    QA.GOAL_INTEGRITY,
    "Make the transmitted parameter set match the code signature.",
)
_r(
    "AS-609",
    "constraint-dropped",
    "Parameter constraint dropped on the wire",
    "warning",
    QA.BOUNDEDNESS,
    "The code bounds this argument but the model is never told; expose it in the transmitted schema.",
)
_r(
    "AS-610",
    "required-made-optional",
    "Required parameter is sent to the model as optional",
    "warning",
    QA.GOAL_INTEGRITY,
    "The model may omit it and the call fails at runtime. (Was AS-604 before 0.1.0.dev0.)",
)
_r(
    "AS-608",
    "fidelity-summary",
    "Schema fidelity summary",
    "info",
    QA.GOAL_INTEGRITY,
    "Fix the individual AS-60x findings for this tool.",
)

# ---------------------------------------------------------------------------
# AS-9xx  meta
# ---------------------------------------------------------------------------
_r(
    "AS-901",
    "execution-model-unverifiable",
    "Execution model cannot be verified (prompt-only / code agent)",
    "warning",
    QA.ACCOUNTABILITY,
    "Tools are described in the prompt, not a schema; switch to a schema-based agent to make them verifiable.",
)
_r(
    "AS-902",
    "extraction-failed",
    "Extraction failed for part of the agent",
    "warning",
    QA.ACCOUNTABILITY,
    "Fix the parse error or report the unsupported construct.",
)
_r(
    "AS-903",
    "no-observation",
    "No runtime capture or trace supplied",
    "info",
    QA.ACCOUNTABILITY,
    "Run `agenticreq mock-run` (or pass --traces) to see the agent run.",
)
_r(
    "AS-904",
    "framework-unrecognized",
    "Agent framework not recognized",
    "info",
    QA.ACCOUNTABILITY,
    "Only generic AST extraction ran; add a recognizer for richer checks.",
)
_r(
    "AS-905",
    "scenario-uncovered",
    "Spec scenario not covered by any trace",
    "info",
    QA.ACCOUNTABILITY,
    "Add a mock run (or a trace) that exercises this clause.",
)
_r(
    "AS-907",
    "needs-spec",
    "Business rules need a spec: none supplied",
    "info",
    QA.ACCOUNTABILITY,
    "Run `agenticreq init <path>` to scaffold spec.yml, fill the TODOs, then `agenticreq check <path>`.",
)
_r(
    "AS-908",
    "framework-version-unvalidated",
    "Framework version is outside the range agenticreq was validated on",
    "info",
    QA.ACCOUNTABILITY,
    "Results for this framework may miss constructs. Use a validated version (see data/supported_versions.yml), or "
    "treat the checks for it as incomplete; nothing is reported as passed on a guess.",
)
_r(
    "AS-909",
    "agents-unresolved",
    "Agents built at runtime could not be counted",
    "warning",
    QA.ACCOUNTABILITY,
    "Build agents from a literal table or a config file the code reads with a static path, or run "
    "`agenticreq mock-run` so the agents are seen running; until then the per-agent checks wait for a mock run.",
)
_r(
    "AS-906",
    "wire-format-unsupported",
    "Captured call uses an unsupported wire format",
    "warning",
    QA.ACCOUNTABILITY,
    "This provider format is outside SCOPE.md; its tools are not counted as observed.",
)
