"""AS-5xx — clauses code cannot show. A person signs them off under ``confirmed:``; agenticreq records, never
verifies."""

from __future__ import annotations

import re
from typing import Any, Iterable

from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding
from agenticreq.models.spec import Attestation
from agenticreq.rules.base import CheckContext, rule
from agenticreq.rules.registry import make_finding
from agenticreq.signoff import Assessment, assess

_RPM_KEY = re.compile(r"^(llm_)?(requests?|calls?)_per_minute$|^rpm$", re.IGNORECASE)

#: clause -> why code cannot prove it
ATTESTATION_CLAUSES: dict[str, str] = {
    "data_rules": "retention / never_store happen in storage systems outside the agent code",
    "out_of_scope": "scope is a business judgement; code shows capability, not intent",
    "escalation": "escalation routing happens in on-call / ticketing systems",
    "owner": "ownership is organisational",
    "log_immutable": "immutability is a property of the log store (WORM bucket, ledger)",
    "kill_switch": "a kill switch lives in deployment / feature-flag infrastructure",
    "rollback_path": "rollback is a deployment procedure",
    "environment_separation": "credentials and environments are configured at deploy time",
    "drift_monitors": "monitors run in observability tooling",
    "regression_triggers": "triggers live in CI/CD",
    "audit_log": "audit logging is usually configured in the platform, not the agent",
    "secrets_policy": "secret storage is infrastructure",
    "termination_authority": "who may stop the agent is an operational decision",
    "cross_agent_trust": "trust between agents depends on deployment identity",
    "context_isolation": "isolation depends on runtime memory/session configuration",
    "failure_isolation": "failure isolation depends on the orchestrator's deployment",
    "conflict_resolution": "a process decision between agents/owners",
    "concurrency_policy": "concurrency is governed by the runtime / queue",
    # Level-2 behaviours that live in prompts, orchestration logic or runtime config, not in a
    # statically checkable construct. Recorded, never verified (SCOPE.md: no prompt parsing).
    "stop_conditions": "when the agent stops is decided by the loop's runtime state, not a code construct",
    "progress_guard": "detecting a no-progress loop needs runtime state comparison",
    "completion_verification": "verifying the result against done_criteria happens at runtime",
    "goal_drift_guard": "goal drift is a property of the conversation, not of the code",
    "tool_result_validation": "how tool results are validated depends on runtime data",
    "tool_error_contract": "the error shape a tool returns is decided at runtime (AS-219 catches silent swallowing)",
    "read_write_separation": "which agent/credential reads vs writes is a deployment decision",
    "context_policy": "what is kept in the context window is decided at runtime",
    "memory_policy": "long-term memory writes are configured in the memory store",
    "fallback_policy": "fallback behaviour is an operational decision",
    "rate_limits": "rate limits are enforced by gateways / providers",
    "identity_propagation": "end-user identity flows through the deployment's auth layer",
    "clarification_policy": "asking for clarification is prompt behaviour (no prompt parsing)",
    "confidence_threshold": "model confidence is not observable in code",
    "handoff_contract": "the handoff payload contract is enforced by the orchestrator",
}
#: clauses whose only content is a number: thresholds.py judges them (spec number + attestation), so the generic
#: "declared, code cannot prove it" path below skips them
_NUMERIC_OWNED = {"progress_guard", "confidence_threshold", "rate_limits"}
_MULTI_ONLY = {
    "cross_agent_trust",
    "context_isolation",
    "failure_isolation",
    "conflict_resolution",
    "concurrency_policy",
}


def _rpm_declared(raw: dict[str, Any]) -> bool:
    """rate_limits holds a requests-per-minute number (or a TODO for it): thresholds.py owns that line."""
    rl = raw.get("rate_limits")
    return isinstance(rl, dict) and any(_RPM_KEY.match(str(k)) for k in rl)


def _truthy(v: Any) -> bool:
    return v not in (None, False, "", [], {})


def _nice(v: Any) -> str:
    """A declared value in plain words, never a raw dict / list dump."""
    if isinstance(v, dict):
        return "settings for " + ", ".join(str(k) for k in v)
    if isinstance(v, (list, tuple)):
        parts = [
            x if isinstance(x, str) else (next(iter(x.values()), "") if isinstance(x, dict) else str(x)) for x in v
        ]
        sample = ", ".join(str(p)[:40] for p in parts[:2] if str(p))
        return f"{len(v)} item(s)" + (f" (e.g. {sample})" if sample else "")
    return " ".join(str(v).split())


def _how_to_attest(clause: str) -> str:
    return (
        f"In spec.yml add an entry under `confirmed:` for '{clause}' with: by (who signs it off), date, "
        "evidence (a link, a ticket or file:line) and expires (when to sign it off again)."
    )


def _answered(clause: str, a: Attestation, st: Assessment, ref: Any, shown: str) -> Finding:
    """The line for an entry that is not a plain yes: the owner says no (a gap), not_needed (done), or the entry
    does not count yet (a field missing, expired, evidence changed: still open)."""
    from agenticreq.models.verdict import SIGN_OFF_QUESTION, sentence

    who = f" by {a.by}" if a.by else ""
    ev = f"confirmed: {clause}: answer {a.answer}{who}"
    if st.state == "no":
        return make_finding(
            "AS-503",
            confidence=Confidence.attestation(),
            spec_ref=ref,
            param=clause,
            title=f"{clause}: owner says this is not in place yet",
            reason=sentence(f"Owner says this is not in place yet: {st.message}"),
            status_hint="GAP",
            detail=f"{clause} = {shown}; answered no{who} on {a.date}: {st.message}",
            evidence=ev,
        )
    if st.state == "not_needed":
        return make_finding(
            "AS-504",
            confidence=Confidence.attestation(),
            spec_ref=ref,
            param=clause,
            title=f"{clause}: owner says it is not needed",
            reason=sentence(st.message),
            status_hint="NOT NEEDED",
            detail=f"{clause} = {shown}; answered not_needed{who} on {a.date}: {st.message}",
            evidence=ev,
        )
    question = SIGN_OFF_QUESTION.get(clause, f"Confirm {clause}.")
    return make_finding(
        "AS-501",
        confidence=Confidence.attestation(),
        spec_ref=ref,
        param=clause,
        title=f"'{clause}' is declared, but code cannot show it",
        status_hint="SIGN-OFF",
        reason=f"{question} Not signed yet: {st.message}. Confirm under confirmed: in spec.yml.",
        detail=f"{clause}: you declared {shown}; the entry under confirmed: does not count yet: {st.message}.",
        evidence=ev,
        remediation=_how_to_attest(clause),
    )


@rule("attest", "AS-501", "AS-502", "AS-503", "AS-504")
def attestations(ctx: CheckContext) -> Iterable[Finding]:
    s = ctx.spec
    if not s:
        return []
    eff = ctx.effective()
    root = ctx.observed.root if ctx.observed is not None else None
    entries: dict[str, Attestation] = {att.clause: att for att in s.confirmed}  # the last entry for a clause wins
    answers: dict[str, Assessment] = {c: assess(a, ctx.today, root) for c, a in entries.items()}
    raw = ctx.loaded.raw if ctx.loaded else {}
    out = []
    from_defaults: list[str] = []
    obs = ctx.observed
    runtime_checked: set[str] = set()
    if obs is not None and obs.real_run_ids():  # AS-228..231 read these from real runs instead
        runtime_checked = {"stop_conditions", "progress_guard"}
        if any(c.timestamp for c in obs.captures if not c.synthetic) or any(sp.start_time for sp in obs.trace_spans):
            runtime_checked.add("rate_limits")
    for clause, why in ATTESTATION_CLAUSES.items():
        if clause in _MULTI_ONLY and not s.is_multi_agent:
            continue
        if clause in runtime_checked and (clause not in answers or answers[clause].state == "open"):
            continue
        if clause in _NUMERIC_OWNED and (clause != "rate_limits" or _rpm_declared(raw)):
            continue
        value = raw.get(clause, eff.get(clause))
        if not _truthy(value):
            continue
        ref = ctx.spec_ref(clause) if clause in raw else None
        shown = _nice(value)
        a, st = entries.get(clause), answers.get(clause)
        if clause not in raw and st is None:
            from_defaults.append(f"{clause}={shown}")
            continue
        if a is not None and st is not None and st.state != "signed":
            out.append(_answered(clause, a, st, ref, shown))
            continue
        if a:
            on = f" on {a.date}" if a.date else ""
            out.append(
                make_finding(
                    "AS-502",
                    confidence=Confidence.attestation(),
                    spec_ref=ref,
                    title=f"{clause} signed by {a.by}",
                    reason=f"Signed by {a.by}{on}.",
                    detail=f"{clause} = {shown}; signed by {a.by}{on}"
                    f"{f', expires {a.expires}' if a.expires else ''}. "
                    "agenticreq records the sign-off; it does not check it.",
                    evidence=a.evidence or f"confirmed: {clause}",
                )
            )
        else:
            out.append(
                make_finding(
                    "AS-501",
                    confidence=Confidence.attestation(),
                    spec_ref=ref,
                    title=f"'{clause}' is declared, but code cannot show it",
                    detail=f"{clause}: you declared {shown}. A person signs it off because {why}.",
                    evidence=f"{clause}: {shown}",
                    remediation=_how_to_attest(clause),
                )
            )
    if from_defaults:
        out.append(
            make_finding(
                "AS-501",
                confidence=Confidence.attestation(),
                spec_ref=ctx.spec_ref("risk_level"),
                title=f"{len(from_defaults)} built-in safeguards (from risk_level="
                f"{ctx.risk_level}) need a person to sign them off",
                detail="You did not write these, they come with your risk level. Code cannot show "
                "any of them: " + ", ".join(c.split("=")[0] for c in from_defaults) + ".",
                evidence="; ".join(from_defaults)[:400],
                remediation="Sign off each under `confirmed:` in spec.yml (by, date, evidence, "
                "expires), or declare your own value for it in the spec.",
            )
        )
    return out
