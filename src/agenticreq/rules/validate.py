"""AS-1xx — spec validation. Needs only the spec; no code."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from agenticreq.defaults import floor_violations
from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding, SpecIssue
from agenticreq.models.spec import TIER1_REQUIRED
from agenticreq.names import norm
from agenticreq.rules.base import CheckContext, Mode, rule
from agenticreq.rules.registry import make_finding

_ALL: tuple[Mode, ...] = ("check", "validate")
P = Confidence.proven


def _fmt(path: tuple[str | int, ...]) -> str:
    return ".".join(str(p) for p in path)


U = Confidence.unknown


def _issue_text(i: SpecIssue) -> tuple[str, str]:
    detail = i.message + ("\n" + "\n".join(f"  - {x}" for x in i.items) if len(i.items) > 1 else "")
    remedy = (
        f"expected: {i.expected}\nwrite it like:\n{i.example}"
        if i.example
        else f"expected: {i.expected}"
        if i.expected
        else ""
    )
    return detail, remedy


@rule("validate", "AS-108", "AS-118", "AS-101", modes=_ALL)
def spec_issues(ctx: CheckContext) -> Iterable[Finding]:
    """Prose and placeholders: the field is undeclared, so whatever it should guard is UNKNOWN (never a pass).
    Shape errors never get here: an invalid spec stops before the rules (engine.run)."""
    if not ctx.loaded:
        return []
    out = []
    for i in ctx.loaded.issues:
        if i.kind not in ("prose", "placeholder"):
            continue
        detail, remedy = _issue_text(i)
        out.append(
            make_finding(
                i.rule_id,
                confidence=U(),
                spec_ref=i.spec_ref,
                detail=detail,
                title=f"{i.field} written as prose; it cannot be verified"
                if i.kind == "prose"
                else "Placeholder value, not expanded",
                evidence="; ".join(i.items)[:300] or i.field,
                remediation=remedy or None,
            )
        )
    return out


@rule("validate", "AS-101", modes=_ALL)
def todos(ctx: CheckContext) -> Iterable[Finding]:
    """Every TODO in one finding: the fields are undeclared (UNKNOWN), and `check` refuses the spec."""
    if not ctx.loaded:
        return []
    paths = [p for p in ctx.loaded.todos if not (p and p[0] == "_derived")]
    if not paths:
        return []
    from agenticreq.shapes import example_yaml, spec_fields

    first = ctx.loaded.ref(*paths[0])
    tops = list(dict.fromkeys(str(p[0]) for p in paths))
    examples = "\n".join(example_yaml(t) for t in tops[:3] if t in spec_fields())
    return [
        make_finding(
            "AS-101",
            confidence=U(),
            spec_ref=first,
            title=f"{len(paths)} field{'s' if len(paths) > 1 else ''} still TODO",
            detail=f"still TODO (your decision; `check` refuses an unfinished spec): {', '.join(_fmt(p) for p in paths)}",
            evidence="; ".join(f"{_fmt(p)} (line {getattr(ctx.loaded.ref(*p), 'line', '?')})" for p in paths)[:600],
            remediation="replace each TODO; `agenticreq params` prints the shape and an example of every field, e.g.\n"
            + examples
            if examples
            else None,
        )
    ]


@rule("validate", "AS-102", modes=_ALL)
def required_fields(ctx: CheckContext) -> Iterable[Finding]:
    if not ctx.loaded:
        return []
    todo_tops = {p[0] for p in ctx.loaded.todos if p}
    missing = [
        n
        for n in TIER1_REQUIRED  # `n: not_needed` answers it
        if not (n in ctx.loaded.raw and ctx.loaded.raw[n] is not None)
        and n not in todo_tops
        and n not in ctx.loaded.not_needed
    ]
    if not missing:
        return []
    from agenticreq.shapes import example_yaml

    return [
        make_finding(
            "AS-102",
            confidence=U(),
            spec_ref=ctx.loaded.ref(),
            title=f"{len(missing)} required Level-1 field{'s' if len(missing) > 1 else ''} missing",
            detail=f"missing: {', '.join(missing)} (an explicit empty list is fine, e.g. `{missing[0]}: []`)",
            evidence=f"keys present: {', '.join(sorted(k for k in ctx.loaded.raw if not k.startswith('_')))[:200]}",
            remediation="add them, e.g.\n" + "\n".join(example_yaml(m) for m in missing[:3]),
        )
    ]


@rule("validate", "AS-103", modes=_ALL)
def conflicting_tiers(ctx: CheckContext) -> Iterable[Finding]:
    s = ctx.spec
    if not s:
        return []
    out = []
    from agenticreq.names import norm_set

    allowed = norm_set(s.role_boundary.allowed_tools) if s.role_boundary else set()
    approval = norm_set(s.approval_tools())
    for idx, p in enumerate(s.prohibited or []):
        if not p.tool:
            continue
        clash = []
        if norm(p.tool) in allowed:
            clash.append("role_boundary.allowed_tools")
        if norm(p.tool) in approval:
            clash.append("approval_required")
        if clash:
            out.append(
                make_finding(
                    "AS-103",
                    confidence=P(),
                    spec_ref=ctx.spec_ref("prohibited", idx),
                    detail=f"{p.tool!r} is prohibited but also listed in {' and '.join(clash)}",
                    evidence=f"prohibited: {p.tool}; {', '.join(clash)}: {p.tool}",
                )
            )
    return out


@rule("validate", "AS-104", modes=_ALL)
def approver_missing(ctx: CheckContext) -> Iterable[Finding]:
    s = ctx.spec
    if not s:
        return []
    return [
        make_finding(
            "AS-104",
            confidence=P(),
            spec_ref=ctx.spec_ref("approval_required", i),
            detail=f"approval_required[{i}] ({a.tool}) names no approver",
            evidence=f"tool: {a.tool}, approver: {a.approver}",
        )
        for i, a in enumerate(s.approval_required or [])
        if not a.approver
    ]


@rule("validate", "AS-105", modes=_ALL)
def limit_contradictions(ctx: CheckContext) -> Iterable[Finding]:
    s = ctx.spec
    if not s:
        return []
    out = []
    seen: dict[tuple[str, str], int] = {}
    for i, lim in enumerate(s.limits or []):
        ref = ctx.spec_ref("limits", i)
        L = lim.limit
        if L.min is not None and L.max is not None and L.min > L.max:
            out.append(
                make_finding(
                    "AS-105",
                    confidence=P(),
                    spec_ref=ref,
                    detail=f"{lim.tool}.{lim.arg}: min {L.min} > max {L.max}",
                    evidence=f"min: {L.min}, max: {L.max}",
                )
            )
        key = (lim.tool, lim.arg)
        if key in seen:
            prev = (s.limits or [])[seen[key]].limit
            out.append(
                make_finding(
                    "AS-105",
                    confidence=P(),
                    spec_ref=ref,
                    detail=f"{lim.tool}.{lim.arg} is limited twice "
                    f"({prev.model_dump(exclude_none=True)} vs {L.model_dump(exclude_none=True)})",
                    evidence=f"limits[{seen[key]}] and limits[{i}]",
                )
            )
        seen[key] = i
    b = s.bounds
    if b and b.max_tool_calls is not None and b.max_steps is not None and b.max_tool_calls > b.max_steps * 16:
        out.append(
            make_finding(
                "AS-105",
                confidence=P(),
                spec_ref=ctx.spec_ref("bounds"),
                detail="bounds.max_tool_calls is not reachable within bounds.max_steps",
                evidence=f"max_steps={b.max_steps}, max_tool_calls={b.max_tool_calls}",
            )
        )
    return out


@rule("validate", "AS-106", modes=_ALL)
def expired(ctx: CheckContext) -> Iterable[Finding]:
    s = ctx.spec
    if not s:
        return []
    out = []
    for i, w in enumerate(s.waivers):
        if w.expires and w.expires < ctx.today:
            out.append(
                make_finding(
                    "AS-106",
                    confidence=P(),
                    spec_ref=ctx.spec_ref("waivers", i),
                    detail=f"waiver for {w.rule_id} ({w.subject or 'all'}) expired on {w.expires}",
                    evidence=f"expires: {w.expires}; today: {ctx.today}",
                )
            )
    from agenticreq.rules.attest import ATTESTATION_CLAUSES

    for i, a in enumerate(s.confirmed):
        if a.clause in ATTESTATION_CLAUSES:
            continue  # the item's own line says "sign-off expired" (rules/attest.py)
        if a.expires and a.expires < ctx.today:
            # filed on the clause's own line: an expired sign-off is an open sign-off again, not a spec error
            out.append(
                make_finding(
                    "AS-106",
                    confidence=P(),
                    spec_ref=ctx.spec_ref("confirmed", i),
                    detail=f"the sign-off for {a.clause} by {a.by} expired on {a.expires}",
                    evidence=f"expires: {a.expires}; today: {ctx.today}",
                    param=_line_of(a.clause),
                )
            )
    return out


def _line_of(clause: str) -> str:
    """The report line a sign-off belongs to: a number's own line (``max_cost_usd`` -> ``bounds.max_cost_usd``)."""
    from agenticreq.thresholds import NUMERIC_PARAMS

    return next((p.key for p in NUMERIC_PARAMS if clause in p.attest_names and clause != p.key.split(".")[0]), clause)


@rule("validate", "AS-109", modes=_ALL)
def prohibition_reason(ctx: CheckContext) -> Iterable[Finding]:
    s = ctx.spec
    if not s:
        return []
    return [
        make_finding(
            "AS-109",
            confidence=P(),
            spec_ref=ctx.spec_ref("prohibited", i),
            detail=f"prohibited[{i}] ({p.tool or p.action}) has no reason",
            evidence=str(p.model_dump(exclude_none=True)),
        )
        for i, p in enumerate(s.prohibited or [])
        if not p.reason
    ]


_SAFE_TIMEOUT = ("safe_stop", "deny")


@rule("validate", "AS-110", modes=_ALL)
def approval_timeout(ctx: CheckContext) -> Iterable[Finding]:
    s = ctx.spec
    if not s:
        return []
    out = []
    strict = ctx.risk_level in ("high", "critical")
    for i, a in enumerate(s.approval_required or []):
        action = a.on_timeout or s.on_approval_timeout or "safe_stop"
        # deny (the action is refused) is as safe as safe_stop; proceed acts without approval; at high /
        # critical risk escalate is not enough either: the action waits on nobody accountable for the timeout
        if action == "proceed" or (strict and action not in _SAFE_TIMEOUT):
            out.append(
                make_finding(
                    "AS-110",
                    confidence=P(),
                    spec_ref=ctx.spec_ref("approval_required", i),
                    detail=f"{a.tool}: on_timeout={action!r} at risk_level={ctx.risk_level}",
                    evidence=f"timeout: {a.timeout}, on_timeout: {action}",
                )
            )
    return out  # a malformed timeout ('soon') is a shape error: the loader reports it and nothing runs


@rule("validate", "AS-111", modes=_ALL)
def floors(ctx: CheckContext) -> Iterable[Finding]:
    if not ctx.loaded or not ctx.spec or not ctx.spec.risk_level:
        return []
    return [
        make_finding(
            "AS-111",
            confidence=P(),
            spec_ref=ctx.spec_ref(*key.split(".")),
            detail=f"{key}={user!r} is weaker than the {ctx.risk_level} floor ({floor!r})",
            evidence=f"{key}: {user}",
        )
        for key, user, floor in floor_violations(ctx.spec.risk_level, ctx.loaded.user_tier2)
    ]


@rule("validate", "AS-112", modes=_ALL)
def goal_measurable(ctx: CheckContext) -> Iterable[Finding]:
    s = ctx.spec
    if not s or not s.goal or s.goal.done_criteria:
        return []
    return [
        make_finding(
            "AS-112",
            confidence=P(),
            spec_ref=ctx.spec_ref("goal"),
            detail="goal has no done_criteria; completion cannot be verified",
            evidence=s.goal.statement[:120],
        )
    ]


@rule("validate", "AS-113", modes=_ALL)
def technical_owner(ctx: CheckContext) -> Iterable[Finding]:
    s = ctx.spec
    if not s or not s.owner or s.technical_owner:
        return []
    return [
        make_finding(
            "AS-113",
            confidence=P(),
            spec_ref=ctx.spec_ref("owner"),
            detail="owner is set but technical_owner is not",
            evidence=f"owner: {s.owner}",
        )
    ]


@rule("validate", "AS-114", modes=_ALL)
def delegation_targets(ctx: CheckContext) -> Iterable[Finding]:
    s = ctx.spec
    if not s or not s.declared_delegation:
        return []
    registry = {a.id for a in s.agent_registry or []}
    if s.agent_id:
        registry.add(s.agent_id)
    known = {norm(r) for r in registry}
    out = []
    for i, e in enumerate(s.declared_delegation):
        for end in (e.from_, e.to):
            if norm(end) not in known:
                out.append(
                    make_finding(
                        "AS-114",
                        confidence=P(),
                        spec_ref=ctx.spec_ref("declared_delegation", i),
                        detail=f"edge {e.from_} → {e.to}: {end!r} is not in agent_registry",
                        evidence=f"agent_registry: {sorted(registry)}",
                    )
                )
    return out


@rule("validate", "AS-116", modes=_ALL)
def data_rules(ctx: CheckContext) -> Iterable[Finding]:
    s = ctx.spec
    if not s or not s.data_rules:
        return []
    clash = set(s.data_rules.never_store) & set(s.data_rules.retention)
    return [
        make_finding(
            "AS-116",
            confidence=P(),
            spec_ref=ctx.spec_ref("data_rules"),
            detail=f"{f!r} is in never_store but has retention {s.data_rules.retention[f]!r}",
            evidence=f"never_store: {s.data_rules.never_store}",
        )
        for f in sorted(clash)
    ]


def tool_index(ctx: CheckContext) -> dict[str, list[str]]:
    """Where each tool name appears in the spec (for cross-referencing)."""
    s = ctx.spec
    idx: dict[str, list[str]] = defaultdict(list)
    if not s:
        return idx
    for p in s.prohibited or []:
        if p.tool:
            idx[p.tool].append("prohibited")
    for a in s.approval_required or []:
        idx[a.tool].append("approval_required")
    for lim in s.limits or []:
        idx[lim.tool].append("limits")
    if s.role_boundary:
        for t in s.role_boundary.allowed_tools:
            idx[t].append("role_boundary")
    return idx
