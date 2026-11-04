"""AS-6xx — schema fidelity drift: what the code declares vs what the model is told.

Spike evidence: CrewAI kept 31/44 fields verbatim (all 9 param descriptions dropped,
an optional param made required, a default lost); OpenAI Agents 42/44; smolagents 42/44.
This is a root cause of hallucinated tool arguments and nothing else checks it.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from agenticreq.ast.classify import canonical
from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding, PassedCheck
from agenticreq.models.observed import ParamSchema, ToolSchema
from agenticreq.rules.base import CheckContext, Mode, rule
from agenticreq.rules.registry import make_finding

_BOTH: tuple[Mode, ...] = ("check", "audit")
_TYPE_EQUIV = {("number", "integer"), ("integer", "number")}


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().rstrip(".").lower()


def _side(p: ParamSchema | None, *, show_type: bool = False) -> str:
    if p is None:
        return "absent"
    parts = ["required" if p.required else "optional"]
    if p.has_default:
        parts.append(f"default {p.default!r}")
    if show_type and p.type:
        parts.append(f"type {p.type}")
    s = " (".join([parts[0], ", ".join(parts[1:])]) + ")" if len(parts) > 1 else parts[0]
    s += f" with description {p.description!r}" if p.description else " with no description"
    return s


def describe(name: str, code: ParamSchema | None, wire: ParamSchema | None) -> str:
    """'code says max_results is optional with description X; the model is told it is required with no description'"""
    show_type = bool(code and wire and code.type and wire.type and code.type != wire.type)
    return (
        f"code says {name} is {_side(code, show_type=show_type)}; "
        f"the model is told it is {_side(wire, show_type=show_type)}"
    )


def _contract(p: ParamSchema) -> str:
    s = "required" if p.required else "optional"
    return s + (f", default={p.default!r}" if p.has_default else ", no default")


def contract_detail(code: ParamSchema, wire: ParamSchema) -> str:
    return (
        f"wire: required=true, default={wire.default!r} | code: {_contract(code)} — the schema the model "
        f"receives for {code.name} contradicts itself: it must always be supplied, yet has a default"
    )


def compare_tool(code: ToolSchema, wire: ToolSchema) -> tuple[list[tuple[str, str, str]], int, int]:
    """→ ([(rule_id, param_or_'', detail)], verbatim_fields, total_fields)."""
    issues: list[tuple[str, str, str]] = []
    total = verbatim = 0
    if code.description:
        total += 1
        cd, wd = _norm(code.description), _norm(wire.description)
        if cd == wd:
            verbatim += 1
        elif not wd:
            issues.append(
                (
                    "AS-601",
                    "",
                    f"code describes {code.name} as {code.description[:80]!r}; the model gets no description",
                )
            )
        elif cd.split("\n")[0][:60] not in wd and wd not in cd:
            issues.append(
                (
                    "AS-601",
                    "",
                    f"code describes {code.name} as {code.description[:80]!r}; "
                    f"the model is told {(wire.description or '')[:80]!r}",
                )
            )
        else:
            verbatim += 1  # same text, framework added/trimmed sections
    wparams = {canonical(p.name): p for p in wire.params}
    cparams = {canonical(p.name): p for p in code.params}
    for key, cp in cparams.items():
        wp = wparams.get(key)
        fields = 2 + (1 if cp.description else 0) + (1 if cp.has_default else 0) + len(cp.constraints)
        total += fields
        if wp is None:
            issues.append(("AS-607", cp.name, f"code declares parameter {cp.name}; the model is never told about it"))
            continue
        d = describe(cp.name, cp, wp)
        # type
        if (
            cp.type is None
            or wp.type is None
            or cp.type == wp.type
            or (cp.type, wp.type) in _TYPE_EQUIV
            or (wp.type and cp.type in wp.type.split("|"))
        ):
            verbatim += 1
        else:
            issues.append(("AS-606", cp.name, d))
        # required — a wire field that is required AND carries a default contradicts itself (AS-604);
        # it supersedes AS-603 for that field so one problem yields one finding
        if wp.required and wp.has_default:
            issues.append(("AS-604", cp.name, contract_detail(cp, wp)))
        elif cp.required == wp.required:
            verbatim += 1
        elif not cp.required and wp.required:
            issues.append(("AS-603", cp.name, d))
        else:
            issues.append(("AS-610", cp.name, d))
        # description
        if cp.description:
            if _norm(cp.description) == _norm(wp.description):
                verbatim += 1
            elif not wp.description:
                issues.append(("AS-602", cp.name, d))
            else:
                issues.append(("AS-602", cp.name, d.replace("; the model", " (altered); the model", 1)))
        # default
        if cp.has_default:
            if wp.has_default and wp.default == cp.default:
                verbatim += 1
            else:
                issues.append(("AS-605", cp.name, d))
        # constraints
        for k, v in cp.constraints.items():
            if wp.constraints.get(k) == v:
                verbatim += 1
            else:
                issues.append(
                    (
                        "AS-609",
                        cp.name,
                        f"code constrains {cp.name} with {k}={v!r}; the model is told {wp.constraints.get(k)!r}",
                    )
                )
    for key, wp in wparams.items():
        if key not in cparams:
            issues.append(
                (
                    "AS-607",
                    wp.name,
                    f"the model is told about parameter {wp.name}, which the code signature does not declare",
                )
            )
    return issues, verbatim, total


def _evidence(wire: ToolSchema, param: str) -> str:
    raw: Any = wire.raw or {}
    params = raw.get("parameters") if isinstance(raw, dict) else None
    if isinstance(params, dict) and param:
        prop = (params.get("properties") or {}).get(param)
        req = param in (params.get("required") or [])
        return f"wire {wire.name}.{param}: {prop} required={req}"
    return f"wire {wire.name}: description={str(wire.description)[:100]!r}"


@rule(
    "fidelity",
    "AS-601",
    "AS-602",
    "AS-603",
    "AS-604",
    "AS-605",
    "AS-606",
    "AS-607",
    "AS-608",
    "AS-609",
    "AS-610",
    modes=_BOTH,
)
def fidelity(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs:
        return []
    out: list[Finding] = []
    wire_by = {canonical(n): t for n, t in {**obs.mcp_tools, **obs.wire_tools}.items()}
    tag = obs.framework_tag()
    for name, code in sorted(obs.declared_tools.items()):
        wire = wire_by.get(canonical(name))
        if wire is None or code.kind == "dict-schema":  # hand-written JSON schema: identical by construction
            continue
        if wire.kind == "wire:hosted":  # the provider's own tool: it carries no schema to compare
            continue
        issues, verbatim, total = compare_tool(code, wire)
        ctx.clause(extracted=True)
        if not issues:
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-608",
                    subject=name,
                    confidence=Confidence.proven(),
                    note=f"{verbatim}/{total} fields verbatim [{tag}]",
                )
            )
            continue
        for rid, param, detail in issues:
            ev = detail.split(" — ")[0] if rid == "AS-604" else _evidence(wire, param)
            out.append(
                make_finding(
                    rid,
                    confidence=Confidence.proven(),
                    code_ref=code.code_ref,
                    title=f"{name}: {_TITLES[rid]}" + (f" ({param})" if param else ""),
                    detail=detail,
                    evidence=f"{tag} | {ev}",
                )
            )
        out.append(
            make_finding(
                "AS-608",
                confidence=Confidence.proven(),
                code_ref=code.code_ref,
                title=f"{name}: {verbatim}/{total} schema fields reach the model verbatim",
                detail=f"{total - verbatim} field(s) lost or altered between code and wire "
                f"({tag}); framework schema rewriting is version-dependent",
                evidence=f"{tag} | " + ", ".join(sorted({r for r, _, _ in issues})),
            )
        )
    return out


_TITLES = {
    "AS-601": "tool description changed on the wire",
    "AS-602": "parameter description dropped",
    "AS-603": "optional parameter sent as required",
    "AS-604": "contradictory field contract",
    "AS-610": "required parameter sent as optional",
    "AS-605": "default dropped on the wire",
    "AS-606": "parameter type changed",
    "AS-607": "parameter set differs",
    "AS-609": "constraint dropped on the wire",
}
