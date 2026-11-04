"""Tier 4 — derived, never written by the user: trace_coverage, enforceability.
Plus `stats`: agent registry and aggregate numbers (counted over the extracted portion only)."""

from __future__ import annotations

from typing import Any

from agenticreq.ast.classify import canonical, is_unbounded, is_write_tool
from agenticreq.graph.analysis import build
from agenticreq.models.finding import PassedCheck
from agenticreq.models.observed import ObservedAgent
from agenticreq.models.spec import AgentSpec
from agenticreq.rules.fidelity import compare_tool


def enforceability(spec: AgentSpec, obs: ObservedAgent | None, passed: list[PassedCheck]) -> dict[str, Any]:
    """Per clause: is it enforced in code, and could an external policy engine enforce it?

    "Enforced in code" is read from the rule results (`passed`: the PASS list of the same run), never decided a
    second time here. A clause whose rule ended UNKNOWN or BLOCKER is not enforced. "export" only says the clause
    CAN be written as an OPA / Cedar policy; nothing in the code shows that a policy engine is deployed."""
    ok = {(p.rule_id, p.subject) for p in passed}
    ok_canon = {(p.rule_id, canonical(p.subject)) for p in passed}
    schema_ok = {p.subject for p in passed if p.rule_id == "AS-207"}
    rows: list[dict[str, str]] = []
    for pr in spec.prohibited or []:
        if pr.tool:
            held = ("AS-203", canonical(pr.tool)) in ok_canon
            rows.append(
                {
                    "clause": f"prohibited:{pr.tool}",
                    "code": "tool absent from code" if held else "",
                    "export": "opa,cedar",
                }
            )
    for a in spec.approval_required or []:
        rows.append(
            {
                "clause": f"approval_required:{a.tool}",
                "code": "gate in code" if ("AS-205", a.tool) in ok or ("AS-205", canonical(a.tool)) in ok_canon else "",
                "export": "opa,cedar",
            }
        )
    for lim in spec.limits or []:
        keys = list(lim.limit.model_dump(exclude_none=True))  # every bound the spec declares must be proven
        held = bool(keys) and all(f"{lim.tool}.{lim.arg}.{k}" in schema_ok for k in keys)
        rows.append(
            {
                "clause": f"limit:{lim.tool}.{lim.arg}",
                "code": "tool schema" if held else "",
                "export": "opa,cedar,schema",
            }
        )
    if spec.role_boundary and spec.role_boundary.allowed_tools:
        rows.append({"clause": "role_boundary", "code": "", "export": "opa,cedar"})
    for e in spec.escalation or []:
        rows.append({"clause": f"escalation:{e.metric}", "code": "", "export": ""})
    for u in spec.untrusted_inputs or []:
        rows.append({"clause": f"untrusted:{u.source}", "code": "", "export": ""})
    if spec.data_rules:
        rows.append({"clause": "data_rules", "code": "", "export": ""})
    if any(p.rule_id == "AS-307" for p in passed):  # the same evidence the rule used
        rows.append({"clause": "end_to_end_trace_id", "code": "one trace id across agents", "export": ""})
    elif obs is not None and obs.trace_evidence:
        rows.append({"clause": "end_to_end_trace_id", "code": "", "export": ""})
    total = len(rows)
    in_code = sum(1 for r in rows if r["code"])
    any_point = sum(1 for r in rows if r["code"] or r["export"])
    return {
        "clauses": total,
        "enforced_in_code": in_code,
        "has_enforcement_point": any_point,
        "exportable_only": sum(1 for r in rows if not r["code"] and r["export"]),
        "sign_off_only": total - any_point,
        "rows": rows,
    }


def trace_coverage(spec: AgentSpec, obs: ObservedAgent | None) -> dict[str, Any]:
    from agenticreq.rules.graph import one_trace_across_agents
    from agenticreq.traces.coverage import coverage_matrix, derive_scenarios

    one = one_trace_across_agents(obs) if obs else None
    end_to_end = {"end_to_end": bool(one), "mechanisms": [e.mechanism for e in obs.trace_evidence] if obs else []}
    if not obs or not obs.trace_spans:
        return {"scenarios": len(derive_scenarios(spec)), "covered": None, "note": "no traces supplied", **end_to_end}
    m = coverage_matrix(spec, obs.trace_spans)
    return {
        "scenarios": len(m),
        "covered": sum(1 for _, n in m if n),
        "matrix": {sc.id: n for sc, n in m},
        **end_to_end,
    }


def stats(obs: ObservedAgent, spec: AgentSpec | None = None, passed: list[PassedCheck] | None = None) -> dict[str, Any]:
    g = build(obs, spec)
    offered = {canonical(n) for n in obs.offered_tools()}
    fidelity_v = fidelity_t = 0
    wire = {canonical(k): v for k, v in obs.wire_tools.items()}
    for name, t in obs.declared_tools.items():
        w = wire.get(canonical(name))
        if w is not None and t.kind != "dict-schema":
            _, v, tot = compare_tool(t, w)
            fidelity_v, fidelity_t = fidelity_v + v, fidelity_t + tot
    has_runtime = obs.has_runtime or bool(obs.mcp_tools)
    out: dict[str, Any] = {
        "framework": obs.framework,
        "framework_version": obs.framework_version,
        "execution_model": obs.execution_model,
        "files": len(obs.files),
        "declared_tools": len(obs.declared_tools),
        "bound_tools": len(obs.bound_tools),
        "observed_tools": len(offered) if has_runtime else None,
        "visible_over_defined": (
            round(len(offered & {canonical(t) for t in obs.declared_tools}) / len(obs.declared_tools), 2)
            if has_runtime and obs.declared_tools
            else None
        ),
        "write_tools": sorted(t.name for t in obs.declared_tools.values() if is_write_tool(t)),
        "unbounded_tools": sorted(t.name for t in obs.declared_tools.values() if is_unbounded(t)),
        "models": sorted({f"{m.name}{'' if m.pinned else ' (unpinned)'}" for m in obs.models}),
        "captured_calls": len(obs.captures),
        "runs": len(obs.run_ids()),
        "trace_spans": len(obs.trace_spans),
        "schema_fidelity": f"{fidelity_v}/{fidelity_t}" if fidelity_t else None,
        "agent_registry": [
            {"id": n, "tools": sorted(t), "prompt_sha": g.hashes.get(n)} for n, t in sorted(g.nodes.items())
        ],
        "delegation_edges": [
            {"from": e.src, "to": e.dst, "via": e.via, "confidence": str(e.confidence)} for e in g.edges
        ],
        "gates": [g_.kind for g_ in obs.gates],
    }
    if spec is not None:
        out["tier4"] = {
            "enforceability": {k: v for k, v in enforceability(spec, obs, passed or []).items() if k != "rows"},
            "trace_coverage": trace_coverage(spec, obs),
        }
    return out
